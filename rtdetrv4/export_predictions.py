#!/usr/bin/env python3
"""Export RT-DETRv4 predictions to canonical COCO result JSON.

Run from the repository root with the rtdetrv4 environment active.
This deliberately uses YAMLConfig, DetSolver resume loading, EMA, the validation
DataLoader, and the repository PostProcessor exactly like `train.py --test-only`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
RTDETR_ROOT = HERE.parent
if RTDETR_ROOT.name != "rtdetrv4":
    raise RuntimeError("Place this script at rtdetrv4/export_predictions.py")
sys.path.insert(0, str(RTDETR_ROOT.parent))

from engine.core import YAMLConfig  # noqa: E402
from engine.solver import TASKS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export canonical COCO JSON from RT-DETRv4.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metadata-out", type=Path)
    p.add_argument("--confidence", type=float, default=0.001)
    p.add_argument("--max-detections", type=int, default=300)
    p.add_argument("--device", default="cuda")
    p.add_argument("--image-ids", type=Path, help="Optional JSON list restricting exported image IDs")
    return p.parse_args()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ids(path: Path | None) -> set[int] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(x, int) for x in value):
        raise ValueError("--image-ids must contain a JSON list of integers")
    if len(value) != len(set(value)):
        raise ValueError("--image-ids contains duplicates")
    return set(value)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0 <= args.confidence <= 1:
        raise SystemExit("--confidence must be in [0,1]")
    if not 1 <= args.max_detections <= 300:
        raise SystemExit("--max-detections must be in [1,300]")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.config.is_file():
        raise SystemExit(f"Config not found: {args.config}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    selected = load_ids(args.image_ids)
    # Solver._setup requires a path-like output_dir even for evaluation.
    # Use an isolated directory beside the exported predictions so that the
    # original training output directory remains untouched.
    solver_output_dir = args.output.parent / "solver-output"
    solver_output_dir.mkdir(parents=True, exist_ok=True)

    cfg = YAMLConfig(
        str(args.config),
        resume=str(args.checkpoint),
        device=args.device,
        output_dir=str(solver_output_dir),
    )
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()  # builds components and loads resume checkpoint
    model = solver.ema.module if solver.ema else solver.model
    model.eval()
    postprocessor = solver.postprocessor
    postprocessor.eval()
    loader = solver.val_dataloader
    device = solver.device

    predictions: list[dict] = []
    seen: set[int] = set()
    raw_per_image = Counter()
    exported_per_image = Counter()

    with torch.inference_mode():
        for batch_index, (samples, targets) in enumerate(loader):
            batch_ids = [int(t["image_id"].item()) for t in targets]
            keep = [i for i, image_id in enumerate(batch_ids) if selected is None or image_id in selected]
            if not keep:
                continue
            # Subset the batch only after DataLoader loading. This avoids changing
            # dataset or collate behavior while still making val50 runs cheap enough.
            samples = samples.to(device)
            targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(samples)
            orig_sizes = torch.stack([t["orig_size"] for t in targets_device], dim=0)
            results = postprocessor(outputs, orig_sizes)

            for i in keep:
                image_id = batch_ids[i]
                out = results[i]
                boxes = out["boxes"].detach().cpu()
                scores = out["scores"].detach().cpu()
                labels = out["labels"].detach().cpu()
                raw_per_image[image_id] = len(scores)

                order = torch.argsort(scores, descending=True)
                kept = 0
                for j in order.tolist():
                    score = float(scores[j])
                    if score < args.confidence:
                        continue
                    if kept >= args.max_detections:
                        break
                    x1, y1, x2, y2 = (float(v) for v in boxes[j].tolist())
                    category_id = int(labels[j])
                    predictions.append({
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    })
                    kept += 1
                exported_per_image[image_id] = kept
                seen.add(image_id)

            if batch_index % 25 == 0:
                print(f"batch={batch_index} images_exported={len(seen)} predictions={len(predictions)}", flush=True)
            if selected is not None and seen == selected:
                break

    if selected is not None:
        missing = selected - seen
        if missing:
            raise SystemExit(f"Requested image IDs not encountered: {sorted(missing)[:20]}")

    write_json(args.output, predictions)
    metadata_path = args.metadata_out or args.output.with_name(args.output.stem + "-metadata.json")
    metadata = {
        "status": "completed",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "output": str(args.output.resolve()),
        "output_sha256": file_sha256(args.output),
        "device": str(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "used_ema": solver.ema is not None,
        "confidence_floor": args.confidence,
        "max_detections": args.max_detections,
        "selected_image_ids": str(args.image_ids.resolve()) if args.image_ids else None,
        "images_exported": len(seen),
        "prediction_count": len(predictions),
        "raw_predictions_per_image_max": max(raw_per_image.values(), default=0),
        "exported_predictions_per_image_max": max(exported_per_image.values(), default=0),
        "category_ids_observed": sorted({p["category_id"] for p in predictions}),
        "coordinate_format": "absolute original-image COCO xywh",
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
