#!/usr/bin/env python3
"""Export YOLO26-S predictions for a COCO-annotated DocLayNet split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO

EXPECTED_CLASSES = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO26-S DocLayNet predictions as canonical COCO result JSON."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path)
    parser.add_argument("--image-ids", type=Path, help="Optional JSON list restricting COCO image IDs")
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    raise TypeError(f"Unexpected model.names type: {type(names)!r}")


def selected_ids(path: Path | None) -> set[int] | None:
    if path is None:
        return None
    value = load_json(path)
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError("--image-ids must contain a JSON list of integers")
    if len(value) != len(set(value)):
        raise ValueError("--image-ids contains duplicates")
    return set(value)


def main() -> None:
    args = parse_arguments()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.annotations.is_file():
        raise SystemExit(f"Annotations not found: {args.annotations}")
    if not args.image_root.is_dir():
        raise SystemExit(f"Image root not found: {args.image_root}")
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be in [0,1]")
    if not 1 <= args.max_detections <= 300:
        raise SystemExit("--max-detections must be in [1,300]")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    coco = load_json(args.annotations)
    requested_ids = selected_ids(args.image_ids)
    image_records = sorted(coco["images"], key=lambda record: int(record["id"]))
    if requested_ids is not None:
        known_ids = {int(record["id"]) for record in image_records}
        missing = requested_ids - known_ids
        if missing:
            raise SystemExit(f"Requested image IDs absent from annotations: {sorted(missing)[:20]}")
        image_records = [record for record in image_records if int(record["id"]) in requested_ids]

    image_paths: list[Path] = []
    for record in image_records:
        path = args.image_root / str(record["file_name"])
        if not path.is_file():
            raise SystemExit(f"Image not found: {path}")
        image_paths.append(path)

    model = YOLO(str(args.checkpoint))
    names = normalized_names(model.names)
    observed_names = [names[index] for index in sorted(names)]
    if observed_names != EXPECTED_CLASSES:
        raise SystemExit(
            "Unexpected checkpoint class mapping.\n"
            f"Expected: {EXPECTED_CLASSES}\n"
            f"Observed: {observed_names}"
        )

    predictions: list[dict[str, Any]] = []
    raw_count = 0
    for start in range(0, len(image_records), args.batch_size):
        batch_records = image_records[start : start + args.batch_size]
        batch_paths = image_paths[start : start + args.batch_size]
        results = model.predict(
            source=[str(path) for path in batch_paths],
            imgsz=args.input_size,
            conf=args.confidence,
            max_det=args.max_detections,
            device=args.device,
            batch=args.batch_size,
            verbose=False,
            save=False,
            stream=False,
        )
        if len(results) != len(batch_records):
            raise RuntimeError(
                f"Ultralytics returned {len(results)} results for {len(batch_records)} images"
            )

        for record, path, result in zip(batch_records, batch_paths, results, strict=True):
            image_id = int(record["id"])
            expected_shape = (int(record["height"]), int(record["width"]))
            if tuple(result.orig_shape) != expected_shape:
                raise RuntimeError(
                    f"Image shape mismatch for {path}: annotations={expected_shape}, result={result.orig_shape}"
                )
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.detach().cpu()
            scores = result.boxes.conf.detach().cpu()
            classes = result.boxes.cls.detach().cpu()
            raw_count += len(scores)
            order = scores.argsort(descending=True)

            for index in order.tolist()[: args.max_detections]:
                class_index = int(classes[index])
                if class_index not in names:
                    raise RuntimeError(f"Unknown YOLO class index: {class_index}")
                x1, y1, x2, y2 = map(float, boxes[index].tolist())
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": class_index + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(scores[index]),
                    }
                )

        completed = min(start + args.batch_size, len(image_records))
        print(
            f"Processed {completed}/{len(image_records)} images; "
            f"exported {len(predictions)} predictions",
            flush=True,
        )

    write_json(args.output, predictions)
    metadata_path = args.metadata_out or args.output.with_name(args.output.stem + "-metadata.json")
    metadata = {
        "status": "completed",
        "detector": "yolo26s-doclaynet",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "annotations": str(args.annotations.resolve()),
        "image_root": str(args.image_root.resolve()),
        "image_ids": str(args.image_ids.resolve()) if args.image_ids else None,
        "image_count": len(image_records),
        "raw_detection_count": raw_count,
        "exported_prediction_count": len(predictions),
        "input_size": args.input_size,
        "confidence_floor": args.confidence,
        "max_detections": args.max_detections,
        "batch_size": args.batch_size,
        "device": str(args.device),
        "class_names": names,
        "category_mapping": "YOLO class 0-10 mapped to canonical COCO category 1-11",
        "coordinate_format": "absolute original-image COCO xywh",
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    write_json(metadata_path, metadata)
    print(f"Predictions: {args.output}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
