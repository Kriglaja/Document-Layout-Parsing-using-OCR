#!/usr/bin/env python3
"""Export, sanitize, validate, and evaluate YOLO26-S on a DocLayNet split."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(arguments: list[object]) -> None:
    command = [sys.executable, *map(str, arguments)]
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "yolo26/checkpoints/yolo26s-doclaynet.pt")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/doclaynet")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/yolo")
    parser.add_argument("--image-ids", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--agnostic-nms",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    annotations = args.data_root / "COCO" / f"{args.split}.json"
    image_root = args.data_root / "PNG"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = output_dir / f"{args.split}-raw.json"
    metadata = output_dir / f"{args.split}-export-metadata.json"
    sanitized = output_dir / f"{args.split}-sanitized.json"
    sanitation = output_dir / f"{args.split}-sanitation.json"
    validation = output_dir / f"{args.split}-validation.json"
    metrics = output_dir / f"{args.split}-metrics.json"

    export_arguments: list[object] = [
        PROJECT_ROOT / "eval/export_yolo_doclaynet.py",
        "--checkpoint", args.checkpoint,
        "--annotations", annotations,
        "--image-root", image_root,
        "--output", raw,
        "--metadata-out", metadata,
        "--confidence", args.confidence,
        "--max-detections", args.max_detections,
        "--input-size", args.input_size,
        "--device", args.device,
        "--batch-size", args.batch_size,
        "--iou", args.iou,
        "--agnostic-nms" if args.agnostic_nms else "--no-agnostic-nms",
    ]
    if args.image_ids:
        export_arguments.extend(["--image-ids", args.image_ids])
    run(export_arguments)

    run(
        [
            PROJECT_ROOT / "eval/sanitize_predictions.py",
            "--gt", annotations,
            "--pred", raw,
            "--out", sanitized,
            "--report-out", sanitation,
        ]
    )

    validation_arguments: list[object] = [
        PROJECT_ROOT / "eval/validate_predictions.py",
        "--gt", annotations,
        "--pred", sanitized,
        "--summary-out", validation,
    ]
    if args.image_ids:
        validation_arguments.extend(["--image-ids", args.image_ids])
    run(validation_arguments)

    evaluation_arguments: list[object] = [
        PROJECT_ROOT / "eval/coco_eval.py",
        "--gt", annotations,
        "--pred", sanitized,
        "--max-dets", 1, 10, 100,
        "--out", metrics,
    ]
    if args.image_ids:
        evaluation_arguments.extend(["--image-ids", args.image_ids])
    run(evaluation_arguments)

    print(f"\nCompleted YOLO {args.split} evaluation")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
