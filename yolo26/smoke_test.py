#!/usr/bin/env python3

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import torch
import ultralytics
from PIL import Image, ImageDraw
from ultralytics import YOLO


EXPECTED_CLASS_COUNT = 11

EXPECTED_DOCLAYNET_CLASSES = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the pretrained YOLO26-S DocLayNet checkpoint."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to yolo26s-doclaynet.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("yolo26/artifacts/smoke-test"),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
    )
    return parser.parse_args()


def make_synthetic_page(path: Path) -> None:
    """Create a simple page-like image for an inference plumbing test."""
    width, height = 816, 1056
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.rectangle((60, 50, 756, 115), outline="black", width=3)
    draw.rectangle((60, 155, 490, 420), outline="black", width=2)
    draw.rectangle((520, 155, 756, 420), outline="black", width=2)
    draw.rectangle((60, 460, 756, 780), outline="black", width=2)

    for y in range(175, 400, 25):
        draw.line((80, y, 465, y), fill="gray", width=2)

    for y in range(485, 755, 25):
        draw.line((80, y, 730, y), fill="gray", width=2)

    draw.rectangle((60, 930, 756, 990), outline="black", width=2)
    image.save(path)


def normalize_names(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}

    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}

    raise TypeError(f"Unexpected model.names type: {type(names)!r}")


def assert_finite_boxes(result) -> None:
    if result.boxes is None or len(result.boxes) == 0:
        return

    xyxy = result.boxes.xyxy.detach().cpu()
    confidence = result.boxes.conf.detach().cpu()
    classes = result.boxes.cls.detach().cpu()

    if not torch.isfinite(xyxy).all():
        raise RuntimeError("Model produced non-finite box coordinates.")

    if not torch.isfinite(confidence).all():
        raise RuntimeError("Model produced non-finite confidence values.")

    if not torch.isfinite(classes).all():
        raise RuntimeError("Model produced non-finite class indices.")

    for value in confidence.tolist():
        if not 0.0 <= value <= 1.0 or not math.isfinite(value):
            raise RuntimeError(f"Invalid confidence value: {value}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.model.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.model}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. The UCloud smoke test is expected "
            "to run on an NVIDIA GPU."
        )

    device = 0
    synthetic_page = args.output_dir / "synthetic-page.png"
    make_synthetic_page(synthetic_page)

    print(f"Loading checkpoint: {args.model}")
    model = YOLO(str(args.model))

    names = normalize_names(model.names)

    if len(names) != EXPECTED_CLASS_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_CLASS_COUNT} classes, found {len(names)}: {names}"
        )

    # This does not silently rewrite checkpoint metadata. It records whether
    # the embedded names match the canonical mapping expected by the project.
    class_name_match = names == EXPECTED_DOCLAYNET_CLASSES

    print(f"Embedded classes: {names}")
    print(f"Canonical class-name match: {class_name_match}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    results = model.predict(
        source=str(synthetic_page),
        imgsz=args.imgsz,
        conf=0.001,
        max_det=300,
        device=device,
        verbose=True,
        save=True,
        project=str(args.output_dir),
        name="prediction",
        exist_ok=True,
    )

    if len(results) != 1:
        raise RuntimeError(f"Expected one result, got {len(results)}")

    result = results[0]
    assert_finite_boxes(result)

    detection_count = 0 if result.boxes is None else len(result.boxes)

    summary = {
        "status": "passed",
        "checkpoint": str(args.model),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "imgsz": args.imgsz,
        "confidence_floor": 0.001,
        "max_detections": 300,
        "embedded_names": names,
        "expected_names": EXPECTED_DOCLAYNET_CLASSES,
        "class_name_match": class_name_match,
        "detections": detection_count,
        "original_shape": list(result.orig_shape),
    }

    summary_path = args.output_dir / "smoke-test-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Smoke test passed. Summary: {summary_path}")


if __name__ == "__main__":
    main()