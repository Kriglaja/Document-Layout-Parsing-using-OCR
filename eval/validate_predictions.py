#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from common import CANONICAL_CATEGORY_IDS, image_dimensions, load_json, selected_ids, sha256, validate_record, validation_summary, write_json


def main() -> None:
    p = argparse.ArgumentParser(description="Validate canonical DocLayNet COCO-result predictions.")
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--image-ids", type=Path)
    p.add_argument("--max-per-image", type=int, default=300)
    p.add_argument("--summary-out", type=Path)
    args = p.parse_args()

    gt, predictions = load_json(args.gt), load_json(args.pred)
    if not isinstance(predictions, list):
        raise SystemExit("Prediction file must contain a JSON list")
    dims = image_dimensions(gt)
    ids = selected_ids(args.image_ids)
    valid_images = set(dims) if ids is None else set(ids)
    unknown_selected = valid_images - set(dims)
    if unknown_selected:
        raise SystemExit(f"Selected image IDs absent from ground truth: {sorted(unknown_selected)[:10]}")

    errors: list[str] = []
    for i, record in enumerate(predictions):
        errors.extend(validate_record(record, i, valid_images, CANONICAL_CATEGORY_IDS))
        if len(errors) >= 100:
            break
    per_image = Counter(p["image_id"] for p in predictions if isinstance(p, dict) and isinstance(p.get("image_id"), int))
    excessive = {k: v for k, v in per_image.items() if v > args.max_per_image}
    if excessive:
        errors.append(f"images exceed max {args.max_per_image}: {list(excessive.items())[:10]}")
    if errors:
        print("Validation failed:")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)

    summary = validation_summary(predictions)
    summary.update({"status": "passed", "prediction_file": str(args.pred), "prediction_sha256": sha256(args.pred), "selected_image_count": len(valid_images)})
    if args.summary_out:
        write_json(args.summary_out, summary)
    print("Validation passed")
    for k, v in summary.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
