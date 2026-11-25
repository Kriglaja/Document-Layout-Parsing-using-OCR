#!/usr/bin/env python3
"""Shared helpers for canonical DocLayNet prediction files."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REQUIRED_KEYS = {"image_id", "category_id", "bbox", "score"}
CANONICAL_CATEGORY_IDS = set(range(1, 12))


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_dimensions(coco: dict[str, Any]) -> dict[int, tuple[int, int]]:
    return {int(i["id"]): (int(i["width"]), int(i["height"])) for i in coco["images"]}


def selected_ids(path: Path | None) -> list[int] | None:
    if path is None:
        return None
    value = load_json(path)
    if not isinstance(value, list) or not all(isinstance(x, int) for x in value):
        raise ValueError("Image-ID file must be a JSON list of integers")
    if len(value) != len(set(value)):
        raise ValueError("Image-ID list contains duplicates")
    return value


def validate_record(record: Any, index: int, valid_images: set[int], allowed_categories: set[int]) -> list[str]:
    errors: list[str] = []
    if not isinstance(record, dict):
        return [f"record {index}: must be an object"]
    if set(record) != REQUIRED_KEYS:
        errors.append(f"record {index}: fields must be exactly {sorted(REQUIRED_KEYS)}, got {sorted(record)}")
        return errors
    image_id, category_id, bbox, score = record["image_id"], record["category_id"], record["bbox"], record["score"]
    if not isinstance(image_id, int) or image_id not in valid_images:
        errors.append(f"record {index}: unknown or non-integer image_id {image_id!r}")
    if not isinstance(category_id, int) or category_id not in allowed_categories:
        errors.append(f"record {index}: invalid category_id {category_id!r}")
    if not isinstance(bbox, list) or len(bbox) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in bbox):
        errors.append(f"record {index}: bbox must be four numbers")
    else:
        if not all(math.isfinite(float(v)) for v in bbox):
            errors.append(f"record {index}: bbox contains non-finite values")
        elif float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            errors.append(f"record {index}: bbox width and height must be positive")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
        errors.append(f"record {index}: score must be finite and in [0,1]")
    return errors


def validation_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    by_image = Counter(int(p["image_id"]) for p in predictions)
    by_class = Counter(int(p["category_id"]) for p in predictions)
    scores = [float(p["score"]) for p in predictions]
    counts = sorted(by_image.values())
    def percentile(q: float) -> float:
        if not counts:
            return 0.0
        pos = (len(counts) - 1) * q
        lo, hi = int(pos), min(int(pos) + 1, len(counts) - 1)
        frac = pos - lo
        return counts[lo] * (1 - frac) + counts[hi] * frac
    return {
        "prediction_count": len(predictions),
        "images_with_predictions": len(by_image),
        "predictions_per_image_max": max(counts, default=0),
        "predictions_per_image_mean": (sum(counts) / len(counts)) if counts else 0.0,
        "predictions_per_image_median": percentile(0.5),
        "predictions_per_image_p95": percentile(0.95),
        "predictions_by_category": dict(sorted(by_class.items())),
        "minimum_score": min(scores, default=None),
        "maximum_score": max(scores, default=None),
    }
