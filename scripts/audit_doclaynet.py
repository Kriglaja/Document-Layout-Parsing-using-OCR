#!/usr/bin/env python3

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[1]

DEFAULT_DATASET_ROOT = (
    REPOSITORY_ROOT / "data" / "doclaynet"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "results"/ "doclaynet_audit.json"
)
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".png"}


def percentile(sorted_values: list[int], probability: float) -> int | None:
    """Return the nearest-rank-style value used by this audit."""
    if not sorted_values:
        return None

    index = round((len(sorted_values) - 1) * probability)
    return sorted_values[index]


def summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }

    sorted_values = sorted(values)

    return {
        "count": len(values),
        "minimum": min(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": percentile(sorted_values, 0.95),
        "p99": percentile(sorted_values, 0.99),
        "maximum": max(values),
    }


def analyze_split(
    split: str,
    annotation_path: Path,
    png_dir: Path,
) -> dict[str, Any]:
    with annotation_path.open(encoding="utf-8") as file:
        coco = json.load(file)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    category_records = coco.get("categories", [])

    categories = {
        category["id"]: category["name"]
        for category in category_records
    }

    image_by_id = {
        image["id"]: image
        for image in images
    }

    filenames = [
        image["file_name"]
        for image in images
    ]
    unique_filenames = set(filenames)

    missing_png_files = sorted(
        filename
        for filename in unique_filenames
        if not (png_dir / filename).is_file()
    )

    image_precedence = Counter()
    annotation_precedence = Counter()

    images_with_precedence = []
    annotations_with_precedence = 0

    dimensions = Counter()
    boxes_per_image = Counter()
    boxes_per_class = Counter()

    invalid_dimension_annotations = []
    missing_image_references = []
    missing_category_references = []
    boundary_violations = []

    for image in images:
        dimensions[
            (image.get("width"), image.get("height"))
        ] += 1

        if "precedence" in image:
            value = image["precedence"]
            image_precedence[str(value)] += 1

            images_with_precedence.append({
                "image_id": image["id"],
                "file_name": image["file_name"],
                "precedence": value,
            })

    for annotation in annotations:
        annotation_id = annotation["id"]
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]

        if "precedence" in annotation:
            value = annotation["precedence"]
            annotation_precedence[str(value)] += 1
            annotations_with_precedence += 1

        boxes_per_image[image_id] += 1
        boxes_per_class[category_id] += 1

        if image_id not in image_by_id:
            missing_image_references.append(annotation_id)
            continue

        if category_id not in categories:
            missing_category_references.append(annotation_id)

        bbox = annotation.get("bbox")

        if bbox is None or len(bbox) != 4:
            invalid_dimension_annotations.append({
                "annotation_id": annotation_id,
                "reason": "bbox missing or does not contain four values",
                "bbox": bbox,
            })
            continue

        x, y, width, height = bbox

        if width <= 0 or height <= 0:
            invalid_dimension_annotations.append({
                "annotation_id": annotation_id,
                "reason": "non-positive width or height",
                "bbox": bbox,
            })

        image = image_by_id[image_id]
        image_width = image["width"]
        image_height = image["height"]

        left_exceedance = max(0.0, -x)
        top_exceedance = max(0.0, -y)
        right_exceedance = max(
            0.0,
            x + width - image_width,
        )
        bottom_exceedance = max(
            0.0,
            y + height - image_height,
        )

        maximum_exceedance = max(
            left_exceedance,
            top_exceedance,
            right_exceedance,
            bottom_exceedance,
        )

        if maximum_exceedance > 1e-6:
            boundary_violations.append({
                "annotation_id": annotation_id,
                "image_id": image_id,
                "file_name": image["file_name"],
                "category_id": category_id,
                "category_name": categories.get(category_id),
                "image_width": image_width,
                "image_height": image_height,
                "bbox": bbox,
                "maximum_exceedance_px": maximum_exceedance,
                "side_exceedance_px": {
                    "left": left_exceedance,
                    "top": top_exceedance,
                    "right": right_exceedance,
                    "bottom": bottom_exceedance,
                },
            })

    page_counts = [
        boxes_per_image.get(image["id"], 0)
        for image in images
    ]
    sorted_page_counts = sorted(page_counts)

    busiest_pages = sorted(
        (
            {
                "image_id": image["id"],
                "file_name": image["file_name"],
                "annotation_count": boxes_per_image.get(
                    image["id"],
                    0,
                ),
            }
            for image in images
        ),
        key=lambda item: item["annotation_count"],
        reverse=True,
    )[:10]

    boundary_violations.sort(
        key=lambda item: item["maximum_exceedance_px"],
        reverse=True,
    )

    boundary_thresholds = (0.01, 0.1, 1.0, 2.0, 5.0, 10.0)

    boundary_threshold_counts = {
        str(threshold): sum(
            violation["maximum_exceedance_px"] > threshold
            for violation in boundary_violations
        )
        for threshold in boundary_thresholds
    }

    class_counts = [
        {
            "category_id": category_id,
            "category_name": categories[category_id],
            "annotation_count": boxes_per_class.get(
                category_id,
                0,
            ),
        }
        for category_id in sorted(categories)
    ]

    dimension_counts = [
        {
            "width": width,
            "height": height,
            "image_count": count,
        }
        for (width, height), count in dimensions.most_common()
    ]

    return {
        "split": split,
        "annotation_file": str(annotation_path),
        "image_records": len(images),
        "unique_filenames": len(unique_filenames),
        "duplicate_filename_records": (
            len(filenames) - len(unique_filenames)
        ),
        "annotation_records": len(annotations),
        "category_count": len(categories),
        "referenced_filenames": sorted(unique_filenames),
        "missing_png_count": len(missing_png_files),
        "missing_png_files": missing_png_files,
        "categories": [
            {
                "id": category_id,
                "name": categories[category_id],
            }
            for category_id in sorted(categories)
        ],
        "image_dimensions": dimension_counts,
        "precedence": {
            "image_records_with_field": len(
                images_with_precedence
            ),
            "image_value_counts": dict(image_precedence),
            "annotation_records_with_field": (
                annotations_with_precedence
            ),
            "annotation_value_counts": dict(
                annotation_precedence
            ),
            "sample_images": images_with_precedence[:10],
        },
        "boxes_per_page": {
            "minimum": min(page_counts) if page_counts else None,
            "mean": (
                statistics.mean(page_counts)
                if page_counts
                else None
            ),
            "median": (
                statistics.median(page_counts)
                if page_counts
                else None
            ),
            "p95": percentile(sorted_page_counts, 0.95),
            "p99": percentile(sorted_page_counts, 0.99),
            "maximum": max(page_counts) if page_counts else None,
            "empty_page_count": sum(
                count == 0
                for count in page_counts
            ),
            "busiest_pages": busiest_pages,
        },
        "instances_per_class": class_counts,
        "integrity": {
            "invalid_box_dimension_count": len(
                invalid_dimension_annotations
            ),
            "invalid_box_dimensions": (
                invalid_dimension_annotations
            ),
            "missing_image_reference_count": len(
                missing_image_references
            ),
            "missing_image_reference_annotation_ids": (
                missing_image_references
            ),
            "missing_category_reference_count": len(
                missing_category_references
            ),
            "missing_category_reference_annotation_ids": (
                missing_category_references
            ),
            "out_of_bounds_count": len(
                boundary_violations
            ),
            "out_of_bounds_percentage": (
                100.0
                * len(boundary_violations)
                / len(annotations)
                if annotations
                else 0.0
            ),
            "out_of_bounds_threshold_counts": (
                boundary_threshold_counts
            ),
            "maximum_boundary_exceedance_px": (
                boundary_violations[0][
                    "maximum_exceedance_px"
                ]
                if boundary_violations
                else 0.0
            ),
            "worst_boundary_violations": (
                boundary_violations[:20]
            ),
        },
    }


def print_split_summary(result: dict[str, Any]) -> None:
    print()
    print("=" * 72)
    print(result["split"].upper())
    print("=" * 72)

    print()
    print("DATASET")
    print(f"  image records:        {result['image_records']:,}")
    print(f"  unique filenames:     {result['unique_filenames']:,}")
    print(
        "  duplicate filenames:  "
        f"{result['duplicate_filename_records']:,}"
    )
    print(
        f"  annotations:          "
        f"{result['annotation_records']:,}"
    )
    print(f"  categories:           {result['category_count']:,}")
    print(f"  missing PNG files:    {result['missing_png_count']:,}")

    precedence = result["precedence"]

    print()
    print("PRECEDENCE")
    print(
        "  image records with field:      "
        f"{precedence['image_records_with_field']:,}"
    )
    print(
        "  image value counts:             "
        f"{precedence['image_value_counts']}"
    )
    print(
        "  annotation records with field: "
        f"{precedence['annotation_records_with_field']:,}"
    )
    print(
        "  annotation value counts:       "
        f"{precedence['annotation_value_counts']}"
    )

    boxes = result["boxes_per_page"]

    print()
    print("BOXES PER PAGE")
    print(f"  minimum:   {boxes['minimum']:,}")
    print(f"  mean:      {boxes['mean']:.2f}")
    print(f"  median:    {boxes['median']:.1f}")
    print(f"  p95:       {boxes['p95']:,}")
    print(f"  p99:       {boxes['p99']:,}")
    print(f"  maximum:   {boxes['maximum']:,}")
    print(f"  empty:     {boxes['empty_page_count']:,}")

    print("  busiest pages:")
    for page in boxes["busiest_pages"]:
        print(
            f"    {page['annotation_count']:4d}  "
            f"{page['file_name']}"
        )

    print()
    print("INSTANCES PER CLASS")
    for class_result in result["instances_per_class"]:
        print(
            f"  {class_result['category_id']:2d} "
            f"{class_result['category_name']:20s} "
            f"{class_result['annotation_count']:,}"
        )

    integrity = result["integrity"]

    print()
    print("INTEGRITY")
    print(
        "  invalid box dimensions:       "
        f"{integrity['invalid_box_dimension_count']:,}"
    )
    print(
        "  out-of-bounds boxes:          "
        f"{integrity['out_of_bounds_count']:,} "
        f"({integrity['out_of_bounds_percentage']:.4f}%)"
    )
    print(
        "  maximum boundary exceedance: "
        f"{integrity['maximum_boundary_exceedance_px']:.6f}px"
    )
    print(
        "  missing image references:     "
        f"{integrity['missing_image_reference_count']:,}"
    )
    print(
        "  missing category references:  "
        f"{integrity['missing_category_reference_count']:,}"
    )

    print("  boundary exceedance thresholds:")
    for threshold, count in (
        integrity["out_of_bounds_threshold_counts"].items()
    ):
        print(f"    > {threshold:>4}px: {count:,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the canonical DocLayNet COCO dataset and "
            "write a machine-readable JSON summary."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "DocLayNet root containing COCO/ and PNG/. "
            f"Default: {DEFAULT_DATASET_ROOT}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Path for the generated JSON report. "
            f"Default: {DEFAULT_OUTPUT}"
        ),
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    coco_dir = dataset_root / "COCO"
    png_dir = dataset_root / "PNG"

    if not coco_dir.is_dir():
        raise FileNotFoundError(
            f"COCO directory not found: {coco_dir}"
        )

    if not png_dir.is_dir():
        raise FileNotFoundError(
            f"PNG directory not found: {png_dir}"
        )

    annotation_files = {
        split: coco_dir / f"{split}.json"
        for split in SPLITS
    }

    for split, annotation_file in annotation_files.items():
        if not annotation_file.is_file():
            raise FileNotFoundError(
                f"{split} annotation file not found: "
                f"{annotation_file}"
            )

    disk_png_files = {
        path.name
        for path in png_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    split_results = {}
    all_referenced_filenames = set()

    for split in SPLITS:
        result = analyze_split(
            split=split,
            annotation_path=annotation_files[split],
            png_dir=png_dir,
        )

        split_results[split] = result
        all_referenced_filenames.update(
            result.pop("referenced_filenames")
        )

        print_split_summary(result)

    unreferenced_png_files = sorted(
        disk_png_files - all_referenced_filenames
    )
    referenced_but_missing = sorted(
        all_referenced_filenames - disk_png_files
    )

    category_mappings = {
        split: result["categories"]
        for split, result in split_results.items()
    }

    mappings_equal = (
        category_mappings["train"]
        == category_mappings["val"]
        == category_mappings["test"]
    )

    total_images = sum(
        result["image_records"]
        for result in split_results.values()
    )
    total_annotations = sum(
        result["annotation_records"]
        for result in split_results.values()
    )
    total_out_of_bounds = sum(
        result["integrity"]["out_of_bounds_count"]
        for result in split_results.values()
    )

    report = {
        "dataset": "DocLayNet v1.0",
        "dataset_root": str(dataset_root),
        "canonical_annotation_format": "COCO",
        "overall": {
            "png_files_on_disk": len(disk_png_files),
            "referenced_unique_filenames": len(
                all_referenced_filenames
            ),
            "unreferenced_png_count": len(
                unreferenced_png_files
            ),
            "unreferenced_png_files": (
                unreferenced_png_files
            ),
            "referenced_but_missing_count": len(
                referenced_but_missing
            ),
            "referenced_but_missing_files": (
                referenced_but_missing
            ),
            "total_image_records": total_images,
            "total_annotation_records": total_annotations,
            "total_out_of_bounds_count": total_out_of_bounds,
            "total_out_of_bounds_percentage": (
                100.0
                * total_out_of_bounds
                / total_annotations
                if total_annotations
                else 0.0
            ),
            "category_mappings_equal_across_splits": (
                mappings_equal
            ),
            "canonical_categories": (
                category_mappings["train"]
            ),
        },
        "splits": split_results,
    }

    print()
    print("=" * 72)
    print("OVERALL")
    print("=" * 72)
    print(f"  PNG files on disk:       {len(disk_png_files):,}")
    print(
        "  referenced filenames:    "
        f"{len(all_referenced_filenames):,}"
    )
    print(
        "  unreferenced PNG files:  "
        f"{len(unreferenced_png_files):,}"
    )
    print(
        "  referenced but missing:  "
        f"{len(referenced_but_missing):,}"
    )
    print(f"  total image records:     {total_images:,}")
    print(f"  total annotations:       {total_annotations:,}")
    print(
        "  out-of-bounds boxes:     "
        f"{total_out_of_bounds:,} "
        f"({report['overall']['total_out_of_bounds_percentage']:.4f}%)"
    )
    print(
        "  category maps identical: "
        f"{mappings_equal}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(f"Audit report written to: {args.output}")


if __name__ == "__main__":
    main()
