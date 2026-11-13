#!/usr/bin/env python3

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[1]

DEFAULT_DATASET_ROOT = REPOSITORY_ROOT / "data" / "doclaynet"
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "results" / "raw"
)

COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
]


def clip_box(x, y, width, height, image_width, image_height):
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(image_width), x + width)
    y2 = min(float(image_height), y + height)

    return x1, y1, x2, y2


def annotation_exceedance(annotation, image):
    x, y, width, height = annotation["bbox"]

    return max(
        max(0.0, -x),
        max(0.0, -y),
        max(0.0, x + width - image["width"]),
        max(0.0, y + height - image["height"]),
    )


def choose_pages(images, annotations_by_image, categories):
    counts = {
        image["id"]: len(annotations_by_image[image["id"]])
        for image in images
    }

    selected = {}

    empty_pages = [
        image for image in images
        if counts[image["id"]] == 0
    ]
    if empty_pages:
        selected["empty"] = empty_pages[0]

    sorted_non_empty = sorted(
        (
            image for image in images
            if counts[image["id"]] > 0
        ),
        key=lambda image: counts[image["id"]],
    )

    if sorted_non_empty:
        selected["median_density"] = sorted_non_empty[
            len(sorted_non_empty) // 2
        ]

    selected["maximum_density"] = max(
        images,
        key=lambda image: counts[image["id"]],
    )

    target_classes = {
        "table": "Table",
        "formula": "Formula",
        "picture": "Picture",
        "caption": "Caption",
        "footnote": "Footnote",
        "title": "Title",
    }

    category_id_by_name = {
        name: category_id
        for category_id, name in categories.items()
    }

    for output_name, category_name in target_classes.items():
        category_id = category_id_by_name.get(category_name)

        if category_id is None:
            continue

        candidates = [
            image
            for image in images
            if any(
                annotation["category_id"] == category_id
                for annotation in annotations_by_image[
                    image["id"]
                ]
            )
        ]

        if candidates:
            # Prefer a moderately busy example rather than an extreme page.
            candidates.sort(
                key=lambda image: counts[image["id"]]
            )
            selected[output_name] = candidates[
                len(candidates) // 2
            ]

    out_of_bounds_candidates = []

    for image in images:
        maximum_exceedance = max(
            (
                annotation_exceedance(annotation, image)
               for annotation in annotations_by_image[
                    image["id"]
                ]
            ),
            default=0.0,
        )

        if maximum_exceedance > 1e-6:
            out_of_bounds_candidates.append(
                (maximum_exceedance, image)
            )

    if out_of_bounds_candidates:
        out_of_bounds_candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )
        selected["out_of_bounds"] = (
            out_of_bounds_candidates[0][1]
        )

    return selected, counts


def draw_page(
    image_record,
    annotations,
    categories,
    png_dir,
    output_path,
):
    image_path = png_dir / image_record["file_name"]

    with Image.open(image_path) as source:
        canvas = source.convert("RGB")

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    image_width, image_height = canvas.size

    for annotation in annotations:
        category_id = annotation["category_id"]
        category_name = categories[category_id]
        color = COLORS[(category_id - 1) % len(COLORS)]

        x, y, width, height = annotation["bbox"]

        x1, y1, x2, y2 = clip_box(
            x,
            y,
            width,
            height,
            image_width,
            image_height,
        )

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=3,
        )

        label = f"{category_id}: {category_name}"

        text_box = draw.textbbox(
            (x1, y1),
            label,
            font=font,
        )

        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]

        label_y = max(0, y1 - text_height - 4)

        draw.rectangle(
            (
                x1,
                label_y,
                min(image_width, x1 + text_width + 6),
                min(image_height, label_y + text_height + 4),
            ),
            fill=color,
        )

        draw.text(
            (x1 + 3, label_y + 2),
            label,
            fill="black",
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select and visualize representative DocLayNet "
            "ground-truth pages."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve() / args.split

    annotation_path = (
        dataset_root / "COCO" / f"{args.split}.json"
    )
    png_dir = dataset_root / "PNG"

    with annotation_path.open(encoding="utf-8") as file:
        coco = json.load(file)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = {
        category["id"]: category["name"]
        for category in coco["categories"]
    }

    annotations_by_image = defaultdict(list)

    for annotation in annotations:
        annotations_by_image[
            annotation["image_id"]
        ].append(annotation)

    selected, counts = choose_pages(
        images,
        annotations_by_image,
        categories,
    )

    manifest = []

    for purpose, image_record in selected.items():
        annotations_for_image = annotations_by_image[
            image_record["id"]
        ]

        output_name = (
            f"{purpose}__"
            f"{image_record['file_name']}"
        )
        output_path = output_dir / output_name

        draw_page(
            image_record=image_record,
            annotations=annotations_for_image,
            categories=categories,
            png_dir=png_dir,
            output_path=output_path,
        )

        class_counts = Counter(
            categories[annotation["category_id"]]
            for annotation in annotations_for_image
        )

        record = {
            "purpose": purpose,
            "image_id": image_record["id"],
            "file_name": image_record["file_name"],
            "annotation_count": counts[image_record["id"]],
            "class_counts": dict(class_counts),
            "output_file": str(output_path),
        }
        manifest.append(record)

        print(
            f"{purpose:18s} "
            f"boxes={record['annotation_count']:3d} "
            f"file={image_record['file_name']}"
        )

    manifest_path = output_dir / "manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(f"Visualizations written to: {output_dir}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
