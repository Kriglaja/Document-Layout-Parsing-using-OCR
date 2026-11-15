#!/usr/bin/env python3

import json
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[1]

INPUT = REPOSITORY_ROOT / "data" / "doclaynet" / "COCO" / "train.json"
OUTPUT = (
    REPOSITORY_ROOT
    / "data"
    / "doclaynet"
    / "subsets"
    / "train_32_all_classes.json"
)

TARGET_IMAGE_COUNT = 32
MINIMUM_PER_CLASS = 3


def main():
    with INPUT.open(encoding="utf-8") as file:
        coco = json.load(file)

    images = sorted(coco["images"], key=lambda image: image["id"])

    annotations_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    category_ids = {category["id"] for category in coco["categories"]}
    category_names = {
        category["id"]: category["name"] for category in coco["categories"]
    }
    global_class_counts = Counter(
        annotation["category_id"] for annotation in coco["annotations"]
    )

    selected_ids = set()
    class_counts = Counter()

    candidates = [
        image for image in images if annotations_by_image[image["id"]]
    ]
    rarest_first = sorted(
        category_ids,
        key=lambda category_id: global_class_counts[category_id],
    )

    while (
        len(selected_ids) < TARGET_IMAGE_COUNT
        and any(
            class_counts[category_id] < MINIMUM_PER_CLASS
            for category_id in category_ids
        )
    ):
        best_image = None
        best_score = None

        for image in candidates:
            image_id = image["id"]
            if image_id in selected_ids:
                continue

            annotations = annotations_by_image[image_id]
            image_class_counts = Counter(
                annotation["category_id"] for annotation in annotations
            )

            coverage_gain = sum(
                min(
                    image_class_counts[category_id],
                    max(0, MINIMUM_PER_CLASS - class_counts[category_id]),
                )
                for category_id in rarest_first
            )
            rare_gain = sum(
                1
                for category_id in rarest_first
                if class_counts[category_id] < MINIMUM_PER_CLASS
                and image_class_counts[category_id] > 0
            )
            density_penalty = max(0, len(annotations) - 40)
            score = (rare_gain, coverage_gain, -density_penalty, -image_id)

            if best_score is None or score > best_score:
                best_score = score
                best_image = image

        if best_image is None:
            break

        image_id = best_image["id"]
        selected_ids.add(image_id)
        for annotation in annotations_by_image[image_id]:
            class_counts[annotation["category_id"]] += 1

    for image in candidates:
        if len(selected_ids) >= TARGET_IMAGE_COUNT:
            break

        image_id = image["id"]
        if image_id in selected_ids:
            continue

        selected_ids.add(image_id)
        for annotation in annotations_by_image[image_id]:
            class_counts[annotation["category_id"]] += 1

    if len(selected_ids) != TARGET_IMAGE_COUNT:
        raise RuntimeError(
            f"Selected {len(selected_ids)} images, expected {TARGET_IMAGE_COUNT}"
        )

    missing_classes = [
        category_id
        for category_id in sorted(category_ids)
        if class_counts[category_id] < MINIMUM_PER_CLASS
    ]
    if missing_classes:
        names = [category_names[category_id] for category_id in missing_classes]
        raise RuntimeError(
            "Could not reach minimum class coverage for: " + ", ".join(names)
        )

    selected_images = [
        image for image in images if image["id"] in selected_ids
    ]
    selected_annotations = []

    for new_id, annotation in enumerate(
        (
            annotation
            for annotation in coco["annotations"]
            if annotation["image_id"] in selected_ids
        ),
        start=1,
    ):
        copied_annotation = annotation.copy()
        copied_annotation["id"] = new_id
        selected_annotations.append(copied_annotation)

    subset = {
        key: value
        for key, value in coco.items()
        if key not in {"images", "annotations"}
    }
    subset["images"] = selected_images
    subset["annotations"] = selected_annotations

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(subset, file, ensure_ascii=False)

    print(f"Input:       {INPUT}")
    print(f"Output:      {OUTPUT}")
    print(f"Images:      {len(selected_images)}")
    print(f"Annotations: {len(selected_annotations)}")
    print("\nClass coverage:")

    for category_id in sorted(category_ids):
        print(
            f"  {category_id:2d} "
            f"{category_names[category_id]:20s} "
            f"{class_counts[category_id]:4d}"
        )


if __name__ == "__main__":
    main()
