#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[1]

DEFAULT_INPUT = (
    REPOSITORY_ROOT / "data" / "doclaynet" / "COCO" / "val.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "data"
    / "doclaynet"
    / "subsets"
    / "val_16.json"
)


def main():
    parser = argparse.ArgumentParser(
        description="Create a deterministic COCO-format subset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--images",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    with input_path.open(encoding="utf-8") as file:
        coco = json.load(file)

    # Deterministic selection using ascending image ID.
    selected_images = sorted(
        coco["images"],
        key=lambda image: image["id"],
    )[:args.images]

    selected_image_ids = {
        image["id"]
        for image in selected_images
    }

    selected_annotations = [
        annotation
        for annotation in coco["annotations"]
        if annotation["image_id"] in selected_image_ids
    ]

    subset = {
        key: value
        for key, value in coco.items()
        if key not in {"images", "annotations"}
    }

    subset["images"] = selected_images
    subset["annotations"] = selected_annotations

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            subset,
            file,
            ensure_ascii=False,
        )

    print(f"Input:       {input_path}")
    print(f"Output:      {output_path}")
    print(f"Images:      {len(selected_images)}")
    print(f"Annotations: {len(selected_annotations)}")


if __name__ == "__main__":
    main()
