#!/usr/bin/env python3
"""Detect document layout, run OCR, and export selected formats."""

from ocr.config import parse_arguments, resolve_settings
from ocr.pipeline import process_documents


def main() -> None:
    arguments = parse_arguments()
    settings = resolve_settings(arguments)
    result = process_documents(settings)
    print(f"\nCompleted successfully: {result.output_directory}")


if __name__ == "__main__":
    main()
