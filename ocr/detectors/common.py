"""Shared detector constants and helpers."""
from __future__ import annotations

import hashlib
from pathlib import Path

DOCLAYNET_LABELS = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "List-item",
    5: "Page-footer",
    6: "Page-header",
    7: "Picture",
    8: "Section-header",
    9: "Table",
    10: "Text",
    11: "Title",
}
VISUAL_LABELS = {"Formula", "Picture", "Table"}
EXPECTED_LABEL_NAMES = list(DOCLAYNET_LABELS.values())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_for(label: str) -> str:
    return "visual" if label in VISUAL_LABELS else "ocr"
