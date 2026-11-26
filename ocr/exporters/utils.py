"""Shared exporter helpers."""
from __future__ import annotations

from collections import defaultdict

from ..models import ProcessedPage, Region

REGION_PRIORITY = {
    "Page-header": 0,
    "Title": 1,
    "Section-header": 2,
    "Text": 3,
    "List-item": 3,
    "Caption": 4,
    "Picture": 4,
    "Table": 4,
    "Formula": 4,
    "Footnote": 5,
    "Page-footer": 6,
}


def ordered_regions(page: ProcessedPage) -> list[Region]:
    return sorted(
        page.regions,
        key=lambda region: (
            REGION_PRIORITY.get(region.label, 4),
            region.bbox.y,
            region.bbox.x,
        ),
    )


def region_text(region: Region) -> str:
    lines: dict[tuple[int, int, int], list] = defaultdict(list)
    for token in region.tokens:
        key = (token.block_number, token.paragraph_number, token.line_number)
        lines[key].append(token)

    ordered_lines = sorted(
        lines.values(),
        key=lambda tokens: min(token.bbox.y for token in tokens),
    )
    return "\n".join(
        " ".join(token.text for token in sorted(tokens, key=lambda token: token.bbox.x))
        for tokens in ordered_lines
    )
