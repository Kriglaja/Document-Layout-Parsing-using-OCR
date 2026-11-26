"""Plain-text exporter."""
from pathlib import Path

from ..models import ProcessedDocument
from .utils import ordered_regions, region_text


def export(document: ProcessedDocument, output_directory: Path) -> None:
    chunks: list[str] = []
    for page in document.pages:
        chunks.append(f"--- Page {page.page_number} ---")
        for region in ordered_regions(page):
            text = region_text(region)
            if text:
                chunks.append(text)

    path = output_directory / f"{document.name}.txt"
    path.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
