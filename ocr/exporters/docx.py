"""Semantic DOCX exporter."""
from pathlib import Path

from docx import Document

from ..models import ProcessedDocument
from .utils import ordered_regions, region_text

STYLE_BY_LABEL = {
    "Title": "Title",
    "Section-header": "Heading 1",
    "List-item": "List Bullet",
}


def export(document: ProcessedDocument, output_directory: Path) -> None:
    word_document = Document()
    for page_index, page in enumerate(document.pages):
        if page_index:
            word_document.add_page_break()
        for region in ordered_regions(page):
            text = region_text(region)
            if text:
                word_document.add_paragraph(text, style=STYLE_BY_LABEL.get(region.label))

    word_document.save(output_directory / f"{document.name}.docx")
