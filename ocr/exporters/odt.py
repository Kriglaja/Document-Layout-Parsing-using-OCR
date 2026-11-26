"""Semantic ODT exporter."""
from pathlib import Path

from odf.opendocument import OpenDocumentText
from odf.text import H, P

from ..models import ProcessedDocument
from .utils import ordered_regions, region_text


def export(document: ProcessedDocument, output_directory: Path) -> None:
    odt_document = OpenDocumentText()
    for page in document.pages:
        for region in ordered_regions(page):
            text = region_text(region)
            if not text:
                continue
            if region.label in {"Title", "Section-header"}:
                node = H(outlinelevel=1, text=text)
            else:
                node = P(text=text)
            odt_document.text.addElement(node)

    odt_document.save(str(output_directory / f"{document.name}.odt"))
