"""Searchable facsimile PDF exporter."""
from pathlib import Path

import pymupdf

from ..models import ProcessedDocument


def export(document: ProcessedDocument, output_directory: Path) -> None:
    pdf = pymupdf.open()
    for source in document.pages:
        page = pdf.new_page(width=source.width, height=source.height)
        page.insert_image(page.rect, filename=str(source.image_path))
        for region in source.regions:
            for token in region.tokens:
                page.insert_text(
                    (token.bbox.x, token.bbox.y + token.bbox.height * 0.85),
                    token.text,
                    fontsize=max(4.0, token.bbox.height * 0.9),
                    render_mode=3,
                    overlay=True,
                )

    path = output_directory / f"{document.name}.pdf"
    pdf.save(path, garbage=4, deflate=True)
    pdf.close()
