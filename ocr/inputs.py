"""Input discovery and PDF rasterization."""
from __future__ import annotations

from pathlib import Path

import pymupdf
from PIL import Image

from .models import PreparedPage

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".pdf"}


def parse_page_selection(specification: str | None, page_count: int) -> set[int]:
    if not specification:
        return set(range(page_count))

    selected: set[int] = set()
    for component in specification.split(","):
        component = component.strip()
        if not component:
            continue
        if "-" in component:
            start, end = map(int, component.split("-", 1))
            if end < start:
                raise ValueError(f"Invalid page range: {component}")
            selected.update(range(start - 1, end))
        else:
            selected.add(int(component) - 1)

    invalid = sorted(index + 1 for index in selected if index < 0 or index >= page_count)
    if invalid:
        raise ValueError(f"Invalid PDF page(s): {invalid}")
    return selected


def discover_inputs(inputs: tuple[Path, ...], recursive: bool) -> list[Path]:
    discovered: list[Path] = []
    for item in inputs:
        if item.is_file():
            candidates = [item]
        elif item.is_dir():
            candidates = item.rglob("*") if recursive else item.glob("*")
        else:
            raise FileNotFoundError(f"Input not found: {item}")

        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                discovered.append(candidate.resolve())

    return sorted(dict.fromkeys(discovered))


def _image_page(source: Path) -> PreparedPage:
    with Image.open(source) as image:
        width, height = image.size
    return PreparedPage(
        document_key=source.stem,
        page_number=1,
        image_key=source.stem,
        image_path=source,
        width=width,
        height=height,
        source_path=source,
    )


def _pdf_pages(
    source: Path,
    work_directory: Path,
    dpi: int,
    selected_pages: str | None,
) -> list[PreparedPage]:
    document = pymupdf.open(source)
    selection = parse_page_selection(selected_pages, document.page_count)
    scale = dpi / 72.0
    pages: list[PreparedPage] = []

    for page_index in sorted(selection):
        image_key = f"{source.stem}-page-{page_index + 1:04d}"
        image_path = work_directory / f"{image_key}.png"
        pixmap = document[page_index].get_pixmap(
            matrix=pymupdf.Matrix(scale, scale),
            alpha=False,
        )
        pixmap.save(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        pages.append(
            PreparedPage(
                document_key=source.stem,
                page_number=page_index + 1,
                image_key=image_key,
                image_path=image_path,
                width=width,
                height=height,
                source_path=source,
            )
        )
    document.close()
    return pages


def prepare_inputs(
    inputs: tuple[Path, ...],
    work_directory: Path,
    *,
    recursive: bool,
    pdf_dpi: int,
    selected_pages: str | None,
) -> list[PreparedPage]:
    work_directory.mkdir(parents=True, exist_ok=True)
    pages: list[PreparedPage] = []

    for source in discover_inputs(inputs, recursive):
        if source.suffix.lower() == ".pdf":
            pages.extend(_pdf_pages(source, work_directory, pdf_dpi, selected_pages))
        else:
            pages.append(_image_page(source))

    if not pages:
        raise ValueError("No supported input documents were found")
    return pages
