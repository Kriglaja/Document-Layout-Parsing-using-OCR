"""End-to-end document processing orchestration."""
from __future__ import annotations

import json
import shutil
from collections import defaultdict

import pytesseract

from .config import PipelineSettings
from .detectors import create_backend
from .exporters import EXPORTERS
from .inputs import prepare_inputs
from .models import PipelineResult, ProcessedDocument
from .recognition import recognize_pages


def _write_run_config(settings: PipelineSettings) -> None:
    payload = {
        "inputs": [str(path) for path in settings.inputs],
        "output": str(settings.output_directory),
        "output_name": settings.output_name,
        "formats": sorted(settings.formats),
        "recursive": settings.recursive,
        "pdf_dpi": settings.pdf_dpi,
        "pages": settings.selected_pages,
        "detector": settings.detector,
        "checkpoint": str(settings.checkpoint),
        "config": str(settings.model_config) if settings.model_config else None,
        "device": settings.device,
        "detector_confidence": settings.detector_confidence,
        "max_detections": settings.max_detections,
        "input_size": settings.input_size,
        "ocr": {
            "language": settings.language,
            "psm": settings.psm,
            "token_confidence": settings.token_confidence,
            "tesseract_command": settings.tesseract_command,
        },
    }
    path = settings.output_directory / "run-config.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def process_documents(settings: PipelineSettings) -> PipelineResult:
    output = settings.output_directory
    work_directory = output / ".work"
    output.mkdir(parents=True, exist_ok=True)
    work_directory.mkdir(parents=True, exist_ok=True)

    try:
        print("\n[prepare inputs]", flush=True)
        prepared_pages = prepare_inputs(
            settings.inputs,
            work_directory / "pages",
            recursive=settings.recursive,
            pdf_dpi=settings.pdf_dpi,
            selected_pages=settings.selected_pages,
        )

        document_keys = {page.document_key for page in prepared_pages}
        if settings.output_name and len(document_keys) != 1:
            raise ValueError("--name is only valid for one logical document")

        print(f"\n[detect layout: {settings.detector}]", flush=True)
        backend = create_backend(settings.detector)
        detections = backend.detect(
            prepared_pages,
            checkpoint=settings.checkpoint,
            device=settings.device,
            confidence=settings.detector_confidence,
            max_detections=settings.max_detections,
            input_size=settings.input_size,
            work_directory=work_directory,
            model_config=settings.model_config,
        )

        print("\n[recognize text]", flush=True)
        processed_pages = recognize_pages(
            prepared_pages,
            detections,
            tesseract_command=settings.tesseract_command,
            language=settings.language,
            psm=settings.psm,
            token_confidence=settings.token_confidence,
        )

        grouped_pages: dict[str, list] = defaultdict(list)
        for page in processed_pages:
            grouped_pages[page.document_key].append(page)

        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_command
        ocr_metadata = {
            "engine": "tesseract",
            "version": str(pytesseract.get_tesseract_version()),
            "language": settings.language,
            "psm": settings.psm,
            "token_confidence": settings.token_confidence,
        }
        documents: list[ProcessedDocument] = []

        print("\n[export documents]", flush=True)
        for document_key, pages in grouped_pages.items():
            name = settings.output_name or document_key
            document = ProcessedDocument(
                name=name,
                detector=detections.detector_name,
                detector_metadata=detections.metadata,
                ocr_metadata=ocr_metadata,
                pages=sorted(pages, key=lambda page: page.page_number),
            )
            for output_format in sorted(settings.formats):
                EXPORTERS[output_format](document, output)
            documents.append(document)
            print(f"Exported {name}: {', '.join(sorted(settings.formats))}")

        _write_run_config(settings)
        return PipelineResult(output_directory=output, documents=documents)
    finally:
        if not settings.preserve_intermediates:
            shutil.rmtree(work_directory, ignore_errors=True)
