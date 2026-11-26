"""Command-line parsing and pipeline configuration."""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_FORMATS = ("png", "pdf", "docx", "odt", "json", "txt")


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    inputs: tuple[Path, ...]
    output_directory: Path
    output_name: str | None
    formats: frozenset[str]
    recursive: bool
    pdf_dpi: int
    selected_pages: str | None
    checkpoint: Path
    model_config: Path
    device: str
    detector_confidence: float
    max_detections: int
    language: str
    psm: int
    token_confidence: float
    tesseract_command: str
    preserve_intermediates: bool


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process images or PDFs with RT-DETRv4-S and Tesseract OCR."
    )
    parser.add_argument("input", nargs="+", type=Path, help="Image, PDF, or directory inputs")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--name", help="Base filename for a single logical document")
    parser.add_argument("--formats", nargs="+", choices=SUPPORTED_FORMATS, default=["png", "pdf"])
    parser.add_argument("--recursive", action="store_true", help="Search input directories recursively")
    parser.add_argument("--pdf-dpi", type=int, default=200)
    parser.add_argument("--pages", help="PDF pages, for example 1,3-5")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models/rtdetrv4/best_stg1.pth")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "models/rtdetrv4/runtime-config.yml")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--detector-confidence", type=float, default=0.50)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--lang", default="eng")
    parser.add_argument("--psm", type=int, default=6)
    parser.add_argument("--token-confidence", type=float, default=30)
    parser.add_argument("--tesseract-command", type=Path)
    parser.add_argument("--preserve-intermediates", action="store_true")
    return parser.parse_args()


def _find_tesseract(explicit: Path | None) -> str:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not candidate.is_file():
            raise SystemExit(f"Tesseract executable not found: {candidate}")
        return str(candidate)

    discovered = shutil.which("tesseract")
    if discovered:
        return discovered

    if sys.platform == "win32":
        common_location = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if common_location.is_file():
            return str(common_location)

    raise SystemExit(
        "Tesseract was not found. Add it to PATH or pass --tesseract-command PATH."
    )


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but the current PyTorch environment cannot access it.")
    return requested


def _safe_name(value: str | None) -> str | None:
    if value is None:
        return None
    forbidden = '<>:"/\\|?*' if sys.platform == "win32" else "/\x00"
    cleaned = "".join("-" if char in forbidden else char for char in value).strip(" .")
    if not cleaned or cleaned in {".", ".."}:
        raise SystemExit("--name must contain a usable filename")
    return cleaned


def resolve_settings(arguments: argparse.Namespace) -> PipelineSettings:
    checkpoint = arguments.checkpoint.expanduser().resolve()
    model_config = arguments.config.expanduser().resolve()
    for path in (checkpoint, model_config):
        if not path.is_file():
            raise SystemExit(f"Missing required model file: {path}")

    if not 0.0 <= arguments.detector_confidence <= 1.0:
        raise SystemExit("--detector-confidence must be in [0,1]")
    if not 0.0 <= arguments.token_confidence <= 100.0:
        raise SystemExit("--token-confidence must be in [0,100]")
    if arguments.max_detections < 1:
        raise SystemExit("--max-detections must be positive")
    if arguments.pdf_dpi < 72:
        raise SystemExit("--pdf-dpi must be at least 72")

    return PipelineSettings(
        inputs=tuple(path.expanduser() for path in arguments.input),
        output_directory=arguments.output.expanduser().resolve(),
        output_name=_safe_name(arguments.name),
        formats=frozenset(arguments.formats) | {"json"},
        recursive=arguments.recursive,
        pdf_dpi=arguments.pdf_dpi,
        selected_pages=arguments.pages,
        checkpoint=checkpoint,
        model_config=model_config,
        device=_resolve_device(arguments.device),
        detector_confidence=arguments.detector_confidence,
        max_detections=arguments.max_detections,
        language=arguments.lang,
        psm=arguments.psm,
        token_confidence=arguments.token_confidence,
        tesseract_command=_find_tesseract(arguments.tesseract_command),
        preserve_intermediates=arguments.preserve_intermediates,
    )
