"""Structured JSON exporter."""
import json
from pathlib import Path

from ..models import ProcessedDocument


def export(document: ProcessedDocument, output_directory: Path) -> None:
    path = output_directory / f"{document.name}.json"
    path.write_text(
        json.dumps(document.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
