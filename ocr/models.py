"""Shared data structures used throughout the OCR pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.width, self.height]

    def clipped(self, image_width: int, image_height: int) -> "BoundingBox | None":
        x1 = max(0.0, self.x)
        y1 = max(0.0, self.y)
        x2 = min(float(image_width), self.x + self.width)
        y2 = min(float(image_height), self.y + self.height)
        if x2 <= x1 or y2 <= y1:
            return None
        return BoundingBox(x1, y1, x2 - x1, y2 - y1)


@dataclass(slots=True)
class PreparedPage:
    document_key: str
    page_number: int
    image_key: str
    image_path: Path
    width: int
    height: int
    source_path: Path


@dataclass(slots=True)
class Detection:
    label: str
    bbox: BoundingBox
    score: float
    action: str


@dataclass(slots=True)
class OcrToken:
    text: str
    confidence: float
    bbox: BoundingBox
    block_number: int
    paragraph_number: int
    line_number: int


@dataclass(slots=True)
class Region:
    label: str
    action: str
    score: float
    bbox: BoundingBox
    tokens: list[OcrToken] = field(default_factory=list)


@dataclass(slots=True)
class ProcessedPage:
    document_key: str
    page_number: int
    image_key: str
    image_path: Path
    width: int
    height: int
    regions: list[Region]


@dataclass(slots=True)
class ProcessedDocument:
    name: str
    detector: str
    detector_metadata: dict[str, Any]
    ocr_metadata: dict[str, Any]
    pages: list[ProcessedPage]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "document_name": self.name,
            "detector": self.detector,
            "detector_metadata": self.detector_metadata,
            "ocr": self.ocr_metadata,
            "pages": [
                {
                    "document_key": page.document_key,
                    "page_number": page.page_number,
                    "image_key": page.image_key,
                    "file_name": str(page.image_path),
                    "width": page.width,
                    "height": page.height,
                    "regions": [
                        {
                            "label": region.label,
                            "action": region.action,
                            "score": region.score,
                            "bbox_xywh": region.bbox.as_list(),
                            "tokens": [
                                {
                                    "text": token.text,
                                    "confidence": token.confidence,
                                    "bbox_xywh": token.bbox.as_list(),
                                    "block_num": token.block_number,
                                    "par_num": token.paragraph_number,
                                    "line_num": token.line_number,
                                }
                                for token in region.tokens
                            ],
                        }
                        for region in page.regions
                    ],
                }
                for page in self.pages
            ],
        }


@dataclass(slots=True)
class PipelineResult:
    output_directory: Path
    documents: list[ProcessedDocument]
