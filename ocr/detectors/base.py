"""Common detector backend interface and result type."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import Detection, PreparedPage


@dataclass(slots=True)
class DetectionResult:
    detector_name: str
    metadata: dict[str, Any]
    pages: dict[str, list[Detection]]


class DetectionBackend(ABC):
    @abstractmethod
    def detect(
        self,
        prepared_pages: list[PreparedPage],
        *,
        checkpoint: Path,
        device: str,
        confidence: float,
        max_detections: int,
        input_size: int,
        work_directory: Path,
        model_config: Path | None,
    ) -> DetectionResult:
        """Detect layout regions in prepared document pages."""
