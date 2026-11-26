"""Ultralytics YOLO26-S detector backend."""
from __future__ import annotations

from pathlib import Path

from ..models import BoundingBox, Detection, PreparedPage
from .base import DetectionBackend, DetectionResult
from .common import EXPECTED_LABEL_NAMES, action_for, file_sha256


class Yolo26Backend(DetectionBackend):
    @staticmethod
    def _normalize_names(names) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(index): str(name) for index, name in names.items()}
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}
        raise TypeError(f"Unexpected Ultralytics class-name type: {type(names)!r}")

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
        del work_directory, model_config
        try:
            import ultralytics
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "YOLO26 requires Ultralytics. Install it with: python -m pip install ultralytics"
            ) from error

        model = YOLO(str(checkpoint))
        names = self._normalize_names(model.names)
        observed_names = [names[index] for index in sorted(names)]
        if observed_names != EXPECTED_LABEL_NAMES:
            raise RuntimeError(
                "Unexpected YOLO class mapping. "
                f"Expected {EXPECTED_LABEL_NAMES}, found {observed_names}"
            )

        device_argument: str | int = 0 if device == "cuda" else "cpu"
        detections_by_page: dict[str, list[Detection]] = {}
        paths = [str(page.image_path) for page in prepared_pages]
        results = model.predict(
            source=paths,
            imgsz=input_size,
            conf=confidence,
            max_det=max_detections,
            device=device_argument,
            verbose=False,
            save=False,
            stream=False,
        )

        if len(results) != len(prepared_pages):
            raise RuntimeError(
                f"Ultralytics returned {len(results)} results for {len(prepared_pages)} pages"
            )

        for page, result in zip(prepared_pages, results, strict=True):
            detections: list[Detection] = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu()
                scores = result.boxes.conf.detach().cpu()
                classes = result.boxes.cls.detach().cpu()
                order = scores.argsort(descending=True)

                for index in order.tolist():
                    score = float(scores[index])
                    if score < confidence:
                        continue
                    if len(detections) >= max_detections:
                        break

                    class_index = int(classes[index])
                    if class_index not in names:
                        raise RuntimeError(f"Unknown YOLO class index: {class_index}")
                    label = names[class_index]
                    x1, y1, x2, y2 = map(float, boxes[index].tolist())
                    detections.append(
                        Detection(
                            label=label,
                            bbox=BoundingBox(x1, y1, x2 - x1, y2 - y1),
                            score=score,
                            action=action_for(label),
                        )
                    )

            detections_by_page[page.image_key] = detections
            print(f"Detected {len(detections)} regions in {page.image_path.name}")

        return DetectionResult(
            detector_name="yolo26s-doclaynet",
            metadata={
                "backend": "yolo26",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "ultralytics": ultralytics.__version__,
                "device": device,
                "input_size": input_size,
                "confidence": confidence,
                "max_detections": max_detections,
                "class_names": names,
            },
            pages=detections_by_page,
        )
