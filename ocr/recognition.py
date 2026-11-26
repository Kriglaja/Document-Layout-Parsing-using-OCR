"""Tesseract OCR over detected text-bearing regions."""
from __future__ import annotations

from pathlib import Path

import pytesseract
from PIL import Image
from pytesseract import Output

from .detection import DetectionResult
from .models import BoundingBox, OcrToken, PreparedPage, ProcessedPage, Region


def padded_box(box: BoundingBox, width: int, height: int) -> BoundingBox:
    pad_x = min(20, max(4, round(box.width * 0.02)))
    pad_y = min(20, max(4, round(box.height * 0.02)))
    padded = BoundingBox(
        box.x - pad_x,
        box.y - pad_y,
        box.width + 2 * pad_x,
        box.height + 2 * pad_y,
    )
    clipped = padded.clipped(width, height)
    if clipped is None:
        raise ValueError("Padding produced an invalid crop")
    return clipped


def recognize_crop(
    crop: Image.Image,
    *,
    language: str,
    psm: int,
    minimum_confidence: float,
    offset_x: float,
    offset_y: float,
) -> list[OcrToken]:
    data = pytesseract.image_to_data(
        crop,
        lang=language,
        config=f"--psm {psm}",
        output_type=Output.DICT,
    )
    tokens: list[OcrToken] = []

    for index, raw_text in enumerate(data["text"]):
        text = str(raw_text).strip()
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1.0

        if not text or confidence < minimum_confidence:
            continue

        tokens.append(
            OcrToken(
                text=text,
                confidence=confidence,
                bbox=BoundingBox(
                    offset_x + int(data["left"][index]),
                    offset_y + int(data["top"][index]),
                    int(data["width"][index]),
                    int(data["height"][index]),
                ),
                block_number=int(data["block_num"][index]),
                paragraph_number=int(data["par_num"][index]),
                line_number=int(data["line_num"][index]),
            )
        )
    return tokens


def recognize_pages(
    prepared_pages: list[PreparedPage],
    detections: DetectionResult,
    *,
    tesseract_command: str,
    language: str,
    psm: int,
    token_confidence: float,
) -> list[ProcessedPage]:
    pytesseract.pytesseract.tesseract_cmd = tesseract_command
    processed_pages: list[ProcessedPage] = []

    for page in prepared_pages:
        with Image.open(page.image_path) as source:
            image = source.convert("RGB")
        regions: list[Region] = []

        for detection in detections.pages.get(page.image_key, []):
            clipped_box = detection.bbox.clipped(page.width, page.height)
            if clipped_box is None:
                continue

            tokens: list[OcrToken] = []
            if detection.action == "ocr":
                crop_box = padded_box(clipped_box, page.width, page.height)
                crop_rectangle = (
                    int(crop_box.x),
                    int(crop_box.y),
                    int(crop_box.x + crop_box.width),
                    int(crop_box.y + crop_box.height),
                )
                tokens = recognize_crop(
                    image.crop(crop_rectangle),
                    language=language,
                    psm=psm,
                    minimum_confidence=token_confidence,
                    offset_x=crop_box.x,
                    offset_y=crop_box.y,
                )

            regions.append(
                Region(
                    label=detection.label,
                    action=detection.action,
                    score=detection.score,
                    bbox=clipped_box,
                    tokens=tokens,
                )
            )

        processed_pages.append(
            ProcessedPage(
                document_key=page.document_key,
                page_number=page.page_number,
                image_key=page.image_key,
                image_path=page.image_path,
                width=page.width,
                height=page.height,
                regions=regions,
            )
        )
        print(f"Recognized text in {page.image_path.name}")

    return processed_pages
