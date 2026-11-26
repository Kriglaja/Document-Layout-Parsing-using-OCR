"""RT-DETRv4 model loading and arbitrary-image inference."""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .models import BoundingBox, Detection, PreparedPage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RTDETR_ROOT = PROJECT_ROOT / "rtdetrv4"
LABELS = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "List-item",
    5: "Page-footer",
    6: "Page-header",
    7: "Picture",
    8: "Section-header",
    9: "Table",
    10: "Text",
    11: "Title",
}
VISUAL_LABELS = {"Formula", "Picture", "Table"}


@dataclass(slots=True)
class DetectionResult:
    detector_name: str
    metadata: dict[str, Any]
    pages: dict[str, list[Detection]]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_model_config(source: Path, destination: Path) -> Path:
    text = source.read_text(encoding="utf-8")
    base_config = (RTDETR_ROOT / "configs/rtv4/rtv4_hgnetv2_s_coco.yml").resolve()
    text = text.replace("./rtv4_hgnetv2_s_coco.yml", base_config.as_posix())
    destination.write_text(text, encoding="utf-8")
    return destination


def _load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    if str(RTDETR_ROOT) not in sys.path:
        sys.path.insert(0, str(RTDETR_ROOT))
    from engine.core import YAMLConfig

    config = YAMLConfig(str(config_path), device=str(device), output_dir=None)
    if "HGNetv2" in config.yaml_cfg:
        config.yaml_cfg["HGNetv2"]["pretrained"] = False

    model = config.model.to(device)
    postprocessor = config.postprocessor.to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    used_ema = False

    if "ema" in checkpoint:
        ema = config.ema
        if ema is None:
            raise RuntimeError("The checkpoint contains EMA weights, but the config defines no EMA model")
        ema.load_state_dict(checkpoint["ema"])
        model = ema.module.to(device)
        used_ema = True
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        raise RuntimeError(f"Checkpoint contains neither 'ema' nor 'model': {list(checkpoint)}")

    model.eval()
    postprocessor.eval()
    return model, postprocessor, used_ema


def detect_layout(
    prepared_pages: list[PreparedPage],
    *,
    checkpoint: Path,
    config: Path,
    resolved_config_path: Path,
    device_name: str,
    confidence: float,
    max_detections: int,
    input_size: int = 1024,
) -> DetectionResult:
    device = torch.device(device_name)
    model, postprocessor, used_ema = _load_model(
        resolved_model_config(config, resolved_config_path),
        checkpoint,
        device,
    )
    detections_by_page: dict[str, list[Detection]] = {}

    with torch.inference_mode():
        for page in prepared_pages:
            with Image.open(page.image_path) as source:
                image = source.convert("RGB")
            tensor = TF.pil_to_tensor(
                TF.resize(image, [input_size, input_size], antialias=True)
            ).float().div(255).unsqueeze(0).to(device)

            outputs = model(tensor)
            original_size = torch.tensor(
                [[page.width, page.height]],
                dtype=torch.float32,
                device=device,
            )
            result = postprocessor(outputs, original_size)[0]
            order = torch.argsort(result["scores"], descending=True)
            detections: list[Detection] = []

            for index in order.tolist():
                score = float(result["scores"][index])
                if score < confidence:
                    continue
                if len(detections) >= max_detections:
                    break

                raw_label = int(result["labels"][index])
                category_id = raw_label if raw_label in LABELS else raw_label + 1
                if category_id not in LABELS:
                    raise RuntimeError(f"Unknown postprocessor label: {raw_label}")

                x1, y1, x2, y2 = map(float, result["boxes"][index].tolist())
                label = LABELS[category_id]
                detections.append(
                    Detection(
                        label=label,
                        bbox=BoundingBox(x1, y1, x2 - x1, y2 - y1),
                        score=score,
                        action="visual" if label in VISUAL_LABELS else "ocr",
                    )
                )

            detections_by_page[page.image_key] = detections
            print(f"Detected {len(detections)} regions in {page.image_path.name}")

    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "used_ema": used_ema,
        "device": str(device),
        "input_size": input_size,
        "confidence": confidence,
        "max_detections": max_detections,
    }
    return DetectionResult(
        detector_name="rtdetrv4-s-doclaynet-seed42-best",
        metadata=metadata,
        pages=detections_by_page,
    )
