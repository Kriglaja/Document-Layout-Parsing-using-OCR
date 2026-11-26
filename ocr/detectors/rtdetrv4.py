"""RT-DETRv4-S detector backend."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from ..models import BoundingBox, Detection, PreparedPage
from .base import DetectionBackend, DetectionResult
from .common import DOCLAYNET_LABELS, action_for, file_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RTDETR_ROOT = PROJECT_ROOT / "rtdetrv4"


class Rtdetrv4Backend(DetectionBackend):
    @staticmethod
    def _resolve_config(source: Path, destination: Path) -> Path:
        text = source.read_text(encoding="utf-8")
        base_config = (RTDETR_ROOT / "configs/rtv4/rtv4_hgnetv2_s_coco.yml").resolve()
        text = text.replace("./rtv4_hgnetv2_s_coco.yml", base_config.as_posix())
        destination.write_text(text, encoding="utf-8")
        return destination

    @staticmethod
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
        if model_config is None:
            raise ValueError("RT-DETRv4 requires a runtime configuration file")

        torch_device = torch.device(device)
        resolved_config = self._resolve_config(
            model_config,
            work_directory / "rtdetrv4-runtime-config.yml",
        )
        model, postprocessor, used_ema = self._load_model(
            resolved_config,
            checkpoint,
            torch_device,
        )
        detections_by_page: dict[str, list[Detection]] = {}

        with torch.inference_mode():
            for page in prepared_pages:
                with Image.open(page.image_path) as source:
                    image = source.convert("RGB")
                tensor = TF.pil_to_tensor(
                    TF.resize(image, [input_size, input_size], antialias=True)
                ).float().div(255).unsqueeze(0).to(torch_device)

                outputs = model(tensor)
                original_size = torch.tensor(
                    [[page.width, page.height]],
                    dtype=torch.float32,
                    device=torch_device,
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
                    category_id = raw_label if raw_label in DOCLAYNET_LABELS else raw_label + 1
                    if category_id not in DOCLAYNET_LABELS:
                        raise RuntimeError(f"Unknown RT-DETR label: {raw_label}")

                    x1, y1, x2, y2 = map(float, result["boxes"][index].tolist())
                    label = DOCLAYNET_LABELS[category_id]
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
            detector_name="rtdetrv4-s-doclaynet-seed42-best",
            metadata={
                "backend": "rtdetrv4",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "used_ema": used_ema,
                "device": str(torch_device),
                "input_size": input_size,
                "confidence": confidence,
                "max_detections": max_detections,
            },
            pages=detections_by_page,
        )
