"""Detector backend registry."""

from .base import DetectionBackend
from .rtdetrv4 import Rtdetrv4Backend
from .yolo26 import Yolo26Backend


def create_backend(name: str) -> DetectionBackend:
    if name == "rtdetrv4":
        return Rtdetrv4Backend()
    if name == "yolo26":
        return Yolo26Backend()
    raise ValueError(f"Unsupported detector backend: {name}")


__all__ = ["DetectionBackend", "create_backend"]
