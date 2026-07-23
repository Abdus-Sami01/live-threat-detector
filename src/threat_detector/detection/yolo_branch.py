"""Per-frame object-detection branch (weapon / fire / smoke / person).

Wraps Ultralytics YOLO behind a narrow interface so the pipeline never imports the
model directly. Torch/Ultralytics are imported lazily; a `DetectorProtocol` lets
tests inject a fake without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..core.events import Detection

THREAT_LABELS = ("person", "weapon", "fire", "smoke")


class DetectorProtocol(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


@dataclass
class YoloConfig:
    weights: str = "yolov8n.pt"
    conf: float = 0.35
    imgsz: int = 640
    half: bool = True
    device: str | None = None
    # maps model class names to our threat vocabulary
    name_map: dict[str, str] | None = None


class YoloDetector:
    """Ultralytics YOLO wrapper. Loads on first use."""

    def __init__(self, cfg: YoloConfig | None = None):
        self.cfg = cfg or YoloConfig()
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.cfg.weights)
        return self._model

    def _map_name(self, name: str) -> str | None:
        if self.cfg.name_map:
            return self.cfg.name_map.get(name)
        return name if name in THREAT_LABELS else None

    def detect(self, image: np.ndarray) -> list[Detection]:
        model = self._load()
        res = model.predict(
            image,
            conf=self.cfg.conf,
            imgsz=self.cfg.imgsz,
            half=self.cfg.half,
            device=self.cfg.device,
            verbose=False,
        )[0]
        out: list[Detection] = []
        names = res.names
        for b in res.boxes:
            label = self._map_name(names[int(b.cls)])
            if label is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append(Detection(label, float(b.conf), (x1, y1, x2, y2)))
        return out


def max_threat_conf(detections: list[Detection]) -> float:
    """Cascade signal `d`: strongest non-person detection confidence."""
    threats = [d.confidence for d in detections if d.label != "person"]
    return max(threats, default=0.0)
