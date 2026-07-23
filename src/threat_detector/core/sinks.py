"""Output sinks: structured event log (JSON) and annotated video writer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .events import Detection, ThreatEvent

_COLORS = {
    "person": (0, 200, 0),
    "weapon": (0, 0, 255),
    "fire": (0, 140, 255),
    "smoke": (180, 180, 180),
}


def write_event_log(events: list[ThreatEvent], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in events], f, indent=2)


def draw_overlay(image: np.ndarray, detections: list[Detection], banner: str | None):
    import cv2

    out = image.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        color = _COLORS.get(d.label, (255, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{d.label} {d.confidence:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if banner:
        cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 255), -1)
        cv2.putText(out, banner, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


class AnnotatedVideoWriter:
    """Writes frames with detection boxes and alert banners to an mp4."""

    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        import cv2

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._w = cv2.VideoWriter(path, fourcc, fps, size)

    def write(self, image, detections, banner=None):
        self._w.write(draw_overlay(image, detections, banner))

    def close(self):
        self._w.release()
