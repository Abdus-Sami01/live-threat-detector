"""FrameSource abstraction: decouples the engine from where frames come from.

Offline file source now; an RTSP/webcam source implements the same iterator
protocol later with zero change to the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np


@dataclass
class Frame:
    index: int
    time_s: float
    image: np.ndarray  # HxWx3 uint8 BGR


class FrameSource(Protocol):
    fps: float

    def __iter__(self) -> Iterator[Frame]: ...


class VideoFileSource:
    """Reads frames from a video file via OpenCV, yielding timestamped frames."""

    def __init__(self, path: str, stride: int = 1):
        import cv2

        self._cv2 = cv2
        self.path = path
        self.stride = max(1, stride)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            raise FileNotFoundError(f"cannot open video: {path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._cap = cap

    def __iter__(self) -> Iterator[Frame]:
        cv2 = self._cv2
        idx = 0
        try:
            while True:
                ok, img = self._cap.read()
                if not ok:
                    break
                if idx % self.stride == 0:
                    yield Frame(index=idx, time_s=idx / self.fps, image=img)
                idx += 1
        finally:
            self._cap.release()


class ArrayFrameSource:
    """In-memory frame source for tests and synthetic clips."""

    def __init__(self, images: list[np.ndarray], fps: float = 25.0):
        self.images = images
        self.fps = fps

    def __iter__(self) -> Iterator[Frame]:
        for idx, img in enumerate(self.images):
            yield Frame(index=idx, time_s=idx / self.fps, image=img)
