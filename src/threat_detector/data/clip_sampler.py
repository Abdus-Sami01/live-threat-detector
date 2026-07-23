"""Uniform temporal clip sampling: decode a video into a fixed-length frame stack."""
from __future__ import annotations

import numpy as np


def sample_clip(path: str, num_frames: int = 16, size: int = 224) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(img, (size, size)))
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded: {path}")
    idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    return np.stack([frames[i] for i in idx]).astype(np.uint8)
