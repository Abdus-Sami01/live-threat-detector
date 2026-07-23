"""N1 cascade primitive: cheap frame-difference motion energy.

Motion energy is the mean absolute per-pixel change between consecutive
(grayscale, downscaled) frames, normalized to [0, 1]. Costs a subtract and a
mean — orders of magnitude cheaper than the 3D action branch it gates.
"""
from __future__ import annotations

import numpy as np


def _gray_small(img: np.ndarray, size: int = 64) -> np.ndarray:
    if img.ndim == 3:
        img = img.mean(axis=2)
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img.astype(np.float32)
    step_h = max(1, h // size)
    step_w = max(1, w // size)
    return img[::step_h, ::step_w].astype(np.float32)


class MotionEnergy:
    """Stateful running motion-energy estimator over a frame stream."""

    def __init__(self, size: int = 64):
        self.size = size
        self._prev: np.ndarray | None = None

    def update(self, img: np.ndarray) -> float:
        cur = _gray_small(img, self.size)
        if self._prev is None or self._prev.shape != cur.shape:
            self._prev = cur
            return 0.0
        energy = float(np.abs(cur - self._prev).mean() / 255.0)
        self._prev = cur
        return energy
