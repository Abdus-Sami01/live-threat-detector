"""N4 scene-adaptive loitering.

A fixed dwell-time threshold flags a bus stop and a bank vault alike. Instead we
track people with a cheap greedy-IoU tracker, accumulate per-track dwell time, and
learn an online per-scene dwell baseline (running mean/variance). A track is a
loiterer when its dwell exceeds `mean + k*std` of the scene it lives in — normalized
to how that scene usually behaves.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.events import Detection

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    box: Box
    dwell_s: float = 0.0
    misses: int = 0


@dataclass
class LoiterConfig:
    iou_match: float = 0.3
    max_misses: int = 15
    k_sigma: float = 2.0
    warmup: int = 20  # completed tracks before the baseline is trusted


class _RunningStats:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def push(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self._m2 += d * (x - self.mean)

    @property
    def std(self) -> float:
        return (self._m2 / self.n) ** 0.5 if self.n > 1 else 0.0


class LoiterTracker:
    """Greedy-IoU person tracker with an online per-scene dwell baseline."""

    def __init__(self, fps: float, cfg: LoiterConfig | None = None):
        self.fps = fps
        self.cfg = cfg or LoiterConfig()
        self._tracks: dict[int, Track] = {}
        self._next_id = 0
        self._stats = _RunningStats()

    def _threshold(self) -> float:
        if self._stats.n < self.cfg.warmup:
            return float("inf")
        return self._stats.mean + self.cfg.k_sigma * self._stats.std

    def update(self, detections: list[Detection]) -> list[int]:
        """Advance one frame; return track ids currently flagged as loitering."""
        people = [d for d in detections if d.label == "person"]
        dt = 1.0 / self.fps
        matched: set[int] = set()
        used: set[int] = set()

        for tid, tr in self._tracks.items():
            best_j, best_iou = -1, self.cfg.iou_match
            for j, det in enumerate(people):
                if j in used:
                    continue
                v = iou(tr.box, det.box)
                if v >= best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0:
                tr.box = people[best_j].box
                tr.dwell_s += dt
                tr.misses = 0
                used.add(best_j)
                matched.add(tid)

        for tid, tr in self._tracks.items():
            if tid not in matched:
                tr.misses += 1

        for j, det in enumerate(people):
            if j not in used:
                self._tracks[self._next_id] = Track(self._next_id, det.box, dt)
                self._next_id += 1

        for tid in [t for t, tr in self._tracks.items() if tr.misses > self.cfg.max_misses]:
            self._stats.push(self._tracks[tid].dwell_s)
            del self._tracks[tid]

        thr = self._threshold()
        return [tid for tid, tr in self._tracks.items() if tr.dwell_s > thr]
