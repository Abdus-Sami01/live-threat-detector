"""N3 hysteresis temporal filter.

A single confident window is not an alert. Confidence is smoothed with an EMA and
gated by two thresholds: an alarm is raised only after the smoothed score crosses
`raise_thr` for `min_consecutive` windows, and cleared only once it falls below
`clear_thr`. The gap between thresholds prevents flicker around a single boundary —
the behaviour that makes naive per-frame systems cry wolf.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HysteresisConfig:
    ema_alpha: float = 0.4
    raise_thr: float = 0.60
    clear_thr: float = 0.35
    min_consecutive: int = 3


class HysteresisGate:
    """Per-class latched alarm state driven by smoothed confidence."""

    def __init__(self, cfg: HysteresisConfig | None = None):
        self.cfg = cfg or HysteresisConfig()
        self._ema: float = 0.0
        self._above = 0
        self._active = False

    @property
    def score(self) -> float:
        return self._ema

    @property
    def active(self) -> bool:
        return self._active

    def update(self, confidence: float) -> bool:
        """Feed one window's confidence, return whether the alarm is active."""
        c = self.cfg
        self._ema = c.ema_alpha * confidence + (1 - c.ema_alpha) * self._ema
        if self._ema >= c.raise_thr:
            self._above += 1
        else:
            self._above = 0
        if not self._active:
            if self._above >= c.min_consecutive:
                self._active = True
        elif self._ema < c.clear_thr:
            self._active = False
        return self._active

    def reset(self) -> None:
        self._ema = 0.0
        self._above = 0
        self._active = False
