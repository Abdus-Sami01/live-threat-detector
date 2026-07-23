"""Turn pipeline alert events into per-window score arrays for FA/hour calibration."""
from __future__ import annotations

import numpy as np

from ..core.events import ThreatEvent


def scores_from_events(events: list[ThreatEvent]) -> np.ndarray:
    return np.array([e.confidence for e in events], dtype=np.float64)


def score_split(per_video_events: list[list[ThreatEvent]]) -> np.ndarray:
    if not per_video_events:
        return np.array([], dtype=np.float64)
    return np.concatenate([scores_from_events(ev) for ev in per_video_events])
