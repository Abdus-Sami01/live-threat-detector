"""Threat event schema shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class ThreatType(str, Enum):
    FIGHT = "fight"
    WEAPON = "weapon"
    FIRE = "fire"
    SMOKE = "smoke"
    FALL = "fall"
    LOITER = "loiter"
    NORMAL = "normal"


@dataclass
class Detection:
    """One per-frame object detection box in xyxy pixel coords."""
    label: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass
class ThreatEvent:
    """A fused, temporally-confirmed alert for one window."""
    threat: ThreatType
    confidence: float
    frame_index: int
    time_s: float
    detections: list[Detection] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_alert(self) -> bool:
        return self.threat is not ThreatType.NORMAL

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["threat"] = self.threat.value
        return d
