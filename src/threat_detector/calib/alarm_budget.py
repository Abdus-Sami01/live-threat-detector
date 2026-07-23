"""N3 alarm-budget calibration.

Production security teams do not care about an ROC point in the abstract; they care
about how many times a night the system pages them. This picks the lowest decision
threshold whose false-alarm rate on normal footage stays within a stated budget
(alarms per hour), then reports the recall that threshold buys — turning a vague
"tune it" into a single, auditable operating point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OperatingPoint:
    threshold: float
    false_alarms_per_hour: float
    recall: float


def false_alarms_per_hour(
    scores: np.ndarray, threshold: float, normal_hours: float
) -> float:
    if normal_hours <= 0:
        return 0.0
    alarms = int((np.asarray(scores) >= threshold).sum())
    return alarms / normal_hours


def calibrate(
    normal_scores: np.ndarray,
    normal_hours: float,
    anomaly_scores: np.ndarray | None = None,
    budget_per_hour: float = 1.0,
    grid: int = 101,
) -> OperatingPoint:
    """Lowest threshold whose FA/hr on normal footage is within budget.

    Lower threshold = higher recall, so the smallest budget-satisfying threshold is
    the most sensitive operating point the budget allows.
    """
    normal_scores = np.asarray(normal_scores, dtype=np.float64)
    thresholds = np.linspace(0.0, 1.0, grid)
    chosen = 1.0
    for t in thresholds:
        if false_alarms_per_hour(normal_scores, t, normal_hours) <= budget_per_hour:
            chosen = float(t)
            break
    fa = false_alarms_per_hour(normal_scores, chosen, normal_hours)
    recall = 0.0
    if anomaly_scores is not None and len(anomaly_scores) > 0:
        recall = float((np.asarray(anomaly_scores) >= chosen).mean())
    return OperatingPoint(threshold=chosen, false_alarms_per_hour=fa, recall=recall)
