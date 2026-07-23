"""Bundle calibration into a publishable FA/hour report."""
from __future__ import annotations

import numpy as np

from ..calib.alarm_budget import calibrate


def build_report(normal_scores, anomaly_scores, normal_hours: float,
                 budget_per_hour: float = 1.0) -> dict:
    normal_scores = np.asarray(normal_scores, dtype=np.float64)
    anomaly_scores = np.asarray(anomaly_scores, dtype=np.float64)
    op = calibrate(normal_scores, normal_hours, anomaly_scores, budget_per_hour)
    return {
        "threshold": op.threshold,
        "false_alarms_per_hour": op.false_alarms_per_hour,
        "recall": op.recall,
        "budget_per_hour": budget_per_hour,
        "n_normal": int(normal_scores.size),
        "n_anomaly": int(anomaly_scores.size),
    }
