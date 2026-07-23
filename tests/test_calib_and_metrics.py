import numpy as np

from threat_detector.calib.alarm_budget import calibrate, false_alarms_per_hour
from threat_detector.eval.metrics import (
    average_precision,
    compute_saved,
    top1_accuracy,
    LatencyReport,
)


def test_false_alarms_per_hour_counts():
    scores = np.array([0.1, 0.9, 0.95, 0.2])
    assert false_alarms_per_hour(scores, 0.5, normal_hours=2.0) == 1.0  # 2 alarms / 2h


def test_calibrate_respects_budget():
    normal = np.concatenate([np.full(99, 0.1), np.array([0.95])])  # one loud outlier
    op = calibrate(normal, normal_hours=10.0, budget_per_hour=0.5)
    assert op.false_alarms_per_hour <= 0.5


def test_calibrate_reports_recall():
    normal = np.full(100, 0.1)
    anomaly = np.full(50, 0.9)
    op = calibrate(normal, normal_hours=10.0, anomaly_scores=anomaly, budget_per_hour=1.0)
    assert op.recall == 1.0


def test_average_precision_perfect():
    gts = [(0, 0, 10, 10)]
    preds = [((0, 0, 10, 10), 0.9)]
    assert average_precision(preds, gts) == 1.0


def test_average_precision_no_overlap():
    gts = [(0, 0, 10, 10)]
    preds = [((50, 50, 60, 60), 0.9)]
    assert average_precision(preds, gts) == 0.0


def test_top1_accuracy():
    assert top1_accuracy(["a", "b", "c"], ["a", "x", "c"]) == 2 / 3


def test_compute_saved():
    assert compute_saved(20, 100) == 0.8


def test_latency_realtime_flag():
    r = LatencyReport(detect_ms=40, classify_ms=40, fuse_ms=1)
    assert not r.realtime
    fast = LatencyReport(detect_ms=5, classify_ms=5, fuse_ms=1)
    assert fast.realtime
