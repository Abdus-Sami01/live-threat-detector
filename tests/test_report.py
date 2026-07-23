import numpy as np
from threat_detector.eval.report import build_report


def test_build_report_fields():
    normal = np.full(100, 0.1)
    anomaly = np.full(20, 0.9)
    r = build_report(normal, anomaly, normal_hours=10.0, budget_per_hour=1.0)
    assert set(r) >= {"threshold", "false_alarms_per_hour", "recall", "n_normal", "n_anomaly"}
    assert r["n_normal"] == 100 and r["n_anomaly"] == 20
    assert r["recall"] == 1.0
