import numpy as np

from threat_detector.core.events import Detection
from threat_detector.core.motion import MotionEnergy
from threat_detector.fusion.loiter_baseline import LoiterTracker, LoiterConfig, iou


def test_motion_zero_on_static():
    m = MotionEnergy()
    img = np.full((64, 64, 3), 100, np.uint8)
    m.update(img)
    assert m.update(img) == 0.0


def test_motion_positive_on_change():
    m = MotionEnergy()
    m.update(np.zeros((64, 64, 3), np.uint8))
    assert m.update(np.full((64, 64, 3), 255, np.uint8)) > 0.5


def test_iou_identical():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_loiter_flags_long_dweller_after_warmup():
    trk = LoiterTracker(fps=10.0, cfg=LoiterConfig(warmup=3, k_sigma=1.0, max_misses=2))
    # seed baseline with short-dwell tracks that then leave
    for _ in range(4):
        for _ in range(2):
            trk.update([Detection("person", 0.9, (0, 0, 10, 10))])
        for _ in range(3):  # exceed max_misses so track completes into baseline
            trk.update([])
    # now a persistent dweller
    flagged = []
    for _ in range(60):
        flagged = trk.update([Detection("person", 0.9, (100, 100, 110, 110))])
    assert len(flagged) >= 1
