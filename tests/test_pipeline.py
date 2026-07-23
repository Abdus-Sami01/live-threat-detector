import numpy as np

from threat_detector.core.events import Detection
from threat_detector.core.frames import ArrayFrameSource
from threat_detector.core.pipeline import Pipeline, PipelineConfig
from threat_detector.fusion.hysteresis import HysteresisConfig


class FakeDetector:
    def __init__(self, dets=None):
        self.dets = dets or []

    def detect(self, image):
        return list(self.dets)


class FakeRecognizer:
    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=float)

    def classify(self, clip):
        return self.logits


def _frames(n, val=0, jitter=False):
    imgs = []
    for i in range(n):
        base = val + (i % 5 if jitter else 0)
        imgs.append(np.full((64, 64, 3), base % 256, np.uint8))
    return ArrayFrameSource(imgs, fps=25.0)


def test_quiet_scene_emits_no_alert_and_saves_compute():
    p = Pipeline(FakeDetector(), FakeRecognizer([0, 0, 0, 5]), fps=25.0,
                 cfg=PipelineConfig(window=16, tau_motion=0.02))
    events = p.run(_frames(60, val=100))  # static -> motion gate never fires
    assert events == []
    assert p.stats.compute_saved > 0.9


def test_weapon_detection_raises_alert():
    det = FakeDetector([Detection("weapon", 0.95, (0, 0, 10, 10))])
    p = Pipeline(det, FakeRecognizer([0, 0, 0, 5]), fps=25.0,
                 cfg=PipelineConfig(hysteresis=HysteresisConfig(min_consecutive=3, ema_alpha=1.0)))
    events = p.run(_frames(20, val=100))
    assert any(e.threat.value == "weapon" for e in events)


def test_active_scene_runs_action_branch():
    p = Pipeline(FakeDetector(), FakeRecognizer([10, 0, 0, 0]), fps=25.0,
                 cfg=PipelineConfig(window=16, tau_motion=0.001,
                                    hysteresis=HysteresisConfig(min_consecutive=2, ema_alpha=1.0)))
    events = p.run(_frames(40, jitter=True))
    assert p.stats.windows_run > 0
    assert any(e.threat.value == "fight" for e in events)
