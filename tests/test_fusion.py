import numpy as np

from threat_detector.core.events import Detection
from threat_detector.fusion.bayes_fuser import fuse, softmax, ACTIONS


def test_softmax_sums_to_one():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(p.sum() - 1.0) < 1e-9


def test_weapon_detection_raises_fight_posterior():
    logits = np.array([0.5, 0.5, 0.5, 0.5])  # flat likelihood over ACTIONS
    no_det = fuse(logits, [])
    with_weapon = fuse(logits, [Detection("weapon", 0.9, (0, 0, 10, 10))])
    assert with_weapon.posterior["fight"] > no_det.posterior["fight"]


def test_fire_suppresses_normal():
    logits = np.zeros(len(ACTIONS))
    base = fuse(logits, [])
    fire = fuse(logits, [Detection("fire", 0.8, (0, 0, 10, 10))])
    assert fire.posterior["normal"] < base.posterior["normal"]


def test_posterior_normalized():
    r = fuse(np.array([2.0, 1.0, 0.0, 3.0]), [Detection("weapon", 0.5, (0, 0, 5, 5))])
    assert abs(sum(r.posterior.values()) - 1.0) < 1e-9
    assert r.label in ACTIONS
