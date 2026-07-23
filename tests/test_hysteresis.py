from threat_detector.fusion.hysteresis import HysteresisGate, HysteresisConfig


def test_single_spike_does_not_alarm():
    g = HysteresisGate(HysteresisConfig(min_consecutive=3))
    assert not g.update(0.99)
    assert not g.active


def test_sustained_high_raises_alarm():
    g = HysteresisGate(HysteresisConfig(ema_alpha=1.0, raise_thr=0.6, min_consecutive=3))
    states = [g.update(0.9) for _ in range(3)]
    assert states[-1] is True


def test_hysteresis_holds_through_dip_above_clear():
    g = HysteresisGate(HysteresisConfig(ema_alpha=1.0, raise_thr=0.6, clear_thr=0.35, min_consecutive=2))
    for _ in range(2):
        g.update(0.9)
    assert g.active
    assert g.update(0.5) is True  # above clear_thr, stays latched


def test_clears_below_clear_threshold():
    g = HysteresisGate(HysteresisConfig(ema_alpha=1.0, raise_thr=0.6, clear_thr=0.35, min_consecutive=2))
    for _ in range(2):
        g.update(0.9)
    assert g.active
    assert g.update(0.1) is False
