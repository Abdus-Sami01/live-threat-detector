import numpy as np
from threat_detector.eval.scorer import scores_from_events, score_split
from threat_detector.core.events import ThreatEvent, ThreatType


def test_scores_from_events_extracts_confidences():
    events = [
        ThreatEvent(ThreatType.FIGHT, 0.8, 10, 0.4),
        ThreatEvent(ThreatType.NORMAL, 0.2, 20, 0.8),
    ]
    s = scores_from_events(events)
    assert np.isclose(s.max(), 0.8)
    assert len(s) == 2


def test_scores_from_events_empty():
    assert len(scores_from_events([])) == 0


def test_score_split_concatenates():
    a = [ThreatEvent(ThreatType.FIGHT, 0.5, 1, 0.1)]
    b = [ThreatEvent(ThreatType.FALL, 0.7, 2, 0.2)]
    assert len(score_split([a, b])) == 2
