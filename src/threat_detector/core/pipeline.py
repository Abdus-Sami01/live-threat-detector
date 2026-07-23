"""Cascade orchestrator: ties the branches and novelty modules into one stream.

Per frame: run the cheap detector, accumulate a 16-frame window, and only invoke
the expensive action branch when the N1 gate (motion OR detection) fires. Detections
prior-shape the action posterior (N2); a per-class hysteresis gate confirms alarms
(N3); a scene-adaptive tracker supplies loitering independently (N4). Latency and
compute-saved are recorded as the stream runs.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from ..detection.yolo_branch import DetectorProtocol, max_threat_conf
from ..action.x3d_branch import RecognizerProtocol
from ..fusion.bayes_fuser import FusionConfig, fuse
from ..fusion.hysteresis import HysteresisConfig, HysteresisGate
from ..fusion.loiter_baseline import LoiterConfig, LoiterTracker
from .events import Detection, ThreatEvent, ThreatType
from .frames import Frame, FrameSource
from .motion import MotionEnergy


@dataclass
class PipelineConfig:
    window: int = 16
    tau_motion: float = 0.02
    tau_detect: float = 0.4
    decay: float = 0.85  # posterior decay when the action branch is skipped
    fusion: FusionConfig = field(default_factory=FusionConfig)
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)
    loiter: LoiterConfig = field(default_factory=LoiterConfig)


@dataclass
class PipelineStats:
    frames: int = 0
    windows_total: int = 0
    windows_run: int = 0
    detect_ms: float = 0.0
    classify_ms: float = 0.0
    fuse_ms: float = 0.0

    @property
    def compute_saved(self) -> float:
        if self.windows_total == 0:
            return 0.0
        return 1.0 - self.windows_run / self.windows_total


class Pipeline:
    """Streaming threat pipeline over any FrameSource."""

    def __init__(
        self,
        detector: DetectorProtocol,
        recognizer: RecognizerProtocol,
        fps: float,
        cfg: PipelineConfig | None = None,
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.cfg = cfg or PipelineConfig()
        self.motion = MotionEnergy()
        self.loiter = LoiterTracker(fps=fps, cfg=self.cfg.loiter)
        self._gates: dict[str, HysteresisGate] = {}
        self._buf: deque[np.ndarray] = deque(maxlen=self.cfg.window)
        self._last_posterior = np.array([0.0, 0.0, 0.0, 1.0])  # start "normal"
        self.stats = PipelineStats()

    def _gate_for(self, label: str) -> HysteresisGate:
        if label not in self._gates:
            self._gates[label] = HysteresisGate(self.cfg.hysteresis)
        return self._gates[label]

    def _should_run_action(self, motion: float, detections: list[Detection]) -> bool:
        return (
            motion > self.cfg.tau_motion
            or max_threat_conf(detections) > self.cfg.tau_detect
        )

    def process_frame(self, frame: Frame) -> ThreatEvent | None:
        c = self.cfg
        self.stats.frames += 1
        img = frame.image
        self._buf.append(img)

        t0 = perf_counter()
        detections = self.detector.detect(img)
        self.stats.detect_ms += (perf_counter() - t0) * 1000
        motion = self.motion.update(img)
        loiter_ids = self.loiter.update(detections)

        # per-frame detections raise their own instant-confirm alarms
        for d in detections:
            if d.label in ("weapon", "fire", "smoke"):
                if self._gate_for(d.label).update(d.confidence):
                    return ThreatEvent(
                        threat=ThreatType(d.label),
                        confidence=d.confidence,
                        frame_index=frame.index,
                        time_s=frame.time_s,
                        detections=detections,
                        meta={"motion": motion},
                    )

        if len(self._buf) < c.window:
            return None

        self.stats.windows_total += 1
        run = self._should_run_action(motion, detections)
        if run:
            clip = np.stack(list(self._buf))
            t1 = perf_counter()
            logits = self.recognizer.classify(clip)
            self.stats.classify_ms += (perf_counter() - t1) * 1000
            t2 = perf_counter()
            result = fuse(logits, detections, c.fusion)
            self.stats.fuse_ms += (perf_counter() - t2) * 1000
            self._last_posterior = np.array(
                [result.posterior[a] for a in ("fight", "fall", "loiter", "normal")]
            )
            self.stats.windows_run += 1
            label, conf = result.label, result.confidence
        else:
            self._last_posterior = self._last_posterior * c.decay
            self._last_posterior[-1] += 1 - self._last_posterior.sum()  # bleed to normal
            i = int(self._last_posterior.argmax())
            label = ("fight", "fall", "loiter", "normal")[i]
            conf = float(self._last_posterior[i])

        if loiter_ids:
            label, conf = "loiter", max(conf, 0.9)

        active = self._gate_for(label).update(conf) if label != "normal" else False
        if active and label != "normal":
            return ThreatEvent(
                threat=ThreatType(label),
                confidence=conf,
                frame_index=frame.index,
                time_s=frame.time_s,
                detections=detections,
                meta={"motion": motion, "action_ran": run, "loiter_ids": loiter_ids},
            )
        return None

    def run(self, source: FrameSource) -> list[ThreatEvent]:
        return [e for f in source if (e := self.process_frame(f)) is not None]
