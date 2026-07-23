"""N2 Bayesian cross-branch fusion.

The action branch supplies a likelihood over actions, L_a = softmax(logits).
Per-frame detections shape a prior pi_a: a visible weapon raises the fight prior,
fire/smoke raises the panic (non-normal) mass. The fused posterior is

    P(a | clip, det) ∝ L_a · pi_a

normalized over actions. This is cheaper than a learned late-fusion head and its
decisions are inspectable: every boost traces to a detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.events import Detection

ACTIONS = ("fight", "fall", "loiter", "normal")


@dataclass
class FusionConfig:
    base_prior: dict[str, float] = field(
        default_factory=lambda: {"fight": 0.15, "fall": 0.15, "loiter": 0.15, "normal": 0.55}
    )
    weapon_boosts_fight: float = 3.0
    fire_smoke_suppresses_normal: float = 2.0


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def _detection_prior(detections: list[Detection], cfg: FusionConfig) -> np.ndarray:
    prior = np.array([cfg.base_prior[a] for a in ACTIONS], dtype=np.float64)
    d_weapon = max((d.confidence for d in detections if d.label == "weapon"), default=0.0)
    d_hazard = max(
        (d.confidence for d in detections if d.label in ("fire", "smoke")), default=0.0
    )
    fight_i, normal_i = ACTIONS.index("fight"), ACTIONS.index("normal")
    prior[fight_i] *= 1.0 + cfg.weapon_boosts_fight * d_weapon
    prior[normal_i] /= 1.0 + cfg.fire_smoke_suppresses_normal * d_hazard
    s = prior.sum()
    return prior / s if s > 0 else prior


@dataclass
class FusionResult:
    label: str
    confidence: float
    posterior: dict[str, float]


def fuse(
    action_logits: np.ndarray,
    detections: list[Detection],
    cfg: FusionConfig | None = None,
) -> FusionResult:
    cfg = cfg or FusionConfig()
    likelihood = softmax(np.asarray(action_logits, dtype=np.float64))
    prior = _detection_prior(detections, cfg)
    joint = likelihood * prior
    s = joint.sum()
    posterior = joint / s if s > 0 else likelihood
    i = int(posterior.argmax())
    return FusionResult(
        label=ACTIONS[i],
        confidence=float(posterior[i]),
        posterior={a: float(p) for a, p in zip(ACTIONS, posterior)},
    )
