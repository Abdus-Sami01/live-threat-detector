"""Clip-level action-recognition branch (fight / fall / loiter / normal).

Wraps an X3D-M backbone (PyTorchVideo, Kinetics-pretrained) with a 4-way head.
Returns raw logits over `bayes_fuser.ACTIONS` so fusion owns the softmax + prior.
Torch is imported lazily; a `RecognizerProtocol` allows a fake in tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..fusion.bayes_fuser import ACTIONS


class RecognizerProtocol(Protocol):
    def classify(self, clip: np.ndarray) -> np.ndarray: ...


@dataclass
class X3DConfig:
    num_frames: int = 16
    crop: int = 224
    mean: tuple[float, float, float] = (0.45, 0.45, 0.45)
    std: tuple[float, float, float] = (0.225, 0.225, 0.225)
    device: str | None = None
    checkpoint: str | None = None  # fine-tuned head weights


class X3DRecognizer:
    """X3D-M with a fine-tuned 4-class head. Loads on first use."""

    def __init__(self, cfg: X3DConfig | None = None):
        self.cfg = cfg or X3DConfig()
        self._model = None
        self._torch = None

    def _load(self):
        if self._model is None:
            import torch
            from pytorchvideo.models.hub import x3d_m

            self._torch = torch
            model = x3d_m(pretrained=True)
            model.blocks[-1].proj = torch.nn.Linear(
                model.blocks[-1].proj.in_features, len(ACTIONS)
            )
            if self.cfg.checkpoint:
                model.load_state_dict(torch.load(self.cfg.checkpoint, map_location="cpu"))
            model.eval()
            if self.cfg.device:
                model.to(self.cfg.device)
            self._model = model
        return self._model

    def _preprocess(self, clip: np.ndarray):
        torch = self._torch
        # clip: T,H,W,3 uint8 BGR -> 1,3,T,H,W normalized RGB
        x = torch.from_numpy(clip[..., ::-1].copy()).float() / 255.0
        x = x.permute(3, 0, 1, 2)  # C,T,H,W
        mean = torch.tensor(self.cfg.mean).view(3, 1, 1, 1)
        std = torch.tensor(self.cfg.std).view(3, 1, 1, 1)
        x = (x - mean) / std
        return x.unsqueeze(0)

    def classify(self, clip: np.ndarray) -> np.ndarray:
        model = self._load()
        torch = self._torch
        x = self._preprocess(clip)
        if self.cfg.device:
            x = x.to(self.cfg.device)
        with torch.no_grad():
            logits = model(x)[0]
        return logits.float().cpu().numpy()
