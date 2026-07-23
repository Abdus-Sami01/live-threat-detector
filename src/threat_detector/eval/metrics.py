"""Evaluation metrics: action accuracy, detection mAP@0.5, latency, compute saved.

False-alarms/hour lives in calib.alarm_budget since it is tied to the operating
point; everything here is stateless and unit-testable on small arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..fusion.loiter_baseline import iou


def top1_accuracy(pred: list[str], true: list[str]) -> float:
    if not true:
        return 0.0
    return float(np.mean([p == t for p, t in zip(pred, true)]))


def _ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """VOC-style area under the precision-recall curve (monotone envelope)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def average_precision(
    preds: list[tuple[tuple[float, float, float, float], float]],
    gts: list[tuple[float, float, float, float]],
    iou_thr: float = 0.5,
) -> float:
    """AP@iou for one class. preds = [(box, score)], gts = [box]."""
    if not gts:
        return 0.0
    if not preds:
        return 0.0
    order = sorted(range(len(preds)), key=lambda i: preds[i][1], reverse=True)
    matched = set()
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for rank, i in enumerate(order):
        box = preds[i][0]
        best_j, best_iou = -1, iou_thr
        for j, gt in enumerate(gts):
            if j in matched:
                continue
            v = iou(box, gt)
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            tp[rank] = 1
            matched.add(best_j)
        else:
            fp[rank] = 1
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    recall = tp_c / len(gts)
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    return _ap(recall, precision)


def mean_ap(per_class_ap: dict[str, float]) -> float:
    return float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0


@dataclass
class LatencyReport:
    detect_ms: float
    classify_ms: float
    fuse_ms: float

    @property
    def total_ms(self) -> float:
        return self.detect_ms + self.classify_ms + self.fuse_ms

    @property
    def fps(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms > 0 else 0.0

    @property
    def realtime(self) -> bool:
        return self.fps >= 15.0


def compute_saved(windows_run: int, windows_total: int) -> float:
    if windows_total <= 0:
        return 0.0
    return 1.0 - windows_run / windows_total
