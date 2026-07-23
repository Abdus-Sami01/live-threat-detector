# Real-Time Video Threat Detector — Design Spec

Date: 2026-07-23
Status: Approved (sections 1–2 by user; remainder industry-standard defaults)

## 1. Goal

Detect fights, weapons, fire/smoke, falls, and loitering in CCTV-style video via
temporal action recognition + per-frame object detection, with a production-grade
false-alarm rate. Offline-first engine, streaming-ready, benchmarked on Modal GPUs.

Non-goals (v1): real webcam capture (no device), full UCF-Crime training,
physical Jetson hardware. Edge story handled via quantization + latency numbers.

## 2. Decisions

- **Scope**: full pipeline (both branches + fusion) built modular.
- **Data**: hybrid — pretrained backbones + small fine-tune sets; UCF-Crime for
  evaluation only (false-alarm-rate benchmark).
- **Runtime**: offline file-first; engine consumes a `FrameSource` abstraction so
  RTSP/webcam drops in later with zero core change.
- **Artifacts**: alert event log (JSON), annotated video, metrics report, minimal
  upload dashboard.
- **Compute**: Modal serverless GPU (A40/L40S/A100). FP16 inference.

## 3. Novelty (all four)

- **N1 Compute-adaptive cascade**: cheap frame-diff motion energy `m` + max YOLO
  threat conf `d` gate the expensive 3D branch. Run X3D only when
  `m > τ_m OR d > τ_d`; else reuse decayed last posterior. Report `compute_saved%`.
- **N2 Bayesian cross-branch fusion**: per-frame detections set a prior on the clip
  action posterior. `P(a|clip,det) ∝ L_a · π_a`, where `L_a` = softmax action
  likelihood and `π_a` = detection-shaped prior (weapon → fight↑, fire → panic↑).
- **N3 Hysteresis temporal filter + alarm-budget calibration**: dual-threshold EMA
  gate (raise 0.60 / clear 0.35, α=0.4, min 3 consecutive windows). Operating point
  calibrated to a target false-alarms/hour budget on UCF-Crime normal split.
- **N4 Scene-adaptive loitering**: cheap IoU tracker accumulates per-track dwell time;
  online per-scene EMA baseline flags dwell `> μ+kσ`. Overrides X3D loiter.

Thesis: *Cascaded, evidence-fused, alarm-budget-calibrated temporal threat detection
for low-compute edge.*

## 4. Architecture

```
threat-detector/
  core/    frames.py motion.py pipeline.py events.py
  detection/ yolo_branch.py
  action/    x3d_branch.py
  fusion/    bayes_fuser.py hysteresis.py loiter_baseline.py
  calib/     alarm_budget.py
  eval/      metrics.py ucf_crime.py
  bench/     modal_app.py
  serve/     dashboard.py
  cli.py
```

Data flow: `FrameSource` yields frames → sliding window buffer (16 frames) →
YOLO per-frame (fast) + gated X3D per-window (slow) → BayesFuser → Hysteresis gate →
`ThreatEvent` → sinks (JSON log, annotated video, dashboard). Branches run at
different cadences; pipeline reports separate + fused latency.

## 5. Models

- **Detection**: YOLOv8n/s (Ultralytics), COCO-pretrained, light fine-tune on merged
  weapon (pistol/knife) + fire/smoke (D-Fire/FireNet) sets. Classes:
  `person, weapon, fire, smoke`. Every frame, FP16.
- **Action**: X3D-M (PyTorchVideo, Kinetics-400 pretrained), head fine-tuned to
  `{fight, fall, loiter, normal}` on RWF-2000 + fall set (URFD/le2i) + normal clips.
  Window 16, adaptive stride via N1.

## 6. Evaluation

- Detection mAP@0.5; action top-1 accuracy.
- **False-alarms/hour** (headline production metric), target ≤ 1/hr.
- Per-stage latency (detect / classify / fuse) + end-to-end FPS; state real-time
  (>15 fps) verdict + hardware.
- `compute_saved%` from N1 cascade.
- UCF-Crime eval-only loader for anomaly/normal FA measurement.

## 7. Testing

pytest unit tests per module against synthetic tensors/frames (no GPU needed for
unit layer). Fusion math, hysteresis state machine, motion gate, loiter baseline,
metrics all deterministically testable. Modal funcs smoke-tested separately.

## 8. Delivery

FastAPI dashboard: upload video → run pipeline → alert timeline (JSON + minimal HTML).
CLI runs offline on a video file producing annotated output + event log.

## 9. Stack

Python 3.11, PyTorch, Ultralytics, PyTorchVideo, OpenCV, NumPy, FastAPI, pytest,
Modal. Config via YAML/dataclasses. Commits small, one-line messages, no co-author.
