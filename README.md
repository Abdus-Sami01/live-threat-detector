# Live Threat Detector

Real-time detection of **fights, weapons, fire/smoke, falls, and loitering** in
CCTV-style video. Temporal action recognition across clips fused with per-frame
object detection — not per-frame classification — tuned for the metric that
actually keeps a security system plugged in: **false alarms per hour**.

## Why it's different

Four low-compute contributions compose into one thesis —
*cascaded, evidence-fused, alarm-budget-calibrated temporal threat detection*:

| | Idea | Payoff |
|---|------|--------|
| **N1** | Compute-adaptive cascade — cheap frame-diff motion + detection gate the 3D branch | 3D FLOPs run only when the scene is interesting; `compute_saved%` reported |
| **N2** | Bayesian cross-branch fusion — detections prior-shape the action posterior | weapon seen → fight prior ↑; explainable, a few multiplies |
| **N3** | Hysteresis filter + alarm-budget calibration | dual-threshold EMA gate; operating point picked to a target FA/hour |
| **N4** | Scene-adaptive loitering baseline | per-scene dwell norm, not a fixed timer |

## Architecture

```
FrameSource ─► [YOLO detect / frame]──┐
             ─► motion energy ─► N1 gate ─► [X3D classify / window]
                                              │
                    detections ──► N2 Bayes fusion ──► N3 hysteresis ──► ThreatEvent
                    person boxes ─► N4 loiter baseline ┘
```

`core/` engine • `detection/` YOLO branch • `action/` X3D branch •
`fusion/` N2/N3/N4 • `calib/` alarm budget • `eval/` metrics • `bench/` Modal •
`serve/` dashboard.

## Install

```bash
pip install -e ".[models,serve,gpu,dev]"
```

The engine core (`core`, `fusion`, `calib`, `eval`) depends only on NumPy/OpenCV and
is unit-tested without a GPU. Torch/Ultralytics/PyTorchVideo load lazily inside the
model branches.

## Run

```bash
# offline on a video file -> outputs/events.json + report.json
threat-detect run input.mp4 --out outputs --device cuda

# dashboard: upload a video, get an alert timeline
uvicorn threat_detector.serve.dashboard:app

# GPU latency benchmark / UCF-Crime FA-rate eval on Modal
modal run -m threat_detector.bench.modal_app::bench
modal run -m threat_detector.bench.modal_app::eval_ucf
```

## Data

Hybrid strategy: COCO/Kinetics-pretrained backbones, light head fine-tune on small
sets (weapons, D-Fire/FireNet, RWF-2000 fights, fall sets). **UCF-Crime is used for
evaluation only** — the false-alarm-rate benchmark. No large training bill.

## Test

```bash
PYTHONPATH=src pytest -q
```

## Status

Engine, novelty modules, fusion, metrics, CLI, dashboard, and Modal harness in place
with a green unit suite. Next: wire fine-tuned checkpoints and publish benchmark
numbers (mAP, action acc, FA/hour, latency, compute-saved).
