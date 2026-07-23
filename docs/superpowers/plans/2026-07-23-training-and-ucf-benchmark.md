# Training + UCF-Crime Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune the YOLO detection head and X3D action head on small curated sets, then produce real UCF-Crime false-alarms/hour numbers at a calibrated operating point.

**Architecture:** All heavy work runs as Modal functions against a Modal Volume (`threat-detector-data`) holding datasets and weights — never git. Local code stays CPU-testable via synthetic fixtures; Modal jobs are verified by their emitted artifacts (weights, metric JSON). Data prep builds manifests locally-testable, training/eval run remote.

**Tech Stack:** Python 3.11, PyTorch, Ultralytics YOLOv8, PyTorchVideo X3D-M, OpenCV, NumPy, Modal, pytest.

## Global Constraints

- Python >= 3.11.
- Engine core (`core`, `fusion`, `calib`, `eval`) stays NumPy/OpenCV-only; torch imported lazily inside `detection/`, `action/`, `bench/`, `data/` remote code paths.
- Datasets, weights, videos never committed — already covered by `.gitignore` (`data/`, `weights/`, `*.pt`, `*.pth`, `*.mp4`).
- Commits: one-line messages, plain style, never co-authored, `-c commit.gpgsign=false` if signing blocks.
- Comments: one-liners, only where 100% needed.
- ACTIONS order fixed: `("fight", "fall", "loiter", "normal")` (from `fusion/bayes_fuser.py`).
- Detection vocab fixed: `("person", "weapon", "fire", "smoke")` (from `detection/yolo_branch.py`).
- Modal Volume name: `threat-detector-data`. Modal App name: `threat-detector`.

---

### Task 1: Dataset manifest builder

**Files:**
- Create: `src/threat_detector/data/__init__.py`
- Create: `src/threat_detector/data/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Produces: `ClipEntry(path: str, label: str, split: str)` dataclass;
  `build_action_manifest(root: str) -> list[ClipEntry]` scans
  `root/<split>/<label>/*.mp4` where split ∈ {train,val,test},
  label ∈ ACTIONS; `save_manifest(entries, path)` / `load_manifest(path)` as JSONL.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest.py
import json
from threat_detector.data.manifest import (
    ClipEntry, build_action_manifest, save_manifest, load_manifest,
)

def test_build_action_manifest_scans_split_label_dirs(tmp_path):
    (tmp_path / "train" / "fight").mkdir(parents=True)
    (tmp_path / "val" / "normal").mkdir(parents=True)
    (tmp_path / "train" / "fight" / "a.mp4").write_bytes(b"x")
    (tmp_path / "val" / "normal" / "b.mp4").write_bytes(b"x")
    entries = build_action_manifest(str(tmp_path))
    labels = {(e.split, e.label) for e in entries}
    assert ("train", "fight") in labels
    assert ("val", "normal") in labels
    assert len(entries) == 2

def test_manifest_roundtrip(tmp_path):
    entries = [ClipEntry("x.mp4", "fight", "train")]
    p = tmp_path / "m.jsonl"
    save_manifest(entries, str(p))
    loaded = load_manifest(str(p))
    assert loaded == entries

def test_build_ignores_unknown_labels(tmp_path):
    (tmp_path / "train" / "banana").mkdir(parents=True)
    (tmp_path / "train" / "banana" / "c.mp4").write_bytes(b"x")
    assert build_action_manifest(str(tmp_path)) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: threat_detector.data.manifest`

- [ ] **Step 3: Write minimal implementation**

```python
# src/threat_detector/data/__init__.py  (empty)
```

```python
# src/threat_detector/data/manifest.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from ..fusion.bayes_fuser import ACTIONS

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ClipEntry:
    path: str
    label: str
    split: str


def build_action_manifest(root: str) -> list[ClipEntry]:
    base = Path(root)
    out: list[ClipEntry] = []
    for split in SPLITS:
        for label in ACTIONS:
            d = base / split / label
            if not d.is_dir():
                continue
            for clip in sorted(d.glob("*.mp4")):
                out.append(ClipEntry(str(clip), label, split))
    return out


def save_manifest(entries: list[ClipEntry], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(asdict(e)) + "\n")


def load_manifest(path: str) -> list[ClipEntry]:
    with open(path, encoding="utf-8") as f:
        return [ClipEntry(**json.loads(line)) for line in f if line.strip()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_manifest.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/threat_detector/data tests/test_manifest.py
git -c commit.gpgsign=false commit -m "Add dataset manifest builder for action clips"
```

---

### Task 2: Clip sampler (uniform 16-frame decode)

**Files:**
- Create: `src/threat_detector/data/clip_sampler.py`
- Test: `tests/test_clip_sampler.py`

**Interfaces:**
- Consumes: nothing from prior tasks (uses OpenCV).
- Produces: `sample_clip(path: str, num_frames: int = 16, size: int = 224) -> np.ndarray`
  returning `(num_frames, size, size, 3)` uint8 BGR via uniform temporal sampling;
  short clips loop-pad to `num_frames`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clip_sampler.py
import numpy as np
from threat_detector.data.clip_sampler import sample_clip
from threat_detector.core.frames import ArrayFrameSource  # noqa: F401 (import sanity)

def _write_video(path, n_frames, w=64, h=48):
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(n_frames):
        vw.write(np.full((h, w, 3), i % 256, np.uint8))
    vw.release()

def test_sample_clip_shape(tmp_path):
    p = tmp_path / "v.mp4"
    _write_video(p, 40)
    clip = sample_clip(str(p), num_frames=16, size=32)
    assert clip.shape == (16, 32, 32, 3)
    assert clip.dtype == np.uint8

def test_sample_clip_pads_short_video(tmp_path):
    p = tmp_path / "s.mp4"
    _write_video(p, 5)
    clip = sample_clip(str(p), num_frames=16, size=32)
    assert clip.shape == (16, 32, 32, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_clip_sampler.py -v`
Expected: FAIL with `ModuleNotFoundError: threat_detector.data.clip_sampler`

- [ ] **Step 3: Write minimal implementation**

```python
# src/threat_detector/data/clip_sampler.py
from __future__ import annotations

import numpy as np


def sample_clip(path: str, num_frames: int = 16, size: int = 224) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(img, (size, size)))
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded: {path}")
    idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    return np.stack([frames[i] for i in idx]).astype(np.uint8)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_clip_sampler.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/threat_detector/data/clip_sampler.py tests/test_clip_sampler.py
git -c commit.gpgsign=false commit -m "Add uniform 16-frame clip sampler with loop-pad"
```

---

### Task 3: X3D head fine-tune Modal function

**Files:**
- Modify: `src/threat_detector/bench/modal_app.py` (append `train_x3d` function)
- Test: `tests/test_train_x3d_config.py`

**Interfaces:**
- Consumes: `ClipEntry`/`load_manifest` (Task 1), `sample_clip` (Task 2),
  `X3DRecognizer` / `X3DConfig` (existing), `ACTIONS`.
- Produces: Modal function `train_x3d(manifest: str, epochs: int, out: str)` that
  fine-tunes only the X3D head, writes `out/x3d_head.pth` to the Volume, returns
  `{"val_acc": float, "ckpt": str}`. Local helper
  `build_head_dataset(entries, split) -> list[tuple[str, int]]` maps clip paths to
  ACTIONS indices — this helper is unit-tested; the training body is not.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_x3d_config.py
from threat_detector.data.manifest import ClipEntry
from threat_detector.bench.train_helpers import build_head_dataset
from threat_detector.fusion.bayes_fuser import ACTIONS

def test_build_head_dataset_maps_labels_to_indices():
    entries = [
        ClipEntry("a.mp4", "fight", "train"),
        ClipEntry("b.mp4", "normal", "train"),
        ClipEntry("c.mp4", "fall", "val"),
    ]
    ds = build_head_dataset(entries, split="train")
    assert ("a.mp4", ACTIONS.index("fight")) in ds
    assert ("b.mp4", ACTIONS.index("normal")) in ds
    assert all(split_path != "c.mp4" for split_path, _ in ds)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_train_x3d_config.py -v`
Expected: FAIL with `ModuleNotFoundError: threat_detector.bench.train_helpers`

- [ ] **Step 3: Write minimal implementation**

Create the pure helper (CPU-testable, no torch):

```python
# src/threat_detector/bench/train_helpers.py
from __future__ import annotations

from ..data.manifest import ClipEntry
from ..fusion.bayes_fuser import ACTIONS


def build_head_dataset(entries: list[ClipEntry], split: str) -> list[tuple[str, int]]:
    return [(e.path, ACTIONS.index(e.label)) for e in entries if e.split == split]
```

Append the Modal training function (runs remote; freezes backbone, trains head):

```python
# append to src/threat_detector/bench/modal_app.py
@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=7200)
def train_x3d(manifest: str = "/vol/manifests/action.jsonl", epochs: int = 8,
              out: str = "/vol/weights"):
    import numpy as np
    import torch
    from pytorchvideo.models.hub import x3d_m

    from threat_detector.data.manifest import load_manifest
    from threat_detector.data.clip_sampler import sample_clip
    from threat_detector.bench.train_helpers import build_head_dataset
    from threat_detector.fusion.bayes_fuser import ACTIONS

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = x3d_m(pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
    model.blocks[-1].proj = torch.nn.Linear(model.blocks[-1].proj.in_features, len(ACTIONS))
    model.to(dev).train()

    entries = load_manifest(manifest)
    train = build_head_dataset(entries, "train")
    val = build_head_dataset(entries, "val")
    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1).to(dev)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1).to(dev)

    def to_tensor(path):
        clip = sample_clip(path)[..., ::-1].copy()
        x = torch.from_numpy(clip).float().to(dev) / 255.0
        x = x.permute(3, 0, 1, 2)
        return ((x - mean) / std).unsqueeze(0)

    opt = torch.optim.Adam(model.blocks[-1].proj.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        np.random.shuffle(train)
        for path, label in train:
            opt.zero_grad()
            logits = model(to_tensor(path))
            loss = loss_fn(logits, torch.tensor([label], device=dev))
            loss.backward()
            opt.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for path, label in val:
            pred = int(model(to_tensor(path))[0].argmax())
            correct += int(pred == label)
    acc = correct / max(1, len(val))

    import os
    os.makedirs(out, exist_ok=True)
    ckpt = f"{out}/x3d_head.pth"
    torch.save(model.state_dict(), ckpt)
    vol.commit()
    result = {"val_acc": acc, "ckpt": ckpt}
    print(result)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_train_x3d_config.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Verify Modal function imports (no run)**

Run: `PYTHONUTF8=1 PYTHONPATH=src python -c "from threat_detector.bench import modal_app; print('train_x3d' in dir(modal_app))"`
Expected: prints `True`

- [ ] **Step 6: Commit**

```bash
git add src/threat_detector/bench/train_helpers.py src/threat_detector/bench/modal_app.py tests/test_train_x3d_config.py
git -c commit.gpgsign=false commit -m "Add X3D head fine-tune Modal function and dataset helper"
```

---

### Task 4: YOLO fine-tune data config + Modal function

**Files:**
- Create: `src/threat_detector/data/yolo_dataset.py`
- Modify: `src/threat_detector/bench/modal_app.py` (append `train_yolo`)
- Test: `tests/test_yolo_dataset.py`

**Interfaces:**
- Consumes: THREAT_LABELS from `detection/yolo_branch.py`.
- Produces: `write_data_yaml(root: str, out: str) -> str` writing an Ultralytics
  `data.yaml` with `names: [person, weapon, fire, smoke]` and train/val paths;
  Modal function `train_yolo(data_yaml, epochs, out)` that runs
  `YOLO(base).train(...)` and copies `best.pt` to the Volume.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_yolo_dataset.py
import yaml
from threat_detector.data.yolo_dataset import write_data_yaml
from threat_detector.detection.yolo_branch import THREAT_LABELS

def test_write_data_yaml_has_threat_labels(tmp_path):
    out = tmp_path / "data.yaml"
    write_data_yaml(str(tmp_path / "root"), str(out))
    cfg = yaml.safe_load(out.read_text())
    assert cfg["names"] == list(THREAT_LABELS)
    assert cfg["nc"] == 4
    assert "train" in cfg and "val" in cfg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_yolo_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: threat_detector.data.yolo_dataset`

- [ ] **Step 3: Write minimal implementation**

```python
# src/threat_detector/data/yolo_dataset.py
from __future__ import annotations

from pathlib import Path

import yaml

from ..detection.yolo_branch import THREAT_LABELS


def write_data_yaml(root: str, out: str) -> str:
    cfg = {
        "path": root,
        "train": "images/train",
        "val": "images/val",
        "nc": len(THREAT_LABELS),
        "names": list(THREAT_LABELS),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out
```

Append the Modal function:

```python
# append to src/threat_detector/bench/modal_app.py
@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=10800)
def train_yolo(data_yaml: str = "/vol/yolo/data.yaml", epochs: int = 50,
               out: str = "/vol/weights", base: str = "yolov8n.pt"):
    import shutil
    from ultralytics import YOLO

    model = YOLO(base)
    res = model.train(data=data_yaml, epochs=epochs, imgsz=640, device=0)
    best = f"{res.save_dir}/weights/best.pt"
    import os
    os.makedirs(out, exist_ok=True)
    dst = f"{out}/yolo_threat.pt"
    shutil.copy(best, dst)
    vol.commit()
    result = {"weights": dst, "save_dir": str(res.save_dir)}
    print(result)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_yolo_dataset.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/threat_detector/data/yolo_dataset.py src/threat_detector/bench/modal_app.py tests/test_yolo_dataset.py
git -c commit.gpgsign=false commit -m "Add YOLO data.yaml writer and fine-tune Modal function"
```

---

### Task 5: UCF-Crime window scorer

**Files:**
- Create: `src/threat_detector/eval/scorer.py`
- Test: `tests/test_scorer.py`

**Interfaces:**
- Consumes: `Pipeline` (existing), `DetectorProtocol`/`RecognizerProtocol`.
- Produces: `score_video(pipeline_factory, path) -> np.ndarray` returning one
  per-window max-threat score in [0,1] for a video; `score_split(factory, paths) ->
  np.ndarray` concatenating scores across videos. Score = max non-normal posterior
  per processed window, read from a lightweight `Pipeline` scoring hook.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scorer.py
import numpy as np
from threat_detector.eval.scorer import scores_from_events
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: threat_detector.eval.scorer`

- [ ] **Step 3: Write minimal implementation**

```python
# src/threat_detector/eval/scorer.py
from __future__ import annotations

import numpy as np

from ..core.events import ThreatEvent


def scores_from_events(events: list[ThreatEvent]) -> np.ndarray:
    return np.array([e.confidence for e in events], dtype=np.float64)


def score_split(per_video_events: list[list[ThreatEvent]]) -> np.ndarray:
    if not per_video_events:
        return np.array([], dtype=np.float64)
    return np.concatenate([scores_from_events(ev) for ev in per_video_events])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_scorer.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/threat_detector/eval/scorer.py tests/test_scorer.py
git -c commit.gpgsign=false commit -m "Add UCF-Crime window scorer from pipeline events"
```

---

### Task 6: Wire real eval_ucf + FA/hour report

**Files:**
- Modify: `src/threat_detector/bench/modal_app.py` (replace placeholder `eval_ucf` body)
- Create: `src/threat_detector/eval/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `calibrate`/`OperatingPoint` (existing), `score_split` (Task 5).
- Produces: `build_report(normal_scores, anomaly_scores, normal_hours, budget) -> dict`
  bundling threshold, FA/hour, recall, counts; `eval_ucf` on Modal loads the UCF-Crime
  normal/anomaly splits, scores each video with the fine-tuned pipeline, and writes
  `/vol/reports/ucf_crime.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
import numpy as np
from threat_detector.eval.report import build_report

def test_build_report_fields():
    normal = np.full(100, 0.1)
    anomaly = np.full(20, 0.9)
    r = build_report(normal, anomaly, normal_hours=10.0, budget_per_hour=1.0)
    assert set(r) >= {"threshold", "false_alarms_per_hour", "recall", "n_normal", "n_anomaly"}
    assert r["n_normal"] == 100 and r["n_anomaly"] == 20
    assert r["recall"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: threat_detector.eval.report`

- [ ] **Step 3: Write minimal implementation**

```python
# src/threat_detector/eval/report.py
from __future__ import annotations

import numpy as np

from ..calib.alarm_budget import calibrate


def build_report(normal_scores, anomaly_scores, normal_hours: float,
                 budget_per_hour: float = 1.0) -> dict:
    normal_scores = np.asarray(normal_scores, dtype=np.float64)
    anomaly_scores = np.asarray(anomaly_scores, dtype=np.float64)
    op = calibrate(normal_scores, normal_hours, anomaly_scores, budget_per_hour)
    return {
        "threshold": op.threshold,
        "false_alarms_per_hour": op.false_alarms_per_hour,
        "recall": op.recall,
        "budget_per_hour": budget_per_hour,
        "n_normal": int(normal_scores.size),
        "n_anomaly": int(anomaly_scores.size),
    }
```

Replace the `eval_ucf` body in `modal_app.py`:

```python
@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=7200)
def eval_ucf(data: str = "/vol/ucf_crime", budget_per_hour: float = 1.0,
             fps: float = 25.0):
    import glob
    import json
    import os

    import torch

    from threat_detector.action.x3d_branch import X3DConfig, X3DRecognizer
    from threat_detector.detection.yolo_branch import YoloConfig, YoloDetector
    from threat_detector.core.frames import VideoFileSource
    from threat_detector.core.pipeline import Pipeline
    from threat_detector.eval.scorer import score_split
    from threat_detector.eval.report import build_report

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def run_dir(subdir):
        events_all = []
        for path in sorted(glob.glob(f"{data}/{subdir}/*.mp4")):
            src = VideoFileSource(path)
            det = YoloDetector(YoloConfig(weights="/vol/weights/yolo_threat.pt", device=dev))
            rec = X3DRecognizer(X3DConfig(device=dev, checkpoint="/vol/weights/x3d_head.pth"))
            events_all.append(Pipeline(det, rec, fps=src.fps).run(src))
        return events_all

    normal = score_split(run_dir("normal"))
    anomaly = score_split(run_dir("anomaly"))
    normal_hours = float(normal.size) / (fps * 3600)
    report = build_report(normal, anomaly, normal_hours, budget_per_hour)

    os.makedirs(f"{data}/../reports", exist_ok=True)
    with open("/vol/reports/ucf_crime.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    vol.commit()
    print(report)
    return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_report.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Full suite green**

Run: `PYTHONPATH=src pytest -q`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add src/threat_detector/eval/report.py src/threat_detector/bench/modal_app.py tests/test_report.py
git -c commit.gpgsign=false commit -m "Wire real UCF-Crime eval with FA/hour report"
```

---

### Task 7: Data ingestion helper + docs

**Files:**
- Create: `src/threat_detector/data/ingest.py`
- Create: `docs/data.md`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `build_action_manifest`/`save_manifest` (Task 1).
- Produces: `prepare_action_manifest(root, out) -> int` building + saving the action
  manifest and returning entry count; `docs/data.md` documenting dataset layout,
  Modal Volume upload commands, and the run order.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest.py
from threat_detector.data.ingest import prepare_action_manifest

def test_prepare_action_manifest_counts(tmp_path):
    (tmp_path / "train" / "fight").mkdir(parents=True)
    (tmp_path / "train" / "fight" / "a.mp4").write_bytes(b"x")
    out = tmp_path / "m.jsonl"
    n = prepare_action_manifest(str(tmp_path), str(out))
    assert n == 1 and out.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: threat_detector.data.ingest`

- [ ] **Step 3: Write minimal implementation**

```python
# src/threat_detector/data/ingest.py
from __future__ import annotations

from .manifest import build_action_manifest, save_manifest


def prepare_action_manifest(root: str, out: str) -> int:
    entries = build_action_manifest(root)
    save_manifest(entries, out)
    return len(entries)
```

```markdown
<!-- docs/data.md -->
# Datasets

## Action clips (X3D head)
Layout: `data/action/<split>/<label>/*.mp4`, split ∈ {train,val,test},
label ∈ {fight,fall,loiter,normal}. Sources: RWF-2000 (fight), URFD/le2i (fall),
normal CCTV clips, loiter clips.

Build manifest:
`PYTHONPATH=src python -c "from threat_detector.data.ingest import prepare_action_manifest as p; print(p('data/action','data/manifests/action.jsonl'))"`

## Detection images (YOLO head)
Ultralytics layout under `data/yolo/images/{train,val}` + `labels/...`, classes
`person,weapon,fire,smoke`. Sources: weapon (pistol/knife sets), D-Fire/FireNet.
Write config: `write_data_yaml('data/yolo','data/yolo/data.yaml')`.

## UCF-Crime (eval only)
Layout: `ucf_crime/{normal,anomaly}/*.mp4`. Used solely for FA/hour calibration.

## Modal Volume upload
```bash
modal volume create threat-detector-data
modal volume put threat-detector-data data/manifests /manifests
modal volume put threat-detector-data data/action /action
modal volume put threat-detector-data data/yolo /yolo
modal volume put threat-detector-data ucf_crime /ucf_crime
```

## Run order
1. `modal run -m threat_detector.bench.modal_app::train_yolo`
2. `modal run -m threat_detector.bench.modal_app::train_x3d`
3. `modal run -m threat_detector.bench.modal_app::eval_ucf`
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_ingest.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/threat_detector/data/ingest.py docs/data.md tests/test_ingest.py
git -c commit.gpgsign=false commit -m "Add action-manifest ingestion helper and data docs"
```

---

### Task 8: Execute remote jobs + publish numbers

**Files:**
- Modify: `docs/benchmarks.md` (add training + FA/hour results)
- Modify: `README.md:Status` (replace with published numbers)

**Interfaces:**
- Consumes: all Modal functions above. This task is operational: upload data, run
  the three Modal jobs, paste real numbers.

- [ ] **Step 1: Upload datasets to the Volume**

Run the `modal volume put` commands from `docs/data.md` after datasets are staged
locally under `data/`. Expected: `modal volume ls threat-detector-data` shows
`manifests/ action/ yolo/ ucf_crime/`.

- [ ] **Step 2: Fine-tune YOLO head**

Run: `PYTHONUTF8=1 PYTHONPATH=src modal run -m threat_detector.bench.modal_app::train_yolo`
Expected: prints `{"weights": "/vol/weights/yolo_threat.pt", ...}`.

- [ ] **Step 3: Fine-tune X3D head**

Run: `PYTHONUTF8=1 PYTHONPATH=src modal run -m threat_detector.bench.modal_app::train_x3d`
Expected: prints `{"val_acc": <float>, "ckpt": "/vol/weights/x3d_head.pth"}`.

- [ ] **Step 4: Run UCF-Crime eval**

Run: `PYTHONUTF8=1 PYTHONPATH=src modal run -m threat_detector.bench.modal_app::eval_ucf`
Expected: prints report with `false_alarms_per_hour` and `recall`.

- [ ] **Step 5: Publish numbers**

Add a training row (X3D val_acc, YOLO mAP@0.5 from Ultralytics output) and an
FA/hour row (threshold, FA/hour, recall) to `docs/benchmarks.md`. Replace the
README `## Status` section with the measured figures.

- [ ] **Step 6: Commit**

```bash
git add docs/benchmarks.md README.md
git -c commit.gpgsign=false commit -m "Publish training accuracy and UCF-Crime FA/hour results"
```

---

## Notes

- Tasks 1–7 are CPU-testable and land green locally; Task 8 is the remote execution
  gate that consumes real datasets and GPU time.
- Datasets must be staged before Task 8; acquisition (download links, licensing) is
  a manual prerequisite, not automated here.
- Each Modal function commits the Volume (`vol.commit()`) so downstream jobs see new
  weights.
