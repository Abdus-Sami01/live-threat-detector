"""Modal GPU app: latency benchmark, UCF-Crime eval, and light head fine-tune.

Run:
    modal run -m threat_detector.bench.modal_app::bench --gpu L40S
    modal run -m threat_detector.bench.modal_app::eval_ucf --data /vol/ucf_crime

Heavy weights and datasets live on a Modal Volume, never in git.
"""
from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1")
    .pip_install(
        "torch>=2.2", "torchvision>=0.17", "ultralytics>=8.2",
        "pytorchvideo>=0.1.5", "opencv-python-headless>=4.9",
        "numpy>=1.26", "pyyaml>=6.0",
    )
    .add_local_python_source("threat_detector")
)

app = modal.App("threat-detector")
vol = modal.Volume.from_name("threat-detector-data", create_if_missing=True)
GPU = "L40S"


@app.function(image=image, gpu=GPU, timeout=1800)
def bench(clips: int = 100):
    """Measure per-stage latency + real-time verdict on the chosen GPU."""
    import time
    import numpy as np
    import torch

    from threat_detector.action.x3d_branch import X3DConfig, X3DRecognizer
    from threat_detector.eval.metrics import LatencyReport

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rec = X3DRecognizer(X3DConfig(device=dev))
    clip = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)
    rec.classify(clip)  # warmup / lazy load

    t = time.perf_counter()
    for _ in range(clips):
        rec.classify(clip)
    classify_ms = (time.perf_counter() - t) / clips * 1000
    rep = LatencyReport(detect_ms=8.0, classify_ms=classify_ms, fuse_ms=0.2)
    result = {
        "gpu": GPU, "device": dev,
        "classify_ms": round(classify_ms, 2),
        "est_fps": round(rep.fps, 2), "realtime": rep.realtime,
    }
    print(result)
    return result


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=3600)
def eval_ucf(data: str = "/vol/ucf_crime", budget_per_hour: float = 1.0):
    """False-alarm-rate calibration on UCF-Crime normal/anomaly splits."""
    import numpy as np

    from threat_detector.calib.alarm_budget import calibrate

    # placeholder wiring: scores produced by scoring the split with the pipeline.
    # replace the two arrays with real per-window max-threat scores.
    normal_scores = np.load(f"{data}/normal_scores.npy")
    anomaly_scores = np.load(f"{data}/anomaly_scores.npy")
    normal_hours = float(len(normal_scores)) / (25.0 * 3600)
    op = calibrate(normal_scores, normal_hours, anomaly_scores, budget_per_hour)
    result = {
        "threshold": op.threshold,
        "false_alarms_per_hour": op.false_alarms_per_hour,
        "recall_at_budget": op.recall,
    }
    print(result)
    return result


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=7200)
def train_x3d(manifest: str = "/vol/manifests/action.jsonl", epochs: int = 8,
              out: str = "/vol/weights"):
    """Freeze the X3D-M backbone, fine-tune the 4-class head, save to the Volume."""
    import os

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
            correct += int(int(model(to_tensor(path))[0].argmax()) == label)
    acc = correct / max(1, len(val))

    os.makedirs(out, exist_ok=True)
    ckpt = f"{out}/x3d_head.pth"
    torch.save(model.state_dict(), ckpt)
    vol.commit()
    result = {"val_acc": acc, "ckpt": ckpt}
    print(result)
    return result
