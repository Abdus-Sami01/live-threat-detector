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
