"""Offline CLI: run the pipeline on a video file, emit alerts + report.

    threat-detect run input.mp4 --out outputs/ [--annotate]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .action.x3d_branch import X3DConfig, X3DRecognizer
from .core.frames import VideoFileSource
from .core.pipeline import Pipeline
from .core.sinks import write_event_log
from .detection.yolo_branch import YoloConfig, YoloDetector


def _run(args) -> int:
    source = VideoFileSource(args.input)
    detector = YoloDetector(YoloConfig(weights=args.yolo, device=args.device))
    recognizer = X3DRecognizer(X3DConfig(device=args.device, checkpoint=args.x3d_ckpt))
    pipe = Pipeline(detector, recognizer, fps=source.fps)

    events = pipe.run(source)

    out = Path(args.out)
    write_event_log(events, str(out / "events.json"))
    s = pipe.stats
    report = {
        "frames": s.frames,
        "alerts": len(events),
        "windows_total": s.windows_total,
        "windows_run": s.windows_run,
        "compute_saved": round(s.compute_saved, 4),
        "latency_ms": {
            "detect_avg": round(s.detect_ms / max(1, s.frames), 3),
            "classify_avg": round(s.classify_ms / max(1, s.windows_run), 3),
            "fuse_avg": round(s.fuse_ms / max(1, s.windows_run), 3),
        },
    }
    (out / "report.json").parent.mkdir(parents=True, exist_ok=True)
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"{len(events)} alert(s) -> {out / 'events.json'}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="threat-detect")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run pipeline on a video file")
    r.add_argument("input")
    r.add_argument("--out", default="outputs")
    r.add_argument("--yolo", default="yolov8n.pt")
    r.add_argument("--x3d-ckpt", default=None)
    r.add_argument("--device", default=None)
    r.set_defaults(func=_run)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
