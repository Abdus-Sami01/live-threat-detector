"""Write an Ultralytics data.yaml pinned to the threat detection vocabulary."""
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
