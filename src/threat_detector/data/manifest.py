"""Action-clip dataset manifest: scan a split/label tree into a flat JSONL list."""
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
