"""Fetch and reorganize the freely-licensed training sets into project layout.

    python scripts/download_data.py rwf     # RWF-2000 fights -> data/action
    python scripts/download_data.py dfire   # D-Fire fire/smoke -> data/yolo

Downloads use kagglehub, which reads your existing Kaggle credentials
(~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY). You must accept each
dataset's terms on Kaggle first.

Licensing: RWF-2000 is research-only — no redistribution or commercial use without
SMIIP Lab approval. D-Fire follows its repository license. This script only pulls to
your machine and rearranges files; it redistributes nothing.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

RWF_SLUG = "vulamnguyen/rwf2000"
DFIRE_SLUG = "sayedgamal99/smoke-fire-detection-yolo"

# RWF class folder -> our action label
RWF_LABEL_MAP = {"fight": "fight", "nonfight": "normal"}
# D-Fire uses class 0=smoke, 1=fire; remap to our detection vocab indices.
DFIRE_CLASS_REMAP = {0: 3, 1: 2}  # smoke->3, fire->2  (person=0, weapon=1)


def _download(slug: str) -> Path:
    import kagglehub

    path = kagglehub.dataset_download(slug)
    print(f"downloaded {slug} -> {path}")
    return Path(path)


def _find_dirs(root: Path, names: set[str]) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_dir() and p.name.lower() in names]


def _copy_videos(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for vid in src.glob("*"):
        if vid.suffix.lower() in (".mp4", ".avi", ".mov"):
            shutil.copy2(vid, dst / f"{src.parent.name}_{vid.name}")
            n += 1
    return n


def prepare_rwf(out: str = "data/action") -> dict:
    """RWF-2000: <root>/{train,val}/{Fight,NonFight} -> data/action/<split>/<label>."""
    root = _download(RWF_SLUG)
    out_base = Path(out)
    counts: dict[str, int] = {}
    for split_dir in _find_dirs(root, {"train", "val"}):
        split = "val" if split_dir.name.lower() == "val" else "train"
        for cls_dir in split_dir.iterdir():
            label = RWF_LABEL_MAP.get(cls_dir.name.lower())
            if not cls_dir.is_dir() or label is None:
                continue
            n = _copy_videos(cls_dir, out_base / split / label)
            counts[f"{split}/{label}"] = counts.get(f"{split}/{label}", 0) + n
    print("rwf counts:", counts)
    return counts


def _remap_label_file(src: Path, dst: Path) -> None:
    lines = []
    for line in src.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = DFIRE_CLASS_REMAP.get(int(parts[0]), int(parts[0]))
        lines.append(" ".join([str(cls), *parts[1:]]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))


def prepare_dfire(out: str = "data/yolo") -> dict:
    """D-Fire YOLO layout -> data/yolo/{images,labels}/{train,val} with remapped ids."""
    root = _download(DFIRE_SLUG)
    out_base = Path(out)
    counts: dict[str, int] = {}
    for img_dir in _find_dirs(root, {"images"}):
        split_name = img_dir.parent.name.lower()
        split = "val" if split_name in ("val", "valid", "validation", "test") else "train"
        lbl_dir = img_dir.parent / "labels"
        for img in img_dir.glob("*"):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            shutil.copy2(img, _ensure(out_base / "images" / split) / img.name)
            src_lbl = lbl_dir / f"{img.stem}.txt"
            dst_lbl = out_base / "labels" / split / f"{img.stem}.txt"
            if src_lbl.exists():
                _remap_label_file(src_lbl, dst_lbl)
            counts[split] = counts.get(split, 0) + 1
    from threat_detector.data.yolo_dataset import write_data_yaml  # local import

    write_data_yaml(str(out_base), str(out_base / "data.yaml"))
    print("dfire counts:", counts)
    return counts


def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="download_data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rwf", help="RWF-2000 fights -> data/action")
    r.add_argument("--out", default="data/action")
    r.set_defaults(func=lambda a: prepare_rwf(a.out))
    d = sub.add_parser("dfire", help="D-Fire fire/smoke -> data/yolo")
    d.add_argument("--out", default="data/yolo")
    d.set_defaults(func=lambda a: prepare_dfire(a.out))
    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
