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
