from threat_detector.data.ingest import prepare_action_manifest


def test_prepare_action_manifest_counts(tmp_path):
    (tmp_path / "train" / "fight").mkdir(parents=True)
    (tmp_path / "train" / "fight" / "a.mp4").write_bytes(b"x")
    out = tmp_path / "m.jsonl"
    n = prepare_action_manifest(str(tmp_path), str(out))
    assert n == 1 and out.exists()
