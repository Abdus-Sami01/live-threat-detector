import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from download_data import _remap_label_file, DFIRE_CLASS_REMAP, RWF_LABEL_MAP


def test_dfire_remap_matches_vocab():
    # smoke(0)->3, fire(1)->2 in person,weapon,fire,smoke vocab
    assert DFIRE_CLASS_REMAP == {0: 3, 1: 2}


def test_rwf_label_map():
    assert RWF_LABEL_MAP["fight"] == "fight"
    assert RWF_LABEL_MAP["nonfight"] == "normal"


def test_remap_label_file_rewrites_class_ids(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.3 0.3\n")
    dst = tmp_path / "out" / "a.txt"
    _remap_label_file(src, dst)
    lines = dst.read_text().splitlines()
    assert lines[0].startswith("3 ")  # smoke -> 3
    assert lines[1].startswith("2 ")  # fire -> 2
