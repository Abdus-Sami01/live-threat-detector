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
