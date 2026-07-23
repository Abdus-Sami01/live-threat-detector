from threat_detector.data.manifest import ClipEntry
from threat_detector.bench.train_helpers import build_head_dataset
from threat_detector.fusion.bayes_fuser import ACTIONS


def test_build_head_dataset_maps_labels_to_indices():
    entries = [
        ClipEntry("a.mp4", "fight", "train"),
        ClipEntry("b.mp4", "normal", "train"),
        ClipEntry("c.mp4", "fall", "val"),
    ]
    ds = build_head_dataset(entries, split="train")
    assert ("a.mp4", ACTIONS.index("fight")) in ds
    assert ("b.mp4", ACTIONS.index("normal")) in ds
    assert all(p != "c.mp4" for p, _ in ds)
