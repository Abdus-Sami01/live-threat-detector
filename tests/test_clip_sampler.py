import numpy as np
from threat_detector.data.clip_sampler import sample_clip


def _write_video(path, n_frames, w=64, h=48):
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(n_frames):
        vw.write(np.full((h, w, 3), i % 256, np.uint8))
    vw.release()


def test_sample_clip_shape(tmp_path):
    p = tmp_path / "v.mp4"
    _write_video(p, 40)
    clip = sample_clip(str(p), num_frames=16, size=32)
    assert clip.shape == (16, 32, 32, 3)
    assert clip.dtype == np.uint8


def test_sample_clip_pads_short_video(tmp_path):
    p = tmp_path / "s.mp4"
    _write_video(p, 5)
    clip = sample_clip(str(p), num_frames=16, size=32)
    assert clip.shape == (16, 32, 32, 3)
