from pathlib import Path

import numpy as np
import pytest

from face_anon.config import MODELS_DIR
from face_anon.video_io import FFmpegWriter


def _models_present() -> bool:
    return (MODELS_DIR / "face_detection_yunet_2023mar.onnx").is_file()


@pytest.mark.skipif(not _models_present(), reason="YuNet weights not downloaded")
def test_empty_scene_writes_same_length(tmp_path: Path):
    from face_anon.config import load_config
    from face_anon.pipeline import FaceAnonymizer
    from face_anon.video_io import probe_video

    src = tmp_path / "blank.mp4"
    writer = FFmpegWriter(src, 320, 240, 10.0, encoder="x264")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:] = (30, 40, 20)
    for _ in range(25):
        writer.write(frame)
    writer.close()

    cfg = load_config()
    cfg["qc"]["enabled"] = False
    cfg["video"]["keep_audio"] = False
    cfg["video"]["encoder"] = "x264"
    out = tmp_path / "out.mp4"
    metrics = FaceAnonymizer(cfg).process_video(src, out, run_quality_check=False)
    assert Path(out).is_file()
    assert metrics["faces_painted"] == 0
    assert abs(probe_video(out)["nframes"] - 25) <= 2
