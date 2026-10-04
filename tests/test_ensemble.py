import numpy as np

from face_anon.detect_ensemble import nms_merge
from face_anon.preprocess import any_tile_dark, resize_for_detect
from face_anon.types import FaceBox


def test_resize_upsamples_small_frames():
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    out, scale = resize_for_detect(img, 960)
    assert scale == 2.0
    assert out.shape[1] == 960
    assert out.shape[0] == 720


def test_resize_downscales_large_frames():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out, scale = resize_for_detect(img, 640)
    assert abs(scale - 640 / 1920) < 1e-6
    assert out.shape[1] == 640


def _five_point_face(x1, y1, x2, y2) -> np.ndarray:
    w, h = x2 - x1, y2 - y1
    return np.array(
        [
            [x1 + 0.30 * w, y1 + 0.35 * h],
            [x1 + 0.70 * w, y1 + 0.35 * h],
            [x1 + 0.50 * w, y1 + 0.55 * h],
            [x1 + 0.35 * w, y1 + 0.78 * h],
            [x1 + 0.65 * w, y1 + 0.78 * h],
        ],
        dtype=np.float32,
    )


def test_rejects_giant_false_face():
    from face_anon.detect_ensemble import is_sane_box

    giant = FaceBox(10, 10, 470, 340, 0.3)
    assert is_sane_box(giant, 480, 360, 6) is False
    torso = FaceBox(160, 145, 327, 364, 0.21)
    assert is_sane_box(torso, 480, 360, 6) is False
    normal = FaceBox(200, 80, 280, 180, 0.9)
    assert is_sane_box(normal, 480, 360, 6) is True


def test_keeps_closeup_that_fills_the_frame():
    from face_anon.detect_ensemble import is_sane_box

    # Officer standing on the lens: the face *is* the frame.
    close = FaceBox(8, 4, 472, 356, 0.91, landmarks=_five_point_face(8, 4, 472, 356))
    assert is_sane_box(close, 480, 360, 6) is True
    # Motion-blur close-up can lose landmarks; a strong score still counts.
    blurry = FaceBox(12, 8, 468, 350, 0.88)
    assert is_sane_box(blurry, 480, 360, 6) is True


def test_nms_keeps_higher_score():
    a = FaceBox(10, 10, 40, 40, 0.9)
    b = FaceBox(12, 12, 42, 42, 0.4)
    c = FaceBox(80, 80, 110, 110, 0.8)
    kept = nms_merge([a, b, c], iou_thresh=0.4)
    assert len(kept) == 2
    assert {round(k.score, 1) for k in kept} == {0.9, 0.8}


def test_dark_doorway_in_bright_frame():
    img = np.full((80, 80, 3), 180, dtype=np.uint8)
    img[60:80, 60:80] = 20
    assert any_tile_dark(img, threshold=72, tiles=4) is True
    bright = np.full((80, 80, 3), 180, dtype=np.uint8)
    assert any_tile_dark(bright, threshold=72, tiles=4) is False
