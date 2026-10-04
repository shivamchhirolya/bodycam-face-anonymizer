from face_anon.boxes import pad_for_face
from face_anon.detect_ensemble import is_sane_box
from face_anon.types import FaceBox


def test_low_score_full_frame_is_rejected():
    assert is_sane_box(FaceBox(10, 10, 470, 350, 0.40), 480, 360, 6) is False


def test_high_score_closeup_is_kept():
    assert is_sane_box(FaceBox(8, 4, 472, 356, 0.91), 480, 360, 6) is True


def test_normal_midground_is_kept():
    assert is_sane_box(FaceBox(200, 80, 300, 220, 0.85), 480, 360, 6) is True


def test_tiny_box_below_min_face_is_rejected():
    assert is_sane_box(FaceBox(100, 140, 104, 144, 0.90), 480, 360, 6) is False


def test_tiny_head_gets_extra_pad():
    tiny = FaceBox(220, 140, 236, 158, 0.80)
    assert pad_for_face(tiny, 480, 360, 0.42) > 0.42
