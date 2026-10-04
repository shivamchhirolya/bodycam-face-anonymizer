import numpy as np

from face_anon.anonymize import anonymize_faces
from face_anon.boxes import pad_for_face
from face_anon.types import FaceBox


def test_hybrid_destroys_structure():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # High-frequency identity-like pattern.
    img[60:140, 60:140] = 0
    img[60:140:4, 60:140] = 255
    img[60:140, 60:140:4] = 180
    before = img[60:140, 60:140].astype(np.float32).var()
    anonymize_faces(
        img,
        [FaceBox(60, 60, 140, 140, 0.99)],
        method="hybrid",
        pad=0.05,
        pixel_blocks=8,
        shape="ellipse",
    )
    after = img[70:130, 70:130].astype(np.float32).var()
    assert after < before * 0.35


def test_blackout_zeros_roi():
    img = np.full((80, 80, 3), 200, dtype=np.uint8)
    anonymize_faces(img, [FaceBox(10, 10, 70, 70, 1.0)], method="blackout", pad=0.0, shape="rect")
    assert img[20:60, 20:60].max() == 0


def test_closeup_uses_tight_pad():
    close = FaceBox(8, 4, 472, 356, 0.95)
    pad = pad_for_face(close, 480, 360, 0.42)
    assert pad <= 0.08
    far = FaceBox(220, 140, 250, 178, 0.9)
    assert pad_for_face(far, 480, 360, 0.42) == 0.42
