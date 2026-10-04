import numpy as np

from face_anon.anonymize import anonymize_faces
from face_anon.qc import residual_visibility, select_qc_indices
from face_anon.types import FaceBox


def test_respects_budget_and_includes_extras():
    idxs = select_qc_indices(
        nframes=10_000,
        fps=30.0,
        interval_sec=1.5,
        max_frames=80,
        extra=[12, 999999, 500],
    )
    assert len(idxs) <= 80
    assert idxs == sorted(set(idxs))
    assert idxs[0] >= 0
    assert idxs[-1] < 10_000


def test_short_video_keeps_all_steps():
    idxs = select_qc_indices(nframes=90, fps=30, interval_sec=1.0, max_frames=200)
    assert idxs == list(range(0, 90, 30))


def test_residual_visibility_rejects_mosaic_keeps_photo():
    photo = np.zeros((120, 100, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:120, 0:100]
    photo[..., 0] = (40 + xx).clip(0, 255)
    photo[..., 1] = (80 + yy // 2).clip(0, 255)
    photo[..., 2] = ((xx * yy) % 180).astype(np.uint8)
    # Fine photographic noise so Laplacian stays high.
    rng = np.random.default_rng(0)
    photo = np.clip(photo.astype(np.int16) + rng.integers(-25, 26, photo.shape), 0, 255).astype(np.uint8)
    assert residual_visibility(photo)["visible"] is True

    redacted = photo.copy()
    anonymize_faces(redacted, [FaceBox(5, 5, 95, 115, 1.0)], method="hybrid", pad=0.0, pixel_blocks=8)
    assert residual_visibility(redacted)["visible"] is False
