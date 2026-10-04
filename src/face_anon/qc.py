from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from face_anon.detect_retinaface import RetinaFaceDetector
from face_anon.types import FaceBox, Leak, QCReport
from face_anon.video_io import iter_frames, probe_video


def residual_visibility(roi: np.ndarray) -> dict:
    """Is a RetinaFace crop still a *photographic* face?

    Detectors fire on coarse mosaics because the oval + hair silhouette
    remains. That is not an identity leak. We measure how well an 8×8
    mosaic reconstructs the crop (already-pixelated regions are cheap to
    reconstruct) and Laplacian energy (heavy blur is cheap to reconstruct
    poorly but has almost no texture).
    """
    if roi is None or roi.size == 0 or min(roi.shape[:2]) < 8:
        return {"visible": False, "mae": 0.0, "laplacian": 0.0}
    h, w = roi.shape[:2]
    # Ignore the feathered ellipse rim: that ring is unredacted background
    # and would otherwise look "photographic" after a correct redact.
    y0, y1 = int(h * 0.12), max(int(h * 0.12) + 8, int(h * 0.88))
    x0, x1 = int(w * 0.12), max(int(w * 0.12) + 8, int(w * 0.88))
    roi = roi[y0:y1, x0:x1]
    if min(roi.shape[:2]) < 8:
        return {"visible": False, "mae": 0.0, "laplacian": 0.0}
    h, w = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    tiny = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_AREA)
    recon = cv2.resize(tiny, (w, h), interpolation=cv2.INTER_NEAREST)
    mae = float(np.mean(np.abs(roi.astype(np.float32) - recon.astype(np.float32))))
    # Mosaic / heavy blur sit well below these; a live face does not.
    visible = mae >= 9.0 and lap >= 22.0
    return {"visible": visible, "mae": mae, "laplacian": lap}


def crop_face(frame: np.ndarray, face: FaceBox, pad: float = 0.05) -> np.ndarray:
    h, w = frame.shape[:2]
    box = face.expanded(pad, w, h)
    x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
    return frame[y1:y2, x1:x2]


def select_qc_indices(
    nframes: int,
    fps: float,
    interval_sec: float,
    max_frames: int,
    extra: Optional[Iterable[int]] = None,
) -> list[int]:
    """Uniform time samples + extra frames (track-loss, dark) under a hard cap.

    RetinaFace-R50 is ~50-150 ms/frame on CPU. A one-hour video at 30 FPS
    cannot be fully rescored. A budgeted, stratified sample is the only
    way to keep QC independent *and* viable at thousands of videos/day.
    """
    if nframes <= 0:
        return []
    step = max(1, int(round(interval_sec * fps)))
    chosen = set(range(0, nframes, step))
    if extra:
        for i in extra:
            if 0 <= i < nframes:
                chosen.add(int(i))
    ordered = sorted(chosen)
    if len(ordered) <= max_frames:
        return ordered
    # Keep temporal coverage when we have to thin the set.
    idxs = np.linspace(0, len(ordered) - 1, max_frames).round().astype(int)
    return [ordered[i] for i in sorted(set(idxs.tolist()))]


def run_qc(
    output_video: str,
    detector: RetinaFaceDetector,
    sample_interval_sec: float = 1.5,
    max_frames: int = 160,
    extra_indices: Optional[Iterable[int]] = None,
    fail_fast: bool = False,
    min_face_px: int = 12,
) -> QCReport:
    meta = probe_video(output_video)
    want = set(
        select_qc_indices(
            meta["nframes"],
            meta["fps"],
            sample_interval_sec,
            max_frames,
            extra_indices,
        )
    )
    leaks: list[Leak] = []
    checked = 0
    detector_hits = 0
    suppressed = 0
    for idx, frame in iter_frames(output_video):
        if idx not in want:
            continue
        checked += 1
        faces = detector.detect(frame)
        for face in faces:
            if face.width < min_face_px or face.height < min_face_px:
                continue
            detector_hits += 1
            vis = residual_visibility(crop_face(frame, face))
            if not vis["visible"]:
                suppressed += 1
                continue
            leaks.append(
                Leak(
                    t_sec=idx / meta["fps"],
                    frame_index=idx,
                    score=face.score,
                    box=face.as_xyxy(),
                )
            )
        if fail_fast and leaks:
            break

    leak_rate = (len(leaks) / checked) if checked else 0.0
    status = "pass" if not leaks else "needs_review"
    notes = []
    if meta["nframes"] and len(want) < meta["nframes"]:
        notes.append(
            f"sampled {len(want)}/{meta['nframes']} frames "
            f"(interval={sample_interval_sec}s, cap={max_frames})"
        )
    notes.append(
        f"retinaface_hits={detector_hits} residual_visible={len(leaks)} "
        f"suppressed_already_redacted={suppressed}"
    )
    return QCReport(
        status=status,
        frames_checked=checked,
        leak_count=len(leaks),
        leak_rate=leak_rate,
        leaks=leaks,
        model="retinaface_r50",
        notes=notes,
    )
