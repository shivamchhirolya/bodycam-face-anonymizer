from __future__ import annotations

from typing import Any

from face_anon.detect_retinaface import RetinaFaceDetector
from face_anon.qc import crop_face, residual_visibility
from face_anon.video_io import iter_frames, probe_video


def privacy_eval(
    input_video: str,
    output_video: str,
    retina: RetinaFaceDetector,
    max_frames: int = 80,
    iou_match: float = 0.2,
) -> dict[str, Any]:
    """Proxy metrics using RetinaFace on the input as stand-in ground truth."""
    src_meta = probe_video(input_video)
    step = max(1, src_meta["nframes"] // max(1, max_frames))
    gt_faces = 0
    leftover = 0
    covered = 0
    frames = 0
    for (si, src), (oi, out) in zip(iter_frames(input_video), iter_frames(output_video)):
        if si != oi:
            break
        if si % step:
            continue
        frames += 1
        gt = retina.detect(src)
        vis = [f for f in retina.detect(out) if residual_visibility(crop_face(out, f))["visible"]]
        gt_faces += len(gt)
        leftover += len(vis)
        for g in gt:
            if not any(g.iou(v) >= iou_match for v in vis):
                covered += 1
        if frames >= max_frames:
            break
    return {
        "frames": frames,
        "input_faces_proxy": gt_faces,
        "residual_visible_faces": leftover,
        "residual_face_rate": leftover / max(1, gt_faces),
        "covered_gt_faces": covered,
        "coverage_rate": covered / max(1, gt_faces),
        "note": (
            "GT is RetinaFace on the input (no human labels). "
            "residual_face_rate is leftover real-looking faces / input faces."
        ),
    }
