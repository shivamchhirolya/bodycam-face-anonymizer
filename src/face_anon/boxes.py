from __future__ import annotations

from face_anon.types import FaceBox


def pad_for_face(face: FaceBox, frame_w: int, frame_h: int, base_pad: float) -> float:
    """Shrink pad as the box already fills the view.

    A 42% pad on a close-up paints walls. Far heads still get the base pad.
    """
    if frame_w <= 0 or frame_h <= 0:
        return base_pad
    area_frac = (face.width * face.height) / float(frame_w * frame_h)
    side_frac = max(face.width / float(frame_w), face.height / float(frame_h))
    if min(face.width, face.height) < 24:
        return base_pad * 1.35
    if area_frac >= 0.40 or side_frac >= 0.75:
        return min(base_pad, 0.08)
    if area_frac >= 0.18 or side_frac >= 0.50:
        return min(base_pad, 0.14)
    if area_frac >= 0.08 or side_frac >= 0.35:
        return min(base_pad, 0.22)
    return base_pad
