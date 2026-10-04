from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np

from face_anon.preprocess import resize_for_detect
from face_anon.types import FaceBox


def nms_merge(faces: Sequence[FaceBox], iou_thresh: float = 0.4) -> List[FaceBox]:
    ordered = sorted(faces, key=lambda f: f.score, reverse=True)
    kept: List[FaceBox] = []
    for face in ordered:
        if any(face.iou(k) >= iou_thresh for k in kept):
            continue
        kept.append(face)
    return kept


def landmarks_look_like_face(face: FaceBox) -> bool:
    """YuNet / RetinaFace order: left eye, right eye, nose, left mouth, right mouth.

    A torso or wall box almost never has this stack. A real close-up does,
    even when the box fills the frame.
    """
    if face.landmarks is None:
        return False
    pts = np.asarray(face.landmarks, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 5:
        return False
    slack = 0.30 * max(face.width, face.height, 1.0)
    inside = 0
    for x, y in pts[:5]:
        if face.x1 - slack <= float(x) <= face.x2 + slack and face.y1 - slack <= float(y) <= face.y2 + slack:
            inside += 1
    if inside < 4:
        return False
    left_eye, right_eye, nose, left_mouth, right_mouth = pts[:5]
    eye_y = 0.5 * (float(left_eye[1]) + float(right_eye[1]))
    mouth_y = 0.5 * (float(left_mouth[1]) + float(right_mouth[1]))
    if not (eye_y + 1.0 < float(nose[1]) < mouth_y - 1.0):
        return False
    if abs(float(left_eye[1]) - float(right_eye[1])) > 0.50 * face.height:
        return False
    eye_dist = abs(float(right_eye[0]) - float(left_eye[0]))
    if eye_dist < 0.10 * face.width or eye_dist > 0.90 * face.width:
        return False
    return True


def is_sane_box(face: FaceBox, width: int, height: int, min_face: float) -> bool:
    """Keep real faces of any size. Drop torso / wall false hits.

    Body-cam footage often has someone centimetres from the lens: the
    box can be the whole frame. Size is never a reject reason. Score and
    5-point geometry are.
    """
    if face.width < min_face or face.height < min_face:
        return False
    aspect = face.width / max(face.height, 1e-3)
    if aspect < 0.40 or aspect > 2.2:
        return False

    area_frac = (face.width * face.height) / max(1.0, width * height)
    large = area_frac > 0.08 or face.width > 0.35 * width or face.height > 0.38 * height
    if not large:
        return True

    if landmarks_look_like_face(face) and face.score >= 0.32:
        return True
    # Motion-blur close-up can scramble landmarks; a strong score still counts.
    if face.score >= 0.75:
        return True
    return False


def scale_boxes(faces: Iterable[FaceBox], inv: float, width: int, height: int, min_face: float) -> List[FaceBox]:
    out: List[FaceBox] = []
    for f in faces:
        box = FaceBox(
            x1=f.x1 * inv,
            y1=f.y1 * inv,
            x2=f.x2 * inv,
            y2=f.y2 * inv,
            score=f.score,
            landmarks=None if f.landmarks is None else f.landmarks * inv,
        )
        clipped = box.clip(width, height)
        if is_sane_box(clipped, width, height, min_face):
            out.append(clipped)
    return out


def detect_multiscale(
    detector,
    image_bgr,
    long_sides: Sequence[int],
    width: int,
    height: int,
    min_face: float = 6.0,
) -> List[FaceBox]:
    """Run the same finder at one or more long-side sizes (upsample or shrink).

    Small body-cam frames (480×360) must be *enlarged* so an 8–12 px head
    becomes 20–30 px. The old resize only shrank, which is why far airport
    faces stayed visible.
    """
    found: List[FaceBox] = []
    seen_sides: set[int] = set()
    for long_side in long_sides:
        if long_side in seen_sides:
            continue
        seen_sides.add(int(long_side))
        det_img, scale = resize_for_detect(image_bgr, int(long_side))
        raw = detector.detect(det_img)
        found.extend(scale_boxes(raw, 1.0 / scale, width, height, min_face))
    return nms_merge(found)


def detect_tiled(
    detector,
    image_bgr,
    width: int,
    height: int,
    tile: int = 640,
    overlap: float = 0.20,
    min_face: float = 6.0,
) -> List[FaceBox]:
    """Slide a tile over an upsampled copy so far heads get more pixels.

    Used only on the high-risk path. A 480×360 frame is enlarged so the long
    side is 2×tile, then 640 windows with overlap are scored and mapped back.
    """
    long_side = max(2 * int(tile), max(width, height))
    det_img, scale = resize_for_detect(image_bgr, long_side)
    dh, dw = det_img.shape[:2]
    step = max(32, int(tile * (1.0 - overlap)))
    found: List[FaceBox] = []
    ys = list(range(0, max(1, dh - tile + 1), step))
    xs = list(range(0, max(1, dw - tile + 1), step))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]
    if ys[-1] + tile < dh:
        ys.append(max(0, dh - tile))
    if xs[-1] + tile < dw:
        xs.append(max(0, dw - tile))
    for y0 in ys:
        for x0 in xs:
            y1 = min(dh, y0 + tile)
            x1 = min(dw, x0 + tile)
            crop = det_img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            for f in detector.detect(crop):
                box = FaceBox(
                    x1=(f.x1 + x0) / scale,
                    y1=(f.y1 + y0) / scale,
                    x2=(f.x2 + x0) / scale,
                    y2=(f.y2 + y0) / scale,
                    score=f.score,
                    landmarks=None if f.landmarks is None else (f.landmarks + np.array([x0, y0])) / scale,
                )
                clipped = box.clip(width, height)
                if is_sane_box(clipped, width, height, min_face):
                    found.append(clipped)
    return nms_merge(found)


def detect_on_frame(
    detector,
    image_bgr,
    width: int,
    height: int,
    long_side: int,
    extra_long_sides: Optional[Sequence[int]] = None,
    strong_detector=None,
    min_face: float = 6.0,
) -> List[FaceBox]:
    sides = [int(long_side), *(int(s) for s in (extra_long_sides or []))]
    boxes = detect_multiscale(detector, image_bgr, sides, width, height, min_face)
    if strong_detector is not None:
        strong = strong_detector.detect(image_bgr)
        extra = []
        for f in strong:
            clipped = f.clip(width, height)
            if is_sane_box(clipped, width, height, min_face):
                extra.append(clipped)
        boxes = nms_merge(boxes + extra)
    return boxes
