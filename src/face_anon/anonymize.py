from __future__ import annotations

import math
from typing import Iterable, Literal

import cv2
import numpy as np

from face_anon.boxes import pad_for_face
from face_anon.types import FaceBox

Method = Literal["hybrid", "pixelate", "blur", "blackout"]
Shape = Literal["ellipse", "rect"]


def _kernel_for_box(w: int, h: int) -> int:
    k = int(max(w, h) * 0.55)
    k = max(15, k | 1)
    return min(k, 151)


def _face_angle_deg(face: FaceBox) -> float:
    if face.landmarks is None or len(face.landmarks) < 2:
        return 0.0
    # YuNet / RetinaFace: 0 = left eye, 1 = right eye.
    lx, ly = float(face.landmarks[0][0]), float(face.landmarks[0][1])
    rx, ry = float(face.landmarks[1][0]), float(face.landmarks[1][1])
    return math.degrees(math.atan2(ry - ly, rx - lx))


def anonymize_faces(
    frame: np.ndarray,
    faces: Iterable[FaceBox],
    method: Method = "hybrid",
    pad: float = 0.40,
    pixel_blocks: int = 10,
    shape: Shape = "ellipse",
) -> np.ndarray:
    h, w = frame.shape[:2]
    for face in faces:
        use_pad = pad_for_face(face, w, h, pad)
        use_shape = "rect" if min(face.width, face.height) < 20 else shape
        box = face.expanded(use_pad, w, h)
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        redacted = _redact_roi(roi, method, pixel_blocks)
        if use_shape == "ellipse":
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            angle = _face_angle_deg(face)
            cv2.ellipse(
                mask,
                (roi.shape[1] // 2, roi.shape[0] // 2),
                (max(1, roi.shape[1] // 2), max(1, roi.shape[0] // 2)),
                angle,
                0,
                360,
                255,
                -1,
            )
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            alpha = (mask.astype(np.float32) / 255.0)[..., None]
            frame[y1:y2, x1:x2] = (alpha * redacted + (1.0 - alpha) * roi).astype(np.uint8)
        else:
            frame[y1:y2, x1:x2] = redacted
    return frame


def _redact_roi(roi: np.ndarray, method: Method, pixel_blocks: int) -> np.ndarray:
    rh, rw = roi.shape[:2]
    if method == "blackout":
        return np.zeros_like(roi)
    k = _kernel_for_box(rw, rh)
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    if method == "blur":
        return blurred
    blocks = max(4, min(pixel_blocks, min(rw, rh) // 2))
    small = cv2.resize(blurred if method == "hybrid" else roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
