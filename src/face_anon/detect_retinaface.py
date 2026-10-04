from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

from face_anon.config import MODELS_DIR
from face_anon.runtime import make_session, session_device
from face_anon.types import FaceBox

RETINAFACE_NAME = "retinaface_r50.onnx"

# biubug6 / Deng et al. RetinaFace ResNet-50, WIDER FACE training recipe.
CFG_R50 = {
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
}


def prior_boxes(height: int, width: int, cfg: dict | None = None) -> np.ndarray:
    cfg = cfg or CFG_R50
    anchors: list[float] = []
    for k, step in enumerate(cfg["steps"]):
        fm_h = int(math.ceil(height / step))
        fm_w = int(math.ceil(width / step))
        for i in range(fm_h):
            for j in range(fm_w):
                for min_size in cfg["min_sizes"][k]:
                    cx = (j + 0.5) * step / width
                    cy = (i + 0.5) * step / height
                    skx = min_size / width
                    sky = min_size / height
                    anchors.extend((cx, cy, skx, sky))
    return np.asarray(anchors, dtype=np.float32).reshape(-1, 4)


def decode_boxes(loc: np.ndarray, priors: np.ndarray, variances: Sequence[float]) -> np.ndarray:
    boxes = np.empty_like(loc)
    boxes[:, :2] = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
    boxes[:, 2:] = priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    boxes[:, :2] -= boxes[:, 2:] / 2.0
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landmarks(pre: np.ndarray, priors: np.ndarray, variances: Sequence[float]) -> np.ndarray:
    landms = np.empty_like(pre)
    for i in range(5):
        landms[:, 2 * i : 2 * i + 2] = (
            priors[:, :2] + pre[:, 2 * i : 2 * i + 2] * variances[0] * priors[:, 2:]
        )
    return landms


def nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> list[int]:
    if len(boxes) == 0:
        return []
    xywh = np.stack(
        [
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
        ],
        axis=1,
    )
    idxs = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.0, thresh)
    if idxs is None or len(idxs) == 0:
        return []
    return [int(i) for i in np.array(idxs).reshape(-1)]


def letterbox(image: np.ndarray, long_side: int) -> tuple[np.ndarray, float, int, int]:
    h, w = image.shape[:2]
    scale = min(1.0, long_side / float(max(h, w)))
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    # Feature maps use ceil(dim/step); pad to a multiple of 32 so priors align.
    pad_w = (32 - new_w % 32) % 32
    pad_h = (32 - new_h % 32) % 32
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR) if scale != 1.0 else image
    if pad_w or pad_h:
        resized = cv2.copyMakeBorder(
            resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(104, 117, 123)
        )
    return resized, scale, pad_w, pad_h


class RetinaFaceDetector:
    """Independent QC detector: RetinaFace-ResNet50 (Deng et al., CVPR 2020).

    Different architecture, priors, and training recipe than YuNet, so it
    does not share the same miss pattern. Used on a small smart sample of
    *output* frames, never as the default per-frame anonymizer.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        score_threshold: float = 0.50,
        nms_threshold: float = 0.40,
        input_long_side: int = 640,
        device: str = "auto",
    ) -> None:
        path = Path(model_path) if model_path else MODELS_DIR / RETINAFACE_NAME
        if not path.is_file():
            raise FileNotFoundError(
                f"RetinaFace model missing at {path}. Run: python -m face_anon download-models"
            )
        self.session = make_session(str(path), device)
        self.device = session_device(self.session)
        self.input_name = self.session.get_inputs()[0].name
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.input_long_side = input_long_side
        self._prior_cache: dict[tuple[int, int], np.ndarray] = {}

    def _priors(self, h: int, w: int) -> np.ndarray:
        key = (h, w)
        if key not in self._prior_cache:
            self._prior_cache[key] = prior_boxes(h, w)
        return self._prior_cache[key]

    def detect(self, image_bgr: np.ndarray) -> List[FaceBox]:
        img, scale, _pad_w, _pad_h = letterbox(image_bgr, self.input_long_side)
        h, w = img.shape[:2]
        blob = img.astype(np.float32)
        blob -= np.array([104.0, 117.0, 123.0], dtype=np.float32)
        blob = blob.transpose(2, 0, 1)[None, ...]

        loc, conf, landms = self.session.run(None, {self.input_name: blob})
        loc = loc[0]
        scores = conf[0][:, 1]
        landms = landms[0]
        priors = self._priors(h, w)
        if priors.shape[0] != loc.shape[0]:
            # Dynamic ONNX vs prior mismatch: rebuild without the cache.
            priors = prior_boxes(h, w)
            self._prior_cache[(h, w)] = priors

        keep = scores > self.score_threshold
        if not np.any(keep):
            return []
        loc = loc[keep]
        scores = scores[keep]
        landms = landms[keep]
        priors = priors[keep]

        boxes = decode_boxes(loc, priors, CFG_R50["variance"])
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        lms = decode_landmarks(landms, priors, CFG_R50["variance"])
        lms[:, 0::2] *= w
        lms[:, 1::2] *= h

        if scale != 1.0:
            boxes /= scale
            lms /= scale

        order = nms(boxes, scores, self.nms_threshold)
        orig_h, orig_w = image_bgr.shape[:2]
        out: List[FaceBox] = []
        for i in order:
            x1, y1, x2, y2 = boxes[i]
            out.append(
                FaceBox(
                    x1=float(np.clip(x1, 0, orig_w - 1)),
                    y1=float(np.clip(y1, 0, orig_h - 1)),
                    x2=float(np.clip(x2, 0, orig_w - 1)),
                    y2=float(np.clip(y2, 0, orig_h - 1)),
                    score=float(scores[i]),
                    landmarks=lms[i].reshape(5, 2),
                )
            )
        return out
