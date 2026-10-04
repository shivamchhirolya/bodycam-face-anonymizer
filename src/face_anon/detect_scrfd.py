from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from face_anon.config import MODELS_DIR
from face_anon.runtime import make_session, session_device
from face_anon.types import FaceBox

SCRFD_NAME = "det_10g.onnx"
_STRIDES = (8, 16, 32)
_NUM_ANCHORS = 2


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    out = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        out.extend([px, py])
    return np.stack(out, axis=-1)


def _letterbox(image_bgr: np.ndarray, long_side: int) -> tuple[np.ndarray, float, int, int]:
    h, w = image_bgr.shape[:2]
    scale = long_side / float(max(h, w))
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    if new_w % 2:
        new_w += 1
    if new_h % 2:
        new_h += 1
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((long_side, long_side, 3), 127, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas, scale, new_w, new_h


class SCRFDDetector:
    """InsightFace buffalo_l det_10g = SCRFD-10G-KPS (Guo et al., 2021).

    Same 5-point order as YuNet / RetinaFace. Heavier than YuNet; better on
    the WIDER FACE Hard / tiny-head tail. ~7× YuNet's cost on CPU, so it is
    the hot path only when a CUDA provider is available.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        score_threshold: float = 0.32,
        nms_threshold: float = 0.40,
        input_long_side: int = 640,
        device: str = "auto",
    ) -> None:
        path = Path(model_path) if model_path else MODELS_DIR / SCRFD_NAME
        if not path.is_file():
            raise FileNotFoundError(f"SCRFD-10G missing at {path}")
        self.session = make_session(str(path), device)
        self.device = session_device(self.session)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.input_long_side = int(input_long_side)
        self._center_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def _centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        cached = self._center_cache.get(key)
        if cached is not None:
            return cached
        yy, xx = np.mgrid[0:height, 0:width]
        centers = np.stack([xx, yy], axis=-1).astype(np.float32).reshape(-1, 2) * stride
        if _NUM_ANCHORS > 1:
            centers = np.stack([centers] * _NUM_ANCHORS, axis=1).reshape(-1, 2)
        self._center_cache[key] = centers
        return centers

    def detect(self, image_bgr: np.ndarray) -> List[FaceBox]:
        img, scale, _nw, _nh = _letterbox(image_bgr, self.input_long_side)
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / 128.0, (self.input_long_side, self.input_long_side),
            (127.5, 127.5, 127.5), swapRB=True,
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        scores_list, bboxes_list, kps_list = [], [], []
        fmc = 3
        input_h = input_w = self.input_long_side
        for idx, stride in enumerate(_STRIDES):
            scores = net_outs[idx].reshape(-1)
            bbox_preds = net_outs[idx + fmc] * stride
            kps_preds = net_outs[idx + fmc * 2] * stride
            height = input_h // stride
            width = input_w // stride
            centers = self._centers(height, width, stride)
            pos = np.where(scores >= self.score_threshold)[0]
            if pos.size == 0:
                continue
            scores_list.append(scores[pos])
            bboxes_list.append(_distance2bbox(centers, bbox_preds)[pos])
            kps_list.append(_distance2kps(centers, kps_preds)[pos])
        if not scores_list:
            return []
        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list) / scale
        kpss = np.concatenate(kps_list) / scale
        order = scores.argsort()[::-1]
        scores, bboxes, kpss = scores[order], bboxes[order], kpss[order]
        xywh = np.stack(
            [bboxes[:, 0], bboxes[:, 1], bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]],
            axis=1,
        )
        keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.0, self.nms_threshold)
        if keep is None or len(keep) == 0:
            return []
        out: List[FaceBox] = []
        h, w = image_bgr.shape[:2]
        for i in np.array(keep).reshape(-1):
            x1, y1, x2, y2 = (float(v) for v in bboxes[int(i)])
            lms = kpss[int(i)].reshape(5, 2).astype(np.float32)
            out.append(
                FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, score=float(scores[int(i)]), landmarks=lms).clip(w, h)
            )
        return out
