from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from face_anon.config import MODELS_DIR
from face_anon.types import FaceBox

YUNET_NAME = "face_detection_yunet_2023mar.onnx"


class YuNetDetector:
    """OpenCV YuNet, the default live detector.

    Chosen because a 232 KB model runs at hundreds of FPS on CPU, so every
    frame of a 720p/1080p body-cam can be scored without a GPU queue.
    Recall on hard poses is recovered by a low threshold, tracking coast,
    and an independent RetinaFace QC/retry, not by making this model heavier.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        score_threshold: float = 0.32,
        nms_threshold: float = 0.40,
        top_k: int = 5000,
    ) -> None:
        path = Path(model_path) if model_path else MODELS_DIR / YUNET_NAME
        if not path.is_file():
            raise FileNotFoundError(
                f"YuNet model missing at {path}. Run: python -m face_anon download-models"
            )
        self._path = str(path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._size: tuple[int, int] | None = None
        self._det = cv2.FaceDetectorYN.create(
            self._path, "", (320, 320), score_threshold, nms_threshold, top_k
        )

    def _ensure_size(self, w: int, h: int) -> None:
        if self._size == (w, h):
            return
        self._det.setInputSize((w, h))
        self._size = (w, h)

    def detect(self, image_bgr: np.ndarray) -> List[FaceBox]:
        h, w = image_bgr.shape[:2]
        self._ensure_size(w, h)
        ok, faces = self._det.detect(image_bgr)
        if faces is None or len(faces) == 0:
            return []
        out: List[FaceBox] = []
        for row in faces:
            x, y, bw, bh = row[:4]
            score = float(row[-1])
            if score < self.score_threshold:
                continue
            lms = np.array(row[4:14], dtype=np.float32).reshape(5, 2)
            out.append(
                FaceBox(
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + bw),
                    y2=float(y + bh),
                    score=score,
                    landmarks=lms,
                )
            )
        return out
