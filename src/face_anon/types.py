from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np


@dataclass
class FaceBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    landmarks: Optional[np.ndarray] = None  # (5, 2) if available
    track_id: Optional[int] = None

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    def clip(self, width: int, height: int) -> "FaceBox":
        return FaceBox(
            x1=float(np.clip(self.x1, 0, width - 1)),
            y1=float(np.clip(self.y1, 0, height - 1)),
            x2=float(np.clip(self.x2, 0, width - 1)),
            y2=float(np.clip(self.y2, 0, height - 1)),
            score=self.score,
            landmarks=self.landmarks,
            track_id=self.track_id,
        )

    def expanded(self, pad: float, width: int, height: int) -> "FaceBox":
        px = self.width * pad
        py = self.height * pad
        out = FaceBox(
            x1=self.x1 - px,
            y1=self.y1 - py,
            x2=self.x2 + px,
            y2=self.y2 + py,
            score=self.score,
            landmarks=self.landmarks,
            track_id=self.track_id,
        )
        return out.clip(width, height)

    def iou(self, other: "FaceBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.width * self.height + other.width * other.height - inter
        return float(inter / union) if union > 0 else 0.0

    def as_xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class Leak:
    t_sec: float
    frame_index: int
    score: float
    box: list[float]


@dataclass
class QCReport:
    status: str  # pass | needs_review | skipped
    frames_checked: int
    leak_count: int
    leak_rate: float
    leaks: list[Leak] = field(default_factory=list)
    model: str = "retinaface_r50"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class JobRecord:
    job_id: str
    upload_id: str
    content_fp: str
    config_hash: str
    input_path: str
    output_path: str
    status: str
    reused: bool = False
    qc_status: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metrics_json: Optional[str] = None
