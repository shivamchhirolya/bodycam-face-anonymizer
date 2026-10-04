from __future__ import annotations

from dataclasses import dataclass
from typing import List

from face_anon.types import FaceBox


@dataclass
class _Track:
    track_id: int
    box: FaceBox
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    age: int = 0
    time_since_update: int = 0


class IoUTracker:
    """IoU tracker with constant-velocity coast.

    Identity does not matter. What matters is that the blur follows the
    face when the camera pans or the person walks during a detector miss.
    """

    def __init__(
        self,
        iou_match: float = 0.22,
        center_match_frac: float = 0.35,
        max_age: int = 14,
        min_hits: int = 1,
        smooth: float = 0.55,
        vel_smooth: float = 0.6,
    ) -> None:
        self.iou_match = iou_match
        self.center_match_frac = center_match_frac
        self.max_age = max_age
        self.min_hits = min_hits
        self.smooth = smooth
        self.vel_smooth = vel_smooth
        self._next_id = 1
        self._tracks: list[_Track] = []

    def reset(self) -> None:
        self._next_id = 1
        self._tracks = []

    def _shift(self, tr: _Track) -> FaceBox:
        return FaceBox(
            x1=tr.box.x1 + tr.vx,
            y1=tr.box.y1 + tr.vy,
            x2=tr.box.x2 + tr.vx,
            y2=tr.box.y2 + tr.vy,
            score=tr.box.score,
            landmarks=tr.box.landmarks,
            track_id=tr.track_id,
        )

    def update(self, detections: List[FaceBox]) -> List[FaceBox]:
        for tr in self._tracks:
            if tr.time_since_update >= 1:
                tr.box = self._shift(tr)
            tr.age += 1
            tr.time_since_update += 1

        unmatched_dets = set(range(len(detections)))
        unmatched_trks = set(range(len(self._tracks)))
        pairs: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(self._tracks):
            for di, det in enumerate(detections):
                iou = tr.box.iou(det)
                if iou >= self.iou_match:
                    pairs.append((iou, ti, di))
                    continue
                diag = (tr.box.width**2 + tr.box.height**2) ** 0.5
                dist = ((tr.box.cx - det.cx) ** 2 + (tr.box.cy - det.cy) ** 2) ** 0.5
                if diag > 1 and dist <= self.center_match_frac * max(diag, 1.0):
                    pairs.append((0.15 + 0.01 * (1.0 - dist / (diag + 1e-6)), ti, di))

        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            unmatched_trks.discard(ti)
            unmatched_dets.discard(di)
            self._apply_detection(self._tracks[ti], detections[di])

        for di in unmatched_dets:
            det = detections[di]
            self._tracks.append(_Track(track_id=self._next_id, box=det))
            self._next_id += 1

        self._tracks = [
            tr
            for i, tr in enumerate(self._tracks)
            if not (i in unmatched_trks and tr.time_since_update > self.max_age)
        ]

        out: List[FaceBox] = []
        for tr in self._tracks:
            if tr.hits >= self.min_hits or tr.time_since_update == 0:
                out.append(
                    FaceBox(
                        x1=tr.box.x1,
                        y1=tr.box.y1,
                        x2=tr.box.x2,
                        y2=tr.box.y2,
                        score=tr.box.score,
                        landmarks=tr.box.landmarks,
                        track_id=tr.track_id,
                    )
                )
        return out

    def _apply_detection(self, tr: _Track, det: FaceBox) -> None:
        a = self.smooth
        old = tr.box
        merged = FaceBox(
            x1=a * old.x1 + (1 - a) * det.x1,
            y1=a * old.y1 + (1 - a) * det.y1,
            x2=a * old.x2 + (1 - a) * det.x2,
            y2=a * old.y2 + (1 - a) * det.y2,
            score=det.score,
            landmarks=det.landmarks,
            track_id=tr.track_id,
        )
        b = self.vel_smooth
        tr.vx = b * tr.vx + (1 - b) * (merged.cx - old.cx)
        tr.vy = b * tr.vy + (1 - b) * (merged.cy - old.cy)
        tr.box = merged
        tr.hits += 1
        tr.time_since_update = 0

    @property
    def active_count(self) -> int:
        return len(self._tracks)

    @property
    def lost_track_ids(self) -> list[int]:
        return [tr.track_id for tr in self._tracks if tr.time_since_update == self.max_age]


class ByteTracker(IoUTracker):
    """ByteTrack-style two-stage association (Zhang et al., 2022), for faces.

    Stage 1: match high-score detections to tracks.
    Stage 2: leftover tracks try the low-score detections that YuNet/SCRFD
    almost dropped. Same velocity coast as IoUTracker. Identity still does
    not matter; the extra stage is only to keep painting through a blink.
    """

    def __init__(self, high_thresh: float = 0.50, low_thresh: float = 0.18, **kwargs) -> None:
        super().__init__(**kwargs)
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh

    def _associate(self, detections: List[FaceBox], track_idxs: set[int], det_idxs: set[int]) -> tuple[set[int], set[int]]:
        pairs: list[tuple[float, int, int]] = []
        for ti in track_idxs:
            tr = self._tracks[ti]
            for di in det_idxs:
                det = detections[di]
                iou = tr.box.iou(det)
                if iou >= self.iou_match:
                    pairs.append((iou, ti, di))
                    continue
                diag = (tr.box.width**2 + tr.box.height**2) ** 0.5
                dist = ((tr.box.cx - det.cx) ** 2 + (tr.box.cy - det.cy) ** 2) ** 0.5
                if diag > 1 and dist <= self.center_match_frac * max(diag, 1.0):
                    pairs.append((0.15 + 0.01 * (1.0 - dist / (diag + 1e-6)), ti, di))
        pairs.sort(reverse=True)
        for _, ti, di in pairs:
            if ti not in track_idxs or di not in det_idxs:
                continue
            track_idxs.discard(ti)
            det_idxs.discard(di)
            self._apply_detection(self._tracks[ti], detections[di])
        return track_idxs, det_idxs

    def update(self, detections: List[FaceBox]) -> List[FaceBox]:
        for tr in self._tracks:
            if tr.time_since_update >= 1:
                tr.box = self._shift(tr)
            tr.age += 1
            tr.time_since_update += 1

        high = [d for d in detections if d.score >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d.score < self.high_thresh]
        leftover_tr = set(range(len(self._tracks)))
        leftover_hi = set(range(len(high)))
        leftover_tr, leftover_hi = self._associate(high, leftover_tr, leftover_hi)
        leftover_lo = set(range(len(low)))
        leftover_tr, leftover_lo = self._associate(low, leftover_tr, leftover_lo)

        for di in leftover_hi:
            det = high[di]
            self._tracks.append(_Track(track_id=self._next_id, box=det))
            self._next_id += 1

        self._tracks = [
            tr
            for i, tr in enumerate(self._tracks)
            if not (i in leftover_tr and tr.time_since_update > self.max_age)
        ]

        out: List[FaceBox] = []
        for tr in self._tracks:
            if tr.hits >= self.min_hits or tr.time_since_update == 0:
                out.append(
                    FaceBox(
                        x1=tr.box.x1,
                        y1=tr.box.y1,
                        x2=tr.box.x2,
                        y2=tr.box.y2,
                        score=tr.box.score,
                        landmarks=tr.box.landmarks,
                        track_id=tr.track_id,
                    )
                )
        return out
