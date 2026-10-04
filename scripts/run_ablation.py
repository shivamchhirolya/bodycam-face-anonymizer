#!/usr/bin/env python3
"""Ablation on a full video: detector × resolution × device, risk-gated tiles, tracker.

Stand-in GT is RetinaFace-R50 on the *input* (no human boxes exist for this
footage). Every detector config is scored on the same sampled frames; the
tracker pass runs on every frame so coasting is real.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from face_anon.detect_ensemble import detect_multiscale, detect_tiled, nms_merge  # noqa: E402
from face_anon.detect_retinaface import RetinaFaceDetector  # noqa: E402
from face_anon.detect_scrfd import SCRFDDetector  # noqa: E402
from face_anon.detect_yunet import YuNetDetector  # noqa: E402
from face_anon.preprocess import any_tile_dark  # noqa: E402
from face_anon.track import ByteTracker, IoUTracker  # noqa: E402
from face_anon.types import FaceBox  # noqa: E402
from face_anon.video_io import iter_frames, probe_video  # noqa: E402

BUCKETS = [("tiny <16px", 0, 16), ("small 16-32px", 16, 32), ("medium 32-64px", 32, 64), ("large >=64px", 64, 10**9)]
MIN_FACE = 6.0


def match(gt: list[FaceBox], pred: list[FaceBox], iou: float = 0.20) -> list[bool]:
    used: set[int] = set()
    hits = []
    for g in gt:
        hit = False
        for j, p in enumerate(pred):
            if j not in used and g.iou(p) >= iou:
                used.add(j)
                hit = True
                break
        hits.append(hit)
    return hits


def bucket_of(face: FaceBox) -> str:
    side = min(face.width, face.height)
    for name, lo, hi in BUCKETS:
        if lo <= side < hi:
            return name
    return BUCKETS[-1][0]


class Score:
    def __init__(self) -> None:
        self.tp = self.gt = self.pred = 0
        self.seconds = 0.0
        self.calls = 0
        self.by_bucket = {name: [0, 0] for name, _, _ in BUCKETS}

    def add(self, gt: list[FaceBox], pred: list[FaceBox], seconds: float) -> None:
        hits = match(gt, pred)
        self.tp += sum(hits)
        self.gt += len(gt)
        self.pred += len(pred)
        self.seconds += seconds
        self.calls += 1
        for g, h in zip(gt, hits):
            b = self.by_bucket[bucket_of(g)]
            b[0] += int(h)
            b[1] += 1

    def row(self) -> dict:
        ms = 1000.0 * self.seconds / max(1, self.calls)
        return {
            "recall": round(self.tp / max(1, self.gt), 4),
            "precision": round(self.tp / max(1, self.pred), 4),
            "gt_faces": self.gt,
            "pred_faces": self.pred,
            "matched": self.tp,
            "missed": self.gt - self.tp,
            "ms_per_frame": round(ms, 2),
            "detect_fps": round(1000.0 / ms, 1) if ms > 0 else None,
            "frames": self.calls,
            "recall_by_size": {
                k: {"recall": round(v[0] / v[1], 4) if v[1] else None, "gt": v[1]}
                for k, v in self.by_bucket.items()
            },
        }


def run_cfg(cfg: dict, det, frame) -> list[FaceBox]:
    h, w = frame.shape[:2]
    if cfg["mode"] == "tiles":
        base = detect_multiscale(det, frame, [640], w, h, MIN_FACE)
        tiled = detect_tiled(det, frame, w, h, tile=640, overlap=0.20, min_face=MIN_FACE)
        return nms_merge(list(base) + list(tiled))
    return detect_multiscale(det, frame, [cfg["long_side"]], w, h, MIN_FACE)


def make_det(model: str, long_side: int, device: str, score: float = 0.28):
    if model == "yunet":
        return YuNetDetector(score_threshold=score)
    return SCRFDDetector(score_threshold=score, input_long_side=long_side, device=device)


def detector_pass(video: Path, every: int, retina: RetinaFaceDetector, log) -> dict:
    cfgs = [
        {"id": "yunet_640_cpu", "model": "yunet", "long_side": 640, "mode": "full", "device": "cpu"},
        {"id": "yunet_960_cpu", "model": "yunet", "long_side": 960, "mode": "full", "device": "cpu"},
        {"id": "yunet_tiles_cpu", "model": "yunet", "long_side": 640, "mode": "tiles", "device": "cpu"},
        {"id": "scrfd_640_gpu", "model": "scrfd", "long_side": 640, "mode": "full", "device": "cuda"},
        {"id": "scrfd_960_gpu", "model": "scrfd", "long_side": 960, "mode": "full", "device": "cuda"},
        {"id": "scrfd_tiles_gpu", "model": "scrfd", "long_side": 640, "mode": "tiles", "device": "cuda"},
    ]
    dets = {c["id"]: make_det(c["model"], c["long_side"], c["device"]) for c in cfgs}
    warm = [f for i, f in iter_frames(video) if i % 50 == 0][:6]
    for c in cfgs:
        for f in warm:
            run_cfg(c, dets[c["id"]], f)
    for f in warm:
        retina.detect(f)
    scores = {c["id"]: Score() for c in cfgs}
    gated = Score()
    gated_frames = 0
    gt_seconds = 0.0
    sampled = 0
    prev_small = None
    risk_motion = 0
    for idx, frame in iter_frames(video):
        small = frame[::16, ::16]
        motion = 0.0
        if prev_small is not None and small.shape == prev_small.shape:
            motion = float((small.astype("int16") - prev_small.astype("int16")).std())
        prev_small = small
        if idx % every:
            continue
        sampled += 1
        t = time.perf_counter()
        gt = [f for f in retina.detect(frame) if f.width >= 8 and f.height >= 8]
        gt_seconds += time.perf_counter() - t
        outs = {}
        for c in cfgs:
            det = dets[c["id"]]
            t = time.perf_counter()
            pred = run_cfg(c, det, frame)
            dt = time.perf_counter() - t
            scores[c["id"]].add(gt, pred, dt)
            outs[c["id"]] = (pred, dt)
        # The proposed graph: SCRFD-640 always, tiles only on a risk signal.
        base, base_dt = outs["scrfd_640_gpu"]
        dark = any_tile_dark(frame, 72.0)
        tiny = any(min(b.width, b.height) < 24 for b in base)
        risky = dark or motion > 18 or tiny
        if motion > 18:
            risk_motion += 1
        if risky:
            gated_frames += 1
            tiles_pred, tiles_dt = outs["scrfd_tiles_gpu"]
            gated.add(gt, nms_merge(list(base) + list(tiles_pred)), base_dt + tiles_dt)
        else:
            gated.add(gt, base, base_dt)
        if sampled % 200 == 0:
            log(f"detector pass: {sampled} sampled frames (frame {idx})")
    rows = []
    for c in cfgs:
        rows.append({**c, **scores[c["id"]].row()})
    rows.append(
        {
            "id": "scrfd_640_gpu+risk_tiles",
            "model": "scrfd",
            "long_side": 640,
            "mode": "risk-gated tiles",
            "device": "cuda",
            **gated.row(),
            "risk_frames": gated_frames,
            "risk_frame_rate": round(gated_frames / max(1, sampled), 4),
        }
    )
    return {
        "every": every,
        "sampled_frames": sampled,
        "gt_ms_per_frame_retinaface_gpu": round(1000.0 * gt_seconds / max(1, sampled), 2),
        "rows": rows,
    }


def latency_cpu_extras(video: Path, log) -> list[dict]:
    """SCRFD and RetinaFace on CPU are too slow for the full pass; time 48 frames."""
    step = max(1, probe_video(video)["nframes"] // 56)
    frames = [f for i, f in iter_frames(video) if i % step == 0][:56]
    out = []
    for name, det in [
        ("scrfd_640_cpu", SCRFDDetector(score_threshold=0.28, input_long_side=640, device="cpu")),
        ("retinaface_640_cpu", RetinaFaceDetector(score_threshold=0.45, input_long_side=640, device="cpu")),
        ("retinaface_640_gpu", RetinaFaceDetector(score_threshold=0.45, input_long_side=640, device="cuda")),
    ]:
        for f in frames[:8]:
            det.detect(f)
        t = time.perf_counter()
        for f in frames[8:]:
            det.detect(f)
        ms = 1000.0 * (time.perf_counter() - t) / len(frames[8:])
        out.append({"id": name, "ms_per_frame": round(ms, 2), "fps": round(1000.0 / ms, 1), "frames": len(frames[8:])})
        log(f"latency {name}: {ms:.2f} ms")
    return out


def tracker_pass(video: Path, every: int, retina: RetinaFaceDetector, log, variants: list[str]) -> list[dict]:
    """Every frame, both detectors, each association rule in `variants`."""
    detectors = {
        "yunet_640_cpu": YuNetDetector(score_threshold=0.18),
        "scrfd_640_gpu": SCRFDDetector(score_threshold=0.18, input_long_side=640, device="cuda"),
    }
    common = dict(iou_match=0.22, center_match_frac=0.35, max_age=14, min_hits=1, smooth=0.55)
    factories = {
        "iou_coast": lambda: IoUTracker(**common),
        "bytetrack": lambda: ByteTracker(high_thresh=0.50, low_thresh=0.18, **common),
        "bytetrack_privacy": lambda: ByteTracker(high_thresh=0.28, low_thresh=0.18, **common),
    }
    names = tuple(variants)

    def trackers():
        return {n: factories[n]() for n in names}

    state = {d: trackers() for d in detectors}
    scores = {(d, t): Score() for d in detectors for t in names}
    raw_scores = {d: Score() for d in detectors}
    painted = {(d, t): 0 for d in detectors for t in names}
    track_seconds = {(d, t): 0.0 for d in detectors for t in names}
    n = 0
    for idx, frame in iter_frames(video):
        n += 1
        h, w = frame.shape[:2]
        gt = None
        if idx % every == 0:
            gt = [f for f in retina.detect(frame) if f.width >= 8 and f.height >= 8]
        for dname, det in detectors.items():
            raw = detect_multiscale(det, frame, [640], w, h, MIN_FACE)
            if gt is not None:
                raw_scores[dname].add(gt, [d for d in raw if d.score >= 0.28], 0.0)
            for tname, tr in state[dname].items():
                dets = [d for d in raw if d.score >= (0.28 if tname == "iou_coast" else 0.18)]
                t = time.perf_counter()
                out = tr.update(dets)
                track_seconds[(dname, tname)] += time.perf_counter() - t
                painted[(dname, tname)] += len(out)
                if gt is not None:
                    scores[(dname, tname)].add(gt, out, 0.0)
        if n % 3000 == 0:
            log(f"tracker pass: frame {idx}")
    rows = []
    for dname in detectors:
        r = raw_scores[dname].row()
        rows.append({"detector": dname, "tracker": "none (raw detections)", "recall": r["recall"],
                     "precision": r["precision"], "missed": r["missed"], "gt_faces": r["gt_faces"]})
        for tname in names:
            r = scores[(dname, tname)].row()
            tr = state[dname][tname]
            rows.append(
                {
                    "detector": dname,
                    "tracker": tname,
                    "recall": r["recall"],
                    "precision": r["precision"],
                    "missed": r["missed"],
                    "gt_faces": r["gt_faces"],
                    "tracks_started": tr._next_id - 1,
                    "boxes_painted_total": painted[(dname, tname)],
                    "track_ms_per_frame": round(1000.0 * track_seconds[(dname, tname)] / max(1, n), 4),
                    "recall_by_size": r["recall_by_size"],
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "data" / "samples" / "airport_full.mp4"))
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "demo" / "ablation_full.json"))
    ap.add_argument("--skip-tracker", action="store_true")
    ap.add_argument("--tracker-only", action="store_true", help="merge tracker rows into an existing --out")
    ap.add_argument("--trackers", default="iou_coast,bytetrack")
    args = ap.parse_args()
    variants = [t.strip() for t in args.trackers.split(",") if t.strip()]

    video = Path(args.video)
    meta = probe_video(video)
    t0 = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", flush=True)

    log(f"video {video.name} {meta['width']}x{meta['height']} {meta['nframes']} frames @ {meta['fps']}")
    retina = RetinaFaceDetector(score_threshold=0.45, input_long_side=640, device="cuda")
    if args.tracker_only:
        report = json.loads(Path(args.out).read_text())
        new_rows = tracker_pass(video, args.every, retina, log, variants)
        kept = [r for r in report.get("tracker", []) if r["tracker"] not in variants]
        report["tracker"] = kept + [r for r in new_rows if r["tracker"] in variants]
        Path(args.out).write_text(json.dumps(report, indent=2))
        log(f"merged tracker rows {variants} into {args.out}")
        return 0
    report = {
        "video": str(video),
        "meta": meta,
        "gt": "RetinaFace-R50 on the input (CUDA), score>=0.45, box>=8px. Not human labels.",
        "hardware": "NVIDIA L40S (onnxruntime-gpu 1.23.2 CUDA EP); YuNet runs on CPU via OpenCV DNN.",
        "latency_cpu_extras": latency_cpu_extras(video, log),
        "detectors": detector_pass(video, args.every, retina, log),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    log("detector pass written")
    if not args.skip_tracker:
        report["tracker"] = tracker_pass(video, args.every, retina, log, variants)
        Path(args.out).write_text(json.dumps(report, indent=2))
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
