#!/usr/bin/env python3
"""Threshold-free detector comparison: PR curve, AP, recall at matched precision.

A single score cut (0.28) is unfair between YuNet and SCRFD because their
scores are calibrated differently. Here every detector keeps boxes down to
0.05 and is scored over a threshold grid against the same stand-in GT.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from face_anon.detect_ensemble import detect_multiscale, detect_tiled, nms_merge  # noqa: E402
from face_anon.detect_retinaface import RetinaFaceDetector  # noqa: E402
from face_anon.detect_scrfd import SCRFDDetector  # noqa: E402
from face_anon.detect_yunet import YuNetDetector  # noqa: E402
from face_anon.types import FaceBox  # noqa: E402
from face_anon.video_io import iter_frames  # noqa: E402

THRESHOLDS = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
MIN_FACE = 6.0


def match_count(gt: list[FaceBox], pred: list[FaceBox], iou: float = 0.20) -> int:
    used: set[int] = set()
    tp = 0
    for g in gt:
        for j, p in enumerate(pred):
            if j not in used and g.iou(p) >= iou:
                used.add(j)
                tp += 1
                break
    return tp


def run(det, frame, mode: str, long_side: int) -> list[FaceBox]:
    h, w = frame.shape[:2]
    if mode == "tiles":
        base = detect_multiscale(det, frame, [640], w, h, MIN_FACE)
        return nms_merge(list(base) + list(detect_tiled(det, frame, w, h, 640, 0.20, MIN_FACE)))
    return detect_multiscale(det, frame, [long_side], w, h, MIN_FACE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "data" / "samples" / "airport_full.mp4"))
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "demo" / "pr_sweep_full.json"))
    args = ap.parse_args()

    cfgs = {
        "yunet_640": (YuNetDetector(score_threshold=0.05), "full", 640),
        "yunet_960": (YuNetDetector(score_threshold=0.05), "full", 960),
        "yunet_tiles": (YuNetDetector(score_threshold=0.05), "tiles", 640),
        "scrfd_640": (SCRFDDetector(score_threshold=0.05, input_long_side=640, device="cuda"), "full", 640),
        "scrfd_960": (SCRFDDetector(score_threshold=0.05, input_long_side=960, device="cuda"), "full", 960),
        "scrfd_tiles": (SCRFDDetector(score_threshold=0.05, input_long_side=640, device="cuda"), "tiles", 640),
    }
    retina = RetinaFaceDetector(score_threshold=0.45, input_long_side=640, device="cuda")
    # tp / pred counts per threshold; gt is shared.
    tp = {k: np.zeros(len(THRESHOLDS), dtype=np.int64) for k in cfgs}
    npred = {k: np.zeros(len(THRESHOLDS), dtype=np.int64) for k in cfgs}
    gt_total = 0
    frames = 0
    for idx, frame in iter_frames(args.video):
        if idx % args.every:
            continue
        frames += 1
        gt = [f for f in retina.detect(frame) if f.width >= 8 and f.height >= 8]
        gt_total += len(gt)
        for name, (det, mode, side) in cfgs.items():
            preds = run(det, frame, mode, side)
            for i, t in enumerate(THRESHOLDS):
                keep = [p for p in preds if p.score >= t]
                npred[name][i] += len(keep)
                tp[name][i] += match_count(gt, keep)
        if frames % 300 == 0:
            print(f"{frames} frames", flush=True)

    report = {"video": args.video, "every": args.every, "frames": frames, "gt_faces": gt_total,
              "thresholds": THRESHOLDS, "models": {}}
    for name in cfgs:
        rec = tp[name] / max(1, gt_total)
        prec = np.where(npred[name] > 0, tp[name] / np.maximum(1, npred[name]), 1.0)
        order = np.argsort(rec)
        r, p = rec[order], prec[order]
        # Monotone precision envelope, then area under PR over the swept range.
        p_env = np.maximum.accumulate(p[::-1])[::-1]
        ap_val = float(np.trapezoid(p_env, r)) if len(r) > 1 else 0.0

        def recall_at_precision(target: float) -> float | None:
            ok = prec >= target
            return round(float(rec[ok].max()), 4) if ok.any() else None

        def precision_at_recall(target: float) -> float | None:
            ok = rec >= target
            return round(float(prec[ok].max()), 4) if ok.any() else None

        report["models"][name] = {
            "recall": [round(float(x), 4) for x in rec],
            "precision": [round(float(x), 4) for x in prec],
            "ap_swept": round(ap_val, 4),
            "max_recall": round(float(rec.max()), 4),
            "recall_at_p0.5": recall_at_precision(0.5),
            "recall_at_p0.6": recall_at_precision(0.6),
            "recall_at_p0.7": recall_at_precision(0.7),
            "precision_at_r0.95": precision_at_recall(0.95),
            "precision_at_r0.97": precision_at_recall(0.97),
        }
        m = report["models"][name]
        print(f"{name:12s} AP={m['ap_swept']:.4f} maxR={m['max_recall']:.4f} R@P.6={m['recall_at_p0.6']} "
              f"R@P.7={m['recall_at_p0.7']} P@R.95={m['precision_at_r0.95']} P@R.97={m['precision_at_r0.97']}")
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
