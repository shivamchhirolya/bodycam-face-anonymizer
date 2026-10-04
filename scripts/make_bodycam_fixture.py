#!/usr/bin/env python3
"""Turn a clean face clip into a body-cam-like stress fixture.

Applies camera shake, exposure drops, motion blur and a resolution change
so the pipeline can be exercised on the failure modes in the brief.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from face_anon.video_io import FFmpegWriter, probe_video  # noqa: E402


def shake(frame: np.ndarray, rng: np.random.Generator, px: int = 12) -> np.ndarray:
    h, w = frame.shape[:2]
    tx = int(rng.integers(-px, px + 1))
    ty = int(rng.integers(-px, px + 1))
    m = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def darken(frame: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def motion_blur(frame: np.ndarray, k: int = 9) -> np.ndarray:
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(frame, -1, kernel)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    meta = probe_video(args.input)
    cap = cv2.VideoCapture(args.input)
    rng = np.random.default_rng(7)
    out_w, out_h = 1280, 720
    writer = FFmpegWriter(args.output, out_w, out_h, meta["fps"])
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (out_w, out_h))
            frame = shake(frame, rng, px=10)
            # Cycle through the brief's failure modes.
            phase = (idx // max(1, int(meta["fps"]))) % 6
            if phase in (1, 2):
                frame = darken(frame, 0.28)
            if phase == 3:
                frame = motion_blur(frame, 11)
            if phase == 4:
                # Unusual viewpoint proxy: strong tilt.
                h, w = frame.shape[:2]
                m = cv2.getRotationMatrix2D((w / 2, h / 2), 18, 1.05)
                frame = cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
            writer.write(frame)
            idx += 1
    finally:
        cap.release()
        writer.close()
    print(f"wrote {args.output} ({idx} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
