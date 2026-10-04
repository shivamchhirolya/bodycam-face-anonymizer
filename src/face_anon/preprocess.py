from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def frame_luma_mean(frame_bgr: np.ndarray) -> float:
    # Subsample: a 1/8 grid is enough to decide whether CLAHE is worth it.
    patch = frame_bgr[::8, ::8]
    b = float(patch[..., 0].mean())
    g = float(patch[..., 1].mean())
    r = float(patch[..., 2].mean())
    return 0.114 * b + 0.587 * g + 0.299 * r


def any_tile_dark(frame_bgr: np.ndarray, threshold: float, tiles: int = 4) -> bool:
    """True if any local tile is dark: a bright street plus dark doorway is night."""
    h, w = frame_bgr.shape[:2]
    tiles = max(1, int(tiles))
    th, tw = max(1, h // tiles), max(1, w // tiles)
    for i in range(tiles):
        for j in range(tiles):
            tile = frame_bgr[i * th : (i + 1) * th, j * tw : (j + 1) * tw]
            if tile.size and frame_luma_mean(tile) < threshold:
                return True
    return False


def maybe_clahe(frame_bgr: np.ndarray, mean_luma: float, clip: float, grid: int, threshold: float) -> np.ndarray:
    if mean_luma >= threshold and not any_tile_dark(frame_bgr, threshold):
        return frame_bgr
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def resize_for_detect(frame_bgr: np.ndarray, long_side: int) -> Tuple[np.ndarray, float]:
    """Scale so the long side equals `long_side`: shrink 4K or enlarge 360p.

    Enlarging is what makes 8–12 px hallway faces detectable.
    """
    h, w = frame_bgr.shape[:2]
    long = max(h, w)
    if long_side <= 0 or long == long_side:
        return frame_bgr, 1.0
    scale = long_side / float(long)
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    if new_w % 2:
        new_w += 1
    if new_h % 2:
        new_h += 1
    interp = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=interp)
    return resized, scale
