from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

from face_anon.config import MODELS_DIR, PACKAGE_ROOT

YUNET_URLS = [
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
]
RETINAFACE_URLS = [
    "https://huggingface.co/nakamura196/retinaface-r50-onnx/resolve/main/retinaface_r50.onnx",
]
SCRFD_URLS = [
    "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx",
]
SAMPLE_URLS = [
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female.mp4",
    "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/head-pose-face-detection-female.mp4",
]

YUNET_MIN_BYTES = 80_000
RETINA_MIN_BYTES = 1_000_000
SCRFD_MIN_BYTES = 8_000_000
SAMPLE_MIN_BYTES = 50_000


def _download(urls: list[str], dest: Path, min_bytes: int) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size >= min_bytes:
        print(f"already present: {dest} ({dest.stat().st_size} bytes)")
        return dest
    last_err: Exception | None = None
    tmp = dest.with_suffix(dest.suffix + ".part")
    for url in urls:
        try:
            print(f"downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "face-anon/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f)
            if tmp.stat().st_size < min_bytes:
                raise RuntimeError(f"downloaded file too small: {tmp.stat().st_size}")
            tmp.replace(dest)
            print(f"saved {dest} ({dest.stat().st_size} bytes)")
            return dest
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"  failed: {exc}", file=sys.stderr)
            if tmp.exists():
                tmp.unlink()
    raise RuntimeError(f"could not download {dest.name}: {last_err}")


def download_all(with_sample: bool = False) -> dict[str, Path]:
    out = {
        "yunet": _download(YUNET_URLS, MODELS_DIR / "face_detection_yunet_2023mar.onnx", YUNET_MIN_BYTES),
        "retinaface": _download(RETINAFACE_URLS, MODELS_DIR / "retinaface_r50.onnx", RETINA_MIN_BYTES),
        "scrfd10g": _download(SCRFD_URLS, MODELS_DIR / "det_10g.onnx", SCRFD_MIN_BYTES),
    }
    if with_sample:
        sample = PACKAGE_ROOT / "data" / "samples" / "faces_short.mp4"
        out["sample"] = _download(SAMPLE_URLS, sample, SAMPLE_MIN_BYTES)
    return out
