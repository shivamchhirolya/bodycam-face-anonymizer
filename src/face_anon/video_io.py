from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


def probe_video(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return {"fps": fps, "width": width, "height": height, "nframes": nframes}


def iter_frames(path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def has_audio(path: str | Path) -> bool:
    if not shutil.which("ffprobe"):
        return False
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=20)
        return bool(out.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def pick_encoder(requested: str = "auto") -> list[str]:
    if requested and requested != "auto":
        if requested == "nvenc":
            return ["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "20", "-b:v", "0"]
        if requested == "x264":
            return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
        return ["-c:v", requested]

    # Prefer NVENC: encode is usually the bottleneck once YuNet is this cheap.
    try:
        encs = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.STDOUT, timeout=10
        ).decode("utf-8", errors="ignore")
        if "h264_nvenc" in encs:
            return ["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "20", "-b:v", "0"]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]


class FFmpegWriter:
    """Raw-BGR stdin → H.264. Avoids OpenCV's slow/fragile VideoWriter."""

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
        encoder: str = "auto",
    ) -> None:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is required to write output video")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            *pick_encoder(encoder),
            "-pix_fmt",
            "yuv420p",
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if self.proc.stdin is None:
            raise RuntimeError("ffmpeg stdin closed")
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg encode failed with code {rc}")


def mux_audio(video_no_audio: str | Path, source: str | Path, dest: str | Path) -> Path:
    dest = Path(dest)
    if not has_audio(source):
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_no_audio),
            "-c:v",
            "copy",
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            str(dest),
        ]
        subprocess.check_call(cmd)
        return dest
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_no_audio),
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-map_metadata",
        "-1",
        "-shortest",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    subprocess.check_call(cmd)
    return dest
