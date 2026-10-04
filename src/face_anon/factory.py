"""Build detectors and trackers from config."""

from __future__ import annotations

from typing import Any

from face_anon.detect_retinaface import RetinaFaceDetector
from face_anon.detect_scrfd import SCRFDDetector
from face_anon.detect_yunet import YuNetDetector
from face_anon.runtime import onnx_providers
from face_anon.track import ByteTracker, IoUTracker


def resolve_detector_name(cfg: dict[str, Any]) -> str:
    name = str(cfg.get("detector", {}).get("name", "yunet")).lower()
    device = str(cfg.get("device", "auto"))
    if name == "auto":
        cuda = onnx_providers(device)[0] == "CUDAExecutionProvider"
        return "scrfd" if cuda else "yunet"
    if name not in {"yunet", "scrfd"}:
        raise ValueError(f"unknown detector.name: {name}")
    return name


def build_detector(cfg: dict[str, Any]):
    dcfg = cfg["detector"]
    name = resolve_detector_name(cfg)
    device = str(cfg.get("device", "auto"))
    if name == "scrfd":
        det = SCRFDDetector(
            score_threshold=float(dcfg["score_threshold"]),
            nms_threshold=float(dcfg["nms_threshold"]),
            input_long_side=int(dcfg.get("input_long_side", 640)),
            device=device,
        )
        return det, name, det.device
    det = YuNetDetector(
        score_threshold=float(dcfg["score_threshold"]),
        nms_threshold=float(dcfg["nms_threshold"]),
    )
    return det, name, "cpu"


def build_tracker(cfg: dict[str, Any]):
    tcfg = cfg["tracker"]
    name = str(tcfg.get("name", "bytetrack")).lower()
    common = dict(
        iou_match=float(tcfg["iou_match"]),
        center_match_frac=float(tcfg["center_match_frac"]),
        max_age=int(tcfg["max_age"]),
        min_hits=int(tcfg["min_hits"]),
        smooth=float(tcfg["smooth"]),
    )
    if name == "bytetrack":
        tracker = ByteTracker(
            high_thresh=float(tcfg.get("high_thresh", 0.28)),
            low_thresh=float(tcfg.get("low_thresh", 0.18)),
            **common,
        )
        return tracker, name
    if name == "iou":
        return IoUTracker(**common), name
    raise ValueError(f"unknown tracker.name: {name}")


def build_qc_detector(cfg: dict[str, Any]) -> RetinaFaceDetector:
    q = cfg["qc"]
    return RetinaFaceDetector(
        score_threshold=float(q["score_threshold"]),
        nms_threshold=float(q["nms_threshold"]),
        input_long_side=int(q["input_long_side"]),
        device=str(cfg.get("device", "auto")),
    )


def qc_budget(qcfg: dict[str, Any], device: str) -> tuple[float, int]:
    on_cpu = device == "cpu"
    interval = float(
        qcfg.get("cpu_sample_interval_sec", qcfg["sample_interval_sec"]) if on_cpu else qcfg["sample_interval_sec"]
    )
    cap = int(qcfg.get("cpu_max_frames", qcfg["max_frames"]) if on_cpu else qcfg["max_frames"])
    return interval, cap
