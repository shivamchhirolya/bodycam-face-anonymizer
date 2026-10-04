from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from face_anon.anonymize import anonymize_faces
from face_anon.config import load_config, merge_config
from face_anon.detect_ensemble import detect_on_frame
from face_anon.detect_retinaface import RetinaFaceDetector
from face_anon.factory import build_detector, build_qc_detector, build_tracker, qc_budget
from face_anon.preprocess import any_tile_dark, frame_luma_mean, maybe_clahe
from face_anon.qc import run_qc
from face_anon.types import QCReport
from face_anon.video_io import FFmpegWriter, iter_frames, mux_audio, probe_video


class FaceAnonymizer:
    def __init__(self, cfg: Optional[dict[str, Any]] = None) -> None:
        self.cfg = cfg or load_config()
        self.device = str(self.cfg.get("device", "auto"))
        self.detector, self.detector_name, self.detector_device = build_detector(self.cfg)
        self.tracker, self.tracker_name = build_tracker(self.cfg)
        self._qc_detector: Optional[RetinaFaceDetector] = None

    def qc_detector(self) -> RetinaFaceDetector:
        if self._qc_detector is None:
            self._qc_detector = build_qc_detector(self.cfg)
        return self._qc_detector

    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
        run_quality_check: Optional[bool] = None,
        override: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = merge_config(self.cfg, override)
        if override and "detector" in override:
            self.detector.score_threshold = float(cfg["detector"]["score_threshold"])

        meta = probe_video(input_path)
        metrics = self._encode(input_path, output_path, cfg, meta)

        do_qc = cfg["qc"]["enabled"] if run_quality_check is None else run_quality_check
        if do_qc:
            metrics = self._quality_check(input_path, output_path, cfg, metrics, override)
        metrics.pop("_qc_hints", None)

        sidecar = output_path.with_suffix(".json")
        sidecar.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    def _encode(
        self,
        input_path: Path,
        output_path: Path,
        cfg: dict[str, Any],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        dcfg, pcfg, acfg, adp = cfg["detector"], cfg["preprocess"], cfg["anonymize"], cfg["adaptive"]
        long_side = int(dcfg.get("input_long_side", 640))
        extra_side = int(dcfg.get("extra_long_side", 960))
        extra_every = int(dcfg.get("extra_every", 0))
        strong_every = int(dcfg.get("strong_every", 0))
        min_face = float(dcfg.get("min_face_px", 6))

        self.tracker.reset()
        faces_painted = 0
        frames_with_faces = 0
        empty_streak = 0
        track_loss_frames: list[int] = []
        dark_frames: list[int] = []
        motion_frames: list[int] = []
        prev_small = None
        detect_seconds = 0.0
        detect_calls = 0
        t0 = time.perf_counter()

        tmp_video = output_path.with_suffix(".video.tmp.mp4")
        writer = FFmpegWriter(
            tmp_video,
            meta["width"],
            meta["height"],
            meta["fps"],
            encoder=str(cfg["video"].get("encoder", "auto")),
        )
        try:
            for idx, frame in iter_frames(input_path):
                luma = frame_luma_mean(frame)
                dark = luma < float(pcfg["clahe_when_mean_below"]) or any_tile_dark(
                    frame, float(pcfg["clahe_when_mean_below"])
                )
                if dark and len(dark_frames) < 80:
                    dark_frames.append(idx)

                small = frame[::16, ::16]
                if prev_small is not None and small.shape == prev_small.shape:
                    motion = float((small.astype("int16") - prev_small.astype("int16")).std())
                    if motion > 18 and len(motion_frames) < 80:
                        motion_frames.append(idx)
                prev_small = small

                detect_every = 1
                if empty_streak >= int(adp["empty_streak_before_skip"]) and self.tracker.active_count == 0:
                    detect_every = int(adp["empty_detect_every"])

                detections = []
                if (idx % detect_every == 0) or (idx in motion_frames):
                    det_src = maybe_clahe(
                        frame,
                        luma,
                        float(pcfg["clahe_clip"]),
                        int(pcfg["clahe_grid"]),
                        float(pcfg["clahe_when_mean_below"]),
                    )
                    extra = [extra_side] if extra_every > 0 and idx % extra_every == 0 else []
                    strong = self.qc_detector() if strong_every > 0 and idx % strong_every == 0 else None
                    t_det = time.perf_counter()
                    detections = detect_on_frame(
                        self.detector,
                        det_src,
                        meta["width"],
                        meta["height"],
                        long_side=long_side,
                        extra_long_sides=extra,
                        strong_detector=strong,
                        min_face=min_face,
                    )
                    detect_seconds += time.perf_counter() - t_det
                    detect_calls += 1

                tracks = self.tracker.update(detections)
                if self.tracker.lost_track_ids:
                    track_loss_frames.append(idx)
                if tracks:
                    empty_streak = 0
                    frames_with_faces += 1
                    faces_painted += len(tracks)
                    anonymize_faces(
                        frame,
                        tracks,
                        method=acfg["method"],
                        pad=float(acfg["pad"]),
                        pixel_blocks=int(acfg["pixel_blocks"]),
                        shape=acfg["shape"],
                    )
                else:
                    empty_streak += 1
                writer.write(frame)
        finally:
            writer.close()

        if cfg["video"].get("keep_audio", True):
            mux_audio(tmp_video, input_path, output_path)
            if tmp_video.exists() and tmp_video.resolve() != output_path.resolve():
                tmp_video.unlink()
        else:
            tmp_video.replace(output_path)

        elapsed = time.perf_counter() - t0
        return {
            "input": str(input_path),
            "output": str(output_path),
            "frames": meta["nframes"],
            "src_fps": meta["fps"],
            "width": meta["width"],
            "height": meta["height"],
            "seconds": round(elapsed, 3),
            "throughput_fps": round((meta["nframes"] / elapsed) if elapsed > 0 and meta["nframes"] else 0.0, 2),
            "frames_with_faces": frames_with_faces,
            "faces_painted": faces_painted,
            "detect_long_side": long_side,
            "detector": self.detector_name,
            "detector_device": self.detector_device,
            "tracker": self.tracker_name,
            "detect_calls": detect_calls,
            "detect_ms_per_call": round(1000.0 * detect_seconds / max(1, detect_calls), 2),
            "detect_seconds": round(detect_seconds, 2),
            "pipeline_version": cfg.get("pipeline_version"),
            "_qc_hints": {
                "track_loss_frames": track_loss_frames,
                "dark_frames": dark_frames,
                "motion_frames": motion_frames,
            },
        }

    def _quality_check(
        self,
        input_path: Path,
        output_path: Path,
        cfg: dict[str, Any],
        metrics: dict[str, Any],
        override: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        qcfg, dcfg = cfg["qc"], cfg["detector"]
        hints = metrics.pop("_qc_hints", {})
        extra = list(hints.get("track_loss_frames", [])) + list(hints.get("dark_frames", [])) + list(
            hints.get("motion_frames", [])
        )
        interval, cap = qc_budget(qcfg, self.qc_detector().device)
        t_qc = time.perf_counter()
        qc_report = run_qc(
            str(output_path),
            self.qc_detector(),
            sample_interval_sec=interval,
            max_frames=cap,
            extra_indices=extra,
            fail_fast=bool(qcfg["fail_fast"]),
            min_face_px=int(dcfg.get("min_face_px", 6)),
        )
        metrics["qc"] = qc_report.to_dict()
        metrics["qc_seconds"] = round(time.perf_counter() - t_qc, 2)
        metrics["qc_device"] = self.qc_detector().device

        if (
            qc_report.status == "needs_review"
            and qcfg.get("retry_on_fail", False)
            and not (override or {}).get("_is_retry")
        ):
            return self._retry(input_path, output_path, qcfg, qc_report, metrics)
        return metrics

    def _retry(
        self,
        input_path: Path,
        output_path: Path,
        qcfg: dict[str, Any],
        qc_report: QCReport,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        retry_out = output_path.with_name(output_path.stem + ".retry.mp4")
        retry_metrics = self.process_video(
            input_path,
            retry_out,
            run_quality_check=True,
            override={
                "_is_retry": True,
                "detector": {
                    "score_threshold": float(qcfg["retry_score_threshold"]),
                    "input_long_side": int(qcfg["retry_input_long_side"]),
                    "extra_long_side": int(qcfg["retry_input_long_side"]),
                    "extra_every": 1,
                    "strong_every": int(qcfg.get("retry_strong_every", 0)),
                    "min_face_px": 6,
                },
                "adaptive": {"empty_streak_before_skip": 10**9, "empty_detect_every": 1},
            },
        )
        retry_qc = retry_metrics.get("qc", {})
        retry_sidecar = retry_out.with_suffix(".json")
        if retry_qc.get("leak_count", 1) <= qc_report.leak_count:
            retry_out.replace(output_path)
            if retry_sidecar.exists():
                retry_sidecar.unlink()
            retry_metrics["retried"] = True
            retry_metrics["first_pass_qc"] = qc_report.to_dict()
            return retry_metrics
        if retry_out.exists():
            retry_out.unlink()
        if retry_sidecar.exists():
            retry_sidecar.unlink()
        metrics["retried"] = True
        metrics["retry_rejected"] = True
        return metrics


def process_file(input_path: str, output_path: str, config_path: Optional[str] = None) -> dict[str, Any]:
    return FaceAnonymizer(load_config(config_path)).process_video(input_path, output_path)
