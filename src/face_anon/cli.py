from __future__ import annotations

import argparse
import json
import sys

from face_anon.config import load_config
from face_anon.jobs import JobService
from face_anon.pipeline import FaceAnonymizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="face-anon",
        description="Anonymize every visible face in body-camera video, then QC with RetinaFace.",
    )
    parser.add_argument("--config", default=None, help="YAML config path")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download-models", help="Fetch YuNet, SCRFD-10G and RetinaFace ONNX weights")
    p_dl.add_argument("--sample", action="store_true", help="Also download a short public face clip")

    p_proc = sub.add_parser("process", help="Run anonymization + QC on one video (sync)")
    p_proc.add_argument("--input", required=True)
    p_proc.add_argument("--output", required=True)
    p_proc.add_argument("--no-qc", action="store_true")

    p_sub = sub.add_parser("submit", help="Enqueue an upload event (idempotent)")
    p_sub.add_argument("--input", required=True)
    p_sub.add_argument("--upload-id", default=None)
    p_sub.add_argument("--output", default=None)
    p_sub.add_argument("--run", action="store_true", help="Process immediately after submit")

    p_w = sub.add_parser("worker", help="Drain the job queue")
    p_w.add_argument("--once", action="store_true")

    p_st = sub.add_parser("status", help="Print a job record")
    p_st.add_argument("--job-id", required=True)

    p_ev = sub.add_parser("eval", help="Proxy recall + residual-face rate (RetinaFace as stand-in GT)")
    p_ev.add_argument("--input", required=True)
    p_ev.add_argument("--output", required=True)
    p_ev.add_argument("--max-frames", type=int, default=80)

    args = parser.parse_args(argv)
    if args.cmd == "download-models":
        from face_anon.download import download_all

        download_all(with_sample=args.sample)
        return 0

    cfg = load_config(args.config)
    if args.cmd == "process":
        metrics = FaceAnonymizer(cfg).process_video(
            args.input, args.output, run_quality_check=not args.no_qc
        )
        print(json.dumps(metrics, indent=2))
        qc = metrics.get("qc") or {}
        return 0 if qc.get("status", "pass") != "needs_review" else 2

    service = JobService(cfg)
    if args.cmd == "submit":
        job = service.submit(args.input, upload_id=args.upload_id, output_path=args.output)
        payload = {
            "job_id": job.job_id,
            "upload_id": job.upload_id,
            "status": job.status,
            "reused": job.reused,
            "output_path": job.output_path,
        }
        if args.run and not job.reused:
            job = service.run_job(job)
            payload["status"] = job.status
            payload["qc_status"] = job.qc_status
        elif args.run and job.reused:
            payload["note"] = "duplicate event: existing job reused, not reprocessed"
        print(json.dumps(payload, indent=2))
        return 0

    if args.cmd == "worker":
        service.worker_loop(once=args.once)
        return 0

    if args.cmd == "status":
        job = service.store.get(args.job_id)
        if job is None:
            print(f"unknown job: {args.job_id}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "job_id": job.job_id,
                    "upload_id": job.upload_id,
                    "status": job.status,
                    "qc_status": job.qc_status,
                    "input_path": job.input_path,
                    "output_path": job.output_path,
                    "error": job.error,
                    "created_at": job.created_at,
                    "updated_at": job.updated_at,
                    "metrics": json.loads(job.metrics_json) if job.metrics_json else None,
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "eval":
        from face_anon.detect_retinaface import RetinaFaceDetector
        from face_anon.eval_metrics import privacy_eval

        q = cfg["qc"]
        retina = RetinaFaceDetector(
            score_threshold=float(q["score_threshold"]),
            nms_threshold=float(q["nms_threshold"]),
            input_long_side=int(q["input_long_side"]),
        )
        report = privacy_eval(args.input, args.output, retina, max_frames=args.max_frames)
        print(json.dumps(report, indent=2))
        return 0 if report["residual_visible_faces"] == 0 else 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
