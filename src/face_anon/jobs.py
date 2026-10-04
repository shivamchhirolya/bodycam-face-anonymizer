from __future__ import annotations

import fcntl
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from face_anon.config import PACKAGE_ROOT, config_hash, load_config, resolve_path
from face_anon.hashutil import content_fingerprint
from face_anon.pipeline import FaceAnonymizer
from face_anon.types import JobRecord

TERMINAL = {"completed", "needs_review", "failed"}
IN_FLIGHT = {"queued", "processing"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    upload_id TEXT NOT NULL,
                    content_fp TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    qc_status TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metrics_json TEXT,
                    UNIQUE(upload_id, config_hash)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_fp ON jobs(content_fp, config_hash, status)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def get_by_upload(self, upload_id: str, config_hash_val: str) -> Optional[JobRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE upload_id=? AND config_hash=?",
                (upload_id, config_hash_val),
            ).fetchone()
        return _row_to_job(row) if row else None

    def get_by_fingerprint(self, content_fp: str, config_hash_val: str) -> Optional[JobRecord]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE content_fp=? AND config_hash=?
                  AND status IN ('queued','processing','completed','needs_review')
                ORDER BY created_at DESC LIMIT 1
                """,
                (content_fp, config_hash_val),
            ).fetchone()
        return _row_to_job(row) if row else None

    def insert(self, job: JobRecord) -> JobRecord:
        now = _utc_now()
        job.created_at = job.created_at or now
        job.updated_at = now
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, upload_id, content_fp, config_hash, input_path,
                    output_path, status, qc_status, error, created_at, updated_at, metrics_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job.job_id,
                    job.upload_id,
                    job.content_fp,
                    job.config_hash,
                    job.input_path,
                    job.output_path,
                    job.status,
                    job.qc_status,
                    job.error,
                    job.created_at,
                    job.updated_at,
                    job.metrics_json,
                ),
            )
        return job

    def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = _utc_now()
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values()) + [job_id]
        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {cols} WHERE job_id=?", vals)

    def claim_next(self) -> Optional[JobRecord]:
        """Atomically move one queued job to processing."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if not row:
                return None
            conn.execute(
                "UPDATE jobs SET status='processing', updated_at=? WHERE job_id=? AND status='queued'",
                (_utc_now(), row["job_id"]),
            )
            if conn.total_changes == 0:
                return None
        job = _row_to_job(row)
        job.status = "processing"
        return job


def _row_to_job(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        job_id=row["job_id"],
        upload_id=row["upload_id"],
        content_fp=row["content_fp"],
        config_hash=row["config_hash"],
        input_path=row["input_path"],
        output_path=row["output_path"],
        status=row["status"],
        qc_status=row["qc_status"],
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        metrics_json=row["metrics_json"],
    )


@contextmanager
def interprocess_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


class JobService:
    """Async job façade with upload-event idempotency.

    Duplicate events are collapsed by (upload_id, config_hash) first, then
    by content fingerprint so a retried webhook with a new id still
    returns the existing work instead of encoding the same hour twice.
    """

    def __init__(self, cfg: Optional[dict[str, Any]] = None) -> None:
        self.cfg = cfg or load_config()
        self.store = JobStore(resolve_path(self.cfg["jobs"]["db_path"]))
        self.lock_dir = resolve_path(self.cfg["jobs"]["lock_dir"])
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.chash = config_hash(self.cfg)

    def submit(
        self,
        input_path: str | Path,
        upload_id: Optional[str] = None,
        output_path: Optional[str | Path] = None,
    ) -> JobRecord:
        input_path = Path(input_path).resolve()
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        upload_id = upload_id or f"upload-{content_fingerprint(input_path)[:16]}"
        lock_path = self.lock_dir / f"{upload_id}.lock"
        with interprocess_lock(lock_path):
            existing = self.store.get_by_upload(upload_id, self.chash)
            if existing:
                existing.reused = True
                return existing
            fp = content_fingerprint(input_path)
            existing = self.store.get_by_fingerprint(fp, self.chash)
            if existing and existing.status in IN_FLIGHT | TERMINAL - {"failed"}:
                existing.reused = True
                return existing
            job_id = uuid.uuid4().hex
            if output_path is None:
                output_path = PACKAGE_ROOT / "outputs" / job_id / "anonymized.mp4"
            output_path = Path(output_path)
            job = JobRecord(
                job_id=job_id,
                upload_id=upload_id,
                content_fp=fp,
                config_hash=self.chash,
                input_path=str(input_path),
                output_path=str(output_path),
                status="queued",
            )
            self.store.insert(job)
            return job

    def run_job(self, job: JobRecord) -> JobRecord:
        self.store.update(job.job_id, status="processing")
        try:
            anon = FaceAnonymizer(self.cfg)
            metrics = anon.process_video(job.input_path, job.output_path)
            qc_status = (metrics.get("qc") or {}).get("status")
            status = "needs_review" if qc_status == "needs_review" else "completed"
            self.store.update(
                job.job_id,
                status=status,
                qc_status=qc_status,
                metrics_json=json.dumps(metrics),
                error=None,
            )
            job.status = status
            job.qc_status = qc_status
            job.metrics_json = json.dumps(metrics)
            return job
        except Exception as exc:  # persist, then re-raise for worker logs
            self.store.update(job.job_id, status="failed", error=str(exc))
            job.status = "failed"
            job.error = str(exc)
            raise

    def worker_loop(self, once: bool = False, idle_sleep: float = 0.5) -> None:
        while True:
            job = self.store.claim_next()
            if job is None:
                if once:
                    return
                time.sleep(idle_sleep)
                continue
            try:
                self.run_job(job)
            except Exception:
                if once:
                    return
                continue
            if once:
                return
