from pathlib import Path

from face_anon.config import load_config
from face_anon.jobs import JobService


def _cfg(tmp_path: Path) -> dict:
    cfg = load_config()
    cfg["jobs"]["db_path"] = str(tmp_path / "jobs.sqlite")
    cfg["jobs"]["lock_dir"] = str(tmp_path / "locks")
    return cfg


def _dummy_video(path: Path) -> Path:
    path.write_bytes(b"FAKEVIDEO" + b"\x00" * 2048)
    return path


def test_same_upload_event_is_idempotent(tmp_path: Path):
    svc = JobService(_cfg(tmp_path))
    video = _dummy_video(tmp_path / "cam.mp4")
    a = svc.submit(video, upload_id="evt-1")
    b = svc.submit(video, upload_id="evt-1")
    assert a.job_id == b.job_id
    assert b.reused is True
    assert a.reused is False


def test_same_bytes_different_event_reuses_job(tmp_path: Path):
    svc = JobService(_cfg(tmp_path))
    video = _dummy_video(tmp_path / "cam.mp4")
    a = svc.submit(video, upload_id="evt-A")
    b = svc.submit(video, upload_id="evt-B")
    assert a.job_id == b.job_id
    assert b.reused is True


def test_different_files_are_distinct_jobs(tmp_path: Path):
    svc = JobService(_cfg(tmp_path))
    a = svc.submit(_dummy_video(tmp_path / "a.mp4"), upload_id="1")
    b = svc.submit(_dummy_video(tmp_path / "b.mp4") if False else _write_other(tmp_path), upload_id="2")
    assert a.job_id != b.job_id


def _write_other(tmp_path: Path) -> Path:
    p = tmp_path / "other.mp4"
    p.write_bytes(b"OTHERVID" + b"\x01" * 4096)
    return p
