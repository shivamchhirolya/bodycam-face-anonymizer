from __future__ import annotations

import hashlib
from pathlib import Path


def content_fingerprint(path: str | Path, chunk: int = 1 << 20) -> str:
    """Cheap collision-resistant id for large body-cam files.

    Full SHA-256 of a multi-hour recording is a sequential disk bottleneck
    when thousands of uploads arrive. Head/mid/tail plus size is enough
    to treat duplicate events as the same asset.
    """
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode("ascii"))
    with open(p, "rb") as f:
        h.update(f.read(chunk))
        if size > 2 * chunk:
            f.seek(size // 2)
            h.update(f.read(chunk))
            f.seek(max(0, size - chunk))
            h.update(f.read(chunk))
    return h.hexdigest()
