"""Exact paths and atomic publication for public frozen scene PNG bytes."""

import hashlib
import os
from pathlib import Path
import re
import tempfile

from skill3d.storage.hot_cache import IntegrityError, _flock

SHARED_SCHEMA = "v11-fit-set-offline-input/2.0"
LEGACY_SCHEMA = "v11-fit-set-offline-input/1.0"


def scene_key(scene):
    return hashlib.sha256(str(scene).encode("utf-8")).hexdigest()


def shared_frame_path(root, scene, frame_set_hash, digest):
    if not all(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value)
               for value in (frame_set_hash, digest)):
        raise IntegrityError("shared image requires exact frame and byte digests")
    # The full frame-set identity remains in each QA receipt. A short bucket
    # keeps USTAR directory prefixes within 155 bytes; the full content digest
    # still identifies every PNG, including any colliding bucket prefixes.
    return (Path(root) / "variants" / "shared_frames" / scene_key(scene)
            / frame_set_hash[:16] / (digest + ".png"))


def reject_links(root, path):
    root, path = Path(root), Path(path)
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise IntegrityError("shared frame escaped frozen output root") from None
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise IntegrityError("shared frame path contains a link")
    if path.is_file() and path.stat().st_nlink != 1:
        raise IntegrityError("shared frame has multiple links")


def publish_frame(root, path, data, *, state_io=None):
    """Serialize only host publication; never expose an incomplete final PNG."""
    if state_io is not None:
        reject_links(root, path)
        state_io.write_bytes_once(path, data)
        return
    root, path = Path(root), Path(path)
    lock = root / "variants" / ".shared-frame-locks" / (
        hashlib.sha256(str(path.parent).encode()).hexdigest() + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with _flock(lock):
        reject_links(root, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not path.is_file() or path.read_bytes() != data:
                raise IntegrityError("immutable shared image changed")
            return
        fd, temporary = tempfile.mkstemp(prefix=".frame-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
