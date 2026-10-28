"""Bounded local storage and durable, content-addressed archive delivery.

This module has no inference or dataset dependencies. All byte counts use decimal
GB. Producers must write through ``Reservation.open`` (or preallocate a sufficient
budget before using a trusted external materializer). Published payloads are
immutable; readers hold a lease and writers use an exclusive lease. Scene identity
validation belongs to storage.relocation and can be supplied as a publish hook.

ACKs attest durability, not authenticity: the transfer callable must represent a
trusted receiver/transport. A digest received from an untrusted peer is not an ACK.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import threading
import time
from typing import Callable, Iterator, Mapping
import uuid

GB = 1_000_000_000
DEFAULT_QUOTA_BYTES = 35 * GB
MAX_QUOTA_BYTES = 40 * GB
MIN_FREE_BYTES = 25 * GB
_HEADROOM = 64 * 1024
_BLOCK = 1024 * 1024
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_MEMBERS = 100_000
_MANIFEST_SCHEMA = "h3d-archive-v1"
_ACK_SCHEMA = "h3d-archive-ack-v1"


class StorageError(ValueError):
    """The storage protocol cannot safely complete the requested operation."""


class QuotaExceeded(StorageError):
    """The shared quota or filesystem free-space floor would be violated."""


class IntegrityError(StorageError):
    """An archive, payload, manifest, or ACK fails verification."""


class IdentityConflict(StorageError):
    """A job/shard/attempt was already bound to different archived bytes."""


class EntryBusy(StorageError):
    """An active lease or a retained slot prevents removal."""


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        c not in "0123456789abcdef" for c in value
    ):
        raise IntegrityError("invalid SHA256")
    return value


def _count(value: object) -> int:
    if type(value) is not int or value < 0:
        raise StorageError("byte counts must be nonnegative integers")
    return value


def _name(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 512 or any(
        ord(c) < 32 or ord(c) == 127 for c in value
    ):
        raise StorageError("identity/key must be a nonempty printable string")
    return value


def _ref(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or any(
        ord(c) < 32 or ord(c) == 127 for c in value
    ):
        raise IntegrityError("unsafe archive member path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(
        p in (".", "..") or ":" in p or p.startswith("~")
        for p in value.split("/")
    ):
        raise IntegrityError("escaping or noncanonical archive member path")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError("duplicate JSON field")
        result[key] = value
    return result


def _read_json(path: Path) -> dict:
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise IntegrityError("oversized JSON document")
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_pairs)
        if not isinstance(value, dict):
            raise IntegrityError("JSON document must be an object")
        return value
    except (ValueError, UnicodeError) as exc:
        raise IntegrityError("unreadable JSON document") from exc


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _charged(info) -> int:
    return max(info.st_size, getattr(info, "st_blocks", 0) * 512)


def _allocation_unit(root: Path) -> int:
    return max(512, os.statvfs(root).f_frsize)


def _rounded(size: int, unit: int) -> int:
    return ((size + unit - 1) // unit) * unit


def _tree_bytes(root: Path, *, logical: bool = False) -> int:
    if not root.exists():
        return 0
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise IntegrityError("storage tree root must be a real directory")
    total = 0 if logical else _charged(info)
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in (*dirs, *files):
            info = (Path(base) / name).lstat()
            if stat.S_ISLNK(info.st_mode) or not (
                stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
            ):
                raise IntegrityError("links and special files are forbidden")
            if stat.S_ISREG(info.st_mode):
                total += info.st_size if logical else _charged(info)
            elif not logical:
                total += _charged(info)
    return total


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(root: Path) -> None:
    _tree_bytes(root)
    for base, _, files in os.walk(root, topdown=False):
        for name in files:
            fd = os.open(Path(base) / name, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        _fsync_dir(Path(base))


def _atomic_json(path: Path, value: dict) -> None:
    temp = path.parent / (".state-" + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        _fsync_dir(path.parent)
    finally:
        temp.unlink(missing_ok=True)


def _root(path: str | Path) -> Path:
    path = Path(path).absolute()
    if path.is_symlink():
        raise IntegrityError("storage root cannot be a symlink")
    path.mkdir(parents=True, exist_ok=True)
    path = path.resolve()
    if path == path.parent:
        raise StorageError("filesystem root cannot be a storage root")
    return path


@contextmanager
def _flock(path: Path, *, exclusive: bool = True, blocking: bool = True):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(fd, mode | (0 if blocking else fcntl.LOCK_NB))
        yield fd
    finally:
        os.close(fd)


def _inventory(root: Path) -> list[dict]:
    _tree_bytes(root)
    members = []
    for path in sorted(root.rglob("*")):
        ref = _ref(path.relative_to(root).as_posix())
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            members.append({"path": ref, "kind": "directory",
                            "size": 0, "sha256": None})
        elif stat.S_ISREG(info.st_mode):
            members.append({"path": ref, "kind": "file",
                            "size": info.st_size, "sha256": _file_hash(path)})
        else:
            raise IntegrityError("archive source must contain only files/directories")
        if len(members) > _MAX_MEMBERS:
            raise IntegrityError("too many archive members")
    # Path orders components; the wire protocol orders whole POSIX strings.
    # A sibling ``buffer.lock`` sorts before ``buffer/stream.json`` on the wire.
    return sorted(members, key=lambda member: member["path"])


@dataclass(frozen=True)
class ArchiveIdentity:
    job: str
    shard: str
    attempt: int
    sha256: str

    def __post_init__(self):
        _name(self.job)
        _name(self.shard)
        _count(self.attempt)
        _sha(self.sha256)

    @property
    def slot(self) -> str:
        return _digest([self.job, self.shard, self.attempt])

    def as_dict(self) -> dict:
        return {"job": self.job, "shard": self.shard, "attempt": self.attempt,
                "sha256": self.sha256}


@dataclass(frozen=True)
class ArchiveAck:
    identity: ArchiveIdentity
    manifest_sha256: str

    def __post_init__(self):
        _sha(self.manifest_sha256)

    def as_dict(self) -> dict:
        return {"schema_version": _ACK_SCHEMA, **self.identity.as_dict(),
                "manifest_sha256": self.manifest_sha256}

    @classmethod
    def parse(cls, value: ArchiveAck | Mapping) -> ArchiveAck:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version", "job", "shard", "attempt", "sha256", "manifest_sha256"
        } or value["schema_version"] != _ACK_SCHEMA:
            raise IntegrityError("invalid ACK schema")
        return cls(ArchiveIdentity(value["job"], value["shard"], value["attempt"],
                                   value["sha256"]), value["manifest_sha256"])


def _manifest(value: Mapping) -> dict:
    if not isinstance(value, Mapping):
        raise IntegrityError("manifest must be an object")
    try:
        encoded = _json_bytes(dict(value))
        if len(encoded) > _MAX_MANIFEST_BYTES:
            raise IntegrityError("manifest is too large")
        data = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise IntegrityError("manifest is not canonical JSON") from exc
    if set(data) != {
        "schema_version", "job", "shard", "attempt", "sha256", "archive_bytes",
        "total_bytes", "members", "members_sha256", "manifest_sha256"
    } or data["schema_version"] != _MANIFEST_SCHEMA:
        raise IntegrityError("invalid manifest schema")
    ArchiveIdentity(data["job"], data["shard"], data["attempt"], data["sha256"])
    _count(data["archive_bytes"])
    _count(data["total_bytes"])
    members = data["members"]
    if not isinstance(members, list) or len(members) > _MAX_MEMBERS:
        raise IntegrityError("invalid member list")
    seen = {}
    for member in members:
        if not isinstance(member, dict) or set(member) != {
            "path", "kind", "size", "sha256"
        }:
            raise IntegrityError("invalid member record")
        ref = _ref(member["path"])
        if ref in seen:
            raise IntegrityError("duplicate manifest member")
        _count(member["size"])
        if member["kind"] == "file":
            _sha(member["sha256"])
        elif member["kind"] != "directory" or member["size"] or member["sha256"] is not None:
            raise IntegrityError("invalid member kind")
        seen[ref] = member["kind"]
    for ref in seen:
        for parent in PurePosixPath(ref).parents:
            if str(parent) != "." and seen.get(str(parent)) != "directory":
                raise IntegrityError("missing or non-directory member parent")
    if data["total_bytes"] != sum(m["size"] for m in members):
        raise IntegrityError("manifest total size mismatch")
    if members != sorted(members, key=lambda m: m["path"]):
        raise IntegrityError("member list must be in canonical path order")
    if _sha(data["members_sha256"]) != _digest(members):
        raise IntegrityError("member-list hash mismatch")
    body = {k: v for k, v in data.items() if k != "manifest_sha256"}
    if _sha(data["manifest_sha256"]) != _digest(body):
        raise IntegrityError("manifest hash mismatch")
    return data


def _identity(manifest: dict) -> ArchiveIdentity:
    return ArchiveIdentity(manifest["job"], manifest["shard"],
                           manifest["attempt"], manifest["sha256"])


def _check_ack(value: ArchiveAck | Mapping, manifest: dict) -> ArchiveAck:
    ack = ArchiveAck.parse(value)
    if ack.identity != _identity(manifest) or (
        ack.manifest_sha256 != manifest["manifest_sha256"]
    ):
        raise IdentityConflict("ACK does not correspond to this exact archive")
    return ack


def _extract(
    archive: Path, target: Path | None, manifest: dict,
    before_write: Callable[[int], object] | None = None,
) -> None:
    expected = {m["path"]: m for m in manifest["members"]}
    seen = set()
    try:
        # Parse fixed USTAR headers directly. Never let tarfile consume an
        # attacker-sized PAX/GNU extension into memory or follow an archive link.
        with gzip.open(archive, "rb") as stream:
            while True:
                header = stream.read(tarfile.BLOCKSIZE)
                if len(header) != tarfile.BLOCKSIZE:
                    raise IntegrityError("truncated tar header")
                if header == bytes(tarfile.BLOCKSIZE):
                    if stream.read(tarfile.BLOCKSIZE) != bytes(tarfile.BLOCKSIZE):
                        raise IntegrityError("tar requires two end-of-archive blocks")
                    tail = stream.read(tarfile.RECORDSIZE + 1)
                    if len(tail) > tarfile.RECORDSIZE or any(tail):
                        raise IntegrityError("nonzero/excess trailing tar data")
                    break
                member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
                ref = _ref(member.name)
                if ref in seen or ref not in expected:
                    raise IntegrityError("duplicate or undeclared archive member")
                seen.add(ref)
                record = expected[ref]
                if member.type == tarfile.DIRTYPE:
                    if record["kind"] != "directory" or member.size != 0:
                        raise IntegrityError("directory member mismatch")
                    if target is not None:
                        (target / ref).mkdir(parents=True, exist_ok=True)
                elif member.type in (tarfile.REGTYPE, tarfile.AREGTYPE):
                    if record["kind"] != "file" or member.size != record["size"]:
                        raise IntegrityError("file member size/kind mismatch")
                    path = target / ref if target is not None else None
                    if path is not None:
                        path.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    remaining = record["size"]
                    with path.open("xb") if path is not None else nullcontext(None) as dest:
                        while remaining:
                            block = stream.read(min(_BLOCK, remaining))
                            if not block:
                                raise IntegrityError("truncated archive member")
                            if dest is not None:
                                if before_write is not None:
                                    before_write(len(block))
                                dest.write(block)
                            digest.update(block)
                            remaining -= len(block)
                    if digest.hexdigest() != record["sha256"]:
                        raise IntegrityError("archive member SHA256 mismatch")
                    padding = (-member.size) % tarfile.BLOCKSIZE
                    if stream.read(padding) != bytes(padding):
                        raise IntegrityError("invalid tar file padding")
                else:
                    raise IntegrityError("links, extensions and special members are forbidden")
        if seen != set(expected):
            raise IntegrityError("missing archive members")
    except (OSError, EOFError, UnicodeError, tarfile.TarError) as exc:
        raise IntegrityError("corrupt or unsafe tar.gz") from exc
    if target is not None and _inventory(target) != manifest["members"]:
        raise IntegrityError("extracted member inventory mismatch")


class ArchiveReceiver:
    """Titan-side local protocol endpoint; callable with (archive_path, manifest).

    Transfer is a real bounded copy into a temporary receive directory. Files are
    manually extracted, verified, fsynced, then published with their ACK in one
    directory rename. Retries return the same ACK; a different hash for the same
    job/shard/attempt is rejected. The caller owns network authentication.
    """

    def __init__(
        self, root: str | Path, *, max_archive_bytes: int = MAX_QUOTA_BYTES,
        max_extract_bytes: int = MAX_QUOTA_BYTES,
        min_free_bytes: int = MIN_FREE_BYTES, disk_usage: Callable = shutil.disk_usage,
    ):
        self.root = _root(root)
        self.max_archive_bytes = _count(max_archive_bytes)
        self.max_extract_bytes = _count(max_extract_bytes)
        self.min_free_bytes = _count(min_free_bytes)
        if self.min_free_bytes < MIN_FREE_BYTES:
            raise StorageError("filesystem free-space floor must be at least 25 GB")
        self.disk_usage = disk_usage
        for name in ("objects", ".incoming"):
            (self.root / name).mkdir(exist_ok=True)
        self.recover()

    def recover(self) -> None:
        with _flock(self.root / ".receiver.lock"):
            for path in (self.root / ".incoming").iterdir():
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            _fsync_dir(self.root / ".incoming")

    def published_path(self, identity: ArchiveIdentity) -> Path:
        return self.root / "objects" / identity.slot / "payload"

    def __call__(self, archive: str | Path, manifest: Mapping) -> ArchiveAck:
        return self.receive(archive, manifest)

    def receive(self, archive: str | Path, manifest: Mapping) -> ArchiveAck:
        data = _manifest(manifest)
        identity = _identity(data)
        final = self.root / "objects" / identity.slot
        with _flock(self.root / ".receiver.lock"):
            if final.exists():
                saved = _manifest(_read_json(final / "manifest.json"))
                if saved != data:
                    raise IdentityConflict("job/shard/attempt already published differently")
                if _inventory(final / "payload") != data["members"]:
                    raise IntegrityError("previously published payload is corrupt")
                return _check_ack(_read_json(final / "ack.json"), data)
            required = data["archive_bytes"] + data["total_bytes"] + 2 * len(
                _json_bytes(data)
            ) + _HEADROOM
            if data["archive_bytes"] > self.max_archive_bytes or (
                data["total_bytes"] > self.max_extract_bytes
            ) or self.disk_usage(self.root).free < self.min_free_bytes + required:
                raise QuotaExceeded("receiver cannot reserve copy and extraction space")
            stage = self.root / ".incoming" / uuid.uuid4().hex
            stage.mkdir()
            try:
                copied = stage / "archive.tar.gz"
                digest = hashlib.sha256()
                remaining = data["archive_bytes"]
                with Path(archive).open("rb") as source, copied.open("xb") as dest:
                    while remaining:
                        block = source.read(min(_BLOCK, remaining))
                        if not block:
                            raise IntegrityError("truncated archive transfer")
                        if self.disk_usage(self.root).free < self.min_free_bytes + len(block):
                            raise QuotaExceeded("receiver free-space floor reached")
                        dest.write(block)
                        digest.update(block)
                        remaining -= len(block)
                    if source.read(1):
                        raise IntegrityError("archive transfer exceeds declared size")
                    dest.flush()
                    os.fsync(dest.fileno())
                if digest.hexdigest() != identity.sha256:
                    raise IntegrityError("archive total SHA256 mismatch")
                payload = stage / "payload"
                payload.mkdir()
                def check_free(size):
                    if self.disk_usage(self.root).free < self.min_free_bytes + size:
                        raise QuotaExceeded("receiver extraction reached free-space floor")

                _extract(copied, payload, data, check_free)
                ack = ArchiveAck(identity, data["manifest_sha256"])
                _atomic_json(stage / "manifest.json", data)
                _atomic_json(stage / "ack.json", ack.as_dict())
                copied.unlink()
                _fsync_tree(stage)
                os.replace(stage, final)
                _fsync_dir(final.parent)
                return ack
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
                    _fsync_dir(stage.parent)


@dataclass(frozen=True)
class ArchivePackage:
    path: Path
    manifest: dict

    @property
    def identity(self) -> ArchiveIdentity:
        return _identity(self.manifest)


class _LimitedWriter:
    def __init__(self, reservation: Reservation, stream):
        self.reservation = reservation
        self.stream = stream

    def write(self, data) -> int:
        with self.reservation.cache._locked() as state:
            self.reservation._check(state)
            if _tree_bytes(self.reservation.path, logical=True) + len(data) > self.reservation.bytes:
                raise QuotaExceeded("write exceeds reservation")
            self.reservation.cache._fits(state, 0)
            unit = _allocation_unit(self.reservation.path)
            info = os.fstat(self.stream.fileno())
            growth = max(0, max(_rounded(info.st_size + len(data), unit),
                                _charged(info)) - _charged(info))
            if _tree_bytes(self.reservation.path) + growth > self.reservation.charged_bytes:
                raise QuotaExceeded("allocated write exceeds reservation envelope")
            if self.reservation.cache.disk_usage(self.reservation.cache.root).free < (
                self.reservation.cache.min_free_bytes + growth
            ):
                raise QuotaExceeded("write would cross filesystem free-space floor")
            return self.stream.write(data)

    def flush(self):
        return self.stream.flush()

    def tell(self):
        return self.stream.tell()

    def fileno(self):
        return self.stream.fileno()


class _ArchiveWriter:
    """Internal compressor only: a fully reserved file and a strict byte cap."""

    def __init__(self, reservation, stream):
        self.reservation, self.stream = reservation, stream
        self.written = 0

    def write(self, data):
        if self.reservation.closed:
            raise StorageError("archive reservation is closed")
        if self.written + len(data) > self.reservation.bytes:
            raise QuotaExceeded("archive exceeds its compression reservation")
        if self.reservation.cache.disk_usage(self.reservation.cache.root).free < (
            self.reservation.cache.min_free_bytes
        ):
            raise QuotaExceeded("archive crosses the free-space floor")
        count = self.stream.write(data)
        self.written += count
        return count

    def flush(self):
        return self.stream.flush()

    def tell(self):
        return self.written


class Reservation:
    """A cross-process byte reservation and crash-recoverable staging tree."""

    def __init__(self, cache: HotCache, token: str, size: int, charged: int, lock_fd: int):
        self.cache, self.token, self.bytes = cache, token, size
        self.charged_bytes = charged
        self.path = cache.root / "work" / token
        self._fd = lock_fd
        self.closed = False

    def _check(self, state: dict):
        if self.closed or self.token not in state["reservations"]:
            raise StorageError("reservation is closed")

    @contextmanager
    def open(self, relative: str) -> Iterator[_LimitedWriter]:
        ref = _ref(relative)
        with self.cache._locked() as state:
            self._check(state)
            path = self.path / ref
            path.parent.mkdir(parents=True, exist_ok=True)
            _tree_bytes(self.path)
            self.cache._fits(state, 0)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb", buffering=0) as stream:
            yield _LimitedWriter(self, stream)
            stream.flush()
            os.fsync(stream.fileno())

    @contextmanager
    def _archive_stream(self):
        """Reserve the entire worst-case archive before trusted compression.

        Preallocation counts the whole envelope immediately, avoiding repeated
        whole-cache walks for each small gzip block. The generic writer remains
        responsible for arbitrary producers; this stream is private to archiving.
        """
        with self.cache._locked() as state:
            self._check(state)
            self.cache._fits(state, 0)
            fd = os.open(self.path / "archive.tar.gz",
                         os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb", buffering=0) as stream:
            allocate = getattr(os, "posix_fallocate", None)
            if allocate is not None and self.bytes:
                allocate(stream.fileno(), 0, self.bytes)
            else:
                stream.truncate(self.bytes)
            writer = _ArchiveWriter(self, stream)
            yield writer
            stream.truncate(writer.written)
            stream.flush()
            os.fsync(stream.fileno())
            with self.cache._locked() as state:
                self._check(state)
                if _tree_bytes(self.path) > self.charged_bytes:
                    raise QuotaExceeded("archive allocated more than its envelope")
                self.cache._fits(state, 0)

    def write_bytes(self, relative: str, data: bytes):
        with self.open(relative) as stream:
            for start in range(0, len(data), _BLOCK):
                stream.write(data[start:start + _BLOCK])

    def commit(
        self, key: str, *, kind: str = "input", payload: str = "payload",
        slots: tuple[str, ...] = (), validator: Callable[[Path], object] | None = None,
        workspace_bytes: int | None = None,
    ) -> Path:
        return self.cache._publish(self, key, kind, payload, slots, validator,
                                   workspace_bytes)

    def close(self):
        if self.closed:
            return
        try:
            with self.cache._locked() as state:
                # A publish intent must survive a crash/error between its durable
                # state write and rename; recovery completes it, not inference.
                referenced = any(e.get("staging", "").startswith(self.token + "/")
                                 for e in state["entries"].values())
                referenced |= any(
                    (e.get("archive_staging") or "").startswith(self.token + "/")
                    for e in state["entries"].values()
                )
                if not referenced:
                    shutil.rmtree(self.path, ignore_errors=False)
                    state["reservations"].pop(self.token, None)
                    self.cache._save(state)
        finally:
            self.closed = True
            os.close(self._fd)

    def __enter__(self) -> Reservation:
        return self

    def __exit__(self, *exc):
        self.close()


@dataclass
class WorkspaceLease:
    """Trusted mutable IO budget, valid during an exclusive workspace lease.

    atomic_writes is a fixed-envelope contract: only WorkspaceIO may mutate the
    tree. Its touched-path accounting replaces scans, not quota admission.
    """

    cache: HotCache
    key: str
    path: Path
    bytes: int
    charged_bytes: int
    active: bool = True
    atomic_writes: bool = False
    _logical_bytes: int | None = field(default=None, init=False, repr=False)
    _allocated_bytes: int | None = field(default=None, init=False, repr=False)
    _io_lock: object = field(default_factory=threading.RLock, init=False, repr=False)
    _accounting_ok: bool = field(default=True, init=False, repr=False)
    _durability_ok: bool = field(default=True, init=False, repr=False)
    _writers: int = field(default=0, init=False, repr=False)
    _atomic_temporaries: set = field(default_factory=set, init=False, repr=False)

    def check_growth(self, *, logical_bytes: int = 0, allocation_bytes: int = 0):
        logical_bytes, allocation_bytes = _count(logical_bytes), _count(allocation_bytes)
        if not self.atomic_writes:
            return self._check_full(logical_bytes, allocation_bytes)
        with self._io_lock:
            if not self.active:
                raise StorageError("workspace lease is closed")
            if not self._accounting_ok or self._logical_bytes is None:
                raise IntegrityError("atomic workspace accounting is unavailable")
            if self._logical_bytes + logical_bytes > self.bytes or (
                self._allocated_bytes + allocation_bytes > self.charged_bytes
            ):
                raise QuotaExceeded("workspace write exceeds its persistent growth envelope")
            # The global quota already charges the fixed ceiling. Other cache
            # operations retain their own admission; check actual free space here.
            if self.cache.disk_usage(self.cache.root).free < (
                self.cache.min_free_bytes + allocation_bytes
            ):
                raise QuotaExceeded("workspace write crosses filesystem free-space floor")

    def _check_full(self, logical_bytes=0, allocation_bytes=0):
        if not self.active:
            raise StorageError("workspace lease is closed")
        with self.cache._locked() as state:
            row = state["entries"].get(self.key)
            if row is None or row["kind"] != "workspace" or not row["ready"]:
                raise StorageError("workspace is not published")
            logical = _tree_bytes(self.path, logical=True)
            allocated = _tree_bytes(self.path)
            if logical + logical_bytes > self.bytes or (
                allocated + allocation_bytes > self.charged_bytes
            ):
                raise QuotaExceeded("workspace write exceeds its persistent growth envelope")
            self.cache._fits(state, 0)
            if self.cache.disk_usage(self.cache.root).free < (
                self.cache.min_free_bytes + allocation_bytes
            ):
                raise QuotaExceeded("workspace write crosses filesystem free-space floor")
            self._logical_bytes, self._allocated_bytes = logical, allocated

    @contextmanager
    def _accounted(self, paths, *, logical_bytes=0, allocation_bytes=0, admit=True):
        """Private WorkspaceIO hook; account actual changes even on failed IO."""
        if not self.atomic_writes:
            raise StorageError("incremental accounting requires atomic workspace IO")
        with self._io_lock:
            touched = set()
            for path in paths:
                path.relative_to(self.path)
                while True:
                    touched.add(path)
                    if path == self.path:
                        break
                    path = path.parent

            def counts():
                logical = allocated = 0
                for path in sorted(touched, key=lambda p: len(p.parts)):
                    try:
                        info = path.lstat()
                    except FileNotFoundError:
                        continue
                    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                        raise IntegrityError("workspace IO forbids links and special files")
                    if stat.S_ISREG(info.st_mode):
                        logical += info.st_size
                    charged = _charged(info)
                    if path in self._atomic_temporaries:
                        # Dirty file extents can be allocated only at fsync.
                        charged = max(charged, _rounded(info.st_size,
                                                       _allocation_unit(self.path)))
                    allocated += charged
                return logical, allocated

            before = counts()
            if admit:
                self.check_growth(logical_bytes=logical_bytes,
                                  allocation_bytes=allocation_bytes)
            try:
                yield
            finally:
                try:
                    after = counts()
                    self._logical_bytes += after[0] - before[0]
                    self._allocated_bytes += after[1] - before[1]
                except BaseException:
                    self._accounting_ok = False
                    raise
                if admit:
                    self.check_growth()


class HotCache:
    """One quota for inputs, results, compression/download temp and checkpoints."""

    _SLOTS = {"current", "next", "latest", "best", "global", "state"}
    _KINDS = {"input", "result", "checkpoint", "workspace"}

    def __init__(
        self, root: str | Path, *, quota_bytes: int = DEFAULT_QUOTA_BYTES,
        min_free_bytes: int = MIN_FREE_BYTES, disk_usage: Callable = shutil.disk_usage,
    ):
        self.quota_bytes = _count(quota_bytes)
        self.min_free_bytes = _count(min_free_bytes)
        if not 0 < self.quota_bytes <= MAX_QUOTA_BYTES:
            raise StorageError("quota must be positive and at most 40 GB")
        if self.min_free_bytes < MIN_FREE_BYTES:
            raise StorageError("filesystem free-space floor must be at least 25 GB")
        self.root = _root(root)
        self.disk_usage = disk_usage
        self._thread_lock = threading.RLock()
        for name in ("entries", "archives", "work", "locks"):
            (self.root / name).mkdir(exist_ok=True)
        with self._locked() as state:
            if state["quota_bytes"] != self.quota_bytes or (
                state["min_free_bytes"] != self.min_free_bytes
            ):
                raise StorageError("all handles must use the same persisted quota policy")
            self._save(state)
        self.recover()

    @contextmanager
    def _locked(self):
        with self._thread_lock, _flock(self.root / ".cache.lock"):
            path = self.root / "state.json"
            state = _read_json(path) if path.exists() else {
                "schema_version": "h3d-hot-cache-v1", "quota_bytes": self.quota_bytes,
                "min_free_bytes": self.min_free_bytes,
                "entries": {}, "slots": {}, "reservations": {},
            }
            if state.get("schema_version") != "h3d-hot-cache-v1":
                raise IntegrityError("unsupported cache state")
            yield state

    def _save(self, state: dict):
        size = _rounded(len(_json_bytes(state)), _allocation_unit(self.root))
        if _tree_bytes(self.root) + size > self.quota_bytes or (
            self.disk_usage(self.root).free < self.min_free_bytes + size
        ):
            raise QuotaExceeded("cannot durably write state within storage limits")
        _atomic_json(self.root / "state.json", state)

    def _entry_path(self, key: str) -> Path:
        return self.root / "entries" / _digest(_name(key))

    def _archive_path(self, key: str) -> Path:
        return self.root / "archives" / (_digest(_name(key)) + ".tar.gz")

    def _lease_path(self, key: str) -> Path:
        return self.root / "locks" / ("e-" + _digest(_name(key)))

    def _unwritten(self, state: dict) -> int:
        scratch = sum(max(0, r["charged_bytes"] - _tree_bytes(self.root / "work" / token))
                      for token, r in state["reservations"].items())
        workspaces = sum(
            max(0, row["workspace_charged_bytes"] - _tree_bytes(self._entry_path(key)))
            for key, row in state["entries"].items()
            if row["kind"] == "workspace" and row["ready"])
        accounting = 0
        for key, row in state["entries"].items():
            if row.get("accounting_charged_bytes"):
                record = self.root / "locks" / ("w-" + _digest(key) + ".json")
                try:
                    actual = _charged(record.lstat())
                except FileNotFoundError:
                    actual = 0
                accounting += max(0, row["accounting_charged_bytes"] - actual)
        return scratch + workspaces + accounting

    def _fits(self, state: dict, additional: int):
        unwritten = self._unwritten(state)
        headroom = max(_HEADROOM, 2 * len(_json_bytes(state)))
        if _tree_bytes(self.root) + unwritten + additional + headroom > self.quota_bytes:
            raise QuotaExceeded("shared hot-cache byte quota exceeded")
        if self.disk_usage(self.root).free < (
            self.min_free_bytes + unwritten + additional + headroom
        ):
            raise QuotaExceeded("filesystem must retain its free-space floor")

    def _pinned(self, state: dict, key: str) -> bool:
        return key in state["slots"].values()

    def _deletable(self, state: dict, key: str, row: dict) -> bool:
        if row["kind"] == "workspace":
            return False
        if self._pinned(state, key) or not row["ready"]:
            return False
        if row["kind"] == "input":
            return row["archive"] is None
        return row["ack"] is not None and row["archive"] is not None

    def _delete(self, state: dict, key: str):
        row = state["entries"][key]
        if row["kind"] == "workspace":
            raise StorageError("mutable workspaces cannot be deleted by cache collection")
        if row["kind"] != "input" or row["archive"] is not None:
            data = _manifest(row["archive"])
            _check_ack(row["ack"], data)
            if not row.get("deleting"):
                if _inventory(self._entry_path(key)) != data["members"]:
                    raise IntegrityError("local payload changed after archival; refusing deletion")
        row["deleting"] = True
        self._save(state)
        path = self._entry_path(key)
        if path.exists():
            shutil.rmtree(path)
        self._archive_path(key).unlink(missing_ok=True)
        _fsync_dir(self.root / "entries")
        _fsync_dir(self.root / "archives")
        del state["entries"][key]
        self._save(state)

    def _make_space(self, state: dict, additional: int):
        candidates = sorted(state["entries"], key=lambda k: state["entries"][k]["last_used"])
        for key in [None, *candidates]:
            try:
                self._fits(state, additional)
                return
            except QuotaExceeded:
                if key is None:
                    continue
            row = state["entries"][key]
            if not self._deletable(state, key, row):
                continue
            try:
                with _flock(self._lease_path(key), blocking=False):
                    self._delete(state, key)
            except BlockingIOError:
                continue
        self._fits(state, additional)

    def reserve(
        self, *, download_bytes: int = 0, extract_bytes: int = 0,
        output_bytes: int = 0, compression_bytes: int = 0,
        allocation_bytes: int | None = None,
    ) -> Reservation:
        """Reserve logical estimates plus an allocated-space envelope.

        ``allocation_bytes`` supplies a larger expected physical allocation, e.g.
        many tiny files. Each token also reserves 64 KiB for directory/block
        overhead. Both sparse logical lengths and preallocated extents count.
        """
        size = sum(_count(n) for n in (
            download_bytes, extract_bytes, output_bytes, compression_bytes
        ))
        allocated = size if allocation_bytes is None else _count(allocation_bytes)
        charged = _rounded(max(size, allocated), _allocation_unit(self.root)) + _HEADROOM
        token = uuid.uuid4().hex
        with self._locked() as state:
            self._make_space(state, charged)
            fd = os.open(self.root / "locks" / ("r-" + token),
                         os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            path = self.root / "work" / token
            try:
                path.mkdir()
                state["reservations"][token] = {"bytes": size, "charged_bytes": charged}
                self._fits(state, 0)
                self._save(state)
                return Reservation(self, token, size, charged, fd)
            except BaseException:
                os.close(fd)
                if path.exists():
                    shutil.rmtree(path)
                raise

    def _publish(self, reservation, key, kind, payload, slots, validator,
                 workspace_bytes=None):
        _name(key)
        if kind not in self._KINDS:
            raise StorageError("unknown cache entry kind")
        if workspace_bytes is not None:
            if kind != "workspace" or _count(workspace_bytes) > reservation.bytes:
                raise StorageError("workspace ceiling must already be reserved")
        elif kind == "workspace":
            workspace_bytes = reservation.bytes
        compatible = {"input": {"current", "next", "global"}, "result": {"state"},
                      "checkpoint": {"latest", "best"}, "workspace": {"state"}}
        if any(slot not in compatible[kind] for slot in slots):
            raise StorageError("slot is incompatible with entry kind")
        source = reservation.path / _ref(payload)
        if not source.is_dir() or source.is_symlink():
            raise IntegrityError("published payload must be a directory")
        if kind == "workspace" and _tree_bytes(source, logical=True) > workspace_bytes:
            raise QuotaExceeded("initial state exceeds the workspace ceiling")
        if validator is not None:
            validator(source)
        _fsync_tree(source)
        with self._locked() as state:
            reservation._check(state)
            if key in state["entries"] or self._entry_path(key).exists():
                raise IdentityConflict("cache keys are immutable; publish a new key")
            if _tree_bytes(reservation.path, logical=True) > reservation.bytes or (
                _tree_bytes(reservation.path) > reservation.charged_bytes
            ):
                raise QuotaExceeded("external producer exceeded its reservation")
            self._fits(state, 0)
            row = {
                "kind": kind, "ready": False, "last_used": time.time_ns(),
                "archive": None, "archive_ready": False, "ack": None,
                "staging": reservation.token + "/" + payload,
            }
            if kind == "workspace":
                row["workspace_bytes"] = workspace_bytes
                row["workspace_charged_bytes"] = reservation.charged_bytes
            state["entries"][key] = row
            for slot in slots:
                state["slots"][slot] = key
            self._save(state)
            os.replace(source, self._entry_path(key))
            _fsync_dir(self.root / "entries")
            _fsync_dir(reservation.path)
            row["ready"] = True
            row.pop("staging")
            if kind == "workspace":
                state["reservations"][reservation.token]["charged_bytes"] = 0
            self._save(state)
        reservation.close()
        return self._entry_path(key)

    def populate(
        self, key: str, producer: Callable[[Reservation], object], *,
        kind: str = "input", download_bytes: int = 0, extract_bytes: int = 0,
        output_bytes: int = 0, allocation_bytes: int | None = None,
        slots: tuple[str, ...] = (),
        validator: Callable[[Path], object] | None = None,
        workspace_bytes: int | None = None,
    ) -> Path:
        """Invoke a real producer; its payload goes under reservation.path/payload."""
        if workspace_bytes is not None:
            if kind != "workspace":
                raise StorageError("only workspaces have a mutable ceiling")
            output_bytes = max(_count(output_bytes), _count(workspace_bytes))
        with self.reserve(download_bytes=download_bytes, extract_bytes=extract_bytes,
                          output_bytes=output_bytes, allocation_bytes=allocation_bytes) as reservation:
            (reservation.path / "payload").mkdir()
            producer(reservation)
            return reservation.commit(key, kind=kind, slots=slots, validator=validator,
                                      workspace_bytes=workspace_bytes)

    def new_workspace(
        self, key: str, producer: Callable[[Reservation], object], *,
        initial_bytes: int, growth_bytes: int = 0, allocation_bytes: int | None = None,
        validator: Callable[[Path], object] | None = None,
    ) -> Path:
        """Admit initial state and future growth before invoking trusted seed IO."""
        return self.populate(
            key, producer, kind="workspace", output_bytes=_count(initial_bytes),
            workspace_bytes=_count(initial_bytes) + _count(growth_bytes),
            allocation_bytes=allocation_bytes, validator=validator)

    def adopt_workspace(
        self, key: str, *, growth_bytes: int = 0,
        workspace_bytes: int | None = None, allocation_bytes: int | None = None,
    ) -> Path:
        """Promote an unarchived staged result in place after growth admission."""
        _count(growth_bytes)
        if workspace_bytes is not None:
            _count(workspace_bytes)
        if allocation_bytes is not None:
            _count(allocation_bytes)
        with self.lease(key, exclusive=True) as path:
            with self._locked() as state:
                row = state["entries"][key]
                if row["kind"] not in {"result", "workspace"} or row["archive"] is not None:
                    raise StorageError("only an unarchived result/workspace can be adopted")
                logical, physical = _tree_bytes(path, logical=True), _tree_bytes(path)
                if workspace_bytes is not None and workspace_bytes < logical + growth_bytes:
                    raise QuotaExceeded("workspace ceiling cannot cover state and requested growth")
                limit = logical + growth_bytes if workspace_bytes is None else workspace_bytes
                if allocation_bytes is not None and allocation_bytes < max(physical, limit):
                    raise QuotaExceeded("workspace allocation ceiling is too small")
                charged = _rounded(max(physical, limit, allocation_bytes or 0),
                                   _allocation_unit(path)) + _HEADROOM
                previous = max(0, row.get("workspace_charged_bytes", physical) - physical)
                limit = max(limit, row.get("workspace_bytes", 0))
                charged = max(charged, row.get("workspace_charged_bytes", 0))
                self._make_space(state, max(0, charged - physical - previous))
                row["kind"] = "workspace"
                row["workspace_bytes"], row["workspace_charged_bytes"] = limit, charged
                self._fits(state, 0)
                self._save(state)
            return path

    @contextmanager
    def workspace_lease(
        self, key: str, *, growth_bytes: int | None = None,
        allocation_bytes: int | None = None, atomic_writes: bool = False,
        incremental: bool = False,
    ) -> Iterator[WorkspaceLease]:
        """Persist peak growth before mutation; crashes retain bytes and budget.

        atomic_writes opts into WorkspaceIO-only writes within an already
        reserved fixed ceiling. Raw writers must retain the default scan/fsync
        path. Boundary scans still reject links, special files and over-quota IO.
        """
        if atomic_writes and (growth_bytes is not None or allocation_bytes is not None):
            raise StorageError("atomic workspace IO requires an already reserved fixed envelope")
        if incremental:
            if not atomic_writes:
                raise StorageError("persistent accounting requires atomic workspace IO")
            with self._incremental_workspace_lease(key) as budget:
                yield budget
            return
        if growth_bytes is not None:
            _count(growth_bytes)
        if allocation_bytes is not None:
            _count(allocation_bytes)
            if growth_bytes is None:
                raise StorageError("allocation estimate requires a logical growth estimate")
        with self.lease(key, exclusive=True) as path:
            with self._locked() as state:
                row = state["entries"][key]
                if row["kind"] != "workspace":
                    raise StorageError("mutable lease requires a workspace")
                if growth_bytes is not None:
                    physical = _tree_bytes(path)
                    allocated = growth_bytes if allocation_bytes is None else allocation_bytes
                    charged = max(
                        row["workspace_charged_bytes"],
                        physical + _rounded(max(growth_bytes, allocated),
                                            _allocation_unit(path)) + _HEADROOM)
                    previous = max(0, row["workspace_charged_bytes"] - physical)
                    self._make_space(state, max(0, charged - physical - previous))
                    row["workspace_bytes"] = max(
                        row["workspace_bytes"], _tree_bytes(path, logical=True) + growth_bytes)
                    row["workspace_charged_bytes"] = charged
                    self._fits(state, 0)
                    self._save(state)
                budget = WorkspaceLease(self, key, path, row["workspace_bytes"],
                                        row["workspace_charged_bytes"],
                                        atomic_writes=atomic_writes)
            try:
                if atomic_writes:
                    budget._check_full()
                else:
                    budget.check_growth()
                yield budget
            finally:
                try:
                    if not atomic_writes:
                        _fsync_tree(path)
                        budget.check_growth()
                    else:
                        budget._check_full()
                    if atomic_writes and budget._writers:
                        raise StorageError("atomic writers must finish before lease release")
                finally:
                    budget.active = False

    @contextmanager
    def _incremental_workspace_lease(self, key):
        """Fixed-ceiling framework writers share durable touched-path counters.

        All mutations must use WorkspaceIO. A dirty record means the previous
        owner crashed, so the next owner reconciles the tree before admission.
        Raw/external workspace leases invalidate this record. Neither a clean
        acquisition nor its release scans the workspace or updates LRU state.
        """
        _name(key)
        path = self._entry_path(key)
        record_path = self.root / "locks" / ("w-" + _digest(key) + ".json")
        fd = os.open(self._lease_path(key), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise EntryBusy("entry has an incompatible active lease") from exc
            with self._locked() as state:
                row = state["entries"].get(key)
                if row is None or not row["ready"] or row["kind"] != "workspace":
                    raise StorageError("mutable lease requires a published workspace")
                info = path.lstat()
                if not stat.S_ISDIR(info.st_mode):
                    raise IntegrityError("workspace root must be a real directory")
                anchor = {
                    "schema_version": "h3d-workspace-accounting-v1", "key": key,
                    "root_device": info.st_dev, "root_inode": info.st_ino,
                    "bytes": row["workspace_bytes"],
                    "charged_bytes": row["workspace_charged_bytes"],
                }
                budget = WorkspaceLease(
                    self, key, path, anchor["bytes"], anchor["charged_bytes"],
                    atomic_writes=True)
            saved = _read_json(record_path) if record_path.exists() else {}
            if saved.get("clean") is True and all(
                    saved.get(name) == value for name, value in anchor.items()):
                budget._logical_bytes = _count(saved["logical_bytes"])
                budget._allocated_bytes = _count(saved["allocated_bytes"])
                budget.check_growth()
            else:
                budget._check_full()
                # A previous owner may have lost directory/file durability.
                # Reconcile both bytes and fsync before declaring this clean.
                _fsync_tree(path)
                budget._check_full()

            def persist(clean):
                document = {
                    **anchor, "clean": clean,
                    "logical_bytes": budget._logical_bytes,
                    "allocated_bytes": budget._allocated_bytes,
                }
                unit = _allocation_unit(self.root)
                size = _rounded(len(_json_bytes(document)), unit)
                required = 2 * size + 4 * unit
                # Short metadata serialization also prevents a global scanner
                # from observing a temporary that is concurrently renamed.
                with self._locked() as state:
                    row = state["entries"][key]
                    previous = row.get("accounting_charged_bytes", 0)
                    if required > previous:
                        self._make_space(state, required - previous)
                        row["accounting_charged_bytes"] = required
                        self._fits(state, 0)
                        self._save(state)
                    if self.disk_usage(self.root).free < self.min_free_bytes + size + unit:
                        raise QuotaExceeded("cannot persist workspace accounting above free-space floor")
                    _atomic_json(record_path, document)

            # fsync the dirty intent before giving out a mutation capability.
            persist(False)
            completed = False
            try:
                yield budget
                completed = True
            finally:
                try:
                    if budget._writers or not budget._accounting_ok:
                        raise IntegrityError("workspace accounting did not finish")
                    budget.check_growth()
                    if completed and budget._durability_ok:
                        persist(True)
                finally:
                    budget.active = False
        finally:
            os.close(fd)

    @contextmanager
    def lease(self, key: str, *, exclusive: bool = False) -> Iterator[Path]:
        """Hold an OS lease; elapsed time/TTL can never revoke an active holder."""
        with self._locked() as state:
            row = state["entries"].get(key)
            if row is None or not row["ready"] or row.get("deleting"):
                raise StorageError("cache entry is not published")
            fd = os.open(self._lease_path(key), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                os.close(fd)
                raise EntryBusy("entry has an incompatible active lease") from exc
            try:
                if exclusive and row["kind"] == "workspace":
                    accounting = self.root / "locks" / ("w-" + _digest(key) + ".json")
                    accounting.unlink(missing_ok=True)
                    _fsync_dir(accounting.parent)
                row["last_used"] = time.time_ns()
                self._save(state)
            except BaseException:
                os.close(fd)
                raise
        try:
            yield self._entry_path(key)
        finally:
            os.close(fd)

    def pin(self, slot: str, key: str | None) -> None:
        if slot not in self._SLOTS:
            raise StorageError("unknown retained slot")
        with self._locked() as state:
            if key is None:
                state["slots"].pop(slot, None)
            else:
                row = state["entries"].get(key)
                expected = (
                    "input" if slot in {"current", "next", "global"}
                    else "result" if slot == "state" else "checkpoint")
                if row is None or not row["ready"] or not (
                    row["kind"] == expected or slot == "state" and row["kind"] == "workspace"):
                    raise StorageError("slot requires a published compatible entry")
                state["slots"][slot] = key
            self._save(state)

    def archive_candidates(self) -> list[str]:
        """Pending results and old checkpoints, including those not compressed yet."""
        with self._locked() as state:
            return [key for key, row in state["entries"].items()
                    if row["ready"] and row["kind"] in {"result", "checkpoint"}
                    and row["ack"] is None and not self._pinned(state, key)]

    def queue_archive(
        self, key: str, *, job: str, shard: str, attempt: int,
        compression_bytes: int | None = None,
    ) -> ArchivePackage:
        _name(job)
        _name(shard)
        _count(attempt)
        with self.lease(key) as source:
            with self._locked() as state:
                row = state["entries"][key]
                if row["kind"] not in {"result", "checkpoint"}:
                    raise StorageError("only results/checkpoints use the archive ACK protocol")
                if row["archive"] is not None:
                    data = _manifest(row["archive"])
                    if (data["job"], data["shard"], data["attempt"]) != (job, shard, attempt):
                        raise IdentityConflict("retry must use the existing archive identity")
                    if not row["archive_ready"]:
                        raise EntryBusy("archive publication needs recovery")
                    return ArchivePackage(self._archive_path(key), data)
            members = _inventory(source)
            total = sum(m["size"] for m in members)
            # Tar padding/PAX metadata and worst-case deflate expansion, not an
            # optimistic compression ratio. A too-small explicit budget fails closed.
            budget = total + total // 1000 + 4096 * (len(members) + 1) + _HEADROOM
            if compression_bytes is not None:
                budget = _count(compression_bytes)
            with self.reserve(compression_bytes=budget) as reservation:
                with reservation._archive_stream() as raw:
                    with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
                        with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.USTAR_FORMAT) as tar:
                            for member in members:
                                info = tarfile.TarInfo(member["path"])
                                info.mtime = info.uid = info.gid = 0
                                info.mode = 0o700 if member["kind"] == "directory" else 0o600
                                info.type = tarfile.DIRTYPE if member["kind"] == "directory" else tarfile.REGTYPE
                                info.size = member["size"]
                                if member["kind"] == "directory":
                                    tar.addfile(info)
                                else:
                                    with (source / member["path"]).open("rb") as stream:
                                        tar.addfile(info, stream)
                if _inventory(source) != members:
                    raise IntegrityError("payload changed during compression")
                path = reservation.path / "archive.tar.gz"
                body = {
                    "schema_version": _MANIFEST_SCHEMA, "job": job, "shard": shard,
                    "attempt": attempt, "sha256": _file_hash(path),
                    "archive_bytes": path.stat().st_size, "total_bytes": total,
                    "members": members, "members_sha256": _digest(members),
                }
                data = _manifest({**body, "manifest_sha256": _digest(body)})
                # Verify the archived bytes, not just the source before/after:
                # an uncooperative source writer can change and restore a file.
                _extract(path, None, data)
                with self._locked() as state:
                    row = state["entries"][key]
                    if row["archive"] is not None:
                        if row["archive"] != data:
                            raise IdentityConflict("concurrent archive identity conflict")
                        return ArchivePackage(self._archive_path(key), data)
                    row["archive"] = data
                    row["archive_staging"] = reservation.token + "/archive.tar.gz"
                    row["archive_ready"] = False
                    self._save(state)
                    os.replace(path, self._archive_path(key))
                    _fsync_dir(self.root / "archives")
                    _fsync_dir(reservation.path)
                    row["archive_ready"] = True
                    row.pop("archive_staging")
                    self._save(state)
                return ArchivePackage(self._archive_path(key), data)

    def transfer_one(
        self, key: str, transfer: Callable[[Path, dict], ArchiveAck | Mapping | None],
        *, delete_acked: bool = False,
    ) -> ArchiveAck | None:
        """Retry the saved archive only. Never invokes an inference/producer hook."""
        with self.lease(key):
            with self._locked() as state:
                row = state["entries"][key]
                if row["archive"] is None or not row["archive_ready"]:
                    raise StorageError("archive is not ready for transfer")
                data = _manifest(row["archive"])
                ack = _check_ack(row["ack"], data) if row["ack"] is not None else None
            if ack is None:
                path = self._archive_path(key)
                if path.stat().st_size != data["archive_bytes"] or _file_hash(path) != data["sha256"]:
                    raise IntegrityError("local queued archive is corrupt")
                response = transfer(path, json.loads(_json_bytes(data)))
                if response is None:
                    return None
                ack = _check_ack(response, data)
                with self._locked() as state:
                    row = state["entries"][key]
                    if row["archive"] != data:
                        raise IdentityConflict("archive changed during transfer")
                    row["ack"] = ack.as_dict()
                    self._save(state)
        if delete_acked:
            self.collect_acknowledged(key)
        return ack

    def transfer_pending(
        self, transfer: Callable[[Path, dict], ArchiveAck | Mapping | None], *,
        delete_acked: bool = False,
    ) -> list[ArchiveAck]:
        with self._locked() as state:
            keys = [k for k, row in state["entries"].items() if row["ready"]
                    and row["archive_ready"] and not row.get("deleting")]
        acks = []
        for key in keys:
            ack = self.transfer_one(key, transfer, delete_acked=delete_acked)
            if ack is not None:
                acks.append(ack)
        return acks

    def collect_acknowledged(self, key: str) -> bool:
        """Delete only a matching ACKed, unchanged, inactive, unpinned payload."""
        with self._locked() as state:
            row = state["entries"].get(key)
            if row is None:
                return False
            if row["ack"] is None:
                raise StorageError("cannot delete local data before its corresponding ACK")
            if self._pinned(state, key):
                return False
            try:
                with _flock(self._lease_path(key), blocking=False):
                    self._delete(state, key)
                    return True
            except BlockingIOError:
                return False

    def stats(self) -> dict:
        with self._locked() as state:
            physical = _tree_bytes(self.root)
            logical = _tree_bytes(self.root, logical=True)
            reserved = self._unwritten(state)
            return {"physical_bytes": physical, "logical_bytes": logical,
                    "reserved_bytes": reserved, "charged_bytes": physical + reserved,
                    "quota_bytes": self.quota_bytes, "min_free_bytes": self.min_free_bytes,
                    "entries": list(state["entries"]), "slots": dict(state["slots"]),
                    "reservations": len(state["reservations"])}

    def recover(self) -> None:
        """Reconcile durable publish intents; discard only abandoned scratch trees."""
        with self._locked() as state:
            abandoned = set()
            with_exit = []
            try:
                tokens = set(state["reservations"]) | {
                    p.name for p in (self.root / "work").iterdir()
                }
                for token in tokens:
                    _ref(token)
                    lock = _flock(self.root / "locks" / ("r-" + token), blocking=False)
                    try:
                        lock.__enter__()
                    except BlockingIOError:
                        continue
                    with_exit.append(lock)
                    abandoned.add(token)
                for key, row in list(state["entries"].items()):
                    if row.get("deleting"):
                        with _flock(self._lease_path(key), blocking=False):
                            self._delete(state, key)
                        continue
                    if not row["ready"]:
                        staging = row["staging"]
                        if staging.split("/")[0] not in abandoned:
                            continue
                        source = self.root / "work" / _ref(staging)
                        final = self._entry_path(key)
                        if not final.exists() and source.exists():
                            _fsync_tree(source)
                            os.replace(source, final)
                            _fsync_dir(final.parent)
                        if final.exists():
                            _fsync_dir(final.parent)
                            row["ready"] = True
                            row.pop("staging")
                            if row["kind"] == "workspace":
                                reservation = state["reservations"].get(staging.split("/")[0])
                                if reservation is not None:
                                    reservation["charged_bytes"] = 0
                        else:
                            # No published bytes exist; never treat this as a
                            # completed result or call a producer to reconstruct it.
                            del state["entries"][key]
                            state["slots"] = {s: k for s, k in state["slots"].items()
                                              if k != key}
                            continue
                    if row["archive"] is not None and not row["archive_ready"]:
                        staging = row["archive_staging"]
                        if staging.split("/")[0] not in abandoned:
                            continue
                        data = _manifest(row["archive"])
                        source = self.root / "work" / _ref(staging)
                        final = self._archive_path(key)
                        candidate = final if final.exists() else source
                        if candidate.exists() and candidate.stat().st_size == data["archive_bytes"] and (
                            _file_hash(candidate) == data["sha256"]
                        ):
                            if candidate != final:
                                os.replace(source, final)
                            _fsync_dir(final.parent)
                            row["archive_ready"] = True
                            row.pop("archive_staging")
                        else:
                            raise IntegrityError("incomplete archive publication is corrupt/missing")
                self._save(state)
                for token in abandoned:
                    path = self.root / "work" / token
                    if path.exists():
                        shutil.rmtree(path)
                    state["reservations"].pop(token, None)
                for path in self.root.glob(".state-*.tmp"):
                    path.unlink()
                _fsync_dir(self.root / "work")
                self._save(state)
            finally:
                for lock in reversed(with_exit):
                    lock.__exit__(None, None, None)
