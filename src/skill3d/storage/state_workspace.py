"""Zero-copy durable state and baseline-anchored delta checkpoints.

For trusted framework IO only, never generated code. Requires HotCache's
workspace API. QuotaExceeded means pause before producing more bytes; no solver,
transport, experiment, or background worker is started by this module.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
import errno
import hashlib
import io
import os
from pathlib import Path
import stat
import time
from types import MappingProxyType
from typing import Mapping
import uuid

from skill3d.storage.hot_cache import (
    EntryBusy, HotCache, IdentityConflict, IntegrityError, QuotaExceeded, StorageError,
    _allocation_unit, _count, _digest, _file_hash, _fsync_dir, _json_bytes,
    _name, _read_json, _ref, _rounded, _sha, _tree_bytes,
)

_META = ".state-workspace"
_JOURNAL = _META + "/journal.json"
_BASELINE = _META + "/baseline.json"
_DELTA = ".workspace-delta.json"
_JOURNAL_BUDGET = 64 * 1024
_BLOCK = 1024 * 1024
_SCHEMA = "h3d-state-workspace-v1"
_DELTA_SCHEMA = "h3d-state-delta-v1"


def _sync_ancestors(path, root):
    while True:
        _fsync_dir(path)
        if path == root:
            return
        path = path.parent


def _raw_ref(value):
    value = _ref(value)
    if value.split("/")[0] in {_META, _DELTA}:
        raise StorageError("workspace protocol metadata is private")
    return value


@dataclass(frozen=True)
class ArchiveRoute:
    """Trusted logical identity; HotCache binds the final content digest."""

    job: str
    shard: str
    attempt: int

    def __post_init__(self):
        _name(self.job)
        _name(self.shard)
        _count(self.attempt)

    def as_dict(self):
        return {"job": self.job, "shard": self.shard, "attempt": self.attempt}


@dataclass(frozen=True)
class BaselineAnchor:
    """Caller-verified Titan baseline identity and complete raw member inventory.

    sha256 is the trusted immutable baseline archive SHA256, not a newly computed
    hash of local state. members_sha256 additionally binds this exact inventory.
    Immutable baseline file contents are not rehashed for delta snapshots.
    """

    sha256: str
    files: Mapping[str, Mapping]
    directories: tuple[str, ...] = ()

    def __post_init__(self):
        _sha(self.sha256)
        files = {}
        for ref, row in self.files.items():
            _raw_ref(ref)
            if set(row) != {"size", "sha256"}:
                raise IntegrityError("baseline files require exact size and sha256")
            files[ref] = MappingProxyType(
                {"size": _count(row["size"]), "sha256": _sha(row["sha256"])})
        directories = set(_raw_ref(ref) for ref in self.directories)
        for ref in files:
            for parent in Path(ref).parents:
                if str(parent) != ".":
                    directories.add(parent.as_posix())
        if directories.intersection(files):
            raise IntegrityError("baseline file/directory paths conflict")
        object.__setattr__(self, "files", MappingProxyType(files))
        object.__setattr__(self, "directories", tuple(sorted(directories)))

    def as_dict(self):
        return {"sha256": self.sha256,
                "files": {ref: dict(row) for ref, row in self.files.items()},
                "directories": list(self.directories)}

    @property
    def members_sha256(self):
        return _digest({"files": self.as_dict()["files"],
                        "directories": list(self.directories)})


def _mutable(relative, roots):
    return any(relative == root or relative.startswith(root + "/") for root in roots)


def _members(root: Path):
    _tree_bytes(root)
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == _META or relative.startswith(_META + "/"):
            continue
        _raw_ref(relative)
        yield relative, path, path.lstat()


def _copy_delta_file(cache, target, source, size, sha256):
    """Trusted materialization: fixed admitted length, per-block floor and digest.

    Admission and commit validate the whole allocation envelope. No per-block
    tree scan, and this raw path capability is never delivered to caller code.
    """
    unit = _allocation_unit(cache.root)
    missing = 0
    parent = target.parent
    while not parent.exists():
        missing += 1
        parent = parent.parent
    if cache.disk_usage(cache.root).free < cache.min_free_bytes + (missing + 1) * unit:
        raise QuotaExceeded("delta directory allocation crosses filesystem free-space floor")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    digest, remaining = hashlib.sha256(), size
    with os.fdopen(fd, "wb") as dest:
        while remaining:
            block = source.read(min(_BLOCK, remaining))
            if not block:
                raise IntegrityError("delta source truncated during snapshot")
            required = _rounded(len(block), unit)
            if cache.disk_usage(cache.root).free < cache.min_free_bytes + required:
                raise QuotaExceeded("delta copy crosses filesystem free-space floor")
            dest.write(block)
            digest.update(block)
            remaining -= len(block)
        if source.read(1) or digest.hexdigest() != sha256:
            raise IntegrityError("delta source identity changed during snapshot")
        dest.flush()
        os.fsync(dest.fileno())


class _Writer:
    def __init__(self, budget, stream, *, path=None):
        self._budget, self._stream = budget, stream
        self._path = path

    def write(self, data):
        if self._budget.atomic_writes:
            with self._budget._io_lock:
                info = os.fstat(self._stream.fileno())
                size = memoryview(data).nbytes
                unit = _allocation_unit(self._budget.path)
                charged = max(_rounded(info.st_size, unit),
                              getattr(info, "st_blocks", 0) * 512)
                growth = max(0, _rounded(info.st_size + size, unit) - charged)
                with self._budget._accounted(
                    (self._path,), logical_bytes=size, allocation_bytes=growth,
                ):
                    return self._stream.write(data)
        info = os.fstat(self._stream.fileno())
        charged = max(info.st_size, getattr(info, "st_blocks", 0) * 512)
        growth = max(0, _rounded(info.st_size + len(data),
                                 _allocation_unit(self._budget.path)) - charged)
        self._budget.check_growth(logical_bytes=len(data), allocation_bytes=growth)
        return self._stream.write(data)

    def flush(self):
        self._stream.flush()

    def tell(self):
        return self._stream.tell()


class WorkspaceIO:
    """Lease-scoped bounded atomic IO. No path, fileno, or materializer escape.

    Peak growth includes old files plus replacement temporaries. Existing
    baseline files are writable only under explicitly declared mutable roots.
    An atomic_writes lease trusts exclusively this capability; raw filesystem
    mutation is outside that contract and still checked at lease boundaries.
    """

    def __init__(self, budget, *, baseline=None, mutable_roots=(), internal=False,
                 declared_only=False):
        self._budget, self._baseline = budget, baseline
        self._roots, self._internal = mutable_roots, internal
        self._declared_only = declared_only

    def _path(self, relative, *, write=False):
        relative = _ref(relative) if self._internal else _raw_ref(relative)
        if write and self._declared_only and not _mutable(relative, self._roots):
            raise StorageError("framework destination is outside declared mutable roots")
        if write and not self._internal and self._baseline is not None:
            if (relative in self._baseline.files or relative in self._baseline.directories) and (
                not _mutable(relative, self._roots)
            ):
                raise StorageError("immutable baseline member cannot be mutated")
        current = self._budget.path
        for component in relative.split("/"):
            current = current / component
            if current.is_symlink():
                raise IntegrityError("workspace IO forbids symlinks")
            if current.exists() and not (current.is_file() or current.is_dir()):
                raise IntegrityError("workspace IO forbids special files")
        return current

    def read_bytes(self, relative):
        self._budget.check_growth()
        fd = os.open(self._path(relative), os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            return stream.read()

    @contextmanager
    def open(self, relative, mode="wb"):
        if mode not in {"wb", "xb"}:
            raise StorageError("workspace writes support atomic wb/xb only")
        if self._budget.atomic_writes:
            with self._open_atomic(relative, mode) as stream:
                yield stream
            return
        self._budget.check_growth()
        path = self._path(relative, write=True)
        if mode == "xb" and path.exists():
            raise FileExistsError(path)
        missing = 0
        parent = path.parent
        while not parent.exists():
            missing += 1
            parent = parent.parent
        temp_dir = self._budget.path / _META / "tmp"
        self._budget.check_growth(
            allocation_bytes=_allocation_unit(self._budget.path) * (missing + 3))
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp = temp_dir / (uuid.uuid4().hex + ".tmp")
        try:
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb", buffering=0) as stream:
                yield _Writer(self._budget, stream)
                stream.flush()
                os.fsync(stream.fileno())
            self._budget.check_growth()
            os.replace(temp, path)
            _sync_ancestors(path.parent, self._budget.path)
            _sync_ancestors(temp_dir, self._budget.path)
        finally:
            temp.unlink(missing_ok=True)
            _fsync_dir(temp_dir)

    def _missing_directories(self, *paths):
        missing = set()
        for path in paths:
            while not path.exists():
                missing.add(path)
                path = path.parent
        return missing

    def _sync_parents(self, *paths):
        for path in paths:
            while not path.exists():
                path = path.parent
            self._sync(path)

    def _sync(self, path):
        try:
            _sync_ancestors(path, self._budget.path)
        except BaseException:
            self._budget._durability_ok = False
            raise

    @contextmanager
    def _open_atomic(self, relative, mode):
        self._budget.check_growth()
        path = self._path(relative, write=True)
        if mode == "xb" and path.exists():
            raise FileExistsError(path)
        temp_dir = WorkspaceIO(self._budget, internal=True)._path(_META + "/tmp")
        temp = temp_dir / (uuid.uuid4().hex + ".tmp")
        missing = self._missing_directories(path.parent, temp_dir)
        changed_dirs = {directory.parent for directory in missing} | {temp_dir}
        unit = _allocation_unit(self._budget.path)
        # Admit directory entries and a temporary inode separately from payload.
        allocation = unit * (len(missing) + len(changed_dirs) + 1)
        fd = None
        registered = False
        try:
            with self._budget._accounted(
                (*missing, path.parent, temp), allocation_bytes=allocation,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                temp_dir.mkdir(parents=True, exist_ok=True)
                fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                self._budget._atomic_temporaries.add(temp)
                self._budget._writers += 1
                registered = True
            with os.fdopen(fd, "wb", buffering=0) as stream:
                fd = None
                yield _Writer(self._budget, stream, path=temp)
                info = os.fstat(stream.fileno())
                charged = max(info.st_size, getattr(info, "st_blocks", 0) * 512)
                growth = max(0, _rounded(info.st_size, unit) - charged)
                with self._budget._accounted((temp,)):
                    if self._budget.cache.disk_usage(self._budget.cache.root).free < (
                        self._budget.cache.min_free_bytes + growth
                    ):
                        raise QuotaExceeded("workspace fsync crosses filesystem free-space floor")
                    try:
                        stream.flush()
                        os.fsync(stream.fileno())
                    except BaseException:
                        self._budget._durability_ok = False
                        raise
            with self._budget._accounted((temp, path), allocation_bytes=2 * unit):
                if mode == "xb" and path.exists():
                    raise FileExistsError(path)
                os.replace(temp, path)
        finally:
            try:
                if fd is not None:
                    os.close(fd)
                # Abort cleanup must work even after actual IO exceeded admission.
                with self._budget._accounted((temp,), admit=False):
                    temp.unlink(missing_ok=True)
                self._sync_parents(path.parent, temp_dir)
            finally:
                if registered:
                    with self._budget._io_lock:
                        self._budget._atomic_temporaries.discard(temp)
                        self._budget._writers -= 1

    def write_bytes(self, relative, data):
        with self.open(relative) as stream:
            for start in range(0, len(data), _BLOCK):
                stream.write(data[start:start + _BLOCK])

    def copy_file(self, relative, source):
        with Path(source).open("rb") as src, self.open(relative) as dst:
            for block in iter(lambda: src.read(_BLOCK), b""):
                dst.write(block)

    def unlink(self, relative):
        self._budget.check_growth()
        path = self._path(relative, write=True)
        if self._budget.atomic_writes:
            with self._budget._accounted((path,)):
                path.unlink()
            self._sync(path.parent)
            return
        path.unlink()
        _fsync_dir(path.parent)

    def mkdir(self, relative):
        self._budget.check_growth()
        path = self._path(relative, write=True)
        if self._budget.atomic_writes:
            missing = self._missing_directories(path)
            changed_dirs = {directory.parent for directory in missing}
            allocation = _allocation_unit(self._budget.path) * (
                len(missing) + len(changed_dirs))
            try:
                with self._budget._accounted(
                    (*missing, path), allocation_bytes=allocation,
                ):
                    path.mkdir(parents=True, exist_ok=True)
            finally:
                self._sync_parents(path)
            return
        missing = 0
        parent = path
        while not parent.exists():
            missing += 1
            parent = parent.parent
        self._budget.check_growth(
            allocation_bytes=_allocation_unit(self._budget.path) * missing)
        path.mkdir(parents=True, exist_ok=True)
        self._budget.check_growth()
        _sync_ancestors(path, self._budget.path)

    def rmdir(self, relative):
        self._budget.check_growth()
        path = self._path(relative, write=True)
        if self._budget.atomic_writes:
            with self._budget._accounted((path,)):
                path.rmdir()
            self._sync(path.parent)
            return
        path.rmdir()
        _fsync_dir(path.parent)


class StateWorkspace:
    """Adopt a staged state in place; archive bounded deltas asynchronously."""

    def __init__(self, cache: HotCache, key: str, *, identity=None, transfer=None,
                 checkpoint_validator=None, bounded=False, lock_timeout_s=120.0,
                 atomic_writes=False, incremental_leases=False,
                 reuse_checkpoint_files=False):
        _name(key)
        self.cache, self.key = cache, key
        self.identity, self.transfer = identity, transfer
        self.checkpoint_validator = checkpoint_validator
        self.bounded = bool(bounded)
        self.atomic_writes = bool(atomic_writes)
        self.incremental_leases = bool(incremental_leases)
        self.reuse_checkpoint_files = bool(reuse_checkpoint_files)
        self._baseline_cache = None
        if self.atomic_writes and not self.bounded:
            raise StorageError("atomic workspace IO requires fixed-ceiling bounded mode")
        self.lock_timeout_s = float(lock_timeout_s)
        if not 0 < self.lock_timeout_s < float("inf"):
            raise StorageError("workspace lock timeout must be finite and positive")

    @contextmanager
    def _write_lease(self, *, growth_bytes=None, allocation_bytes=None):
        """Retry acquisition only; never retry a producer or enlarge bounded state."""
        if not self.bounded:
            with self.cache.workspace_lease(
                self.key, growth_bytes=growth_bytes, allocation_bytes=allocation_bytes,
                atomic_writes=self.atomic_writes,
                incremental=self.incremental_leases,
            ) as budget:
                yield budget
            return
        deadline = time.monotonic() + self.lock_timeout_s
        while True:
            lease = self.cache.workspace_lease(
                self.key, atomic_writes=self.atomic_writes,
                incremental=self.incremental_leases)
            try:
                budget = lease.__enter__()
                break
            except EntryBusy:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        try:
            yield budget
        finally:
            lease.__exit__(None, None, None)

    @classmethod
    def adopt(cls, cache, key, *, baseline_anchor, baseline_members, mutable_roots,
              baseline_directories=(),
              growth_bytes=0, workspace_bytes=None, allocation_bytes=None,
              identity=None, transfer=None, checkpoint_validator=None):
        """No payload copying. Inject the verified archive SHA and member closure."""
        baseline = BaselineAnchor(baseline_anchor, baseline_members,
                                  tuple(baseline_directories))
        roots = tuple(sorted(set(_raw_ref(ref) for ref in mutable_roots)))
        baseline_bytes = _json_bytes(baseline.as_dict())
        cache.adopt_workspace(
            key, growth_bytes=max(_count(growth_bytes),
                                  2 * len(baseline_bytes) + _JOURNAL_BUDGET),
            workspace_bytes=workspace_bytes, allocation_bytes=allocation_bytes,
        )
        workspace = cls(cache, key, identity=identity, transfer=transfer,
                        checkpoint_validator=checkpoint_validator)
        with cache.workspace_lease(key) as budget:
            io = WorkspaceIO(budget, internal=True)
            if (budget.path / _JOURNAL).exists():
                journal = workspace._journal(budget)
                stored = workspace._baseline(budget, journal)
                if stored.as_dict() != baseline.as_dict() or journal["mutable_roots"] != list(roots):
                    raise IdentityConflict("adoption cannot change baseline or mutation scope")
            else:
                actual = {ref: info.st_size for ref, _, info in _members(budget.path)
                          if stat.S_ISREG(info.st_mode)}
                if actual != {ref: row["size"] for ref, row in baseline.files.items()}:
                    raise IntegrityError("adoption requires the complete verified baseline member closure")
                if (budget.path / _BASELINE).exists() and (
                    _read_json(budget.path / _BASELINE) != baseline.as_dict()
                ):
                    raise IdentityConflict("partial adoption bound a different baseline")
                io.write_bytes(_BASELINE, baseline_bytes)
                workspace._save_journal(budget, {
                    "schema_version": _SCHEMA, "key": key,
                    "baseline_archive_sha256": baseline.sha256,
                    "baseline_members_sha256": baseline.members_sha256,
                    "mutable_roots": list(roots), "outputs": {},
                })
        return workspace

    def _journal(self, budget):
        budget.check_growth()
        value = _read_json(budget.path / _JOURNAL)
        if value.get("schema_version") != _SCHEMA or value.get("key") != self.key or (
            not isinstance(value.get("outputs"), dict)
            or not isinstance(value.get("mutable_roots"), list)
        ):
            raise IntegrityError("workspace journal identity/schema mismatch")
        for root in value["mutable_roots"]:
            _raw_ref(root)
        for key, record in value["outputs"].items():
            _name(key)
            if not isinstance(record, dict) or record.get("kind") not in {"result", "checkpoint"}:
                raise IntegrityError("invalid durable output intent")
            ArchiveRoute(**record["route"])
        return value

    def _baseline(self, budget, journal):
        path = budget.path / _BASELINE
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise IntegrityError("baseline must be a regular file")
        stamp = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
                 journal["baseline_archive_sha256"], journal["baseline_members_sha256"])
        if self._baseline_cache is not None and self._baseline_cache[0] == stamp:
            return self._baseline_cache[1]
        value = _read_json(path)
        baseline = BaselineAnchor(**value)
        if baseline.sha256 != journal["baseline_archive_sha256"] or (
            baseline.members_sha256 != journal["baseline_members_sha256"]
        ):
            raise IntegrityError("persisted baseline identity changed")
        self._baseline_cache = (stamp, baseline)
        return baseline

    def _save_journal(self, budget, value):
        WorkspaceIO(budget, internal=True).write_bytes(_JOURNAL, _json_bytes(value))

    def _journal_growth(self):
        with self.cache.lease(self.key) as path:
            return (path / _JOURNAL).stat().st_size + _JOURNAL_BUDGET

    def extend_mutable_roots(self, roots, *, expected_roots, baseline_sha256,
                             baseline_members_sha256):
        """Explicit operator authorization while ALL application writers are paused.

        Compare-and-swap the verified scope only; neither baseline identity nor
        quota changes. The caller must record this authorization in deployment
        configuration. A file lease does not prove application quiescence.
        """
        if not self.bounded:
            raise StorageError("scope extension requires explicit bounded workspace mode")
        additions = set(_raw_ref(ref) for ref in roots)
        expected = sorted(set(_raw_ref(ref) for ref in expected_roots))
        with self._write_lease() as budget:
            journal = self._journal(budget)
            self._baseline(budget, journal)
            if journal["mutable_roots"] != expected or (
                journal["baseline_archive_sha256"] != _sha(baseline_sha256)
                or journal["baseline_members_sha256"] != _sha(baseline_members_sha256)
            ):
                raise IdentityConflict("paused scope authorization identity changed")
            journal["mutable_roots"] = sorted(set(expected) | additions)
            self._save_journal(budget, journal)
            return tuple(journal["mutable_roots"])

    def mutate(self, producer, *, estimated_growth_bytes, allocation_bytes=None):
        """Reserve peak growth before trusted IO; quota refusal invokes nobody."""
        with self._write_lease(
            growth_bytes=estimated_growth_bytes,
            allocation_bytes=allocation_bytes,
        ) as budget:
            journal = self._journal(budget)
            if self.bounded:
                budget.check_growth(
                    logical_bytes=_count(estimated_growth_bytes),
                    allocation_bytes=_count(allocation_bytes or 0))
            return producer(WorkspaceIO(
                budget, baseline=self._baseline(budget, journal),
                mutable_roots=tuple(journal["mutable_roots"]),
                declared_only=self.bounded))

    def _route(self, key, kind, route):
        if route is None:
            if self.identity is None:
                raise StorageError("archive identity is required before production")
            route = self.identity(key, kind)
        if not isinstance(route, ArchiveRoute):
            raise StorageError("identity provider must return ArchiveRoute")
        return route

    def _delta(self, source, baseline, roots):
        from skill3d.storage.cold_history import COLD_HISTORY_PATH, validate_cold_index
        cold_path = source / COLD_HISTORY_PATH
        cold_document = _read_json(cold_path) if cold_path.exists() else None
        cold = (validate_cold_index(cold_document, workspace_key=self.key)
                if cold_document is not None else {})
        files, directories, present = {}, [], set()
        for relative, path, info in _members(source):
            present.add(relative)
            if stat.S_ISDIR(info.st_mode):
                if relative not in baseline.directories:
                    directories.append(relative)
            elif relative not in baseline.files or _mutable(relative, roots):
                identity = {"size": info.st_size, "sha256": _file_hash(path)}
                if relative not in baseline.files or identity != dict(baseline.files[relative]):
                    files[relative] = identity
            elif info.st_size != baseline.files[relative]["size"]:
                raise IntegrityError("immutable baseline file size changed")
        missing = set(baseline.files) - present
        for ref in missing.intersection(cold):
            if cold[ref].identity != dict(baseline.files[ref]):
                raise IntegrityError("cold baseline reference changed")
        deleted_files = sorted(missing - set(cold))
        deleted_dirs = sorted(set(baseline.directories) - present)
        if any(not _mutable(ref, roots) for ref in (*deleted_files, *deleted_dirs)):
            raise IntegrityError("immutable baseline member disappeared")
        return {
            "schema_version": _DELTA_SCHEMA,
            "baseline_archive_sha256": baseline.sha256,
            "baseline_members_sha256": baseline.members_sha256,
            "files": files, "directories": directories,
            "deleted_files": deleted_files, "deleted_directories": deleted_dirs,
            **({"cold_history": cold_document} if cold_document is not None else {}),
        }

    @contextmanager
    def _previous_checkpoint(self, journal):
        """Lease only this workspace's immutable latest snapshot for reuse.

        Linking two immutable checkpoints preserves a standalone payload and
        archive. Live workspace files are never linked, so numeric permits and
        other in-place writers cannot change a previously published checkpoint.
        Quota admission still charges each complete logical/allocated payload.
        """
        if not (self.reuse_checkpoint_files and self.atomic_writes
                and self.incremental_leases and self.bounded):
            yield None
            return
        with self.cache._locked() as state:
            key = state["slots"].get("latest")
            output = journal["outputs"].get(key)
            row = state["entries"].get(key)
            usable = output and output["kind"] == "checkpoint" and row and (
                row["kind"] == "checkpoint" and row["ready"] and not row.get("deleting"))
        if not usable:
            yield None
            return
        with ExitStack() as held:
            try:
                path = held.enter_context(self.cache.lease(key, touch=False))
            except EntryBusy:
                yield None
                return
            except StorageError as exc:
                if type(exc) is StorageError and not self.cache.contains(key):
                    yield None
                    return
                raise
            descriptor = _read_json(path / _DELTA)
            if descriptor.get("baseline_archive_sha256") != journal["baseline_archive_sha256"] or (
                descriptor.get("baseline_members_sha256") != journal["baseline_members_sha256"]
            ):
                raise IntegrityError("previous checkpoint baseline identity changed")
            yield path, descriptor

    def _link_checkpoint_file(self, previous, relative, identity, target):
        """Reuse verified immutable bytes; final checkpoint validation is strict."""
        if previous is None or previous[1].get("files", {}).get(relative) != identity:
            return False
        source = previous[0] / relative
        info = source.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != identity["size"]:
            raise IntegrityError("previous checkpoint member changed")
        missing, parent = 0, target.parent
        while not parent.exists():
            missing += 1
            parent = parent.parent
        unit = _allocation_unit(self.cache.root)
        if self.cache.disk_usage(self.cache.root).free < (
            self.cache.min_free_bytes + (missing + 1) * unit
        ):
            raise QuotaExceeded("checkpoint link crosses filesystem free-space floor")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target, follow_symlinks=False)
        except OSError as exc:
            if exc.errno in {errno.EXDEV, errno.EOPNOTSUPP, errno.EPERM}:
                return False
            raise
        return True

    def _publish(self, key, kind, producer, *, estimated_bytes, allocation_bytes,
                 route, latest, best):
        _name(key)
        route = self._route(key, kind, route)
        if estimated_bytes is not None:
            _count(estimated_bytes)
        growth = None if self.bounded else self._journal_growth()
        with self._write_lease(
            growth_bytes=growth, allocation_bytes=growth,
        ) as budget:
            journal = self._journal(budget)
            if key in journal["outputs"] or key == self.key:
                raise IdentityConflict("output intent keys cannot be reused")
            delta = None
            required_allocation = None
            if kind == "checkpoint":
                delta = self._delta(budget.path, self._baseline(budget, journal),
                                    journal["mutable_roots"])
                exact = sum(row["size"] for row in delta["files"].values()) + len(_json_bytes(delta))
                if estimated_bytes is None:
                    estimated_bytes = exact
                if estimated_bytes < exact:
                    raise QuotaExceeded("delta exceeds estimate; pause before copying")
                unit = _allocation_unit(budget.path)
                directories = set(delta["directories"])
                for relative in (*delta["files"], *delta["directories"]):
                    directories.update(parent.as_posix() for parent in Path(relative).parents
                                       if str(parent) != ".")
                required_allocation = sum(_rounded(row["size"], unit)
                                          for row in delta["files"].values())
                required_allocation += unit * (len(directories) + 2 * len(delta["files"]) + 4)
                required_allocation += _rounded(len(_json_bytes(delta)), unit)
                if allocation_bytes is None:
                    allocation_bytes = required_allocation
            # Admit workspace journal peak AND immutable output before production.
            previous_context = (
                self._previous_checkpoint(journal) if delta is not None else nullcontext(None))
            with previous_context as previous, self.cache.reserve(
                output_bytes=estimated_bytes, allocation_bytes=allocation_bytes
            ) as reservation:
                if required_allocation is not None and required_allocation > reservation.charged_bytes:
                    raise QuotaExceeded("delta allocation estimate is too small; pause before copying")
                journal["outputs"][key] = {"kind": kind, "route": route.as_dict(), "ack": None}
                self._save_journal(budget, journal)
                (reservation.path / "payload").mkdir()
                if delta is None:
                    producer(reservation)
                else:
                    for relative in delta["directories"]:
                        if self.cache.disk_usage(self.cache.root).free < (
                            self.cache.min_free_bytes + required_allocation
                        ):
                            raise QuotaExceeded("delta allocation crosses filesystem free-space floor")
                        (reservation.path / "payload" / relative).mkdir(parents=True, exist_ok=True)
                    for relative, row in delta["files"].items():
                        target = reservation.path / "payload" / relative
                        if self._link_checkpoint_file(previous, relative, row, target):
                            continue
                        with (budget.path / relative).open("rb") as src:
                            _copy_delta_file(
                                self.cache, target, src, row["size"], row["sha256"])
                    descriptor = _json_bytes(delta)
                    _copy_delta_file(
                        self.cache, reservation.path / "payload" / _DELTA,
                        io.BytesIO(descriptor), len(descriptor),
                        hashlib.sha256(descriptor).hexdigest())
                slots = tuple(slot for slot, enabled in (
                    ("latest", latest), ("best", best)) if enabled)

                def validate(path):
                    self.validate_checkpoint(path, self._baseline(budget, journal))
                    if self.checkpoint_validator is not None:
                        self.checkpoint_validator(path)
                return reservation.commit(
                    key, kind=kind, slots=slots,
                    validator=validate if delta is not None else None)

    def publish_output(self, key, producer, *, estimated_bytes,
                       allocation_bytes=None, route=None):
        return self._publish(
            key, "result", producer, estimated_bytes=estimated_bytes,
            allocation_bytes=allocation_bytes, route=route, latest=False, best=False)

    def checkpoint(self, key, *, latest=True, best=False, route=None,
                   estimated_bytes=None, allocation_bytes=None):
        """Delta contains changed and new members plus baseline tombstones."""
        return self._publish(
            key, "checkpoint", None, estimated_bytes=estimated_bytes,
            allocation_bytes=allocation_bytes, route=route, latest=latest, best=best)

    @staticmethod
    def validate_checkpoint(path, baseline):
        """Validate the delta closure and its baseline anchor before resume."""
        path = Path(path)
        _tree_bytes(path)
        data = _read_json(path / _DELTA)
        if data.get("schema_version") != _DELTA_SCHEMA or (
            data.get("baseline_archive_sha256") != baseline.sha256
            or data.get("baseline_members_sha256") != baseline.members_sha256
        ):
            raise IntegrityError("delta checkpoint baseline identity mismatch")
        if "cold_history" in data:
            from skill3d.storage.cold_history import validate_cold_index
            cold = validate_cold_index(data["cold_history"])
            for ref, record in cold.items():
                if ref in baseline.files and record.identity != dict(baseline.files[ref]):
                    raise IntegrityError("checkpoint cold baseline reference changed")
            if set(cold).intersection(data["deleted_files"]):
                raise IntegrityError("checkpoint deleted a durable cold reference")
        expected_files = {_DELTA}
        expected_dirs = set()
        for relative, row in data["files"].items():
            _raw_ref(relative)
            _count(row["size"])
            _sha(row["sha256"])
            expected_files.add(relative)
            expected_dirs.update(parent.as_posix() for parent in Path(relative).parents
                                 if str(parent) != ".")
            member = path / relative
            if not member.is_file() or member.stat().st_size != row["size"] or (
                _file_hash(member) != row["sha256"]
            ):
                raise IntegrityError("delta checkpoint file identity changed")
        for relative in (*data["directories"], *data["deleted_files"], *data["deleted_directories"]):
            _raw_ref(relative)
        for relative in data["directories"]:
            expected_dirs.add(relative)
            expected_dirs.update(parent.as_posix() for parent in Path(relative).parents
                                 if str(parent) != ".")
        actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
        if actual != expected_files:
            raise IntegrityError("delta checkpoint file closure changed")
        actual_dirs = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_dir()}
        if actual_dirs != expected_dirs:
            raise IntegrityError("delta checkpoint directory closure changed")
        if not set(data["deleted_files"]).issubset(baseline.files) or (
            not set(data["deleted_directories"]).issubset(baseline.directories)
        ):
            raise IntegrityError("delta tombstone is not a baseline member")
        if set(data["files"]).intersection(data["deleted_files"]) or (
            set(data["directories"]).intersection(data["deleted_directories"])
        ):
            raise IntegrityError("delta both replaces and deletes a member")
        return data

    def archive_pending(self):
        """Resume saved identities, persist matching ACKs, then collect old output."""
        if self.transfer is None:
            raise StorageError("trusted archive transport must be injected")
        with self._write_lease() as budget:
            outputs = dict(self._journal(budget)["outputs"])
        acknowledged = []
        for key, record in outputs.items():
            if key not in self.cache.stats()["entries"]:
                continue  # Never regenerate uncommitted intents or collected ACKs.
            route = ArchiveRoute(**record["route"])
            with self.cache.lease(key):
                self.cache.queue_archive(key, **route.as_dict())
                ack = self.cache.transfer_one(key, self.transfer, delete_acked=False)
                if ack is None:
                    continue
                growth = None if self.bounded else self._journal_growth()
                with self._write_lease(
                    growth_bytes=growth, allocation_bytes=growth,
                ) as budget:
                    journal = self._journal(budget)
                    journal["outputs"][key]["ack"] = ack.as_dict()
                    self._save_journal(budget, journal)
            self.cache.collect_acknowledged(key)
            acknowledged.append(ack)
        if self.bounded:
            # Retained latest/best keep their saved identity. Only remove intents
            # whose matching ACK was persisted and whose payload was collected.
            with self._write_lease() as budget:
                journal = self._journal(budget)
                entries = set(self.cache.stats()["entries"])
                pruned = {key: row for key, row in journal["outputs"].items()
                          if row["ack"] is None or key in entries}
                if pruned != journal["outputs"]:
                    journal["outputs"] = pruned
                    self._save_journal(budget, journal)
        return acknowledged

    def archive_async(self, executor):
        """Caller-owned executor; submission retries persisted archive identities."""
        return executor.submit(self.archive_pending)


__all__ = ["ArchiveRoute", "BaselineAnchor", "StateWorkspace", "WorkspaceIO"]
