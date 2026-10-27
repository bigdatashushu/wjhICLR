"""ACK-backed immutable history for explicitly selected completed raw files.

Trusted framework IO only. ``collect(paths, route, key)`` does not discover
completed cases: the parent supplies that selection and uses ``lease(paths)``
while those references are active. Locks are bounded, striped flock reference
sets; collisions can conservatively report EntryBusy. Collectors share one
cache-wide retention lock, including across processes.

The private index requires a bounded StateWorkspace and uses its existing
envelope, including atomic-write peak space. Provision that envelope before use;
this module never extends it. At most 100,000 references and strictly less than
64,000,000 bytes of canonical JSON are
admitted. Records and output keys cannot be replaced or logically deleted.

``fetch_member(ack, member_key, writer)`` is an injected trusted synchronous
transport. It writes the exact member through a bounded writer, without receiving
a workspace path or reservation capability. ``materialize`` yields a physical
HotCache input path only for its lease scope; ``read_bytes`` also caps memory.
Neither operation copies bytes into the state workspace.

``records`` returns ACKed ColdReference objects; ``identity_for`` returns the raw
size/SHA identity. ``validate_cold_index`` and the optional already-held ``budget``
argument let checkpoint/deployment validators account for absent cold references
without nesting workspace leases. Parents must integrate this explicitly: the
existing StateWorkspace delta format neither carries this index nor distinguishes
cold references from tombstones.

ACKs attest durability, not authenticity. The workspace/index and both transports
must be trusted, and all raw writers must honor workspace/reference leases.
Crash recovery retries saved published archives, never a producer. An intent
without a durable published output remains unresolved with local bytes intact.
"""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Mapping

from skill3d.storage.hot_cache import (
    ArchiveAck, EntryBusy, IdentityConflict, IntegrityError, QuotaExceeded,
    StorageError, _allocation_unit, _check_ack, _count, _digest, _flock,
    _json_bytes, _manifest, _name, _pairs, _rounded, _sha,
)
from skill3d.storage.state_workspace import (
    ArchiveRoute, StateWorkspace, WorkspaceIO, _raw_ref,
)

COLD_HISTORY_PATH = ".state-workspace/cold-history.json"
MAX_COLD_RECORDS = 100_000
MAX_COLD_ARCHIVE_RECORDS = 30_000
MAX_COLD_INDEX_BYTES = 64_000_000 - 1
_SCHEMA = "h3d-cold-history-v1"
_BLOCK = 1024 * 1024
_STRIPES = 256
_MAX_REF_BYTES = 4096


def _cold_ref(ref):
    ref = _raw_ref(ref)
    if len(ref.encode("utf-8")) > _MAX_REF_BYTES:
        raise IntegrityError("cold reference is too long")
    return ref


def _selection(paths):
    if isinstance(paths, (str, bytes)):
        raise StorageError("select an explicit iterable of raw references")
    refs = set()
    for ref in paths:
        ref = _cold_ref(ref)
        if ref in refs:
            raise IdentityConflict("duplicate cold reference selection")
        refs.add(ref)
        if len(refs) > MAX_COLD_RECORDS:
            raise QuotaExceeded("cold reference count exceeds admission limit")
    if not refs:
        raise StorageError("cold reference selection cannot be empty")
    return tuple(sorted(refs))


def _directories(files):
    result = set()
    for ref in files:
        result.update(p.as_posix() for p in Path(ref).parents if str(p) != ".")
    if result.intersection(files):
        raise IntegrityError("cold file/directory paths conflict")
    return result


def _members(files):
    rows = [{"path": ref, "kind": "file", "size": row["size"],
             "sha256": row["sha256"]} for ref, row in files.items()]
    rows.extend({"path": ref, "kind": "directory", "size": 0, "sha256": None}
                for ref in _directories(files))
    return sorted(rows, key=lambda row: row["path"])


def _document(body):
    value = {**body, "index_sha256": _digest(body)}
    data = _json_bytes(value)
    if len(data) > MAX_COLD_INDEX_BYTES:
        raise QuotaExceeded("cold index exceeds its bounded JSON admission limit")
    return value, data


@dataclass(frozen=True)
class ColdReference:
    """Validated raw identity and its exact durable archive member receipt."""

    ref: str
    size: int
    sha256: str
    archive_key: str
    member_key: str
    ack: ArchiveAck

    @property
    def identity(self):
        return {"size": self.size, "sha256": self.sha256}

    def as_dict(self):
        return {**self.identity, "archive_key": self.archive_key,
                "member_key": self.member_key, "ack": self.ack.as_dict()}


def validate_cold_index(value: Mapping, *, workspace_key=None):
    """Validate the full manifest/ACK/hash chain; return ACKed raw references.

    Call once per loaded index, including from a parent-held workspace lease.
    The checksum detects corruption; it is not an authentication signature.
    """
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version", "workspace_key", "archives", "index_sha256"
    } or value["schema_version"] != _SCHEMA:
        raise IntegrityError("invalid cold index schema")
    _name(value["workspace_key"])
    if workspace_key is not None and value["workspace_key"] != workspace_key:
        raise IntegrityError("cold index belongs to a different workspace")
    archives = value["archives"]
    if not isinstance(archives, dict) or len(archives) > MAX_COLD_RECORDS:
        raise IntegrityError("invalid cold archive inventory")
    records, seen, slots = {}, set(), set()
    for key, archive in archives.items():
        _name(key)
        if key == value["workspace_key"] or not isinstance(archive, dict) or set(archive) != {
            "route", "files", "manifest", "ack"
        }:
            raise IntegrityError("invalid cold archive intent")
        if not isinstance(archive["route"], dict) or set(archive["route"]) != {
            "job", "shard", "attempt"
        }:
            raise IntegrityError("invalid cold archive route")
        route = ArchiveRoute(**archive["route"])
        slot = _digest(route.as_dict())
        if slot in slots:
            raise IdentityConflict("cold archive routes cannot be reused")
        slots.add(slot)
        files = archive["files"]
        if not isinstance(files, dict) or not files:
            raise IntegrityError("cold intent must contain exact raw file identities")
        if len(seen) + len(files) > MAX_COLD_RECORDS:
            raise QuotaExceeded("cold reference count exceeds admission limit")
        for ref, row in files.items():
            _cold_ref(ref)
            if ref in seen:
                raise IdentityConflict("cold raw references are immutable")
            seen.add(ref)
            if not isinstance(row, dict) or set(row) != {"size", "sha256", "member_key"}:
                raise IntegrityError("invalid cold raw identity")
            _count(row["size"])
            _sha(row["sha256"])
            if _cold_ref(row["member_key"]) != ref:
                raise IntegrityError("cold member key differs from the exact raw reference")
        manifest, ack = archive["manifest"], archive["ack"]
        if manifest is None:
            if ack is not None:
                raise IntegrityError("cold ACK has no bound manifest")
            continue
        manifest = _manifest(manifest)
        if any(manifest[field] != archive["route"][field] for field in archive["route"]):
            raise IntegrityError("cold manifest route differs from saved intent")
        if manifest["members"] != _members(files):
            raise IntegrityError("cold manifest differs from exact raw inventory")
        if ack is not None:
            ack = _check_ack(ack, manifest)
            for ref, row in files.items():
                records[ref] = ColdReference(
                    ref, row["size"], row["sha256"], key, row["member_key"], ack)
    _directories(dict.fromkeys(seen))
    body = {key: value[key] for key in value if key != "index_sha256"}
    _, encoded = _document(body)
    if _sha(value["index_sha256"]) != _digest(body):
        raise IntegrityError("cold index checksum changed")
    if len(encoded) > MAX_COLD_INDEX_BYTES:
        raise QuotaExceeded("cold index is too large")
    return records


class _MemberWriter:
    """Fixed length/SHA; coalesce transport chunks into at most one MiB."""

    def __init__(self, stream, size, sha256):
        self._stream, self._size, self._sha256 = stream, size, sha256
        self._written, self._digest = 0, hashlib.sha256()
        self._pending = bytearray()
        self._active = True

    def write(self, data):
        if not self._active:
            raise StorageError("cold member writer is closed")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise StorageError("cold transport must write bytes")
        data = memoryview(data).cast("B")
        if self._written + len(data) > self._size:
            raise IntegrityError("cold member exceeds its exact saved length")
        start = 0
        while start < len(data):
            count = min(_BLOCK - len(self._pending), len(data) - start)
            self._pending.extend(data[start:start + count])
            self._written += count
            start += count
            if len(self._pending) == _BLOCK:
                self._flush()
        return len(data)

    def _flush(self):
        if self._pending:
            count = self._stream.write(self._pending)
            if count != len(self._pending):
                raise IntegrityError("short cold member write")
            self._digest.update(self._pending)
            self._pending.clear()

    def finish(self):
        self._active = False
        self._flush()
        if self._written != self._size or self._digest.hexdigest() != self._sha256:
            raise IntegrityError("cold member size/SHA differs from its saved identity")


class _ArchiveWorkspace(StateWorkspace):
    """Use archive_pending for one intent, preserving all unrelated journal rows."""

    def __init__(self, workspace, key, transfer):
        super().__init__(
            workspace.cache, workspace.key, identity=workspace.identity,
            transfer=transfer, bounded=workspace.bounded,
            lock_timeout_s=workspace.lock_timeout_s,
            atomic_writes=workspace.atomic_writes)
        self._archive_key = key

    def _journal(self, budget):
        journal = super()._journal(budget)
        journal["outputs"] = {
            key: row for key, row in journal["outputs"].items()
            if key == self._archive_key}
        return journal

    def _save_journal(self, budget, value):
        journal = super()._journal(budget)
        journal["outputs"].pop(self._archive_key, None)
        journal["outputs"].update(value["outputs"])
        super()._save_journal(budget, journal)


class ColdHistory:
    def __init__(self, workspace: StateWorkspace, fetch_member, *,
                 max_read_bytes=64 * 1024 * 1024):
        if not isinstance(workspace, StateWorkspace) or not callable(fetch_member):
            raise StorageError("cold history requires a workspace and trusted member transport")
        if not workspace.bounded:
            raise StorageError("cold history requires a fixed-ceiling bounded workspace")
        self.workspace, self.cache = workspace, workspace.cache
        self.fetch_member = fetch_member
        self.max_read_bytes = _count(max_read_bytes)

    def _load(self, budget):
        if budget.cache is not self.cache or budget.key != self.workspace.key or not budget.active:
            raise StorageError("cold index requires this workspace's active lease")
        io = WorkspaceIO(budget, internal=True)
        path = io._path(COLD_HISTORY_PATH)
        if not path.exists():
            value, _ = _document({
                "schema_version": _SCHEMA, "workspace_key": self.workspace.key,
                "archives": {}})
            return value
        if not path.is_file() or path.stat().st_size > MAX_COLD_INDEX_BYTES:
            raise IntegrityError("invalid or oversized cold index file")
        try:
            value = json.loads(io.read_bytes(COLD_HISTORY_PATH), object_pairs_hook=_pairs)
        except (ValueError, UnicodeError) as exc:
            raise IntegrityError("unreadable cold history index") from exc
        validate_cold_index(value, workspace_key=self.workspace.key)
        return value

    def _save(self, budget, value):
        body = {key: value[key] for key in value if key != "index_sha256"}
        value, data = _document(body)
        validate_cold_index(value, workspace_key=self.workspace.key)
        WorkspaceIO(budget, internal=True).write_bytes(COLD_HISTORY_PATH, data)

    @contextmanager
    def _lock(self, name, *, exclusive=True):
        path = self.cache.root / "locks" / name
        with self.cache._locked() as state:
            if not path.exists():
                self.cache._fits(state, _allocation_unit(self.cache.root))
            lock = _flock(path, exclusive=exclusive, blocking=False)
            try:
                lock.__enter__()
            except BlockingIOError as exc:
                raise EntryBusy("cold reference/retention set has an active lease") from exc
            try:
                self.cache._fits(state, 0)
            except BaseException:
                lock.__exit__(None, None, None)
                raise
        try:
            yield
        finally:
            lock.__exit__(None, None, None)

    def _stripe(self, ref):
        return int(_digest([self.workspace.key, ref])[:8], 16) % _STRIPES

    @contextmanager
    def _references(self, refs, *, exclusive):
        with ExitStack() as stack:
            for stripe in sorted({self._stripe(ref) for ref in refs}):
                stack.enter_context(self._lock("cold-ref-" + str(stripe), exclusive=exclusive))
            yield

    @contextmanager
    def lease(self, paths):
        """Protect an explicit active raw reference set before collection starts.

        All framework users of raw paths must participate, including readers
        before a reference becomes cold. The scope is flock-backed, not TTL-based.
        """
        refs = _selection(paths)
        with self._references(refs, exclusive=False):
            yield refs

    def records(self, *, budget=None):
        if budget is not None:
            return validate_cold_index(self._load(budget), workspace_key=self.workspace.key)
        with self.workspace._write_lease() as held:
            return self.records(budget=held)

    def identity_for(self, ref, *, budget=None):
        return self.validate_reference(ref, budget=budget).identity

    def validate_reference(self, ref, *, expected_identity=None, budget=None):
        ref = _cold_ref(ref)
        record = self.records(budget=budget).get(ref)
        if record is None:
            raise StorageError("raw reference has no saved durable cold ACK")
        if expected_identity is not None and record.identity != dict(expected_identity):
            raise IntegrityError("cold reference differs from the expected raw identity")
        return record

    def _raw_identity(self, budget, ref):
        path = WorkspaceIO(budget, internal=True)._path(ref)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise IntegrityError("cold history requires regular raw files")
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(_BLOCK), b""):
                digest.update(block)
            after = os.fstat(stream.fileno())
        current = path.lstat()
        signature = lambda info: (
            info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if signature(before) != signature(after) or signature(after) != signature(current):
            raise IntegrityError("local raw bytes changed while hashing")
        return {"size": current.st_size, "sha256": digest.hexdigest()}

    def _projected(self, value, key):
        """Admit the final full manifest and ACK before any output production."""
        projected = json.loads(_json_bytes(value))
        archive = projected["archives"][key]
        members = _members(archive["files"])
        total = sum(row["size"] for row in members)
        body = {
            "schema_version": "h3d-archive-v1", **archive["route"], "sha256": "0" * 64,
            "archive_bytes": total + total // 1000 + 4096 * (len(members) + 1) + 65536,
            "total_bytes": total, "members": members, "members_sha256": _digest(members),
        }
        manifest = _manifest({**body, "manifest_sha256": _digest(body)})
        archive["manifest"] = manifest
        archive["ack"] = ArchiveAck.parse({
            "schema_version": "h3d-archive-ack-v1", **archive["route"],
            "sha256": manifest["sha256"], "manifest_sha256": manifest["manifest_sha256"],
        }).as_dict()
        projected, data = _document(
            {field: projected[field] for field in projected if field != "index_sha256"})
        validate_cold_index(projected, workspace_key=self.workspace.key)
        return len(data)

    def _bind_manifest(self, key, manifest):
        manifest = _manifest(manifest)
        with self.workspace._write_lease() as budget:
            value = self._load(budget)
            archive = value["archives"][key]
            if archive["manifest"] is not None and archive["manifest"] != manifest:
                raise IdentityConflict("cold retry changed the saved archive identity")
            archive["manifest"] = manifest
            self._save(budget, value)

    def _save_ack(self, key, ack):
        with self.workspace._write_lease() as budget:
            value = self._load(budget)
            archive = value["archives"][key]
            if archive["manifest"] is None:
                raise IntegrityError("cannot save a cold ACK without a bound manifest")
            ack = _check_ack(ack, archive["manifest"])
            if archive["ack"] is not None:
                if ArchiveAck.parse(archive["ack"]) != ack:
                    raise IdentityConflict("cold saved ACK cannot change")
                return ack
            archive["ack"] = ack.as_dict()
            self._save(budget, value)
        return ack

    def _remove_raw(self, refs, key):
        with self.workspace._write_lease() as budget:
            value = self._load(budget)
            archive = value["archives"][key]
            if archive["ack"] is None:
                return {}
            records = validate_cold_index(value, workspace_key=self.workspace.key)
            io = WorkspaceIO(budget, internal=True)
            present = []
            # Confirm the entire selected set before removing its first member.
            for ref in refs:
                path = io._path(ref)
                if path.exists():
                    if self._raw_identity(budget, ref) != records[ref].identity:
                        raise IntegrityError("local bytes changed after ACK; refusing cold removal")
                    present.append(ref)
            for ref in present:
                # Reconfirm under the same lease immediately before each unlink.
                if self._raw_identity(budget, ref) != records[ref].identity:
                    raise IntegrityError("local bytes changed before cold unlink")
                io.unlink(ref)
            return {ref: records[ref] for ref in refs}

    def collect(self, paths, route: ArchiveRoute, key: str):
        """Archive a caller-selected completed set; retry only its saved identity.

        Return this set's ACKed records, or {} when the trusted receiver supplies
        no ACK. Failure never authorizes raw unlink. Existing intents with no
        published output are deliberately not regenerated.
        """
        refs = _selection(paths)
        if len(refs) > MAX_COLD_ARCHIVE_RECORDS:
            raise QuotaExceeded("split cold history into bounded member manifests")
        _name(key)
        if not isinstance(route, ArchiveRoute) or key == self.workspace.key:
            raise StorageError("cold output requires its own key and ArchiveRoute")
        if not callable(self.workspace.transfer):
            raise StorageError("trusted archive transport must be injected")
        with self._lock("cold-retention"), self._references(refs, exclusive=True), ExitStack() as output:
            output.enter_context(self._lock(self.cache._lease_path(key).name, exclusive=False))
            new = False
            with self.workspace._write_lease() as budget:
                value = self._load(budget)
                archive = value["archives"].get(key)
                if archive is None:
                    journal = self.workspace._journal(budget)
                    if key in journal["outputs"] or key in self.cache.stats()["entries"]:
                        raise IdentityConflict("cold output key was already used")
                    seen = {ref for row in value["archives"].values() for ref in row["files"]}
                    if seen.intersection(refs):
                        raise IdentityConflict("cold raw references cannot be rebound")
                    if len(seen) + len(refs) > MAX_COLD_RECORDS:
                        raise QuotaExceeded("cold reference count exceeds admission limit")
                    if any(row["route"] == route.as_dict()
                           for row in value["archives"].values()):
                        raise IdentityConflict("cold archive routes cannot be reused")
                    files = {ref: {**self._raw_identity(budget, ref), "member_key": ref}
                             for ref in refs}
                    archive = {"route": route.as_dict(), "files": files,
                               "manifest": None, "ack": None}
                    value["archives"][key] = archive
                    final_bytes = self._projected(value, key)
                    unit = _allocation_unit(budget.path)
                    budget.check_growth(
                        logical_bytes=2 * final_bytes,
                        allocation_bytes=2 * _rounded(final_bytes, unit) + 4 * unit)
                    self._save(budget, value)
                    new = True
                elif archive["route"] != route.as_dict() or tuple(sorted(archive["files"])) != refs:
                    raise IdentityConflict("cold retry must use the exact saved key, route and paths")
                if archive["ack"] is not None:
                    saved_ack = True
                else:
                    saved_ack = False
                    for ref, row in archive["files"].items():
                        if self._raw_identity(budget, ref) != {
                            "size": row["size"], "sha256": row["sha256"]
                        }:
                            raise IntegrityError("cold intent's local raw bytes changed")
            if saved_ack:
                result = self._remove_raw(refs, key)
                output.close()
                self._collect_saved_output(key)
                return result
            if new:
                self._publish_raw(key, archive, route)
            if key not in self.cache.stats()["entries"]:
                raise StorageError("unresolved cold intent has no published output; never regenerate")
            # This lease also prevents generic cache collection from discarding
            # an ACKed output before its receipt has reached the cold index.
            with self.cache.lease(key):
                package = self.cache.queue_archive(key, **route.as_dict())
                self._bind_manifest(key, package.manifest)

                def transfer(path, manifest):
                    if _manifest(manifest) != package.manifest:
                        raise IntegrityError("cold transfer changed its exact bound manifest")
                    response = self.workspace.transfer(path, manifest)
                    return None if response is None else self._save_ack(
                        key, _check_ack(response, package.manifest))

                archiver = _ArchiveWorkspace(self.workspace, key, transfer)
                acknowledged = archiver.archive_pending()
                for ack in acknowledged:
                    self._save_ack(key, _check_ack(ack, package.manifest))
                # A concurrent archiver may already have persisted the cache ACK
                # without invoking our wrapper. The output lease protects it.
                with self.cache._locked() as state:
                    row = state["entries"][key]
                    cached_ack = row["ack"]
                if cached_ack is not None:
                    self._save_ack(key, _check_ack(cached_ack, package.manifest))
            result = self._remove_raw(refs, key)
            output.close()
            if result:
                self._collect_saved_output(key)
            return result

    def _collect_saved_output(self, key):
        with self.workspace._write_lease() as budget:
            archive = self._load(budget)["archives"][key]
        with self.cache._locked() as state:
            row = state["entries"].get(key)
            if row is not None:
                if row["archive"] != archive["manifest"]:
                    raise IntegrityError("collected output differs from saved cold ACK")
                if row["ack"] is not None and (
                    _check_ack(row["ack"], row["archive"]) != ArchiveAck.parse(archive["ack"])
                ):
                    raise IntegrityError("collected output ACK differs from saved cold ACK")
        if row is not None:
            def saved_receipt(path, manifest):
                if _manifest(manifest) != archive["manifest"]:
                    raise IntegrityError("cold resume changed its bound archive")
                return ArchiveAck.parse(archive["ack"])
            # Recover a crash after the cold receipt write but before the cache
            # ACK write; no transport call or producer is required.
            self.cache.transfer_one(key, saved_receipt)
            self.cache.collect_acknowledged(key)
        if self.workspace.bounded:
            with self.workspace._write_lease() as budget:
                journal = self.workspace._journal(budget)
                record = journal["outputs"].get(key)
                if record is not None and key not in self.cache.stats()["entries"]:
                    if record["route"] != archive["route"] or (
                        record["ack"] is not None
                        and ArchiveAck.parse(record["ack"]) != ArchiveAck.parse(archive["ack"])
                    ):
                        raise IntegrityError("cold output journal differs from its saved ACK")
                    journal["outputs"].pop(key)
                    self.workspace._save_journal(budget, journal)

    def _publish_raw(self, key, archive, route):
        files = archive["files"]
        total = sum(row["size"] for row in files.values())
        unit = _allocation_unit(self.cache.root)
        allocation = sum(_rounded(row["size"], unit) for row in files.values())
        allocation += unit * (len(_directories(files)) + 2 * len(files) + 4)

        def producer(reservation):
            # publish_output already owns the exclusive workspace lease.
            path = self.cache._entry_path(self.workspace.key)
            for ref, row in files.items():
                current = path
                for part in ref.split("/"):
                    current = current / part
                    if current.is_symlink():
                        raise IntegrityError("cold producer forbids source symlinks")
                fd = os.open(current, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(fd, "rb") as source, reservation.open("payload/" + ref) as dest:
                    if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                        raise IntegrityError("cold source is not a regular file")
                    writer = _MemberWriter(dest, row["size"], row["sha256"])
                    try:
                        for block in iter(lambda: source.read(_BLOCK), b""):
                            writer.write(block)
                        writer.finish()
                    finally:
                        writer._active = False

        self.workspace.publish_output(
            key, producer, estimated_bytes=total, allocation_bytes=allocation, route=route)

    @contextmanager
    def materialize(self, ref):
        """Yield a hash-validated transient physical path under a HotCache lease."""
        ref = _cold_ref(ref)
        with self.lease((ref,)), ExitStack() as held:
            record = self.validate_reference(ref)
            key = "cold-member:" + _digest([self.workspace.key, record.as_dict()])
            with self._lock("cold-fetch-" + str(self._stripe(ref))):
                # Acquire the future entry's shared flock before publishing, so
                # quota eviction cannot race the commit-to-lease transition.
                entry_lock = self.cache._lease_path(key)
                held.enter_context(self._lock(entry_lock.name, exclusive=False))
                if key not in self.cache.stats()["entries"]:
                    with self.cache.reserve(output_bytes=record.size) as reservation:
                        with reservation.open("payload/member") as dest:
                            writer = _MemberWriter(dest, record.size, record.sha256)
                            try:
                                self.fetch_member(record.ack, record.member_key, writer)
                                writer.finish()
                            finally:
                                writer._active = False
                        reservation.commit(key, kind="input")
                path = held.enter_context(self.cache.lease(key)) / "member"
                if path.is_symlink() or not path.is_file() or path.stat().st_size != record.size:
                    raise IntegrityError("hot cold-member copy has changed")
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(fd, "rb") as stream:
                    digest = hashlib.sha256()
                    for block in iter(lambda: stream.read(_BLOCK), b""):
                        digest.update(block)
                if digest.hexdigest() != record.sha256:
                    raise IntegrityError("hot cold-member SHA has changed")
            yield path

    def read_bytes(self, ref):
        record = self.validate_reference(ref)
        if record.size > self.max_read_bytes:
            raise QuotaExceeded("cold read exceeds memory cap; use materialize")
        with self.materialize(ref) as path:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as stream:
                data = stream.read(record.size + 1)
            if len(data) != record.size or hashlib.sha256(data).hexdigest() != record.sha256:
                raise IntegrityError("cold member changed during read")
            return data


__all__ = [
    "COLD_HISTORY_PATH", "MAX_COLD_RECORDS", "MAX_COLD_INDEX_BYTES",
    "ColdHistory", "ColdReference", "validate_cold_index",
]
