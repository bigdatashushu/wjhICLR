"""Opt-in bounded IO for trusted fit-set framework writers, not generated code.

Private method copies adapt legacy writer helpers without patching modules or
builtins globally. A section spans file IO only. Models and streaming callbacks
must run outside sections. Reopening requires a previously adopted workspace.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import inspect
import json
import os
from pathlib import Path
import threading
from types import FunctionType, MethodType, SimpleNamespace
import uuid

from skill3d.storage.hot_cache import (
    EntryBusy, HotCache, IntegrityError, QuotaExceeded, StorageError,
    _HEADROOM, _allocation_unit, _flock, _rounded,
)
from skill3d.storage.state_workspace import ArchiveRoute, StateWorkspace

MAX_WORKSPACE_BYTES = 10_000_000_000


class FrameworkStorageRefused(BaseException):
    """Stop scheduling; storage refusal is not a scientific/model failure."""


def _function_copy(function, replacements):
    namespace = {**function.__globals__, **replacements}
    copied = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    copied.__kwdefaults__ = function.__kwdefaults__
    return copied


def bind_framework_writers(instance, owner, replacements):
    """Instance-local adapters; nothing is attached to the online sandbox."""
    for name, function in vars(owner).items():
        if not name.startswith("__") and inspect.isfunction(function) and (
            set(function.__code__.co_names).intersection(replacements)
        ):
            setattr(instance, name, MethodType(
                _function_copy(function, replacements), instance))


class FrameworkStateIO:
    def __init__(self, workspace, *, job, attempt=0, max_file_bytes=64 * 1024 * 1024,
                 max_workspace_bytes=MAX_WORKSPACE_BYTES):
        if not workspace.bounded:
            raise StorageError("framework IO requires fixed-ceiling StateWorkspace")
        ArchiveRoute(job, "validation", attempt)
        if not isinstance(max_file_bytes, int) or max_file_bytes < 1:
            raise StorageError("framework file ceiling must be positive")
        if not 0 < max_workspace_bytes <= MAX_WORKSPACE_BYTES:
            raise StorageError("framework workspace ceiling must be at most 10 GB")
        self._workspace = workspace
        workspace.atomic_writes = True
        workspace.incremental_leases = True
        workspace.reuse_checkpoint_files = True
        self.job, self.attempt = job, attempt
        self.max_file_bytes = max_file_bytes
        self.max_workspace_bytes = max_workspace_bytes
        self._cold_fetch = None
        self._cold_history = None
        with workspace.cache._locked() as state:
            row = state["entries"].get(workspace.key)
            if row is None or row["kind"] != "workspace" or not row["ready"]:
                raise StorageError("adopt and anchor the workspace while paused first")
            physical_ceiling = _rounded(
                max_workspace_bytes, _allocation_unit(workspace.cache.root)) + _HEADROOM
            if row["workspace_bytes"] > max_workspace_bytes or (
                row["workspace_charged_bytes"] > physical_ceiling
            ):
                raise StorageError("persisted workspace envelope exceeds the fixed ceiling")
        self.root = workspace.cache._entry_path(workspace.key)
        with workspace._write_lease() as budget:
            journal = workspace._journal(budget)
            workspace._baseline(budget, journal)
            self.mutable_roots = tuple(journal["mutable_roots"])
        self._reset_process()

    def _reset_process(self):
        self._pid = os.getpid()
        self._local = threading.local()
        self._executor = None
        self._future = None
        self._background_lock = threading.Lock()
        self._governors = {}
        self._governor_lock = threading.Lock()

    def __getstate__(self):
        state = {key: value for key, value in self.__dict__.items()
                 if key not in {"_workspace", "_local", "_executor", "_future",
                                "_background_lock", "_cold_history",
                                "_governors", "_governor_lock"}}
        workspace = self._workspace
        cache = workspace.cache
        state["_cold_fetch"] = self._compact_transport_callback(self._cold_fetch)
        state["_workspace_config"] = {
            "root": cache.root, "quota_bytes": cache.quota_bytes,
            "min_free_bytes": cache.min_free_bytes, "disk_usage": cache.disk_usage,
            "key": workspace.key, "transfer": self._compact_transport_callback(workspace.transfer),
            "lock_timeout_s": workspace.lock_timeout_s,
            "atomic_writes": True,
            "incremental_leases": True,
            "reuse_checkpoint_files": True,
        }
        return state

    @staticmethod
    def _compact_transport_callback(callback):
        """Transport callbacks cannot pull a full scene runtime into a child."""
        name = getattr(callback, "__name__", None)
        owner = getattr(callback, "__self__", None)
        if name in {"archive_transfer", "fetch_cold_member"}:
            from skill3d.storage.scene_runtime import SceneHotRuntime
            if type(owner) is SceneHotRuntime and not any(
                method in owner.__dict__ for method in (
                    "_request", "archive_transfer", "fetch_cold_member", "archive_transport")
            ):
                return getattr(owner.archive_transport(), name)
        return callback

    def __setstate__(self, state):
        config = dict(state.pop("_workspace_config"))
        cache = HotCache(config.pop("root"), quota_bytes=config.pop("quota_bytes"),
                         min_free_bytes=config.pop("min_free_bytes"),
                         disk_usage=config.pop("disk_usage"))
        self.__dict__.update(state)
        self._workspace = StateWorkspace(cache, bounded=True, **config)
        self._reset_process()
        self._cold_history = None
        if self._cold_fetch is not None:
            self.enable_cold_history(self._cold_fetch)

    def _process(self):
        if self._pid != os.getpid():
            self._reset_process()

    def relative(self, path):
        path = Path(path)
        if not path.is_absolute() or ".." in path.parts:
            raise StorageError("framework destinations require absolute lexical paths")
        try:
            relative = path.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise StorageError("framework destination is outside adopted workspace") from exc
        if not any(relative == root or relative.startswith(root + "/")
                   for root in self.mutable_roots):
            raise StorageError("framework destination is outside declared mutable roots")
        return relative

    def validate_root(self, path):
        path = Path(path)
        if not path.is_absolute() or ".." in path.parts:
            raise StorageError("framework root requires an absolute lexical path")
        try:
            ref = path.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise StorageError("framework root is outside adopted workspace") from exc
        if not any(ref == root or ref.startswith(root + "/") or root.startswith(ref + "/")
                   for root in self.mutable_roots):
            raise StorageError("framework root has no declared mutable destinations")

    def validate_destinations(self, paths):
        """Check the configured phase's write scope before dispatch, without IO."""
        try:
            for path in paths:
                self.relative(path)
        except StorageError as exc:
            raise FrameworkStorageRefused("missing declared framework write root") from exc

    @contextmanager
    def section(self):
        self._process()
        if self._future is not None and self._future.done():
            self._future.result()
        active = getattr(self._local, "io", None)
        if active is not None:
            yield active
            return
        try:
            with self._workspace._write_lease() as budget:
                journal = self._workspace._journal(budget)
                from skill3d.storage.state_workspace import WorkspaceIO
                active = WorkspaceIO(
                    budget, baseline=self._workspace._baseline(budget, journal),
                    mutable_roots=tuple(journal["mutable_roots"]), declared_only=True)
                self._local.io = active
                try:
                    yield active
                finally:
                    self._local.io = None
        except (QuotaExceeded, EntryBusy) as exc:
            raise FrameworkStorageRefused(type(exc).__name__) from exc

    def mkdir(self, path):
        ref = self.relative(path)
        with self.section() as io:
            io.mkdir(ref)

    def write_bytes(self, path, data, *, once=False, exclusive=False):
        if not isinstance(data, bytes):
            raise TypeError("trusted framework IO requires bytes")
        if len(data) > self.max_file_bytes:
            raise FrameworkStorageRefused("framework file ceiling exceeded")
        ref = self.relative(path)
        with self.section() as io:
            destination = io._path(ref, write=True)
            if destination.exists() and (once or exclusive):
                if exclusive:
                    raise FileExistsError(path)
                if io.read_bytes(ref) != data:
                    raise IntegrityError("immutable framework file changed")
                return
            io.write_bytes(ref, data)

    def write_bytes_once(self, path, data):
        self.write_bytes(path, data, once=True)

    def write_json(self, path, value):
        self.write_bytes(path, self._json_bytes(value))

    def write_json_once(self, path, value):
        self.write_bytes_once(path, self._json_bytes(value))

    @staticmethod
    def _json_bytes(value):
        return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False, default=str) + "\n").encode("utf-8")

    def write_text_once(self, path, text):
        self.write_bytes_once(path, text.encode("utf-8"))

    def append_bytes(self, path, data):
        ref = self.relative(path)
        with self.section() as io:
            destination = io._path(ref, write=True)
            old_size = destination.stat().st_size if destination.exists() else 0
            if old_size + len(data) > self.max_file_bytes:
                raise FrameworkStorageRefused("framework trace ceiling exceeded")
            old = io.read_bytes(ref) if destination.exists() else b""
            self.write_bytes(path, old + data)

    def trace_store(self, root):
        return WorkspaceTraceStore(self, root)

    def enable_cold_history(self, fetch_member):
        from skill3d.storage.cold_history import ColdHistory
        self._cold_fetch = fetch_member
        self._cold_history = ColdHistory(self._workspace, fetch_member)

    def cold_reference(self, path, *, records=None):
        if self._cold_history is None:
            return None
        path = Path(path)
        if not path.is_absolute() or ".." in path.parts:
            raise IntegrityError("cold reads require absolute lexical paths")
        try:
            ref = path.relative_to(self.root).as_posix()
        except ValueError:
            return None
        if records is None:
            with self.section() as io:
                records = self._cold_history.records(budget=io._budget)
        return records.get(ref)

    def read_history_bytes(self, path):
        path = Path(path)
        if self._cold_history is None:
            return path.read_bytes()
        if not path.is_absolute() or ".." in path.parts:
            raise IntegrityError("history reads require an absolute lexical path")
        ref = path.relative_to(self.root).as_posix()
        with self._cold_history.lease((ref,)):
            record = self.cold_reference(path)
            if record is None:
                return path.read_bytes()
            return self._cold_history.read_bytes(record.ref)

    def _archive_completed_history(self):
        """Retire only immutable completed episodes and finalized gate campaigns."""
        from skill3d.storage.cold_history import MAX_COLD_ARCHIVE_RECORDS
        from skill3d.storage.hot_cache import _digest
        history = self._cold_history
        if history is None:
            return
        with self._workspace._write_lease() as budget:
            document = history._load(budget)
        # Resume saved identities before any generic output collection.
        for key, archive in document["archives"].items():
            if archive["ack"] is None or any(
                    (self.root / ref).exists() for ref in archive["files"]):
                history.collect(tuple(archive["files"]),
                                ArchiveRoute(**archive["route"]), key)
        with self._workspace._write_lease() as budget:
            existing = history.records(budget=budget)
        selected = []
        def add(path):
            if path.is_file() and not path.is_symlink():
                ref = path.relative_to(self.root).as_posix()
                if ref not in existing:
                    selected.append((ref, path.stat().st_size))
        for phase in ("initial_validation", "accumulation", "final_validation"):
            base = self.root / phase
            for result in (base / "results").glob("*.json"):
                variant = base / "variants" / result.stem
                for directory in ("trace", "offline_images"):
                    for path in (variant / directory).rglob("*"):
                        add(path)
                if phase != "accumulation":
                    add(variant / "outcome.json")
            if phase == "accumulation":
                for worker in (base / "workers").glob("*"):
                    for campaign in (worker / "campaigns").glob("*"):
                        if not (campaign / "result.json").is_file():
                            continue
                        add(campaign / "experience_bundle.json")
                        # result.json is written after gate scoring/publication;
                        # later recovery returns it before reopening gate episodes.
                        for path in (campaign / "gate").rglob("*"):
                            if "trace" in path.parts or path.name == "outcome.json":
                                add(path)
        selected = sorted(set(selected))
        batches, current, size = [], [], 0
        for ref, amount in selected:
            if current and (size + amount > 128 * 1024 * 1024
                            or len(current) >= MAX_COLD_ARCHIVE_RECORDS):
                batches.append(current)
                current, size = [], 0
            current.append(ref)
            size += amount
        if current:
            batches.append(current)
        for refs in batches:
            key = "history-" + _digest([self.job, refs])[:32]
            history.collect(refs, ArchiveRoute(self.job, key, self.attempt), key)

    def request_governor(self, root, config):
        self._process()
        key = (Path(root).resolve(), config)
        with self._governor_lock:
            if key not in self._governors:
                self._governors[key] = WorkspaceRequestGovernor(self, key[0], config)
            return self._governors[key]

    def adapt_reviser(self, reviser):
        from skill3d.evolution.fitset_revision_v11 import FitSetSlotReviser
        self.validate_root(reviser.root)
        bind_framework_writers(reviser, FitSetSlotReviser, {
            "_write_create_once": self.write_json_once,
            "_write_text_create_once": self.write_text_once,
        })
        reviser.framework_state_io = self

    def adapt_analysis_reviser(self, reviser):
        from skill3d.evolution.revision_edits_v11 import AnchoredSlotReviser
        from skill3d.evolution.trajectory_analysis_v11 import TrajectoryAnalyst
        self.validate_root(reviser.root)
        replacements = {
            "_write_create_once": self.write_json_once,
            "_write_text_create_once": self.write_text_once,
        }
        bind_framework_writers(reviser, AnchoredSlotReviser, replacements)
        if isinstance(reviser, TrajectoryAnalyst):
            bind_framework_writers(reviser, TrajectoryAnalyst, replacements)

    def adapt_diagnoser(self, diagnoser, context):
        from skill3d.evolution.diagnosis_v11 import V11CaseDiagnoser, _sha
        root = diagnoser.audit_root / _sha(context)[:24]
        self.mkdir(root)
        # The existing diagnosis lock spans its request, but it contains no
        # bytes and never holds workspace EX during that request.
        self.write_bytes_once(root / "request.lock", b"")
        bind_framework_writers(diagnoser, V11CaseDiagnoser, {
            "_write_once": self.write_json_once,
        })

    def write_snapshot(self, root, snapshot, manifest, **kwargs):
        from skill3d.skills import v11_library as library
        adapted = _function_copy(library.write_v11_snapshot, {
            "_write_create_once": self.write_bytes_once,
            "_write_pointer_atomic": self._pointer,
        })
        with self.section():
            return adapted(root, snapshot, manifest, **kwargs)

    def _pointer(self, store_dir, snapshot_id):
        from skill3d.skills.v11_library import _json_bytes
        self.write_bytes(Path(store_dir) / "active_snapshot.json",
                         _json_bytes({"snapshot_id": snapshot_id}))

    @contextmanager
    def _publication_lock(self, store_dir):
        from skill3d.skills.promote_atomic import PublishLockedError, _now_iso
        lock = Path(store_dir) / ".publish.lock"
        with self.section() as io:
            try:
                self.write_bytes(lock, json.dumps(
                    {"pid": os.getpid(), "at": _now_iso()}).encode(), exclusive=True)
            except FileExistsError as exc:
                raise PublishLockedError("library publication requires lock recovery") from exc
            try:
                yield
            finally:
                io.unlink(self.relative(lock))

    def publish_candidate(self, root, candidate, **kwargs):
        from skill3d.skills import v11_library as library
        from skill3d.skills import promote_atomic
        # publish_v11_candidate imports its lock locally. Override that one
        # import in a private function namespace; all validators stay original.
        original_import = __import__

        def local_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "skill3d.skills.promote_atomic" and level == 0:
                return SimpleNamespace(
                    publish_lock=self._publication_lock,
                    read_active_snapshot=promote_atomic.read_active_snapshot)
            return original_import(name, globals, locals, fromlist, level)

        builtins = dict(vars(__import__("builtins")))
        builtins["__import__"] = local_import
        adapted = _function_copy(library.publish_v11_candidate, {
            "__builtins__": builtins,
            "_write_create_once": self.write_bytes_once,
            "_write_pointer_atomic": self._pointer,
            "write_v11_snapshot": self.write_snapshot,
        })
        return adapted(root, candidate, **kwargs)

    def execute_episode_pool(self, runner, pending, accept):
        from skill3d.evaluation import episode_pool_v11 as pool
        self.write_bytes_once(runner.root / "episode_pool.lock", b"")
        locked = _function_copy(pool._execute_locked, {
            "_write_create_once": self.write_json_once,
            "_write_json_atomic": self.write_json,
        })
        # Keep the original spawn target picklable. Only coordinator writers
        # are copied; children use the explicitly serialized runner adapter.
        execute = _function_copy(pool.execute_episode_pool, {"_execute_locked": locked})
        return execute(runner, pending, accept)

    def _checkpoint_and_archive(self, best):
        try:
            workspace = self._workspace
            lock = workspace.cache.root / ("framework-archive-" + workspace.key + ".lock")
            # One saved-identity transport at a time across all owners. This is
            # not the workspace lease and does not block episode writers.
            with _flock(lock):
                self._archive_completed_history()
                workspace.archive_pending()
                token = uuid.uuid4().hex
                workspace.checkpoint(
                    "fw-" + token, latest=True, best=best,
                    route=ArchiveRoute(self.job, "batch-" + token, self.attempt))
                return workspace.archive_pending()
        except Exception as exc:
            raise FrameworkStorageRefused(type(exc).__name__) from exc

    def after_batch(self, *, best=False):
        """One outstanding checkpoint/archive per process; saved intents retry."""
        self._process()
        with self._background_lock:
            if self._future is not None:
                self._future.result()
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="framework-state-archive")
            self._future = self._executor.submit(self._checkpoint_and_archive, best)

    def finish(self):
        self._process()
        with self._background_lock:
            try:
                if self._future is not None:
                    self._future.result()
            finally:
                if self._executor is not None:
                    self._executor.shutdown(wait=True)
                self._executor = self._future = None


class WorkspaceTraceStore:
    """The online TraceStore protocol, with exact JSONL encoding and bounded IO."""

    def __init__(self, io, root):
        self.io, self.root = io, Path(root)
        io.mkdir(self.root)

    def append(self, topic, record):
        from pydantic import BaseModel
        safe = "".join(c for c in topic if c.isalnum() or c in "-_")
        line = record.model_dump_json() if isinstance(record, BaseModel) else json.dumps(
            record, default=str)
        self.io.append_bytes(self.root / (safe + ".jsonl"), (line + "\n").encode("utf-8"))

    def append_episode(self, record):
        self.append("episode_trace", record)

    def claim_run(self, run_id):
        if not run_id or any(c not in
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
            raise ValueError("invalid trace run_id")
        try:
            self.io.write_bytes(self.root / ".runs" / (run_id + ".json"),
                                json.dumps({"run_id": run_id}).encode(), exclusive=True)
        except FileExistsError as exc:
            raise ValueError("trace run_id already exists; use a new run") from exc

    def bind(self, **identity):
        from skill3d.trace.store import BoundTraceStore
        return BoundTraceStore(self, identity)


from skill3d.synthesis.request_governor import RequestGovernor  # noqa: E402


class WorkspaceRequestGovernor(RequestGovernor):
    """Original request policy, with admitted ledger/permit writes before calls."""

    def __init__(self, io, root, config):
        self.io = io
        self._permits = threading.local()
        root = Path(root).resolve()
        io.mkdir(root)
        io.write_bytes_once(root / "state.lock", b"")
        for index in range(config.max_inflight):
            path = root / f"slot-{index:03d}.lock"
            # Existing numeric permit contents and their locked inodes survive.
            with io.section() as active:
                if not active._path(io.relative(path)).exists():
                    io.write_bytes_once(path, b"")
        initializer = _function_copy(RequestGovernor.__init__, {"_atomic": self._atomic})
        initializer(self, root, config)

    def _atomic(self, path, value):
        data = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        self.io.write_bytes(path, data)

    @contextmanager
    def _locked(self):
        # This order is used for every ledger writer. No ledger lock is ever
        # held while waiting for workspace EX, or during a model request.
        with self.io.section(), super()._locked():
            yield

    def _slot(self):
        slot, permit = super()._slot()
        if permit is not None:
            try:
                active = self.io._local.io
                digits = len(str(self.config.max_requests))
                if digits > self.io.max_file_bytes:
                    raise FrameworkStorageRefused("permit file ceiling exceeded")
                active._budget.check_growth(
                    logical_bytes=digits, allocation_bytes=_allocation_unit(self.io.root))
            except BaseException:
                permit.close()
                raise
            self._permits.stream = permit
            if active._budget.atomic_writes:
                permit = _AccountedPermit(
                    permit, active._budget, self.root / f"slot-{slot:03d}.lock")
                self._permits.stream = permit
        return slot, permit

    @contextmanager
    def request(self, category, context):
        original = _function_copy(inspect.unwrap(RequestGovernor.request),
                                  {"_atomic": self._atomic})
        previous = getattr(self._permits, "stream", None)
        self._permits.stream = None
        try:
            with contextmanager(original)(self, category, context) as receipt:
                yield receipt
        finally:
            # The legacy generator has not yet entered its finally when
            # initial ledger admission fails. Release that permit as well.
            permit = getattr(self._permits, "stream", None)
            try:
                if permit is not None:
                    permit.close()
            finally:
                self._permits.stream = previous


class _AccountedPermit:
    """Track and fsync the original numeric governor permit in its short IO lease."""

    def __init__(self, stream, budget, path):
        self._stream, self._budget, self._path = stream, budget, path

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def truncate(self, *args):
        with self._budget._accounted((self._path,)):
            return self._stream.truncate(*args)

    def write(self, data):
        info = os.fstat(self._stream.fileno())
        unit = _allocation_unit(self._budget.path)
        allocated = max(info.st_size, getattr(info, "st_blocks", 0) * 512)
        growth = max(0, _rounded(info.st_size + len(data), unit) - allocated)
        with self._budget._accounted(
                (self._path,), logical_bytes=len(data), allocation_bytes=growth):
            written = self._stream.write(data)
            self._stream.flush()
            os.fsync(self._stream.fileno())
            return written

    def flush(self):
        self._stream.flush()
        os.fsync(self._stream.fileno())
