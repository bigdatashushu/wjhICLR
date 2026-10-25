#!/usr/bin/env python3
"""Pinned transport-only v3: one sealed SSD/HDD spool, original publication.

The parent operator owns the all-eight-running barrier and old-server retirement.
This process only warms read-only, waits for an exact activation, and serves the
unchanged frozen routes through the reviewed v2 input/interim hooks.
"""
from __future__ import annotations

import argparse
import base64
import errno
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import threading
import time
import types
import uuid


BASE_ADAPTER_SHA256 = "0f8bcfc192fbac00faf69a91c564a43bb8e9281bc1a69be851fb5c962f79f787"
METHOD_SHA256 = "cbe98d6f222dc2e2d3d9c2dec2be74367d9444da4449ec20a1fec454e3afb12e"
SSD_ROOT = Path("/var/tmp/harness3d-r04-ssd-publication-v3")
MAX_ARCHIVE_BYTES = 20 * 1024**3
SSD_BUDGET_BYTES = 16 * 1024**3
SYSTEM_RESERVE_BYTES = 10 * 1024**3
PATHS = {
    "spool": ".publication_spool_v3",
    "events": "receipts/publication_fifo_events_v3.jsonl",
    "prepared": "receipts/publication_fifo_prepared_v3.json",
    "activation": "receipts/publication_fifo_activate_v3.json",
    "startup": "receipts/coordinator_fifo_startup_v3.json",
}
TEMP_NAMES = (".uploads", ".staging", ".publication_spool_v1",
              ".publication_spool_v2", PATHS["spool"])
BLOCK_BYTES = 4 * 1024**2


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_base_adapter(path):
    """Hash the exact bytes executed, without writing beside the original."""
    path = Path(path).absolute()
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == BASE_ADAPTER_SHA256,
            "base adapter SHA256 mismatch")
    module = types.ModuleType("_harness3d_pinned_publication_v2")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def require_empty(path):
    path = Path(path)
    if not os.path.lexists(path):
        return
    require(not path.is_symlink() and path.is_dir() and not any(path.iterdir()),
            "handover/startup requires empty directory: " + str(path))


class Evidence:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def write(self, event, **fields):
        raw = json.dumps({"schema": "harness3d-publication-fifo-event/3",
                          "at": time.time(), "event": event, **fields},
                         sort_keys=True, allow_nan=False).encode() + b"\n"
        with self.lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT |
                         os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "ab") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())

    def observe(self, event, **fields):
        try:
            self.write(event, **fields)
        except OSError as exc:
            print("v3 observation evidence unavailable: " + type(exc).__name__,
                  file=sys.stderr, flush=True)


class PrivateDirectory:
    """Anchor spool operations to an owned private directory, never a link."""

    def __init__(self, path):
        self.path = Path(path).absolute()
        for parent in reversed(self.path.parents):
            require(not parent.is_symlink(), "unsafe spool ancestor: " + str(parent))
        try:
            self.path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        require(not self.path.is_symlink(), "unsafe spool directory link")
        require(self.path.is_dir(), "spool path is existing non-directory evidence")
        self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(self.fd)
            require(info.st_uid == os.geteuid() and stat.S_ISDIR(info.st_mode)
                    and info.st_mode & 0o077 == 0, "spool directory must be owner/private")
            require(not os.listdir(self.fd), "previous spool needs explicit review: " + str(path))
            self.identity = (info.st_dev, info.st_ino)
        except BaseException:
            os.close(self.fd)
            raise

    def verify(self):
        info = self.path.lstat()
        require(stat.S_ISDIR(info.st_mode) and
                (info.st_dev, info.st_ino) == self.identity and
                info.st_uid == os.geteuid() and info.st_mode & 0o077 == 0,
                "spool directory identity changed")
        for parent in self.path.parents:
            require(not parent.is_symlink(), "unsafe spool ancestor")

    def close(self):
        os.close(self.fd)


class Spool:
    def __init__(self, manager, directory, name, fd, size, backend, fallback_reason):
        self.manager, self.directory, self.name, self.fd = manager, directory, name, fd
        self.path = directory.path / name
        self.size, self.backend = size, backend
        self.fallback_reason = fallback_reason
        self.bytes_write_count = 0
        self.sealed = None
        self.archive_sha256 = None

    @staticmethod
    def identity(info):
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
                info.st_ctime_ns, info.st_nlink, info.st_mode)

    def check_identity(self):
        self.directory.verify()
        opened = os.fstat(self.fd)
        named = os.stat(self.name, dir_fd=self.directory.fd, follow_symlinks=False)
        require(stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1 and
                self.identity(opened) == self.sealed == self.identity(named),
                "publication spool file identity changed")

    def receive(self, stream, expected_sha):
        remaining = self.size
        digest = hashlib.sha256()
        while remaining:
            block = stream.read(min(BLOCK_BYTES, remaining))
            if not block or len(block) > remaining:
                raise ValueError("incomplete publication stream")
            view = memoryview(block)
            while view:
                written = os.write(self.fd, view)
                if written <= 0:
                    raise OSError("short publication spool write")
                self.bytes_write_count += written
                view = view[written:]
            digest.update(block)
            remaining -= len(block)
        self.archive_sha256 = digest.hexdigest()
        os.fchmod(self.fd, 0o400)
        os.fsync(self.fd)
        os.fsync(self.directory.fd)
        self.sealed = self.identity(os.fstat(self.fd))
        if self.bytes_write_count != self.size or self.archive_sha256 != expected_sha:
            raise ValueError("publication archive digest mismatch")
        self.check_identity()

    def verify_full(self):
        """Second full read, after FIFO admission; no second archive write."""
        self.check_identity()
        fd = os.open(self.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.directory.fd)
        try:
            require(self.identity(os.fstat(fd)) == self.sealed,
                    "publication spool reopened identity changed")
            digest, total = hashlib.sha256(), 0
            while block := os.read(fd, BLOCK_BYTES):
                total += len(block)
                digest.update(block)
            require(total == self.size and digest.hexdigest() == self.archive_sha256,
                    "publication spool second hash mismatch")
            require(self.identity(os.fstat(fd)) == self.sealed,
                    "publication spool changed during second hash")
            self.check_identity()
            return digest.hexdigest()
        finally:
            os.close(fd)

    def cleanup(self):
        # Do not delete a replacement path or release its conservative budget.
        try:
            info = os.stat(self.name, dir_fd=self.directory.fd, follow_symlinks=False)
            opened = os.fstat(self.fd)
            require((info.st_dev, info.st_ino) == (opened.st_dev, opened.st_ino),
                    "refusing to clean replaced spool evidence")
            os.unlink(self.name, dir_fd=self.directory.fd)
            os.fsync(self.directory.fd)
            self.manager.release(self)
        finally:
            os.close(self.fd)


class SpoolManager:
    """Declared-byte reservations cover all writers until publication cleanup."""

    def __init__(self, root, ssd_root, reserve):
        self.root, self.reserve = Path(root), reserve
        self.lock = threading.Lock()
        self.ssd_reserved = 0
        self.root_reserved = 0
        self.jobs = 0
        self.hdd = PrivateDirectory(self.root / PATHS["spool"])
        self.ssd = None
        self.ssd_unavailable = None
        try:
            self.ssd = PrivateDirectory(ssd_root)
        except OSError as exc:
            self.ssd_unavailable = type(exc).__name__
        except BaseException:
            self.hdd.close()
            raise

    def allocate(self, size):
        require(type(size) is int and 0 < size <= MAX_ARCHIVE_BYTES,
                "archive size outside limit")
        with self.lock:
            self.hdd.verify()
            require(self.jobs < 8, "spool job capacity exceeded")
            if shutil.disk_usage(self.root).free < 2 * size + self.reserve + self.root_reserved:
                raise ValueError("publication reserve would be crossed")
            directory, reason = self.ssd, self.ssd_unavailable
            if directory is not None:
                directory.verify()
                if self.ssd_reserved + size > SSD_BUDGET_BYTES:
                    directory, reason = None, "ssd_declared_budget"
                else:
                    try:
                        free = shutil.disk_usage(directory.path).free
                    except OSError as exc:
                        directory, reason = None, type(exc).__name__
                    else:
                        if free < SYSTEM_RESERVE_BYTES + size + self.ssd_reserved:
                            directory, reason = None, "ssd_system_reserve"
            if directory is not None:
                try:
                    spool = self._create(directory, size, "ssd", None)
                except OSError as exc:
                    directory, reason = None, "ssd_allocate_" + str(exc.errno)
            if directory is None:
                spool = self._create(self.hdd, size, "hdd", reason)
            self.jobs += 1
            self.root_reserved += 2 * size
            if spool.backend == "ssd":
                self.ssd_reserved += size
            return spool

    def _create(self, directory, size, backend, reason):
        name = uuid.uuid4().hex + ".tar"
        fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory.fd)
        try:
            if backend == "ssd" and sys.platform.startswith("linux") and hasattr(os, "posix_fallocate"):
                try:
                    os.posix_fallocate(fd, 0, size)
                except OSError as exc:
                    if exc.errno not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
                        raise
            return Spool(self, directory, name, fd, size, backend, reason)
        except BaseException:
            try:
                os.close(fd)
                os.unlink(name, dir_fd=directory.fd)
                os.fsync(directory.fd)
            except Exception as exc:
                # allocate holds the lock. Retain a failed job's reservation
                # rather than falling back with unaccounted SSD evidence.
                self.jobs += 1
                self.root_reserved += 2 * size
                if backend == "ssd":
                    self.ssd_reserved += size
                raise RuntimeError("spool allocation cleanup failed; reservation retained") from exc
            raise

    def release(self, spool):
        with self.lock:
            self.jobs -= 1
            self.root_reserved -= 2 * spool.size
            if spool.backend == "ssd":
                self.ssd_reserved -= spool.size

    def close(self):
        self.hdd.close()
        if self.ssd is not None:
            self.ssd.close()


def adapt_handler(handler, coordinator, evidence, *, base_adapter, ssd_root,
                  interim_seconds=15, spool_manager=None):
    manager = spool_manager or SpoolManager(coordinator.root, ssd_root, coordinator.reserve)
    # Execute the reviewed input/phase hooks with v3 paths, without rebinding the
    # original module or invoking its receive_archive (which writes .uploads).
    hooks = types.FunctionType(base_adapter.adapt_handler.__code__,
                               {**base_adapter.__dict__, "PATHS": PATHS})
    hooks.__kwdefaults__ = base_adapter.adapt_handler.__kwdefaults__
    try:
        input_handler = hooks(handler, coordinator, evidence, interim_seconds=interim_seconds)
    except BaseException:
        if spool_manager is None:
            manager.close()
        raise
    spool_slots = threading.BoundedSemaphore(8)
    frozen = sys.modules[coordinator.__class__.__module__]
    original_publish = coordinator.publish

    class AdaptedHandler(input_handler):
        def do_POST(self):
            if self.path != "/publish":
                return super().do_POST()
            try:
                return super().do_POST()
            finally:
                state = getattr(self, "_publication_state", None)
                if state is not None:
                    state.close()
                    evidence.observe("publication_finished", **state.timing(),
                                     finished_at=time.time(),
                                     finished_seconds=time.monotonic() - state.started,
                                     ack=state.ack)
                    self._publication_state = None

        def receive_archive(self):
            state = base_adapter.InterimResponse(self, interim_seconds)
            self._publication_state = state
            state.fields["bytes_write_count"] = 0
            state.start()
            spool = None
            admitted = False
            spool_admitted = False
            try:
                # Exact original compact/full lease construction and checkpoint
                # binding; never accept client metadata in place of stored state.
                raw = base64.b64decode(self.headers["X-Lease-JSON"], validate=True)
                wire = json.loads(raw)
                if wire.get("schema") == "harness3d-lease-header/1":
                    job_key = wire["job_key"]
                    if job_key not in coordinator.specs:
                        raise frozen.QueueError("publication is outside frozen scope")
                    job = coordinator.queue.get_job(job_key)
                    lease = frozen.Lease(
                        coordinator.specs[job_key], frozen.WorkerIdentity(**wire["worker"]),
                        wire["token"], wire["expires_at"], wire["attempt"], "publish",
                        frozen.canonical_json(job["output_metadata"]))
                else:
                    lease = frozen.Lease.from_dict(wire)
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_ARCHIVE_BYTES:
                    raise frozen.QueueError("archive size outside limit")
                declared_sha = self.headers["X-Archive-SHA256"]
                if (len(declared_sha) != 64 or
                        any(c not in "0123456789abcdef" for c in declared_sha)):
                    raise frozen.QueueError("noncanonical archive SHA256")
                if lease.job_key not in coordinator.specs:
                    raise frozen.QueueError("publication is outside frozen scope")
                state.fields.update(job_key=lease.job_key, attempt=lease.attempt,
                                    lease_header_sha256=hashlib.sha256(raw).hexdigest(),
                                    archive_bytes=size, declared_archive_sha256=declared_sha)
                spool_slots.acquire()
                spool_admitted = True
                spool = manager.allocate(size)
                state.fields.update(archive_path=str(spool.path), backend=spool.backend,
                                    fallback_reason=spool.fallback_reason,
                                    receive_started_at=time.time())
                receive_started = time.monotonic()
                evidence.write("spool_started", **state.fields)
                try:
                    spool.receive(self.rfile, declared_sha)
                finally:
                    state.fields.update(receive_finished_at=time.time(),
                                        receive_seconds=time.monotonic() - receive_started,
                                        bytes_write_count=spool.bytes_write_count,
                                        archive_sha256=spool.archive_sha256)
                state.fields["queue_entered_at"] = time.time()
                evidence.write("spool_sealed", **state.fields)
                queued = time.monotonic()
                coordinator.publish_slots.acquire()
                admitted = True
                state.fields.update(queue_admitted_at=time.time(),
                                    queue_seconds=time.monotonic() - queued)
                verify_started = time.monotonic()
                state.fields["second_archive_sha256"] = spool.verify_full()
                state.fields["second_hash_seconds"] = time.monotonic() - verify_started
                spool.check_identity()
                state.fields["publish_started_at"] = time.time()
                publish_started = time.monotonic()
                try:
                    result = original_publish(lease, spool.path, spool.archive_sha256, size)
                    spool.check_identity()
                finally:
                    state.fields.update(publish_finished_at=time.time(),
                                        publish_seconds=time.monotonic() - publish_started)
            except Exception as exc:
                state.fields["error_type"] = type(exc).__name__
                if isinstance(exc, (frozen.QueueError, ValueError, KeyError, OSError)):
                    raise
                raise frozen.QueueError("publication transport invariant rejected") from exc
            finally:
                if admitted:
                    coordinator.publish_slots.release()
                try:
                    if spool is not None:
                        state.fields["bytes_write_count"] = spool.bytes_write_count
                        try:
                            spool.cleanup()
                        except Exception as exc:
                            state.fields["cleanup_error_type"] = type(exc).__name__
                            state.fields["error_type"] = type(exc).__name__
                            raise frozen.QueueError("publication spool cleanup failed") from exc
                        state.fields["spool_cleaned"] = True
                finally:
                    if spool_admitted:
                        spool_slots.release()
            self.reply(200, result)

    AdaptedHandler.spool_manager = manager
    return AdaptedHandler


def validate_activation(command, root, *, adapter_sha256, warm_pid, ssd_root):
    require(command["status"] == "old_coordinator_exited", "old coordinator not retired")
    require(command["method_config_sha256"] == METHOD_SHA256, "activation method mismatch")
    require(command["adapter_sha256"] == adapter_sha256, "activation adapter mismatch")
    require(command["base_adapter_sha256"] == BASE_ADAPTER_SHA256, "activation base adapter mismatch")
    require(type(command["warm_pid"]) is int and command["warm_pid"] == warm_pid,
            "activation warm PID mismatch")
    require(command["ssd_root"] == str(ssd_root), "activation SSD root mismatch")
    old_pid = command["old_pid"]
    require(type(old_pid) is int and old_pid > 0 and old_pid != warm_pid, "invalid old PID")
    require(Path("/proc/self/stat").is_file(), "activation requires Linux process evidence")
    require(not os.path.lexists(Path("/proc") / str(old_pid)), "exact old PID still exists")
    for name in TEMP_NAMES:
        require_empty(Path(root) / name)
    require_empty(ssd_root)


def verify_frozen_imports(root, method):
    """Verify every old method code pin, then every loaded frozen dependency."""
    source = root / "source_frozen"
    pins = method["code_sha256"]
    require(isinstance(pins, dict) and pins, "missing frozen code pins")
    for name, expected in pins.items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts,
                "unsafe frozen code path")
        require(hashlib.sha256((source / name).read_bytes()).hexdigest() == expected,
                "frozen code hash mismatch: " + name)
    sys.path[:0] = [str(source / "scripts"), str(source / "src")]
    module = importlib.import_module("cluster_coordinator")
    require(Path(module.__file__).resolve() == source / "scripts/cluster_coordinator.py",
            "coordinator import is not from source_frozen")
    queue = sys.modules[module.SQLiteJobQueue.__module__]
    require(Path(queue.__file__).resolve() == source / "src/skill3d/cluster/queue.py",
            "queue import is not from source_frozen")
    for name, loaded in list(sys.modules.items()):
        if name == "cluster_coordinator" or name == "skill3d" or name.startswith("skill3d."):
            path = Path(loaded.__file__).resolve()
            require(path.is_relative_to(source), "dependency import outside source_frozen: " + name)
            relative = str(path.relative_to(source))
            require(relative in pins and hashlib.sha256(path.read_bytes()).hexdigest() == pins[relative],
                    "unbound frozen dependency: " + name)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-adapter", type=Path, required=True)
    parser.add_argument("--ssd-root", type=Path, required=True)
    parser.add_argument("--expected-method-sha256", required=True, choices=[METHOD_SHA256])
    parser.add_argument("--port", type=int, default=27049)
    args = parser.parse_args(argv)
    require(args.ssd_root == SSD_ROOT, "SSD root must be the explicit v3 system-disk path")
    base = load_base_adapter(args.base_adapter)
    root = args.root.resolve()
    raw = (root / "method_config.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == METHOD_SHA256, "frozen method SHA256 mismatch")
    method = json.loads(raw)
    frozen = verify_frozen_imports(root, method)
    for name in ("prepared", "activation", "startup", "events"):
        require(not os.path.lexists(root / PATHS[name]), "v3 path already exists: " + PATHS[name])
    # Old spools may be active during warm; only new spool evidence must be empty.
    manager = SpoolManager(root, args.ssd_root, 160 * 1024**3)
    try:
        jobs = root / "coordinator_jobs.json"
        jobs_sha = hashlib.sha256(jobs.read_bytes()).hexdigest()
        db_stat = (root / "state/jobs.sqlite3").stat()
        coordinator = base.build_warm_coordinator(frozen, root, jobs)
        require(len(coordinator.specs) == 230, "warm admission requires the full 230-job scope")
        adapter_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        interface = {
            "method_config_sha256": METHOD_SHA256, "adapter_sha256": adapter_sha,
            "base_adapter_sha256": BASE_ADAPTER_SHA256, "ssd_root": str(args.ssd_root),
            "publication_capacity": 4, "spool_capacity": 8,
            "ssd_budget_bytes": SSD_BUDGET_BYTES, "system_reserve_bytes": SYSTEM_RESERVE_BYTES,
            "single_spool": True, "second_full_hash": True, "uploads_written": False,
            "input_interim_seconds": 15, "input_keepalive_phase": "before_final_response_only",
            "input_digest_cache": False, "frozen_validators_delegated": True,
            "parent_owns_all8_running_barrier": True, "model_requests": 0, "label_access": False,
        }
        frozen.atomic_json(root / PATHS["prepared"], {
            **interface, "schema": "harness3d-publication-fifo-prepared/3",
            "status": "prepared", "at": time.time(), "pid": os.getpid(),
            "jobs_sha256": jobs_sha, "code_files": len(method["code_sha256"]),
            "warm_admission": coordinator.warm_admission,
            "ssd_available": manager.ssd is not None,
        })
        activation = root / PATHS["activation"]
        while not activation.exists():
            time.sleep(0.5)
        validate_activation(json.loads(activation.read_text()), root,
                            adapter_sha256=adapter_sha, warm_pid=os.getpid(),
                            ssd_root=args.ssd_root)
        require(hashlib.sha256(jobs.read_bytes()).hexdigest() == jobs_sha, "jobs changed after warm")
        current = (root / "state/jobs.sqlite3").stat()
        require((db_stat.st_dev, db_stat.st_ino) == (current.st_dev, current.st_ino),
                "queue database replaced after warm")
        require(hashlib.sha256((root / "method_config.json").read_bytes()).hexdigest() == METHOD_SHA256,
                "method changed after warm")
        verify_frozen_imports(root, method)
        require(hashlib.sha256(args.base_adapter.read_bytes()).hexdigest() == BASE_ADAPTER_SHA256,
                "base adapter changed after warm")
        evidence = Evidence(root / PATHS["events"])
        original_server, original_atomic = frozen.ThreadingHTTPServer, frozen.atomic_json

        class AdaptedServer(original_server):
            def __init__(self, address, handler):
                super().__init__(address, adapt_handler(
                    handler, coordinator, evidence, base_adapter=base,
                    ssd_root=args.ssd_root, spool_manager=manager))

        def startup_json(path, value):
            if Path(path) == root / "receipts/coordinator_startup.json":
                path = root / PATHS["startup"]
                require(not os.path.lexists(path), "v3 startup receipt already exists")
                value = {**value, **interface,
                         "schema": "harness3d-publication-fifo-coordinator-startup/3",
                         "original_startup_preserved": True}
            original_atomic(path, value)

        frozen.ThreadingHTTPServer, frozen.atomic_json = AdaptedServer, startup_json
        try:
            frozen.serve(coordinator, "0.0.0.0", args.port)
        finally:
            frozen.ThreadingHTTPServer, frozen.atomic_json = original_server, original_atomic
    finally:
        manager.close()


if __name__ == "__main__":
    main()
