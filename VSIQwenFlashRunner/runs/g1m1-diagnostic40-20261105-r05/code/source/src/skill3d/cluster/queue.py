"""Durable single-master SQLite queue. Workers use HTTP, never a shared DB file."""

from __future__ import annotations

import contextlib
import json
import math
import secrets
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .identity import FrozenIdentity, IdentityMismatch, QueueError, canonical_json, digest_json, safe_text
from .publication import (
    PublicationConflict, PublicationError, inspect_receipt, validate_artifacts,
)

ACTIVE = ("leased", "running", "publishing")
TERMINAL = ("complete", "skipped", "failed")
PIPELINE_STAGES = ("reconstruction", "scale", "binding")
STAGES = (*PIPELINE_STAGES, "gpu_aggregate")


class EnqueueConflict(QueueError):
    pass


class LeaseLost(QueueError):
    pass


class InvalidTransition(QueueError):
    pass


class UnsafeDatabasePath(QueueError):
    pass


@dataclass(frozen=True)
class WorkerIdentity:
    worker_id: str
    host: str
    gpu_id: str

    def __post_init__(self) -> None:
        for field in ("worker_id", "host", "gpu_id"):
            safe_text(getattr(self, field), field=field, limit=256)

    def to_dict(self) -> dict[str, str]:
        return {"worker_id": self.worker_id, "host": self.host, "gpu_id": self.gpu_id}


@dataclass(frozen=True)
class JobSpec:
    job_key: str
    completed_key: str
    identity: FrozenIdentity
    stage: str
    dependencies: tuple[str, ...] = ()
    max_attempts: int = 3
    priority: int = 0
    dependency_policy: str = "complete"

    def __post_init__(self) -> None:
        safe_text(self.job_key, field="job_key", limit=512)
        safe_text(self.completed_key, field="completed_key", limit=512)
        if self.stage not in STAGES:
            raise QueueError(f"stage must be one of {STAGES}")
        if not isinstance(self.identity, FrozenIdentity):
            raise IdentityMismatch("identity must be FrozenIdentity")
        self.identity.__post_init__()
        if (not isinstance(self.dependencies, (tuple, list)) or
                any(not isinstance(dep, str) for dep in self.dependencies)):
            raise QueueError("dependencies must be a sequence of job keys")
        object.__setattr__(self, "dependencies", tuple(sorted(self.dependencies)))
        for dep in self.dependencies:
            safe_text(dep, field="dependency", limit=512)
        if self.job_key in self.dependencies or len(set(self.dependencies)) != len(self.dependencies):
            raise QueueError("self or duplicate dependencies")
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise QueueError("max_attempts must be a positive integer")
        if type(self.priority) is not int or not -(2**31) <= self.priority < 2**31:
            raise QueueError("priority must be a 32-bit integer")
        if self.dependency_policy not in ("complete", "complete_or_skipped"):
            raise QueueError("invalid dependency policy")
        if self.stage == "reconstruction" and self.dependencies:
            raise QueueError("reconstruction must not have dependencies")
        if self.stage in ("scale", "binding") and not self.dependencies:
            raise QueueError("scale/binding require upstream dependencies")

    def to_dict(self) -> dict[str, Any]:
        return {"job_key": self.job_key, "completed_key": self.completed_key,
                "identity": self.identity.to_dict(), "stage": self.stage,
                "dependencies": list(self.dependencies), "max_attempts": self.max_attempts,
                "priority": self.priority, "dependency_policy": self.dependency_policy}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JobSpec:
        fields = {"job_key", "completed_key", "identity", "stage", "dependencies",
                  "max_attempts", "priority", "dependency_policy"}
        if (not isinstance(value, Mapping) or set(value) != fields or
                not isinstance(value["dependencies"], list)):
            raise QueueError("invalid job spec envelope")
        return cls(value["job_key"], value["completed_key"],
                   FrozenIdentity.from_dict(value["identity"]), value["stage"],
                   tuple(value["dependencies"]), value["max_attempts"], value["priority"],
                   value["dependency_policy"])


@dataclass(frozen=True)
class Lease:
    spec: JobSpec
    worker: WorkerIdentity
    token: str
    expires_at: float
    attempt: int
    phase: str
    output_metadata_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.spec, JobSpec) or not isinstance(self.worker, WorkerIdentity):
            raise QueueError("lease requires a JobSpec and WorkerIdentity")
        if (not isinstance(self.token, str) or len(self.token) != 64 or
                any(c not in "0123456789abcdef" for c in self.token)):
            raise QueueError("invalid lease fencing token")
        if (isinstance(self.expires_at, bool) or not isinstance(self.expires_at, (int, float))
                or not math.isfinite(self.expires_at) or self.expires_at < 0):
            raise QueueError("invalid lease expiry")
        if type(self.attempt) is not int or not 0 <= self.attempt <= self.spec.max_attempts:
            raise QueueError("invalid lease attempt number")
        if self.phase not in ("leased", "running", "publish"):
            raise QueueError("invalid lease phase")
        try:
            metadata = json.loads(self.output_metadata_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise QueueError("invalid output metadata JSON") from exc
        if not isinstance(metadata, dict) or canonical_json(metadata) != self.output_metadata_json:
            raise QueueError("output metadata must be a canonical JSON object")

    @property
    def job_key(self) -> str:
        return self.spec.job_key

    def to_dict(self) -> dict[str, Any]:
        return {"spec": self.spec.to_dict(), "worker": self.worker.to_dict(),
                "token": self.token, "expires_at": self.expires_at,
                "attempt": self.attempt, "phase": self.phase,
                "output_metadata": json.loads(self.output_metadata_json)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Lease:
        fields = {"spec", "worker", "token", "expires_at", "attempt", "phase", "output_metadata"}
        if (not isinstance(value, Mapping) or set(value) != fields or
                not isinstance(value["worker"], dict) or
                set(value["worker"]) != {"worker_id", "host", "gpu_id"}):
            raise QueueError("invalid lease envelope")
        return cls(JobSpec.from_dict(value["spec"]), WorkerIdentity(**value["worker"]),
                   value["token"], value["expires_at"], value["attempt"], value["phase"],
                   canonical_json(value["output_metadata"]))


@dataclass(frozen=True)
class EnqueueResult:
    job_key: str
    created: bool
    status: str


@dataclass(frozen=True)
class CompletionResult:
    job_key: str
    completed_key: str
    receipt_hash: str
    fingerprint: str
    duplicate: bool
    status: str = "complete"


_SCHEMA = """
CREATE TABLE jobs (
    job_key TEXT PRIMARY KEY,
    completed_key TEXT NOT NULL UNIQUE,
    spec_json TEXT NOT NULL,
    spec_hash TEXT NOT NULL,
    stage TEXT NOT NULL CHECK(stage IN ('reconstruction','scale','binding','gpu_aggregate')),
    identity_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('queued','leased','running','publishing',
                                          'complete','skipped','failed')),
    dependency_policy TEXT NOT NULL,
    priority INTEGER NOT NULL,
    max_attempts INTEGER NOT NULL CHECK(max_attempts > 0),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0 AND attempts <= max_attempts),
    claims INTEGER NOT NULL DEFAULT 0,
    prefetch_failures INTEGER NOT NULL DEFAULT 0,
    return_failures INTEGER NOT NULL DEFAULT 0,
    resume_phase TEXT NOT NULL DEFAULT 'prefetch' CHECK(resume_phase IN ('prefetch','publish')),
    output_metadata_json TEXT NOT NULL DEFAULT '{}',
    available_at REAL NOT NULL,
    lease_token TEXT,
    lease_until REAL,
    worker_id TEXT,
    host TEXT,
    gpu_id TEXT,
    error_code TEXT,
    error_detail TEXT,
    skip_reason TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    CHECK(status != 'failed' OR (error_code IS NOT NULL AND error_detail IS NOT NULL
                                AND length(error_code) > 0 AND length(error_detail) > 0)),
    CHECK(status != 'skipped' OR (skip_reason IS NOT NULL AND length(skip_reason) > 0)),
    CHECK((status IN ('leased','running','publishing') AND lease_token IS NOT NULL
           AND lease_until IS NOT NULL AND worker_id IS NOT NULL AND host IS NOT NULL
           AND gpu_id IS NOT NULL) OR
          (status NOT IN ('leased','running','publishing') AND lease_token IS NULL
           AND lease_until IS NULL))
);
CREATE TABLE dependencies (
    job_key TEXT NOT NULL REFERENCES jobs(job_key),
    depends_on TEXT NOT NULL REFERENCES jobs(job_key),
    PRIMARY KEY(job_key, depends_on),
    CHECK(job_key != depends_on)
);
CREATE TABLE completions (
    completed_key TEXT PRIMARY KEY REFERENCES jobs(completed_key),
    job_key TEXT NOT NULL UNIQUE REFERENCES jobs(job_key),
    identity_hash TEXT NOT NULL,
    stage TEXT NOT NULL,
    receipt_hash TEXT NOT NULL,
    receipt_json TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    accepted_lease_token TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    host TEXT NOT NULL,
    gpu_id TEXT NOT NULL,
    completed_at REAL NOT NULL
);
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_key TEXT NOT NULL REFERENCES jobs(job_key),
    event TEXT NOT NULL,
    at REAL NOT NULL,
    attempt INTEGER NOT NULL,
    worker_id TEXT,
    host TEXT,
    gpu_id TEXT,
    detail_json TEXT NOT NULL
);
CREATE INDEX ready_jobs ON jobs(status, available_at, priority, created_at);
CREATE INDEX expiry_jobs ON jobs(status, lease_until);
CREATE INDEX dependent_jobs ON dependencies(depends_on);
CREATE INDEX job_events ON events(job_key, id);
CREATE TRIGGER immutable_job_identity BEFORE UPDATE OF
    job_key, completed_key, spec_json, spec_hash, stage, identity_hash,
    dependency_policy, priority, max_attempts ON jobs
BEGIN SELECT RAISE(ABORT, 'immutable job identity'); END;
CREATE TRIGGER immutable_completion_update BEFORE UPDATE ON completions
BEGIN SELECT RAISE(ABORT, 'immutable completion'); END;
CREATE TRIGGER immutable_completion_delete BEFORE DELETE ON completions
BEGIN SELECT RAISE(ABORT, 'immutable completion'); END;
CREATE TRIGGER immutable_dependency_update BEFORE UPDATE ON dependencies
BEGIN SELECT RAISE(ABORT, 'immutable dependencies'); END;
CREATE TRIGGER immutable_dependency_delete BEFORE DELETE ON dependencies
BEGIN SELECT RAISE(ABORT, 'immutable dependencies'); END;
CREATE TRIGGER complete_requires_receipt BEFORE UPDATE OF status ON jobs
WHEN NEW.status = 'complete' AND NOT EXISTS (
    SELECT 1 FROM completions WHERE job_key = NEW.job_key)
BEGIN SELECT RAISE(ABORT, 'complete requires validated receipt'); END;
CREATE TRIGGER immutable_terminal BEFORE UPDATE ON jobs
WHEN OLD.status IN ('complete','skipped','failed')
BEGIN SELECT RAISE(ABORT, 'immutable terminal job'); END;
"""


def _local_database_path(path: str | Path) -> Path:
    if str(path) == ":memory:":
        raise UnsafeDatabasePath("a durable queue requires a file, not :memory:")
    resolved = Path(path).expanduser().resolve()
    ancestor = resolved.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    fs_type: str | None = None
    if sys.platform.startswith("linux"):
        try:
            mounts = []
            with open("/proc/self/mountinfo", encoding="utf-8") as stream:
                for line in stream:
                    fields = line.split()
                    marker = fields.index("-")
                    mount = fields[4]
                    for code, char in (("\\040", " "), ("\\011", "\t"),
                                       ("\\012", "\n"), ("\\134", "\\")):
                        mount = mount.replace(code, char)
                    if resolved.is_relative_to(Path(mount)):
                        mounts.append((len(mount), fields[marker + 1]))
            if mounts:
                fs_type = max(mounts)[1]
        except (OSError, ValueError) as exc:
            raise UnsafeDatabasePath("cannot establish database filesystem type") from exc
    elif sys.platform == "darwin":
        try:
            fs_type = subprocess.check_output(
                ["/usr/bin/stat", "-f", "%T", str(ancestor)], text=True, timeout=5
            ).strip().lower()
        except (OSError, subprocess.SubprocessError) as exc:
            raise UnsafeDatabasePath("cannot establish database filesystem type") from exc
    else:
        raise UnsafeDatabasePath("filesystem check supports Linux and macOS only")
    if fs_type is None or any(x in fs_type for x in ("nfs", "smb", "cifs", "sshfs",
                                                    "9p", "ceph", "gluster", "lustre",
                                                    "afs", "fuse", "webdav", "autofs")):
        raise UnsafeDatabasePath("SQLite queue must reside on a verified local filesystem")
    return resolved


class SQLiteJobQueue:
    """One connection per operation; safe to share across threads or open per process."""

    def __init__(self, db_path: str | Path, *, clock: Callable[[], float] = time.time,
                 busy_timeout: float = 30.0) -> None:
        self.db_path = _local_database_path(db_path)
        self._clock = clock
        self._timeout = self._positive(busy_timeout, "busy_timeout")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            if db.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() != "wal":
                raise QueueError("SQLite WAL mode is required")
            db.execute("BEGIN IMMEDIATE")
            try:
                version = db.execute("PRAGMA user_version").fetchone()[0]
                application_id = int.from_bytes(b"S3DQ", "big")
                if version == 0:
                    if db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone():
                        raise QueueError("refusing to initialize a non-queue SQLite database")
                    # Execute statements without executescript's implicit transaction commit.
                    statement = ""
                    for line in _SCHEMA.splitlines():
                        statement += line + "\n"
                        if sqlite3.complete_statement(statement):
                            db.execute(statement)
                            statement = ""
                    db.execute("PRAGMA user_version=1")
                    db.execute(f"PRAGMA application_id={application_id}")
                elif version != 1:
                    raise QueueError(f"unsupported queue schema version: {version}")
                elif db.execute("PRAGMA application_id").fetchone()[0] != application_id:
                    raise QueueError("not a skill3d cluster queue database")
                db.commit()
            except BaseException:
                db.rollback()
                raise

    @staticmethod
    def _positive(value: float, name: str, *, zero: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise QueueError(f"{name} must be a finite number")
        if not math.isfinite(value) or (value < 0 if zero else value <= 0):
            raise QueueError(f"{name} must be {'nonnegative' if zero else 'positive'}")
        return float(value)

    def _now(self) -> float:
        return self._positive(self._clock(), "coordinator time", zero=True)

    @contextlib.contextmanager
    def _connect(self, *, readonly: bool = False) -> Iterator[sqlite3.Connection]:
        uri = self.db_path.as_uri() + "?mode=ro" if readonly else str(self.db_path)
        db = sqlite3.connect(uri, uri=readonly, timeout=self._timeout, isolation_level=None)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            db.execute(f"PRAGMA busy_timeout={int(self._timeout * 1000)}")
            if readonly:
                db.execute("PRAGMA query_only=ON")
            else:
                db.execute("PRAGMA synchronous=FULL")
            yield db
        finally:
            db.close()

    @contextlib.contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    @staticmethod
    def _spec(row: sqlite3.Row) -> JobSpec:
        try:
            spec = JobSpec.from_dict(json.loads(row["spec_json"]))
        except (TypeError, json.JSONDecodeError) as exc:
            raise IdentityMismatch("stored job specification is invalid") from exc
        if (digest_json(spec.to_dict()) != row["spec_hash"] or
                spec.identity.identity_hash != row["identity_hash"] or
                spec.job_key != row["job_key"] or spec.completed_key != row["completed_key"] or
                spec.stage != row["stage"] or spec.max_attempts != row["max_attempts"] or
                spec.priority != row["priority"] or
                spec.dependency_policy != row["dependency_policy"]):
            raise IdentityMismatch("stored job identity mismatch")
        return spec

    @classmethod
    def _verify_dependencies(cls, db: sqlite3.Connection, row: sqlite3.Row) -> None:
        spec = cls._spec(row)
        dependencies = tuple(item[0] for item in db.execute(
            "SELECT depends_on FROM dependencies WHERE job_key=? ORDER BY depends_on",
            (row["job_key"],),
        ))
        if dependencies != spec.dependencies:
            raise IdentityMismatch("stored DAG differs from frozen dependencies")

    @classmethod
    def _lease(cls, row: sqlite3.Row) -> Lease:
        return Lease(cls._spec(row), WorkerIdentity(row["worker_id"], row["host"], row["gpu_id"]),
                     row["lease_token"], row["lease_until"], row["attempts"],
                     "publish" if row["resume_phase"] == "publish" else row["status"],
                     row["output_metadata_json"])

    @staticmethod
    def _event(db: sqlite3.Connection, row: sqlite3.Row, event: str, now: float,
               detail: Mapping[str, Any] | None = None) -> None:
        db.execute(
            "INSERT INTO events(job_key,event,at,attempt,worker_id,host,gpu_id,detail_json)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (row["job_key"], event, now, row["attempts"], row["worker_id"], row["host"],
             row["gpu_id"], canonical_json(dict(detail or {}))),
        )

    def enqueue(self, spec: JobSpec) -> EnqueueResult:
        return self.enqueue_many([spec])[0]

    def enqueue_many(self, specs: Sequence[JobSpec]) -> list[EnqueueResult]:
        """Atomic topologically ordered batch. Existing keys must match the entire spec."""
        now = self._now()
        results = []
        with self._write() as db:
            for spec in specs:
                if not isinstance(spec, JobSpec):
                    raise QueueError("enqueue requires JobSpec values")
                spec.__post_init__()
                body = canonical_json(spec.to_dict())
                old = db.execute("SELECT * FROM jobs WHERE job_key=?", (spec.job_key,)).fetchone()
                if old is not None:
                    self._verify_dependencies(db, old)
                    if old["spec_json"] != body:
                        raise EnqueueConflict("job key already frozen with another specification")
                    results.append(EnqueueResult(spec.job_key, False, old["status"]))
                    continue
                if db.execute("SELECT 1 FROM jobs WHERE completed_key=?",
                              (spec.completed_key,)).fetchone():
                    raise EnqueueConflict("completed key is already owned by another job")
                for dep in spec.dependencies:
                    upstream = db.execute("SELECT * FROM jobs WHERE job_key=?", (dep,)).fetchone()
                    if upstream is None:
                        raise QueueError("dependencies must be enqueued first")
                    upstream_spec = self._spec(upstream)
                    if upstream_spec.identity != spec.identity:
                        raise IdentityMismatch("dependencies must have the same frozen identity")
                    allowed = (("reconstruction",) if spec.stage == "scale" else
                               ("reconstruction", "scale") if spec.stage == "binding" else STAGES)
                    if upstream_spec.stage not in allowed:
                        raise QueueError("dependency must precede its stage")
                db.execute(
                    "INSERT INTO jobs(job_key,completed_key,spec_json,spec_hash,stage,identity_hash,"
                    "status,dependency_policy,priority,max_attempts,available_at,created_at,updated_at)"
                    " VALUES(?,?,?,?,?,?,'queued',?,?,?,?,?,?)",
                    (spec.job_key, spec.completed_key, body, digest_json(spec.to_dict()), spec.stage,
                     spec.identity.identity_hash, spec.dependency_policy, spec.priority,
                     spec.max_attempts, now, now, now),
                )
                db.executemany("INSERT INTO dependencies(job_key,depends_on) VALUES(?,?)",
                               [(spec.job_key, dep) for dep in spec.dependencies])
                row = db.execute("SELECT * FROM jobs WHERE job_key=?", (spec.job_key,)).fetchone()
                self._event(db, row, "enqueued", now)
                results.append(EnqueueResult(spec.job_key, True, "queued"))
        return results

    def enqueue_pipeline(self, prefix: str, identity: FrozenIdentity, *,
                         max_attempts: int = 3, priority: int = 0) -> list[EnqueueResult]:
        """Conservative reconstruction -> scale -> binding dependencies."""
        specs = []
        for i, stage in enumerate(PIPELINE_STAGES):
            key = f"{prefix}/{stage}"
            deps = () if i == 0 else (specs[-1].job_key,)
            completed = digest_json({"identity_hash": identity.identity_hash,
                                     "prefix": prefix, "stage": stage})
            specs.append(JobSpec(key, completed, identity, stage, deps, max_attempts, priority))
        return self.enqueue_many(specs)

    def _terminal(self, db: sqlite3.Connection, row: sqlite3.Row, now: float, *,
                  error_code: str | None = None, detail: str | None = None,
                  reason: str | None = None) -> None:
        status = "skipped" if reason is not None else "failed"
        if reason is not None:
            safe_text(reason, field="skip reason")
        else:
            safe_text(error_code, field="error_code", limit=128)
            safe_text(detail, field="error detail")
        db.execute(
            "UPDATE jobs SET status=?,lease_token=NULL,lease_until=NULL,error_code=?,"
            "error_detail=?,skip_reason=?,updated_at=? WHERE job_key=?",
            (status, error_code, detail, reason, now, row["job_key"]),
        )
        self._event(db, row, status, now, {"error_code": error_code, "detail": detail, "reason": reason})

    def _settle(self, db: sqlite3.Connection, now: float) -> int:
        count = 0
        expired = db.execute(
            "SELECT * FROM jobs WHERE status IN ('leased','running','publishing') AND lease_until<=?",
            (now,),
        ).fetchall()
        for row in expired:
            self._verify_dependencies(db, row)
            self._event(db, row, "lease_expired", now, {"phase": row["status"]})
            if row["status"] == "running" and row["attempts"] >= row["max_attempts"]:
                self._terminal(db, row, now, error_code="attempt_budget_exhausted",
                               detail="Model lease expired after final allowed attempt")
            else:
                db.execute(
                    "UPDATE jobs SET status='queued',lease_token=NULL,lease_until=NULL,"
                    "available_at=?,updated_at=? WHERE job_key=?", (now, now, row["job_key"]),
                )
            count += 1
        while True:
            blocked = db.execute(
                "SELECT j.*,u.job_key AS upstream_key,u.status AS upstream_status "
                "FROM jobs j JOIN dependencies d ON d.job_key=j.job_key "
                "JOIN jobs u ON u.job_key=d.depends_on WHERE j.status='queued' AND "
                "(u.status='failed' OR (u.status='skipped' AND j.dependency_policy='complete')) "
                "ORDER BY CASE u.status WHEN 'failed' THEN 0 ELSE 1 END,u.job_key"
            ).fetchall()
            if not blocked:
                break
            seen = set()
            for row in blocked:
                if row["job_key"] in seen:
                    continue
                seen.add(row["job_key"])
                self._verify_dependencies(db, row)
                if row["upstream_status"] == "failed":
                    self._terminal(db, row, now, error_code="dependency_failed",
                                   detail=f"Required upstream job failed: {row['upstream_key']}")
                else:
                    self._terminal(db, row, now,
                                   reason=f"Required upstream job skipped: {row['upstream_key']}")
                count += 1
        return count

    def reap_expired(self) -> int:
        with self._write() as db:
            return self._settle(db, self._now())

    def claim(self, worker: WorkerIdentity, *, lease_seconds: float = 300,
              stages: Sequence[str] | None = None) -> Lease | None:
        if not isinstance(worker, WorkerIdentity):
            raise QueueError("claim requires WorkerIdentity")
        worker.__post_init__()
        duration = self._positive(lease_seconds, "lease_seconds")
        permitted = tuple(STAGES if stages is None else stages)
        if not permitted or any(stage not in STAGES for stage in permitted):
            raise QueueError("claim stages must be a nonempty supported stage list")
        with self._write() as db:
            now = self._now()
            self._settle(db, now)
            placeholders = ",".join("?" for _ in permitted)
            row = db.execute(
                "SELECT j.* FROM jobs j WHERE j.status='queued' AND j.available_at<=? "
                f"AND j.stage IN ({placeholders}) AND NOT EXISTS ("
                "SELECT 1 FROM dependencies d JOIN jobs u ON u.job_key=d.depends_on "
                "WHERE d.job_key=j.job_key AND NOT (u.status='complete' OR "
                "(u.status='skipped' AND j.dependency_policy='complete_or_skipped'))) "
                "ORDER BY j.priority DESC,j.created_at,j.job_key LIMIT 1",
                (now, *permitted),
            ).fetchone()
            if row is None:
                return None
            self._verify_dependencies(db, row)
            phase = "publishing" if row["resume_phase"] == "publish" else "leased"
            db.execute(
                "UPDATE jobs SET status=?,lease_token=?,lease_until=?,worker_id=?,host=?,gpu_id=?,"
                "claims=claims+1,updated_at=? WHERE job_key=?",
                (phase, secrets.token_hex(32), now + duration, worker.worker_id, worker.host,
                 worker.gpu_id, now, row["job_key"]),
            )
            current = db.execute("SELECT * FROM jobs WHERE job_key=?", (row["job_key"],)).fetchone()
            self._event(db, current, "claimed", now, {"phase": phase})
            return self._lease(current)

    def _owned(self, db: sqlite3.Connection, lease: Lease, now: float) -> sqlite3.Row:
        if not isinstance(lease, Lease):
            raise QueueError("operation requires Lease")
        row = db.execute("SELECT * FROM jobs WHERE job_key=?", (lease.job_key,)).fetchone()
        if row is None:
            raise LeaseLost("unknown job")
        spec = self._spec(row)
        self._verify_dependencies(db, row)
        if spec != lease.spec:
            raise IdentityMismatch("lease job specification differs from frozen identity")
        if (row["status"] not in ACTIVE or row["lease_until"] <= now or
                row["lease_token"] != lease.token or
                (row["worker_id"], row["host"], row["gpu_id"]) !=
                (lease.worker.worker_id, lease.worker.host, lease.worker.gpu_id)):
            raise LeaseLost("lease expired, fenced, terminal, or owned by another worker")
        return row

    def heartbeat(self, lease: Lease, *, lease_seconds: float = 300) -> Lease:
        duration = self._positive(lease_seconds, "lease_seconds")
        with self._write() as db:
            now = self._now()
            row = self._owned(db, lease, now)
            until = max(row["lease_until"], now + duration)
            db.execute("UPDATE jobs SET lease_until=?,updated_at=? WHERE job_key=?",
                       (until, now, lease.job_key))
            self._event(db, row, "heartbeat", now)
            return self._lease(db.execute("SELECT * FROM jobs WHERE job_key=?",
                                         (lease.job_key,)).fetchone())

    def start_attempt(self, lease: Lease) -> Lease:
        """Idempotent within the same lease; call immediately before model execution."""
        with self._write() as db:
            now = self._now()
            row = self._owned(db, lease, now)
            if row["status"] == "running":
                return self._lease(row)
            if row["status"] != "leased" or row["resume_phase"] != "prefetch":
                raise InvalidTransition("cannot run the model during publication")
            if row["attempts"] >= row["max_attempts"]:
                raise InvalidTransition("model attempt budget exhausted")
            db.execute(
                "UPDATE jobs SET status='running',attempts=attempts+1,error_code=NULL,"
                "error_detail=NULL,updated_at=? WHERE job_key=?", (now, lease.job_key),
            )
            current = db.execute("SELECT * FROM jobs WHERE job_key=?", (lease.job_key,)).fetchone()
            self._event(db, current, "attempt_started", now)
            return self._lease(current)

    def finish_attempt(self, lease: Lease, *, output_metadata: Mapping[str, Any]) -> Lease:
        """Checkpoint output locators/digests, not output bytes or reference answers."""
        if not isinstance(output_metadata, Mapping):
            raise QueueError("output_metadata must be an object")
        metadata = canonical_json(dict(output_metadata))
        if len(metadata) > 65536:
            raise QueueError("output metadata exceeds 64 KiB")
        with self._write() as db:
            now = self._now()
            row = self._owned(db, lease, now)
            if row["status"] == "publishing":
                if row["output_metadata_json"] != metadata:
                    raise PublicationConflict("output checkpoint is already frozen")
                return self._lease(row)
            if row["status"] != "running":
                raise InvalidTransition("finish_attempt requires a running model attempt")
            db.execute(
                "UPDATE jobs SET status='publishing',resume_phase='publish',output_metadata_json=?,"
                "updated_at=? WHERE job_key=?", (metadata, now, lease.job_key),
            )
            self._event(db, row, "output_checkpointed", now)
            return self._lease(db.execute("SELECT * FROM jobs WHERE job_key=?",
                                         (lease.job_key,)).fetchone())

    def transfer_failed(self, lease: Lease, *, phase: str, error_code: str, detail: str,
                        retry_delay: float = 0) -> None:
        if phase not in ("prefetch", "return"):
            raise QueueError("transfer phase must be prefetch or return")
        safe_text(error_code, field="error_code", limit=128)
        safe_text(detail, field="transfer error detail")
        delay = self._positive(retry_delay, "retry_delay", zero=True)
        with self._write() as db:
            now = self._now()
            row = self._owned(db, lease, now)
            expected = "leased" if phase == "prefetch" else "publishing"
            if row["status"] != expected:
                raise InvalidTransition(f"{phase} failure requires {expected} state")
            column = "prefetch_failures" if phase == "prefetch" else "return_failures"
            db.execute(
                f"UPDATE jobs SET status='queued',{column}={column}+1,"
                "lease_token=NULL,lease_until=NULL,available_at=?,updated_at=?,"
                "error_code=?,error_detail=? WHERE job_key=?",
                (now + delay, now, error_code, detail, lease.job_key),
            )
            self._event(db, row, f"{phase}_failed", now, {"error_code": error_code, "detail": detail})

    def fail(self, lease: Lease, *, error_code: str, detail: str,
             retryable: bool = False, retry_delay: float = 0) -> str:
        """Explicit model/output failure. Transport errors use transfer_failed instead."""
        safe_text(error_code, field="error_code", limit=128)
        safe_text(detail, field="error detail")
        if type(retryable) is not bool:
            raise QueueError("retryable must be bool")
        delay = self._positive(retry_delay, "retry_delay", zero=True)
        with self._write() as db:
            now = self._now()
            row = self._owned(db, lease, now)
            if retryable and row["status"] == "leased":
                raise InvalidTransition("prefetch failures must not be model retries")
            if retryable and row["attempts"] < row["max_attempts"]:
                db.execute(
                    "UPDATE jobs SET status='queued',resume_phase='prefetch',"
                    "output_metadata_json='{}',lease_token=NULL,lease_until=NULL,"
                    "available_at=?,updated_at=?,error_code=?,error_detail=? WHERE job_key=?",
                    (now + delay, now, error_code, detail, lease.job_key),
                )
                self._event(db, row, "model_retry", now, {"error_code": error_code, "detail": detail})
                return "queued"
            self._terminal(db, row, now, error_code=error_code, detail=detail)
            return "failed"

    def skip(self, lease: Lease, *, reason: str) -> None:
        with self._write() as db:
            now = self._now()
            row = self._owned(db, lease, now)
            self._terminal(db, row, now, reason=reason)
            self._settle(db, now)

    def complete(self, lease: Lease, *, receipt_path: str | Path,
                 expected_receipt_hash: str, artifact_root: str | Path) -> CompletionResult:
        """Validate outside the write lock, then recheck lease and atomically commit.

        Exact retries by the accepted lease succeed even after expiry; new output for
        an existing completed_key always conflicts. Retry does not rehash artifacts.
        """
        publication = inspect_receipt(receipt_path, expected_receipt_hash=expected_receipt_hash)
        receipt = publication.receipt
        if (receipt.job_key, receipt.completed_key, receipt.identity_hash, receipt.stage) != (
                lease.job_key, lease.spec.completed_key, lease.spec.identity.identity_hash,
                lease.spec.stage):
            raise IdentityMismatch("receipt does not match lease identity")
        with self._connect(readonly=True) as db:
            db.execute("BEGIN")
            stored = db.execute("SELECT * FROM jobs WHERE job_key=?", (lease.job_key,)).fetchone()
            if stored is None:
                raise LeaseLost("unknown job")
            if self._spec(stored) != lease.spec:
                raise IdentityMismatch("completion lease differs from frozen job specification")
            self._verify_dependencies(db, stored)
            existing = db.execute("SELECT * FROM completions WHERE completed_key=?",
                                  (receipt.completed_key,)).fetchone()
            if existing is not None:
                return self._duplicate(existing, lease, publication.fingerprint)
            row = self._owned(db, lease, self._now())
            if row["status"] != "publishing" or receipt.attempt != row["attempts"]:
                raise InvalidTransition("completion requires checkpointed output and matching attempt")
        validate_artifacts(publication, artifact_root)
        with self._write() as db:
            now = self._now()
            existing = db.execute("SELECT * FROM completions WHERE completed_key=?",
                                  (receipt.completed_key,)).fetchone()
            if existing is not None:
                return self._duplicate(existing, lease, publication.fingerprint)
            row = self._owned(db, lease, now)
            if row["status"] != "publishing" or receipt.attempt != row["attempts"]:
                raise InvalidTransition("completion requires checkpointed output and matching attempt")
            db.execute(
                "INSERT INTO completions(completed_key,job_key,identity_hash,stage,receipt_hash,"
                "receipt_json,fingerprint,attempt,accepted_lease_token,worker_id,host,gpu_id,"
                "completed_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (receipt.completed_key, receipt.job_key, receipt.identity_hash, receipt.stage,
                 publication.receipt_hash, canonical_json(receipt.to_dict()),
                 publication.fingerprint, receipt.attempt, lease.token, lease.worker.worker_id,
                 lease.worker.host, lease.worker.gpu_id, now),
            )
            db.execute(
                "UPDATE jobs SET status='complete',lease_token=NULL,lease_until=NULL,"
                "error_code=NULL,error_detail=NULL,updated_at=? WHERE job_key=?", (now, lease.job_key),
            )
            self._event(db, row, "completed", now, {"receipt_hash": publication.receipt_hash,
                                                  "fingerprint": publication.fingerprint})
            return CompletionResult(lease.job_key, receipt.completed_key, publication.receipt_hash,
                                    publication.fingerprint, False)

    def _duplicate(self, row: sqlite3.Row, lease: Lease, fingerprint: str) -> CompletionResult:
        if row["identity_hash"] != lease.spec.identity.identity_hash or row["stage"] != lease.spec.stage:
            raise IdentityMismatch("completed identity differs from lease")
        if row["fingerprint"] != fingerprint or row["job_key"] != lease.job_key:
            raise PublicationConflict("completed key already contains different artifacts/receipt")
        if (row["accepted_lease_token"] != lease.token or
                (row["worker_id"], row["host"], row["gpu_id"]) !=
                (lease.worker.worker_id, lease.worker.host, lease.worker.gpu_id)):
            raise LeaseLost("duplicate publication is not from the accepted lease owner")
        return CompletionResult(row["job_key"], row["completed_key"], row["receipt_hash"],
                                row["fingerprint"], True)

    def get_job(self, job_key: str) -> dict[str, Any] | None:
        """Read only. Lease tokens are deliberately excluded from progress output."""
        with self._connect(readonly=True) as db:
            db.execute("BEGIN")
            row = db.execute("SELECT * FROM jobs WHERE job_key=?", (job_key,)).fetchone()
            if row is None:
                return None
            spec = self._spec(row)
            self._verify_dependencies(db, row)
            fields = ("status", "attempts", "max_attempts", "claims", "prefetch_failures",
                      "return_failures", "resume_phase", "available_at", "lease_until",
                      "worker_id", "host", "gpu_id", "error_code", "error_detail",
                      "skip_reason", "created_at", "updated_at")
            result = {name: row[name] for name in fields}
            result["spec"] = spec.to_dict()
            result["output_metadata"] = json.loads(row["output_metadata_json"])
            completion = db.execute(
                "SELECT completed_key,receipt_hash,fingerprint,receipt_json,completed_at "
                "FROM completions WHERE job_key=?", (job_key,),
            ).fetchone()
            result["completion"] = None if completion is None else {
                "completed_key": completion["completed_key"], "receipt_hash": completion["receipt_hash"],
                "fingerprint": completion["fingerprint"],
                "receipt": json.loads(completion["receipt_json"]),
                "completed_at": completion["completed_at"],
            }
            return result

    def progress_summary(self) -> dict[str, Any]:
        """Snapshot only: never claims/reaps/retries or changes timestamps."""
        now = self._now()
        with self._connect(readonly=True) as db:
            db.execute("BEGIN")
            counts = {status: 0 for status in ("queued", *ACTIVE, *TERMINAL)}
            for row in db.execute("SELECT status,COUNT(*) AS n FROM jobs GROUP BY status"):
                counts[row["status"]] = row["n"]
            totals = db.execute(
                "SELECT COALESCE(SUM(attempts),0) AS model_attempts,"
                "COALESCE(SUM(prefetch_failures),0) AS prefetch_failures,"
                "COALESCE(SUM(return_failures),0) AS return_failures FROM jobs"
            ).fetchone()
            workers = [dict(row) for row in db.execute(
                "SELECT worker_id,host,gpu_id,status,COUNT(*) AS jobs FROM jobs "
                "WHERE worker_id IS NOT NULL GROUP BY worker_id,host,gpu_id,status "
                "ORDER BY worker_id,host,gpu_id,status"
            )]
            expired = db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('leased','running','publishing') "
                "AND lease_until<=?", (now,),
            ).fetchone()[0]
            ready = db.execute(
                "SELECT COUNT(*) FROM jobs j WHERE status='queued' AND available_at<=? "
                "AND NOT EXISTS (SELECT 1 FROM dependencies d JOIN jobs u ON u.job_key=d.depends_on "
                "WHERE d.job_key=j.job_key AND NOT (u.status='complete' OR "
                "(u.status='skipped' AND j.dependency_policy='complete_or_skipped')))", (now,),
            ).fetchone()[0]
            db.rollback()
            return {"as_of": now, "counts": counts, "total": sum(counts.values()),
                    "ready": ready, "expired_leases": expired, **dict(totals), "workers": workers}

    def events(self, *, after_id: int = 0, limit: int = 100,
               job_key: str | None = None) -> list[dict[str, Any]]:
        if type(after_id) is not int or after_id < 0 or type(limit) is not int or not 1 <= limit <= 10000:
            raise QueueError("invalid event cursor/limit")
        with self._connect(readonly=True) as db:
            query = "SELECT * FROM events WHERE id>?"
            params: list[Any] = [after_id]
            if job_key is not None:
                query += " AND job_key=?"
                params.append(job_key)
            query += " ORDER BY id LIMIT ?"
            params.append(limit)
            return [{**{k: row[k] for k in row.keys() if k != "detail_json"},
                     "detail": json.loads(row["detail_json"])} for row in db.execute(query, params)]
