# Durable Queue API

This package uses Python 3.11 standard-library modules only. The Titan CPU
coordinator owns the SQLite database. GPU workers talk to HTTP endpoints; they
must never open the database over NFS or copy a live database to another host.
No model, configuration, prepared-binding or relocation modules are imported.

## Identity And Scope

`FrozenIdentity.freeze(manifest, method_config)` freezes canonical JSON snapshots,
their SHA-256 digests and a combined, versioned identity digest. Optional
`expected_manifest_hash` and `expected_method_config_hash` reject mismatches.
`to_dict()` / `from_dict()` are the wire format; decoding revalidates all hashes.
These are canonical JSON hashes, not hashes of an arbitrary source file's bytes.

Only pass public, explicitly whitelisted manifest/configuration metadata. The
queue rejects known credential, reference-answer and GT fields, credential URLs,
nonfinite numbers and obvious credential strings. This is defense in depth, not
a classifier capable of recognizing every secret in arbitrary text. Do not
pass raw exception traces, private receipts, signed URLs or raw dataset rows.
`Lease.to_dict()` includes the public manifest/configuration snapshots.

Scope comes exclusively from the caller's frozen manifest. This package does
not discover scenes or expand a benchmark. For the current deployment the
caller seeds the frozen r03 1154-question scope, with a new unified M5 method
identity, rather than mixing the old 579 outputs into a new completed key.
Archiving old bytes, private identity receipts and validated reuse of existing
geometry for 219 scenes remain outside this package.

## Constructor And Seed

```python
from skill3d.cluster import FrozenIdentity, JobSpec, SQLiteJobQueue, WorkerIdentity

queue = SQLiteJobQueue("/titan-local/cluster/queue.sqlite3")
identity = FrozenIdentity.freeze(public_manifest, public_method_config)

# One end-to-end scene task. The wrapper validates the entire QA group.
spec = JobSpec(
    job_key="r03/scene-a/gpu_aggregate",
    completed_key=f"{identity.identity_hash}/scene-a/gpu_aggregate",
    identity=identity,
    stage="gpu_aggregate",
    max_attempts=3,
)
queue.enqueue(spec)

# Alternatively, a conservative three-stage pipeline.
queue.enqueue_pipeline("scene-b", identity, max_attempts=3, priority=0)
```

`enqueue(spec)` returns `EnqueueResult(job_key, created, status)`.
Identical full specifications are idempotent even after a terminal outcome.
A different specification under the same job key or reuse of another job's
completed key raises `EnqueueConflict`. A terminal job is never reset by enqueue.

`enqueue_many(specs)` is an atomic, topologically ordered DAG seed. Dependencies
must already exist or occur earlier in the batch and must share the exact
frozen identity. An error rolls back the entire batch. Supported stages:

- `reconstruction`: root.
- `scale`: requires reconstruction dependencies.
- `binding`: requires reconstruction or scale dependencies.
- `gpu_aggregate`: a root scene task, or a task depending on previously seeded jobs.

`enqueue_pipeline()` seeds only reconstruction -> scale -> binding.
Default dependency policy is `complete`. An upstream failure makes blocked
descendants explicitly `failed` with `dependency_failed`; an upstream skip
makes blocked descendants `skipped` with a reason, never `complete`.
`dependency_policy="complete_or_skipped"` explicitly permits a skipped upstream
job; the wrapper must implement the corresponding fallback. A failure is never
accepted as a satisfied dependency.

## Lease And Attempt Lifecycle

```python
worker = WorkerIdentity("3090-worker-0", "3090-host", "0")
lease = queue.claim(worker, lease_seconds=300, stages=["gpu_aggregate"])
if lease is not None:
    lease = queue.heartbeat(lease, lease_seconds=300)
    if lease.phase != "publish":
        # Prefetch first, without consuming a model attempt.
        lease = queue.start_attempt(lease)
        # Execute model and persist a recoverable output directory.
        lease = queue.finish_attempt(
            lease,
            output_metadata={"output_locator": "scene-a/attempt-1"},
        )
    # Upload/recover retained output; never execute models for phase="publish".
```

`claim()` uses `BEGIN IMMEDIATE`, returns one job or `None`, and atomically
reaps expired leases and propagates blocked terminal dependencies. Each claim
gets a random fencing token and records worker ID, host and GPU identity.
Serialize using `lease.to_dict()`; decode using `Lease.from_dict(payload)`.
Client attempt/phase/expiry values are informational: transitions always use
the coordinator's persisted state, fencing token and clock.

`start_attempt(lease)` consumes one model attempt, exactly once for the same
lease. Repeating the request after an ACK loss returns the running attempt.
`finish_attempt(lease, output_metadata=...)` checkpoints at most 64 KiB of
public output locators/digests, changes state to `publishing` and is idempotent
for identical metadata. Different metadata in the same checkpoint conflicts.

`heartbeat(lease, lease_seconds=300)` never shortens a live lease. Workers
heartbeat during model execution, upload and coordinator hashing. An expired
lease cannot renew, mutate state or complete. All times come from the
coordinator; keep its clock synchronized and use a lease much longer than the
heartbeat period, for example 300 seconds and a 30-second heartbeat.

On expiry before model start, reclaim does not consume a model attempt.
On expiry during execution, reclaim can start another attempt up to the budget.
At the final model attempt, expiry becomes `failed/attempt_budget_exhausted`.
On expiry after checkpoint, reclaim returns `lease.phase == "publish"` and the
original output metadata/attempt, even when the model budget is exhausted.
The output must be recoverable from that locator; this is not a guarantee that
a crashed host's unreplicated local disk remains reachable.

`transfer_failed(lease, phase="prefetch"|"return", error_code=..., detail=...,
retry_delay=0)` releases the lease and increments a separate transport counter.
Prefetch requires `leased`; return requires `publishing` and preserves the
checkpoint. Neither consumes a model attempt. Transport retries have no
implicit model budget: the coordinator may enforce its own transfer policy and
terminate with `fail(..., retryable=False)` if recovery is impossible.

`fail(lease, error_code=..., detail=..., retryable=False, retry_delay=0)` requires
an explicit, sanitized error. A retryable model/output error requeues with a
fresh-model phase when budget remains; otherwise it becomes terminal `failed`
with the original error. `skip(lease, reason=...)` requires a nonempty
applicability reason. No skip/failure creates a completed receipt.
`reap_expired()` is an explicit maintenance operation; no read endpoint reaps.

## Validated Publication

After the wrapper's scene/QA-group validation, the coordinator-local immutable
output directory contains the actual artifact bytes and a receipt:

```python
import json
from skill3d.cluster import ArtifactDigest, PublicationReceipt, hash_file

digest, size = hash_file(local_scene_bin)
receipt = PublicationReceipt(
    job_key=lease.job_key,
    completed_key=lease.spec.completed_key,
    identity_hash=lease.spec.identity.identity_hash,
    stage=lease.spec.stage,
    attempt=lease.attempt,
    validator="scene-qa-validator-v1",
    artifacts=(ArtifactDigest("scene.bin", digest, size),),
)
local_receipt_path.write_text(
    json.dumps(receipt.to_dict(), sort_keys=True), encoding="utf-8"
)
result = queue.complete(
    lease,
    receipt_path=local_receipt_path,
    expected_receipt_hash=hash_file(local_receipt_path)[0],
    artifact_root=local_immutable_scene_directory,
)
```

Receipt schema is `skill3d.cluster.publication.v1`. Its exact fields are
`schema/job_key/completed_key/identity_hash/stage/attempt/validation/artifacts`.
Validation is `{"status":"validated","validator":"scene-qa-validator-v1"}`.
Artifacts are nonempty and have unique relative POSIX names, SHA-256 hashes
and sizes. Names with traversal or symlinks escaping the root are rejected.
Duplicate JSON keys, mismatched receipt bytes, wrong identity/stage/attempt,
missing artifacts and content mismatches cannot create `complete`.

The queue verifies receipt structure and actual bytes, not model correctness.
The HTTP coordinator must trust/allowlist the semantic validator and reject
unvalidated QA groups before invoking this API. Do not accept an arbitrary
worker-supplied server path as `receipt_path` or `artifact_root`.

The storage layer must finalize/retain an immutable directory, fsync uploaded
files and receipt plus directory metadata, and atomically place it under a
non-overwriting, identity-specific publication location before `complete()`.
Keep files immutable while/after hashing; SQLite does not lock artifact files.
Hashing happens outside the SQLite write lock, with a second lease check before
the single commit of receipt, job status and completion event. A failed commit
can leave an unreferenced staged directory, never a fake completed database row.

`CompletionResult` has `job_key/completed_key/receipt_hash/fingerprint/duplicate`
and `status="complete"`. The fingerprint includes the exact receipt-byte hash.
The same accepted lease owner may repeat the original receipt after ACK loss,
even after expiry or restart; the persisted commit ACK is returned without
rehashing artifact bytes or rerunning models. Keep the original receipt bytes
for retransmission. Different receipt bytes or artifacts under the same
completed key raise `PublicationConflict`, even if a new receipt is semantically
similar. A fenced old lease cannot submit a new completion.

## HTTP And Operations

Suggested endpoints map directly to these methods:

- POST seed/claim/heartbeat/start/checkpoint/transfer-failure/fail/skip/commit.
- GET job/progress/events via `get_job()`, `progress_summary()` and `events()`.
- `IdentityMismatch`, `UnsafePayload` and invalid envelopes: HTTP 400/422.
- `LeaseLost`, `EnqueueConflict`, `PublicationConflict`, `InvalidTransition`: 409.
- Invalid content: 422 with a sanitized error, never a success ACK.
- SQLite busy/unavailable: 503; retry idempotent requests with bounded backoff.

Worker authentication/transport secrets belong to the HTTP layer and must not
be written to manifests, output metadata, receipts or events. Bind each
authenticated worker to its declared host/GPU identity. Fencing tokens travel
only in lease mutation requests, not in progress/events/logs. Use restrictive
permissions on the database directory; the database retains fencing tokens for
durable commit ACK matching. Deploy a single active coordinator/master.

SQLite uses a separate connection per operation, WAL, `synchronous=FULL`,
foreign keys, immutable identity/completion triggers and a 30-second default
busy timeout. Construction rejects known NFS/SMB/FUSE/network filesystems on
Linux/macOS before creating the database. Deployment must additionally verify
the Titan mount is truly a local persistent disk, not an ambiguous bind mount.
Do not copy only the `.sqlite3` file from a running WAL database; use SQLite's
backup API or a controlled shutdown with a checkpoint for backups.

Progress methods use read-only SQLite connections and a consistent summary
snapshot. They distinguish queued, leased, running, publishing, complete,
skipped and failed, expose transport counters/expired leases/worker host+GPU
counts, and do not silently repair or advance anything.

CPU-only verification:

```bash
PYTHONPATH=/Users/bytedance/wangjh/harness3d-cluster-development/src \
  /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -m pytest \
  tests/unit/test_cluster_queue.py -q
```
