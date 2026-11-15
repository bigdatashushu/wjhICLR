"""Stdlib-only primitives for a CPU coordinator and HTTP-connected GPU workers."""

from .identity import FrozenIdentity, IdentityMismatch, QueueError, UnsafePayload
from .publication import (
    ArtifactDigest, PublicationConflict, PublicationError, PublicationReceipt, hash_file,
)
from .queue import (
    CompletionResult, EnqueueConflict, EnqueueResult, InvalidTransition, JobSpec,
    Lease, LeaseLost, SQLiteJobQueue, UnsafeDatabasePath, WorkerIdentity,
)

__all__ = [
    "ArtifactDigest", "CompletionResult", "EnqueueConflict", "EnqueueResult",
    "FrozenIdentity", "IdentityMismatch", "InvalidTransition", "JobSpec", "Lease",
    "LeaseLost", "PublicationConflict", "PublicationError", "PublicationReceipt",
    "QueueError", "SQLiteJobQueue", "UnsafeDatabasePath", "UnsafePayload",
    "WorkerIdentity", "hash_file",
]
