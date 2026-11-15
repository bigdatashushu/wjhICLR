"""Bounded, deterministic scheduling across identical native reviewer replicas."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Lock

from skill3d.infra.m5_identity import content_sha256

POOL_VERSION = "intern-review-pool-v1"


class InternReviewPool:
    def __init__(self, clients):
        if not clients:
            raise ValueError("review pool must have at least one client")
        self.clients = tuple(clients)
        self.provider = self.clients[0].provider
        self.target_binary = bool(getattr(self.clients[0], "target_binary", False))
        if any(client.audit_identity() | {"endpoint_sha256": "", "expected_identity_sha256": ""}
               != self.clients[0].audit_identity() | {
                   "endpoint_sha256": "", "expected_identity_sha256": ""}
               for client in self.clients[1:]):
            raise ValueError("review pool client protocols differ")
        self._locks = tuple(Lock() for _ in self.clients)
        self._identity_lock = Lock()
        self._identity = None

    def identity(self):
        with self._identity_lock:
            if self._identity is None:
                members = [client.identity() for client in self.clients]
                method = {k: v for k, v in members[0].items() if k != "device"}
                if any({k: v for k, v in member.items() if k != "device"} != method
                       for member in members[1:]):
                    raise ValueError("review pool model/runtime/calibration identities differ")
                self._identity = {
                    "version": POOL_VERSION, "method": method, "members": members,
                    "scheduling": "ordered-round-robin-one-request-per-replica",
                }
            return self._identity

    def audit_identity(self, *, seed=None):
        return {
            "provider": self.provider, "version": POOL_VERSION, "cache_eligible": False,
            "members": [client.audit_identity(seed=seed) for client in self.clients],
            "scheduling": "ordered-round-robin-one-request-per-replica",
        }

    def _observe(self, index, candidate, frame, mask, **kwargs):
        client = self.clients[index]
        with self._locks[index]:
            receipt = client.observe_candidate(candidate, frame, mask, **kwargs)
        receipt["review_route"] = {
            "version": POOL_VERSION, "replica_index": index,
            "endpoint_sha256": content_sha256(client.endpoint),
            "pool_identity_sha256": content_sha256(self.identity()),
        }
        if kwargs.get("out_dir") is not None:
            from pathlib import Path
            from .mask_semantic_review import _write
            _write(Path(kwargs["out_dir"]) / "visual" / f"{candidate['candidate_id']}.json",
                   receipt)
        return receipt

    def observe_candidate(self, candidate, frame, mask, **kwargs):
        self.identity()
        return self._observe(0, candidate, frame, mask, **kwargs)

    def observe_candidates(self, observations, frames, **kwargs):
        """Yield in input order with at most one outstanding request per replica."""
        self.identity()
        count = len(self.clients)
        with ThreadPoolExecutor(max_workers=count, thread_name_prefix="intern-review") as executor:
            for start in range(0, len(observations), count):
                futures = []
                for index, observation in enumerate(observations[start:start + count]):
                    candidate = observation["candidate"]
                    futures.append(executor.submit(
                        copy_context().run, self._observe, index, candidate,
                        frames[candidate["frame_slot"]], observation["raw_mask"], **kwargs))
                for future in futures:
                    yield future.result()
