"""Scene-grained transport, episode leases, and bounded durable output delivery.

This layer moves bytes only. It cannot invoke a solver or retry an API request.
Every process reconstructs its cache handle; the on-disk quota and admission lock
are shared by the learning coordinator, episode children, and admission gates.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile
import time
import urllib.parse
import urllib.request

from skill3d.infra.m5_identity import file_identity
from skill3d.storage.hot_cache import (
    HotCache, IntegrityError, QuotaExceeded, StorageError, _flock, _ref,
)
from skill3d.storage.published_transport import PublishedTransportLocator


def materialize_pinned_member(reservation, name, source, identity):
    """Trusted fixed-size transport, under a previously admitted full envelope.

    The producer cannot grow a member past its declared size. Publication still
    checks the complete logical and allocated trees against the reservation.
    Avoid a whole-cache scan for every network block of a multi-GB scene.
    """
    if reservation.closed:
        raise StorageError("transport reservation is closed")
    name = _ref(name)
    dest = reservation.path / "payload" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    checksum = hashlib.sha256()
    remaining = identity["size"]
    with dest.open("xb", buffering=0) as target:
        while remaining:
            block = source.read(min(1024 * 1024, remaining))
            if not block:
                raise IntegrityError("truncated pinned member")
            if reservation.cache.disk_usage(reservation.cache.root).free < (
                reservation.cache.min_free_bytes + len(block)
            ):
                raise QuotaExceeded("transport would cross the free-space floor")
            target.write(block)
            checksum.update(block)
            remaining -= len(block)
    if checksum.hexdigest() != identity["sha256"]:
        raise IntegrityError("pinned member hash changed")


class SceneHotRuntime:
    """Load complete pinned shards and hold them until an episode releases them."""

    def __init__(self, *, closure_path, cache_root, endpoint, token_file,
                 closure_sha256=None):
        self.closure_path = Path(closure_path)
        self.cache_root = Path(cache_root)
        self.endpoint = endpoint.rstrip("/")
        self.token_file = Path(token_file)
        self.closure_sha256 = file_identity(self.closure_path)["sha256"]
        if closure_sha256 is not None and closure_sha256 != self.closure_sha256:
            raise IntegrityError("scene closure identity changed")
        self.closure = json.loads(self.closure_path.read_bytes())
        if self.closure.get("schema_version") != "h3d-scene-closure-v1":
            raise IntegrityError("unsupported scene closure")
        self.locator = PublishedTransportLocator(self.closure["descriptor"])
        self.scenes = tuple(sorted(
            key for key in self.closure["shards"] if key not in {"global", "state"}))
        cache = self.cache()
        for spec in self.closure["shards"].values():
            prefix = cache._entry_path(spec["key"])
            for name, row in spec["members"].items():
                declared = self.closure["descriptor"]["files"].get(row["source"])
                if declared != {"path": str(prefix / name), "identity": row["identity"]}:
                    raise IntegrityError("scene transport destination differs from cache layout")

    def cache(self):
        return HotCache(self.cache_root)

    def _request(self, path, body=None, headers=None):
        headers = dict(headers or {})
        headers["Authorization"] = "Bearer " + self.token_file.read_text().strip()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(urllib.request.Request(
            self.endpoint + path, data=body, headers=headers), timeout=1800)

    @staticmethod
    def validate(path, spec):
        actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
        if actual != set(spec["members"]):
            raise IntegrityError("scene shard member closure changed")
        for name, row in spec["members"].items():
            if file_identity(path / name) != row["identity"]:
                raise IntegrityError("scene shard member bytes changed")

    def _populate(self, cache, scene, *, slot=None):
        spec = self.closure["shards"][scene]
        if spec["key"] in cache.stats()["entries"]:
            if slot is not None:
                cache.pin(slot, spec["key"])
            return
        if slot is not None:
            # Current/next pins cannot prevent the following shard from making
            # space forever. Active readers remain protected by their OS leases.
            cache.pin(slot, None)
            if slot == "current":
                cache.pin("next", None)

        def produce(reservation):
            with self._request("/scene/" + scene) as response:
                with tarfile.open(fileobj=response, mode="r|") as archive:
                    seen = set()
                    for member in archive:
                        if not member.isfile() or member.name not in spec["members"] or (
                            member.name in seen
                        ):
                            raise IntegrityError("unexpected or repeated scene archive member")
                        identity = spec["members"][member.name]["identity"]
                        if member.size != identity["size"]:
                            raise IntegrityError("scene archive member size changed")
                        with archive.extractfile(member) as source:
                            materialize_pinned_member(
                                reservation, member.name, source, identity)
                        seen.add(member.name)
                    if seen != set(spec["members"]):
                        raise IntegrityError("incomplete scene archive")

        cache.populate(
            spec["key"], produce, kind="result" if scene == "state" else "input",
            extract_bytes=spec["bytes"], allocation_bytes=spec["allocation_bytes"],
            slots=(slot,) if slot is not None else (),
            # Every member was size/hash checked during its single incoming
            # pass, and `seen` proves the full closure. Do not reread the shard.
        )

    @contextmanager
    def scene_lease(self, scene, *, wait_for_space=False, max_wait_s=900):
        if scene not in self.scenes:
            raise StorageError("episode is outside the pinned scene scope")
        cache = self.cache()
        # All runtime loaders hold this lock through publish and lease acquisition.
        # The current pin bridges the otherwise unsafe publish-to-lease interval.
        deadline = time.monotonic() + max_wait_s
        while True:
            held = ExitStack()
            try:
                with _flock(cache.root / ".scene-admission.lock"):
                    for name in ("global", "state"):
                        if name in self.closure["shards"]:
                            self._populate(cache, name, slot=name)
                            key = self.closure["shards"][name]["key"]
                            with cache._locked() as state:
                                adopted = name == "state" and (
                                    state["entries"][key]["kind"] == "workspace")
                            # Adopted workspaces are never evictable. Holding SH
                            # for an episode would deadlock its short EX writes.
                            if not adopted:
                                held.enter_context(cache.lease(key))
                    self._populate(cache, scene, slot="current")
                    path = held.enter_context(cache.lease(self.closure["shards"][scene]["key"]))
                break
            except QuotaExceeded:
                held.close()
                if not wait_for_space or time.monotonic() >= deadline:
                    raise
                time.sleep(1)
            except BaseException:
                held.close()
                raise
        with held:
            yield path

    def prefetch(self, scene):
        """Quota refusal is a deferred download; it never dispatches an episode."""
        if scene not in self.scenes:
            raise StorageError("prefetch is outside the pinned scene scope")
        cache = self.cache()
        try:
            with _flock(cache.root / ".scene-admission.lock"):
                self._populate(cache, scene, slot="next")
            return {"status": "ready", "scene": scene}
        except QuotaExceeded:
            return {"status": "deferred_storage_quota", "scene": scene}

    @contextmanager
    def overlap_next(self, scene):
        """Begin one asynchronous prefetch after current input has been leased."""
        index = self.scenes.index(scene)
        next_scene = self.scenes[index + 1] if index + 1 < len(self.scenes) else None
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="scene-prefetch") as pool:
            future = pool.submit(self.prefetch, next_scene) if next_scene else None
            yield future
            # Exceptions remain observable. Closing cannot drop a transfer intent.
            if future is not None:
                future.result()

    def archive_transfer(self, path, manifest):
        with self._request("/archive-intent", json.dumps(manifest).encode(),
                           {"Content-Type": "application/json"}) as response:
            result = json.load(response)
        if result != {"intent": manifest["manifest_sha256"]}:
            raise IntegrityError("receiver bound a different archive intent")
        with path.open("rb") as source:
            with self._request("/archive/" + result["intent"], source, {
                "Content-Type": "application/octet-stream",
                "Content-Length": str(manifest["archive_bytes"]),
            }) as response:
                return json.load(response)

    def fetch_cold_member(self, ack, member_key, writer):
        ref = urllib.parse.quote(member_key, safe="")
        with self._request("/archive-member/" + ack.manifest_sha256 + "/" + ref) as response:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                writer.write(block)

    def publish_output(self, key, producer, *, estimated_bytes, job, shard,
                       attempt, checkpoint=False, latest=False, best=False):
        """Reserve before production; results stay local until a matching ACK."""
        cache = self.cache()
        path = cache.populate(key, producer, output_bytes=estimated_bytes,
                              kind="checkpoint" if checkpoint else "result")
        if latest or best:
            if not checkpoint:
                raise StorageError("retained output must be a checkpoint")
            if latest:
                cache.pin("latest", key)
            if best:
                cache.pin("best", key)
        # Queue even retained checkpoints so durability does not depend on pins.
        package = cache.queue_archive(key, job=job, shard=shard, attempt=attempt)
        return path, package

    def archive_pending(self):
        """Retry persisted transport identities only, retaining latest and best."""
        return self.cache().transfer_pending(self.archive_transfer, delete_acked=True)


class SceneHotInputProvider:
    """Wrap a sealed input provider with a lease spanning the entire episode."""

    def __init__(self, provider, runtime):
        self.provider = provider
        self.runtime = runtime
        reuse = getattr(provider, "reuse_verified_publications", None)
        if reuse is not None:
            reuse()

    def __getattr__(self, name):
        if name in {"provider", "runtime"}:
            raise AttributeError(name)
        return getattr(self.provider, name)

    def __call__(self, variant):
        return self.provider(variant)

    @property
    def input_contract_identity(self):
        identity = deepcopy(self.provider.input_contract_identity)
        record = identity["manifest"]
        origin = self.runtime.closure["original_published_manifest"]
        if origin not in self.runtime.closure["descriptor"]["files"]:
            raise IntegrityError("input manifest lacks an explicit physical identity")
        actual = self.runtime.locator.verified_file(origin, record["sha256"])
        if str(actual) != record["path"]:
            raise IntegrityError("input contract manifest differs from approved locator")
        record["path"] = origin
        return identity

    def _logical_receipt(self, receipt, entry):
        """Keep the original delivered receipt after verifying physical bytes."""
        from skill3d.infra.m5_identity import content_sha256
        logical = deepcopy(receipt)
        paths = {
            "input_manifest": self.runtime.closure["original_published_manifest"],
            "frozen_scene": str(Path(entry["input_root"]) / "scene.json"),
            "scene_input_identity_file": entry["scene_input_identity_file"],
            "migration_receipt_file": entry["migration_receipt_file"],
        }
        for field, origin in paths.items():
            record = logical[field]
            actual = self.runtime.locator.verified_file(origin, record["sha256"])
            if str(actual) != record["path"]:
                raise IntegrityError("receipt physical file differs from approved locator")
            record["path"] = origin
        logical.pop("physical_transport", None)
        logical.pop("receipt_sha256")
        logical["receipt_sha256"] = content_sha256(logical)
        return logical

    def receipt(self, variant, item):
        receipt = self.provider.receipt(variant, item)
        return self._logical_receipt(receipt, self.provider.public_entry(variant))

    def prepare_published(self, variant, item):
        prepared = self.provider.prepare_published(variant, item)
        prepared["physical_transport_receipt"] = deepcopy(prepared["transport_receipt"])
        prepared["transport_receipt"] = self._logical_receipt(
            prepared["transport_receipt"], self.provider.public_entry(variant))
        if "derived_migration_receipt" in prepared:
            prepared["physical_derived_migration_receipt"] = deepcopy(
                prepared["derived_migration_receipt"])
            prepared["derived_migration_receipt"] = deepcopy(
                self.runtime.closure["derived_replay"]["scenes"][
                    str(variant["scene_id"])]["migration_receipts"][str(variant["qa_id"])])
        return prepared

    @contextmanager
    def episode_lease(self, variant):
        with self.runtime.scene_lease(str(variant["scene_id"]), wait_for_space=True):
            yield

    def for_spawned_episode(self, variant):
        return type(self)(self.provider.for_spawned_episode(variant), self.runtime)

    def logical_file(self, path, expected_sha256=None):
        return self.runtime.locator.file(path, expected_sha256)

    def logical_path(self, path):
        physical = str(Path(path).absolute())
        matches = [old for old, row in self.runtime.closure["descriptor"]["files"].items()
                   if row["path"] == physical]
        if len(matches) != 1:
            raise IntegrityError("path has no unique declared logical origin")
        return Path(matches[0])


__all__ = ["SceneHotRuntime", "SceneHotInputProvider"]
