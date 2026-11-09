"""Explicit, immutable deployment authorization without changing run lineage."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Mapping, Sequence

from skill3d.evaluation.fitset_v11 import (
    _require, _write_create_once, content_sha256,
)

DEPLOYMENT_CONTINUATION_SCHEMA = "v11-deployment-continuation/1.0"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_REVISION_INPUT_LIMITS = {
    "revision_input_max_bytes": (4096, 128000),
    "revision_input_max_chars": (4096, 72000),
}


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    _require(bool(value) and not path.is_absolute() and ".." not in path.parts
             and path.as_posix() == value and value != ".",
             "continuation requires a canonical relative file reference")
    return value


def _absolute(value: str | Path) -> Path:
    path = Path(value)
    _require(path.is_absolute() and str(path) == str(value)
             and ".." not in path.parts,
             "continuation requires a canonical absolute locator")
    return path


def _archive_ref(value: str) -> str:
    ref = _relative(value)
    _require(PurePosixPath(ref).parts[0] == "raw_state"
             and len(PurePosixPath(ref).parts) > 1,
             "ledger archive must be inside deployment raw_state")
    return ref


def _identity(data: bytes) -> dict:
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def _checked_file(root: Path, ref: str) -> Path:
    path = root / _relative(ref)
    _require(path.resolve().is_relative_to(root.resolve()) and path.is_file(),
             f"continuation file missing or escaped its locator: {ref}")
    return path


def _write_raw_state_once(root: Path, ref: str, data: bytes) -> None:
    target = root / _relative(ref)
    _require(target.resolve().is_relative_to(root.resolve()),
             "raw state archive escaped its deployment directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        _require(target.read_bytes() == data, "immutable raw state archive changed")


def _manifest_identity(value: Mapping) -> str:
    identity = content_sha256({
        key: child for key, child in value.items() if key != "identity_sha256"})
    _require(value.get("identity_sha256") == identity,
             "continuation manifest content hash changed")
    return identity


class DeploymentContinuation:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.raw = self.path.read_bytes()
        self.value = json.loads(self.raw)
        _require(set(self.value) == {
            "schema_version", "deployment_id", "logical_root", "physical_root", "path_map",
            "run_identity", "original_manifest_file", "current_manifest_sha256",
            "source_identity", "revision_protocol", "preserved_files", "ledger_anchors",
            "external_ledger_anchors", "allowed_changes", "identity_sha256",
        } and self.value.get("schema_version") == DEPLOYMENT_CONTINUATION_SCHEMA,
                 "unsupported deployment continuation schema")
        self.identity_sha256 = _manifest_identity(self.value)
        self.deployment_id = self.value["deployment_id"]
        _require(bool(_SAFE_ID.fullmatch(self.deployment_id)),
                 "unsafe continuation deployment_id")
        self.logical_root = _absolute(self.value["logical_root"])
        self.physical_root = _absolute(self.value["physical_root"])
        self.locations = {
            str(_absolute(old)): _absolute(new)
            for old, new in self.value["path_map"].items()
        }
        _require(self.locations.get(str(self.logical_root)) == self.physical_root,
                 "continuation lacks the exact system locator")
        _require(len(set(self.locations.values())) == len(self.locations),
                 "continuation physical locators are ambiguous")

    def _map(self, path: Path, *, reverse: bool = False) -> Path:
        pairs = [(target, Path(old)) if reverse else (Path(old), target)
                 for old, target in self.locations.items()]
        matches = [(old, target) for old, target in pairs if path.is_relative_to(old)]
        _require(bool(matches), "path has no continuation locator")
        old, target = max(matches, key=lambda pair: len(pair[0].parts))
        return target / path.relative_to(old)

    def physical_path(self, logical: str | Path) -> Path:
        path = _absolute(logical)
        result = self._map(path)
        target = max((root for root in self.locations.values()
                      if result.is_relative_to(root)), key=lambda root: len(root.parts))
        _require(self._map(result, reverse=True) == path
                 and result.resolve().is_relative_to(target.resolve()),
                 "continuation physical path escaped its locator")
        return result

    def logical_path(self, physical: str | Path) -> str:
        path = _absolute(physical)
        logical = self._map(path, reverse=True)
        _require(self.physical_path(logical) == path,
                 "continuation physical path escaped its locator")
        return str(logical)

    @property
    def request_governor_root(self) -> Path:
        internal = [self.physical_root / _relative(ref)
                    for ref in self.value["ledger_anchors"]]
        external = [self.physical_path(logical)
                    for logical in self.value["external_ledger_anchors"]]
        _require(len(internal) + len(external) == 1,
                 "continuation requires one unambiguous request ledger")
        ledger = (internal + external)[0]
        _require(ledger.name == "state.json", "continuation request ledger must be state.json")
        return ledger.parent

    def _changes(self, original: Mapping, current: Mapping) -> list[dict]:
        changes = []

        def revision_change(old, new, fields, *, old_present=True):
            if fields == ["execution", "reviser_concurrency"]:
                owners = current.get("execution", {}).get("skill_owners")
                _require(type(owners) is int and 1 <= owners <= 19
                         and type(new) is int and 1 <= new <= owners
                         and current["execution"].get("scheduler")
                         == original["execution"].get("scheduler")
                         == "coordinator_owned_episode_spawn_pool_v1"
                         and current.get("evolution") is not None,
                         "unbounded reviser concurrency configuration")
                changes.append({
                    "fields": fields, "kind": "reviser_concurrency",
                    "old_present": old_present, "old": old, "new": new,
                })
                return True
            allowed = (len(fields) == 3 and fields[:2] in (
                ["evolution", "config"], ["request_input_config", "evolution_config"])
                and fields[2] in _REVISION_INPUT_LIMITS)
            if not allowed:
                return False
            minimum, maximum = _REVISION_INPUT_LIMITS[fields[2]]
            _require(type(new) is int and minimum <= new <= maximum,
                     f"unbounded revision protocol configuration: {fields}")
            changes.append({
                "fields": fields, "kind": "revision_protocol", "old_present": old_present,
                "old": old, "new": new, "protocol": self.value["revision_protocol"]["current"],
            })
            return True

        def walk(old, new, fields):
            if old == new:
                return
            if revision_change(old, new, fields):
                return
            if fields == ["code_identity"]:
                _require(isinstance(old, dict) and isinstance(new, dict) and bool(new),
                         "continuation requires a real new source identity")
                changes.append({"fields": fields, "kind": "code", "old": old, "new": new})
                return
            if isinstance(old, dict) and isinstance(new, dict):
                for key in sorted(old.keys() | new.keys()):
                    child_fields = [*fields, key]
                    if key not in old:
                        _require(revision_change(None, new[key], child_fields, old_present=False),
                                 f"unauthorized continuation manifest addition: {child_fields}")
                    else:
                        _require(key in new,
                                 f"unauthorized continuation manifest deletion: {child_fields}")
                        walk(old[key], new[key], child_fields)
                return
            if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
                for index, (before, after) in enumerate(zip(old, new)):
                    walk(before, after, [*fields, index])
                return
            _require(fields[:1] == ["input_contract_identity"]
                     and isinstance(old, str) and isinstance(new, str),
                     f"unauthorized continuation manifest difference: {fields}")
            _require(str(self.physical_path(old)) == new,
                     f"continuation location difference is not mapped: {fields}")
            changes.append({"fields": fields, "kind": "location", "old": old, "new": new})

        walk({key: value for key, value in original.items() if key != "identity_sha256"},
             {key: value for key, value in current.items() if key != "identity_sha256"}, [])
        return changes

    def validate(self, *, system_root: Path, current_manifest: Mapping,
                 revision_protocol: str, framework_state_io=None) -> dict:
        _require(self.path.read_bytes() == self.raw, "immutable continuation sidecar changed")
        _require(system_root == self.physical_root, "continuation system locator changed")
        original_data = (system_root / "manifest.json").read_bytes()
        original = json.loads(original_data)
        _require(_identity(original_data) == self.value["original_manifest_file"]
                 and _manifest_identity(original) == self.value["run_identity"],
                 "continuation original manifest changed")
        _require(_manifest_identity(current_manifest) == self.value["current_manifest_sha256"]
                 and current_manifest["code_identity"] == self.value["source_identity"],
                 "continuation executing source or manifest changed")
        _require(self.value["revision_protocol"]["current"] == revision_protocol
                 and bool(self.value["revision_protocol"]["original"]),
                 "continuation revision protocol changed")
        _require(self._changes(original, current_manifest) == self.value["allowed_changes"],
                 "continuation allowed differences changed")
        files = self.value["preserved_files"]
        _require("manifest.json" in files
                 and bool(self.value["ledger_anchors"] or self.value["external_ledger_anchors"]),
                 "continuation lacks original state or ledger anchors")
        _require(set(self.value["ledger_anchors"]) <= set(files)
                 and any(re.fullmatch(r"workers/[0-9]+/buffer/stream.json", ref)
                         for ref in files),
                 "continuation lacks preserved worker state")
        marker = self.path.parent / "activated.json"
        activation = {"continuation_identity_sha256": self.identity_sha256,
                      "run_identity": self.value["run_identity"]}
        activated = marker.exists()
        if activated:
            _require(json.loads(marker.read_bytes()) == activation,
                     "continuation activation identity changed")
        cold_records = {}
        history = getattr(framework_state_io, "_cold_history", None)
        if history is not None:
            with framework_state_io.section() as io:
                cold_records = history.records(budget=io._budget)
        for ref, expected in files.items():
            archive_ref = expected["archive_ref"]
            candidate = system_root / ref
            if archive_ref is None and cold_records and not candidate.exists():
                cold = framework_state_io.cold_reference(candidate, records=cold_records)
                if cold is not None:
                    _require(cold.identity == expected["identity"],
                             f"continuation cold file identity changed: {ref}")
                    continue
            live = _checked_file(system_root, ref).read_bytes()
            if re.fullmatch(r"workers/[0-9]+/buffer/stream.json", ref):
                _require(archive_ref == f"raw_state/{ref}",
                         "continuation requires original stream bytes")
            if ref in self.value["ledger_anchors"]:
                _require(archive_ref == f"raw_state/{ref}",
                         "continuation requires original ledger bytes")
            if archive_ref is None:
                _require(_identity(live) == expected["identity"],
                         f"continuation original immutable file changed: {ref}")
                continue
            archived = _checked_file(self.path.parent, archive_ref).read_bytes()
            _require(_identity(archived) == expected["identity"],
                     f"continuation raw state archive changed: {ref}")
            if not activated:
                _require(live == archived, f"continuation initial raw state changed: {ref}")
            elif ref in self.value["ledger_anchors"]:
                self._validate_ledger(archived, live)
            else:
                before, after = json.loads(archived), json.loads(live)
                _require(before["stream_id"] == after["stream_id"]
                         and before["config"] == after["config"]
                         and after["attempts"][:len(before["attempts"])] == before["attempts"]
                         and set(before["seen_case_ids"]) <= set(after["seen_case_ids"])
                         and set(before["slots"]) <= set(after["slots"]),
                         f"continuation original stream lineage was lost: {ref}")
                for slot_key, old_slot in before["slots"].items():
                    new_slot = after["slots"][slot_key]
                    if old_slot["parent_content_sha256"] == new_slot["parent_content_sha256"]:
                        _require(new_slot["attempt_count"] >= old_slot["attempt_count"]
                                 and set(old_slot["case_refs"]) <= set(new_slot["case_refs"]),
                                 "continuation original parent buffer was reset")
                        if new_slot["attempt_count"] == old_slot["attempt_count"]:
                            _require(new_slot["new_since_attempt"] >= old_slot["new_since_attempt"]
                                     and new_slot["attempted_ever"] == old_slot["attempted_ever"],
                                     "continuation advanced a buffer without an attempt receipt")
            if re.fullmatch(r"workers/[0-9]+/buffer/stream.json", ref):
                worker_id = int(PurePosixPath(ref).parts[1])
                state = json.loads(archived)
                worker_ref = ref.replace("buffer/stream.json", "worker_identity.json")
                _require(worker_ref in files
                         and state["stream_id"] == (
                             f"fit-worker-{worker_id:02d}-{self.value['run_identity'][:12]}")
                         and json.loads(_checked_file(system_root, worker_ref).read_bytes())[
                             "run_identity"] == self.value["run_identity"],
                         "continuation original worker lineage changed")
        for logical, expected in self.value["external_ledger_anchors"].items():
            physical = self.physical_path(logical)
            _require(set(expected) == {"identity", "archive_ref"}
                     and not physical.is_relative_to(system_root) and physical.is_file(),
                     "continuation external ledger locator changed")
            archived = _checked_file(
                self.path.parent, _archive_ref(expected["archive_ref"])).read_bytes()
            live = physical.read_bytes()
            _require(_identity(archived) == expected["identity"],
                     "continuation external ledger raw archive changed")
            if not activated:
                _require(live == archived, "continuation initial external ledger changed")
            else:
                self._validate_ledger(archived, live)
        if framework_state_io is None:
            _write_create_once(marker, activation)
        else:
            framework_state_io.write_json_once(marker, activation)
        return original

    @staticmethod
    def _validate_ledger(original: bytes, live: bytes) -> None:
        before, after = json.loads(original), json.loads(live)
        _require(before["policy"] == after["policy"]
                 and len(after["requests"]) >= len(before["requests"]),
                 "continuation ledger budget or reservations were reset")
        _require(after["requests"][:len(before["requests"])] == before["requests"],
                 "continuation original ledger anchor changed; use a separate recovery receipt")


def write_deployment_continuation(
    path: str | Path, *, system_root: str | Path, logical_root: str | Path,
    deployment_id: str, current_manifest: Mapping, original_revision_protocol: str,
    preserved_paths: Sequence[str], ledger_paths: Sequence[str] = (),
    path_map: Mapping[str, str] | None = None,
    external_ledger_anchors: Mapping[str, Mapping] | None = None,
) -> dict:
    """Pin explicit fixed files and raw streams, not summaries/library pointers.

    External ledger identities may come from a separately verified immutable
    archive. Only their explicitly mapped live files are read here.
    """
    from skill3d.evolution.fitset_revision_v11 import FIT_REVISION_PROMPT_VERSION

    root = _absolute(system_root)
    destination = Path(path).resolve()
    _require(bool(_SAFE_ID.fullmatch(deployment_id)), "unsafe continuation deployment_id")
    _require(destination.parent == root / "deployments" / deployment_id,
             "sidecar must live in the independent deployment directory")
    refs = {_relative(ref) for ref in preserved_paths} | {"manifest.json"}
    ledgers = sorted({_relative(ref) for ref in ledger_paths
                      if not Path(ref).is_absolute() and ".." not in PurePosixPath(ref).parts})
    _require(bool(ledger_paths or external_ledger_anchors) and set(ledgers) <= refs,
             "internal ledger anchors must be included in the preserved file set")
    streams = {ref for ref in refs if re.fullmatch(r"workers/[0-9]+/buffer/stream.json", ref)}
    _require(bool(streams), "continuation requires raw worker stream state")
    _require({path.relative_to(root).as_posix()
              for path in (root / "workers").glob("*/buffer/stream.json")} <= streams,
             "continuation omitted an original worker stream")
    for ref in streams:
        _require(ref.replace("buffer/stream.json", "worker_identity.json") in refs,
                 "continuation requires the original worker identity")
    original_data = _checked_file(root, "manifest.json").read_bytes()
    original = json.loads(original_data)
    _require(bool(original_revision_protocol), "continuation requires the original revision protocol")
    for ref in refs:
        if re.fullmatch(r"workers/[0-9]+/campaigns/[^/]+/revision/slots/[^/]+/[^/]+/"
                        r"request_intent_[^/]+.json", ref):
            intent = json.loads(_checked_file(root, ref).read_bytes())
            _require(intent["request"]["prompt_version"] == original_revision_protocol,
                     "original revision protocol does not match the preserved intent")
    body = {
        "schema_version": DEPLOYMENT_CONTINUATION_SCHEMA, "deployment_id": deployment_id,
        "logical_root": str(_absolute(logical_root)), "physical_root": str(root),
        "path_map": {**dict(path_map or {}), str(logical_root): str(root)},
        "run_identity": _manifest_identity(original),
        "original_manifest_file": _identity(original_data),
        "current_manifest_sha256": _manifest_identity(current_manifest),
        "source_identity": copy.deepcopy(current_manifest["code_identity"]),
        "revision_protocol": {"original": original_revision_protocol,
                              "current": FIT_REVISION_PROMPT_VERSION},
        "ledger_anchors": ledgers,
    }
    probe = object.__new__(DeploymentContinuation)
    probe.value = body
    probe.locations = {str(_absolute(old)): _absolute(new)
                       for old, new in body["path_map"].items()}
    body["allowed_changes"] = probe._changes(original, current_manifest)
    external = copy.deepcopy(dict(external_ledger_anchors or {}))
    for logical, expected in external.items():
        _absolute(logical)
        _require(set(expected) == {"identity", "archive_ref"},
                 "external ledger requires identity and archive_ref")
        _archive_ref(expected["archive_ref"])
        physical = probe.physical_path(logical)
        _require(not physical.is_relative_to(root) and physical.is_file()
                 and _identity(physical.read_bytes()) == expected["identity"],
                 "external ledger differs from its supplied immutable anchor")
    for ref in sorted(set(ledger_paths) - set(ledgers)):
        logical = (str(_absolute(ref)) if Path(ref).is_absolute() else
                   os.path.normpath(str(_absolute(logical_root) / ref)))
        physical = probe.physical_path(logical)
        _require(not physical.is_relative_to(root) and physical.is_file(),
                 "external ledger must use an explicit external locator")
        data = physical.read_bytes()
        inferred = {"identity": _identity(data),
                    "archive_ref": f"raw_state/external/{content_sha256(logical)}.json"}
        if logical in external:
            _require(external[logical]["identity"] == inferred["identity"],
                     "conflicting external ledger anchors")
        else:
            external[logical] = inferred
    _require(len(ledgers) + len(external) == 1,
             "continuation requires one unambiguous request ledger")
    _require(all(PurePosixPath(ref).name == "state.json" for ref in [*ledgers, *external]),
             "continuation request ledger must be state.json")
    files = {}
    for ref in sorted(refs):
        data = _checked_file(root, ref).read_bytes()
        archive_ref = f"raw_state/{ref}" if ref in streams or ref in ledgers else None
        files[ref] = {"identity": _identity(data), "archive_ref": archive_ref}
        if archive_ref is not None:
            _write_raw_state_once(destination.parent, archive_ref, data)
    body["preserved_files"] = files
    body["external_ledger_anchors"] = external
    for logical, expected in external.items():
        data = probe.physical_path(logical).read_bytes()
        _require(_identity(data) == expected["identity"], "external ledger changed during sealing")
        _write_raw_state_once(destination.parent, expected["archive_ref"], data)
    value = {**body, "identity_sha256": content_sha256(body)}
    _write_create_once(destination, value)
    return value


def revision_intents_used(worker_root: Path, slot_key: str) -> int:
    """Count an intent once, never its exchange; identical copied intents alias."""
    identities = set()
    for request_path in sorted((worker_root / "campaigns").glob("*/request.json")):
        request = json.loads(_checked_file(
            worker_root, request_path.relative_to(worker_root).as_posix()).read_bytes())
        if request["slot"] != slot_key:
            continue
        campaign = request_path.parent
        namespaces = [campaign, *sorted((campaign / "deployments").glob("*"))]
        for namespace in namespaces:
            paths = [*namespace.glob("revision/slots/*/*/request_intent_*.json"),
                     *namespace.glob("revision/slots/*/*/*/request_intent_*.json"),
                     *namespace.glob("diagnosis/*/request_intent.json")]
            for path in paths:
                _require(path.resolve().is_relative_to(worker_root.resolve()),
                         "revision intent escaped the worker locator")
                intent = json.loads(path.read_bytes())
                if path.name == "request_intent.json":
                    _require(bool(intent.get("input_sha256"))
                             and isinstance(intent.get("request_identity"), dict),
                             "invalid diagnosis request intent")
                    identity = {"intent": intent,
                                "logical_intent_ref": path.relative_to(worker_root).as_posix()}
                elif path.parent.name in {"analysis", "synthesis"}:
                    _require(bool(intent.get("prompt_sha256"))
                             and isinstance(intent.get("request_identity"), dict)
                             and type(intent.get("ordinal")) is int,
                             "invalid analysis/synthesis request intent")
                    identity = {"slot": slot_key, "intent": intent,
                                "logical_intent_ref": path.relative_to(worker_root).as_posix()}
                else:
                    _require(isinstance(intent.get("request"), dict)
                             and bool(intent.get("started_at")), "invalid revision request intent")
                    identity = {"slot": slot_key, "intent": intent}
                identities.add(content_sha256(identity))
    return len(identities)
