"""Reviewed, cache-only replay of an interrupted solver's original calls.

This is not ordinary resume and must NOT be wrapped in GovernedClient or
SeededGateClient: both write the original request history. The owner injects
frozen dispatch/input/skill evidence and a reviewed mapping from seed indices
and governor integers to gateway UUIDs. No model client is constructed here.
``finish_replay`` settles recovered responses, not episode answers or labels.
"""

from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
from types import SimpleNamespace
from typing import Mapping, Sequence
import uuid

from skill3d.cluster.perception_client import (
    MAX_BODY_BYTES, RPC_SCHEMA, json_object, validate_result,
)
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.request_audit import describe_request, payload_bytes
from skill3d.synthesis.request_context import current_request_phase

RECOVERY_SCHEMA = "harness3d.cached-episode-recovery/1.0"
PREFIX_RESUME_SCHEMA = "harness3d.cached-prefix-resume/1.0"
_FROZEN_KEYS = {
    "run_identity", "shared_input_sha256", "skill_identity", "skill_content_sha256",
    "solver_config_sha256", "concurrency_limit", "scheduler",
}
_REQUEST_KEYS = {
    "round_index", "phase", "prompt_version", "execution_contract_sha256",
    "tool_docs_version", "tool_directory_sha256", "tool_execution_allowed",
    "delivered_method_hashes", "image_round", "images_in_order",
}


class RecoveryStop(BaseException):
    """A terminal refusal, deliberately outside normal model-error recovery."""


def evidence_sha256(value: object) -> str:
    """Fit-set canonical hash for already JSON-serialized engineering evidence."""
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise RecoveryStop(code)


def _equal(left: object, right: object) -> bool:
    # Canonical JSON also distinguishes bool/int, unlike Python dict equality.
    return evidence_sha256(left) == evidence_sha256(right)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))

def _skill_content_sha256(skill: Mapping) -> str:
    if "skill_md" in skill:
        from skill3d.schemas import SkillSpecV11
        computed = SkillSpecV11.model_validate({
            key: value for key, value in skill.items() if key != "content_sha256"
        }).content_sha256
        _require("content_sha256" not in skill or skill["content_sha256"] == computed,
                 "skills_mismatch")
        return computed
    # Compact synthetic engineering fixtures may supply just an explicit digest.
    _require(set(skill) == {"content_sha256"}
             and _digest(skill.get("content_sha256")), "skills_mismatch")
    return skill["content_sha256"]


def _wire_model_identity(identity: Mapping, *, seed: int | None) -> dict:
    # Seed audit JSON sorts keys; gateway identity hashes use Bailian's wire order.
    options = SimpleNamespace(
        provider=identity["provider"], model=identity["model"],
        model_revision=identity["model_revision"], endpoint_hash=identity["endpoint_sha256"],
        temperature=identity["temperature"], max_output_tokens=identity["max_tokens"],
    )
    return BailianClient.audit_identity(options, seed=seed)


def _create_once(path: Path, value: Mapping, framework_state_io=None) -> None:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    if framework_state_io is not None:
        framework_state_io.write_bytes(path, raw + b"\n", exclusive=True)
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class ReviewedSolverCall:
    """Original records plus explicit review; seed records never supply text.

    ``approval`` requires decision=approved_cache_replay, review_id, hashes of
    seed_record/governor_record/response_record (including None), and ``source``.
    Source requires kind, request_id (gateway UUID), payload_sha256,
    identity_sha256, response_identity and usage. gateway_cache additionally
    requires reply_sha256; round_trace requires trace_ref and record_sha256.
    A completed call requires its original SeededGateClient response record;
    the last call may instead be started and lack that record, but still needs
    reviewed source evidence. A completed prefix may precede unfinished execution.
    """

    seed_record: Mapping
    governor_record: Mapping
    approval: Mapping | None
    response_record: Mapping | None = None
    round_record: Mapping | None = None


class GatewaySQLiteCache:
    """Only SELECT from the existing gateway schema, with no HTTP fallback.

    payload_sha256 and identity_sha256 are the existing requests table columns.
    reply_sha256 is separate reviewed SHA-256(reply BLOB), NOT the request hash.
    upstream_audit is historical evidence and is never modified or required here.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def read(self, source: Mapping) -> dict:
        _require(self.path.is_file() and not self.path.is_symlink(), "cache_unavailable")
        try:
            # Never create a DB, acquire the gateway owner lock, or repair pending rows.
            uri = self.path.resolve().as_uri() + "?mode=ro"
            with closing(sqlite3.connect(uri, uri=True, timeout=0)) as db:
                db.execute("PRAGMA query_only=ON")
                row = db.execute(
                    "SELECT payload_sha256, identity_sha256, state, http_status, reply "
                    "FROM requests WHERE request_id=?", (source["request_id"],),
                ).fetchone()
            _require(row is not None, "cache_miss")
            _require(row[0] == source["payload_sha256"]
                     and row[1] == source["identity_sha256"], "cache_identity_mismatch")
            _require(row[2] == "complete" and row[3] == 200, "cache_not_successful")
            _require(isinstance(row[4], bytes) and 0 < len(row[4]) <= MAX_BODY_BYTES,
                     "cache_invalid_reply")
            _require(_sha(row[4]) == source["reply_sha256"], "cache_reply_hash_mismatch")
            return json_object(row[4])
        except (sqlite3.Error, OSError, ValueError, TypeError, KeyError, UnicodeError):
            raise RecoveryStop("cache_invalid") from None


def _round_result(call: ReviewedSolverCall, identity: Mapping) -> dict:
    source = call.approval["source"]
    record = call.round_record
    _require(isinstance(record, Mapping), "round_trace_missing")
    _require(evidence_sha256(record) == source["record_sha256"],
             "round_trace_hash_mismatch")
    _require(record["index"] == call.seed_record["request_context"]["round_index"],
             "round_trace_index_mismatch")
    generation = deepcopy(record["generation"])
    usage = generation.pop("usage")
    return {
        "schema_version": RPC_SCHEMA, "request_id": source["request_id"],
        "payload_sha256": source["payload_sha256"], "audit_identity": dict(identity),
        "content": record["response_text"], "last_usage": usage,
        "last_generation": generation,
        "last_response_identity": generation["response_identity"],
        "last_request_identity": generation["request_identity"],
    }


class CachedEpisodeRecoveryClient:
    """The normal Bailian ``chat``/audit/context contract, without transport.

    Use ``open_cached_episode_recovery``. Supply original frozen context to
    set_audit_context and let normal session supply per-round request context.
    chat applies the original SeededGateClient seed override and compares the
    full effective request audit before retrieving any text. There is no
    delegate, opener, environment lookup, new UUID, budget reservation or retry.
    """

    def __init__(self, *, dispatch: Mapping, frozen_context: Mapping,
                 model_identity: Mapping, calls: Sequence[ReviewedSolverCall],
                 cache: GatewaySQLiteCache | None, continuation_root: Path,
                 binding_sha256: str, framework_state_io=None):
        self._dispatch = deepcopy(dict(dispatch))
        self._frozen_context = deepcopy(dict(frozen_context))
        self._model_identity = deepcopy(dict(model_identity))
        self._calls = deepcopy(tuple(calls))
        self._cache = cache
        self._root = continuation_root
        self.continuation_trace_dir = continuation_root / "trace"
        self._binding_sha256 = binding_sha256
        self._framework_state_io = framework_state_io
        self._audit_context = None
        self._request_context = None
        self._index = 0
        self._stopped = False
        self._finished = False
        self._replayed = []
        self.model = self._model_identity["model"]
        self.temperature = self._model_identity["temperature"]
        self.max_output_tokens = self._model_identity["max_tokens"]
        self.last_usage = {}
        self.last_generation = {}
        self.last_response_identity = {}
        self.last_request_identity = {}

    @property
    def configured(self) -> bool:
        return not self._stopped and not self._finished

    def check_service(self) -> tuple[bool, str]:
        return self.configured, "reviewed cache-only replay; no model transport"

    def audit_identity(self, *, seed: int | None = None) -> dict:
        return _wire_model_identity(self._model_identity, seed=seed)

    def build_payload(self, messages: list[dict], max_tokens: int = 2048,
                      seed: int | None = None, temperature: float | None = None,
                      response_format: dict | None = None, enable_thinking: bool = False,
                      top_p: float | None = None) -> dict:
        return BailianClient.build_payload(
            self, messages, max_tokens=max_tokens, seed=seed, temperature=temperature,
            response_format=response_format, enable_thinking=enable_thinking, top_p=top_p,
        )

    def set_audit_context(self, context: Mapping) -> None:
        self._audit_context = deepcopy(dict(context))

    def set_request_context(self, context: Mapping) -> None:
        self._request_context = deepcopy(dict(context))

    def _stop(self, code: str) -> None:
        self._stopped = True
        self.last_usage = {}
        self.last_generation = {}
        self.last_response_identity = {}
        self.last_request_identity = {}
        try:
            self._write_once(self._root / "stopped.json", {
                "schema_version": RECOVERY_SCHEMA, "code": code,
                "replayed_calls": self._index, "new_requests": 0,
            })
        except OSError:
            pass
        raise RecoveryStop(code)

    def _write_once(self, path: Path, value: Mapping) -> None:
        _create_once(path, value, self._framework_state_io)

    def chat(self, messages: list[dict], max_tokens: int = 2048,
             seed: int | None = None, temperature: float | None = None,
             response_format: dict | None = None, enable_thinking: bool = False,
             top_p: float | None = None) -> str:
        try:
            _require(not self._stopped and not self._finished, "recovery_closed")
            _require(self._index < len(self._calls), "unreviewed_call")
            call = self._calls[self._index]
            original = call.seed_record
            _require(_equal(self._audit_context, self._frozen_context),
                     "frozen_context_mismatch")
            _require(_equal(self._request_context, original["request_context"]),
                     "request_context_mismatch")
            _require(current_request_phase() == original["request_context"]["phase"],
                     "request_phase_mismatch")
            # SeededGateClient overwrote cfg.seed; reproducing that is not resampling.
            effective_seed = original["request_seed"]
            payload = self.build_payload(
                messages, max_tokens=max_tokens, seed=effective_seed, temperature=temperature,
                response_format=response_format, enable_thinking=enable_thinking, top_p=top_p,
            )
            identity = self.audit_identity(seed=effective_seed)
            audit = describe_request(payload, model_identity=identity)
            _require(_equal(audit, original["request_identity"]), "request_hash_mismatch")
            source = call.approval["source"]
            if source["kind"] == "gateway_cache":
                _require(self._cache is not None, "cache_unavailable")
                result = self._cache.read(source)
            else:
                result = _round_result(call, identity)
            validate_result(result, request_id=source["request_id"],
                            payload=payload, audit_identity=identity)
            _require(_equal(result["audit_identity"], identity)
                     and _equal(result["last_request_identity"], audit)
                     and _equal(result["last_generation"]["request_identity"], audit),
                     "response_request_identity_mismatch")
            _require(_equal(result["last_response_identity"], source["response_identity"])
                     and _equal(result["last_usage"], source["usage"]),
                     "reviewed_response_mismatch")
            response = call.response_record
            if response is not None:
                _require(_equal(response["actual_request_identity"], audit)
                         and _equal(response["response_identity"], result["last_response_identity"])
                         and _equal(response["usage"], result["last_usage"]),
                         "seed_response_mismatch")
            replay = {
                "schema_version": RECOVERY_SCHEMA, "request_index": self._index + 1,
                "governor_request_id": call.governor_record["request_id"],
                "original_status": call.governor_record["status"],
                "governor_record_sha256": evidence_sha256(call.governor_record),
                "review_sha256": evidence_sha256(call.approval),
                "source": deepcopy(source), "request_identity_sha256": evidence_sha256(audit),
                "response_text_sha256": _sha(result["content"].encode("utf-8")),
                "usage": deepcopy(result["last_usage"]), "new_requests": 0,
            }
            self._write_once(
                self.continuation_trace_dir / f"replayed-{self._index + 1:03d}.json", replay)
            self._replayed.append(replay)
            self._index += 1
            self.last_usage = deepcopy(result["last_usage"])
            self.last_generation = deepcopy(result["last_generation"])
            self.last_response_identity = deepcopy(result["last_response_identity"])
            self.last_request_identity = deepcopy(result["last_request_identity"])
            return result["content"]
        except RecoveryStop as exc:
            self._stop(str(exc))
        except (OSError, ValueError, TypeError, KeyError, IndexError, UnicodeError, OverflowError):
            self._stop("invalid_replay_evidence")

    def finish_replay(self) -> dict:
        """Create one explicit settlement receipt; never alter a governor row."""
        try:
            _require(not self._stopped and not self._finished, "recovery_closed")
            _require(self._index == len(self._calls), "replay_incomplete")
            receipt = {
                "schema_version": RECOVERY_SCHEMA, "status": "cache_replay_complete",
                "settlement": "response_recovered", "binding_sha256": self._binding_sha256,
                "variant_id": self._dispatch["variant_id"],
                "run_identity": self._dispatch["run_identity"],
                "replayed_calls": deepcopy(self._replayed), "new_requests": 0,
                "recovered_started_request_ids": [
                    row["governor_request_id"] for row in self._replayed
                    if row["original_status"] == "started"],
            }
            receipt["identity_sha256"] = evidence_sha256(receipt)
            self._write_once(self._root / "recovery_receipt.json", receipt)
            self._finished = True
            return receipt
        except RecoveryStop as exc:
            self._stop(str(exc))
        except (OSError, ValueError, TypeError, KeyError):
            self._stop("receipt_write_failed")

    def recover_original_responses(self) -> dict:
        """Recover reviewed gateway replies without reexecuting an episode.

        The opener already requires context rebuilt from actual inputs and the
        original dispatch, Skills, seeds and ledger. This mode verifies the
        original effective request audit against each gateway reply. It does
        not claim that prompts or subsequent tool execution were reproduced.
        The original trace and request ledger remain immutable.
        """
        try:
            _require(not self._stopped and not self._finished and self._index == 0,
                     "recovery_closed")
            verified = []
            for index, call in enumerate(self._calls, 1):
                source = call.approval["source"]
                _require(source["kind"] == "gateway_cache" and self._cache is not None,
                         "original_gateway_source_required")
                result = self._cache.read(source)
                audit = call.seed_record["request_identity"]
                model = _wire_model_identity(
                    self._model_identity, seed=call.seed_record["request_seed"])
                _require(result["schema_version"] == RPC_SCHEMA
                         and result["request_id"] == source["request_id"]
                         and result["payload_sha256"] == source["payload_sha256"]
                         and _equal(result["audit_identity"], model)
                         and _equal(result["last_request_identity"], audit)
                         and _equal(result["last_generation"]["request_identity"], audit),
                         "response_request_identity_mismatch")
                _require(_equal(result["last_response_identity"], source["response_identity"])
                         and _equal(result["last_generation"]["response_identity"],
                                    source["response_identity"])
                         and _equal(result["last_usage"], source["usage"]),
                         "reviewed_response_mismatch")
                response = call.response_record
                if response is not None:
                    _require(_equal(response["actual_request_identity"], audit)
                             and _equal(response["response_identity"], result["last_response_identity"])
                             and _equal(response["usage"], result["last_usage"]),
                             "seed_response_mismatch")
                _require(isinstance(result["content"], str), "cache_invalid_content")
                verified.append({
                    "schema_version": RECOVERY_SCHEMA, "request_index": index,
                    "governor_request_id": call.governor_record["request_id"],
                    "original_status": call.governor_record["status"],
                    "governor_record_sha256": evidence_sha256(call.governor_record),
                    "review_sha256": evidence_sha256(call.approval),
                    "source": deepcopy(source),
                    "request_identity_sha256": evidence_sha256(audit),
                    "response_text_sha256": _sha(result["content"].encode("utf-8")),
                    "usage": deepcopy(result["last_usage"]), "new_requests": 0,
                })
            receipt = {
                "schema_version": RECOVERY_SCHEMA,
                "status": "original_cached_responses_recovered",
                "recovery_mode": "verified_original_gateway_responses",
                "episode_execution_reproduced": False,
                "settlement": "response_recovered", "binding_sha256": self._binding_sha256,
                "variant_id": self._dispatch["variant_id"],
                "run_identity": self._dispatch["run_identity"],
                "replayed_calls": verified, "new_requests": 0,
                "recovered_started_request_ids": [
                    row["governor_request_id"] for row in verified
                    if row["original_status"] == "started"],
            }
            receipt["identity_sha256"] = evidence_sha256(receipt)
            self._write_once(self._root / "recovery_receipt.json", receipt)
            self._finished = True
            return receipt
        except RecoveryStop as exc:
            self._stop(str(exc))
        except (OSError, ValueError, TypeError, KeyError, UnicodeError):
            self._stop("invalid_original_response_custody")


def open_cached_episode_recovery(
    *, frozen_dispatch: Mapping, resumed_dispatch: Mapping,
    frozen_context: Mapping, resumed_context: Mapping,
    skills: Sequence[Mapping], model_identity: Mapping,
    calls: Sequence[ReviewedSolverCall], original_trace_dir: str | Path,
    continuation_root: str | Path, cache_path: str | Path | None = None,
    framework_state_io=None,
) -> CachedEpisodeRecoveryClient:
    """Validate supplied engineering evidence and claim a fresh continuation.

    The owner must compute resumed_context from actual prepared inputs/config,
    not simply copy frozen_context. skills are actual model_dump(mode="json")
    dictionaries in original order. No task, prompt, image or label is read.
    Each approval is an explicit review artifact supplied by the owner, not a
    signature or a substitute for their source/ledger custody audit.
    """
    try:
        _require(_equal(frozen_dispatch, resumed_dispatch), "dispatch_mismatch")
        _require(set(frozen_dispatch) == {
            "variant_id", "slot_key", "skill_identity", "skills_sha256", "run_identity",
        }, "dispatch_fields_missing")
        _require(all(isinstance(frozen_dispatch[key], str) and frozen_dispatch[key]
                     for key in ("variant_id", "slot_key", "run_identity")),
                 "dispatch_identity_missing")
        _require(set(frozen_context) == _FROZEN_KEYS
                 and _equal(frozen_context, resumed_context), "frozen_context_mismatch")
        _require(frozen_dispatch["run_identity"] == frozen_context["run_identity"]
                 and _equal(frozen_dispatch["skill_identity"], frozen_context["skill_identity"]),
                 "dispatch_context_mismatch")
        _require(evidence_sha256(list(skills)) == frozen_dispatch["skills_sha256"]
                 and _equal([_skill_content_sha256(skill) for skill in skills],
                            frozen_context["skill_content_sha256"]), "skills_mismatch")
        _require(all(_digest(frozen_context[key]) for key in (
            "shared_input_sha256", "solver_config_sha256")), "input_identity_missing")
        _require(set(model_identity) == {
            "enabled", "provider", "model", "model_revision", "endpoint_sha256",
            "seed", "temperature", "max_tokens", "cache_eligible",
        } and model_identity["enabled"] is True
            and model_identity["provider"] == "bailian"
            and type(model_identity["max_tokens"]) is int
            and model_identity["max_tokens"] > 0
            and type(model_identity["temperature"]) in (int, float)
            and math.isfinite(model_identity["temperature"])
            and type(model_identity["seed"]) is int
            and isinstance(model_identity["model"], str) and bool(model_identity["model"])
            and isinstance(model_identity["model_revision"], str)
            and _digest(model_identity["endpoint_sha256"]), "model_identity_invalid")
        _require(_equal(model_identity, _wire_model_identity(
            model_identity, seed=model_identity["seed"])), "model_identity_invalid")
        _require(bool(calls), "no_reviewed_calls")
        ids = set()
        gateway_ids = set()
        previous_id = -1
        for index, call in enumerate(calls, 1):
            request = call.seed_record
            governor = call.governor_record
            approval = call.approval
            _require(isinstance(approval, Mapping)
                     and approval.get("decision") == "approved_cache_replay"
                     and isinstance(approval.get("review_id"), str)
                     and bool(approval["review_id"]), "unreviewed_call")
            _require(approval["seed_record_sha256"] == evidence_sha256(request)
                     and approval["governor_record_sha256"] == evidence_sha256(governor)
                     and approval["response_record_sha256"] == evidence_sha256(call.response_record),
                     "review_record_hash_mismatch")
            _require(type(request["request_index"]) is int and request["request_index"] == index
                     and type(request["request_seed"]) is int
                     and request["request_seed"] == model_identity["seed"]
                     and request["retry_of"] is None,
                     "seed_call_identity_mismatch")
            _require(_equal(request["frozen_context"], frozen_context)
                     and set(request["request_context"]) == _REQUEST_KEYS
                     and request["request_context"]["phase"] in {"program", "finalization"},
                     "seed_context_mismatch")
            audit = request["request_identity"]
            expected_model = _wire_model_identity(model_identity, seed=request["request_seed"])
            _require(_equal(audit["model_identity"], expected_model)
                     and audit["parameters"]["seed"] == request["request_seed"]
                     and audit["parameter_status"] == "effective_payload"
                     and _digest(audit["wire_payload_sha256"]), "seed_audit_mismatch")
            _require(governor["category"] == "solver"
                     and _equal(governor["context"], {
                         "variant_id": frozen_dispatch["variant_id"],
                         "run_identity": frozen_dispatch["run_identity"],
                     }) and type(governor["request_id"]) is int
                     and governor["request_id"] > previous_id
                     and governor["request_id"] not in ids,
                     "governor_call_mismatch")
            ids.add(governor["request_id"])
            previous_id = governor["request_id"]
            expected_status = governor["status"]
            _require(expected_status == "completed" or (
                index == len(calls) and expected_status == "started"),
                "governor_status_mismatch")
            response = call.response_record
            _require(response is not None or expected_status == "started",
                     "seed_response_missing")
            source = approval["source"]
            _require(source["kind"] in {"gateway_cache", "round_trace"}, "source_not_allowed")
            gateway_id = source["request_id"]
            _require(str(uuid.UUID(gateway_id)) == gateway_id and gateway_id not in gateway_ids,
                     "gateway_call_mismatch")
            gateway_ids.add(gateway_id)
            _require(source["payload_sha256"] == audit["wire_payload_sha256"]
                     and source["identity_sha256"] == _sha(payload_bytes(expected_model))
                     and isinstance(source["response_identity"], Mapping)
                     and isinstance(source["usage"], Mapping), "review_source_identity_mismatch")
            _require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                         for key, value in source["usage"].items()
                         if key in {"prompt_tokens", "completion_tokens", "total_tokens"}),
                     "review_usage_invalid")
            if source["kind"] == "gateway_cache":
                _require(cache_path is not None and _digest(source["reply_sha256"]),
                         "cache_review_missing")
            else:
                _require(isinstance(source["trace_ref"], str) and bool(source["trace_ref"])
                         and _digest(source["record_sha256"]), "round_review_missing")
            if response is not None:
                _require(response["status"] == "completed"
                         and response["exception_type"] is None
                         and response["request_index"] == index
                         and response["request_seed"] == request["request_seed"]
                         and response["request_identity_sha256"] == evidence_sha256(audit),
                         "seed_response_identity_mismatch")
            if expected_status == "completed":
                measured = {key: value for key, value in source["usage"].items()
                            if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
                            and type(value) in (int, float)}
                _require(_equal(governor.get("usage"), measured), "governor_usage_mismatch")
        original = Path(original_trace_dir).resolve()
        continuation = Path(continuation_root).resolve()
        _require(original.is_dir(), "original_trace_missing")
        _require(original != continuation and original not in continuation.parents
                 and continuation not in original.parents, "continuation_overlaps_original")
        _require(not continuation.exists(), "continuation_already_exists")
        binding = {
            "schema_version": RECOVERY_SCHEMA, "dispatch": deepcopy(dict(frozen_dispatch)),
            "frozen_context": deepcopy(dict(frozen_context)),
            "model_identity": deepcopy(dict(model_identity)),
            "review_sha256s": [evidence_sha256(call.approval) for call in calls],
            "original_trace_dir": str(original), "new_requests": 0,
        }
        binding_hash = evidence_sha256(binding)
        if framework_state_io is None:
            continuation.mkdir(mode=0o700)
            (continuation / "trace").mkdir(mode=0o700)
        else:
            framework_state_io.mkdir(continuation)
            framework_state_io.mkdir(continuation / "trace")
        _create_once(continuation / "binding.json",
                     {**binding, "identity_sha256": binding_hash}, framework_state_io)
        return CachedEpisodeRecoveryClient(
            dispatch=frozen_dispatch, frozen_context=frozen_context,
            model_identity=model_identity, calls=calls,
            cache=GatewaySQLiteCache(cache_path) if cache_path is not None else None,
            continuation_root=continuation, binding_sha256=binding_hash,
            framework_state_io=framework_state_io,
        )
    except RecoveryStop:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        raise RecoveryStop("invalid_recovery_binding") from None


def recovery_accounting(governor_state: Mapping, receipts: Sequence[Mapping]) -> dict:
    """Read-only overlay: count original reservations once and settle started rows.

    The caller must ensure the original processes are stopped. This does not
    inspect OS permits or replace RequestGovernor.summary/state. Only newly
    recovered started usage is added; replayed completed usage is not duplicated.
    """
    try:
        rows = {row["request_id"]: row for row in governor_state["requests"]}
        _require(len(rows) == len(governor_state["requests"]), "duplicate_governor_id")
        recovered = set()
        added_usage = {}
        for receipt in receipts:
            unsigned = {key: value for key, value in receipt.items() if key != "identity_sha256"}
            _require(receipt["schema_version"] == RECOVERY_SCHEMA
                     and receipt["identity_sha256"] == evidence_sha256(unsigned)
                     and (receipt["status"] == "cache_replay_complete" or (
                         receipt["status"] == "original_cached_responses_recovered"
                         and receipt.get("recovery_mode") == "verified_original_gateway_responses"
                         and receipt.get("episode_execution_reproduced") is False))
                     and receipt["settlement"] == "response_recovered"
                     and receipt["new_requests"] == 0, "invalid_recovery_receipt")
            settled = []
            for replay in receipt["replayed_calls"]:
                row = rows[replay["governor_request_id"]]
                _require(evidence_sha256(row) == replay["governor_record_sha256"]
                         and row["status"] == replay["original_status"]
                         and row["category"] == "solver"
                         and _equal(row["context"], {
                             "variant_id": receipt["variant_id"],
                             "run_identity": receipt["run_identity"],
                         }) and replay["new_requests"] == 0, "receipt_governor_mismatch")
                if row["status"] == "started":
                    request_id = row["request_id"]
                    _require(request_id not in recovered, "duplicate_recovery_settlement")
                    recovered.add(request_id)
                    settled.append(request_id)
                    for key, value in replay["usage"].items():
                        if key in {"prompt_tokens", "completion_tokens", "total_tokens"}:
                            _require(type(value) in (int, float)
                                     and math.isfinite(value) and value >= 0,
                                     "invalid_recovered_usage")
                            added_usage[key] = added_usage.get(key, 0) + value
            _require(_equal(settled, receipt["recovered_started_request_ids"])
                     and bool(receipt["replayed_calls"])
                     and all(replay["original_status"] in {"completed", "started"}
                             for replay in receipt["replayed_calls"]),
                     "receipt_settlement_mismatch")
        return {
            "reserved_requests": len(rows), "recovered_requests": len(recovered),
            "completed_requests": sum(r["status"] == "completed" for r in rows.values()),
            "error_requests": sum(r["status"] == "error" for r in rows.values()),
            "unsettled_started_requests": sum(
                r["status"] == "started" and r["request_id"] not in recovered
                for r in rows.values()),
            "recovered_usage": added_usage, "new_requests": 0,
        }
    except RecoveryStop:
        raise
    except (ValueError, TypeError, KeyError, OverflowError):
        raise RecoveryStop("invalid_recovery_receipt") from None


def _private_file(path: Path) -> bytes:
    _require(path.is_absolute() and ".." not in path.parts
             and not any(parent.is_symlink() for parent in (path, *path.parents)),
             "recovery_file_locator_invalid")
    info = path.stat()
    _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
             and info.st_mode & 0o077 == 0, "recovery_file_not_owner_only")
    return path.read_bytes()


class CachedPrefixRecoverySpec:
    """Owner-only spec sealed by an already validated DeploymentContinuation.

    Exact JSON fields are enforced below. ``calls`` contains all five
    ReviewedSolverCall fields (use null for absent response/round records).
    ``original_plan_sha256`` hashes file bytes; ``identity_sha256`` hashes the
    canonical JSON object excluding that field. cache_path is a physical,
    standalone requests-only SQLite locator, not the original gateway database.
    Spec, custody receipt and DB must be preserved_files with archive_ref=null.
    """

    def __init__(self, path, *, continuation, system_root: Path, run_identity: str):
        try:
            _require(continuation is not None, "recovery_requires_deployment")
            self.path = Path(path)
            self.root = Path(system_root)
            self.continuation = continuation
            _require(self.root == continuation.physical_root, "recovery_system_locator_changed")
            self.raw = _private_file(self.path)
            self.value = json.loads(self.raw)
            value = self.value
            _require(set(value) == {
                "schema_version", "recovery_id", "deployment_id", "original_trace_dir",
                "original_plan_sha256", "frozen_dispatch", "frozen_context",
                "model_identity", "calls", "cache_path", "custody_receipt_ref",
                "identity_sha256",
            } and value["schema_version"] == PREFIX_RESUME_SCHEMA,
                     "recovery_spec_schema_invalid")
            _require(value["identity_sha256"] == evidence_sha256({
                key: child for key, child in value.items() if key != "identity_sha256"}),
                     "recovery_spec_hash_mismatch")
            _require(isinstance(value["recovery_id"], str)
                     and re.fullmatch(r"[A-Za-z0-9_-]+", value["recovery_id"])
                     and value["deployment_id"] == continuation.deployment_id
                     and value["frozen_dispatch"]["run_identity"] == run_identity
                     and continuation.value["run_identity"] == run_identity,
                     "recovery_spec_deployment_mismatch")
            _require(_digest(value["original_plan_sha256"]), "recovery_plan_hash_invalid")
            trace = Path(value["original_trace_dir"])
            _require(trace.is_absolute() and ".." not in trace.parts
                     and str(trace) == value["original_trace_dir"],
                     "recovery_trace_locator_invalid")
            self.original_trace_dir = continuation.physical_path(trace)
            self.cache_path = Path(value["cache_path"])
            _require(str(self.cache_path) == value["cache_path"],
                     "recovery_cache_locator_invalid")
            ref = value["custody_receipt_ref"]
            _require(isinstance(ref, str) and not Path(ref).is_absolute()
                     and ".." not in Path(ref).parts and Path(ref).as_posix() == ref
                     and ref != ".", "recovery_custody_locator_invalid")
            self.custody_path = self.root / ref
            self.custody = json.loads(_private_file(self.custody_path))
            _require(isinstance(value["calls"], list) and bool(value["calls"]),
                     "no_reviewed_calls")
            _require(all(set(row) == {
                "seed_record", "governor_record", "approval", "response_record", "round_record",
            } for row in value["calls"]), "recovery_call_schema_invalid")
            self.calls = tuple(ReviewedSolverCall(**row) for row in value["calls"])
            self.output_root = self.root / "solver_recoveries" / value["recovery_id"]
            self.verify_files()
            _require(self.custody.get("status") == "original_cached_responses_recovered"
                     and self.custody.get("episode_execution_reproduced") is False
                     and self.custody["variant_id"] == self.variant_id
                     and self.custody["run_identity"] == run_identity
                     and len(self.custody["replayed_calls"]) == len(self.calls),
                     "recovery_custody_mismatch")
            for index, (call, replay) in enumerate(zip(
                    self.calls, self.custody["replayed_calls"]), 1):
                _require(call.approval["source"]["kind"] == "gateway_cache"
                         and replay["request_index"] == index
                         and replay["governor_request_id"] == call.governor_record["request_id"]
                         and replay["governor_record_sha256"] == evidence_sha256(
                             call.governor_record)
                         and replay["review_sha256"] == evidence_sha256(call.approval)
                         and replay["request_identity_sha256"] == evidence_sha256(
                             call.seed_record["request_identity"])
                         and _equal(replay["source"], call.approval["source"])
                         and _equal(replay["usage"], call.approval["source"]["usage"]),
                         "recovery_custody_mismatch")
        except RecoveryStop:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            raise RecoveryStop("invalid_recovery_spec") from None

    @property
    def variant_id(self):
        return self.value["frozen_dispatch"]["variant_id"]

    def verify_files(self):
        _require(self.continuation.path.read_bytes() == self.continuation.raw,
                 "immutable_deployment_changed")
        for path in (self.path, self.custody_path, self.cache_path):
            raw = _private_file(path)
            ref = path.relative_to(self.root).as_posix()
            anchor = self.continuation.value["preserved_files"].get(ref)
            _require(isinstance(anchor, Mapping) and anchor["archive_ref"] is None
                     and _equal(anchor["identity"], {"sha256": _sha(raw), "size": len(raw)}),
                     "recovery_file_not_sealed")
        _require(self.path.read_bytes() == self.raw, "recovery_spec_changed")
        _require(not any(Path(str(self.cache_path) + suffix).exists()
                         for suffix in ("-wal", "-shm", "-journal")),
                 "recovery_cache_not_standalone")

    def audit_path(self):
        # Same key as fitset._variant_key, imported only inside the execution hook.
        from skill3d.evaluation.fitset_v11 import _variant_key
        return self.root / "request_seeds" / (_variant_key(self.variant_id) + ".json")

    def verify_original(self, *, ledger_path: Path, prepared_plan=None,
                        resumed_dispatch=None, resumed_context=None):
        """Read-only check, repeated before every new request and at completion."""
        self.verify_files()
        audit = self.audit_path()
        for index, call in enumerate(self.calls, 1):
            seed_path = audit.with_name(f"{audit.stem}-{index:03d}.json")
            response_path = audit.with_name(f"{audit.stem}-response-{index:03d}.json")
            _require(_equal(json.loads(seed_path.read_bytes()), call.seed_record),
                     "original_seed_changed")
            _require((call.response_record is None and not response_path.exists()) or (
                call.response_record is not None and response_path.is_file()
                and _equal(json.loads(response_path.read_bytes()), call.response_record)),
                     "original_seed_response_changed")
        dispatch_path = self.root / "dispatch" / (audit.stem + ".json")
        _require(_equal(json.loads(dispatch_path.read_bytes()), self.value["frozen_dispatch"]),
                 "original_dispatch_changed")
        plan_path = self.original_trace_dir.parent / "plan.json"
        original_plan = json.loads(plan_path.read_bytes())
        _require(_sha(plan_path.read_bytes()) == self.value["original_plan_sha256"],
                 "original_plan_changed")
        if prepared_plan is not None:
            _require(_equal(original_plan, prepared_plan), "prepared_plan_mismatch")
            _require(_equal(resumed_dispatch, self.value["frozen_dispatch"]),
                     "dispatch_mismatch")
            _require(_equal(resumed_context, self.value["frozen_context"]),
                     "frozen_context_mismatch")
        ledger = json.loads(ledger_path.read_bytes())
        target = [row for row in ledger["requests"]
                  if row["category"] == "solver" and _equal(row["context"], {
                      "variant_id": self.variant_id,
                      "run_identity": self.value["frozen_dispatch"]["run_identity"],
                  })]
        _require(_equal(target[:len(self.calls)], [
            call.governor_record for call in self.calls]), "original_ledger_changed")
        recovery_accounting(ledger, [self.custody])
        return ledger

    def require_next_seed_absent(self, index: int):
        audit = self.audit_path()
        _require(not any(path.exists() or path.is_symlink() for path in (
            audit.with_name(f"{audit.stem}-{index:03d}.json"),
            audit.with_name(f"{audit.stem}-response-{index:03d}.json"),
        )), "continuation_seed_already_exists")

    def require_no_future_seeds(self):
        audit = self.audit_path()
        pattern = re.compile(re.escape(audit.stem) + r"-(?:response-)?([0-9]+)\.json$")
        for path in audit.parent.glob(audit.stem + "-*.json"):
            match = pattern.fullmatch(path.name)
            _require(match is not None and int(match[1]) <= len(self.calls),
                     "continuation_seed_already_exists")

    def completed_result(self):
        """Only an execution receipt can promote a separate continuation result."""
        path = self.output_root / "continuation_receipt.json"
        if not path.exists():
            _require(not (self.output_root / "result.json").exists(),
                     "continuation_result_without_receipt")
            return None
        receipt = json.loads(path.read_bytes())
        _require(receipt.get("schema_version") == PREFIX_RESUME_SCHEMA
                 and receipt.get("status") == "episode_continuation_complete"
                 and receipt.get("identity_sha256") == evidence_sha256({
                     key: value for key, value in receipt.items() if key != "identity_sha256"})
                 and receipt["spec_identity_sha256"] == self.value["identity_sha256"]
                 and receipt["deployment_identity_sha256"] == self.continuation.identity_sha256
                 and receipt["custody_identity_sha256"] == self.custody["identity_sha256"]
                 and receipt["variant_id"] == self.variant_id
                 and receipt["cached_calls"] == len(self.calls)
                 and type(receipt["new_completed_calls"]) is int
                 and receipt["new_completed_calls"] >= 1, "invalid_continuation_receipt")
        ledger = self.verify_original(
            ledger_path=self.continuation.request_governor_root / "state.json")
        length = receipt["original_ledger_length"]
        _require(type(length) is int and length >= len(self.calls)
                 and evidence_sha256(ledger["requests"][:length])
                 == receipt["original_ledger_prefix_sha256"],
                 "original_ledger_prefix_changed")
        new_calls = self.continuation_calls(
            ledger, length=length, count=receipt["new_completed_calls"])
        _require(_equal(new_calls, receipt["new_calls"]), "continuation_call_evidence_changed")
        for name in ("result.json", "outcome.json", "evaluation_run.json", "recovery_receipt.json"):
            _require(_sha((self.output_root / name).read_bytes()) == receipt["files"][name],
                     "continuation_output_changed")
        replay = json.loads((self.output_root / "recovery_receipt.json").read_bytes())
        _require(_equal(replay["replayed_calls"], self.custody["replayed_calls"]),
                 "recovery_custody_mismatch")
        result = json.loads((self.output_root / "result.json").read_bytes())
        _require(result["variant_id"] == self.variant_id, "continuation_result_identity_changed")
        canonical = self.original_trace_dir.parent / "outcome.json"
        if canonical.exists():
            _require(_sha(canonical.read_bytes()) == receipt["files"]["outcome.json"],
                     "canonical_recovery_outcome_changed")
        return result

    def continuation_calls(self, ledger, *, length, count):
        _require(type(count) is int and count >= 1
                 and len(ledger["requests"]) >= length + count,
                 "continuation_ledger_incomplete")
        audit = self.audit_path()
        verified = []
        for offset in range(count):
            index = len(self.calls) + offset + 1
            row = ledger["requests"][length + offset]
            request = json.loads(audit.with_name(f"{audit.stem}-{index:03d}.json").read_bytes())
            response = json.loads(audit.with_name(
                f"{audit.stem}-response-{index:03d}.json").read_bytes())
            _require(row["category"] == "solver" and row["status"] == "completed"
                     and _equal(row["context"], {
                         "variant_id": self.variant_id,
                         "run_identity": self.value["frozen_dispatch"]["run_identity"],
                     }) and request["request_index"] == index
                     and request["request_seed"] == self.value["model_identity"]["seed"]
                     and request["retry_of"] is None
                     and _equal(request["frozen_context"], self.value["frozen_context"])
                     and response["status"] == "completed" and response["exception_type"] is None
                     and response["request_index"] == index
                     and response["request_seed"] == request["request_seed"]
                     and response["request_identity_sha256"] == evidence_sha256(
                         request["request_identity"])
                     and _equal(response["actual_request_identity"], request["request_identity"])
                     and _equal(row["usage"], {
                         key: value for key, value in response["usage"].items()
                         if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
                         and type(value) in (int, float)
                     }), "continuation_call_evidence_changed")
            verified.append({
                "request_index": index, "governor_request_id": row["request_id"],
                "governor_record_sha256": evidence_sha256(row),
                "seed_record_sha256": evidence_sha256(request),
                "response_record_sha256": evidence_sha256(response),
            })
        return verified


def resumed_seeded_gate_client(client, *, seed, audit_path, calls, framework_state_io=None):
    """Normal SeededGateClient contract, strict exclusive writes, starting at N."""
    from datetime import datetime, timezone
    import time
    from skill3d.evolution.fitset_stream_v11 import SeededGateClient

    class StrictResumedSeededClient(SeededGateClient):
        def __init__(self):
            super().__init__(client, seed, Path(audit_path), framework_state_io)
            self.requests = [deepcopy(dict(call.seed_record)) for call in calls]
            self._terminal = False
            self.new_completed_calls = 0

        def _write_create_once(self, path, value):
            _create_once(path, value, self.framework_state_io)

        def chat(self, *args, **kwargs):
            _require(not self._terminal, "continuation_request_pending_or_failed")
            # Any exception, including write failures, permanently closes this client.
            self._terminal = True
            index = len(self.requests) + 1
            _require(not any(path.exists() or path.is_symlink() for path in (
                self.audit_path.with_name(f"{self.audit_path.stem}-{index:03d}.json"),
                self.audit_path.with_name(f"{self.audit_path.stem}-response-{index:03d}.json"),
            )), "continuation_seed_already_exists")
            kwargs["seed"] = self.seed
            identity = describe_request(
                self.client.build_payload(*args, **kwargs),
                model_identity=self.client.audit_identity(seed=self.seed))
            request = {
                "request_seed": self.seed, "request_index": index,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "request_identity": identity, "frozen_context": deepcopy(self.audit_context),
                "request_context": deepcopy(self.request_context), "retry_of": None,
            }
            self._write_create_once(self.audit_path.with_name(
                f"{self.audit_path.stem}-{index:03d}.json"), request)
            self.requests.append(deepcopy(request))
            started = time.perf_counter()
            try:
                text = self.client.chat(*args, **kwargs)
            except BaseException as exc:
                self._write_create_once(self.audit_path.with_name(
                    f"{self.audit_path.stem}-response-{index:03d}.json"), {
                        "request_seed": self.seed, "request_index": index,
                        "request_identity_sha256": evidence_sha256(identity),
                        "status": "error", "exception_type": type(exc).__name__,
                        "usage": {}, "response_identity": {}, "actual_request_identity": {},
                    })
                raise RecoveryStop("continuation_request_failed_no_retry") from None
            _require(isinstance(text, str)
                     and _equal(getattr(self.client, "last_request_identity", None), identity),
                     "continuation_request_identity_mismatch")
            self._write_create_once(self.audit_path.with_name(
                f"{self.audit_path.stem}-response-{index:03d}.json"), {
                    "request_seed": self.seed, "request_index": index,
                    "request_identity_sha256": evidence_sha256(identity),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_s": time.perf_counter() - started,
                    "status": "completed", "exception_type": None,
                    "usage": dict(getattr(self.client, "last_usage", {}) or {}),
                    "response_identity": dict(
                        getattr(self.client, "last_response_identity", {}) or {}),
                    "actual_request_identity": dict(self.client.last_request_identity),
                })
            self.new_completed_calls += 1
            self._terminal = False
            return text

    return StrictResumedSeededClient()


class CachedPrefixResumeClient:
    """Cache-only prefix followed by a lazily constructed governed normal client."""

    def __init__(self, cached, *, spec, delegate_factory, ledger_path):
        self.cached = cached
        self.spec = spec
        self.delegate_factory = delegate_factory
        self.ledger_path = Path(ledger_path)
        self.delegate = None
        self._terminal = False
        self._prefix_receipt = None
        self._audit_context = None
        self._request_context = None
        ledger = spec.verify_original(ledger_path=self.ledger_path)
        self._ledger_prefix = deepcopy(ledger["requests"])
        _require(len([row for row in self._ledger_prefix
                      if row["category"] == "solver" and row["context"].get("variant_id")
                      == spec.variant_id]) == len(spec.calls), "unreviewed_original_call")

    def __getattr__(self, name):
        return getattr(self.delegate if self.delegate is not None else self.cached, name)

    def set_audit_context(self, context):
        self._audit_context = deepcopy(dict(context))
        self.cached.set_audit_context(context)
        if self.delegate is not None:
            self.delegate.set_audit_context(context)

    def set_request_context(self, context):
        self._request_context = deepcopy(dict(context))
        self.cached.set_request_context(context)
        if self.delegate is not None:
            self.delegate.set_request_context(context)

    def _verify_ledger(self):
        ledger = self.spec.verify_original(ledger_path=self.ledger_path)
        _require(_equal(ledger["requests"][:len(self._ledger_prefix)], self._ledger_prefix),
                 "original_ledger_prefix_changed")

    def chat(self, *args, **kwargs):
        _require(not self._terminal, "prefix_resume_closed")
        self._terminal = True
        try:
            self._verify_ledger()
            if self.cached._index < len(self.spec.calls):
                text = self.cached.chat(*args, **kwargs)
            else:
                if self.delegate is None:
                    self._prefix_receipt = self.cached.finish_replay()
                    _require(_equal(self._prefix_receipt["replayed_calls"],
                                    self.spec.custody["replayed_calls"]),
                             "recovery_custody_mismatch")
                    self.spec.require_no_future_seeds()
                    self.spec.require_next_seed_absent(len(self.spec.calls) + 1)
                    self.delegate = self.delegate_factory()
                    self.delegate.set_audit_context(self._audit_context)
                    self.delegate.set_request_context(self._request_context)
                self.spec.require_next_seed_absent(
                    len(self.spec.calls) + self.delegate.new_completed_calls + 1)
                text = self.delegate.chat(*args, **kwargs)
        except RecoveryStop:
            raise
        except BaseException:
            raise RecoveryStop("prefix_resume_failed_no_retry") from None
        self._terminal = False
        return text

    def finish_execution(self, *, files):
        _require(not self._terminal and self.cached._index == len(self.spec.calls),
                 "prefix_replay_incomplete")
        _require(self.delegate is not None and self.delegate.new_completed_calls >= 1,
                 "unexpected_early_finish")
        self._verify_ledger()
        ledger = json.loads(self.ledger_path.read_bytes())
        new_calls = self.spec.continuation_calls(
            ledger, length=len(self._ledger_prefix), count=self.delegate.new_completed_calls)
        receipt = {
            "schema_version": PREFIX_RESUME_SCHEMA, "status": "episode_continuation_complete",
            "variant_id": self.spec.variant_id,
            "spec_identity_sha256": self.spec.value["identity_sha256"],
            "deployment_identity_sha256": self.spec.continuation.identity_sha256,
            "custody_identity_sha256": self.spec.custody["identity_sha256"],
            "cached_calls": len(self.spec.calls),
            "new_completed_calls": self.delegate.new_completed_calls,
            "original_ledger_length": len(self._ledger_prefix),
            "original_ledger_prefix_sha256": evidence_sha256(self._ledger_prefix),
            "new_calls": new_calls,
            "files": {name: _sha(Path(path).read_bytes()) for name, path in files.items()},
        }
        receipt["identity_sha256"] = evidence_sha256(receipt)
        _create_once(self.spec.output_root / "continuation_receipt.json", receipt,
                     self.cached._framework_state_io)
        self._terminal = True
        return receipt

    def require_execution_continued(self):
        try:
            _require(not self._terminal and self.cached._index == len(self.spec.calls),
                     "prefix_replay_incomplete")
            _require(self.delegate is not None and self.delegate.new_completed_calls >= 1,
                     "unexpected_early_finish")
            self._verify_ledger()
        except BaseException:
            self._terminal = True
            raise
