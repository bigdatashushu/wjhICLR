"""Decision-scoped result receipts and bounded observation reads."""
from __future__ import annotations

from action_contract import digest, integer, plain, public_hash
from protocol import ContractError, canonical
from skill3d.schemas.tool_response import ModelToolResponse, model_tool_response
from skill3d.small_image.tools import TOOL_NAMES, PUBLIC_FIELDS, build_registry

READER_VERSION = "embodied-observation-reader/1"
AUTH_FIELDS = frozenset({
    "episode_id", "tool_id", "argument_digest", "evidence_version", "dependency_refs",
    "allowed", "reason_codes", "metric_gate_result", "decision_version", "result_id",
    "scene_route", "question_tool_scope", "invalidated_by"})


def public_authorization(receipt):
    return {key: plain(receipt[key]) for key in AUTH_FIELDS if key in receipt}


def scope_of(snapshot):
    obs = snapshot.observation
    return {"episode_id": obs.episode_id, "decision_id": snapshot.decision_id,
            "observation_revision": obs.observation_revision,
            "evidence_revision": snapshot.evidence_revision,
            "snapshot_sha256": snapshot.snapshot_sha256,
            "evidence_manifest_sha256": snapshot.evidence_manifest_sha256}


class ResultStore:
    def __init__(self, snapshot, records=(), *, max_reads=3, max_results=32):
        self.snapshot, self.scope = snapshot, scope_of(snapshot)
        self.records, self.reads = {}, []
        self.max_reads, self.max_results = max_reads, max_results
        self.revoked = False
        for row in records:
            self.accept(row)

    def accept(self, row):
        if (type(row) is not dict or set(row) != {"scope", "response", "authorization", "sha256"}
                or row["scope"] != self.scope or self.revoked
                or row["sha256"] != public_hash({k: v for k, v in row.items() if k != "sha256"})):
            raise ContractError("result_scope_mismatch")
        response = ModelToolResponse.model_validate(row["response"]).model_dump()
        tool = response["source_tool"]
        auth = row["authorization"]
        if (tool not in TOOL_NAMES or response["status"] != "ok"
                or not isinstance(response["data"], dict)
                or set(response["data"]) - PUBLIC_FIELDS[tool]
                or not isinstance(auth, dict) or auth.get("allowed") is not True
                or set(auth) - AUTH_FIELDS
                or auth.get("result_id") != response["result_id"]
                or auth.get("episode_id") != self.scope["episode_id"]
                or auth.get("tool_id") != tool or auth.get("scene_route") != "small_image"
                or auth.get("decision_version") != "small-image-tools/1"
                or auth.get("invalidated_by")
                or auth.get("reason_codes")):
            raise ContractError("untrusted_result")
        digest(auth.get("argument_digest"), name="tool_argument_digest")
        result_id = response["result_id"]
        if result_id in self.records:
            if self.records[result_id] != row:
                raise ContractError("result_collision")
            return
        if len(self.records) >= self.max_results:
            raise ContractError("result_cap")
        self.records[result_id] = plain(row)

    def observe(self, result):
        row = {"scope": self.scope, "response": model_tool_response(result),
               "authorization": public_authorization(result.authorization)}
        row["sha256"] = public_hash(row)
        self.accept(row)

    def authorize(self, result_id, snapshot):
        return (not self.revoked and scope_of(snapshot) == self.scope
                and result_id in self.records)

    def read(self, result_id, fields=None, offset=0, limit=8):
        if not self.authorize(result_id, self.snapshot):
            raise ContractError("reader_scope_denied")
        if len(self.reads) >= self.max_reads:
            raise ContractError("reader_cap")
        integer(offset, name="reader_offset", maximum=128)
        integer(limit, name="reader_limit", maximum=32)
        if limit < 1:
            raise ContractError("reader_limit")
        data = self.records[result_id]["response"]["data"]
        if fields is None:
            fields = sorted(data)
        if (not isinstance(fields, (tuple, list)) or len(fields) > 32
                or any(not isinstance(f, str) or f not in data for f in fields)
                or len(set(fields)) != len(fields)):
            raise ContractError("reader_fields")
        page = {key: (data[key][offset:offset + limit] if isinstance(data[key], list) else data[key])
                for key in fields}
        row = {"schema_version": READER_VERSION, "result_id": result_id, "scope": self.scope,
               "data": page, "reason_code": "allowed", "offset": offset, "limit": limit}
        if len(canonical(row)) > 4000:
            raise ContractError("reader_page_cap")
        row["payload_sha256"] = public_hash(page)
        self.reads.append(row)
        return plain(row)

    def revoke(self):
        self.revoked = True


def action_registry(prepared, binding, *, segment_id):
    from types import SimpleNamespace
    # A segment identity avoids result-id collisions across fresh processes.
    view = SimpleNamespace(
        frames=prepared.frames, candidates=prepared.candidates,
        geometries=prepared.geometries,
        provenance={**prepared.provenance, "segment_id": segment_id})
    return build_registry(view, SimpleNamespace(
        input_sha256=binding.input_sha256, benchmark=binding.environment))
