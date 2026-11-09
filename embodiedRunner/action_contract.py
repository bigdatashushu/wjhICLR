"""Immutable public decision data. Raw simulator/scorer packets stay in the host."""
from __future__ import annotations

from dataclasses import dataclass, field
import io
import re
from collections.abc import Mapping
from types import MappingProxyType

from protocol import ContractError, canonical, json_object, sha256

VERSION = "embodied-decision/1"
ENVIRONMENTS = frozenset({"eb_navigation", "eb_alfred", "eb_habitat", "eb_manipulation"})
MAX_PACKET_BYTES = 12_000_000
MAX_EVIDENCE_REFS = 32
HASH = re.compile(r"^[0-9a-f]{64}$")


def integer(value, *, name, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ContractError("invalid_" + name)
    return value


def identity(value, *, name):
    if not isinstance(value, str) or not value or len(value) > 200:
        raise ContractError("invalid_" + name)
    return value


def digest(value, *, name):
    if not isinstance(value, str) or HASH.fullmatch(value) is None:
        raise ContractError("invalid_" + name)
    return value


def freeze(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ContractError("nonstring_json_key")
        return MappingProxyType({key: freeze(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(child) for child in value)
    if value is None or type(value) in (str, bool, int, float):
        try:
            canonical(value)
        except ValueError:
            raise ContractError("nonfinite_public_data") from None
        return value
    raise ContractError("nonjson_public_data")


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [plain(child) for child in value]
    return value


def public_hash(value):
    return sha256(canonical(plain(value)))


def canonical_action_table(names):
    if (not isinstance(names, (list, tuple)) or not 0 <= len(names) <= 1024
            or any(not isinstance(name, str) or not name or len(name) > 2000 for name in names)):
        raise ContractError("invalid_action_table")
    return tuple(names)


def table_records(names):
    return [{"action_id": index, "action_name": name} for index, name in enumerate(names)]


def table_hash(names):
    return public_hash(table_records(canonical_action_table(names)))


@dataclass(frozen=True)
class EpisodeBinding:
    run_id: str
    episode_id: str
    official_task_id: str
    environment: str
    subset: str
    input_sha256: str
    split_role: str = "development"

    def __post_init__(self):
        for name in ("run_id", "episode_id", "official_task_id", "subset"):
            identity(getattr(self, name), name=name)
        digest(self.input_sha256, name="input_sha256")
        if self.environment not in ENVIRONMENTS:
            raise ContractError("invalid_environment")
        if self.split_role not in {"development", "learning", "validation", "external_holdout", "final_test"}:
            raise ContractError("invalid_split_role")


def validate_memory(value, *, episode_id, observation_revision, decision_id=""):
    if value is None:
        value = {"subgoal": "", "records": []}
    if not isinstance(value, Mapping) or set(value) != {"subgoal", "records"}:
        raise ContractError("invalid_memory_fields")
    subgoal, records = value["subgoal"], value["records"]
    if not isinstance(subgoal, str) or len(subgoal) > 1000:
        raise ContractError("invalid_memory_subgoal")
    if not isinstance(records, (tuple, list)) or len(records) > 16:
        raise ContractError("invalid_memory_count")
    total = len(subgoal)
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
                "text", "source", "episode_id", "observation_revision", "decision_id", "confidence"}:
            raise ContractError("invalid_memory_record")
        if (not isinstance(record["text"], str) or len(record["text"]) > 1000
                or record["source"] not in {"visual", "official_feedback"}
                or record["confidence"] not in {"observed", "uncertain", "historical"}
                or record["episode_id"] != episode_id):
            raise ContractError("invalid_memory_source")
        revision = integer(record["observation_revision"], name="memory_revision")
        identity(record["decision_id"], name="memory_decision")
        if revision > observation_revision or (
                revision == observation_revision and decision_id and record["decision_id"] != decision_id):
            raise ContractError("memory_scope_mismatch")
        total += len(record["text"])
    if total > 4000:
        raise ContractError("memory_text_cap")
    return freeze(value)


@dataclass(frozen=True)
class PublicObservation:
    episode_id: str
    official_task_id: str
    environment: str
    subset: str
    instruction: str
    environment_step: int
    mutation_revision: int
    observation_revision: int
    png: bytes = field(repr=False)
    rgb_sha256: str
    pixel_sha256: str
    rgb_size: tuple[int, int]
    view_id: str
    action_table: tuple[str, ...]
    action_table_sha256: str
    public_feedback: str
    feedback_sha256: str
    public_memory: Mapping
    memory_sha256: str
    manipulation_input: Mapping = field(default_factory=lambda: MappingProxyType({}))
    planner_png: bytes = field(default=b"", repr=False)

    def public_dict(self):
        """Only this representation may enter prompts/execution."""
        value = {
            "schema": VERSION, "episode_id": self.episode_id,
            "official_task_id": self.official_task_id, "environment": self.environment,
            "subset": self.subset, "instruction": self.instruction,
            "environment_step": self.environment_step,
            "mutation_revision": self.mutation_revision,
            "observation_revision": self.observation_revision,
            "rgb_sha256": self.rgb_sha256, "pixel_sha256": self.pixel_sha256,
            "rgb_size": list(self.rgb_size), "view_id": self.view_id,
            "action_table": table_records(self.action_table),
            "action_table_sha256": self.action_table_sha256,
            "public_feedback": self.public_feedback, "feedback_sha256": self.feedback_sha256,
            "public_memory": plain(self.public_memory), "memory_sha256": self.memory_sha256,
        }
        if self.environment == "eb_manipulation":
            value["manipulation_input"] = plain(self.manipulation_input)
            value["planner_rgb_sha256"] = sha256(self.planner_png)
        return value


def public_observation(binding, *, png, instruction, environment_step, mutation_revision,
                       observation_revision, action_table, public_feedback, public_memory=None,
                       resolution=500, manipulation_input=None, planner_png=None):
    """Explicit whitelist: never construct by forwarding **raw_packet."""
    if type(png) is not bytes or not 0 < len(png) <= MAX_PACKET_BYTES:
        raise ContractError("invalid_rgb_bytes")
    if not isinstance(instruction, str) or not instruction or len(instruction) > 32000:
        raise ContractError("invalid_instruction")
    if not isinstance(public_feedback, str) or len(public_feedback) > 32000:
        raise ContractError("invalid_public_feedback")
    for name, value in (("environment_step", environment_step),
                        ("mutation_revision", mutation_revision),
                        ("observation_revision", observation_revision)):
        integer(value, name=name)
    try:
        from PIL import Image
        with Image.open(io.BytesIO(png)) as image:
            if image.format != "PNG" or image.size != (resolution, resolution):
                raise ContractError("unexpected_rgb_shape")
            pixels = image.convert("RGB").tobytes()
    except (OSError, ValueError, TypeError):
        raise ContractError("invalid_rgb") from None
    table = canonical_action_table(action_table)
    if binding.environment == "eb_manipulation":
        from manipulation_protocol import public_input
        if table:
            raise ContractError("manipulation_has_no_action_ids")
        manipulation_input = freeze(public_input(manipulation_input, binding.subset))
        if type(planner_png) is not bytes or not 0 < len(planner_png) <= MAX_PACKET_BYTES:
            raise ContractError("invalid_manipulation_planner_rgb")
        with Image.open(io.BytesIO(planner_png)) as image:
            if image.format != "PNG" or image.size != (resolution, resolution):
                raise ContractError("unexpected_planner_rgb_shape")
            image.verify()
    elif not table or manipulation_input is not None or planner_png is not None:
        raise ContractError("invalid_environment_action_input")
    else:
        manipulation_input = freeze({})
        planner_png = b""
    memory = validate_memory(public_memory, episode_id=binding.episode_id,
                             observation_revision=observation_revision)
    return PublicObservation(
        binding.episode_id, binding.official_task_id, binding.environment, binding.subset,
        instruction, environment_step, mutation_revision, observation_revision,
        png, sha256(png), sha256(pixels), (resolution, resolution),
        f"{binding.episode_id}:observation:{observation_revision}:view:0",
        table, table_hash(table), public_feedback, public_hash(public_feedback),
        memory, public_hash(memory), manipulation_input, planner_png,
    )


@dataclass(frozen=True)
class DecisionSnapshot:
    observation: PublicObservation
    decision_id: str
    method_sha256: str
    budget_sha256: str
    evidence_revision: int = 0
    evidence_manifest_sha256: str = ""
    capabilities: tuple[str, ...] = ("current_rgb",)
    input_policy: str = "current_only"
    execution_policy: str = "single_action"
    task_state: Mapping = field(default_factory=lambda: MappingProxyType({}), kw_only=True)
    snapshot_sha256: str = field(init=False)

    def __post_init__(self):
        identity(self.decision_id, name="decision_id")
        if self.method_sha256:
            digest(self.method_sha256, name="method_sha256")
        digest(self.budget_sha256, name="budget_sha256")
        integer(self.evidence_revision, name="evidence_revision")
        if self.evidence_manifest_sha256:
            digest(self.evidence_manifest_sha256, name="evidence_manifest_sha256")
        object.__setattr__(self, "task_state", freeze(self.task_state))
        if (self.input_policy != "current_only" or self.execution_policy != "single_action"
                or type(self.capabilities) is not tuple
                or any(not isinstance(c, str) for c in self.capabilities)
                or len(set(self.capabilities)) != len(self.capabilities)
                or "current_rgb" not in self.capabilities):
            raise ContractError("unsupported_snapshot_policy")
        object.__setattr__(self, "snapshot_sha256", public_hash(self.public_dict(include_hash=False)))

    def public_dict(self, *, include_hash=True):
        value = {
            "schema": VERSION, "observation": self.observation.public_dict(),
            "decision_id": self.decision_id, "method_sha256": self.method_sha256,
            "budget_sha256": self.budget_sha256, "evidence_revision": self.evidence_revision,
            "evidence_manifest_sha256": self.evidence_manifest_sha256,
            "capabilities": list(self.capabilities), "input_policy": self.input_policy,
            "execution_policy": self.execution_policy,
        }
        if include_hash:
            value["snapshot_sha256"] = self.snapshot_sha256
        if self.task_state:
            value["task_state"] = plain(self.task_state)
        return value


@dataclass(frozen=True)
class ExecutionEnvelope:
    episode_id: str
    session_id: str
    mutation_revision: int
    environment_step: int
    decision_id: str
    snapshot_sha256: str


@dataclass(frozen=True)
class ActionPayload:
    decision_id: str
    observation_revision: int
    snapshot_sha256: str
    action_table_sha256: str
    actions: tuple
    evidence_refs: tuple[str, ...] = ()
    memory_update: Mapping | None = None

    def __post_init__(self):
        if not isinstance(self.actions, (list, tuple)) or not isinstance(self.evidence_refs, (list, tuple)):
            raise ContractError("invalid_action_payload_lists")
        object.__setattr__(self, "actions", tuple(
            tuple(action) if isinstance(action, (list, tuple)) else action for action in self.actions))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        if self.memory_update is not None:
            object.__setattr__(self, "memory_update", freeze(self.memory_update))

    def to_dict(self):
        return {
            "decision_id": self.decision_id, "observation_revision": self.observation_revision,
            "snapshot_sha256": self.snapshot_sha256,
            "action_table_sha256": self.action_table_sha256,
            "actions": [list(a) if isinstance(a, tuple) else a for a in self.actions],
            "evidence_refs": list(self.evidence_refs), "memory_update": plain(self.memory_update),
        }


def decode_action_payload(value):
    if isinstance(value, (str, bytes)):
        if len(value) > MAX_PACKET_BYTES:
            raise ContractError("action_packet_cap")
        value = json_object(value)
    if not isinstance(value, Mapping) or set(value) != {
            "decision_id", "observation_revision", "snapshot_sha256", "action_table_sha256",
            "actions", "evidence_refs", "memory_update"}:
        raise ContractError("unexpected_action_payload_fields")
    return ActionPayload(**value)


def validate_action_payload(payload, snapshot, envelope, *, session_id, max_actions,
                            actual_result_ids=(), authorize_result=None):
    actual_result_ids = tuple(actual_result_ids)
    if not isinstance(payload, ActionPayload):
        raise ContractError("typed_action_payload_required")
    obs = snapshot.observation
    expected = ExecutionEnvelope(obs.episode_id, session_id, obs.mutation_revision,
                                 obs.environment_step, snapshot.decision_id, snapshot.snapshot_sha256)
    if envelope != expected:
        raise ContractError("execution_envelope_mismatch")
    if (payload.decision_id != snapshot.decision_id
            or type(payload.observation_revision) is not int
            or payload.observation_revision != obs.observation_revision
            or payload.snapshot_sha256 != snapshot.snapshot_sha256
            or payload.action_table_sha256 != obs.action_table_sha256):
        raise ContractError("stale_action_payload")
    integer(max_actions, name="max_actions", maximum=30)
    if len(payload.actions) > max_actions:
        raise ContractError("action_plan_cap")
    for action in payload.actions:
        if obs.environment == "eb_manipulation":
            from manipulation_protocol import vector
            vector(action)
        else:
            integer(action, name="action_id", maximum=len(obs.action_table) - 1)
    for result_id in payload.evidence_refs:
        identity(result_id, name="result_id")
    if len(payload.evidence_refs) > MAX_EVIDENCE_REFS or len(set(payload.evidence_refs)) != len(payload.evidence_refs):
        raise ContractError("invalid_evidence_refs")
    for result_id in actual_result_ids:
        identity(result_id, name="actual_result_id")
    dependencies = tuple(dict.fromkeys((*payload.evidence_refs, *actual_result_ids)))
    if len(dependencies) > MAX_EVIDENCE_REFS:
        raise ContractError("actual_evidence_refs_cap")
    for result_id in dependencies:
        if authorize_result is None or authorize_result(result_id, snapshot) is not True:
            raise ContractError("evidence_scope_rejected")
    if payload.memory_update is not None:
        validate_memory(payload.memory_update, episode_id=obs.episode_id,
                        observation_revision=obs.observation_revision, decision_id=snapshot.decision_id)
    return dependencies
