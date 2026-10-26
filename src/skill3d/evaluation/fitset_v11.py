"""Fit-set Harness3D execution and direct-Qwen comparison for v11.

The legacy exploratory path reuses questions for induction and comparison;
scene-disjoint evolution instead freezes a separate held-out gate. Neither path
modifies the production active Skill pointer. Every reachable slot receives an
independent candidate body; the frozen bundle also preserves the active parent
for physical slots that are not revised by the selected panel.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import copy
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import queue
import random
import re
import signal
import shutil
import subprocess
import threading
import time
import traceback
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

import numpy as np

from skill3d.adapters.episode_source import EpisodeItem
from skill3d.evaluation.accuracy import extract_option_letter
from skill3d.evaluation.direct_qwen_v11 import (
    DIRECT_RESPONSE_SCHEMA,
    DIRECT_RUN_SCHEMA,
    file_identity,
)
from skill3d.evaluation.mra import mra_single, parse_numeric_answer
from skill3d.infra.experiment_startup import current_code_identity
from skill3d.routing.task_classifier import TASK_TYPES, canonical_task
from skill3d.schemas import SkillSpecV11
from skill3d.schemas.answer import CANONICAL_UNIT_BY_QUESTION_TYPE
from skill3d.schemas.data_access import (
    learning_authorization_for, normalize_experiment_split,
)
from skill3d.schemas.evidence import SKILL_EVIDENCE_STATES
from skill3d.schemas.reconstruction import METRIC_TASK_TYPES
from skill3d.schemas.skill import parse_skill_markdown
from skill3d.skills.delivery import DEFAULT_METHOD_CONTEXT_MAX_CHARS
from skill3d.synthesis.prompt_builder import PROMPT_TEMPLATE_VERSION
from skill3d.tools.docs_v11 import TOOL_DOCS_VERSION

if TYPE_CHECKING:
    from skill3d.online.runner import OnlineRunConfig

PANEL_SCHEMA = "v11-same-question-system-vs-direct-qwen-proposal/2.0"
FIT_SKILL_BUNDLE_SCHEMA = "v11-fit-set-slot-skill-bundle/2.0"
SYSTEM_RUN_SCHEMA = "v11-fit-set-system-run/1.0"
SYSTEM_RESULT_SCHEMA = "v11-fit-set-system-result/1.0"
COMPARISON_SCHEMA = "v11-fit-set-system-vs-direct-qwen/1.0"
FIT_BINDING_MIGRATION_SCHEMA = "v11-fit-set-binding-migrations/1.0"
PANEL_NAMES = ("ready_579", "recommended_core_897")
CORE_G1M1_TASKS = frozenset({
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
})
ALL_FIT_SLOTS = tuple(
    (question_type, evidence_state)
    for question_type in TASK_TYPES
    for evidence_state in SKILL_EVIDENCE_STATES
)
REACHABLE_FIT_SLOTS = (
    *((question_type, "g0m0") for question_type in TASK_TYPES),
    *((question_type, "g1m0") for question_type in TASK_TYPES),
    *((question_type, "g1m1") for question_type in TASK_TYPES
      if question_type in METRIC_TASK_TYPES),
)
UNREACHABLE_FIT_SLOTS = tuple(
    slot for slot in ALL_FIT_SLOTS if slot not in set(REACHABLE_FIT_SLOTS)
)
FIT_SLOT_GROUPING_FIELDS = ("question_type", "evidence_state")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.@-]+$")
_EMPTY_M4_PARENT_QUESTION_BINDING_SHA256 = (
    "945301bce9b0f5167bd5c8feb3878ec0873ac384288930bccb0e7b702a2cd72f"
)
_EMPTY_M4_CURRENT_QUESTION_BINDING_SHA256 = (
    "8d163a595a0bb584b3a907c4fd51d8e6fc8b3b74962a9a335c04ce8bfb5c5bf6"
)
PROCESS_SCHEDULER = "stable_slot_spawn_process_v1"
SINGLE_PROCESS_SCHEDULER = "single_main_process_v1"
PREFLIGHT_SCHEDULER = "stable_scene_threaded_preflight_v1"


class FitSetError(RuntimeError):
    """A frozen fit-set input or result cannot be trusted."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def content_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def fit_slot_key(question_type: str, evidence_state: str) -> str:
    return f"{canonical_task(question_type)}/{evidence_state}"


def _json_value(value):
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json"))
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(child)
                for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_create_once(path: Path, value: object) -> None:
    data = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise FitSetError(f"immutable artifact changed: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FitSetError(message)


def _variant_key(variant_id: str) -> str:
    return hashlib.sha256(variant_id.encode("utf-8")).hexdigest()[:20]


def _stable_slot_shards(
    variants: Sequence[Mapping],
    workers: int,
) -> list[list[dict]]:
    """Assign every ``(question_type, evidence_state)`` slot to one worker.

    One Skill lineage belongs to exactly one slot, so exclusive slot ownership is
    what lets every worker publish into its own run-local library without a
    cross-process read-modify-write race on the shared active snapshot.

    Inside a worker the episodes are ordered by scene, so all questions of one
    scene run back to back and that scene is decoded once. A scene-grouped order
    is still a deterministic total order over each slot's own episodes, so every
    slot keeps its relative stream order and its own batch boundaries.
    """
    _require(workers > 0, "process worker count must be positive")
    by_slot: dict[str, list[dict]] = {}
    for raw in variants:
        row = dict(raw)
        scene_id = str(row.get("scene_id") or "")
        _require(bool(scene_id), "slot scheduling requires scene_id")
        slot_key = f"{row['question_type']}/{row['evidence_state']}"
        row["slot_key"] = slot_key
        by_slot.setdefault(slot_key, []).append(row)
    shards: list[list[dict]] = [[] for _ in range(workers)]
    for slot_key in sorted(by_slot, key=lambda key: (-len(by_slot[key]), key)):
        index = min(range(workers), key=lambda value: (len(shards[value]), value))
        shards[index].extend(by_slot[slot_key])
    for shard in shards:
        shard.sort(key=lambda row: (
            str(row["scene_id"]),
            str(row["slot_key"]),
            str(row["variant_id"]),
        ))
    return shards


def _process_shard_worker(
    runner,
    shard_index: int,
    owned_variants: Sequence[Mapping],
    variants: Sequence[Mapping],
    stop_event,
    messages,
    episode_semaphore=None,
) -> None:
    """Run one slot shard on the child process main thread."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    runner._episode_semaphore = episode_semaphore
    if threading.current_thread() is not threading.main_thread():
        messages.put({
            "kind": "error",
            "shard_index": shard_index,
            "error": "fit-set process shard did not start on its main thread",
        })
        return
    try:
        def accept(result):
            messages.put({
                "kind": "result",
                "shard_index": shard_index,
                "result": result,
            })
        runner._run_shard(shard_index, owned_variants, variants, stop_event, accept)
    except BaseException as exc:  # noqa: BLE001 - parent freezes worker failure
        messages.put({
            "kind": "error",
            "shard_index": shard_index,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        stop_event.set()
    finally:
        messages.put({"kind": "done", "shard_index": shard_index})


def _gate_episode_worker(gate_runner, variant, messages) -> None:
    """Run one gate episode on a spawned child's main thread.

    The global permit is held by the parent gate process on this child's
    behalf, so the child never touches the semaphore. The episode result is
    written to the arm's own result directory; the parent reads it back during
    the unchanged scoring phase.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    variant_id = str(variant["variant_id"])
    try:
        if threading.current_thread() is not threading.main_thread():
            raise FitSetError("gate episode did not start on its main thread")
        gate_runner._episode_semaphore = None
        gate_runner._run_one(variant)
    except BaseException as exc:  # noqa: BLE001 - parent records child failure
        messages.put({
            "kind": "error",
            "variant_id": variant_id,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
    else:
        messages.put({"kind": "done", "variant_id": variant_id})


def _source_code_identity() -> dict:
    root = Path(__file__).resolve().parents[3]
    receipt = current_code_identity(root)
    if receipt:
        return {"source": "startup_receipt", **receipt}
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise FitSetError(
            f"cannot establish fit-set source identity: {exc}") from exc
    dirty = bool(status.strip())
    if os.environ.get("SKILL3D_RUN_MODE") == "formal" and dirty:
        raise FitSetError("formal fit-set execution requires a clean source worktree")
    return {
        "source": "git_worktree",
        "source_root": str(root),
        "base_commit": commit,
        "git_dirty": dirty,
        "status_sha256": hashlib.sha256(
            status.encode("utf-8")).hexdigest(),
    }


def _lexical_absolute(path: str | Path) -> str:
    """Normalize a declared path without touching a possibly remote filesystem."""
    return os.path.abspath(os.path.expanduser(str(path)))


def _normalized_variant(raw: Mapping) -> dict:
    variant_id = str(raw.get("variant_id") or "")
    qa_id = str(raw.get("qa_id") or "")
    state = str(raw.get("evidence_state") or "")
    task = canonical_task(str(raw.get("question_type") or ""))
    _require(bool(_SAFE_ID.fullmatch(variant_id)),
             f"invalid system variant_id: {variant_id!r}")
    _require(bool(_SAFE_ID.fullmatch(qa_id)), f"invalid qa_id: {qa_id!r}")
    _require(variant_id == f"{qa_id}@{state}",
             f"{variant_id}: variant identity does not match qa/state")
    _require(state in SKILL_EVIDENCE_STATES,
             f"{variant_id}: invalid evidence state {state!r}")
    artifact = str(raw.get("artifact_ref") or "")
    _require(bool(artifact), f"{variant_id}: artifact_ref is missing")
    return {
        **dict(raw),
        "variant_id": variant_id,
        "qa_id": qa_id,
        "scene_id": str(raw.get("scene_id") or ""),
        "question_type": task,
        "evidence_state": state,
        "split": str(raw.get("split") or ""),
        "artifact_ref": artifact,
        "binding_key": str(raw.get("binding_key") or ""),
        "binding_manifest_sha256": str(
            raw.get("binding_manifest_sha256") or ""),
    }


def load_fitset_panel(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(value.get("schema_version") == PANEL_SCHEMA,
             "unsupported fit-set panel schema")
    claimed = str(value.get("proposal_sha256") or "")
    proposal_body = {
        key: child for key, child in value.items()
        if key != "proposal_sha256"
    }
    _require(claimed == content_sha256(proposal_body),
             "fit-set panel proposal hash changed")
    accumulation = value.get("skill_accumulation")
    _require(isinstance(accumulation, dict),
             "fit-set panel lacks a Skill accumulation contract")
    _require(
        accumulation.get("grouping_key") == list(FIT_SLOT_GROUPING_FIELDS),
        "fit-set feedback must group by question_type and evidence_state",
    )
    _require(
        accumulation.get("active_snapshot_physical_slot_count")
        == len(ALL_FIT_SLOTS),
        "fit-set active snapshot must declare 24 physical slots",
    )
    _require(
        accumulation.get("reachable_slot_count") == len(REACHABLE_FIT_SLOTS),
        "fit-set panel must declare 19 reachable slots",
    )
    _require(
        accumulation.get("unreachable_slots")
        == [fit_slot_key(*slot) for slot in UNREACHABLE_FIT_SLOTS],
        "fit-set panel unreachable slot declaration changed",
    )
    panels = value.get("panels")
    _require(isinstance(panels, dict),
             "fit-set panel lacks panel definitions")
    for panel_name in PANEL_NAMES:
        panel_row = panels.get(panel_name)
        _require(isinstance(panel_row, dict),
                 f"fit-set panel definition is missing: {panel_name}")
        declared_slots = list(panel_row.get("skill_slots") or [])
        _require(
            panel_row.get("skill_slot_count") == len(declared_slots)
            and len(declared_slots) == len(set(declared_slots)),
            f"{panel_name}: invalid Skill slot count",
        )
        _require(
            set(declared_slots)
            <= {fit_slot_key(*slot) for slot in REACHABLE_FIT_SLOTS},
            f"{panel_name}: panel declares an unreachable Skill slot",
        )
    raw_variants = value.get("system_variants")
    direct_rows = value.get("direct_qwen_rows")
    _require(isinstance(raw_variants, list) and raw_variants,
             "system_variants must be a non-empty list")
    _require(isinstance(direct_rows, list) and direct_rows,
             "direct_qwen_rows must be a non-empty list")
    variants = [_normalized_variant(row) for row in raw_variants]
    by_variant: dict[str, dict] = {}
    for row in variants:
        variant_id = row["variant_id"]
        _require(
            (row["question_type"], row["evidence_state"])
            in set(REACHABLE_FIT_SLOTS),
            f"{variant_id}: non-metric g1m1 is unreachable and cannot "
            "participate in the fit-set experiment",
        )
        _require(variant_id not in by_variant,
                 f"duplicate system variant: {variant_id}")
        by_variant[variant_id] = row
    direct_by_qa: dict[str, dict] = {}
    mapped: set[str] = set()
    for raw in direct_rows:
        qa_id = str(raw.get("qa_id") or "")
        _require(qa_id and qa_id not in direct_by_qa,
                 f"duplicate direct qa_id: {qa_id!r}")
        variant_ids = [str(item) for item in (
            raw.get("system_variant_ids") or [])]
        _require(bool(variant_ids), f"{qa_id}: direct row has no variants")
        _require(not (set(variant_ids) & mapped),
                 f"{qa_id}: a system variant maps to multiple direct rows")
        for variant_id in variant_ids:
            _require(variant_id in by_variant,
                     f"{qa_id}: unknown mapped variant {variant_id}")
            _require(by_variant[variant_id]["qa_id"] == qa_id,
                     f"{variant_id}: direct qa_id mismatch")
            _require(
                by_variant[variant_id]["scene_id"]
                == str(raw.get("scene_id") or "")
                and by_variant[variant_id]["question_type"]
                == canonical_task(str(raw.get("question_type") or ""))
                and by_variant[variant_id]["split"]
                == str(raw.get("split") or ""),
                f"{variant_id}: direct row identity mismatch",
            )
        mapped.update(variant_ids)
        direct_by_qa[qa_id] = {**dict(raw), "qa_id": qa_id,
                               "system_variant_ids": variant_ids}
    _require(mapped == set(by_variant),
             "direct rows do not exactly cover system variants")
    return {
        **value,
        "system_variants": variants,
        "direct_qwen_rows": list(direct_by_qa.values()),
        "panel_file_sha256": file_identity(path)["sha256"],
    }


def select_panel_variants(panel: Mapping, panel_name: str) -> list[dict]:
    _require(panel_name in PANEL_NAMES,
             f"unknown panel {panel_name!r}; expected one of {PANEL_NAMES}")
    variants = list(panel["system_variants"])
    if panel_name == "ready_579":
        selected = [
            row for row in variants
            if row.get("system_input_status") == "ready"
        ]
    elif panel_name == "recommended_core_897":
        selected = [
            row for row in variants
            if row.get("system_input_status") == "ready"
            or (
                row["evidence_state"] == "g1m1"
                and row["question_type"] in CORE_G1M1_TASKS
            )
        ]
    else:
        raise AssertionError(f"unhandled fit-set panel: {panel_name}")
    expected = int((panel.get("panels") or {}).get(
        panel_name, {}).get("system_input_variants") or 0)
    _require(expected > 0 and len(selected) == expected,
             f"{panel_name}: selected {len(selected)} variants, expected {expected}")
    actual_slots = {
        fit_slot_key(row["question_type"], row["evidence_state"])
        for row in selected
    }
    panel_row = (panel.get("panels") or {}).get(panel_name) or {}
    declared_slots = set(panel_row.get("skill_slots") or [])
    _require(
        actual_slots == declared_slots
        and panel_row.get("skill_slot_count") == len(actual_slots),
        f"{panel_name}: selected Skill slots differ from the panel contract",
    )
    return selected


def direct_rows_for_variants(
    panel: Mapping,
    variants: Sequence[Mapping],
) -> list[dict]:
    wanted = {str(row["variant_id"]) for row in variants}
    selected = []
    covered: set[str] = set()
    for row in panel["direct_qwen_rows"]:
        mapped = wanted & set(row["system_variant_ids"])
        if mapped:
            selected.append({
                **row,
                "system_variant_ids": [
                    value for value in row["system_variant_ids"]
                    if value in wanted
                ],
            })
            covered.update(mapped)
    _require(covered == wanted, "selected direct rows do not cover the panel")
    return selected


def load_variant_bindings(
    path: str | Path,
    variants: Sequence[Mapping],
) -> dict[str, dict]:
    """Load a binding manifest keyed by variant_id, falling back to qa_id.

    qa_id fallback is accepted only when its frozen artifact and binding
    identities exactly match the selected variant. This supports the original
    579-row manifest while allowing g1m0/g1m1 variants of one QA to coexist.
    """
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") == FIT_BINDING_MIGRATION_SCHEMA:
        claimed = str(manifest.get("identity_sha256") or "")
        body = {
            key: value for key, value in manifest.items()
            if key != "identity_sha256"
        }
        _require(
            claimed == content_sha256(body),
            "fit-set binding migration manifest hash changed",
        )
    rows = manifest.get("bindings")
    _require(isinstance(rows, dict), "prepared binding manifest lacks bindings")
    migrations = manifest.get("migrations") or {}
    _require(isinstance(migrations, dict),
             "prepared binding migrations must be a mapping")
    resolved: dict[str, dict] = {}
    for variant in variants:
        variant_id = str(variant["variant_id"])
        qa_id = str(variant["qa_id"])
        reference = rows.get(variant_id)
        if reference is None:
            reference = rows.get(qa_id)
        _require(isinstance(reference, dict),
                 f"{variant_id}: prepared binding is missing")
        parent_matches = (
            reference.get("key") == variant["binding_key"]
            and reference.get("manifest_sha256")
            == variant["binding_manifest_sha256"]
        )
        if not parent_matches:
            migration = migrations.get(variant_id)
            _require(
                isinstance(migration, dict)
                and migration.get("parent_key") == variant["binding_key"]
                and migration.get("parent_manifest_sha256")
                == variant["binding_manifest_sha256"]
                and migration.get("new_key") == reference.get("key")
                and migration.get("new_manifest_sha256")
                == reference.get("manifest_sha256")
                and migration.get("proof", {}).get("passed") is True,
                f"{variant_id}: binding identity changed without a valid migration",
            )
        _require(
            _lexical_absolute(reference.get("scene_artifact_ref") or "")
            == _lexical_absolute(variant["artifact_ref"]),
            f"{variant_id}: binding artifact path changed",
        )
        resolved[variant_id] = copy.deepcopy(reference)
    return resolved


def migrate_empty_m4_bindings(
    *,
    panel: Mapping,
    variants: Sequence[Mapping],
    source_manifest_path: str | Path,
    output_dir: str | Path,
    min_free_gib: float = 80.0,
) -> dict:
    """Republish only empty M4-unavailable bindings across one reviewed source diff."""
    from skill3d.infra.m5_identity import (
        content_sha256 as m5_content_sha256,
        file_identity as m5_file_identity,
        source_identity,
    )
    from skill3d.segmentation import (
        prepared_binding_store as prepared_store_module,
        question_binding,
    )
    from skill3d.segmentation.prepared_binding_store import (
        PreparedBindingStore,
        load_prepared_binding_reference,
    )

    reserve = float(min_free_gib)
    _require(math.isfinite(reserve) and reserve >= 0,
             "min_free_gib must be finite and non-negative")
    root = Path(output_dir).resolve()
    _require(not root.exists(), f"binding migration output exists: {root}")
    source_path = Path(source_manifest_path).resolve()
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    source_rows = source_manifest.get("bindings")
    _require(isinstance(source_rows, dict),
             "source prepared binding manifest lacks bindings")
    root.mkdir(parents=True)
    target_store = PreparedBindingStore(root / "store")
    code_root = Path(question_binding.__file__).resolve().parents[1]
    current_sources = source_identity(code_root)
    current_question_binding = current_sources.get(
        "segmentation/question_binding.py") or {}
    _require(
        current_question_binding.get("sha256")
        == _EMPTY_M4_CURRENT_QUESTION_BINDING_SHA256,
        "current question_binding.py is outside the reviewed migration diff",
    )
    current_store_source = m5_file_identity(
        Path(prepared_store_module.__file__))
    bindings: dict[str, dict] = {}
    migrations: dict[str, dict] = {}
    migrated_scenes: set[str] = set()
    for variant in variants:
        variant_id = str(variant["variant_id"])
        qa_id = str(variant["qa_id"])
        parent = source_rows.get(variant_id) or source_rows.get(qa_id)
        _require(isinstance(parent, dict),
                 f"{variant_id}: source binding is missing")
        binding = load_prepared_binding_reference(
            parent,
            scene_input_identity={
                "sha256": parent["scene_input_identity_sha256"],
            },
        )
        generation = copy.deepcopy(binding.stats.get("generation") or {})
        identity = copy.deepcopy(generation.get("identity") or {})
        frozen_sources = identity.get("sources") or {}
        diff_keys = sorted(
            key for key in set(frozen_sources) | set(current_sources)
            if frozen_sources.get(key) != current_sources.get(key)
        )
        if not diff_keys:
            bindings[variant_id] = copy.deepcopy(parent)
            continue
        parent_question_binding = frozen_sources.get(
            "segmentation/question_binding.py") or {}
        proof = {
            "passed": False,
            "source_diff_keys": diff_keys,
            "parent_question_binding_sha256": (
                parent_question_binding.get("sha256")),
            "current_question_binding_sha256": (
                current_question_binding.get("sha256")),
            "binding_status": binding.stats.get("status"),
            "object_count": len(binding.objects),
            "materialized": bool(binding.materialized),
            "object_parse_present": bool(
                binding.stats.get("object_parse")),
            "binding_error_present": bool(
                binding.stats.get("binding_error")),
            "parse_error_present": bool(
                binding.stats.get("parse_error")),
        }
        _require(
            diff_keys == ["segmentation/question_binding.py"]
            and proof["parent_question_binding_sha256"]
            == _EMPTY_M4_PARENT_QUESTION_BINDING_SHA256
            and proof["current_question_binding_sha256"]
            == _EMPTY_M4_CURRENT_QUESTION_BINDING_SHA256
            and proof["binding_status"] == "m4_unavailable"
            and proof["object_count"] == 0
            and proof["materialized"] is False
            and proof["object_parse_present"] is False
            and proof["binding_error_present"] is False
            and proof["parse_error_present"] is False
            and not binding.prepared_measurements,
            f"{variant_id}: binding is not eligible for empty-M4 migration",
        )
        _require(
            generation.get("sha256") == m5_content_sha256(identity),
            f"{variant_id}: parent generation identity does not replay",
        )
        generation_sha256_before = generation["sha256"]
        identity["sources"] = current_sources
        generation["identity"] = identity
        generation["sha256"] = m5_content_sha256(identity)
        binding.stats["generation"] = generation
        request = copy.deepcopy(parent["request_identity"])
        request.pop("sha256", None)
        request["implementation_sources"] = current_sources
        request["store_source"] = current_store_source
        request["sha256"] = m5_content_sha256(request)
        free_gib = shutil.disk_usage(root).free / (1024 ** 3)
        _require(
            free_gib >= reserve,
            f"{variant_id}: free disk {free_gib:.2f} GiB is below "
            f"the {reserve:.2f} GiB reserve",
        )
        migrated = target_store.publish(
            request_identity=request,
            scene_artifact=parent["scene_artifact_ref"],
            scene_input_identity={
                "sha256": parent["scene_input_identity_sha256"],
            },
            binding=binding,
        )
        proof["passed"] = True
        bindings[variant_id] = migrated
        migrations[variant_id] = {
            "parent_key": parent["key"],
            "parent_manifest_sha256": parent["manifest_sha256"],
            "new_key": migrated["key"],
            "new_manifest_sha256": migrated["manifest_sha256"],
            "reason": (
                "empty m4_unavailable binding; reviewed source diff is "
                "unreachable before object parsing"
            ),
            "generation_sha256_before": generation_sha256_before,
            "generation_sha256_after": generation["sha256"],
            "proof": proof,
        }
        migrated_scenes.add(str(variant["scene_id"]))
    body = {
        "schema_version": FIT_BINDING_MIGRATION_SCHEMA,
        "role": "fit-set current-source binding compatibility overlay",
        "panel_proposal_sha256": panel["proposal_sha256"],
        "panel_file_sha256": panel["panel_file_sha256"],
        "source_manifest": file_identity(source_path),
        "source_code_identity": _source_code_identity(),
        "migration_contract": {
            "parent_question_binding_sha256": (
                _EMPTY_M4_PARENT_QUESTION_BINDING_SHA256),
            "current_question_binding_sha256": (
                _EMPTY_M4_CURRENT_QUESTION_BINDING_SHA256),
            "allowed_source_diff_keys": [
                "segmentation/question_binding.py"],
            "required_binding_status": "m4_unavailable",
            "required_object_count": 0,
            "required_materialized": False,
        },
        "store_root": str(target_store.root),
        "n_variants": len(variants),
        "n_migrated": len(migrations),
        "n_reused": len(variants) - len(migrations),
        "n_migrated_scenes": len(migrated_scenes),
        "min_free_gib": reserve,
        "bindings": {
            key: bindings[key] for key in sorted(bindings)
        },
        "migrations": {
            key: migrations[key] for key in sorted(migrations)
        },
    }
    manifest = {**body, "identity_sha256": content_sha256(body)}
    _write_create_once(root / "prepared_bindings.json", manifest)
    load_variant_bindings(
        root / "prepared_bindings.json",
        variants,
    )
    return manifest


@dataclass(frozen=True)
class FitSlotSkill:
    question_type: str
    evidence_state: str
    skill_id: str
    version: str
    source_ref: str
    content_sha256: str
    skill_md: str
    role: str
    parent_skill_version: str

    @property
    def slot_key(self) -> str:
        return fit_slot_key(self.question_type, self.evidence_state)

    def as_spec(self) -> SkillSpecV11:
        return SkillSpecV11(
            skill_id=self.skill_id,
            version=self.version,
            question_type=self.question_type,
            evidence_state=self.evidence_state,
            skill_md=self.skill_md,
        )


@dataclass(frozen=True)
class FitSkillBundle:
    bundle_id: str
    manifest_sha256: str
    slots: Mapping[tuple[str, str], FitSlotSkill]
    manifest: Mapping

    def skill_for(
        self,
        question_type: str,
        evidence_state: str,
    ) -> FitSlotSkill:
        slot = (canonical_task(question_type), evidence_state)
        _require(slot in self.slots, f"fit Skill bundle lacks slot {slot}")
        return self.slots[slot]


def load_fit_skill_bundle(path: str | Path) -> FitSkillBundle:
    manifest_path = Path(path).resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(value.get("schema_version") == FIT_SKILL_BUNDLE_SCHEMA,
             "unsupported fit Skill bundle schema")
    claimed = str(value.get("manifest_sha256") or "")
    body = {key: child for key, child in value.items()
            if key != "manifest_sha256"}
    _require(claimed == content_sha256(body),
             "fit Skill bundle manifest hash changed")
    rows = value.get("slots")
    expected_keys = {fit_slot_key(*slot) for slot in ALL_FIT_SLOTS}
    _require(isinstance(rows, dict) and set(rows) == expected_keys,
             "fit Skill bundle must contain all 24 physical slots")
    slots: dict[tuple[str, str], FitSlotSkill] = {}
    role_counts: dict[str, int] = {}
    for question_type, evidence_state in ALL_FIT_SLOTS:
        key = fit_slot_key(question_type, evidence_state)
        row = rows[key]
        _require(isinstance(row, dict),
                 f"{key}: slot Skill entry must be a mapping")
        _require(
            row.get("question_type") == question_type
            and row.get("evidence_state") == evidence_state,
            f"{key}: slot identity changed",
        )
        role = str(row.get("role") or "")
        _require(
            role in {
                "revised_candidate",
                "active_parent_pending",
                "active_parent_unreachable",
            },
            f"{key}: invalid frozen slot role",
        )
        if (question_type, evidence_state) in set(UNREACHABLE_FIT_SLOTS):
            _require(role == "active_parent_unreachable",
                     f"{key}: unreachable slot must preserve its active parent")
        else:
            _require(role != "active_parent_unreachable",
                     f"{key}: reachable slot has an unreachable role")
        source_ref = str(row.get("source_ref") or "")
        source_path = (manifest_path.parent / source_ref).resolve()
        _require(source_path.is_relative_to(manifest_path.parent),
                 f"{key}: source_ref escapes the bundle")
        text = source_path.read_text(encoding="utf-8")
        _require(hashlib.sha256(text.encode("utf-8")).hexdigest()
                 == row.get("content_sha256"),
                 f"{key}: canonical SKILL.md changed")
        _require(len(text) <= DEFAULT_METHOD_CONTEXT_MAX_CHARS,
                 f"{key}: canonical SKILL.md exceeds service limit")
        front, _ = parse_skill_markdown(text)
        _require(front["name"] == row.get("name"),
                 f"{key}: stable method name changed")
        skill = FitSlotSkill(
            question_type=question_type,
            evidence_state=evidence_state,
            skill_id=str(row.get("skill_id") or ""),
            version=str(row.get("version") or ""),
            source_ref=source_ref,
            content_sha256=str(row["content_sha256"]),
            skill_md=text,
            role=role,
            parent_skill_version=str(row.get("parent_skill_version") or ""),
        )
        spec = skill.as_spec()
        _require(
            spec.name == row.get("name")
            and spec.content_sha256 == skill.content_sha256,
            f"{key}: frozen Skill identity changed",
        )
        parent_id, separator, parent_version = (
            skill.parent_skill_version.partition("@"))
        _require(
            bool(separator)
            and parent_id == skill.skill_id
            and len(parent_version.split(".")) == 3,
            f"{key}: invalid parent Skill identity",
        )
        if role == "revised_candidate":
            _require(
                tuple(map(int, skill.version.split(".")))
                > tuple(map(int, parent_version.split("."))),
                f"{key}: candidate version did not advance",
            )
        else:
            _require(
                skill.version == parent_version,
                f"{key}: preserved parent version changed",
            )
        role_counts[role] = role_counts.get(role, 0) + 1
        slots[(question_type, evidence_state)] = skill
    revised = role_counts.get("revised_candidate", 0)
    _require(
        value.get("physical_slot_count") == len(ALL_FIT_SLOTS)
        and value.get("reachable_slot_count") == len(REACHABLE_FIT_SLOTS)
        and value.get("revised_slot_count") == revised
        and value.get("preserved_parent_slot_count")
        == len(ALL_FIT_SLOTS) - revised
        and value.get("preserved_unreachable_slot_count")
        == len(UNREACHABLE_FIT_SLOTS)
        and value.get("unreachable_slots")
        == [fit_slot_key(*slot) for slot in UNREACHABLE_FIT_SLOTS],
        "fit Skill bundle slot counts changed",
    )
    return FitSkillBundle(
        bundle_id=str(value.get("bundle_id") or ""),
        manifest_sha256=claimed,
        slots=slots,
        manifest=value,
    )


def build_fit_skill_bundle(
    *,
    output_dir: str | Path,
    bundle_id: str,
    candidates: Mapping[tuple[str, str], SkillSpecV11],
    active_parents: Mapping[tuple[str, str], SkillSpecV11],
    source_run_identity: Mapping,
) -> dict:
    """Freeze exact-slot candidates and unchanged active parents."""
    root = Path(output_dir).resolve()
    _require(not root.exists(), f"fit Skill bundle already exists: {root}")
    _require(set(active_parents) == set(ALL_FIT_SLOTS),
             "fit Skill bundle requires all 24 active parent slots")
    _require(
        set(candidates) <= set(REACHABLE_FIT_SLOTS),
        "candidate Skills must cover only reachable slots",
    )
    _require(
        len({spec.content_sha256 for spec in candidates.values()})
        == len(candidates),
        "each revised slot must have an independent SKILL.md body",
    )
    root.mkdir(parents=True)
    slot_rows = {}
    for question_type, evidence_state in ALL_FIT_SLOTS:
        slot = (question_type, evidence_state)
        key = fit_slot_key(*slot)
        parent = active_parents[slot]
        _require(
            parent.question_type == question_type
            and parent.evidence_state == evidence_state,
            f"{key}: active parent occupies another slot",
        )
        spec = candidates.get(slot, parent)
        _require(
            spec.question_type == question_type
            and spec.evidence_state == evidence_state,
            f"{key}: frozen Skill occupies another slot",
        )
        if slot in candidates:
            _require(
                spec.skill_id == parent.skill_id
                and tuple(map(int, spec.version.split(".")))
                > tuple(map(int, parent.version.split("."))),
                f"{key}: candidate left its active lineage",
            )
        _require(len(spec.skill_md) <= DEFAULT_METHOD_CONTEXT_MAX_CHARS,
                 f"{key}: frozen Skill exceeds service limit")
        relative = Path("skills") / question_type / evidence_state / "SKILL.md"
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(spec.skill_md, encoding="utf-8")
        if slot in candidates:
            role = "revised_candidate"
        elif slot in set(UNREACHABLE_FIT_SLOTS):
            role = "active_parent_unreachable"
        else:
            role = "active_parent_pending"
        slot_rows[key] = {
            "question_type": question_type,
            "evidence_state": evidence_state,
            "skill_id": spec.skill_id,
            "version": spec.version,
            "name": spec.name,
            "source_ref": relative.as_posix(),
            "content_sha256": spec.content_sha256,
            "role": role,
            "parent_skill_version": f"{parent.skill_id}@{parent.version}",
        }
    body = {
        "schema_version": FIT_SKILL_BUNDLE_SCHEMA,
        "bundle_id": bundle_id,
        "physical_slot_count": len(ALL_FIT_SLOTS),
        "reachable_slot_count": len(REACHABLE_FIT_SLOTS),
        "revised_slot_count": len(candidates),
        "preserved_parent_slot_count": len(ALL_FIT_SLOTS) - len(candidates),
        "preserved_reachable_pending_slot_count": (
            len(REACHABLE_FIT_SLOTS) - len(candidates)
        ),
        "preserved_unreachable_slot_count": len(UNREACHABLE_FIT_SLOTS),
        "unreachable_slots": [
            fit_slot_key(*slot) for slot in UNREACHABLE_FIT_SLOTS
        ],
        "runtime_contract": (
            "load exactly one immutable Skill by question_type/evidence_state; "
            "no cross-state body reuse or runtime identity adaptation"
        ),
        "source_run_identity": dict(source_run_identity),
        "slots": slot_rows,
    }
    manifest = {**body, "manifest_sha256": content_sha256(body)}
    _write_create_once(root / "manifest.json", manifest)
    load_fit_skill_bundle(root / "manifest.json")
    return manifest


def _system_result_row(
    outcome,
    *,
    variant: Mapping,
    options: Sequence[str],
    elapsed_s: float,
    skill_identity: Mapping,
) -> dict:
    service_errors = [
        error
        for row in outcome.rounds
        for error in (row.get("service_errors") or [])
    ]
    incomplete = outcome.final_state == "unavailable" or bool(service_errors)
    legal = outcome.answer is not None and outcome.predicted is not None
    if outcome.is_mca:
        legal = legal and outcome.predicted in list(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(options)]
        )
    return {
        "schema_version": SYSTEM_RESULT_SCHEMA,
        "variant_id": variant["variant_id"],
        "qa_id": variant["qa_id"],
        "scene_id": variant["scene_id"],
        "split": variant["split"],
        "question_type": variant["question_type"],
        "evidence_state": variant["evidence_state"],
        "status": "incomplete" if incomplete else "completed",
        "final_state": outcome.final_state,
        "answer": outcome.answer,
        "predicted": outcome.predicted,
        "score": None if incomplete else outcome.score,
        "scoring_status": (
            "incomplete" if incomplete else outcome.scoring_status
        ),
        "legal_answer": bool(legal),
        "answer_parse_status": outcome.answer_parse_status,
        "submission_status": outcome.submission_status,
        "answer_basis": outcome.answer_basis,
        "answer_source": outcome.answer_source,
        "episode_status": outcome.episode_status,
        "failure_code": outcome.failure_code,
        "service_errors": service_errors,
        "static_check_errors": list(outcome.static_check_errors),
        "agent_rounds": int(outcome.agent_rounds),
        "yield_count": int(outcome.yield_count),
        "model_request_count": int(outcome.agent_rounds) + int(
            bool(outcome.eval_visual_fallback.get("request_count"))
        ),
        "tool_call_count": sum(
            len(row.get("results") or []) for row in outcome.rounds
        ),
        "elapsed_s": float(elapsed_s),
        "main_gate_passed": outcome.main_gate_passed,
        "quality_status": outcome.quality_status,
        "scene_route": outcome.scene_route,
        "question_tool_scope": outcome.question_tool_scope,
        "initial_evidence_states": dict(outcome.initial_evidence_states),
        "initial_skill_selection": dict(outcome.initial_skill_selection),
        "delivered_skill_versions": list(
            outcome.delivered_skill_versions
        ),
        "skill_identity": dict(skill_identity),
    }


def audit_fitset_pilot(
    *,
    output_root: str | Path,
    expected_variants: int,
) -> dict:
    """Fail closed before a pilot may expand to a full Harness3D run."""
    root = Path(output_root).resolve()
    _require(expected_variants > 0, "pilot size must be positive")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    result_paths = sorted((root / "results").glob("*.json"))
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in result_paths
    ]
    outcomes = []
    missing_outcomes = []
    for row in results:
        variant_id = str(row.get("variant_id") or "")
        path = root / "variants" / _variant_key(variant_id) / "outcome.json"
        if not path.is_file():
            missing_outcomes.append(variant_id)
        else:
            outcomes.append(json.loads(path.read_text(encoding="utf-8")))
    all_rounds = [
        round_row
        for outcome in outcomes
        for round_row in (outcome.get("rounds") or [])
    ]
    execution_errors = [
        {
            "variant_id": str(outcome.get("qa_id") or ""),
            "error_code": str(round_row.get("error_code") or ""),
            "error_type": str(
                (round_row.get("execution_diagnostics") or {}).get(
                    "error_type") or ""),
            "error_message": str(
                (round_row.get("execution_diagnostics") or {}).get(
                    "error_message") or ""),
        }
        for outcome in outcomes
        for round_row in (outcome.get("rounds") or [])
        if round_row.get("error_code") in {
            "violation_runtime",
            "violation_syntax",
        }
    ]
    service_errors = [
        str(error)
        for row in results
        for error in (row.get("service_errors") or [])
    ]
    executed_rounds = sum(bool(row.get("executed")) for row in all_rounds)
    no_execution_variants = sum(
        not any(row.get("executed") for row in (outcome.get("rounds") or []))
        for outcome in outcomes
    )
    tool_program_answers = sum(
        row.get("answer_source") == "tool_program" for row in results)
    visual_fallback_answers = sum(
        row.get("answer_source") == "direct_vlm_routed" for row in results)
    evidence_mismatches = sum(
        row.get("evidence_state")
        != (row.get("initial_skill_selection") or {}).get("evidence_state")
        for row in results
    )
    infrastructure_error_counts: dict[str, int] = {}
    for error in execution_errors:
        key = (
            f"{error['error_code']}:{error['error_type']}:"
            f"{error['error_message']}"
        )
        infrastructure_error_counts[key] = (
            infrastructure_error_counts.get(key, 0) + 1)
    problems = []
    manifest_body = {key: value for key, value in manifest.items()
                     if key != "identity_sha256"}
    if (manifest.get("identity_sha256") != content_sha256(manifest_body)
            or summary.get("manifest_identity_sha256") != manifest.get("identity_sha256")):
        problems.append("pilot manifest identity/hash differs from its summary")
    ids = [row.get("variant_id") for row in results]
    if len(set(ids)) != len(ids) or not set(ids) <= set(manifest.get("variant_ids") or []):
        problems.append("pilot result IDs are duplicated or outside the panel")
    if len(results) != expected_variants:
        problems.append("pilot result count differs from the declared size")
    if (
        summary.get("completed_variants") != len(results)
        or summary.get("pending_variants")
        != len(manifest.get("variant_ids") or []) - len(results)
    ):
        problems.append("pilot summary and result files disagree")
    if missing_outcomes:
        problems.append("pilot result lacks a matching outcome")
    if execution_errors:
        problems.append("pilot contains sandbox infrastructure errors")
    if no_execution_variants or executed_rounds < expected_variants:
        problems.append("pilot did not execute at least one program per variant")
    if tool_program_answers <= 0:
        problems.append("pilot has no accepted Harness3D program answer")
    if visual_fallback_answers:
        problems.append("pilot success depends on direct visual fallback")
    if service_errors:
        problems.append("pilot contains online-model service errors")
    if evidence_mismatches:
        problems.append("pilot runtime evidence state changed")
    body = {
        "schema_version": "v11-fit-set-pilot-audit/2.0",
        "status": "passed" if not problems else "failed",
        "expand_to_full": not problems,
        "expected_variants": expected_variants,
        "result_files": len(results),
        "outcomes": len(outcomes),
        "missing_outcomes": len(missing_outcomes),
        "executed_rounds": executed_rounds,
        "no_execution_variants": no_execution_variants,
        "tool_program_answers": tool_program_answers,
        "visual_fallback_answers": visual_fallback_answers,
        "service_error_rows": sum(
            bool(row.get("service_errors")) for row in results),
        "http_429_mentions": sum("429" in error for error in service_errors),
        "http_5xx_mentions": sum(
            any(str(code) in error for code in range(500, 600))
            for error in service_errors
        ),
        "evidence_mismatches": evidence_mismatches,
        "infrastructure_error_counts": infrastructure_error_counts,
        "problems": problems,
    }
    return {**body, "identity_sha256": content_sha256(body)}


class SystemVariantRunner:
    """Run one frozen system input per isolated, resumable variant directory."""

    def __init__(
        self,
        *,
        output_root: str | Path,
        panel: Mapping,
        panel_name: str,
        variants: Sequence[Mapping],
        item_provider: Callable[[Mapping], EpisodeItem],
        input_receipt_provider: Callable[
            [Mapping, EpisodeItem], Mapping],
        input_contract_identity: Mapping,
        binding_refs: Mapping[str, Mapping],
        base_cfg: OnlineRunConfig,
        phase: str,
        skills_for_variant: Callable[[Mapping], tuple[Sequence[SkillSpecV11], dict]],
        client_factory: Callable[[str], object] | None = None,
        min_free_gib: float = 80.0,
        workers: int = 1,
        evolution_config=None,
        label_metadata: Mapping[str, Mapping] | None = None,
        revision_client_factory=None,
        gate_client_factory=None,
        episode_concurrency: int | None = None,
        scheduler: str = "slot_shards",
        skill_owners: int = 4,
        request_governor_config=None,
        published_inputs=None,
        gate_variants: Sequence[Mapping] | None = None,
        experiment_split: dict | None = None,
    ) -> None:
        _require(phase in {"accumulation", "final_system"},
                 "system phase must be accumulation or final_system")
        self.root = Path(output_root).resolve()
        self.panel = panel
        self.panel_name = panel_name
        self.variants = [dict(row) for row in variants]
        _require(bool(self.variants), "learning/execution variants must not be empty")
        self.gate_variant_rows: dict[str, dict] = {}
        self.variant_rows: dict[str, dict] = {}
        for rows, is_gate in ((self.variants, False), (gate_variants or (), True)):
            for raw in rows:
                row = dict(raw)
                variant_id = str(row.get("variant_id") or "")
                _require(bool(variant_id) and variant_id not in self.variant_rows,
                         f"duplicate or empty frozen variant_id: {variant_id!r}")
                key = fit_slot_key(row["question_type"], row["evidence_state"])
                _require(row.get("slot_key", key) == key,
                         f"{variant_id}: frozen slot identity changed")
                row["slot_key"] = key
                self.variant_rows[variant_id] = row
                if is_gate:
                    self.gate_variant_rows[variant_id] = row
        self.variants = [self.variant_rows[row["variant_id"]] for row in self.variants]
        self.item_provider = item_provider
        self.published_inputs = published_inputs
        self.input_receipt_provider = input_receipt_provider
        self.input_contract_identity = copy.deepcopy(
            dict(input_contract_identity))
        self.binding_refs = {
            str(key): copy.deepcopy(dict(value))
            for key, value in binding_refs.items()
        }
        self.base_cfg = copy.deepcopy(base_cfg)
        self.phase = phase
        self.skills_for_variant = skills_for_variant
        self.client_factory = client_factory
        self.min_free_gib = float(min_free_gib)
        self.workers = int(workers)
        self.evolution_config = evolution_config
        try:
            self.experiment_split = (
                normalize_experiment_split(experiment_split)
                if experiment_split is not None else None)
        except ValueError as exc:
            raise FitSetError(str(exc)) from exc
        _require(self.experiment_split is None or phase == "final_system" or (
            evolution_config is not None and evolution_config.gate_mode == "qa_disjoint"),
            "experiment_split requires opt-in qa_disjoint evolution")
        if self.experiment_split is not None and phase == "final_system":
            _require(
                self.experiment_split.get("execution_role") == "validation"
                and evolution_config is None and not self.gate_variant_rows,
                "final_system overlay requires execution_role=validation without gate/evolution")
            _require(all(
                self.experiment_split["qa_roles"].get(row["qa_id"]) == "validation"
                and row.get("split") in {"induction", "inner_validation"}
                and row.get("source_split", row.get("split")) == row.get("split")
                and all(row.get(field, "validation") == "validation"
                        for field in ("experiment_role", "experiment_split"))
                for row in self.variants),
                "final_system requires mapped validation QA and unchanged source split")
        # Public-input capture survives removal of coordinator-only evolution data.
        self.capture_offline_case_input = (
            evolution_config is not None
            and evolution_config.learning_input_mode != "aggregate_v1")
        self.label_metadata = copy.deepcopy(dict(label_metadata or {}))
        self.revision_client_factory = revision_client_factory
        self.gate_client_factory = gate_client_factory
        self.scheduler = scheduler
        self.skill_owners = int(skill_owners)
        self.request_governor_config = request_governor_config
        self.episode_concurrency = (
            None if episode_concurrency is None else int(episode_concurrency))
        # The cross-process permit is a runtime-only resource. It is created by
        # the parent ``run`` and handed to workers as an explicit spawn argument;
        # it is never pickled into result files or snapshot identities.
        self._episode_semaphore = None
        self.seed_skills = []
        self.gate_panels = {}
        self.gate_deferred_reasons = {}
        self.run_identity = ""
        self.evolution_finished = False
        _require(
            math.isfinite(self.min_free_gib) and self.min_free_gib >= 0,
            "min_free_gib must be finite and non-negative",
        )
        _require(scheduler in {"slot_shards", "episode_pool"},
                 "unknown episode scheduler")
        _require(1 <= self.workers <= (32 if scheduler == "episode_pool" else 16),
                 "workers must be 1..16 for slot_shards or 1..32 for episode_pool")
        _require(1 <= self.skill_owners <= 19, "skill_owners must be 1..19")
        _require(
            self.episode_concurrency is None
            or 1 <= self.episode_concurrency <= 128,
            "episode_concurrency must be between 1 and 128 when set",
        )
        self.execution_variants = [
            row
            for _, row in sorted(
                enumerate(self.variants),
                key=lambda pair: (
                    str(pair[1]["scene_id"]),
                    pair[0],
                ),
            )
        ]
        _require(not self.gate_variant_rows or self.evolution_config is not None,
                 "separate gate variants require accumulation evolution")
        if self.evolution_config is not None:
            _require(phase == "accumulation", "只有 accumulation 可启动演化")
            _require(self.revision_client_factory is not None
                     and self.gate_client_factory is not None,
                     "演化必须配置真实 reviser 和 paired client factory")
            skills, _ = self.skills_for_variant(self.variants[0])
            self.seed_skills = [skill.model_copy(deep=True) for skill in skills]
            _require(
                {(s.question_type, s.evidence_state) for s in self.seed_skills}
                == set(ALL_FIT_SLOTS),
                "演化初始 provider 必须提供完整 24 槽位 seed",
            )
            learning_mode = self.evolution_config.learning_input_mode
            gate_mode = self.evolution_config.gate_mode
            legacy = learning_mode == "aggregate_v1" and gate_mode == "same_question"
            _require(gate_mode in {"same_question", "scene_disjoint", "qa_disjoint"},
                     "unknown gate mode")
            _require(gate_mode != "qa_disjoint" or self.experiment_split is not None,
                     "qa_disjoint requires explicit experiment_split")
            if gate_mode == "qa_disjoint":
                learning_qa = {row["qa_id"] for row in self.variants}
                gate_qa = {row["qa_id"] for row in self.gate_variant_rows.values()}
                _require(not learning_qa & gate_qa,
                         "gate QA overlaps learning QA across frozen variants")
                for row in self.variant_rows.values():
                    variant_id = row["variant_id"]
                    expected_role = (
                        "validation" if variant_id in self.gate_variant_rows else "learning")
                    _require(self.experiment_split["qa_roles"].get(row["qa_id"]) == expected_role,
                             f"{variant_id}: QA role is not authorized {expected_role}")
                    _require(all(row.get(field, expected_role) == expected_role
                                 for field in ("experiment_role", "experiment_split")),
                             f"{variant_id}: declared QA role differs from frozen role map")
                    _require(row.get("split") in {"induction", "inner_validation"}
                             and row.get("source_split", row["split"]) == row["split"],
                             f"{variant_id}: source_split must preserve the source catalog")
            _require(learning_mode in {
                "aggregate_v1", "full_cases_v1", "diagnosis_v1", "diagnosis_multimodal_v1",
            }, "unknown learning input mode")
            _require(gate_mode != "same_question" or not self.gate_variant_rows,
                     "same_question cannot accept separate gate variants")
            by_slot: dict[str, list[dict]] = {}
            for row in [*self.execution_variants, *self.gate_variant_rows.values()]:
                variant_id = row["variant_id"]
                _require(bool(row.get("scene_id")), f"{variant_id}: scene_id is missing")
                key = fit_slot_key(row["question_type"], row["evidence_state"])
                _require((row["question_type"], row["evidence_state"])
                         in set(REACHABLE_FIT_SLOTS), "evolution contains an unreachable slot")
                label = self.label_metadata.get(row["qa_id"])
                _require(label is not None, f"{variant_id}: evolution lacks label metadata")
                _require(label.get("scene_id") == row["scene_id"]
                         and label.get("question_type") == row["question_type"],
                         f"{variant_id}: label metadata identity changed")
                if variant_id not in self.gate_variant_rows:
                    _require(legacy or gate_mode == "qa_disjoint" or row.get("split") == "induction",
                             f"{variant_id}: rich/disjoint learning requires induction-only variants")
                    by_slot.setdefault(key, []).append(row)
            gate_by_slot: dict[str, list[dict]] = {}
            if gate_mode in {"scene_disjoint", "qa_disjoint"}:
                _require(gate_mode == "qa_disjoint" or bool(self.gate_variant_rows),
                         "scene_disjoint requires supplied held-out gate variants")
                learning_scenes = {row["scene_id"] for row in self.variants}
                for row in self.gate_variant_rows.values():
                    variant_id = row["variant_id"]
                    _require(row.get("split") in {"induction", "inner_validation"},
                             f"{variant_id}: gate requires held-out induction or inner_validation")
                    _require(gate_mode == "qa_disjoint" or row["scene_id"] not in learning_scenes,
                             f"{variant_id}: gate scene overlaps learning scenes")
                    _require(variant_id in self.binding_refs,
                             f"{variant_id}: gate prepared binding reference is missing")
                    gate_by_slot.setdefault(row["slot_key"], []).append(row)
            for key, rows in by_slot.items():
                if gate_mode == "qa_disjoint":
                    gate_rows = gate_by_slot.get(key, ())
                    self.gate_panels[key] = [row["variant_id"] for row in gate_rows]
                    n_qa = len({row["qa_id"] for row in gate_rows})
                    if n_qa < self.evolution_config.gate_size:
                        self.gate_deferred_reasons[key] = (
                            f"qa_disjoint admission deferred: {n_qa} validation QA; "
                            f"gate_size={self.evolution_config.gate_size}; parent retained")
                    continue
                selected = []
                scenes = set()
                gate_rows = gate_by_slot.get(key, ()) if gate_mode == "scene_disjoint" else rows
                for row in gate_rows:
                    if row["scene_id"] not in scenes:
                        selected.append(row["variant_id"])
                        scenes.add(row["scene_id"])
                    if len(selected) == self.evolution_config.gate_size:
                        break
                _require(legacy or len(selected) == self.evolution_config.gate_size,
                         f"{key}: incomplete frozen gate_size; distinct scenes required")
                self.gate_panels[key] = selected

    def learning_authorization(self, variant: Mapping) -> dict:
        if self.experiment_split is None:
            return {}
        variant_id = variant.get("variant_id")
        _require(variant_id not in self.gate_variant_rows
                 and self.variant_rows.get(variant_id) == dict(variant),
                 "validation or unfrozen variant cannot enter learning ingest")
        try:
            return learning_authorization_for(self.experiment_split, [variant["qa_id"]])
        except ValueError as exc:
            raise FitSetError(str(exc)) from exc

    def __getstate__(self) -> dict:
        # Spawned workers receive the live permit as an explicit process
        # argument, never through the pickled runner, so drop it here to keep
        # the runner picklable and to avoid duplicating the global permit.
        state = dict(self.__dict__)
        state["_episode_semaphore"] = None
        return state

    def __setstate__(self, state: Mapping) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("_episode_semaphore", None)

    @property
    def execution_scheduler(self) -> str:
        if self.scheduler == "episode_pool":
            from skill3d.evaluation.episode_pool_v11 import EPISODE_POOL_SCHEDULER
            return EPISODE_POOL_SCHEDULER
        return (
            SINGLE_PROCESS_SCHEDULER
            if self.workers == 1
            else PROCESS_SCHEDULER
        )

    def _planned_slot_shards(
        self,
        variants: Sequence[Mapping] | None = None,
    ) -> list[list[dict]]:
        return _stable_slot_shards(
            self.execution_variants if variants is None else variants,
            self.skill_owners if self.scheduler == "episode_pool" else self.workers,
        )

    def _freeze_public_input_capture(self) -> None:
        # Allow setup to finish before freezing identity, not after dispatch.
        if not self.run_identity:
            self.capture_offline_case_input = (
                self.evolution_config is not None
                and self.evolution_config.learning_input_mode != "aggregate_v1")

    def _manifest(self) -> dict:
        self._freeze_public_input_capture()
        identities = []
        for row in self.variants:
            skills, identity = self.skills_for_variant(row)
            identities.append({
                "variant_id": row["variant_id"],
                "skills": [
                    {
                        "skill_version": f"{skill.skill_id}@{skill.version}",
                        "question_type": skill.question_type,
                        "evidence_state": skill.evidence_state,
                        "content_sha256": skill.content_sha256,
                    }
                    for skill in skills
                ],
                "identity": identity,
            })
        body = {
            "schema_version": SYSTEM_RUN_SCHEMA,
            "phase": self.phase,
            "panel_name": self.panel_name,
            "panel_proposal_sha256": self.panel["proposal_sha256"],
            "panel_file_sha256": self.panel["panel_file_sha256"],
            "variant_ids": [row["variant_id"] for row in self.variants],
            "variant_rows_sha256": content_sha256(self.variants),
            "gate_variant_ids": list(self.gate_variant_rows),
            "gate_variant_rows": list(self.gate_variant_rows.values()),
            "gate_variant_rows_sha256": content_sha256(list(self.gate_variant_rows.values())),
            "frozen_union_rows_sha256": content_sha256(list(self.variant_rows.values())),
            "binding_refs_sha256": content_sha256(self.binding_refs),
            "skill_identities_sha256": content_sha256(identities),
            "input_contract_identity": self.input_contract_identity,
            "min_free_gib": self.min_free_gib,
            "execution": {
                "workers": self.workers,
                "scheduler": self.execution_scheduler,
                "process_start_method": (
                    "none" if self.workers == 1 and self.scheduler == "slot_shards"
                    else "spawn"
                ),
                "sandbox_timeout_runtime": "main_thread_sigalrm",
                "variant_ids": [
                    row["variant_id"]
                    for row in self.execution_variants
                ],
                "variant_ids_sha256": content_sha256([
                    row["variant_id"]
                    for row in self.execution_variants
                ]),
                "slot_shards": [
                    {
                        "slot_keys": sorted(
                            {row["slot_key"] for row in shard}),
                        "variant_ids": [row["variant_id"] for row in shard],
                    }
                    for shard in self._planned_slot_shards()
                ],
                "slot_ownership": {
                    row["slot_key"]: index
                    for index, shard in enumerate(self._planned_slot_shards())
                    for row in shard
                },
            },
            "seed": int(self.base_cfg.seed),
            "online_model": self.base_cfg.online_model,
            "online_model_revision": self.base_cfg.online_model_revision,
            "prompt_version": PROMPT_TEMPLATE_VERSION,
            "tool_docs_version": TOOL_DOCS_VERSION,
            "code_identity": _source_code_identity(),
            "evolution": (
                {
                    "config": _json_value(asdict(self.evolution_config)),
                    "seed_skills": [
                        skill.model_dump(mode="json") for skill in self.seed_skills],
                    "gate_panels": self.gate_panels,
                    "label_metadata_sha256": content_sha256(self.label_metadata),
                    "source_split_preserved": True,
                    "final_version_policy": "last_successfully_published_per_slot",
                }
                if self.evolution_config is not None else None
            ),
            "reference_answer_retained_for_input_identity_and_m12": self.published_inputs is None,
            "online_executor_input_is_label_free": self.published_inputs is not None,
            "reference_answer_visible_to_online_model": False,
            "solver_config_sha256": content_sha256(_json_value(asdict(self.base_cfg))),
            "request_input_config": {
                "solver_config": _json_value(asdict(self.base_cfg)),
                "evolution_config": (
                    _json_value(asdict(self.evolution_config))
                    if self.evolution_config is not None else None),
                "capture_offline_case_input": self.capture_offline_case_input,
                "offline_case_input_schema": "v11-fit-set-offline-input/1.0",
            },
        }
        if self.scheduler == "episode_pool":
            body["execution"].update({
                "skill_owners": self.skill_owners,
                "episode_concurrency": self.episode_concurrency or self.workers,
                "lineage_ingest_order": "scene_id_then_variant_id",
                "learning_window_policy": "remaining_cases_before_revision_boundary",
            })
        if self.request_governor_config is not None:
            body["request_policy"] = asdict(self.request_governor_config)
        if self.experiment_split is not None:
            body["experiment_split"] = copy.deepcopy(self.experiment_split)
            body["experiment_split_sha256"] = self.experiment_split["identity_sha256"]
            if body["evolution"] is not None:
                body["evolution"]["gate_deferred_reasons"] = dict(self.gate_deferred_reasons)
                body["evolution"]["gate_selection_policy"] = "all_validation_variants_per_slot"
        return {**body, "identity_sha256": content_sha256(body)}

    def initialize(self, *, resume: bool) -> dict:
        manifest = self._manifest()
        path = self.root / "manifest.json"
        if self.root.exists():
            _require(resume, "system output exists; use resume for the same run")
            _require(path.is_file(), "existing system output lacks manifest")
            _require(
                json.loads(path.read_text(encoding="utf-8")) == manifest,
                "system run manifest changed during resume",
            )
        else:
            self.root.mkdir(parents=True)
            _write_create_once(path, manifest)
        self.run_identity = manifest["identity_sha256"]
        return manifest

    def _result_path(self, variant_id: str) -> Path:
        return self.root / "results" / f"{_variant_key(variant_id)}.json"

    def _validate_gate_inputs(self) -> None:
        if not self.gate_variant_rows:
            return
        # Validate every held-out input before any learning request, even when
        # --max-variants restricts execution to a smaller learning subset.
        for variant in self.gate_variant_rows.values():
            values = self._prepare_one(variant, preflight=True)
            _require(values["state"] == variant["evidence_state"],
                     f"{variant['variant_id']}: gate runtime evidence state changed")
            for field, reference_field in (
                ("binding_key", "key"), ("binding_manifest_sha256", "manifest_sha256"),
            ):
                _require(not variant.get(field)
                         or variant[field] == values["reference"][reference_field],
                         f"{variant['variant_id']}: frozen gate binding identity changed")
            _write_create_once(
                self.root / "gate_preflight" / f"{_variant_key(variant['variant_id'])}.json",
                {
                    "variant": variant,
                    "input_identity_sha256": values["identity"]["sha256"],
                    "runtime_evidence_state": values["state"],
                    "binding_key": values["reference"]["key"],
                    "binding_manifest_sha256": values["reference"]["manifest_sha256"],
                    "model_requests": 0,
                },
            )

    def _freeze_offline_case_input(self, variant: Mapping, item: EpisodeItem) -> None:
        from PIL import Image

        variant_root = self.root / "variants" / _variant_key(str(variant["variant_id"]))
        fs = item.episode.frame_set
        _require(fs is not None and len(fs.frame_ids) == len(item.pixels) == 32,
                 "offline case requires the frozen 32-frame public input")
        image_refs = []
        for index, pixels in enumerate(item.pixels):
            _require(pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[2] == 3,
                     "offline case requires original RGB uint8 frames")
            image = Image.fromarray(pixels)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            data = buffer.getvalue()
            path = variant_root / "offline_images" / f"{index:02d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("xb") as stream:
                    stream.write(data)
            except FileExistsError:
                _require(path.read_bytes() == data, f"immutable offline image changed: {path}")
            digest = hashlib.sha256(data).hexdigest()
            image_refs.append({
                "path": str(path), "file_sha256": digest, "content_sha256": digest,
                "image_id": f"{variant['variant_id']}:frame:{fs.frame_ids[index]}",
                "kind": "frame", "frame_id": fs.frame_ids[index],
                "source_frame_index": fs.source_frame_indices[index],
                "source_hw": [image.height, image.width],
                "frame_set_hash": fs.frame_set_hash,
            })
        _write_create_once(variant_root / "offline_input.json", {
            "schema_version": "v11-fit-set-offline-input/1.0",
            "variant_id": variant["variant_id"],
            "variant_sha256": content_sha256(variant),
            "qa_id": variant["qa_id"], "scene_id": variant["scene_id"],
            "split": variant["split"],
            "question_text": item.episode.question,
            "options": list(item.episode.options or []),
            "frame_set_hash": fs.frame_set_hash,
            "image_refs": image_refs,
        })

    def offline_case_input(self, variant: Mapping) -> dict:
        """Read the public input frozen during execution; never recover labels."""
        path = (self.root / "variants" / _variant_key(str(variant["variant_id"]))
                / "offline_input.json")
        _require(path.is_file(), f"{variant['variant_id']}: offline_input.json is missing")
        value = json.loads(path.read_text(encoding="utf-8"))
        _require(set(value) == {
            "schema_version", "variant_id", "variant_sha256", "qa_id", "scene_id",
            "split", "question_text", "options", "frame_set_hash", "image_refs",
        } and value.get("schema_version") == "v11-fit-set-offline-input/1.0"
                 and value.get("variant_id") == variant["variant_id"]
                 and value.get("variant_sha256") == content_sha256(variant)
                 and value.get("qa_id") == variant["qa_id"]
                 and value.get("scene_id") == variant["scene_id"]
                 and value.get("split") == variant["split"],
                 f"{variant['variant_id']}: offline input identity changed")
        result_path = self._result_path(str(variant["variant_id"]))
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            _require(result.get("offline_input_sha256") == file_identity(path)["sha256"],
                     f"{variant['variant_id']}: offline input differs from completed result")
        _require(isinstance(value["image_refs"], list) and len(value["image_refs"]) == 32,
                 f"{variant['variant_id']}: incomplete frozen offline image refs")
        for ref in value["image_refs"]:
            _require(set(ref) == {
                "path", "file_sha256", "content_sha256", "image_id", "kind",
                "frame_id", "source_frame_index", "source_hw", "frame_set_hash",
            } and ref["frame_set_hash"] == value["frame_set_hash"],
                     f"{variant['variant_id']}: offline image provenance changed")
            image_path = Path(ref["path"])
            _require(image_path.parent == path.parent / "offline_images"
                     and image_path.is_file()
                     and file_identity(image_path)["sha256"] == ref["file_sha256"]
                     == ref["content_sha256"],
                     f"{variant['variant_id']}: frozen offline image changed")
        return value

    def _require_storage_reserve(self, variant_id: str) -> None:
        free_gib = shutil.disk_usage(self.root).free / (1024 ** 3)
        _require(
            free_gib >= self.min_free_gib,
            f"{variant_id}: free disk {free_gib:.2f} GiB is below "
            f"the {self.min_free_gib:.2f} GiB reserve",
        )

    def completed(self) -> dict[str, dict]:
        rows = {}
        for path in sorted((self.root / "results").glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            variant_id = str(row.get("variant_id") or "")
            _require(variant_id and variant_id not in rows,
                     f"duplicate or empty system result: {path}")
            _require(path.stem == _variant_key(variant_id),
                     f"system result path identity changed: {path}")
            rows[variant_id] = row
        expected = {row["variant_id"] for row in self.variants}
        _require(set(rows) <= expected,
                 "system results contain variants outside the frozen panel")
        return rows

    def _prepare_one(self, variant: Mapping, *, preflight: bool = False) -> dict:
        from skill3d.evaluation.skill_ablation_v11 import (
            input_identity,
            initial_state_for_item,
        )
        from skill3d.legacy.readers import load_artifact_v5
        from skill3d.online import runner
        from skill3d.segmentation.prepared_binding_store import (
            load_prepared_binding_reference,
        )
        from skill3d.segmentation.question_binding import (
            binding_parameters,
            question_input,
            validate_prepared_binding,
        )

        variant_id = str(variant["variant_id"])
        self._require_storage_reserve(variant_id)
        item = self.item_provider(variant)
        _require(isinstance(item, EpisodeItem),
                 f"{variant_id}: episode provider returned an invalid item")
        item = copy.deepcopy(item)
        input_receipt = dict(self.input_receipt_provider(variant, item))
        _require(item.input_error is None,
                 f"{variant_id}: input-error variants require explicit rows")
        _require(item.episode.scene_name == variant["scene_id"],
                 f"{variant_id}: episode scene changed")
        _require(canonical_task(item.episode.question_type)
                 == variant["question_type"],
                 f"{variant_id}: episode question type changed")
        _require(item.episode.split == variant["split"],
                 f"{variant_id}: episode split changed")
        frame_set = item.episode.frame_set
        _require(
            frame_set is not None
            and len(frame_set.frame_ids) == 32
            and len(frame_set.readable_frame_ids) == 32
            and len(item.pixels) == 32,
            f"{variant_id}: system input requires exactly 32 readable frames",
        )
        artifact_relocation = None
        transport_receipt = None
        derived_reference = None
        derived_migration_receipt = None
        binding_derivation_context = None
        if self.published_inputs is not None:
            from skill3d.schemas.evidence import skill_evidence_state
            published = self.published_inputs.prepare_published(variant, item)
            artifact_path = published["artifact_path"]
            artifact = published["artifact"]
            identity = published["identity"]
            reference = published["reference"]
            prepared = published["prepared"]
            artifact_relocation = published["artifact_relocation"]
            transport_receipt = published["transport_receipt"]
            derived_reference = published.get("derived_reference")
            derived_migration_receipt = published.get("derived_migration_receipt")
            binding_derivation_context = published.get("binding_derivation_context")
            _require(reference == self.binding_refs.get(variant_id),
                     f"{variant_id}: published binding reference changed")
            quality_artifact = runner._prepare_m4_quality(
                artifact, item.pixels, persist_quality=False)
            state = skill_evidence_state(runner.build_scene_state(
                quality_artifact, question_type=variant["question_type"]).evidence_profile)
        else:
            artifact_path = Path(variant["artifact_ref"]).resolve()
            _require(artifact_path.is_file(),
                     f"{variant_id}: frozen artifact is missing")
            state = initial_state_for_item(item, artifact_path)
            identity = input_identity(item, artifact_path)
            reference = self.binding_refs.get(variant_id)
            _require(reference is not None,
                     f"{variant_id}: prepared binding reference is missing")
            _require(reference.get("scene_input_identity_sha256")
                     == identity["sha256"],
                     f"{variant_id}: prepared binding input identity changed")
            prepared = load_prepared_binding_reference(
                reference, scene_input_identity=identity)
            artifact = load_artifact_v5(artifact_path)
        _require(state == variant["evidence_state"],
                 f"{variant_id}: runtime evidence state changed to {state}")
        frozen_parameters = dict(
            ((derived_reference or reference).get("request_identity") or {}).get("parameters") or {})
        frozen_review = copy.deepcopy(
            frozen_parameters.get("experimental_semantic_review"))
        runtime_base_cfg = replace(
            copy.deepcopy(self.base_cfg),
            semantic_review_config=frozen_review,
        )
        if self.published_inputs is not None:
            binding_seed = reference["request_identity"]["seed"]
            _require(type(binding_seed) is int,
                     f"{variant_id}: invalid sealed binding seed")
            runtime_base_cfg = replace(
                runtime_base_cfg, prepared_binding_seed=binding_seed)
        runtime_parameters = binding_parameters(
            semantic_min_pixels=runtime_base_cfg.semantic_min_pixels,
            semantic_max_pixels=runtime_base_cfg.semantic_max_pixels,
            semantic_relation_batch_size=(
                runtime_base_cfg.semantic_relation_batch_size),
            semantic_review_config=runtime_base_cfg.semantic_review_config,
        )
        _require(runtime_parameters == frozen_parameters,
                 f"{variant_id}: sealed M5 parameters no longer replay")
        validate_prepared_binding(
            prepared,
            frames=item.pixels,
            depth=runner._load_npy(artifact.depth_maps),
            c2w=runner._load_npy(artifact.c2w_list),
            intrinsics=runner._load_npy(artifact.intrinsics),
            point_conf=runner._load_npy(artifact.depth_conf),
            depth_conf=runner._load_npy(artifact.depth_conf),
            point_map=runner._load_npy(artifact.point_map),
            inputs=question_input(
                item.episode.question,
                canonical_task(item.episode.question_type),
                item.episode.options,
            ),
            scene_identity=str(
                artifact.reconstruction_input_identity
                or artifact.frame_set_hash
                or artifact.artifact_id
            ),
            seed=(runtime_base_cfg.prepared_binding_seed
                  if self.published_inputs is not None else int(runtime_base_cfg.seed)),
            model_weights_sha256="",
            parameters=runtime_parameters,
            source_compatibility=(
                runtime_base_cfg.prepared_binding_source_compatibility),
            binding_derivation_context=binding_derivation_context,
        )
        compatibility_receipt = None
        if runtime_base_cfg.prepared_binding_source_compatibility is not None:
            from skill3d.infra.m5_identity import source_identity
            from skill3d.segmentation.frozen_source_compatibility import (
                reviewed_source_compatibility,
            )
            compatibility_receipt = reviewed_source_compatibility(
                prepared.stats["generation"]["identity"]["sources"],
                source_identity(Path(__file__).resolve().parents[1]),
                runtime_base_cfg.prepared_binding_source_compatibility,
            )
        skills, skill_identity = self.skills_for_variant(variant)
        _require(bool(skills), f"{variant_id}: no runtime Skill was supplied")
        matching = [
            skill for skill in skills
            if skill.question_type == variant["question_type"]
            and skill.evidence_state == state
        ]
        _require(len(matching) == 1,
                 f"{variant_id}: runtime Skill match is not unique")
        variant_root = self.root / "variants" / _variant_key(variant_id)
        plan = {
            "schema_version": "v11-fit-set-system-plan/1.0",
            "variant": dict(variant),
            "variant_sha256": content_sha256(variant),
            "input_identity_sha256": identity["sha256"],
            "direct_input_receipt": input_receipt,
            "binding_key": reference["key"],
            "binding_manifest_sha256": reference["manifest_sha256"],
            "binding_source_compatibility": compatibility_receipt,
            "m5_parameters_sha256": content_sha256(frozen_parameters),
            "skill_identity": dict(skill_identity),
        }
        if transport_receipt is not None:
            plan["published_transport"] = transport_receipt
        if derived_reference is not None:
            plan["derived_binding_reference"] = copy.deepcopy(derived_reference)
            plan["derived_binding_migration"] = copy.deepcopy(derived_migration_receipt)
            plan["binding_derivation_sha256"] = prepared.stats["binding_derivation"]["sha256"]
        if not preflight:
            _write_create_once(variant_root / "plan.json", plan)
        return {
            "variant_id": variant_id,
            "item": item,
            "artifact_path": artifact_path,
            "artifact_relocation": artifact_relocation,
            "state": state,
            "identity": identity,
            "reference": reference,
            "prepared": prepared,
            "binding_derivation_context": binding_derivation_context,
            "runtime_base_cfg": runtime_base_cfg,
            "skills": skills,
            "skill_identity": skill_identity,
            "variant_root": variant_root,
            "plan": plan,
        }

    @contextmanager
    def _episode_permit(self):
        """Hold one global episode permit for the body, if one is configured.

        A streaming episode blocks until a permit is free. Because it owns no
        other permit while waiting, concurrent episodes cannot deadlock; the
        permit simply caps how many episodes execute at once across every
        worker and every gate child process.
        """
        semaphore = self._episode_semaphore
        if semaphore is None:
            yield
            return
        semaphore.acquire()
        try:
            yield
        finally:
            semaphore.release()

    def _run_one(self, variant: Mapping) -> dict:
        with self._episode_permit():
            return self._run_one_locked(variant)

    def _govern_client(self, client, category: str, context: Mapping):
        if self.request_governor_config is None:
            return client
        _require(client is not None, "governed runs require an explicit model client")
        from skill3d.synthesis.request_governor import GovernedClient, RequestGovernor
        return GovernedClient(
            client, RequestGovernor(self.request_governor_root, self.request_governor_config),
            category=category, context=context,
        )

    @property
    def request_governor_root(self) -> Path:
        return getattr(self, "_request_governor_root", self.root / "request_governor")

    def _run_one_locked(self, variant: Mapping) -> dict:
        from skill3d.online import runner
        from skill3d.sandbox.kernel import _Timeout

        _require(threading.current_thread() is threading.main_thread(),
                 "fit-set sandbox 必须在进程主线程执行，模型请求尚未发出")
        with _Timeout(1):
            pass

        self._freeze_public_input_capture()
        prepared_run = self._prepare_one(variant)
        variant_id = prepared_run["variant_id"]
        item = prepared_run["item"]
        if getattr(self, "_qa_disjoint_gate", False):
            _require(not self.capture_offline_case_input and not self.label_metadata
                     and not item.episode.ground_truth,
                     "qa_disjoint gate executor requires label-free public input")
        artifact_path = prepared_run["artifact_path"]
        state = prepared_run["state"]
        prepared = prepared_run["prepared"]
        runtime_base_cfg = prepared_run["runtime_base_cfg"]
        skills = prepared_run["skills"]
        skill_identity = prepared_run["skill_identity"]
        variant_root = prepared_run["variant_root"]
        trace_dir = variant_root / "trace"
        _require(not trace_dir.exists(),
                 f"{variant_id}: incomplete trace exists without a result")
        if self.capture_offline_case_input:
            self._freeze_offline_case_input(variant, item)
        cfg = replace(
            copy.deepcopy(runtime_base_cfg),
            skills=[skill.model_copy(deep=True) for skill in skills],
            active_snapshot_ref=str(skill_identity["snapshot_ref"]),
            active_snapshot_manifest_sha256=str(
                skill_identity["manifest_sha256"]),
            trace_dir=str(trace_dir),
            work_dir=str(variant_root / "work"),
            recon_dir=str(variant_root / "recon"),
            reuse_artifact=str(artifact_path),
            evaluation_binding=None,
        )
        llm = (
            self.client_factory(variant_id)
            if self.client_factory is not None
            else None
        )
        llm = self._govern_client(
            llm, getattr(self, "_request_category", "solver"),
            {"variant_id": variant_id, "run_identity": self.run_identity},
        )
        set_context = getattr(llm, "set_audit_context", None)
        if callable(set_context):
            shared_plan = {key: value for key, value in prepared_run["plan"].items()
                           if key != "skill_identity"}
            set_context({
                "run_identity": self.run_identity,
                "shared_input_sha256": content_sha256(shared_plan),
                "skill_identity": skill_identity,
                "skill_content_sha256": [skill.content_sha256 for skill in skills],
                "solver_config_sha256": content_sha256(_json_value(asdict(runtime_base_cfg))),
                "concurrency_limit": self.episode_concurrency,
                "scheduler": self.execution_scheduler,
            })
        started = time.perf_counter()
        outcomes, evaluation_run = runner.run_split(
            [item],
            cfg,
            llm=llm,
            prepared_bindings={item.episode.qa_id: prepared},
            **({"artifact_relocations": {
                item.episode.qa_id: prepared_run["artifact_relocation"]}}
               if prepared_run["artifact_relocation"] is not None else {}),
            **({"binding_derivation_contexts": {
                item.episode.qa_id: prepared_run["binding_derivation_context"]}}
               if prepared_run.get("binding_derivation_context") is not None else {}),
        )
        elapsed = time.perf_counter() - started
        _require(len(outcomes) == 1,
                 f"{variant_id}: system runner returned multiple outcomes")
        outcome = outcomes[0]
        _require(outcome.initial_skill_selection.get("evidence_state") == state,
                 f"{variant_id}: selected Skill state changed")
        _write_create_once(
            variant_root / "outcome.json",
            _json_value(asdict(outcome)),
        )
        _write_create_once(
            variant_root / "evaluation_run.json",
            evaluation_run.model_dump(mode="json"),
        )
        row = _system_result_row(
            outcome,
            variant=variant,
            options=list(item.episode.options or []),
            elapsed_s=elapsed,
            skill_identity=skill_identity,
        )
        if self.capture_offline_case_input:
            row["offline_input_sha256"] = file_identity(
                variant_root / "offline_input.json")["sha256"]
        _write_create_once(self._result_path(variant_id), row)
        if any(
            (record.get("execution_diagnostics") or {}).get("error_message")
            == "cell timeout requires a main-thread SIGALRM runtime"
            for record in outcome.rounds
        ):
            raise FitSetError("SIGALRM 基础设施故障，停止调度")
        return row

    def _run_shard(self, index, owned, pending, stop_event, accept):
        evolution = None
        if self.evolution_config is not None:
            from skill3d.evolution.fitset_stream_v11 import FitSetWorkerEvolution

            evolution = FitSetWorkerEvolution(self, index, owned)
            self.skills_for_variant = evolution.provider
            evolution.recover()
            existing = self.completed()
            for row in owned:
                if row["variant_id"] in existing:
                    evolution.ingest_result(row, existing[row["variant_id"]])
        for variant in pending:
            if stop_event is not None and stop_event.is_set():
                break
            result = self._run_one(variant)
            accept(result)
            if evolution is not None:
                evolution.ingest_result(variant, result)

    def _execute_preflight_pending(
        self,
        pending: Sequence[Mapping],
        operation: Callable[[Mapping], dict],
        accept: Callable[[dict], None],
    ) -> None:
        if self.workers == 1:
            for variant in pending:
                accept(operation(variant))
            return
        executor = ThreadPoolExecutor(
            max_workers=self.workers,
            thread_name_prefix=f"fitset-{self.phase}",
        )
        futures: dict[Future, str] = {
            executor.submit(operation, variant): str(variant["variant_id"])
            for variant in pending
        }
        try:
            for future in as_completed(futures):
                accept(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    def _execute_process_pending(
        self,
        pending: Sequence[Mapping],
        accept: Callable[[dict], None],
    ) -> None:
        if self.scheduler == "episode_pool":
            from skill3d.evaluation.episode_pool_v11 import execute_episode_pool
            execute_episode_pool(self, pending, accept)
            return
        context = multiprocessing.get_context("spawn")
        # One global permit pool, created from the spawn context so it can be
        # inherited by shard workers and their gate children. It bounds the
        # total number of episodes executing at once across the whole run.
        if self.episode_concurrency is not None and self._episode_semaphore is None:
            self._episode_semaphore = context.BoundedSemaphore(
                self.episode_concurrency)
        if self.workers == 1:
            owned = self._planned_slot_shards()[0]
            pending_ids = {row["variant_id"] for row in pending}
            worker = copy.copy(self)
            worker._run_shard(
                0, owned, [row for row in owned
                           if row["variant_id"] in pending_ids], None, accept)
            return
        owned_shards = self._planned_slot_shards()
        pending_ids = {row["variant_id"] for row in pending}
        shards = {
            index: [row for row in shard if row["variant_id"] in pending_ids]
            for index, shard in enumerate(owned_shards)
            if shard and (pending_ids or self.evolution_config is not None)
        }
        if not shards:
            return
        messages = context.Queue()
        stop_event = context.Event()
        processes: dict[int, multiprocessing.Process] = {}
        start_errors: list[str] = []
        for shard_index, shard in shards.items():
            process = context.Process(
                target=_process_shard_worker,
                args=(self, shard_index, owned_shards[shard_index],
                      shard, stop_event, messages, self._episode_semaphore),
                name=f"fitset-{self.phase}-{shard_index:02d}",
                daemon=False,
            )
            try:
                process.start()
            except BaseException as exc:  # noqa: BLE001
                start_errors.append(
                    f"shard {shard_index}: {type(exc).__name__}: {exc}")
                stop_event.set()
                break
            processes[shard_index] = process
        if start_errors:
            for process in processes.values():
                process.join()
            messages.close()
            messages.join_thread()
            raise FitSetError(
                "fit-set process worker startup failed: "
                + "; ".join(start_errors)
            )

        done: set[int] = set()
        errors: list[str] = []
        abort = None
        prior_handler = signal.getsignal(signal.SIGINT)

        def stop_after_current(_signum, _frame):
            nonlocal abort
            abort = KeyboardInterrupt()
            stop_event.set()

        signal.signal(signal.SIGINT, stop_after_current)
        try:
            while len(done) < len(processes):
                try:
                    message = messages.get(timeout=0.5)
                except queue.Empty:
                    for shard_index, process in processes.items():
                        if shard_index not in done and process.exitcode is not None:
                            done.add(shard_index)
                            if process.exitcode != 0:
                                errors.append(
                                    f"shard {shard_index} exited with code "
                                    f"{process.exitcode}")
                                stop_event.set()
                    continue
                except KeyboardInterrupt as exc:
                    abort = exc
                    stop_event.set()
                    continue
                kind = str(message.get("kind") or "")
                shard_index = int(message.get("shard_index", -1))
                if kind == "result":
                    if abort is None:
                        try:
                            accept(dict(message["result"]))
                        except BaseException as exc:
                            abort = exc
                            stop_event.set()
                elif kind == "error":
                    errors.append(f"shard {shard_index}: {message.get('error')}")
                    stop_event.set()
                elif kind == "done":
                    done.add(shard_index)
                else:
                    errors.append(f"shard {shard_index}: invalid message {kind!r}")
                    stop_event.set()
            for shard_index, process in processes.items():
                process.join()
                if process.exitcode != 0:
                    errors.append(
                        f"shard {shard_index} exited with code {process.exitcode}")
        finally:
            stop_event.set()
            signal.signal(signal.SIGINT, prior_handler)
            messages.close()
            messages.join_thread()
        if abort is not None:
            raise abort
        if errors:
            raise FitSetError(
                "fit-set process worker failed: " + "; ".join(errors))

    @staticmethod
    def _gate_spawn_runner(gate_runner, variant):
        provider = getattr(gate_runner, "published_inputs", None)
        factory = getattr(provider, "for_spawned_episode", None)
        if not callable(factory):
            return gate_runner
        _require(gate_runner.item_provider is provider,
                 "spawn replay requires one shared published input provider")
        clone = copy.copy(gate_runner)
        clone.published_inputs = factory(variant)
        clone.item_provider = clone.published_inputs
        clone.input_receipt_provider = clone.published_inputs.receipt
        return clone

    def _run_gate_episodes(
        self,
        items: Sequence[tuple[object, Mapping]],
    ) -> None:
        """Execute gate episodes under the global permit, writing each result.

        Each ``(gate_runner, variant)`` already has a persisted result when a
        prior attempt completed it, so only the missing ones are launched. The
        parent acquires one global permit before spawning a single-episode
        child and releases it when that child exits. Permits are taken with a
        non-blocking try so that several concurrent gates (one per slot worker)
        can never hold-and-wait on each other; a full pool simply means the
        parent reaps a finished child before starting the next.
        """
        pending = [
            (gate_runner, variant)
            for gate_runner, variant in items
            if not gate_runner._result_path(str(variant["variant_id"])).exists()
        ]
        if not pending:
            return
        semaphore = self._episode_semaphore
        if semaphore is None:
            for gate_runner, variant in pending:
                gate_runner._run_one(variant)
            return
        context = multiprocessing.get_context("spawn")
        messages = context.Queue()
        running: dict[int, tuple[multiprocessing.Process, bool]] = {}
        errors: list[str] = []
        index = 0
        try:
            while (index < len(pending) and not errors) or running:
                started = False
                if index < len(pending) and not errors:
                    acquired = semaphore.acquire(block=False)
                    if acquired:
                        gate_runner, variant = pending[index]
                        try:
                            process = context.Process(
                                target=_gate_episode_worker,
                                args=(self._gate_spawn_runner(gate_runner, variant),
                                      dict(variant), messages),
                                name=f"fitset-gate-{index:03d}", daemon=False,
                            )
                            process.start()
                        except BaseException as exc:  # noqa: BLE001
                            semaphore.release()
                            errors.append(
                                f"{variant['variant_id']}: "
                                f"{type(exc).__name__}")
                        else:
                            running[process.pid] = (process, False)
                            index += 1
                            started = True
                self._drain_gate_messages(
                    messages, running, errors, semaphore, block=not started)
        finally:
            self._join_gate_children(running, semaphore)
            messages.close()
            messages.join_thread()
        if errors:
            raise FitSetError("fit-set gate episode failed: " + "; ".join(errors))

    @staticmethod
    def _drain_gate_messages(messages, running, errors, semaphore, *, block):
        """Reap finished gate children, releasing each permit exactly once."""
        try:
            message = messages.get(timeout=0.5 if block else 0.05)
        except queue.Empty:
            message = None
        if message is not None and str(message.get("kind") or "") == "error":
            errors.append(
                f"{message.get('variant_id')}: {message.get('error')}")
        for pid, (process, released) in list(running.items()):
            if process.exitcode is None:
                continue
            if not released:
                semaphore.release()
            process.join()
            running.pop(pid, None)
            if process.exitcode != 0:
                errors.append(f"gate child {pid} exited with {process.exitcode}")

    @staticmethod
    def _join_gate_children(running, semaphore):
        for pid, (process, released) in list(running.items()):
            process.join()
            if not released:
                semaphore.release()
                running[pid] = (process, True)

    def _preflight_one(self, variant: Mapping) -> dict:
        values = self._prepare_one(variant, preflight=True)
        binding = values["prepared"]
        receipt = {
            "schema_version": "v11-fit-set-system-preflight/1.0",
            "variant_id": values["variant_id"],
            "input_identity_sha256": values["identity"]["sha256"],
            "binding_key": values["reference"]["key"],
            "binding_manifest_sha256": values["reference"][
                "manifest_sha256"],
            "m5_parameters_sha256": values["plan"][
                "m5_parameters_sha256"],
            "binding_object_count": len(binding.objects),
            "binding_materialized": bool(binding.materialized),
            "binding_status": str(
                binding.stats.get("status") or "available"),
            "runtime_evidence_state": values["state"],
            "skill_identity": dict(values["skill_identity"]),
            "model_requests": 0,
        }
        if values["plan"].get("published_transport") is not None:
            receipt["published_transport"] = values["plan"]["published_transport"]
        _write_create_once(
            self.root / "preflight"
            / f"{_variant_key(values['variant_id'])}.json",
            receipt,
        )
        return receipt

    def preflight(
        self,
        *,
        resume: bool,
        max_variants: int = 0,
        variant_ids: Sequence[str] = (),
    ) -> dict:
        manifest = self.initialize(resume=resume)
        self._validate_gate_inputs()
        wanted = set(variant_ids)
        known = {row["variant_id"] for row in self.variants}
        _require(not (wanted - known),
                 f"requested variants are outside the panel: {sorted(wanted - known)}")
        completed = {}
        for path in sorted((self.root / "preflight").glob("*.json")):
            receipt = json.loads(path.read_text(encoding="utf-8"))
            variant_id = str(receipt.get("variant_id") or "")
            _require(
                variant_id in known
                and path.stem == _variant_key(variant_id)
                and variant_id not in completed,
                f"invalid system preflight receipt: {path}",
            )
            completed[variant_id] = receipt
        pending = [
            row for row in self.execution_variants
            if row["variant_id"] not in completed
            and (not wanted or row["variant_id"] in wanted)
        ]
        if max_variants:
            _require(max_variants > 0, "max_variants must be non-negative")
            pending = pending[:max_variants]
        def accept(receipt: dict) -> None:
            completed[receipt["variant_id"]] = receipt

        self._execute_preflight_pending(
            pending, self._preflight_one, accept)
        summary = {
            "schema_version": "v11-fit-set-system-preflight/1.0",
            "status": (
                "complete"
                if len(completed) == len(self.variants)
                else "partial"
            ),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "selected_variants": len(self.variants),
            "completed_preflights": len(completed),
            "pending_preflights": len(self.variants) - len(completed),
            "model_requests": 0,
            "workers": self.workers,
            "scheduler": PREFLIGHT_SCHEDULER,
            "updated_at": time.time(),
        }
        _write_json_atomic(self.root / "preflight_summary.json", summary)
        return summary

    def run(
        self,
        *,
        resume: bool,
        max_variants: int = 0,
        variant_ids: Sequence[str] = (),
    ) -> dict:
        manifest = self.initialize(resume=resume)
        self._validate_gate_inputs()
        completed = self.completed()
        wanted = set(variant_ids)
        known = {row["variant_id"] for row in self.variants}
        _require(not (wanted - known),
                 f"requested variants are outside the panel: {sorted(wanted - known)}")
        pending = [
            row for row in self.execution_variants
            if row["variant_id"] not in completed
            and (not wanted or row["variant_id"] in wanted)
        ]
        if max_variants:
            _require(max_variants > 0, "max_variants must be non-negative")
            pending = pending[:max_variants]
        def accept(result: dict) -> None:
            completed[result["variant_id"]] = result
            self._write_summary(manifest, completed)

        try:
            self._execute_process_pending(pending, accept)
        except BaseException:
            completed = self.completed()
            self._write_summary(manifest, completed)
            raise
        self.evolution_finished = True
        return self._write_summary(manifest, self.completed())

    def _write_summary(self, manifest: Mapping, completed: Mapping[str, dict]) -> dict:
        counts: dict[str, int] = {}
        for row in completed.values():
            status = str(row.get("status") or "unknown")
            counts[status] = counts.get(status, 0) + 1
        summary = {
            "schema_version": SYSTEM_RUN_SCHEMA,
            "phase": self.phase,
            "status": (
                "complete"
                if len(completed) == len(self.variants)
                and (self.evolution_config is None or self.evolution_finished)
                else "partial"
            ),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "selected_variants": len(self.variants),
            "completed_variants": len(completed),
            "pending_variants": len(self.variants) - len(completed),
            "status_counts": counts,
            "workers": self.workers,
            "scheduler": self.execution_scheduler,
            "evolution_finished": (
                self.evolution_finished if self.evolution_config is not None else None),
            "updated_at": time.time(),
        }
        if self.request_governor_config is not None:
            from skill3d.synthesis.request_governor import RequestGovernor
            summary["requests"] = RequestGovernor(
                self.request_governor_root, self.request_governor_config).summary()
        _write_json_atomic(self.root / "summary.json", summary)
        return summary


def load_label_metadata(path: str | Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        raw_id = raw.get("qa_id")
        if raw_id is None or raw_id == "":
            raw_id = raw.get("id")
        qa_id = "" if raw_id is None else str(raw_id)
        _require(qa_id and qa_id not in rows,
                 f"duplicate or empty label qa_id: {qa_id!r}")
        rows[qa_id] = {
            "qa_id": qa_id,
            "scene_id": str(raw.get("scene_name") or ""),
            "question_type": canonical_task(
                str(raw.get("question_type") or "")),
            "options": [str(value) for value in (raw.get("options") or [])],
            "reference_answer": str(raw.get("ground_truth") or ""),
        }
    return rows


def score_answer(
    *,
    answer: object,
    task: str,
    options: Sequence[str],
    reference: str,
    available: bool,
) -> dict:
    unit = CANONICAL_UNIT_BY_QUESTION_TYPE[task]
    metric = "Accuracy" if unit == "option" else "MRA"
    if unit == "option":
        predicted = extract_option_letter(
            None if answer is None else str(answer))
        allowed = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(options)])
        ground_truth = extract_option_letter(reference) or reference.upper()
        _require(ground_truth in allowed,
                 f"{task}: invalid MCA reference {reference!r}")
        legal = predicted in allowed
        score = float(bool(available and legal and predicted == ground_truth))
    else:
        numeric = parse_numeric_answer(
            None if answer is None else str(answer))
        ground_truth = parse_numeric_answer(reference)
        _require(ground_truth is not None and ground_truth > 0,
                 f"{task}: invalid numeric reference {reference!r}")
        predicted = numeric
        legal = numeric is not None
        score = (
            mra_single(float(numeric), float(ground_truth))
            if available and legal
            else 0.0
        )
    return {
        "metric": metric,
        "predicted": predicted,
        "legal_answer": bool(legal),
        "available": bool(available),
        "score": float(score),
    }


def _load_system_results(root: Path, expected: set[str]) -> dict[str, dict]:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    _require(summary.get("schema_version") == SYSTEM_RUN_SCHEMA,
             "unsupported system run schema")
    _require(summary.get("status") == "complete",
             "final system run is not complete")
    rows = {}
    for path in sorted((root / "results").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        variant_id = str(row.get("variant_id") or "")
        _require(row.get("schema_version") == SYSTEM_RESULT_SCHEMA,
                 f"{path}: unsupported system result schema")
        _require(variant_id and variant_id not in rows,
                 f"duplicate system result: {variant_id!r}")
        rows[variant_id] = row
    _require(set(rows) == expected,
             "system results do not exactly cover selected variants")
    return rows


def _load_direct_results(
    root: Path,
    expected_qa_ids: set[str],
) -> tuple[dict[str, dict], dict[str, dict]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == DIRECT_RUN_SCHEMA,
             "unsupported direct run manifest")
    _require(summary.get("schema_version") == DIRECT_RUN_SCHEMA,
             "unsupported direct run summary")
    _require(summary.get("status") == "complete",
             "direct-Qwen run is not complete")
    selected_qa_ids = set(manifest.get("selected_qa_ids") or [])
    _require(expected_qa_ids <= selected_qa_ids,
             "direct-Qwen manifest does not cover selected QA IDs")
    responses = {}
    preflights = {}
    for qa_id in sorted(expected_qa_ids):
        response_path = root / "responses" / f"{qa_id}.json"
        preflight_path = root / "preflight" / f"{qa_id}.json"
        _require(response_path.is_file() and preflight_path.is_file(),
                 f"{qa_id}: direct response or preflight is missing")
        response = json.loads(response_path.read_text(encoding="utf-8"))
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        _require(response.get("schema_version") == DIRECT_RESPONSE_SCHEMA,
                 f"{qa_id}: direct response schema changed")
        _require(response.get("request_sha256")
                 == preflight.get("request_sha256"),
                 f"{qa_id}: direct request hash changed")
        responses[qa_id] = response
        preflights[qa_id] = preflight
    return responses, preflights


def _mean(rows: Sequence[Mapping], field: str) -> float | None:
    values = [float(row[field]) for row in rows
              if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def _cluster_interval(
    rows: Sequence[Mapping],
    *,
    delta_field: str,
    seed: int = 137,
    samples: int = 2000,
) -> list[float] | None:
    by_scene: dict[str, list[float]] = {}
    for row in rows:
        by_scene.setdefault(str(row["scene_id"]), []).append(
            float(row[delta_field]))
    scenes = sorted(by_scene)
    if len(scenes) < 2:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        sampled = [scenes[rng.randrange(len(scenes))]
                   for _ in range(len(scenes))]
        flat = [value for scene in sampled for value in by_scene[scene]]
        values.append(sum(flat) / len(flat))
    values.sort()
    lower = values[max(0, math.floor(0.025 * (samples - 1)))]
    upper = values[min(samples - 1, math.ceil(0.975 * (samples - 1)))]
    return [lower, upper]


def _summarize_group(rows: Sequence[Mapping]) -> dict:
    return {
        "n_variants": len(rows),
        "n_unique_qa": len({row["qa_id"] for row in rows}),
        "n_scenes": len({row["scene_id"] for row in rows}),
        "system_score": _mean(rows, "system_score"),
        "direct_score": _mean(rows, "direct_score"),
        "system_minus_direct": _mean(rows, "delta"),
        "scene_cluster_bootstrap_95pct": _cluster_interval(
            rows, delta_field="delta"),
        "system_legal_answer_rate": _mean(rows, "system_legal_answer"),
        "direct_legal_answer_rate": _mean(rows, "direct_legal_answer"),
        "system_available_rate": _mean(rows, "system_available"),
        "direct_available_rate": _mean(rows, "direct_available"),
        "system_run_error_rate": _mean(rows, "system_run_error"),
        "system_service_error_rate": _mean(rows, "system_service_error"),
        "system_mean_model_requests": _mean(
            rows, "system_model_request_count"),
        "system_mean_tool_calls": _mean(rows, "system_tool_call_count"),
        "system_mean_elapsed_s": _mean(rows, "system_elapsed_s"),
        "direct_mean_elapsed_s": _mean(rows, "direct_elapsed_s"),
    }


def build_fitset_comparison(
    *,
    panel: Mapping,
    panel_name: str,
    system_root: str | Path,
    direct_root: str | Path,
    label_metadata: Mapping[str, Mapping],
    output_dir: str | Path,
) -> dict:
    variants = select_panel_variants(panel, panel_name)
    expected_variants = {row["variant_id"] for row in variants}
    direct_rows = direct_rows_for_variants(panel, variants)
    expected_qa_ids = {row["qa_id"] for row in direct_rows}
    system = _load_system_results(
        Path(system_root).resolve(), expected_variants)
    direct, preflights = _load_direct_results(
        Path(direct_root).resolve(), expected_qa_ids)
    pairs = []
    for variant in variants:
        variant_id = variant["variant_id"]
        qa_id = variant["qa_id"]
        label = label_metadata.get(qa_id)
        _require(label is not None, f"{variant_id}: label metadata is missing")
        _require(label["scene_id"] == variant["scene_id"]
                 and label["question_type"] == variant["question_type"],
                 f"{variant_id}: label metadata identity changed")
        system_row = system[variant_id]
        _require(
            system_row.get("question_type") == variant["question_type"]
            and system_row.get("evidence_state") == variant["evidence_state"],
            f"{variant_id}: final system slot identity changed",
        )
        direct_row = direct[qa_id]
        preflight = preflights[qa_id]
        _require(variant_id in preflight.get("system_variant_ids", []),
                 f"{variant_id}: direct preflight mapping changed")
        system_available = (
            system_row.get("status") == "completed"
            and system_row.get("final_state") == "answer"
            and not system_row.get("service_errors")
        )
        direct_available = (
            direct_row.get("status") == "accepted"
            and bool(direct_row.get("accepted"))
        )
        direct_answer = (
            (direct_row.get("answer") or {}).get("value")
            if isinstance(direct_row.get("answer"), dict)
            else direct_row.get("answer")
        )
        system_score = score_answer(
            answer=system_row.get("answer"),
            task=variant["question_type"],
            options=label["options"],
            reference=label["reference_answer"],
            available=system_available,
        )
        direct_score = score_answer(
            answer=direct_answer,
            task=variant["question_type"],
            options=label["options"],
            reference=label["reference_answer"],
            available=direct_available,
        )
        _require(system_score["metric"] == direct_score["metric"],
                 f"{variant_id}: arm metrics differ")
        pair = {
            "variant_id": variant_id,
            "qa_id": qa_id,
            "scene_id": variant["scene_id"],
            "question_type": variant["question_type"],
            "evidence_state": variant["evidence_state"],
            "slot_key": fit_slot_key(variant["question_type"], variant["evidence_state"]),
            "system_skill_identity": system_row.get("skill_identity"),
            "split": variant["split"],
            "metric": system_score["metric"],
            "system_score": system_score["score"],
            "direct_score": direct_score["score"],
            "delta": system_score["score"] - direct_score["score"],
            "system_predicted": system_score["predicted"],
            "direct_predicted": direct_score["predicted"],
            "system_legal_answer": float(system_score["legal_answer"]),
            "direct_legal_answer": float(direct_score["legal_answer"]),
            "system_available": float(system_score["available"]),
            "system_run_error": float(system_row.get("final_state") == "run_error"),
            "system_service_error": float(bool(system_row.get("service_errors"))),
            "direct_available": float(direct_score["available"]),
            "system_status": system_row.get("status"),
            "direct_status": direct_row.get("status"),
            "system_model_request_count": system_row.get(
                "model_request_count"),
            "system_tool_call_count": system_row.get("tool_call_count"),
            "system_elapsed_s": system_row.get("elapsed_s"),
            "direct_elapsed_s": direct_row.get("elapsed_s"),
            "direct_request_sha256": direct_row.get("request_sha256"),
        }
        pairs.append(pair)
    groups = {"all": _summarize_group(pairs)}
    for field in ("question_type", "evidence_state", "slot_key", "split"):
        values = sorted({str(row[field]) for row in pairs})
        for value in values:
            groups[f"{field}:{value}"] = _summarize_group([
                row for row in pairs if str(row[field]) == value
            ])
    body = {
        "schema_version": COMPARISON_SCHEMA,
        "status": "complete",
        "panel_name": panel_name,
        "panel_proposal_sha256": panel["proposal_sha256"],
        "n_variants": len(pairs),
        "n_unique_qa": len(expected_qa_ids),
        "interpretation": (
            "fit-set comparison of the final complete Harness3D system versus "
            "qwen3.8-flash direct 32-frame visual answering; not a "
            "generalization, final-test, or isolated Skill-effect estimate"
        ),
        "denominator_contract": (
            "Every selected system variant remains in the denominator. "
            "Unavailable, service-error, missing, and illegal answers score zero."
        ),
        "groups": groups,
        "inputs": {
            "system_root": str(Path(system_root).resolve()),
            "direct_root": str(Path(direct_root).resolve()),
        },
    }
    report = {**body, "identity_sha256": content_sha256(body)}
    root = Path(output_dir).resolve()
    _require(not root.exists(), f"comparison output already exists: {root}")
    root.mkdir(parents=True)
    _write_create_once(root / "report.json", report)
    with (root / "paired_results.jsonl").open("wb") as stream:
        for row in pairs:
            stream.write(canonical_json_bytes(row) + b"\n")
    return report


__all__ = [
    "COMPARISON_SCHEMA",
    "FIT_BINDING_MIGRATION_SCHEMA",
    "FIT_SKILL_BUNDLE_SCHEMA",
    "FitSetError",
    "FitSkillBundle",
    "FitSlotSkill",
    "PANEL_NAMES",
    "REACHABLE_FIT_SLOTS",
    "ALL_FIT_SLOTS",
    "UNREACHABLE_FIT_SLOTS",
    "audit_fitset_pilot",
    "fit_slot_key",
    "SystemVariantRunner",
    "build_fit_skill_bundle",
    "build_fitset_comparison",
    "direct_rows_for_variants",
    "load_fit_skill_bundle",
    "load_fitset_panel",
    "load_label_metadata",
    "load_variant_bindings",
    "migrate_empty_m4_bindings",
    "score_answer",
    "select_panel_variants",
]
