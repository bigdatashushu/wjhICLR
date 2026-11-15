"""把旧同题方案升级为 19 可达槽位合同，旧文件保持不可变。"""

from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path

from skill3d.evaluation.fitset_v11 import (
    ALL_FIT_SLOTS, PANEL_SCHEMA, REACHABLE_FIT_SLOTS, UNREACHABLE_FIT_SLOTS,
    FitSetError, _require, _write_create_once, content_sha256, fit_slot_key,
    load_fitset_panel, select_panel_variants,
)


def upgrade_fitset_panel(source: str | Path, destination: str | Path) -> dict:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    _require(source != destination and not destination.exists(),
             "panel 升级必须使用新的目标文件")
    old = json.loads(source.read_text())
    _require(old.get("schema_version")
             == "v11-same-question-system-vs-direct-qwen-proposal/1.0",
             "仅接受旧 v1 panel")
    old_body = {key: value for key, value in old.items()
                if key != "proposal_sha256"}
    _require(old["proposal_sha256"] == content_sha256(old_body), "旧 panel hash 不符")
    body = copy.deepcopy(old_body)
    variants = [
        row for row in old["system_variants"]
        if (row["question_type"], row["evidence_state"]) in set(REACHABLE_FIT_SLOTS)
    ]
    ids = {row["variant_id"] for row in variants}
    direct = []
    for row in old["direct_qwen_rows"]:
        mapped = [value for value in row["system_variant_ids"] if value in ids]
        if mapped:
            direct.append({**row, "system_variant_ids": mapped})
    body["schema_version"] = PANEL_SCHEMA
    body["system_name"] = "Harness3D"
    body["experiment_id"] = old["experiment_id"] + "-slot-stream-v2"
    body["status"] = "planned_slot_stream_not_executed"
    body["recommended_panel"] = "recommended_core_897"
    body["system_variants"] = variants
    body["direct_qwen_rows"] = direct
    body["upgraded_from"] = {
        "proposal_sha256": old["proposal_sha256"],
        "input_file_sha256": __import__("hashlib").sha256(source.read_bytes()).hexdigest(),
        "ready_input_identity_preserved": True,
        "completed_direct_run_reusable_by_qa_subset": True,
        "removed_unreachable_variants": len(old["system_variants"]) - len(variants),
    }
    body["panels"].pop("maximum_1166", None)
    for name, panel in body["panels"].items():
        selected = ([
            row for row in variants if row["system_input_status"] == "ready"
        ] if name == "ready_579" else variants)
        slots = {fit_slot_key(row["question_type"], row["evidence_state"])
                 for row in selected}
        _require(panel["system_input_variants"] == len(selected),
                 "合法 panel 题量改变")
        panel["skill_slots"] = sorted(slots)
        panel["skill_slot_count"] = len(slots)
        counts = Counter(fit_slot_key(row["question_type"], row["evidence_state"])
                         for row in selected)
        panel["slot_case_counts"] = dict(sorted(counts.items()))
        panel["planned_full_batch_attempts"] = sum(n // 12 for n in counts.values())
        panel["slots_below_batch_size"] = [key for key, n in sorted(counts.items()) if n < 12]
        panel["slots_below_gate_size"] = [key for key, n in sorted(counts.items()) if n < 8]
    body["skill_accumulation"] = {
        "grouping_key": ["question_type", "evidence_state"],
        "skill_count": len(REACHABLE_FIT_SLOTS),
        "active_snapshot_physical_slot_count": len(ALL_FIT_SLOTS),
        "reachable_slot_count": len(REACHABLE_FIT_SLOTS),
        "unreachable_slots": [fit_slot_key(*slot) for slot in UNREACHABLE_FIT_SLOTS],
        "batch_size": 12,
        "gate_size": 8,
        "gate_seeds": [137, 138],
        "publication_policy": "strict_score_increase_error_nonincrease_legal_nondecrease",
        "worker_library": "isolated_seed_with_exclusive_slot_ownership",
        "source_split_preserved": True,
        "offline_prompt_raw_sample_content": False,
        "final_version_policy": "last_successfully_published_per_slot",
        "tail_revision": False,
        "gate_outputs_enter_revision": False,
    }
    body["execution_gates"] = [
        "Only 19 reachable question_type/evidence_state slots participate.",
        "Use spawn processes with unique slot ownership; sandbox runs on the main thread.",
        "Freeze full-panel slot ownership, gate inputs, model, tools, budgets and seeds before running.",
        "Every 12 new eligible traces triggers one exact-parent revision attempt.",
        "Publish only after strict dual-seed A/B, identity, permission and leakage checks.",
        "Never send gate outcomes to the offline reviser.",
        "Freeze each slot's last published version; preserve all other active parents.",
        "Preserve original source split and explicitly report fit-set reuse.",
        "Run audit-pilot before expanding; preserve the 80 GiB reserve.",
    ]
    inventory = body.get("task_inventory") or []
    for row in inventory:
        if row["question_type"] not in {
                task for task, state in REACHABLE_FIT_SLOTS if state == "g1m1"}:
            row["g1m1"] = {"total": 0, "reachable": False}
    for key, value in body.get("comparison_arms", {}).items():
        if key != "D_qwen_direct":
            value["description"] = "Final frozen Harness3D with exact-slot Skills and frozen evidence"
    body["g1m1_preparation"] = {
        **(body.get("g1m1_preparation") or {}),
        "potential_variants": sum(row["evidence_state"] == "g1m1" for row in variants),
        "reachable_metric_tasks_only": True,
    }
    panel = {**body, "proposal_sha256": content_sha256(body)}
    _write_create_once(destination, panel)
    loaded = load_fitset_panel(destination)
    for name in body["panels"]:
        select_panel_variants(loaded, name)
    return panel
