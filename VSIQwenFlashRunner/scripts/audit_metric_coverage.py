#!/usr/bin/env python3
"""Read-only VSI1000 metric coverage and exact-quota replacement audit.

The default inventory needs only the standard library. --scan must run where
the original FullQA catalog's physical files exist and the shared skill3d
dependencies are installed. No models, bindings, labels or datasets are changed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "vsi1000-metric-coverage/1"


class AuditError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def checked(condition, code):
    if not condition:
        raise AuditError(code)


def load_json(path, sha256=None):
    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    checked(sha256 is None or actual == sha256, "input_file_hash_changed")
    return json.loads(data), actual


def canonical_task(raw):
    return "object_rel_direction" if raw.startswith("object_rel_direction_") else raw


def make_inventory(data_root):
    manifest, manifest_hash = load_json(data_root / "sampling_manifest.json")
    m4, m4_hash = load_json(
        data_root / manifest["m4_source"]["receipt_file"],
        manifest["m4_source"]["receipt_file_sha256"])
    panel, panel_hash = load_json(data_root / "panel.json")
    question_bytes = (data_root / manifest["online_questions"]["file"]).read_bytes()
    checked(hashlib.sha256(question_bytes).hexdigest()
            == manifest["online_questions"]["sha256"], "question_file_hash_changed")
    selected = []
    for line in question_bytes.splitlines():
        row = json.loads(line)
        checked(not any(row.get(name) not in (None, "") for name in (
            "answer", "ground_truth", "reference_answer", "label", "score")),
            "public_question_contains_labels")
        selected.append({
            "qa_id": str(row["qa_id"]), "scene_id": str(row["scene_name"]),
            "raw_type": row["question_type"], "source_split": row["source_split"],
            "experiment_split": row["experiment_split"],
        })
    checked(len({r["qa_id"] for r in selected}) == len(selected)
            == manifest["n_questions"], "selected_question_count_changed")
    checked(set(r["qa_id"] for r in selected) == set(manifest["selected_qa_ids"]),
            "selected_question_identity_changed")
    checked(Counter(r["raw_type"] for r in selected) == manifest["raw_type_quotas"],
            "selected_type_quotas_changed")
    for role in ("train", "test"):
        checked(Counter(r["raw_type"] for r in selected
                        if r["experiment_split"] == role)
                == manifest[f"{role}_raw_type_quotas"], "selected_role_quotas_changed")
    panel_rows = panel["system_variants"]
    checked(len(panel_rows) == len(selected)
            and {r["qa_id"] for r in panel_rows} == {r["qa_id"] for r in selected},
            "panel_question_identity_changed")
    scenes = []
    for scene in m4["scenes"]:
        scenes.append({
            "scene_id": scene["scene_id"], "source_split": scene["source_split"],
            "qa_ids": scene["qa_ids"],
            "m4_passed": bool(scene["variants"]["base"]["main_gate_passed"]
                              and scene["variants"]["metric"]["main_gate_passed"]),
        })
    checked(m4["fullqa_identity_hash"] == manifest["m4_source"]["fullqa_identity_hash"],
            "fullqa_identity_changed")
    checked(sum(len(r["qa_ids"]) for r in scenes if r["m4_passed"])
            == manifest["m4_source"]["n_questions"], "m4_pool_count_changed")
    checked(all(r["scene_id"] in {s["scene_id"] for s in scenes if s["m4_passed"]}
                for r in selected), "selected_scene_not_m4_success")
    return {
        "schema": SCHEMA, "selection_id": manifest["selection_id"],
        "seed": manifest["seed"], "selected": selected, "scenes": scenes,
        "raw_type_quotas": manifest["raw_type_quotas"],
        "task_quotas": manifest["task_quotas"],
        "m4_raw_type_upper_bounds": manifest["successful_raw_type_counts"],
        "published_states": dict(Counter(r["evidence_state"] for r in panel_rows)),
        "anchors": {
            "catalog_sha256": manifest["prepared_inputs"]["original_catalog_sha256"],
            "catalog_identity_hash": m4["fullqa_identity_hash"],
            "sampling_manifest_sha256": manifest_hash,
            "m4_receipts_sha256": m4_hash, "panel_sha256": panel_hash,
            "questions_sha256": manifest["online_questions"]["sha256"],
        },
    }


def checked_identity(entry, validation):
    identity, _ = load_json(entry["scene_input_identity_file"],
                            entry["scene_input_identity_file_sha256"])
    checked(hashlib.sha256(canonical({k: v for k, v in identity.items()
                                     if k != "sha256"})).hexdigest()
            == identity["sha256"] == entry["reference"]["scene_input_identity_sha256"],
            "logical_input_hash_changed")
    episode = identity["episode"]
    checked(episode.get("ground_truth") == "", "reference_answer_not_empty")
    checked(str(episode["qa_id"]) == entry["qa_id"]
            and str(episode["scene_name"]) == entry["scene_id"],
            "episode_identity_changed")
    migration = validation["variants"]["metric"]["migration_receipts"][entry["qa_id"]]
    checked(isinstance(migration, dict), "migration_must_be_embedded")
    checked(migration["logical_input_identity"] == identity,
            "migration_logical_identity_changed")
    checked(hashlib.sha256(canonical({k: v for k, v in migration.items()
                                     if k != "receipt_sha256"})).hexdigest()
            == migration["receipt_sha256"], "migration_hash_changed")
    checked(identity["artifact_path"] == entry["reference"]["scene_artifact_ref"],
            "source_artifact_changed")
    checked(migration["destination_root"] == entry["publication_root"]
            and str(Path(entry["publication_root"]) / migration["artifact"]["target_ref"])
            == entry["scene_artifact"], "published_artifact_location_changed")
    raw = episode["question_type"]
    return identity, migration, {
        "qa_id": entry["qa_id"], "scene_id": entry["scene_id"], "raw_type": raw,
        "source_split": entry["split"],
    }


def native_scene_check(entry, identity, migration):
    """Use native relocation and numerical gates; task authorization is separate."""
    from skill3d.reconstruction.metric_evidence import validated_receipt
    from skill3d.reconstruction_gate.evidence_profile import evaluate_metric_gate
    from skill3d.reconstruction_gate.scene_state import scene_route_from_quality
    from skill3d.schemas.evidence import GATE_SUBCONDITIONS, metric_scale_state_from_gate
    from skill3d.storage.relocation import validate_relocation_receipt

    resolver = validate_relocation_receipt(
        migration, expected_input_sha256=identity["sha256"],
        expected_source_artifact=entry["reference"]["scene_artifact_ref"],
        destination_root=entry["publication_root"])
    checked(resolver.logical_input_identity == identity, "resolved_identity_changed")
    checked(str(resolver.artifact_path) == entry["scene_artifact"],
            "resolved_artifact_location_changed")
    art = resolver.resolved_artifact()
    checked(art.scene_name == entry["scene_id"], "artifact_scene_changed")
    receipt = validated_receipt(art)
    route = scene_route_from_quality(
        art.quality, quality_status=art.quality_status,
        quality_metric_version=art.quality_metric_version)
    gate = evaluate_metric_gate(
        scene_route=route, main_gate_passed=bool(getattr(art.quality, "main_gate_passed", False)),
        fusion_status=art.scale_fusion_status,
        scale_self_consistency=art.scale_self_consistency,
        valid_frame_ratio=receipt["valid_frame_ratio"], metric_scale=art.metric_scale,
        question_type="")
    passed = metric_scale_state_from_gate(gate) == "available"
    return {
        "status": "pass" if passed else "fail",
        "scene_route": route, "gate_version": gate.gate_version,
        "scale_subconditions": {k: gate.sub_results[k] for k in GATE_SUBCONDITIONS[:5]},
        "metric_scale": art.metric_scale,
        "scale_self_consistency": art.scale_self_consistency,
        "valid_frame_ratio": receipt["valid_frame_ratio"],
    }


def scan(inventory, catalog_path, source_root, workers=1):
    sys.path.insert(0, str(source_root))
    catalog, _ = load_json(catalog_path, inventory["anchors"]["catalog_sha256"])
    checked(catalog["status"] == "verified" and catalog["no_labels"] is True
            and catalog["identity_hash"] == inventory["anchors"]["catalog_identity_hash"],
            "catalog_identity_changed")
    entries = catalog["variants"]["metric"]
    checked(set(entries) == {q for s in inventory["scenes"] for q in s["qa_ids"]},
            "catalog_question_scope_changed")
    def check_scene(scene):
        public_rows, signatures = [], []
        scene_result = {"status": "unknown"}
        try:
            first = entries[scene["qa_ids"][0]]
            validation, _ = load_json(
                first["migration_receipt_file"], first["migration_receipt_file_sha256"])
            checked(validation["status"] == "validated"
                    and validation["no_labels"] is True
                    and validation["scene_id"] == scene["scene_id"],
                    "scene_validation_identity_changed")
            representative = None
            for qa in scene["qa_ids"]:
                entry = entries[qa]
                checked(entry["qa_id"] == qa and entry["scene_id"] == scene["scene_id"]
                        and entry["split"] == scene["source_split"]
                        and entry["variant"] == "metric"
                        and entry["migration_receipt_file"] == first["migration_receipt_file"]
                        and entry["migration_receipt_file_sha256"]
                        == first["migration_receipt_file_sha256"],
                        "catalog_scene_entry_changed")
                identity, migration, public = checked_identity(entry, validation)
                checked(public["raw_type"] in inventory["raw_type_quotas"],
                        "unknown_question_type")
                signatures.append({
                    "artifact": migration["artifact"], "dependencies": migration["dependencies"],
                    "destination_root": migration["destination_root"],
                    "files": identity["files"], "pixels": identity["pixels"],
                    "source": identity["source"], "video_path": identity["video_path"],
                    "artifact_id": identity["artifact_id"],
                    "artifact_version": identity["artifact_version"],
                    "artifact_json_sha256": identity["artifact_json_sha256"],
                })
                public_rows.append(public)
                representative = representative or (entry, identity, migration)
            checked(all(s == signatures[0] for s in signatures),
                    "same_scene_geometry_identity_changed")
            scene_result = native_scene_check(*representative)
        except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
            # Only the exception class is retained, never raw exception text.
            scene_result = {"status": "unknown", "reason": type(exc).__name__}
        statuses = {r["qa_id"]: r for r in public_rows}
        scene_rows = []
        for qa in scene["qa_ids"]:
            row = statuses.get(qa, {
                "qa_id": qa, "scene_id": scene["scene_id"],
                "raw_type": None, "source_split": scene["source_split"],
            })
            scene_rows.append({**row, "status": scene_result["status"]})
        return scene_rows, {"scene_id": scene["scene_id"], **scene_result}

    rows, scene_results = [], []
    candidates = [s for s in inventory["scenes"] if s["m4_passed"]]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for scene_rows, scene_result in executor.map(check_scene, candidates):
            rows.extend(scene_rows)
            scene_results.append(scene_result)
            print(f"metric audit {len(scene_results)}/{len(candidates)}: "
              f"{scene_result['status']}", file=sys.stderr, flush=True)
    return rows, scene_results


def quota_table(inventory, rows, canonical_types=False):
    convert = canonical_task if canonical_types else lambda value: value
    selected = inventory["selected"]
    status_by_qa = {r["qa_id"]: r["status"] for r in rows}
    quotas = Counter(convert(r["raw_type"]) for r in selected)
    upper = Counter()
    for raw, count in inventory["m4_raw_type_upper_bounds"].items():
        upper[convert(raw)] += count
    passing, failing, outside = Counter(), Counter(), Counter()
    selected_ids = {r["qa_id"] for r in selected}
    for row in rows:
        if row["raw_type"] is None:
            continue
        task = convert(row["raw_type"])
        if row["status"] == "pass":
            passing[task] += 1
            outside[task] += int(row["qa_id"] not in selected_ids)
        elif row["status"] == "fail":
            failing[task] += 1
    table = []
    for task, target in quotas.items():
        current = Counter(status_by_qa.get(r["qa_id"], "unknown")
                          for r in selected if convert(r["raw_type"]) == task)
        lower, maximum = passing[task], upper[task] - failing[task]
        checked(lower <= maximum, "pool_counts_exceed_inventory")
        feasibility = ("enough" if lower >= target else
                       "insufficient" if maximum < target else "unknown")
        table.append({
            "type": task, "target": target, "current_pass": current["pass"],
            "current_fail": current["fail"], "current_unknown": current["unknown"],
            "m4_candidate_upper_bound": upper[task],
            "eligible_lower_bound": lower, "eligible_upper_bound": maximum,
            "confirmed_replacements_outside_current": outside[task],
            "quota_feasibility": feasibility,
        })
    return table


def make_report(inventory, rows=(), scenes=(), scanned=False):
    rows = list(rows)
    selected_ids = {r["qa_id"] for r in inventory["selected"]}
    by_qa = {r["qa_id"]: r for r in rows}
    current = Counter(by_qa.get(qa, {}).get("status", "unknown") for qa in selected_ids)
    for selected in inventory["selected"]:
        actual = by_qa.get(selected["qa_id"])
        if actual is not None and actual["raw_type"] is not None:
            checked(all(actual[k] == selected[k] for k in (
                "raw_type", "scene_id", "source_split")), "selected_public_metadata_changed")
    raw_table = quota_table(inventory, rows)
    # Source split is an immutable property; joint-cell quotas detect hidden
    # shortages even when all aggregate task quotas have enough candidates.
    joint_targets = Counter((r["raw_type"], r["source_split"])
                            for r in inventory["selected"])
    joint_pass = Counter((r["raw_type"], r["source_split"])
                         for r in rows if r["status"] == "pass")
    complete = scanned and all(r["status"] != "unknown" for r in rows)
    joint_feasibility = ("enough" if all(joint_pass[cell] >= n
                                       for cell, n in joint_targets.items()) else
                         "insufficient" if complete else "unknown")
    raw_feasibility = ("enough" if all(r["quota_feasibility"] == "enough" for r in raw_table)
                       else "insufficient" if any(r["quota_feasibility"] == "insufficient"
                                                 for r in raw_table) else "unknown")
    selected_scenes = {r["scene_id"] for r in inventory["selected"]}
    observed_scenes = {r["scene_id"]: r["status"] for r in scenes}
    scene_counts = Counter(observed_scenes.get(s, "unknown") for s in selected_scenes)
    return {
        "schema": SCHEMA, "observed_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if complete else "pending_metric_evidence",
        "selection_id": inventory["selection_id"], "seed": inventory["seed"],
        "anchors": inventory["anchors"], "model_requests": 0,
        "reference_answers_read": False, "dataset_mutated": False,
        "published_states": inventory["published_states"],
        "current_metric_qualification": {k: current[k] for k in ("pass", "fail", "unknown")},
        "current_scene_qualification": {
            k: scene_counts[k] for k in ("pass", "fail", "unknown")},
        "fullqa_questions": sum(len(s["qa_ids"]) for s in inventory["scenes"]),
        "m4_candidate_questions": sum(inventory["m4_raw_type_upper_bounds"].values()),
        "m4_candidate_scenes": sum(s["m4_passed"] for s in inventory["scenes"]),
        "raw_type_quota_feasibility": raw_feasibility,
        "source_split_and_raw_type_quota_feasibility": joint_feasibility,
        "raw_types": raw_table,
        "task_types": quota_table(inventory, rows, canonical_types=True),
        "joint_source_type_quotas": [
            {"raw_type": cell[0], "source_split": cell[1], "target": n,
             "confirmed_eligible": joint_pass[cell]} for cell, n in joint_targets.items()],
        "scene_checks": list(scenes),
        "qualification_contract": (
            "Native metric gate's first five subconditions determine g1m1; "
            "the sixth only authorizes metric tools for metric question types."),
        "scene_coverage_contract": (
            "Any failed selected scene prevents preserving every original scene "
            "in an all-g1m1 panel, even when question-type quotas are feasible."),
        "evidence_limits": (
            "M4 candidate counts come from the pinned prior M4 audit and are "
            "upper bounds before scale qualification. Unknown is not failure."),
    }


def markdown(report):
    current = report["current_metric_qualification"]
    text = [
        "# VSI1000 米制覆盖核验",
        "",
        f"状态：{report['status']}。",
        f"当前发布状态：{json.dumps(report['published_states'])}。",
        f"Metric 版本已核实合格 {current['pass']} 题、不合格 {current['fail']} 题、"
        f"待核实 {current['unknown']} 题。待核实不表示尺度不可用。",
        "",
        "| 题型 | 原配额 | M4 候选上限 | g1m1 已确认 / 可能上限 | 替补可行性 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report["task_types"]:
        text.append(f"| {row['type']} | {row['target']} | "
                    f"{row['m4_candidate_upper_bound']} | "
                    f"{row['eligible_lower_bound']} / {row['eligible_upper_bound']} | "
                    f"{row['quota_feasibility']} |")
    text.extend([
        "", "方向题细配额：easy 42、medium 74、hard 73。",
        "替补核验还检查题型与原始 source split 的联合配额；train/test 的原题型配额保留。",
        "全 g1m1 会排除尺度门不合格的整场景，因此原 164 场景覆盖需单独检查。",
        "本次不读取参考答案，不调用模型，不修改数据包。",
        "", "核验入口：`scripts/audit_metric_coverage.py`。默认只盘点；"
        "使用 `--scan` 才读取原目录中带哈希锚点的 metric 证据。",
    ])
    return "\n".join(text) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=ROOT.parent.parent / "vsibench-m4-success-1000")
    parser.add_argument("--inventory", type=Path,
                        help="Public inventory bundle, for auditing on the source host")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT / "src")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent read-only scene audits (1-4)")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    checked(1 <= args.workers <= 4, "workers_must_be_between_one_and_four")
    inventory = load_json(args.inventory)[0] if args.inventory else make_inventory(args.data_root)
    checked(inventory["schema"] == SCHEMA, "unsupported_inventory")
    rows, scenes = (scan(inventory, args.catalog or args.data_root / "source_catalog.json",
                         args.source_root, args.workers) if args.scan else ([], []))
    report = make_report(inventory, rows, scenes, scanned=args.scan)
    if args.output_dir:
        output = args.output_dir.resolve()
        checked(output.is_relative_to((ROOT / "runs").resolve()),
                "output_must_be_inside_runner_runs")
        output.mkdir(parents=True, exist_ok=False)
        for name, value in (("inventory.json", inventory), ("report.json", report)):
            (output / name).write_text(
                json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                           allow_nan=False) + "\n")
        (output / "report.md").write_text(markdown(report))
        print(output / "report.json")
    else:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"metric audit stopped: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(2)
