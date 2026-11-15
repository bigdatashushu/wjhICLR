#!/usr/bin/env python3
"""Verify the paired request evidence and compute official VSI scores offline."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys


NAMES = {
    "object_rel_distance": "相对距离", "object_rel_direction": "相对方向",
    "object_counting": "物体计数",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_root(root, content_sha256):
    experiment = read(root / "diagnostic.json")
    assert content_sha256({k: v for k, v in experiment.items()
                           if k != "identity_sha256"}) == experiment["identity_sha256"]
    collection = read(root / "collection_receipt.json")
    assert collection["diagnostic_identity"] == experiment["identity_sha256"]
    for relative, identity in collection["files"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == identity["sha256"]
    for relative, sha in experiment["input_files"].items():
        assert hashlib.sha256((root / "inputs" / relative).read_bytes()).hexdigest() == sha
    for row in read(root / "code/code_identity.json")["files"]:
        assert hashlib.sha256((root / "code/source" / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    return experiment


def verify_reused_direct_inputs(root, experiment, direct_root, direct_experiment):
    # The RGB baseline can be reused across base/metric runs when its QA and
    # frozen pixels are identical. Geometry and Skill references are arm-specific.
    geometry_fields = {
        "artifact_ref", "binding_key", "binding_manifest_sha256",
        "evidence_state", "variant_id",
    }
    def public_variants(manifest):
        return sorted(
            ({k: v for k, v in row.items() if k not in geometry_fields}
             for row in manifest["variants"]),
            key=lambda row: row["qa_id"],
        )
    assert public_variants(experiment) == public_variants(direct_experiment)
    arm_manifests = {"panel.json", "published_inputs.json", "scene_overviews/manifest.json"}
    assert {k: v for k, v in experiment["input_files"].items() if k not in arm_manifests} == {
        k: v for k, v in direct_experiment["input_files"].items() if k not in arm_manifests}
    published = read(root / "inputs/published_inputs.json")
    direct_published = read(direct_root / "inputs/published_inputs.json")
    assert published["source_catalog_sha256"] == direct_published["source_catalog_sha256"]
    for row in experiment["variants"]:
        qa = row["qa_id"]
        for key in ("qa_id", "scene_id", "split", "input_root", "frozen_scene_sha256"):
            assert published["entries"][qa][key] == direct_published["entries"][qa][key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--direct-run-root", type=Path,
                        help="Reuse a fully completed direct arm from a separately collected run.")
    parser.add_argument("--system-completion-root", type=Path,
                        help="Complete only documented transport-interrupted episodes from another run.")
    args = parser.parse_args()
    root = args.run_root.resolve()
    sys.path.insert(0, str(root / "code/source/src"))
    from skill3d.evaluation.fitset_v11 import (
        content_sha256, load_label_metadata, score_answer, _variant_key)
    from skill3d.infra.experiment_campaign import digest
    from skill3d.synthesis.prompt_compaction import COMPACTION_VERSION
    assert COMPACTION_VERSION == "vsi-prompt-compaction/1.0"
    experiment = verify_root(root, content_sha256)
    completion_root = (args.system_completion_root.resolve()
                       if args.system_completion_root else None)
    completion_experiment = None
    completion_requests = []
    storage_reconciliation = None
    recovery_key = "transport_completion"
    if completion_root:
        completion_experiment = verify_root(completion_root, content_sha256)
        if "storage_completion" in completion_experiment:
            recovery_key = "storage_completion"
            from reconcile_storage_requests import verify
            storage_reconciliation = verify(root)
    assert read(root / "exit.json")["status"] == ("stopped" if completion_root else "complete")
    direct_root = args.direct_run_root.resolve() if args.direct_run_root else root
    direct_experiment = experiment
    if direct_root != root:
        direct_experiment = verify_root(direct_root, content_sha256)
        verify_reused_direct_inputs(root, experiment, direct_root, direct_experiment)
        assert read(direct_root / "direct_exit.json")["exit_code"] == 0
        assert read(direct_root / "direct_receipt.json")["status"] == "complete"
        assert read(direct_root / "direct_start.json")["automatic_model_retry"] is False
    system_identity = read(root / "system/manifest.json")
    assert system_identity["reference_answer_visible_to_online_model"] is False
    assert system_identity["online_executor_input_is_label_free"] is True
    system_summary = read(root / "system/summary.json")
    assert system_summary["status"] == ("partial" if completion_root else "complete")
    if storage_reconciliation:
        assert not (root / "system/episode_pool_receipt.json").exists()
        assert read(root / "exit.json")["postmortem"] is True
    else:
        scheduler = read(root / "system/episode_pool_receipt.json")
        assert scheduler["status"] == ("failed" if completion_root else "drained")
    direct_identity = read(direct_root / "direct/manifest.json")
    assert direct_identity["campaign_identity"] == direct_experiment["identity_sha256"]
    assert direct_identity["enable_thinking"] is direct_identity["rgb_only"] is True
    assert direct_identity["max_requests_per_question"] == 1
    assert direct_identity["model"] == "qwen3.8-flash"
    system = {read(p)["qa_id"]: read(p) for p in (root / "system/results").glob("*.json")}
    direct = {read(p)["qa_id"]: read(p) for p in (direct_root / "direct/responses").glob("*.json")}
    expected = {v["qa_id"] for v in experiment["variants"]}
    expected_states = {v["variant_id"]: v["evidence_state"] for v in experiment["variants"]}
    system_roots = {qa: root for qa in system}
    if storage_reconciliation:
        assert 0 <= system_summary["completed_variants"] <= len(system)
        assert len(system) == len(read(root / "storage_interruption_receipt.json")["completed_variant_ids"])
    else:
        assert system_summary["completed_variants"] == len(system)
    if completion_root:
        completion_experiment = verify_root(completion_root, content_sha256)
        recovery = completion_experiment[recovery_key]
        assert recovery["origin_identity"] == experiment["identity_sha256"]
        if direct_root != root:
            assert experiment["comparison"]["arms"] == ["system"]
            assert recovery["reused_direct_identity"] == direct_experiment["identity_sha256"]
        if "origin_collection_sha256" in recovery:
            assert recovery["origin_collection_sha256"] == hashlib.sha256(
                (root / "collection_receipt.json").read_bytes()).hexdigest()
        if "interruption_receipt_sha256" in recovery:
            assert recovery["interruption_receipt_sha256"] == hashlib.sha256(
                (root / "transport_interruption_receipt.json").read_bytes()).hexdigest()
        if storage_reconciliation:
            assert recovery["storage_interruption_receipt_sha256"] == hashlib.sha256(
                (root / "storage_interruption_receipt.json").read_bytes()).hexdigest()
            assert recovery["gateway_request_evidence_sha256"] == hashlib.sha256(
                (root / "gateway_request_evidence_receipt.json").read_bytes()).hexdigest()
        assert recovery["selection_used_reference_answers"] is False
        assert recovery["automatic_model_retry"] is False
        assert completion_experiment["source_sha256"] == experiment["source_sha256"]
        assert completion_experiment["comparison"]["arms"] == ["system"]
        assert read(completion_root / "exit.json")["status"] == "complete"
        completion_summary = read(completion_root / "system/summary.json")
        assert completion_summary["status"] == "complete"
        assert read(completion_root / "system/episode_pool_receipt.json")["status"] == "drained"
        completion_identity = read(completion_root / "system/manifest.json")
        assert completion_identity["reference_answer_visible_to_online_model"] is False
        assert completion_identity["online_executor_input_is_label_free"] is True
        added = {read(p)["qa_id"]: read(p)
                 for p in (completion_root / "system/results").glob("*.json")}
        assert set(added) == expected - set(system)
        assert completion_summary["completed_variants"] == len(added)
        assert completion_experiment["n_questions"] == len(added)
        assert completion_experiment["variants"] == [
            v for v in experiment["variants"] if v["qa_id"] in added]
        assert set(recovery.get("missing_variant_ids", recovery["failed_variant_ids"])) == {
            r["variant_id"] for r in added.values()}
        assert set(recovery["origin_variants_not_repeated"]) == {
            r["variant_id"] for r in system.values()}
        for name, sha in experiment["input_files"].items():
            if name not in {"online.yaml", "panel.json", "sample_questions.json"}:
                assert completion_experiment["input_files"][name] == sha
        import yaml
        original_cfg = yaml.safe_load((root / "inputs/online.yaml").read_text())
        completion_cfg = yaml.safe_load((completion_root / "inputs/online.yaml").read_text())
        assert original_cfg["online_model"].pop("timeout_s") == recovery["previous_timeout_s"]
        assert completion_cfg["online_model"].pop("timeout_s") == recovery["new_timeout_s"]
        assert original_cfg == completion_cfg
        system.update(added)
        system_roots.update({qa: completion_root for qa in added})
    assert set(system) == set(direct) == expected
    assert Counter(r["question_type"] for r in system.values()) == experiment["task_counts"]
    state = read(root / "requests/state.json")
    assert state["policy"]["max_inflight"] == experiment["runtime"]["request_concurrency"]
    assert state["peak_inflight"] <= state["policy"]["max_inflight"]
    requests = storage_reconciliation["requests"] if storage_reconciliation else state["requests"]
    failed_requests = [r for r in requests if r["status"] == "error"]
    if completion_root:
        recovery = completion_experiment[recovery_key]
        assert {r["context"]["variant_id"] for r in failed_requests} == set(
            recovery["failed_variant_ids"])
        assert [r["request_id"] for r in failed_requests] == recovery["failed_request_ids"]
        assert [{k: v for k, v in r.items() if k != "gateway_reconciliation"}
                for r in failed_requests] == recovery["origin_failed_requests"]
        assert len(failed_requests) == len(recovery["failed_variant_ids"])
        assert all(r["exception_type"] == "DiagnosticOperationalStop" for r in failed_requests)
        assert all(recovery["previous_timeout_s"] - 5 <= r["completed_at"] - r["started_at"]
                   <= recovery["previous_timeout_s"] + 10 for r in failed_requests)
        completion_state = read(completion_root / "requests/state.json")
        assert completion_state["policy"]["max_inflight"] == state["policy"]["max_inflight"]
        assert completion_state["peak_inflight"] <= completion_state["policy"]["max_inflight"]
        completion_requests = completion_state["requests"]
        assert all(r["status"] == "completed" and r["exception_type"] is None
                   and not r["context"]["variant_id"].startswith("baseline:")
                   for r in completion_requests)
        assert len(completion_requests) <= completion_experiment["comparison"]["system_request_cap"]
        assert len(requests) + len(completion_requests) <= experiment["runtime"]["max_requests"]
    else:
        assert not failed_requests
    assert all(r["status"] == "completed" and r["exception_type"] is None
               for r in requests if r not in failed_requests)
    system_requests = [r for r in requests if not r["context"]["variant_id"].startswith("baseline:")]
    system_requests += completion_requests
    direct_state = state if direct_root == root else read(direct_root / "requests/state.json")
    assert direct_state["policy"]["max_inflight"] == state["policy"]["max_inflight"]
    assert direct_state["peak_inflight"] <= direct_state["policy"]["max_inflight"]
    direct_requests = [r for r in direct_state["requests"]
                       if r["context"]["variant_id"].startswith("baseline:")]
    assert all(r["status"] == "completed" and r["exception_type"] is None
               for r in direct_requests)
    if direct_root != root:
        # A reusable direct arm can come from a completed paired run.
        # Its historical system requests are not part of this run's API traffic.
        assert len(requests) + len(completion_requests) == len(system_requests)
    assert len(direct_requests) == experiment["n_questions"]
    direct_elapsed = {}
    baseline_ids = {"baseline:" + digest(qa)[:24]: qa for qa in expected}
    assert len(baseline_ids) == len(expected)
    for row in direct_requests:
        qa = baseline_ids[row["context"]["variant_id"]]
        assert qa not in direct_elapsed
        direct_elapsed[qa] = row["completed_at"] - row["started_at"]
    assert set(direct_elapsed) == expected
    assert len(system_requests) - len(failed_requests) <= experiment["comparison"]["system_request_cap"]
    health = read(root / "gateway_final_health.json")
    assert health["identity"] == read(root / "solve_health.json")["identity"]
    assert health["identity"] == read(direct_root / "direct_health.json")["identity"]
    upstream = health["upstream"]
    current_direct_requests = direct_requests if direct_root == root else []
    assert upstream["started"] == (
        len(system_requests) - len(completion_requests) + len(current_direct_requests))
    assert upstream["succeeded"] + upstream["failed"] == upstream["started"]
    assert upstream["failed"] == len(failed_requests)
    assert upstream["active"] == upstream["unknown"] == 0
    assert upstream["peak"] <= health["max_concurrency"] == state["policy"]["max_inflight"]
    completion_upstream = None
    if completion_root:
        completion_health = read(completion_root / "gateway_final_health.json")
        assert completion_health["identity"] == health["identity"]
        assert completion_health["identity"] == read(completion_root / "solve_health.json")["identity"]
        completion_upstream = completion_health["upstream"]
        assert completion_upstream["started"] == completion_upstream["succeeded"] == len(completion_requests)
        assert completion_upstream["active"] == completion_upstream["failed"] == completion_upstream["unknown"] == 0
        assert completion_upstream["peak"] <= completion_health["max_concurrency"]
    labels = load_label_metadata(root / "inputs/host_evaluation/metadata.jsonl")
    questions = {row["qa_id"]: row for row in read(root / "inputs/sample_questions.json")}
    pairs = []
    for qa in sorted(expected):
        s, d, label = system[qa], direct[qa], labels[qa]
        system_root = system_roots[qa]
        assert s["scoring_status"] in {"reference_unavailable", "incomplete"}
        assert s["score"] is None
        tool_faults = s["service_errors"]
        if tool_faults:
            assert s["status"] == "incomplete"
            assert all(error["tool"] == "detect_objects"
                       and "endpoint is not configured / prompt is empty / frame is None"
                       in error["error"] for error in tool_faults)
        assert d["reference_answer_visible"] is False and d["tool_calls"] == 0
        assert d["campaign_identity"] == direct_experiment["identity_sha256"]
        actual = d["actual_request_identity"]["parameters"]
        assert actual["enable_thinking"] is True
        assert actual["seed"] == 137 and actual["temperature"] == 0
        assert actual["max_tokens"] == 8192
        assert actual["model"] == "qwen3.8-flash"
        key = _variant_key(s["variant_id"])
        plan = read(system_root / "system/variants" / key / "plan.json")
        outcome = read(system_root / "system/variants" / key / "outcome.json")
        assert outcome["initial_skill_selection"]["evidence_state"] == expected_states[s["variant_id"]]
        assert outcome["scoring_status"] in {"reference_unavailable", "incomplete"}
        assert outcome["score"] is None and outcome["correct"] is None
        assert [error for row in outcome["rounds"]
                for error in row.get("service_errors", [])] == tool_faults
        tool_results = [
            {"round": row["index"], "tool": result["tool"],
             "status": result["status"], "error_code": result.get("error_code")}
            for row in outcome["rounds"] for result in (row.get("results") or [])]
        assert len(tool_results) == s["tool_call_count"]
        assert outcome["first_synthesis"]["prompt_compaction_version"] == COMPACTION_VERSION
        assert all(r["prompt_compaction_version"] == COMPACTION_VERSION for r in outcome["rounds"])
        first_images = outcome["first_synthesis"]["image_round"]
        source_jpegs = [sha for image_id, sha in zip(
            first_images["image_ids"], first_images["jpeg_sha256"])
            if image_id.startswith("frame-")]
        assert len(source_jpegs) == 32
        assert source_jpegs == [r["jpeg_sha256"]
                                for r in d["actual_request_identity"]["images_in_order"]]
        transport = plan["published_transport"]
        receipt = read(direct_root / "direct/preflight" / (digest(qa)[:24] + ".json"))["input_receipt"]
        for field in ("qa_id", "n_images", "frame_set_hash", "source_frame_indices",
                      "pixel_sha256", "source_catalog_sha256"):
            assert transport[field] == receipt[field]
        assert transport["frozen_scene"]["sha256"] == receipt["frozen_scene_sha256"]
        assert receipt["n_images"] == 32 and receipt["reference_answer_visible"] is False
        seed_rows = [read(p) for p in (system_root / "system/request_seeds").glob(key + "*.json")]
        input_rows = [r for r in seed_rows if "request_identity" in r]
        output_rows = [r for r in seed_rows if "request_identity_sha256" in r]
        assert len(input_rows) == len(output_rows) == s["model_request_count"]
        assert 1 <= len(input_rows) <= 7
        for row in input_rows:
            params = row["request_identity"]["parameters"]
            assert row["request_seed"] == 137 and row["retry_of"] is None
            assert params["enable_thinking"] is True and params["temperature"] == 0
            assert params["seed"] == 137 and params["max_tokens"] == 8192
            assert params["model"] == "qwen3.8-flash"
        for row in output_rows:
            assert row["status"] == "completed" and row["exception_type"] is None
            params = row["actual_request_identity"]["parameters"]
            assert params["enable_thinking"] is True and params["temperature"] == 0
            assert params["seed"] == 137 and params["max_tokens"] == 8192
            assert params["model"] == "qwen3.8-flash"
        scores = {}
        for arm, row in (("system", s), ("direct", d)):
            scores[arm] = score_answer(
                answer=row["answer"], task=label["question_type"], options=label["options"],
                reference=label["reference_answer"],
                available=row["status"] == "completed" and row["final_state"] == "answer")
        pairs.append({
            "qa_id": qa, "question_type": label["question_type"], "question": questions[qa]["question"],
            "original_question_type": questions[qa]["question_type"],
            "scene_id": questions[qa]["scene_name"], "options": questions[qa]["options"],
            "system_answer": s["answer"], "direct_answer": d["answer"],
            "reference_answer": label["reference_answer"],
            "system_score": scores["system"]["score"], "direct_score": scores["direct"]["score"],
            "delta": scores["system"]["score"] - scores["direct"]["score"],
            "system_evaluation": scores["system"], "direct_evaluation": scores["direct"],
            "same_frozen_rgb_identity": True, "system_model_requests": s["model_request_count"],
            "system_tool_calls": s["tool_call_count"], "system_rounds": s["agent_rounds"],
            "system_elapsed_s": s["elapsed_s"], "system_answer_source": s["answer_source"],
            "direct_request_elapsed_s": direct_elapsed[qa],
            "system_tool_results": tool_results,
            "system_status": s["status"], "system_service_errors": tool_faults,
            "system_result_run_identity": read(system_root / "diagnostic.json")["identity_sha256"],
            "system_transport_completion": system_root != root,
            "system_scored_as_unavailable": s["status"] != "completed",
            "system_initial_skill_version": outcome["initial_skill_selection"]["skill_version"],
            "initial_prompt_characters": (len(outcome["m8_prompt"])
                                          if outcome.get("m8_prompt") else None),
        })
    groups = {}
    for task, n in experiment["task_counts"].items():
        rows = [r for r in pairs if r["question_type"] == task]
        assert len(rows) == n
        groups[task] = {
            "name": NAMES[task], "n": n, "metric": "MRA" if task == "object_counting" else "Accuracy",
            "system_score": statistics.mean(r["system_score"] for r in rows),
            "direct_score": statistics.mean(r["direct_score"] for r in rows),
            "delta": statistics.mean(r["delta"] for r in rows),
        }
    usage = {}
    for arm, rows in (("system", system_requests), ("direct", direct_requests)):
        counter = Counter()
        for row in rows:
            counter.update({k: v for k, v in row["usage"].items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)})
        usage[arm] = dict(counter)
    if storage_reconciliation:
        assert sum(p["system_model_requests"] for p in pairs) == (
            len(system_requests) - len(failed_requests)
            - storage_reconciliation["discarded_episode_successful_requests"])
    tool_counts = Counter()
    tool_statuses = Counter()
    tool_errors = Counter()
    for pair in pairs:
        for call in pair["system_tool_results"]:
            tool_counts[call["tool"]] += 1
            tool_statuses[call["status"]] += 1
            if call["error_code"]:
                tool_errors[call["error_code"]] += 1
    difficulty = {}
    for original_task in sorted({r["original_question_type"] for r in pairs}):
        rows = [r for r in pairs if r["original_question_type"] == original_task]
        difficulty[original_task] = {
            "n": len(rows), "system_correct": sum(r["system_score"] == 1 for r in rows),
            "direct_correct": sum(r["direct_score"] == 1 for r in rows),
            "system_score": statistics.mean(r["system_score"] for r in rows),
            "direct_score": statistics.mean(r["direct_score"] for r in rows)}
    summary = {
        "status": ("complete_with_tool_faults"
                   if any(r["system_service_errors"] for r in pairs) else "complete"),
        "n_pairs": len(pairs),
        "system_tool_fault_questions": [
            r["qa_id"] for r in pairs if r["system_service_errors"]],
        "system_score": statistics.mean(r["system_score"] for r in pairs),
        "direct_score": statistics.mean(r["direct_score"] for r in pairs),
        "delta": statistics.mean(r["delta"] for r in pairs), "by_task": groups,
        "improved": sum(r["delta"] > 0 for r in pairs), "decreased": sum(r["delta"] < 0 for r in pairs),
        "tied": sum(r["delta"] == 0 for r in pairs),
        "system_requests": len(system_requests), "direct_requests": len(direct_requests),
        "system_tool_calls": sum(r["system_tool_calls"] for r in pairs), "token_usage": usage,
        "known_total_token_ratio": (
            usage["system"]["total_tokens"] / usage["direct"]["total_tokens"]),
        "system_questions_with_tool_calls": sum(r["system_tool_calls"] > 0 for r in pairs),
        "system_tool_counts": dict(tool_counts), "system_tool_result_statuses": dict(tool_statuses),
        "system_tool_error_codes": dict(tool_errors), "by_original_question_type": difficulty,
        "system_episode_median_elapsed_s": statistics.median(r["system_elapsed_s"] for r in pairs),
        "direct_request_median_elapsed_s": statistics.median(direct_elapsed.values()),
        "configured_api_concurrency": state["policy"]["max_inflight"],
        "observed_request_peak": max(
            state["peak_inflight"], direct_state["peak_inflight"] if direct_root == root else 0),
        "reused_direct_source_request_peak": (
            direct_state["peak_inflight"] if direct_root != root else None),
        "observed_upstream_peak": upstream["peak"],
        "both_enable_thinking": True, "seed": 137, "temperature": 0,
        "system_scene_overview": True, "direct_scene_overview": False,
        "diagnostic_identity": experiment["identity_sha256"],
        "direct_diagnostic_identity": direct_experiment["identity_sha256"],
        "direct_results_reused": direct_root != root,
        "system_evidence_states": sorted({v["evidence_state"] for v in experiment["variants"]}),
        "system_initial_skill_versions": sorted({
            r["system_initial_skill_version"] for r in pairs}),
        "direct_source_evidence_states": sorted({
            v["evidence_state"] for v in direct_experiment["variants"]}),
    }
    if completion_root:
        summary["status"] = ("complete_after_storage_interruption" if storage_reconciliation
                             else "complete_after_transport_interruption")
        summary[recovery_key] = {
            **completion_experiment[recovery_key],
            "completion_identity": completion_experiment["identity_sha256"],
            "completion_model_requests": len(completion_requests),
            "failed_model_requests": len(failed_requests),
            "failed_request_token_usage_unknown": upstream["usage_unknown"],
            "known_usage_excludes_failed_request": True,
        }
        if storage_reconciliation:
            discarded_usage = Counter()
            missing_variants = set(completion_experiment[recovery_key]["missing_variant_ids"])
            for row in storage_reconciliation["requests"]:
                if row["context"]["variant_id"] in missing_variants and row["status"] == "completed":
                    discarded_usage.update(row["usage"])
            final_episode_usage = Counter(usage["system"])
            final_episode_usage.subtract(discarded_usage)
            summary["completed_episode_known_token_usage"] = dict(final_episode_usage)
            summary[recovery_key].update(
                discarded_episode_successful_requests=storage_reconciliation[
                    "discarded_episode_successful_requests"],
                discarded_episode_known_token_usage=dict(discarded_usage),
                discarded_episode_known_tokens_included=True,
                reconciled_requests_receipt_sha256=hashlib.sha256(
                    (root / "reconciled_requests_receipt.json").read_bytes()).hexdigest())
        summary["observed_request_peak"] = max(
            summary["observed_request_peak"], completion_state["peak_inflight"])
        summary["observed_upstream_peak"] = max(
            summary["observed_upstream_peak"], completion_upstream["peak"])
    result = {"summary": summary, "pairs": pairs}
    (root / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report_data = args.report.with_suffix(".json")
    report_data.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# VSI " + "、".join(g["name"] for g in groups.values()) + "：工具组与 Qwen 直答", "",
        f"共 {len(pairs)} 道题。工具臂 {summary['system_score']:.2%}，"
        f"直答 {summary['direct_score']:.2%}，差值 {summary['delta'] * 100:+.2f} 个百分点。", "",
        "| 题型 | 数量 | 指标 | 精简工具臂 | Qwen 直答 | 差值 |",
        "|---|---:|---|---:|---:|---:|"]
    for task in groups:
        g = groups[task]
        lines.append(f"| {g['name']} | {g['n']} | {g['metric']} | {g['system_score']:.1%} | "
                     f"{g['direct_score']:.1%} | {g['delta'] * 100:+.1f} pp |")
    if len(difficulty) > len(groups):
        lines.extend(["", "| 原始难度 | 数量 | 工具正确 | 直答正确 |",
                      "|---|---:|---:|---:|"])
        for task, group in difficulty.items():
            lines.append(f"| {task.rsplit('_', 1)[-1]} | {group['n']} | "
                         f"{group['system_correct']}/{group['n']} | "
                         f"{group['direct_correct']}/{group['n']} |")
    sampling_description = (
        f"沿用原冻结样本的相同 {len(pairs)} 题。" if experiment.get("prior_sample_sha256") else
        f"每类从固定测试集按 seed 137 随机抽取 {experiment['questions_per_type']} 题。")
    if experiment.get("excluded_qa_ids"):
        sampling_description += (
            f"抽样前排除上轮 {len(experiment['excluded_qa_ids'])} 道题，本轮样本与其不重叠。")
    lines.extend([
        "", sampling_description
        + "两臂使用相同题目和原始 32 帧；"
        "均为 qwen3.8-flash、thinking=true、temperature=0、max_tokens=8192。"
        "工具臂使用精简提示词、完整冻结 Skill 和场景总览；直答每题一次原始 RGB 请求。"
        "工具臂沿用最多 6 个求解轮及原配置的一次视觉 fallback。参考答案仅供离线评分。", "",
        f"工具组实际证据状态：{', '.join(summary['system_evidence_states'])}；"
        f"起始 Skill：{', '.join(summary['system_initial_skill_versions'])}。", "",
        (f"物体计数使用官方 MRA；总分为 {len(pairs)} 题均值。" if set(groups) == {"object_counting"} else
         "；".join(f"{g['name']}使用 {g['metric']}" for g in groups.values())
         + "；总分为全部配对题的平均得分。"), "",
        f"相对于 Qwen 直答，逐题提高 {summary['improved']}，下降 {summary['decreased']}，持平 {summary['tied']}。"
        f"工具臂 {summary['system_requests']} 次模型请求、{summary['system_tool_calls']} 次工具调用；"
        f"直答 {summary['direct_requests']} 次模型请求。", "",
        f"API 并发上限 {summary['configured_api_concurrency']}，实际调度峰值 "
        f"{summary['observed_request_peak']}，上游实际峰值 {summary['observed_upstream_peak']}。"
        f"已知总 Token 用量为直答的 {summary['known_total_token_ratio']:.2f} 倍。", "",
        "| 组别 | 输入 Token | 输出 Token（服务报告） | 总 Token |",
        "|---|---:|---:|---:|",
        f"| 工具组 | {usage['system']['prompt_tokens']:,} | "
        f"{usage['system']['completion_tokens']:,} | {usage['system']['total_tokens']:,} |",
        f"| 直答 | {usage['direct']['prompt_tokens']:,} | "
        f"{usage['direct']['completion_tokens']:,} | {usage['direct']['total_tokens']:,} |", "",
        f"{summary['system_questions_with_tool_calls']}/{len(pairs)} 题主动调用过工具。"
        f"调用明细：{json.dumps(dict(tool_counts), ensure_ascii=False)}；"
        f"返回状态：{json.dumps(dict(tool_statuses), ensure_ascii=False)}；"
        f"错误码：{json.dumps(dict(tool_errors), ensure_ascii=False)}。"
        "调用数包含被前置条件拒绝的尝试；ok 只表示工具正常返回，不代表语义已验证。", "",
        f"完成题的工具 episode 耗时中位数 {summary['system_episode_median_elapsed_s']:.1f} 秒，"
        f"直答模型请求耗时中位数 {summary['direct_request_median_elapsed_s']:.1f} 秒。"
        "工具耗时包含该次 episode 的全部模型轮次与工具执行，补跑题采用补跑耗时，"
        "原中断的等待另计。", "",
        f"逐题答案及分数见 [{report_data.name}]({report_data.name})。"
        f"冻结实验身份：`{experiment['identity_sha256']}`。"])
    if direct_root != root:
        lines.extend(["", f"直答结果复用自 `{direct_root.name}`（{len(direct_requests)} 题全部完成，未重复调用）。"
                      f"直答身份：`{direct_experiment['identity_sha256']}`。"])
    if completion_root:
        recovered = completion_experiment[recovery_key]
        if storage_reconciliation:
            lines.extend([
                "", f"运行中断与补跑：原批次有 17 题完整结果；QA2425 请求超时，"
                "QA1558 和 QA3588 因 ENOSPC 未保存完整 episode。"
                "原日志、请求状态和未更新的 summary 均保留；没有正常的调度结束回执。"
                "缺失题目仅按结果是否完整选择，没有使用参考答案。",
                "网关数据库的只读备份、原始请求种子和请求正文哈希逐条核对，"
                "确认 35 次成功、1 次超时；旁路账本补齐两条未落盘的请求终态，"
                "没有改写原请求记录。两道存储故障题在原批次中的 4 次成功调用 "
                "Token 全部计入本轮成本，包括 QA1558 在未落盘的一轮前已完成的两轮。",
                f"仅缺结果的 3 题在独立批次补跑，等待上限由 "
                f"{recovered['previous_timeout_s']:g} 秒延长至 {recovered['new_timeout_s']:g} 秒，"
                "模型、thinking、帧、Skill 和求解预算保持一致；17 道已完成题和全部直答题未重跑。",
                f"请求总数包含 {len(failed_requests)} 次原失败请求，其 Token 未由服务返回。"
                f"补跑身份：`{completion_experiment['identity_sha256']}`。"])
        else:
            lines.extend([
            "", f"传输中断与补跑：原工具组的 {', '.join(recovered['failed_variant_ids'])} "
            f"请求在约 {recovered['previous_timeout_s']:g} 秒等待后失败，原失败记录完整保留。"
            f"仅这些未产出结果的题在独立批次补跑，请求等待上限延长至 "
            f"{recovered['new_timeout_s']:g} 秒，模型、thinking、帧、Skill 和求解预算保持一致。"
            "原来已完成的工具题和全部直答题没有重跑。",
            f"请求总数包含 {len(failed_requests)} 次原失败请求；其 Token 未由服务返回，"
            "上述 Token 是已返回的已知用量。"
                f"补跑身份：`{completion_experiment['identity_sha256']}`。"])
    if summary["system_tool_fault_questions"]:
        lines.extend([
            "", "工具运行故障："
            + "、".join(summary["system_tool_fault_questions"])
            + " 的 detect_objects 返回检测服务地址未配置或输入为空的错误。"
            "这些题保留在配对统计中，沿用运行器的 incomplete 状态按 0 分计；"
            "逐题 JSON 保留原始错误、最终答案和状态。上游模型请求全部完成。"])
    lines.extend([
        "", "| QA | 题型 | 工具答案 | 直答 | 参考答案 | 工具得分 | 直答得分 | 工具状态 |",
        "|---|---|---|---|---|---:|---:|---|"])
    for row in pairs:
        answers = [str(row[key]).replace("|", "\\|").replace("\n", " ")
                   for key in ("system_answer", "direct_answer", "reference_answer")]
        lines.append(f"| {row['qa_id']} | {NAMES[row['question_type']]} | "
                     f"{answers[0]} | {answers[1]} | {answers[2]} | "
                     f"{row['system_score']:.1%} | {row['direct_score']:.1%} | "
                     f"{row['system_status']} |")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
