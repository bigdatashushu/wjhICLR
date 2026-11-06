#!/usr/bin/env python3
"""BLINK SR: offline check by default, explicit preparation/execution, host scoring."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sr_data import (
    DEFAULT_ROOT, ROOT, download, file_sha, input_summary, load_inputs, load_references,
)

from skill3d.evaluation.blink_experiment import (
    OPERATIONAL, RecordedClient, read, write_once,
)
from skill3d.small_image.blink_protocol import (
    explicit_answer, image_identities, original_messages, response_record, sha_value, strict_json,
)
from sr_adapter import (
    SRProblem as SmallImageProblem, SRSolver as SmallImageSolver,
    solver_config, FullRecordedClient, FullResponseClient,
)
from skill3d.small_image.precompute import SmallImagePrecomputer
from skill3d.small_image.runner import (
    SmallImageRunnerConfig, _validate_prepared,
    load_problem_frames, validate_request_config,
)
from sr_protocol import DIRECT_SUFFIX, TOOLS, VERSION, relation_for, build_prompt
from skill3d.small_image.runner import _interfaces
from skill3d.small_image.storage import load_prepared, save_prepared
from skill3d.small_image.tools import build_registry
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig

SCHEMA = "harness3d-blink-sr/1"
CONFIG = Path(__file__).with_name("sr_config.json")
SKILL = Path(__file__).with_name("sr_skill.txt")


def configuration(path=CONFIG):
    value = strict_json(Path(path).read_text(encoding="utf-8"))
    if set(value) != {"schema", "model", "solver"} or value["schema"] != SCHEMA:
        raise ValueError("invalid_sr_config")
    if set(value["model"]) != {
            "model", "base_url", "api_key_env", "timeout_s", "temperature", "max_output_tokens"}:
        raise ValueError("sr_model_config_contains_unexpected_fields")
    config = solver_config(value["solver"])
    client = BailianClient(**value["model"])
    validate_request_config(client, config)
    if (config.max_requests != 3
            or config.max_solve_rounds != 2 or config.seed != 137
            or config.max_tokens != 8192 or client.temperature != 0
            or client.model != "qwen3.8-flash" or config.expected_model != client.model
            or not all((config.enable_thinking, config.original_image_bytes,
                        config.explicit_final_answer, config.capture_responses))):
        raise ValueError("invalid_sr_request_protocol")
    return value, config, client


def source_identity():
    paths = [
        *sorted((ROOT / "src" / "skill3d").rglob("*.py")),
        *sorted(Path(__file__).parent.glob("sr*")),
        ROOT / "scripts" / "run_small_image.py", ROOT / "BLINK" / "download_data.py",
    ]
    return {str(path.relative_to(ROOT)): file_sha(path) for path in paths}


def scoped_registry(prepared, problem):
    registry, scene = build_registry(prepared, problem)
    registry._model_tools = TOOLS
    # Shared names_for_scope iterates registered names, not model_names.
    names_for_scope = registry.names_for_scope
    registry.names_for_scope = lambda *args, **kwargs: [
        name for name in names_for_scope(*args, **kwargs) if name in TOOLS]
    return registry, scene


def prepared_for(problem, root):
    prepared = load_prepared(Path(root) / problem.input_sha256, problem=problem)
    if _validate_prepared(prepared, problem, load_problem_frames(problem)):
        raise ValueError("sr_prepared_cache_mismatch")
    if not prepared.has_relative_depth:
        raise ValueError("sr_skill_tools_requires_precomputed_relative_depth")
    return prepared


def check(data_root, splits, config_path=CONFIG, prepared_root=None, sample_ids=None):
    """Validate all original inputs and initial payloads without opening answers."""
    manifest, items = load_inputs(data_root, splits)
    if sample_ids is not None:
        by_id = {p.sample_id: (p, prompt) for p, prompt in items}
        if len(set(sample_ids)) != len(sample_ids) or set(sample_ids) - set(by_id):
            raise ValueError("sr_selected_input_identity_mismatch")
        items = [by_id[sample_id] for sample_id in sample_ids]
    return inspect_inputs(manifest, items, config_path, prepared_root)


def inspect_inputs(manifest, items, config_path=CONFIG, prepared_root=None):
    """Inspect public problems while retaining their frozen absolute image paths."""
    profile, config, client = configuration(config_path)
    skill = SKILL.read_text(encoding="utf-8")
    raw_precomputer = SmallImagePrecomputer()
    identities = []
    for problem, prompt in items:
        prepared = (prepared_for(problem, prepared_root) if prepared_root is not None
                    else raw_precomputer.process(problem, categories=()))
        registry, scene = scoped_registry(prepared, problem)
        allowed = set(registry.names_for_scope(
            scene.question_tool_scope, scene.available_artifacts,
            evidence_profile=scene.evidence_profile,
            question_type=scene.question_type, local_facts=scene.local_facts)) & TOOLS
        solver_prompt = build_prompt(
            problem, prepared.frames, prompt=prompt, skill_text=skill,
            interfaces=_interfaces(registry, allowed), feedback=())
        kwargs = {"max_tokens": config.max_tokens, "seed": config.seed,
                  "temperature": 0.0, "enable_thinking": config.enable_thinking}
        direct_payload = client.build_payload(original_messages(problem, prompt + DIRECT_SUFFIX), **kwargs)
        skill_payload = client.build_payload(original_messages(problem, solver_prompt), **kwargs)
        identities.append({
            "problem": problem.public_dict(), "official_prompt": prompt,
            "images": image_identities(problem), "relation": relation_for(problem.question),
            "direct_payload_sha256": sha_value(direct_payload),
            "skill_initial_payload_sha256": sha_value(skill_payload),
            "prepared_artifact_sha256": prepared.artifact_sha256 if prepared_root else None,
            "available_tools": sorted(allowed),
        })
    return {
        "schema": SCHEMA, "protocol": VERSION,
        **input_summary(manifest, items), "configuration": profile,
        "sources": source_identity(), "skill_sha256": file_sha(SKILL),
        "endpoint_sha256": client.endpoint_hash, "automatic_retry": False,
        "skill_tools_ready": prepared_root is not None,
        "prepared_root": str(Path(prepared_root).resolve()) if prepared_root else None,
        "inputs": identities,
    }


def precompute(data_root, splits, prepared_root, backend_config):
    """Explicit native preparation using existing DINO/SAM2/Intern/MoGe adapters."""
    import yaml
    from sr_native import build_precomputer
    _, items = load_inputs(data_root, splits)
    profile = yaml.safe_load(Path(backend_config).read_text(encoding="utf-8"))
    precomputer = build_precomputer(profile["precompute"], seed=137)
    for problem, _ in items:
        target = Path(prepared_root) / problem.input_sha256
        if target.exists():
            prepared_for(problem, prepared_root)
            continue
        relation = relation_for(problem.question)
        prepared = precomputer.process(
            problem, categories=tuple(dict.fromkeys((relation["subject"], relation["target"]))))
        if prepared.errors or not prepared.has_relative_depth:
            raise ValueError("sr_native_precompute_failed")
        save_prepared(prepared, target)
        prepared_for(problem, prepared_root)


def solve_one(spec, client):
    """Direct receives only public input; geometry is opened only by skill_tools."""
    public = spec["problem"]
    problem = SmallImageProblem(**{
        k: tuple(v) if k in {"options", "image_paths"} else v
        for k, v in public.items() if k != "input_sha256"})
    if problem.input_sha256 != public["input_sha256"]:
        raise ValueError("sr_worker_input_changed")
    config = solver_config(spec["solver"])
    if spec["arm"] == "direct":
        if set(spec) != {"arm", "problem", "solver", "official_prompt"}:
            raise ValueError("sr_direct_received_extra_evidence")
        kwargs = {"max_tokens": config.max_tokens, "seed": config.seed,
                  "temperature": 0.0, "enable_thinking": config.enable_thinking}
        text = client.chat(original_messages(problem, spec["official_prompt"] + DIRECT_SUFFIX), **kwargs)
        row = {"phase": "direct", "request_parameters": kwargs, **response_record(client, text)}
        actual = row["response_identity"]
        if actual.get("model") != config.expected_model or actual.get("model_reported") is not True:
            raise ValueError("sr_response_model_mismatch")
        try:
            answer, method = explicit_answer(text, problem.options)
            row.update(status="answered", answer_parse_method=method)
        except ValueError:
            answer = None
            row["status"] = "invalid_answer"
        return {
            **public, "status": row["status"], "answer": answer, "rounds": [row],
            "request_count": 1, "attempted_tool_calls": 0, "succeeded_tool_calls": 0,
            "reference_answer_visible": False, "automatic_retry": False,
        }
    if spec["arm"] != "skill_tools":
        raise ValueError("invalid_sr_arm")
    prepared = prepared_for(problem, spec["prepared_root"])
    if prepared.artifact_sha256 != spec["artifact_sha256"]:
        raise ValueError("sr_worker_prepared_changed")
    solver = SmallImageSolver(client, config=config, registry_builder=scoped_registry,
                              skill_text=spec["skill_text"],
                              official_prompt=spec["official_prompt"])
    result = solver.solve(problem, prepared=prepared, categories=prepared.categories)
    return {
        **result.to_dict(), "execution_trace": solver.execution_trace,
        "reference_answer_visible": False, "automatic_retry": False,
    }


def run(data_root, splits, config_path, prepared_root, run_root, arms):
    run_root = Path(run_root).resolve()
    inspection = check(data_root, splits, config_path, prepared_root)
    if "skill_tools" in arms and not inspection["skill_tools_ready"]:
        raise ValueError("sr_tool_arm_needs_a_verified_prepared_root")
    profile = inspection["configuration"]
    client_env = profile["model"]["api_key_env"]
    if not os.environ.get(client_env, "").strip():
        raise ValueError("sr_api_credential_must_be_in_environment")
    skill = SKILL.read_text(encoding="utf-8")
    jobs = []
    for index, item in enumerate(inspection["inputs"]):
        for arm in arms:
            spec = {"arm": arm, "problem": item["problem"], "solver": profile["solver"],
                    "official_prompt": item["official_prompt"]}
            if arm == "skill_tools":
                spec.update(prepared_root=inspection["prepared_root"], skill_text=skill,
                            artifact_sha256=item["prepared_artifact_sha256"])
            jobs.append({"operation_id": f"sr-{index:04d}-{arm}", "spec": spec})
    cap = len(inspection["inputs"]) * sum(1 if a == "direct" else 3 for a in arms)
    frozen = write_once(run_root / "manifest.json", {
        "schema": SCHEMA, "data_root": str(Path(data_root).resolve()),
        "inspection": inspection, "arms": list(arms), "request_cap": cap, "jobs": jobs,
        "reference_answer_visible": False, "automatic_retry": False,
    })
    with (run_root / "active.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("sr_run_already_active") from None
        governor = RequestGovernor(run_root / "budget", RequestGovernorConfig(cap, 1))
        summary = governor.summary()
        if summary["error_requests"] or summary["unknown_requests"] or summary["active_requests"]:
            raise ValueError("sr_incomplete_or_failed_request_retry_forbidden")
        for job in jobs:
            if source_identity() != inspection["sources"] or file_sha(SKILL) != inspection["skill_sha256"]:
                raise ValueError("sr_source_or_skill_changed_during_run")
            path = run_root / "episodes" / (job["operation_id"] + ".json")
            if path.exists():
                receipt = read(path)
                result = receipt["result"]
                if (receipt["job_sha256"] != sha_value(job)
                        or result["input_sha256"] != job["spec"]["problem"]["input_sha256"]
                        or result["sample_id"] != job["spec"]["problem"]["sample_id"]
                        or result.get("reference_answer_visible") is not False):
                    raise ValueError("sr_completed_episode_identity_changed")
                if result["status"] in OPERATIONAL:
                    raise ValueError("sr_failed_episode_retry_forbidden")
                continue
            request_dir = run_root / "requests" / job["operation_id"]
            if request_dir.exists() and any(request_dir.iterdir()):
                raise ValueError("sr_unfinished_episode_retry_forbidden")
            base = FullResponseClient(**profile["model"])
            client = FullRecordedClient(base, governor, run_root, job["operation_id"], "solver")
            result = solve_one(job["spec"], client)
            result["request_trace"] = client.request_trace
            write_once(path, {"job_sha256": sha_value(job), "result": result})
            if result["status"] in OPERATIONAL:
                raise ValueError("sr_operational_failure_stopped")
        budget = governor.summary()
        if budget["error_requests"] or budget["unknown_requests"] or budget["active_requests"]:
            raise ValueError("sr_request_audit_failed")
        return write_once(run_root / "completed.json", {
            "manifest_sha256": frozen["content_sha256"], "episodes": len(jobs), "budget": budget,
            "reference_answers_read": False, "model_requests": budget["reserved_requests"],
        })


def score(run_root):
    """Read reference labels only after all solving is finished and locked out."""
    run_root = Path(run_root).resolve()
    with (run_root / "active.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("sr_cannot_score_during_solving") from None
        manifest = read(run_root / "manifest.json")
        completed = read(run_root / "completed.json")
        if completed["manifest_sha256"] != manifest["content_sha256"]:
            raise ValueError("sr_completion_identity_mismatch")
        data, items = load_inputs(manifest["data_root"], manifest["inspection"]["splits"])
        if data["input_index_sha256"] != manifest["inspection"]["input_index_sha256"]:
            raise ValueError("sr_scoring_input_changed")
        by_id = {p.sample_id: p for p, _ in items}
        receipts = []
        for job in manifest["jobs"]:
            receipt = read(run_root / "episodes" / (job["operation_id"] + ".json"))
            result = receipt["result"]
            public = job["spec"]["problem"]
            if (receipt["job_sha256"] != sha_value(job)
                    or result["sample_id"] != public["sample_id"]
                    or result["input_sha256"] != public["input_sha256"]
                    or result["source_split"] != public["source_split"]
                    or result.get("reference_answer_visible") is not False):
                raise ValueError("sr_scoring_receipt_identity_mismatch")
            receipts.append((job["spec"]["arm"], result))
        expected = {(arm, p.sample_id) for arm in manifest["arms"] for p, _ in items}
        if (len(receipts) != len(expected)
                or {(a, r["sample_id"]) for a, r in receipts} != expected):
            raise ValueError("sr_scoring_denominator_mismatch")
        # Every receipt is verified before opening host answers.
        references = load_references(manifest["data_root"], data)
        groups = {}
        for arm in manifest["arms"]:
            for split in manifest["inspection"]["splits"]:
                selected = [r for a, r in receipts if a == arm and r["source_split"] == split]
                correct = scored = legal = 0
                family_totals, family_correct = Counter(), Counter()
                for result in selected:
                    legal += result["status"] == "answered" and result["answer"] in {"A", "B"}
                    reference = references[result["sample_id"]]
                    if reference is None:
                        continue
                    family = relation_for(by_id[result["sample_id"]].question)["family"]
                    scored += 1
                    hit = result["status"] == "answered" and result["answer"] == reference
                    correct += hit
                    family_totals[family] += 1
                    family_correct[family] += hit
                groups[f"{arm}/{split}"] = {
                    "questions": len(selected), "legal_answers": legal, "scored": scored,
                    "correct": correct if scored else None,
                    "accuracy": correct / scored if scored else None,
                    "scoring_status": "scored" if scored else "official_test_labels_hidden",
                    "families": {f: {"questions": n, "correct": family_correct[f],
                                     "accuracy": family_correct[f] / n}
                                 for f, n in sorted(family_totals.items())},
                }
        return write_once(run_root / "scores.json", {
            "schema": SCHEMA, "manifest_sha256": manifest["content_sha256"], "groups": groups,
            "reference_access": "host scoring after completed solving",
        })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--download", action="store_true")
    action.add_argument("--precompute", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--score", action="store_true")
    parser.add_argument("--check", action="store_true", help="default; no requests or reference reads")
    parser.add_argument("--execute", action="store_true", help="enable precompute or model execution")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--split", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--backend-config", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--arm", nargs="+", choices=["direct", "skill_tools"],
                        default=["direct", "skill_tools"])
    parser.add_argument("--output", type=Path, help="optional offline check JSON file")
    args = parser.parse_args(argv)
    if args.check and any((args.download, args.precompute, args.run, args.score)):
        parser.error("--check cannot be combined with another action")
    if args.execute and not (args.precompute or args.run):
        parser.error("--execute requires --precompute or --run")
    if args.precompute or args.run:
        if not args.execute:
            parser.error("precompute/run requires explicit --execute")
    if (args.run or args.score) and args.run_root is None:
        parser.error("--run-root is required")
    if args.precompute and (args.prepared_root is None or args.backend_config is None):
        parser.error("precompute requires --prepared-root and --backend-config")
    if len(set(args.arm)) != len(args.arm):
        parser.error("duplicate arms")
    if args.download:
        value = download(args.data_root, args.revision)
    elif args.precompute:
        precompute(args.data_root, args.split, args.prepared_root, args.backend_config)
        value = check(args.data_root, args.split, args.config, args.prepared_root)
    elif args.run:
        value = run(args.data_root, args.split, args.config, args.prepared_root, args.run_root, args.arm)
    elif args.score:
        value = score(args.run_root)
    else:
        value = check(args.data_root, args.split, args.config, args.prepared_root)
    if args.output:
        write_once(args.output, value)
    print(json.dumps({k: v for k, v in value.items()
                      if k not in {"inputs", "jobs", "sources", "files"}},
                     ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "stopped", "error_type": type(exc).__name__}))
        raise SystemExit(1)
