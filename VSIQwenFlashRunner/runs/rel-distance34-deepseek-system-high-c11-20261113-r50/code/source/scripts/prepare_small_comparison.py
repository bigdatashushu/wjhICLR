#!/usr/bin/env python3
"""Freeze a paired held-out sample for specified VSI question types."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skill3d.infra.m5_identity import content_sha256
from skill3d.infra.vsi_runner_source import freeze_source
from skill3d.synthesis.xcode_client import get_xcode_client_class


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--native-root", required=True)
    parser.add_argument("--native-catalog", required=True)
    parser.add_argument("--perception-endpoint", required=True)
    parser.add_argument("--types", nargs="+", required=True)
    parser.add_argument("--per-type", type=int, default=10)
    parser.add_argument("--per-difficulty", type=int,
                        help="For relative direction, sample this many easy/medium/hard questions each.")
    parser.add_argument("--system-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--direct-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-provider", choices=("bailian", "xcodeapi"), default="bailian")
    parser.add_argument("--model", default="")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--max-output-tokens", type=int,
                        help="Explicit shared output budget, frozen before either arm starts.")
    parser.add_argument("--model-context-tokens", type=int,
                        help="Documented context capacity of the explicitly selected model.")
    parser.add_argument("--framework-context-tokens", type=int,
                        help="Shared framework context budget; does not claim the model's full capacity.")
    parser.add_argument("--system-effort", choices=("high", "medium", "low", "max"), default="high")
    parser.add_argument("--direct-effort", choices=("high", "medium", "low", "max"), default="low")
    parser.add_argument("--derived-replay", type=Path)
    parser.add_argument("--publish-sample-only", action="store_true",
                        help="Expose sampled QAs while retaining each complete source-scene audit scope.")
    parser.add_argument("--derivation-root", default="")
    parser.add_argument("--binding-config", type=Path)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--sample-from", type=Path,
                        help="Reuse the exact requested-type QA IDs from a prior sample.")
    parser.add_argument("--exclude-sample", type=Path, action="append", default=[],
                        help="Exclude QA IDs from a prior sample; may be repeated.")
    parser.add_argument("--detector-endpoint", default="",
                        help="Record and enable an existing native-compatible detector proxy.")
    parser.add_argument("--min-free-gib", type=float, default=80)
    parser.add_argument("--timeout-s", type=float, default=660,
                        help="Transport wait limit; does not change model output budgets.")
    parser.add_argument("--arms", nargs="+", choices=("system", "direct"),
                        default=("system", "direct"))
    parser.add_argument("--preflight-mode", choices=("required_zero_model_calls",
                                                   "per_episode_before_first_call"),
                        default="per_episode_before_first_call")
    args = parser.parse_args()
    root, package = args.run_root.resolve(), args.package.resolve()
    assert not root.exists() and root.parent.is_dir()
    assert args.per_type > 0 and len(set(args.types)) == len(args.types)
    if args.per_difficulty is not None:
        assert args.types == ["object_rel_direction"] and args.per_difficulty > 0
        assert args.per_type == 3 * args.per_difficulty
        assert args.sample_from is None
    assert 1 <= args.concurrency <= 64
    assert math.isfinite(args.min_free_gib) and args.min_free_gib >= 0
    assert math.isfinite(args.timeout_s) and 0 < args.timeout_s <= 1800
    assert args.max_output_tokens is None or 0 < args.max_output_tokens <= 393216
    assert args.model_context_tokens is None or 0 < args.model_context_tokens <= 1048576
    assert args.framework_context_tokens is None or 0 < args.framework_context_tokens <= 1048576
    assert args.model_context_tokens is None or args.framework_context_tokens is None
    assert len(set(args.arms)) == len(args.arms)
    assert not args.publish_sample_only or args.derived_replay is not None
    panel = json.loads((package / "panel.json").read_text())
    by_qa = {row["qa_id"]: row for row in panel["system_variants"]}
    tests = [json.loads(line) for line in (package / "test.jsonl").read_text().splitlines()]
    rng, selected = random.Random(137), []
    previous_sample = (json.loads(args.sample_from.read_text(encoding="utf-8"))
                       if args.sample_from else None)
    excluded_samples = [
        {
            "source": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "qa_ids": sorted({row["qa_id"] for row in json.loads(path.read_text(encoding="utf-8"))}),
        }
        for path in args.exclude_sample
    ]
    excluded_ids = {qa for sample in excluded_samples for qa in sample["qa_ids"]}
    for task in sorted(args.types):
        pool = sorted((row for row in tests
                       if by_qa[row["qa_id"]]["question_type"] == task
                       and row["qa_id"] not in excluded_ids),
                      key=lambda row: row["qa_id"])
        assert len(pool) >= args.per_type, f"insufficient held-out questions after exclusions: {task}"
        if previous_sample is None:
            if args.per_difficulty is None:
                selected.extend(rng.sample(pool, args.per_type))
            else:
                for difficulty in ("easy", "medium", "hard"):
                    stratum = [row for row in pool
                               if row["question_type"] == task + "_" + difficulty]
                    assert len(stratum) >= args.per_difficulty, difficulty
                    selected.extend(rng.sample(stratum, args.per_difficulty))
        else:
            fixed_ids = {row["qa_id"] for row in previous_sample
                         if by_qa[row["qa_id"]]["question_type"] == task}
            fixed = [row for row in pool if row["qa_id"] in fixed_ids]
            assert len(fixed) == len(fixed_ids) == args.per_type
            selected.extend(fixed)
    selected.sort(key=lambda row: (by_qa[row["qa_id"]]["question_type"], row["qa_id"]))
    ids = {row["qa_id"] for row in selected}
    n = len(ids)
    assert n == args.per_type * len(args.types)
    assert not ids & excluded_ids
    difficulty_counts = dict(sorted(Counter(row["question_type"] for row in selected).items()))
    if args.per_difficulty is not None:
        assert set(difficulty_counts.values()) == {args.per_difficulty}
        assert len(difficulty_counts) == 3
    assert all(row["experiment_split"] == "test" for row in selected)
    root.mkdir()
    inputs = root / "inputs"
    inputs.mkdir()
    for name in ("online.yaml", "split.json", "questions.jsonl", "train.jsonl", "test.jsonl"):
        shutil.copyfile(package / name, inputs / name)
    shutil.copytree(package / "host_evaluation", inputs / "host_evaluation")
    shutil.copytree(package / "scene_overviews", inputs / "scene_overviews")
    # Transport timeout is independent of model budgets and recorded before freezing.
    import yaml
    config_path = inputs / "online.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["online_model"]["timeout_s"] = args.timeout_s
    if args.max_output_tokens is not None:
        config["online_model"]["max_tokens"] = args.max_output_tokens
    if args.model_context_tokens is not None:
        config["online_model"]["max_model_len"] = args.model_context_tokens
    if args.framework_context_tokens is not None:
        config["online_model"]["max_model_len"] = args.framework_context_tokens
    assert config["online_model"]["max_tokens"] < config["online_model"]["max_model_len"]
    if args.model_provider == "xcodeapi":
        assert args.model and args.base_url
        if "system" in args.arms:
            assert args.derived_replay and args.derivation_root and args.binding_config
        config["online_model"].update(
            provider="xcodeapi", model=args.model, base_url=args.base_url,
            model_revision=args.model_revision,
            api_key_env=("XCODE_CLAUDE_API_KEY" if args.model.startswith("claude-") else "XCODE_GPT_API_KEY"),
            reasoning_effort=args.system_effort)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    write(inputs / "sample_questions.json", selected)
    panel["system_variants"] = [row for row in panel["system_variants"] if row["qa_id"] in ids]
    panel["direct_qwen_rows"] = [row for row in panel["direct_qwen_rows"] if row["qa_id"] in ids]
    panel["experiment_id"] += "-" + root.name
    slots = sorted({row["question_type"] + "/" + row["evidence_state"]
                    for row in panel["system_variants"]})
    for value in panel["panels"].values():
        value["system_input_variants"] = n
        value["skill_slots"] = slots
        value["skill_slot_count"] = len(slots)
    panel["proposal_sha256"] = content_sha256({k: v for k, v in panel.items()
                                             if k != "proposal_sha256"})
    write(inputs / "panel.json", panel)
    published = json.loads((package / "published_inputs.json").read_text())
    original_locator = published["source_catalog_file"]
    published["source_catalog_file"] = args.native_catalog
    overview_scope_adaptation = None
    if args.derived_replay:
        scenes = {row["scene_name"] for row in selected}
        published["entries"] = {qa: entry for qa, entry in published["entries"].items()
                                if entry["scene_id"] in scenes
                                and (not args.publish_sample_only or qa in ids)}
        published["scene_scopes"] = {scene: value for scene, value in published["scene_scopes"].items()
                                    if scene in scenes}
        overview_path = inputs / "scene_overviews/manifest.json"
        original_overview = overview_path.read_bytes()
        overview = json.loads(original_overview)
        overview["images"] = [row for row in overview["images"] if row["scene_id"] in scenes]
        assert {row["scene_id"] for row in overview["images"]} == scenes
        overview["n_scenes"] = len(scenes)
        overview["n_questions"] = len(published["entries"])
        overview_path.parent.chmod(0o755)
        overview_path.chmod(0o644)
        (overview_path.parent / "source_manifest.json").write_bytes(original_overview)
        write(overview_path, overview)
        overview_scope_adaptation = {
            "source_manifest_sha256": hashlib.sha256(original_overview).hexdigest(),
            "selected_scene_ids": sorted(scenes),
            "source_rows_and_image_bytes_unchanged": True,
        }
        replay_bytes = args.derived_replay.read_bytes()
        replay = json.loads(replay_bytes)
        assert scenes <= set(replay["scenes"])
        replay["scenes"] = {scene: row for scene, row in replay["scenes"].items() if scene in scenes}
        (inputs / "source_derived_replay.json").write_bytes(replay_bytes)
        write(inputs / "derived_replay.json", replay)
        if args.binding_config:
            shutil.copyfile(args.binding_config, inputs / "binding_online.yaml")
    write(inputs / "published_inputs.json", published)
    runner = Path(__file__).resolve().parents[1]
    receipt = freeze_source(runner, root / "code")
    receipt["source_root"] = str(Path(args.native_root) / "code/source")
    write(root / "code/code_identity.json", receipt)
    shutil.copyfile(runner / "scripts/run_g1m1_diagnostic.py", root / "controller.py")
    shutil.copyfile(runner / "scripts/launch_small_comparison.py", root / "launcher.py")
    manifest = {
        "schema": "vsi-small-paired-comparison/1", "seed": 137, "role": "test",
        "selection": "random.Random(137), sorted requested types and lexically sorted held-out QA pools",
        "questions_per_type": args.per_type, "n_questions": n,
        "n_scenes": len({row["scene_name"] for row in selected}),
        "task_counts": dict(sorted(Counter(by_qa[q]["question_type"] for q in ids).items())),
        "original_question_type_counts": difficulty_counts,
        "variants": panel["system_variants"], "native_root": args.native_root,
        "runtime": {
            "workers": args.concurrency, "episode_concurrency": min(n, args.concurrency),
            "request_concurrency": args.concurrency, "direct_workers": min(n, args.concurrency),
            "max_requests": n * 8, "scene_cache_size": n, "min_free_gib": args.min_free_gib,
            "phase": "final_system", "automatic_model_retry": False, "skill_evolution": False,
            "enable_thinking": args.system_thinking,
            "direct_enable_thinking": args.direct_thinking,
            "max_solver_rounds": 6, "max_requests_per_episode": 7,
            "visual_fallback_max_requests": 1, "question_preflight": args.preflight_mode,
            "input_scheduler": "independent_scene_providers_v1",
            "perception_endpoint": args.perception_endpoint,
            "transport_timeout_s": args.timeout_s,
        },
        "comparison": {
            "arms": list(args.arms), "same_questions_and_rgb_bytes": True,
            "both_enable_thinking": args.system_thinking and args.direct_thinking,
            "system_enable_thinking": args.system_thinking,
            "direct_enable_thinking": args.direct_thinking,
            "direct_protocol": "prepared-32",
            "direct_rgb_only": True, "system_request_cap": n * 7, "direct_request_cap": n,
            "system_prompt": "vsi-prompt-compaction/1.0",
            "system_scene_overview": True, "direct_scene_overview": False,
        },
        "source_package": str(package),
        "source_panel_sha256": hashlib.sha256((package / "panel.json").read_bytes()).hexdigest(),
        "source_sha256": receipt["source_sha256"],
        "published_locator_adaptation": {
            "original_locator": original_locator, "native_locator": args.native_catalog,
            "catalog_sha256": published["source_catalog_sha256"],
            "entries_and_scopes_unchanged": not bool(args.derived_replay),
            "retained_scene_entries_and_scopes_unchanged": True,
        },
        "input_files": {str(p.relative_to(inputs)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(inputs.rglob("*")) if p.is_file()},
        "controller_sha256": hashlib.sha256((root / "controller.py").read_bytes()).hexdigest(),
        "launcher_sha256": hashlib.sha256((root / "launcher.py").read_bytes()).hexdigest(),
        "reference_answer_visible": False,
    }
    if args.sample_from:
        manifest["selection"] = "Exact requested-type QA IDs from a prior frozen sample"
        manifest["prior_sample_sha256"] = hashlib.sha256(args.sample_from.read_bytes()).hexdigest()
    if args.max_output_tokens is not None:
        manifest["comparison"]["shared_max_output_tokens"] = args.max_output_tokens
    if args.model_context_tokens is not None:
        manifest["comparison"]["documented_model_context_tokens"] = args.model_context_tokens
    if args.framework_context_tokens is not None:
        manifest["comparison"]["framework_context_tokens"] = args.framework_context_tokens
    if overview_scope_adaptation is not None:
        manifest["scene_overview_scope_adaptation"] = overview_scope_adaptation
    if args.publish_sample_only:
        manifest["published_qa_scope"] = {
            "mode": "sample_only", "qa_ids": sorted(ids),
            "complete_source_scene_audit_scopes_preserved": True}
    if args.model_provider == "xcodeapi":
        manifest["runtime"].update(system_reasoning_effort=args.system_effort,
                                    direct_reasoning_effort=args.direct_effort)
        manifest["comparison"].update(
            model=args.model, provider="xcodeapi",
            model_revision=args.model_revision,
            protocol=get_xcode_client_class(args.model).protocol,
            native_seed=None, framework_seed=137,
            system_reasoning_effort=args.system_effort, direct_reasoning_effort=args.direct_effort)
        if args.derived_replay:
            manifest["derived_replay"] = {"file": "derived_replay.json",
                                        "derivation_root": args.derivation_root,
                                        "binding_config": "binding_online.yaml",
                                        "source_replay_sha256": hashlib.sha256(
                                            args.derived_replay.read_bytes()).hexdigest(),
                                        "mode": "read_only_sealed_replay"}
    if args.per_difficulty is not None:
        manifest["questions_per_difficulty"] = args.per_difficulty
        manifest["selection"] = (
            "random.Random(137), easy/medium/hard strata with lexically sorted held-out QA pools")
    if excluded_samples:
        manifest["excluded_samples"] = excluded_samples
        manifest["excluded_qa_ids"] = sorted(excluded_ids)
        manifest["selection"] += "; prior sample QA IDs excluded before sampling"
    if args.detector_endpoint:
        manifest["runtime"]["detector_endpoint"] = args.detector_endpoint
    if args.types == ["object_counting"]:
        from skill3d.synthesis.counting_context import COUNTING_CONTEXT_VERSION
        manifest["comparison"]["system_prompt"] = COUNTING_CONTEXT_VERSION
    manifest["identity_sha256"] = content_sha256(manifest)
    write(root / "diagnostic.json", manifest)
    # Published input directories may be read-only; deployment needs to populate them.
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
    print(json.dumps({k: manifest[k] for k in (
        "n_questions", "n_scenes", "task_counts", "identity_sha256", "source_sha256")},
        sort_keys=True))


if __name__ == "__main__":
    main()
