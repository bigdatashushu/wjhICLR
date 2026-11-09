#!/usr/bin/env python3
"""VSI1000 runner: check, prepare Qwen8B M5 evidence, then use the original v11 engine."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def default_data_root():
    local = ROOT.parent.parent / "vsibench-m4-success-1000"
    return local if local.is_dir() else Path(
        "/data/wangy/wangjh/harness3d/experiment-inputs/vsibench-m4-success-1000-seed137")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_inputs(data_root, online_config):
    """Check the selected public subset without geometry, labels or model I/O."""
    data = Path(data_root).absolute()
    rows = [json.loads(line) for line in (data / "questions.jsonl").read_text().splitlines() if line]
    train = [json.loads(line) for line in (data / "train.jsonl").read_text().splitlines() if line]
    test = [json.loads(line) for line in (data / "test.jsonl").read_text().splitlines() if line]
    selected = {str(row["qa_id"]) for row in rows}
    train_ids = {str(row["qa_id"]) for row in train}
    test_ids = {str(row["qa_id"]) for row in test}
    if (len(rows) != 1000 or len(selected) != 1000 or len(train) != 300 or len(test) != 700
            or train_ids & test_ids or train_ids | test_ids != selected):
        raise ValueError("VSI1000 subset must preserve the original 300/700 split")
    forbidden = {"answer", "answers", "ground_truth", "reference_answer", "label", "score"}
    def contains_private(value):
        if isinstance(value, dict):
            return bool(forbidden & set(value)) or any(contains_private(v) for v in value.values())
        return isinstance(value, list) and any(contains_private(v) for v in value)
    if any(contains_private(row) for row in rows + train + test):
        raise ValueError("public questions contain private reference fields")
    scenes = {row["scene_name"] for row in rows}
    if (len(scenes) != 164 or {row["scene_name"] for row in train} != scenes
            or {row["scene_name"] for row in test} != scenes):
        raise ValueError("VSI1000 scene coverage changed")
    split = json.loads((data / "split.json").read_text())
    if (split["questions_sha256"] != file_hash(data / "questions.jsonl")
            or any(split["files"][role]["sha256"] != file_hash(data / f"{role}.jsonl")
                   for role in ("train", "test"))):
        raise ValueError("subset file hashes changed")
    manifest = json.loads((data / "published_inputs.json").read_text())
    if set(manifest["entries"]) != selected:
        raise ValueError("published input subset differs")
    from skill3d.online.config import load_config
    from skill3d.synthesis.eval_visual_fallback import ISOLATED_CHOICE_PROMPT_VERSION
    config = load_config(online_config)
    original = load_config(data / "online.yaml")
    changed = json.loads(json.dumps(config))
    changed["online_model"]["mask_review"].pop("target_description")
    changed["online_model"]["enable_thinking"] = original["online_model"]["enable_thinking"]
    changed["isolated_choice_visual_fallback"] = original["isolated_choice_visual_fallback"]
    # Existing thinking pilots use 900s transport timeout; keep the allowed delta exact.
    timeouts = (original["online_model"]["timeout_s"], config["online_model"]["timeout_s"])
    if timeouts[0] != timeouts[1] and timeouts != (180.0, 900.0):
        raise ValueError("runner solver timeout changes outside the existing 180s-to-900s pilot override")
    changed["online_model"]["timeout_s"] = original["online_model"]["timeout_s"]
    # Service location may be overridden, while the original reviewer pin is explicit.
    for field in ("endpoint", "expected_identity_sha256"):
        changed["online_model"]["mask_review"][field] = original["online_model"]["mask_review"][field]
    if changed != original:
        raise ValueError("runner changes runtime values outside description, thinking, or choice fallback")
    missing_remote_roots = sorted({entry["input_root"] for entry in manifest["entries"].values()
                                   if not Path(entry["input_root"]).is_dir()})
    return {
        "status": "complete", "questions": len(rows), "train": len(train), "test": len(test),
        "scenes": len(scenes), "model_requests": 0,
        "audit_questions": sum(len(scope["qa_ids"]) for scope in manifest["scene_scopes"].values()),
        "remote_scene_inputs_available": not missing_remote_roots,
        "description_model": config["online_model"]["mask_review"]["target_description"]["model"],
        "reviewer": "Intern-Decision-4B", "reviewer_yes_threshold":
            config["online_model"]["mask_review"]["yes_threshold"],
        "solver_enable_thinking": config["online_model"]["enable_thinking"],
        "solver_timeout_s": config["online_model"]["timeout_s"],
        "original_solver_timeout_s": original["online_model"]["timeout_s"],
        "isolated_choice_visual_fallback": config["isolated_choice_visual_fallback"],
        "isolated_choice_prompt_version": ISOLATED_CHOICE_PROMPT_VERSION,
        "eval_visual_fallback_max_requests": int(
            config["eval_visual_fallback"] or config["isolated_choice_visual_fallback"]),
        "public_questions_sha256": file_hash(data / "questions.jsonl"),
        "published_inputs_sha256": file_hash(data / "published_inputs.json"),
    }


def load_fitset_cli():
    spec = importlib.util.spec_from_file_location("_vsi8b_fitset_cli", ROOT / "scripts/run_fitset_v11.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def campaign(args):
    from skill3d.infra.experiment_campaign import main
    profile = json.loads((ROOT / "configs/experiments/vsi8b.json").read_text())
    profile["inputs"]["derived_binding_root"] = str(Path(args.cache_root).absolute())
    if Path(args.online_config).absolute() != ROOT / "runner.yaml":
        profile["inputs"]["online_config"] = str(Path(args.online_config).absolute())
    argv = ["--dataset", "vsi-bench", "--data-root", str(Path(args.data_root).absolute())]
    if args.output_dir:
        argv += ["--output-dir", str(Path(args.output_dir).absolute())]
    if args.max_requests is not None:
        argv += ["--max-requests", str(args.max_requests)]
    if args.action == "dry-run":
        argv.append("--dry-run")
    if args.action == "resume":
        if not args.output_dir:
            raise ValueError("resume requires the existing --output-dir")
        argv.append("--resume")
        return main(argv)
    with tempfile.TemporaryDirectory(prefix="vsi8b-profile-") as temporary:
        path = Path(temporary) / "profile.json"
        path.write_text(json.dumps(profile, ensure_ascii=False, indent=2))
        return main([*argv, "--config", str(path)])


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("check", "dry-run", "prepare", "run", "resume"))
    result.add_argument("--data-root", type=Path, default=default_data_root())
    result.add_argument("--online-config", type=Path, default=ROOT / "runner.yaml")
    result.add_argument("--cache-root", type=Path, default=ROOT / "prepared_cache")
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--max-requests", type=int,
                        help="explicit cap for preparation or the unchanged solver campaign")
    result.add_argument("--min-free-gib", type=float, default=80,
                        help="storage reserve for M5 preparation")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.action == "check":
            print(json.dumps(check_inputs(args.data_root, args.online_config), ensure_ascii=False, indent=2))
            return 0
        if args.action in {"dry-run", "run", "resume"}:
            if args.action == "run":
                check_inputs(args.data_root, args.online_config)
                if not (args.cache_root / "refresh_manifest.json").is_file():
                    raise ValueError("prepare the independent Qwen8B M5 cache before run")
            return campaign(args)
        check_inputs(args.data_root, args.online_config)
        from skill3d.evaluation.qwen8b_prepare import prepare
        from skill3d.online.config import load_config
        config = load_config(args.online_config)
        result = prepare(
            args.data_root / "published_inputs.json", args.cache_root,
            load_fitset_cli()._base_config(config),
            max_requests=args.max_requests, min_free_gib=args.min_free_gib)
        print(json.dumps({k: result[k] for k in (
            "status", "n_questions", "n_audit_questions", "n_scenes")}, indent=2))
        return 0
    except Exception as exc:
        # These errors are generated locally; external provider errors are type-only upstream.
        print(f"VSIbench8Brunner stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
