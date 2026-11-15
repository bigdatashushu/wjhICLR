#!/usr/bin/env python3
"""Freeze a deterministic five-per-type test sample and its online inputs."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil


def digest(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=str).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--native-root", required=True)
    parser.add_argument("--native-catalog", required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    args = parser.parse_args()
    assert not args.run_root.exists()
    tests = [json.loads(line) for line in (args.package / "test.jsonl").read_text().splitlines()]
    panel = json.loads((args.package / "panel.json").read_text())
    by_qa = {row["qa_id"]: row for row in panel["system_variants"]}
    rng = random.Random(137)
    selected = []
    for task in sorted({by_qa[row["qa_id"]]["question_type"] for row in tests}):
        pool = sorted([row for row in tests if by_qa[row["qa_id"]]["question_type"] == task],
                      key=lambda row: row["qa_id"])
        selected.extend(rng.sample(pool, 5))
    selected.sort(key=lambda row: (by_qa[row["qa_id"]]["question_type"], row["qa_id"]))
    ids = {row["qa_id"] for row in selected}
    assert len(ids) == 40 and len(Counter(by_qa[q]["question_type"] for q in ids)) == 8
    assert all(row["experiment_split"] == "test" for row in selected)
    args.run_root.mkdir()
    inputs = args.run_root / "inputs"
    inputs.mkdir()
    for name in ("online.yaml", "split.json", "questions.jsonl", "train.jsonl", "test.jsonl"):
        shutil.copyfile(args.package / name, inputs / name)
    shutil.copytree(args.package / "host_evaluation", inputs / "host_evaluation")
    shutil.copytree(args.package / "scene_overviews", inputs / "scene_overviews")
    shutil.copytree(args.source_snapshot, args.run_root / "code")
    write(inputs / "sample_questions.json", selected)
    original_panel_sha = hashlib.sha256((args.package / "panel.json").read_bytes()).hexdigest()
    panel["system_variants"] = [row for row in panel["system_variants"] if row["qa_id"] in ids]
    panel["direct_qwen_rows"] = [row for row in panel["direct_qwen_rows"] if row["qa_id"] in ids]
    panel["experiment_id"] += "-diagnostic40-20261105"
    for value in panel["panels"].values():
        value["system_input_variants"] = 40
    panel["proposal_sha256"] = digest({k: v for k, v in panel.items() if k != "proposal_sha256"})
    write(inputs / "panel.json", panel)
    published = json.loads((args.package / "published_inputs.json").read_text())
    original_locator = published["source_catalog_file"]
    published["source_catalog_file"] = args.native_catalog
    write(inputs / "published_inputs.json", published)
    controller = Path(__file__).with_name("run_g1m1_diagnostic.py")
    shutil.copyfile(controller, args.run_root / "controller.py")
    manifest = {
        "schema": "vsi-g1m1-diagnostic40/1", "seed": 137, "role": "test",
        "selection": "random.Random(137), sorted canonical types and lexically sorted QA pools",
        "questions_per_type": 5, "n_questions": 40,
        "n_scenes": len({row["scene_name"] for row in selected}),
        "task_counts": dict(sorted(Counter(by_qa[q]["question_type"] for q in ids).items())),
        "variants": panel["system_variants"],
        "native_root": args.native_root,
        "runtime": {"workers": 16, "episode_concurrency": 16, "request_concurrency": 8,
                    "max_requests": 280, "scene_cache_size": 34, "min_free_gib": 80,
                    "phase": "final_system", "automatic_model_retry": False,
                    "skill_evolution": False, "max_solver_rounds": 6,
                    "max_requests_per_episode": 7, "visual_fallback_max_requests": 1,
                    "input_scheduler": "independent_scene_providers_v1",
                    "question_preflight": "skipped_per_user_instruction"},
        "source_sha256": json.loads(
            (args.source_snapshot / "code_identity.json").read_text())["source_sha256"],
        "source_panel_sha256": original_panel_sha,
        "source_package": str(args.package),
        "published_locator_adaptation": {
            "original_locator": original_locator, "native_locator": args.native_catalog,
            "catalog_sha256": published["source_catalog_sha256"],
            "entries_and_scopes_unchanged": True},
        "input_files": {str(p.relative_to(inputs)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(inputs.rglob("*")) if p.is_file()},
        "controller_sha256": hashlib.sha256(controller.read_bytes()).hexdigest(),
        "reference_answer_visible": False,
    }
    manifest["identity_sha256"] = digest(manifest)
    write(args.run_root / "diagnostic.json", manifest)
    for path in args.run_root.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
    print(json.dumps({k: manifest[k] for k in ("n_questions", "n_scenes", "task_counts",
                                             "identity_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
