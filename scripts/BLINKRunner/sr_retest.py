#!/usr/bin/env python3
"""Public-only SRv2 development retest with immutable prior input/cache identity."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr_trial as original

sr, FullResponseClient, now = original.sr, original.FullResponseClient, original.now
SAMPLE_SEED, SIZE, CAP, CONCURRENCY = 20261105, 20, 60, 8
ROLE = "internal_test_development_retest"


def sample_ids(split):
    ids = split["internal_test"]
    chosen = set(random.Random(SAMPLE_SEED).sample(ids, SIZE))
    return [sid for sid in ids if sid in chosen]


def cohort(root, split):
    """Keep the first public sample fixed and bind its exact 80-question complement."""
    root = Path(root)
    path = root / "retest_cohort.json"
    if not path.exists():
        return {"questions": SIZE, "request_cap": CAP, "concurrency": CONCURRENCY,
                "role": ROLE, "prefix": "sr-v2-retest",
                "selection": "uniform public-only internal_test sample"}
    value = sr.read(path)
    parent = sr.read(root / "parent_retest20_manifest.json")
    completed = sr.read(root / "parent_retest20_completed.json")
    if (value["schema"] != "blink-sr-development-retest-cohort/1"
            or value["selection"] != "remaining80" or value["questions"] != 80
            or value["request_cap"] != 240 or value["concurrency"] != CONCURRENCY
            or value["role"] != ROLE + "_remaining" or value["prefix"] != "sr-v2-remaining"
            or value["sample_seed"] != SAMPLE_SEED
            or value["parent_manifest_sha256"] != parent["content_sha256"]
            or value["parent_completed_sha256"] != completed["content_sha256"]
            or parent["schema"] != "blink-sr-development-retest/1"
            or parent["split_sha256"] != split["content_sha256"]
            or parent["sample_seed"] != SAMPLE_SEED
            or [r["problem"]["sample_id"] for r in parent["inspection"]["inputs"]] != sample_ids(split)
            or completed["manifest_sha256"] != parent["content_sha256"]
            or completed["episodes"] != SIZE
            or any(completed["budget"][k] for k in (
                "error_requests", "unknown_requests", "active_requests"))):
        raise ValueError("sr_retest_remaining_parent_changed")
    if (parent["inspection"]["configuration"] != sr.configuration()[0]
            or parent["inspection"]["skill_sha256"] != sr.file_sha(sr.SKILL)):
        raise ValueError("sr_retest_remaining_scientific_protocol_changed")
    orchestration = {"scripts/BLINKRunner/sr_retest.py"}
    if {k: v for k, v in parent["inspection"]["sources"].items() if k not in orchestration} != {
            k: v for k, v in sr.source_identity().items() if k not in orchestration}:
        raise ValueError("sr_retest_remaining_scientific_source_changed")
    return value


def selected_ids(root, split):
    profile = cohort(root, split)
    chosen = sample_ids(split)
    if profile["selection"] == "remaining80":
        chosen = [sid for sid in split["internal_test"] if sid not in set(chosen)]
    if len(chosen) != profile["questions"] or len(set(chosen)) != len(chosen):
        raise ValueError("sr_retest_cohort_denominator_changed")
    return chosen


def public_inputs(root):
    root = Path(root).resolve()
    split = sr.read(root / "split.json")
    if (len(split["internal_test"]) != 100 or len(split["train"]) != 43
            or set(split["train"]) & set(split["internal_test"])):
        raise ValueError("sr_retest_invalid_split")
    # Validate the complete public index independently of prior answer trajectories.
    data, current = sr.load_inputs(root / "dataset", ("val",))
    if original.split_manifest(root / "dataset") != {
            k: v for k, v in split.items() if k != "content_sha256"}:
        raise ValueError("sr_retest_public_split_changed")
    by_id = {p.sample_id: (p, prompt) for p, prompt in current}
    prior, parents = {}, {}
    for name in ("parent20_manifest.json", "parent80_manifest.json"):
        parent = sr.read(root / name)
        if parent["split_sha256"] != split["content_sha256"]:
            raise ValueError("sr_retest_parent_split_changed")
        parents[name] = parent["content_sha256"]
        for item in parent["inspection"]["inputs"]:
            sid = item["problem"]["sample_id"]
            if sid in prior:
                raise ValueError("sr_retest_parent_overlap")
            prior[sid] = item
    if set(prior) != set(split["internal_test"]):
        raise ValueError("sr_retest_parent_coverage_changed")
    items, bindings = [], []
    for sid in selected_ids(root, split):
        old = prior[sid]
        public = old["problem"]
        problem = sr.SmallImageProblem(**{
            k: tuple(v) if k in {"options", "image_paths"} else v
            for k, v in public.items() if k != "input_sha256"})
        current_problem, prompt = by_id[sid]
        if (problem.public_dict() != public or prompt != old["official_prompt"]
                or {k: v for k, v in public.items() if k not in {
                    "image_paths", "input_sha256"}} !=
                {k: v for k, v in current_problem.public_dict().items() if k not in {
                    "image_paths", "input_sha256"}}
                or sr.image_identities(problem) != old["images"]
                or sr.image_identities(current_problem)[0]["sha256"] != old["images"][0]["sha256"]):
            raise ValueError("sr_retest_original_public_input_changed")
        prepared = sr.prepared_for(problem, root / "prepared")
        if prepared.artifact_sha256 != old["prepared_artifact_sha256"]:
            raise ValueError("sr_retest_original_cache_changed")
        items.append((problem, prompt))
        bindings.append({
            "sample_id": sid, "input_sha256": problem.input_sha256,
            "image_sha256": old["images"][0]["sha256"],
            "prepared_artifact_sha256": prepared.artifact_sha256,
            "reused_direct_payload_sha256": old["direct_payload_sha256"],
        })
    return data, items, split, parents, bindings


def inspection(root):
    data, items, split, parents, bindings = public_inputs(root)
    ready = sr.inspect_inputs(data, items, prepared_root=Path(root).resolve() / "prepared")
    current = ready["configuration"]
    for name in parents:
        previous = sr.read(Path(root) / name)["inspection"]["configuration"]
        if (previous["model"] != current["model"]
                or {k: v for k, v in previous["solver"].items() if k != "prompt_profile"} !=
                {k: v for k, v in current["solver"].items() if k != "prompt_profile"}):
            raise ValueError("sr_retest_sampling_parameters_changed")
    if any(item["direct_payload_sha256"] != bind["reused_direct_payload_sha256"]
           for item, bind in zip(ready["inputs"], bindings)):
        raise ValueError("sr_retest_direct_payload_changed")
    return ready, split, parents, bindings


def jobs_for(ready, prefix="sr-v2-retest"):
    skill = sr.SKILL.read_text(encoding="utf-8")
    return [
        {"operation_id": f"{prefix}-{i:03d}-skill_tools",
         "spec": {"arm": "skill_tools", "problem": item["problem"],
                  "solver": ready["configuration"]["solver"],
                  "official_prompt": item["official_prompt"],
                  "prepared_root": ready["prepared_root"], "skill_text": skill,
                  "artifact_sha256": item["prepared_artifact_sha256"]}}
        for i, item in enumerate(ready["inputs"])
    ]


def freeze(root):
    root = Path(root).resolve()
    ready, split, parents, bindings = inspection(root)
    profile = cohort(root, split)
    size = profile["questions"]
    delivery = original.check_actual_initial_prompts(root, ready)
    manifest = sr.write_once(root / "manifest.json", {
        "schema": "blink-sr-development-retest/1",
        "inspection": ready, "jobs": jobs_for(ready, profile["prefix"]), "questions_per_arm": size,
        "split_sha256": split["content_sha256"], "parents": parents,
        "sample_seed": SAMPLE_SEED, "selection": profile["selection"],
        "public_bindings": bindings, "request_cap": profile["request_cap"], "concurrency": CONCURRENCY,
        "created_at": now(), "evaluation_role": profile["role"], "initial_skill_frozen": True,
        "training_evolution": False, "test_informed_prompt_development": True,
        "reference_answer_visible": False, "automatic_retry": False,
        "reused_comparison_arms": ["direct", "sr_v1"],
        **({"cohort_sha256": profile["content_sha256"]} if "content_sha256" in profile else {}),
    })
    return sr.write_once(root / "preparation.json", {
        "manifest_sha256": manifest["content_sha256"], "model_requests": 0,
        "reference_answers_read": False, "questions_per_arm": size, "new_episodes": size,
        "reused_direct_episodes": size, "reused_sr_v1_episodes": size,
        "zero_request_delivery_sha256": delivery["content_sha256"],
    })


def verify(root, manifest):
    ready, split, parents, bindings = inspection(root)
    profile = cohort(root, split)
    if (manifest["schema"] != "blink-sr-development-retest/1"
            or manifest["inspection"] != ready or manifest["jobs"] != jobs_for(ready, profile["prefix"])
            or manifest["split_sha256"] != split["content_sha256"]
            or manifest["parents"] != parents or manifest["public_bindings"] != bindings
            or manifest["sample_seed"] != SAMPLE_SEED
            or manifest["request_cap"] != profile["request_cap"] or manifest["concurrency"] != CONCURRENCY
            or manifest["evaluation_role"] != profile["role"]
            or manifest["questions_per_arm"] != profile["questions"]
            or manifest["selection"] != profile["selection"]
            or manifest.get("cohort_sha256") != profile.get("content_sha256")
            or manifest["reference_answer_visible"] is not False
            or manifest["automatic_retry"] is not False
            or len(manifest["jobs"]) != profile["questions"]):
        raise ValueError("sr_retest_frozen_identity_changed")


def initial_jobs(root, manifest, governor):
    return original.initial_jobs(root, manifest, governor)


def run(root):
    return original.run(root, verify_fn=verify, initial_jobs_fn=initial_jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify", "run"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        value = freeze(args.root)
    elif args.command == "run":
        value = run(args.root)
    else:
        verify(args.root, sr.read(args.root / "manifest.json"))
        value = {"status": "verified", "model_requests": 0}
    print(__import__("json").dumps(value))


if __name__ == "__main__":
    main()
