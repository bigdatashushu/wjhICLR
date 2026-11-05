"""Prepare one public frozen case for offline checks while full preflight runs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .adapters import ROOT, mind_config, problem_for, public_spec, setup, solver_config
from .traces import digest, read, write


def seed(dataset):
    source = setup(dataset)
    origin = ROOT / "frozen" / dataset
    if dataset == "mmsi":
        old = read(origin / "manifest.json")
        sample = old["jobs"][0]["problem"]["sample_id"]
        specs = []
        for job in old["jobs"]:
            if job["problem"]["sample_id"] != sample:
                continue
            spec = public_spec(job)
            spec["arm"] = "direct" if spec["arm"] == "direct" else "harness"
            if spec["arm"] == "direct":
                for key in ("prepared_root", "artifact_sha256", "skill", "allowed_tools"):
                    spec.pop(key, None)
            specs.append(spec)
        return specs
    if dataset == "cv3d":
        import json
        from CV3D.data import PublicInput
        from CV3Drunner.config import RunnerConfig
        from CV3Drunner.program import prepare_request
        from CV3Drunner.protocol import FrozenSkills
        old = read(origin / "manifest.json")
        sample = old["test_ids"][0]
        public = old["request_identities"]["direct"][sample]["problem"]
        problem = problem_for(public)
        rows = (json.loads(line) for line in
                Path(old["config"]["official_annotations"]).read_text().splitlines() if line.strip())
        prompt = next(row["prompt"] for row in rows
                      if (row["task"], row["question"], tuple(row["choices"]))
                      == (problem.task_type, problem.question, problem.options))
        config = RunnerConfig(**solver_config(old["runner"]))
        frozen = FrozenSkills(source / "scripts/CV3Drunner/skills/frozen_skills.json")
        directory = str(Path(old["config"]["prepared_root"]) / problem.input_sha256)
        specs = []
        for arm, legacy in (("direct", "direct"), ("harness", "tools")):
            prepared, _, _, _, identity = prepare_request(
                PublicInput(problem, prompt), directory if arm == "harness" else None,
                arm=legacy, frozen=frozen, config=config)
            previous = old["request_identities"][legacy][sample]
            if identity["initial_messages_sha256"] != previous["initial_messages_sha256"]:
                raise ValueError("offline_seed_initial_input_changed")
            spec = {"arm": arm, "problem": public, "official_prompt": prompt,
                    "solver": asdict(config), "initial_messages_sha256": identity["initial_messages_sha256"]}
            if arm == "harness":
                if prepared.artifact_sha256 != previous["artifact_sha256"]:
                    raise ValueError("offline_seed_prepared_changed")
                spec.update(prepared_directory=directory, artifact_sha256=prepared.artifact_sha256)
            specs.append(spec)
        return specs
    if dataset == "mindcube-tiny":
        from copy import deepcopy
        from MindcubeRunner.data import load, select
        from MindcubeRunner.protocol import messages
        old = read(origin / "run.config.json")
        config = mind_config(old["runner"])
        item = select(old["cache_root"], old["selection"], pins=old["input_pins"])[0]
        specs = []
        for arm, legacy in (("direct", "direct"), ("harness", "skill")):
            cache = load(item, prepared=arm == "harness")
            _, initial = messages(item, cache, arm=legacy, config=config)
            public = deepcopy(item)
            if arm == "direct":
                for view in public["views"]:
                    view.pop("prepared_directory", None)
            specs.append({"arm": arm, "item": public, "config": config.public(),
                          "initial_receipt": initial})
        return specs
    raise ValueError("unsupported_offline_seed_dataset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mmsi", "cv3d", "mindcube-tiny"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    specs = seed(args.dataset)
    if sorted(s["arm"] for s in specs) != ["direct", "harness"]:
        raise ValueError("offline_seed_pair_invalid")
    write(args.output, {"dataset": args.dataset, "specs": specs, "identity": digest(specs),
                        "model_requests": 0, "reference_answers_read": False,
                        "benchmark_evidence": False, "full_preflight": False}, exclusive=True)
    print(args.dataset, "single frozen offline fixture prepared")


if __name__ == "__main__":
    main()
