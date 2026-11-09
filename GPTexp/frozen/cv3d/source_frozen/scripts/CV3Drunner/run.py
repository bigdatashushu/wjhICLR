#!/usr/bin/env python3
"""Zero-request preflight by default; --execute runs the frozen paired protocol."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
import fcntl
import os
from pathlib import Path
import sys
import traceback

if not __package__:
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "scripts"), str(root / "tests")]
    __package__ = "CV3Drunner"

import yaml

from skill3d.small_image.datasets import normalize_answer
from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig
from CV3D.client import NativeToolClient
from CV3D.common import InferenceConfig, digest, file_sha256, read_json, write_json
from CV3D.data import index_prepared, replay_inputs
from CV3D.governor import SharedSlots
from CV3D.paired import schedule, STOP_STATUSES
from CV3D.program_worker import RunStop
from CV3D.transport import ProxyOpener, health

from .config import RunnerConfig
from .program import PROTOCOL, prepare_request
from .protocol import FrozenSkills
from .worker import ProcessSolver


def now():
    return datetime.now().astimezone().isoformat()


def source_identity():
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "src").rglob("*.py"))
    paths += list((root / "tests/CV3D").glob("*.py"))
    paths += [p for p in Path(__file__).parent.rglob("*")
              if p.suffix in {".py", ".json", ".yaml", ".md"}]
    return {str(p.relative_to(root)): file_sha256(p)
            for p in sorted(set(paths)) if p.is_file() and not p.name.startswith("test_")}


def select_inputs(inputs, selection=None, *, diagnostic=False, test_only=False):
    train = [p for p in inputs if p.problem.source_split == "train"]
    test = [p for p in inputs if p.problem.source_split == "test"]
    if test_only:
        if diagnostic or selection is not None:
            raise ValueError("full test run cannot select a subset or training questions")
        if (Counter(p.problem.task_type for p in test) != {"Depth": 420, "Distance": 420}
                or len({p.problem.sample_id for p in test}) != 840):
            raise ValueError("full test run requires all 840 unique author test questions")
        return [], test
    if diagnostic:
        if not isinstance(selection, dict) or set(selection) != {"test_ids"}:
            raise ValueError("diagnostic trial requires explicit test_ids only")
        ids = selection["test_ids"]
        if (not isinstance(ids, list) or len(ids) != 10
                or any(not isinstance(i, str) or not i for i in ids)
                or len(set(ids)) != 10):
            raise ValueError("diagnostic trial requires exactly ten unique test IDs")
        index = {p.problem.sample_id: p for p in test}
        if set(ids) - set(index):
            raise ValueError("diagnostic ID outside author test split")
        selected = [index[i] for i in ids]
        if Counter(p.problem.task_type for p in selected) != {"Depth": 5, "Distance": 5}:
            raise ValueError("diagnostic trial requires five questions per task")
        return [], selected
    if selection is None:
        if Counter(p.problem.task_type for p in test) != {"Depth": 420, "Distance": 420}:
            raise ValueError("expected 840 author test questions")
        pilot = [p for task in ("Depth", "Distance")
                 for p in [p for p in train if p.problem.task_type == task][:2]]
        if len(pilot) != 4:
            raise ValueError("four training protocol questions required")
    else:
        if set(selection) != {"train_ids", "test_ids"}:
            raise ValueError("selection requires frozen train_ids and test_ids")

        def selected(pool, ids):
            if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
                    or any(not isinstance(i, str) for i in ids)):
                raise ValueError("selected IDs must be unique nonempty strings")
            index = {p.problem.sample_id: p for p in pool}
            if set(ids) - set(index):
                raise ValueError("selected ID outside requested split")
            return [index[i] for i in ids]

        pilot = selected(train, selection["train_ids"])
        test = selected(test, selection["test_ids"])
        if {p.problem.task_type for p in pilot} != {"Depth", "Distance"}:
            raise ValueError("training protocol must cover both tasks")
    if len({p.problem.sample_id for p in pilot + test}) != len(pilot + test):
        raise ValueError("phase IDs must be unique")
    return pilot, test


def preflight(config):
    output = Path(config["output"])
    if config["concurrency"] != 8:
        raise ValueError("paired protocol requires eight slots, four per arm")
    runner_path = config.get("runner_config", str(Path(__file__).with_name("runner.yaml")))
    skill_path = config.get("skills_manifest", str(Path(__file__).parent / "skills/frozen_skills.json"))
    runner = RunnerConfig(**yaml.safe_load(Path(runner_path).read_text())["solver"])
    frozen = FrozenSkills(skill_path)
    model = InferenceConfig(timeout_s=config["timeout_s"])
    if (runner.seed != model.seed or runner.max_tokens != model.max_tokens
            or runner.enable_thinking != model.thinking or runner.expected_model != model.model):
        raise ValueError("runner and paired model configuration disagree")
    # Resolve local defaults once into the manifest used by the child process.
    config = {**config, "runner_config": str(Path(runner_path).resolve()),
              "skills_manifest": str(Path(skill_path).resolve())}
    inputs = replay_inputs(config["public_manifest"], config["dataset_root"],
                           config["official_annotations"])
    run_kind = config.get("run_kind", "full_paired")
    if run_kind not in {"full_paired", "full_test_paired", "bounded_diagnostic_trial"}:
        raise ValueError("unknown run_kind")
    diagnostic = run_kind == "bounded_diagnostic_trial"
    pilot, test = select_inputs(
        inputs, config.get("selection"), diagnostic=diagnostic,
        test_only=run_kind == "full_test_paired")
    total = len(pilot) + len(test)
    if type(config["max_requests"]) is not int or config["max_requests"] < total * 7:
        raise ValueError("budget must cover six Skill requests and one direct request per question")
    if diagnostic and config["max_requests"] != 70:
        raise ValueError("ten-question diagnostic request cap must be seventy")
    if run_kind == "full_test_paired" and config["max_requests"] != 5880:
        raise ValueError("full test paired request cap must be 5880")
    index = index_prepared(config["prepared_root"])
    identities = {"tools": {}, "direct": {}}

    def check(public):
        return public.problem.sample_id, {
            arm: prepare_request(
                public, index.get(public.problem.input_sha256), arm=arm,
                frozen=frozen, config=runner)[-1] for arm in identities}

    with ThreadPoolExecutor(max_workers=4) as pool:
        for ordinal, (sample_id, checked) in enumerate(pool.map(check, pilot + test), 1):
            for arm in identities:
                identities[arm][sample_id] = checked[arm]
            if ordinal % 10 == 0 or ordinal == total:
                write_json(output / "preflight_progress.json", {
                    "at": now(), "checked": ordinal, "total": total, "model_requests": 0})
    manifest = {
        "protocol": PROTOCOL, "evaluation_role": "test_development", "run_kind": run_kind,
        "config": config,
        "source": source_identity(), "runner": asdict(runner),
        "model": InferenceConfig(timeout_s=config["timeout_s"]).public_dict(),
        "arms": {"tools": "skill_program", "direct": "qwen_direct"},
        "public_manifest_sha256": file_sha256(config["public_manifest"]),
        "official_annotations_sha256": file_sha256(config["official_annotations"]),
        "host_answer_file_sha256": file_sha256(config["answers_path"]),
        "train_protocol_ids": [p.problem.sample_id for p in pilot],
        "test_ids": [p.problem.sample_id for p in test], "request_identities": identities,
        "skills_manifest_sha256": frozen.manifest_sha256,
    }
    if (output / "manifest.json").exists():
        if read_json(output / "manifest.json") != manifest:
            raise ValueError("frozen CV3Drunner manifest changed")
        receipt = read_json(output / "preflight.json")
        if (receipt["manifest_sha256"] != digest(manifest)
                or receipt["model_requests"] != 0 or receipt["gpu_requests"] != 0):
            raise ValueError("preflight receipt changed")
    else:
        write_json(output / "manifest.json", manifest, exclusive=True)
        write_json(output / "preflight.json", {
            "status": "passed", "at": now(), "checked_questions": total,
            "model_requests": 0, "gpu_requests": 0, "manifest_sha256": digest(manifest)},
            exclusive=True)
    return pilot, test, index, identities, manifest, runner, frozen


def run(config, *, execute=False):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "operator.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pilot, test, index, identities, manifest, runner, frozen = preflight(config)
        if not execute:
            return 0
        config = manifest["config"]
        sys.path.insert(0, config["auth_module_root"])
        from transport_auth import load_transport_token
        gateway = health(config["gateway"], load_transport_token)
        if (gateway["source_sha256"] != config["gateway_source_sha256"]
                or gateway["max_inflight"] != 8
                or gateway["endpoint_sha256"] != config["upstream_endpoint_sha256"]):
            raise ValueError("gateway identity or capacity changed")
        governor = RequestGovernor(output / "budget", RequestGovernorConfig(
            config["max_requests"], 8))
        stop = RunStop(output / "stop_admission")
        total = len(pilot) + len(test)
        wrapped = SharedSlots(
            governor, reserve_direct=total, max_slots=8, stop_event=stop,
            stop_on_request_error=True)
        model_config = InferenceConfig(timeout_s=config["timeout_s"])
        os.environ["CV3D_PROXY_PLACEHOLDER"] = "local-proxy-configuration"
        run_id = digest(manifest)
        process_solver = ProcessSolver(config, manifest, runner, frozen)

        def client_factory(arm, public):
            if arm == "tools":
                return None
            return NativeToolClient(
                model_config, api_key_env="CV3D_PROXY_PLACEHOLDER",
                opener=ProxyOpener(
                    gateway=config["gateway"], secret_loader=load_transport_token,
                    operation_id=digest({"arm": arm, "input": public.problem.input_sha256}),
                    run_identity=run_id))

        # Scoring labels stay in the parent, outside every solver job.
        answer_map = {(row["source_split"], row["sample_id"]): row["reference_answer"]
                      for row in read_json(config["answers_path"]) if row["benchmark"] == "cv3d"}
        for stage, selected in (("protocol_train", pilot), ("test", test)):
            if not selected:
                continue
            result, rows = schedule(
                selected, identities, index, output=output / stage, governor=wrapped,
                client_factory=client_factory, concurrency=8, lend_idle=False,
                answers={p.problem.sample_id: normalize_answer(
                    answer_map[(p.problem.source_split, p.problem.sample_id)], p.problem.options)
                         for p in selected},
                solver=process_solver, protocol=PROTOCOL, stop_event=stop)
            if (not result["complete"] or any(
                    row["status"] in STOP_STATUSES for arm in rows for row in rows[arm].values())):
                raise ValueError("paired stage stopped")
            if stage == "protocol_train":
                calls = sum(row["succeeded_tool_calls"] for row in rows["tools"].values())
                tasks_answered = {row["task_type"] for row in rows["tools"].values()
                                  if row["status"] == "answered"}
                if (not calls or tasks_answered != {"Depth", "Distance"}
                        or any(row["status"] != "answered" for row in rows["direct"].values())):
                    raise ValueError("training did not establish answers and tool interaction")
                validation = output / "protocol_validation.json"
                if not validation.exists():
                    write_json(validation, {
                        "status": "passed", "at": now(), "train_questions": len(pilot),
                        "successful_tool_calls": calls, "requests": governor.summary(),
                        "model_failure_statuses_retained": dict(Counter(
                            row["status"] for row in rows["tools"].values())),
                        "gate_uses_accuracy": False,
                        "test_scores_used_to_change_configuration": False}, exclusive=True)
        completion = output / "completion.json"
        if not completion.exists():
            write_json(completion, {
                "status": "complete", "at": now(), "questions": len(test),
                "episodes": 2 * len(test), "requests": governor.summary(),
                "manifest_sha256": run_id, "evaluation_role": "test_development"}, exclusive=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f"startup.{os.getpid()}.json", {
        "pid": os.getpid(), "at": now(), "phase": "preflight"}, exclusive=True)
    try:
        return run(config, execute=args.execute)
    except BaseException as exc:
        write_json(output / f"failure.{os.getpid()}.json", {
            "at": now(), "error_type": type(exc).__name__, "automatic_retry": False,
            "locations": [{"file": Path(f.filename).name, "line": f.lineno}
                          for f in traceback.extract_tb(exc.__traceback__)]})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
