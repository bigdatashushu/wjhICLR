#!/usr/bin/env python3
"""Frozen main Skill program and Qwen direct answer, four episodes per arm."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import fcntl
from functools import partial
import json
import os
from pathlib import Path
import sys
import traceback

if not __package__:
    sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "src"),
                   str(Path(__file__).resolve().parent.parent)]
    __package__ = "CV3D"

import yaml

from skill3d.small_image.cv3d_protocol import FrozenCV3DSkills
from skill3d.small_image.datasets import normalize_answer
from skill3d.small_image.runner import SmallImageRunnerConfig
from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig

from .client import NativeToolClient
from .common import InferenceConfig, digest, file_sha256, read_json, write_json
from .data import index_prepared, replay_inputs
from .governor import SharedSlots
from .paired import schedule
from .program import PROTOCOL, ProgramClient, prepare_request, solve
from .transport import ProxyOpener, health


def now():
    return datetime.now().astimezone().isoformat()


def source_identity():
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "src").rglob("*.py"))
    paths += list((root / "tests/CV3D").glob("*.py"))
    paths += [root / "configs/cv3d_program.yaml"]
    paths += list((root / "skill_library/candidates/cv3d-compact-testdev-1").rglob("*"))
    return {str(path.relative_to(root)): file_sha256(path)
            for path in sorted(paths) if path.is_file() and not path.name.startswith("test_")}


def preflight(config):
    output = Path(config["output"])
    if config["concurrency"] != 8 or config["max_requests"] < 844:
        raise ValueError("requires eight slots and a complete direct budget")
    raw_config = yaml.safe_load(Path(config["runner_config"]).read_text())
    runner = SmallImageRunnerConfig(**raw_config["solver"])
    frozen = FrozenCV3DSkills(config["skills_manifest"])
    inputs = replay_inputs(config["public_manifest"], config["dataset_root"],
                           config["official_annotations"])
    train = [p for p in inputs if p.problem.source_split == "train"]
    test = [p for p in inputs if p.problem.source_split == "test"]
    if Counter(p.problem.task_type for p in test) != {"Depth": 420, "Distance": 420}:
        raise ValueError("expected 840 author test questions")
    pilot = [p for task in ("Depth", "Distance")
             for p in [p for p in train if p.problem.task_type == task][:2]]
    if len(pilot) != 4 or len({p.problem.sample_id for p in pilot + test}) != 844:
        raise ValueError("requires four train questions and unique phase IDs")
    index = index_prepared(config["prepared_root"])
    identities = {"tools": {}, "direct": {}}
    for ordinal, public in enumerate(pilot + test, 1):
        for arm in identities:
            identities[arm][public.problem.sample_id] = prepare_request(
                public, index.get(public.problem.input_sha256), arm=arm,
                frozen=frozen, config=runner)[-1]
        if ordinal % 10 == 0 or ordinal == 844:
            write_json(output / "preflight_progress.json", {
                "at": now(), "checked": ordinal, "total": 844, "model_requests": 0})
    manifest = {
        "protocol": PROTOCOL, "evaluation_role": "test_development",
        "config": config, "source": source_identity(),
        "runner": asdict(runner),
        "model": InferenceConfig(timeout_s=config["timeout_s"]).public_dict(),
        "arms": {"tools": "skill_program", "direct": "qwen_direct"},
        "public_manifest_sha256": file_sha256(config["public_manifest"]),
        "official_annotations_sha256": file_sha256(config["official_annotations"]),
        "host_answer_file_sha256": file_sha256(config["answers_path"]),
        "train_protocol_ids": [p.problem.sample_id for p in pilot],
        "test_ids": [p.problem.sample_id for p in test],
        "request_identities": identities, "skills_manifest_sha256": frozen.manifest_sha256,
    }
    if (output / "manifest.json").exists():
        if read_json(output / "manifest.json") != manifest:
            raise ValueError("frozen main paired manifest changed")
    else:
        write_json(output / "manifest.json", manifest, exclusive=True)
        write_json(output / "preflight.json", {
            "status": "passed", "at": now(), "checked_questions": 844,
            "model_requests": 0, "gpu_requests": 0, "manifest_sha256": digest(manifest),
        }, exclusive=True)
    return pilot, test, index, identities, manifest, runner, frozen


def run(config, *, execute=False):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "operator.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pilot, test, index, identities, manifest, runner, frozen = preflight(config)
        if not execute:
            return 0
        sys.path.insert(0, config["auth_module_root"])
        from transport_auth import load_transport_token
        gateway = health(config["gateway"], load_transport_token)
        if (gateway["source_sha256"] != config["gateway_source_sha256"]
                or gateway["max_inflight"] != 8
                or gateway["endpoint_sha256"] != config["upstream_endpoint_sha256"]):
            raise ValueError("gateway identity or capacity changed")
        governor = RequestGovernor(output / "budget", RequestGovernorConfig(
            config["max_requests"], config["concurrency"]))
        wrapped = SharedSlots(governor, reserve_direct=844, max_slots=8)
        model_config = InferenceConfig(timeout_s=config["timeout_s"])
        os.environ["CV3D_PROXY_PLACEHOLDER"] = "local-proxy-configuration"
        run_id = digest(manifest)

        def client_factory(arm, public):
            operation = digest({"arm": arm, "input": public.problem.input_sha256})
            cls = ProgramClient if arm == "tools" else NativeToolClient
            return cls(model_config, api_key_env="CV3D_PROXY_PLACEHOLDER",
                       opener=ProxyOpener(
                           gateway=config["gateway"], secret_loader=load_transport_token,
                           operation_id=operation, run_identity=run_id))

        # Labels remain in this host scoring function and never enter solver kwargs.
        answer_map = {(row["source_split"], row["sample_id"]): row["reference_answer"]
                      for row in read_json(config["answers_path"]) if row["benchmark"] == "cv3d"}
        for stage, selected in (("protocol_train", pilot), ("test", test)):
            result, rows = schedule(
                selected, identities, index, output=output / stage,
                governor=wrapped, client_factory=client_factory, concurrency=8, lend_idle=False,
                answers={p.problem.sample_id: normalize_answer(
                    answer_map[(p.problem.source_split, p.problem.sample_id)], p.problem.options)
                         for p in selected},
                solver=partial(solve, frozen=frozen, config=runner), protocol=PROTOCOL)
            if not result["complete"]:
                raise ValueError("main paired stage stopped")
            if stage == "protocol_train":
                calls = sum(row["succeeded_tool_calls"] for row in rows["tools"].values())
                if not calls or any(row["status"] != "answered"
                                    for arm in rows for row in rows[arm].values()):
                    raise ValueError("training did not establish legal answers and tool interaction")
                write_json(output / "protocol_validation.json", {
                    "status": "passed", "at": now(), "train_questions": 4,
                    "successful_tool_calls": calls, "requests": governor.summary(),
                    "test_scores_used_to_change_configuration": False,
                }, exclusive=True)
        write_json(output / "completion.json", {
            "status": "complete", "at": now(), "questions": 840, "episodes": 1680,
            "requests": governor.summary(), "manifest_sha256": run_id,
            "evaluation_role": "test_development",
        }, exclusive=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    startup = {"pid": os.getpid(), "at": now(), "phase": "preflight"}
    process = Path("/proc") / str(os.getpid())
    if process.exists():
        startup.update(
            start_ticks=(process / "stat").read_text().split(") ", 1)[1].split()[19],
            command_sha256=file_sha256(process / "cmdline"))
    write_json(output / "startup.json", startup, exclusive=True)
    try:
        code = run(config, execute=args.execute)
    except BaseException as exc:
        write_json(output / "failure.json", {
            "at": now(), "error_type": type(exc).__name__, "automatic_retry": False,
            "locations": [{"file": Path(f.filename).name, "line": f.lineno}
                          for f in traceback.extract_tb(exc.__traceback__)],
        })
        code = 2
    write_json(output / "exit.json", {"at": now(), "returncode": code})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
