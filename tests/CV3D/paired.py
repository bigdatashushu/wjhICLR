#!/usr/bin/env python3
"""Shared scheduling for a fixed training protocol check and paired 840-test run."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

if not __package__:
    sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "src"),
                   str(Path(__file__).resolve().parent.parent)]
    __package__ = "CV3D"

from skill3d.small_image.datasets import normalize_answer
from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig

from .client import NativeToolClient
from .common import InferenceConfig, digest, file_sha256, read_json, write_json
from .data import index_prepared, replay_inputs
from .evaluate import source_identity, summarize
from .governor import SharedSlots
from .inference import prepare_request, solve
from .transport import ProxyOpener, health

PAIRED_PROTOCOL = "cv3d-simplified-versus-direct/1"
STOP_STATUSES = {"service_error", "host_error", "budget_exhausted", "cancelled"}


def now():
    return datetime.now().astimezone().isoformat()


def scores(inputs, answers, rows):
    arms = {arm: summarize(inputs, answers, rows[arm]) for arm in ("tools", "direct")}
    counts = Counter()
    for public in inputs:
        key = public.problem.sample_id
        if not all(key in rows[arm] for arm in rows):
            continue
        a, b = (rows[arm][key].get("answer") == answers[key]
                and rows[arm][key]["status"] == "answered" for arm in ("tools", "direct"))
        counts["both_correct" if a and b else "tools_only" if a else
               "direct_only" if b else "both_wrong"] += 1
    return {
        "protocol": PAIRED_PROTOCOL, "at": now(), "arms": arms,
        "paired": dict(counts), "paired_count": sum(counts.values()),
        "complete": all(s["complete"] for s in arms.values()),
    }


def schedule(inputs, identities, prepared_index, *, output, governor, client_factory,
             answers, stop_event=None, concurrency=4, resume_sessions=False):
    """Start 2 episodes per arm; lend idle capacity when one arm has no queued work."""
    stop_event = stop_event or threading.Event()
    if concurrency < 2 or concurrency % 2:
        raise ValueError("paired concurrency must be an even number >= 2")
    if isinstance(governor, SharedSlots):
        governor.stop_event = stop_event
    queues = {arm: deque(inputs) for arm in ("tools", "direct")}
    rows = {arm: {} for arm in queues}
    for arm in queues:
        pending = deque()
        for public in queues[arm]:
            p = output / "episodes" / arm / public.problem.sample_id
            if (p / "result.json").exists():
                row = read_json(p / "result.json")
                payload = {k:v for k,v in row.items() if k != "receipt_sha256"}
                if (digest(payload) != row.get("receipt_sha256")
                        or row["input_identity"] != identities[arm][public.problem.sample_id]
                        or row["status"] in STOP_STATUSES):
                    raise ValueError("saved episode requires diagnosis")
                rows[arm][public.problem.sample_id] = row
            elif p.exists() and any(p.iterdir()) and not resume_sessions:
                raise ValueError("unfinished attempt cannot be retried")
            else:
                pending.append(public)
        queues[arm] = pending

    def worker(arm, public):
        return solve(
            public, prepared_index.get(public.problem.input_sha256),
            arm=arm, client=client_factory(arm, public), governor=governor,
            output_directory=output / "episodes" / arm / public.problem.sample_id,
            stop_event=stop_event, expected_identity=identities[arm][public.problem.sample_id],
            **({"resume": True} if resume_sessions else {}))

    def publish():
        value = scores(inputs, answers, rows)
        value["budget"] = governor.summary()
        write_json(output / "scores.json", value)
        print(json.dumps({
            "at": now(), "phase": output.name,
            "completed": {arm: len(rows[arm]) for arm in rows},
            "requests": value["budget"]["reserved_requests"],
            "active_requests": value["budget"]["active_requests"],
            "complete": value["complete"],
            "accuracy": {arm: value["arms"][arm]["overall"]["accuracy_recorded_denominator"]
                         for arm in rows},
        }), flush=True)
        return value

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        active = {}
        def fill():
            while len(active) < concurrency and not stop_event.is_set():
                eligible = []
                for arm in queues:
                    running = sum(a == arm for a, _ in active.values())
                    other = "direct" if arm == "tools" else "tools"
                    cap = concurrency // 2 if queues[other] else concurrency
                    if queues[arm] and running < cap:
                        eligible.append((running, arm))
                if not eligible:
                    break
                _, arm = min(eligible)
                public = queues[arm].popleft()
                active[pool.submit(worker, arm, public)] = (arm, public)
        fill()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                arm, public = active.pop(future)
                try:
                    row = future.result()
                except BaseException as exc:
                    stop_event.set()
                    row = {
                        "arm": arm, "sample_id": public.problem.sample_id,
                        "source_split": public.problem.source_split,
                        "task_type": public.problem.task_type, "status": "host_error",
                        "answer": None, "error_code": type(exc).__name__,
                        "input_identity": identities[arm][public.problem.sample_id],
                    }
                    row["receipt_sha256"] = digest(row)
                    path = output / "episodes" / arm / public.problem.sample_id / "result.json"
                    if not path.exists():
                        write_json(path, row, exclusive=True)
                rows[arm][public.problem.sample_id] = row
                if row["status"] in STOP_STATUSES:
                    stop_event.set()
            publish()
            fill()
    return publish(), rows


def preflight(config):
    output = Path(config["output"])
    inputs = replay_inputs(config["public_manifest"], config["dataset_root"],
                           config["official_annotations"])
    train = [p for p in inputs if p.problem.source_split == "train"]
    test = [p for p in inputs if p.problem.source_split == "test"]
    if Counter(p.problem.task_type for p in test) != {"Depth": 420, "Distance": 420}:
        raise ValueError("expected author 840-test split")
    pilot = [p for task in ("Depth", "Distance") for p in train
             if p.problem.task_type == task]
    pilot = ([p for p in pilot if p.problem.task_type == "Depth"][:2]
             + [p for p in pilot if p.problem.task_type == "Distance"][:2])
    if len(pilot) != 4:
        raise ValueError("four training protocol examples required")
    index = index_prepared(config["prepared_root"])
    identities = {"tools": {}, "direct": {}}
    for ordinal, public in enumerate(pilot + test, 1):
        for arm in identities:
            *_, identity = prepare_request(
                public, index.get(public.problem.input_sha256), arm=arm)
            identities[arm][public.problem.sample_id] = identity
        if ordinal % 20 == 0:
            write_json(output / "preflight_progress.json", {
                "at": now(), "checked": ordinal, "total": 844, "model_requests": 0})
    identity = {
        "protocol": PAIRED_PROTOCOL, "config": config, "source": source_identity(),
        "public_manifest_sha256": file_sha256(config["public_manifest"]),
        "official_annotations_sha256": file_sha256(config["official_annotations"]),
        "train_protocol_ids": [p.problem.sample_id for p in pilot],
        "test_ids": [p.problem.sample_id for p in test],
        "request_identities": identities,
        "host_answer_file_sha256": file_sha256(config["answers_path"]),
    }
    if (output / "manifest.json").exists():
        if read_json(output / "manifest.json") != identity:
            raise ValueError("paired manifest changed")
    else:
        write_json(output / "manifest.json", identity, exclusive=True)
        write_json(output / "preflight.json", {
            "status": "passed", "at": now(), "checked_questions": 844,
            "model_requests": 0, "gpu_requests": 0, "manifest_sha256": digest(identity),
        }, exclusive=True)
    return pilot, test, index, identities, identity


def run(config, *, execute=False):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "operator.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pilot, test, index, identities, manifest = preflight(config)
        if not execute:
            return 0
        sys.path.insert(0, config["auth_module_root"])
        from transport_auth import load_transport_token
        gateway_health = health(config["gateway"], load_transport_token)
        if gateway_health["source_sha256"] != config["gateway_source_sha256"]:
            raise ValueError("gateway source changed")
        inference_config = InferenceConfig(timeout_s=config.get("timeout_s", 600.0))
        governor = RequestGovernor(
            output / "budget", RequestGovernorConfig(config["max_requests"], 4))
        wrapped = SharedSlots(
            governor, config.get("shared_slots_root"), reserve_direct=844)
        os.environ["CV3D_PROXY_PLACEHOLDER"] = "local-proxy-configuration"
        run_id = digest(manifest)
        def client_factory(arm, public):
            operation = digest({"arm": arm, "input": public.problem.input_sha256})
            return NativeToolClient(
                inference_config, api_key_env="CV3D_PROXY_PLACEHOLDER",
                opener=ProxyOpener(
                    gateway=config["gateway"], secret_loader=load_transport_token,
                    operation_id=operation, run_identity=run_id))
        # Host-only answer read. It never enters worker args, prompts or observations.
        host_answers = read_json(config["answers_path"])
        answer_map = {(r["source_split"], r["sample_id"]): r["reference_answer"]
                      for r in host_answers if r["benchmark"] == "cv3d"}
        def answers_for(selected):
            return {p.problem.sample_id: normalize_answer(
                answer_map[(p.problem.source_split, p.problem.sample_id)], p.problem.options)
                for p in selected}
        for stage, selected in (("protocol_train", pilot), ("test", test)):
            result, rows = schedule(
                selected, identities, index, output=output / stage,
                governor=wrapped, client_factory=client_factory,
                answers=answers_for(selected))
            if not result["complete"]:
                raise ValueError("paired stage stopped")
            if stage == "protocol_train":
                tool_calls = sum(r["succeeded_tool_calls"] for r in rows["tools"].values())
                if (not tool_calls or any(r["status"] != "answered"
                        for arm in rows for r in rows[arm].values())):
                    raise ValueError("training protocol check did not establish valid tool interaction")
                write_json(output / "protocol_validation.json", {
                    "status": "passed", "at": now(), "train_questions": 4,
                    "successful_tool_calls": tool_calls, "requests": governor.summary(),
                    "test_scores_used_to_change_configuration": False,
                }, exclusive=not (output / "protocol_validation.json").exists())
        write_json(output / "completion.json", {
            "status": "complete", "at": now(), "questions": 840, "episodes": 1680,
            "requests": governor.summary(), "manifest_sha256": run_id,
        }, exclusive=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    process = Path("/proc") / str(os.getpid())
    startup = {"pid": os.getpid(), "at": now(), "phase": "preflight"}
    if process.exists():
        startup.update(
            start_ticks=(process / "stat").read_text().split(") ", 1)[1].split()[19],
            command_sha256=file_sha256(process / "cmdline"))
    write_json(output / "startup.json", startup, exclusive=True)
    try:
        code = run(config, execute=args.execute)
    except BaseException as exc:
        import traceback
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
