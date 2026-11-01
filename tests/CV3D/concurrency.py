"""Continue the frozen paired experiment after an audited concurrency change."""
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

if not __package__:
    sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "src"),
                   str(Path(__file__).resolve().parent.parent)]
    __package__ = "CV3D"

from skill3d.small_image.datasets import normalize_answer
from skill3d.synthesis.request_governor import (
    RequestGovernor, RequestGovernorConfig, _atomic,
)
from .client import NativeToolClient
from .common import InferenceConfig, digest, file_sha256, read_json, write_json
from .data import index_prepared, replay_inputs
from .governor import SharedSlots
from .paired import schedule
from .transport import ProxyOpener, health


def now():
    return datetime.now().astimezone().isoformat()


def revise_policy(root, ops):
    before = read_json(root / "budget/state.json")
    assert before["policy"]["max_requests"] == 6000
    assert before["policy"]["max_inflight"] == 4
    assert all(r["status"] == "completed" for r in before["requests"])
    old = RequestGovernor(root / "budget", RequestGovernorConfig(6000, 4))
    with old._locked():
        state = old._state()
        assert state == before
        write_json(ops / "budget-before.json", state, exclusive=True)
        request_identity = digest(state["requests"])
        state["policy_history"] = [{
            "at": now(), "old": state["policy"].copy(),
            "new_max_inflight": 8, "authorization": "user_requested_concurrency_increase",
            "prior_request_count": len(state["requests"]),
            "prior_requests_sha256": request_identity,
        }]
        state["policy"]["max_inflight"] = 8
        _atomic(root / "budget/state.json", state)
        assert digest(read_json(root / "budget/state.json")["requests"]) == request_identity
    write_json(ops / "policy-change.json", {
        "at": now(), "prior_request_count": len(state["requests"]),
        "prior_requests_sha256": request_identity, "max_requests": 6000,
        "cv3d_max_inflight": 8, "mmsi_max_inflight": 4,
        "shared_mmsi_slots_removed_from_cv3d": True,
    }, exclusive=True)


def run(config):
    root = Path(config["root"])
    ops = Path(config["ops"])
    with (root / "operator.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert read_json(ops / "drained.json")["status"] == "drained"
        assert not (root / "failure.json").exists()
        assert read_json(root / "preflight.json")["status"] == "passed"
        assert read_json(root / "protocol_validation.json")["status"] == "passed"
        manifest = read_json(root / "manifest.json")
        original = manifest["config"]
        sys.path.insert(0, original["auth_module_root"])
        from transport_auth import load_transport_token
        gateway = health(config["gateway"], load_transport_token)
        assert gateway["source_sha256"] == config["gateway_source_sha256"]
        assert gateway["max_inflight"] == 8
        public = replay_inputs(
            original["public_manifest"], original["dataset_root"],
            original["official_annotations"])
        selected = set(manifest["test_ids"])
        inputs = [p for p in public if p.problem.sample_id in selected]
        assert len(inputs) == 840 and all(p.problem.source_split == "test" for p in inputs)
        index = index_prepared(original["prepared_root"])
        inference_config = InferenceConfig(timeout_s=original["timeout_s"])
        revise_policy(root, ops)
        governor = RequestGovernor(root / "budget", RequestGovernorConfig(6000, 8))
        wrapped = SharedSlots(governor, reserve_direct=844)
        os.environ["CV3D_PROXY_PLACEHOLDER"] = "local-proxy-configuration"
        run_id = digest(manifest)

        def client_factory(arm, p):
            proxy = ProxyOpener(
                gateway=config["gateway"], secret_loader=load_transport_token,
                operation_id=digest({"arm": arm, "input": p.problem.input_sha256}),
                run_identity=run_id)
            directory = root / "test/episodes" / arm / p.problem.sample_id
            proxy.count = len(list(directory.glob("*.intent.json")))
            return NativeToolClient(
                inference_config, api_key_env="CV3D_PROXY_PLACEHOLDER", opener=proxy)

        # Host-only scoring. These values are never passed to inference workers.
        raw = read_json(original["answers_path"])
        labels = {(r["source_split"], r["sample_id"]): r["reference_answer"]
                  for r in raw if r["benchmark"] == "cv3d"}
        answers = {p.problem.sample_id: normalize_answer(
            labels[("test", p.problem.sample_id)], p.problem.options) for p in inputs}
        result, rows = schedule(
            inputs, manifest["request_identities"], index, output=root / "test",
            governor=wrapped, client_factory=client_factory, answers=answers,
            concurrency=8, resume_sessions=True)
        if not result["complete"]:
            raise ValueError("concurrent stage stopped")
        write_json(root / "completion.json", {
            "status": "complete", "at": now(), "questions": 840, "episodes": 1680,
            "requests": governor.summary(), "manifest_sha256": run_id,
            "operational_revision": str(ops),
        }, exclusive=True)


def main():
    config = read_json(sys.argv[1])
    root, ops = Path(config["root"]), Path(config["ops"])
    process = Path("/proc") / str(os.getpid())
    write_json(ops / "startup.json", {
        "at": now(), "pid": os.getpid(),
        "start_ticks": (process / "stat").read_text().split(") ", 1)[1].split()[19],
        "command_sha256": file_sha256(process / "cmdline"),
        "cv3d_max_inflight": 8, "per_arm_initial_concurrency": 4,
        "original_manifest_sha256": file_sha256(root / "manifest.json"),
    }, exclusive=True)
    try:
        run(config)
        code = 0
    except BaseException as exc:
        import traceback
        failure = {
            "at": now(), "error_type": type(exc).__name__, "automatic_retry": False,
            "locations": [{"file": Path(f.filename).name, "line": f.lineno}
                          for f in traceback.extract_tb(exc.__traceback__)],
            "operational_revision": str(ops),
        }
        write_json(ops / "failure.json", failure)
        write_json(root / "failure.json", failure)
        code = 2
    write_json(root / "exit.json", {"at": now(), "returncode": code})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
