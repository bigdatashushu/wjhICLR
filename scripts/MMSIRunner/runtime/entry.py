"""Run frozen MMSI solvers with audited transport and host-only trace storage."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
for key in list(os.environ):
    if any(s in key.upper() for s in ("API_KEY", "TOKEN", "ENDPOINT", "BASE_URL", "MODELS_KEY")):
        os.environ.pop(key, None)
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
sys.path.insert(0, str(ROOT))
import pilot
import provider
import audited_eval_transport
from host_trace_adapter import trace_result
import watcher

provider.ProxyOpener = audited_eval_transport.ProxyOpener
pilot.model_tool_response = trace_result


def solve(job):
    return pilot.solve_worker(job)


def verify_current():
    manifest = pilot.read(ROOT / "manifest.json")
    state = pilot.verify(manifest)
    original = json.loads(Path(manifest["original_budget_root"], "state.json").read_text())
    baseline = pilot.read(ROOT / "budget_reservation.json")
    prefix = json.loads((ROOT / "original_budget_prefix.json").read_text())
    assert original["policy"]["max_requests"] == 20000
    assert original["requests"][:len(prefix["requests"])] == prefix["requests"]
    assert len(original["requests"]) + len(state["requests"]) <= 20000
    assert baseline["reserved_mmsi_requests"] == 1900
    assert len(state["requests"]) <= 1900
    assert all(r["status"] in {"completed", "error"} for r in original["requests"])
    cv = Path(manifest["cv3d_root"])
    assert not watcher.process_state(pilot.read(cv / "startup.json"))["alive"]
    return manifest, state


def final_audit(manifest, state):
    assert all(r["status"] == "completed" for r in state["requests"])
    private = json.loads((ROOT / "test_budget/state.json").read_text())
    assert private["policy"] == manifest["test_policy"]
    assert len(private["requests"]) == len(state["requests"])
    assert all(r["status"] == "completed" for r in private["requests"])
    frozen = {}
    paired = {}
    actual_requests = 0
    for job in manifest["jobs"]:
        path = ROOT / "results" / job["arm"] / (job["operation_id"] + ".json")
        row = pilot.read(path)
        assert row["job_sha256"] == pilot.sha_value(job)
        assert row["status"] == "complete" and row["reference_answer_visible"] is False
        assert row["result"]["input_sha256"] == job["input_sha256"]
        for relative, expected in row["request_records"].items():
            assert pilot.sha(ROOT / relative) == expected
            if not relative.endswith(".response.json"):
                continue
            response = pilot.read(ROOT / relative)
            actual_requests += 1
            assert response["status"] == "received"
            identity = response["request_identity"]
            params = identity["parameters"]
            assert params["model"] == "qwen3.8-flash"
            assert params["seed"] == 137 and params["temperature"] == 0
            assert params["enable_thinking"] is True and params["max_tokens"] == 8192
            assert [r["jpeg_sha256"] for r in identity["images_in_order"]] == [
                r["sha256"] for r in job["images"]]
            assert response["response_identity"]["model"] == "qwen3.8-flash"
            assert response["response_identity"]["model_reported"] is True
            if relative.endswith("001.response.json"):
                paired.setdefault(job["sample_id"], {})[job["arm"]] = (job, identity)
        frozen[str(path.relative_to(ROOT))] = pilot.sha(path)
    assert len(frozen) == 730 and len(paired) == 365
    assert actual_requests == len(state["requests"])
    for arms in paired.values():
        assert set(arms) == {"skill", "direct"}
        a, ai = arms["skill"]
        b, bi = arms["direct"]
        assert a["input_sha256"] == b["input_sha256"]
        assert ai["parameters"] == bi["parameters"]
        assert ai["images_in_order"] == bi["images_in_order"]
    pilot.write(ROOT / "request_pairing_audit.json", {
        "status": "passed", "pairs": 365, "requests": actual_requests,
        "parameters_and_original_images_equal": True, "at": pilot.now()})
    pilot.write(ROOT / "predictions_frozen.json", {
        "predictions": frozen, "questions": 365, "episodes": 730, "at": pilot.now()})
    pilot.score(final=True)
    scores = pilot.read(ROOT / "scores.json")
    assert scores["status"] == "complete" and len(scores["rows"]) == 730
    assert all(scores["summary"][a]["completed"] == 365 for a in ("skill", "direct"))
    pilot.write(ROOT / "completion.json", {
        "status": "complete", "episodes": 730, "questions": 365, "at": pilot.now(),
        "manifest_sha256": manifest["content_sha256"], "reference_answers_read": False,
        "request_usage": pilot.governor(ROOT / "requests", manifest["shared_policy"]).summary(),
        "online_train_requests": 0, "test_only": True,
        "request_pairing_audit_sha256": pilot.sha(ROOT / "request_pairing_audit.json"),
        "scores_sha256": pilot.sha(ROOT / "scores.json"),
        "frozen_predictions_sha256": pilot.sha(ROOT / "predictions_frozen.json")})
    pilot.update(ROOT / "progress.json", {
        "status": "complete", "episodes": 730, "at": pilot.now(), "test_only": True})


def run():
    manifest = pilot.read(ROOT / "manifest.json")
    assert pilot.read(ROOT / "ready.json")["status"] == "ready"
    with ExitStack() as locks:
        for path in (ROOT / "operator.lock", Path(manifest["original_budget_root"]) /
                     "thinking_pilot_operator.lock"):
            stream = locks.enter_context(path.open("a+b"))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, state = verify_current()
        assert not state["requests"] and not (ROOT / "intents").exists()
        h = audited_eval_transport.health()
        assert h["source_sha256"] == manifest["audited_gateway_source_sha256"]
        assert h["schema_version"] == manifest["jobs"][0]["proxy_schema"]
        proc = Path("/proc") / str(os.getpid())
        fields = (proc / "stat").read_text().split(") ", 1)[1].split()
        pilot.write(ROOT / "startup.json", {
            "status": "started", "pid": os.getpid(), "start_ticks": fields[19],
            "command_sha256": pilot.sha(proc / "cmdline"), "at": pilot.now(),
            "automatic_retry": False, "max_workers": 4, "warmup_episodes": 1,
            "entry_sha256": pilot.sha(__file__), "manifest_sha256": manifest["content_sha256"]})
        pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"))
        iterator = iter(manifest["jobs"])
        pending = {}
        done = 0
        first_validated = False

        def dispatch():
            job = next(iterator, None)
            if job is not None:
                pilot.write(ROOT / "intents" / (job["operation_id"] + ".json"), {
                    "job_sha256": pilot.sha_value(job), "at": pilot.now()})
                pending[pool.submit(solve, job)] = job

        try:
            dispatch()
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                failures = []
                for future in completed:
                    job = pending.pop(future)
                    try:
                        row = future.result()
                    except BaseException as exc:
                        failures.append(exc)
                        continue
                    done += 1
                    pilot.update(ROOT / "progress.json", {
                        "status": "running", "phase": "test_only", "completed_episodes": done,
                        "expected_episodes": 730, "last_arm": job["arm"], "last_question": job["ordinal"],
                        "last_status": row["result"]["status"], "at": pilot.now(),
                        "request_usage": pilot.governor(ROOT / "requests", manifest["shared_policy"]).summary()})
                    print(json.dumps({"arm": job["arm"], "question": job["ordinal"],
                                      "status": row["result"]["status"],
                                      "seconds": round(row["seconds"], 2),
                                      "tools": [row["result"]["succeeded_tool_calls"],
                                                row["result"]["attempted_tool_calls"]]}), flush=True)
                if failures:
                    raise failures[0]
                if done % 4 == 0 or not first_validated:
                    pilot.score()
                verify_current()
                if not first_validated:
                    first_validated = True
                    for _ in range(4 - len(pending)):
                        dispatch()
                else:
                    for _ in completed:
                        dispatch()
        except BaseException:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            pilot.score()
            raise
        else:
            pool.shutdown(wait=True)
        manifest, state = verify_current()
        final_audit(manifest, state)


if __name__ == "__main__":
    code = 0
    try:
        run()
    except BaseException as exc:
        code = 2
        pilot.update(ROOT / "failure.json", {
            "status": "stopped", "error_type": type(exc).__name__, "at": pilot.now(),
            "automatic_retry": False,
            "locations": [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                          for f in traceback.extract_tb(exc.__traceback__)]})
    pilot.update(ROOT / "exit.json", {
        "returncode": code, "at": pilot.now(), "automatic_retry": False})
    sys.exit(code)
