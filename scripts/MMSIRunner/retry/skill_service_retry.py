"""One authorized solver retry per Skill service failure; separate host statistics."""
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
from copy import deepcopy
import fcntl
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

ORIGINAL = Path("/home/wangy/wangjh/harness3d/four-datasets-20261031/mmsi-365-test-rerun-20261101")
ROOT = ORIGINAL.parent / "mmsi-skill-service-retry-20261102-r01"
for k in list(os.environ):
    if any(t in k.upper() for t in ("API_KEY", "TOKEN", "ENDPOINT", "BASE_URL", "MODELS_KEY")):
        os.environ.pop(k, None)
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
sys.path.insert(0, str(ORIGINAL))
sys.path.insert(0, str(ORIGINAL / "ops"))
import entry
import provider
import timeout_extension06
import timeout_transport06
import timeout_handling08
pilot = entry.pilot
ORIGINAL_MANIFEST_SHA = "b933e38f0b02be6a334cd05ddfb16378ad2548a9198b9e555aa84c696e22a69b"
SCORER_SHA = "7b9a38068a1c7aa36e54d40eeb09da33ccbeab0059dacd27aac05213585e5a09"
ORDINALS = [204, 253, 300, 316, 318, 322, 323, 327, 351, 361, 362, 365]
AUTHORIZATION = "只重试 skill 这边的然后重新统计"
RETRY_SCOPE = "mmsi-skill-service-retry-20261102-r01"


def process_receipt():
    proc = Path("/proc") / str(os.getpid())
    fields = (proc / "stat").read_text().split(") ", 1)[1].split()
    return {"pid": os.getpid(), "start_ticks": fields[19],
            "command_sha256": pilot.sha(proc / "cmdline"), "at": pilot.now()}


def locks(stack):
    m = pilot.read(ORIGINAL / "manifest.json")
    for path in (ROOT / "operator.lock", ORIGINAL / "operator.lock",
                 Path(m["original_budget_root"]) / "thinking_pilot_operator.lock"):
        stream = stack.enter_context(path.open("a+b"))
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def originals():
    m = pilot.read(ORIGINAL / "manifest.json")
    assert m["content_sha256"] == ORIGINAL_MANIFEST_SHA and len(m["jobs"]) == 730
    previous_root = pilot.ROOT
    try:
        pilot.ROOT = ORIGINAL
        pilot.verify(m)
    finally:
        pilot.ROOT = previous_root
    final = pilot.read(ORIGINAL / "completion.json")
    assert final["status"] == "complete" and final["episodes"] == 730
    scores = pilot.read(ORIGINAL / "scores.json")
    assert scores["status"] == "complete" and len(scores["rows"]) == 730
    assert scores["summary"]["skill"]["correct"] == 192
    assert scores["summary"]["direct"]["correct"] == 163
    for path in [ORIGINAL / "startup.json", ORIGINAL / "ops/startup.json",
                 *sorted((ORIGINAL / "ops").glob("resume*_launch.json"))]:
        value = pilot.read(path)
        if value.get("pid") or value.get("runner_pid"):
            assert not entry.watcher.process_state(value)["alive"]
    return m, scores


def snapshot():
    m, scores = originals()
    hashes = {}
    frozen = pilot.read(ORIGINAL / "predictions_frozen.json")
    assert len(frozen["predictions"]) == 730
    for relative, expected in frozen["predictions"].items():
        path = ORIGINAL / relative
        assert pilot.sha(path) == expected
        hashes[str(path)] = expected
        row = pilot.read(path)
        for rel, digest in row["request_records"].items():
            assert pilot.sha(ORIGINAL / rel) == digest
            hashes[str(ORIGINAL / rel)] = digest
    for job in m["jobs"]:
        path = ORIGINAL / "intents" / (job["operation_id"] + ".json")
        value = pilot.read(path)
        assert value["job_sha256"] == pilot.sha_value(job)
        hashes[str(path)] = pilot.sha(path)
    for relative in ("manifest.json", "completion.json", "scores.json", "predictions_frozen.json",
                     "request_pairing_audit.json", "host_evaluation_binding.json",
                     "requests/state.json", "test_budget/state.json"):
        hashes[str(ORIGINAL / relative)] = pilot.sha(ORIGINAL / relative)
    for relative, digest in {**m["source_files"], **m["driver_files"]}.items():
        hashes[str(ORIGINAL / relative)] = digest
    for path in sorted((ORIGINAL / "ops").glob("*.py")):
        hashes[str(path)] = pilot.sha(path)
    other_path = Path(m["original_budget_root"]) / "state.json"
    hashes[str(other_path)] = pilot.sha(other_path)
    states = {name: json.loads((ORIGINAL / name / "state.json").read_text())
              for name in ("requests", "test_budget")}
    for state in states.values():
        assert len(state["requests"]) == 745 and state["peak_inflight"] <= 4
        assert all(r["status"] in {"completed", "error"} for r in state["requests"])
    other = json.loads(other_path.read_text())
    assert all(r["status"] in {"completed", "error"} for r in other["requests"])
    assert len(other["requests"]) + 745 + 48 <= 20000
    return {"hashes": hashes, "original_requests": 745, "other_requests": len(other["requests"]),
            "original_combined_requests": len(other["requests"]) + 745,
            "original_scores": scores["summary"], "original_results": 730,
            "original_request_records": 1490, "model_requests": 0}


def verify_snapshot(value):
    for path, digest in value["hashes"].items():
        assert pilot.sha(path) == digest, "original_file_changed"


def health():
    h = timeout_transport06.health()
    assert h["status"] == "ready" and h["max_inflight"] == 4
    assert h["source_sha256"] == timeout_extension06.GATEWAY_SOURCE_SHA
    reader = timeout_handling08.health()
    assert reader["status"] == "ready"
    return {k: h[k] for k in ("status", "schema_version", "model", "source_sha256",
                             "max_inflight", "upstream_timeout_s")}


def wire_check(operation_id, count, request):
    manifest = pilot.read(ROOT / "retry_manifest.json")
    mappings = {r["retry_operation_id"]: r for r in manifest["mapping"]}
    assert operation_id in mappings and 1 <= count <= 4
    if count == 1:
        assert hashlib.sha256(request.data).hexdigest() == mappings[operation_id]["first_wire_sha256"]


class CheckedProxy(timeout_transport06.ProxyOpener):
    def __call__(self, request, timeout):
        wire_check(self.operation_id, self.count + 1, request)
        return super().__call__(request, timeout)


def install_runtime(root, offline=False):
    timeout_extension06.apply()
    if offline:
        def fake_forward(request, timeout):
            assert timeout == 1230
            raise TimeoutError("offline_transport_fixture")
        provider.ProxyOpener = lambda op, run, schema: CheckedProxy(
            op, run, schema, secret="offline_transport_fixture", transport=fake_forward)
    else:
        provider.ProxyOpener = CheckedProxy
    pilot.ROOT = root


def audit_result(job, root=ROOT, offline=False):
    path = root / "results/skill" / (job["operation_id"] + ".json")
    row = pilot.read(path)
    assert row["operation_id"] == job["operation_id"] and row["job_sha256"] == pilot.sha_value(job)
    assert row["reference_answer_visible"] is False
    r = row["result"]
    assert r["input_sha256"] == job["input_sha256"] and r["source_split"] == "test"
    assert 1 <= r["request_count"] <= 4
    records = {str(p.relative_to(root)): pilot.sha(p)
               for p in (root / "request_records" / job["operation_id"]).glob("*.json")}
    assert row["request_records"] == records and len(records) == 2 * r["request_count"]
    for index in range(1, r["request_count"] + 1):
        base = root / "request_records" / job["operation_id"] / f"{index:03d}"
        intent = pilot.read(str(base) + ".intent.json")
        response = pilot.read(str(base) + ".response.json")
        identity = response["request_identity"]
        params = identity["parameters"]
        assert params == {"model": "qwen3.8-flash", "max_tokens": 8192, "temperature": 0.0,
                          "seed": 137, "enable_thinking": True, "top_p": None, "response_format": None}
        assert [x["jpeg_sha256"] for x in identity["images_in_order"]] == [x["sha256"] for x in job["images"]]
        assert intent["images"] == job["images"] and intent["reference_answer_visible"] is False
        if response["status"] == "received":
            assert response["response_identity"]["model"] == "qwen3.8-flash"
            assert response["response_identity"]["model_reported"] is True
        else:
            assert index == r["request_count"] and r["status"] == "service_error"
            assert row["status"] == "stopped" and r["answer"] is None
            assert response["status"] == "error" and response["error_type"] == "ServiceUnavailable"
            assert response["model_response"] is None and response["response_identity"] == {}
            if not offline:
                request_id = hashlib.sha256((job["run_identity"] + ":" + job["operation_id"] + ":" + str(index)).encode()).hexdigest()
                proof = timeout_handling08.get("/request-receipt/" + request_id)
                assert proof["status"] == "receipt_available"
                assert proof["intent"]["payload_sha256"] == identity["wire_payload_sha256"]
                assert proof["intent"]["audit_identity"] == {"run_identity": job["run_identity"], "operation_id": job["operation_id"]}
                assert proof["response"]["status"] == "failed"
                assert proof["response"]["automatic_retry"] is False
                target = ROOT / "upstream_error_receipts" / (request_id + ".json")
                if not target.exists():
                    pilot.write(target, proof)
    if r["status"] != "service_error":
        assert row["status"] == "complete"
    mapping = next(r for r in pilot.read(ROOT / "retry_manifest.json")["mapping"]
                   if r["retry_operation_id"] == job["operation_id"])
    first = pilot.read(root / "request_records" / job["operation_id"] / "001.response.json")
    assert first["request_identity"]["wire_payload_sha256"] == mapping["first_wire_sha256"]
    return row


def solve_once(job):
    manifest = pilot.read(ROOT / "retry_manifest.json")
    assert job in manifest["jobs"] and job["arm"] == "skill"
    install_runtime(ROOT)
    try:
        entry.solve(job)
    except RuntimeError as exc:
        path = ROOT / "results/skill" / (job["operation_id"] + ".json")
        if not path.exists() or pilot.read(path)["result"]["status"] != "service_error":
            raise
        assert str(exc) == "operational_failure_no_retry"
    return audit_result(job)


def ledger_audit(active=False):
    m = pilot.read(ROOT / "retry_manifest.json")
    ops = {j["operation_id"] for j in m["jobs"]}
    states = {}
    for name in ("requests", "test_budget"):
        path = ROOT / name / "state.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        assert s["policy"] == m["policy"] and s["peak_inflight"] <= 4
        assert len(s["requests"]) <= 48
        assert [r["request_id"] for r in s["requests"]] == list(range(len(s["requests"])))
        assert all(r["status"] in ({"started", "completed", "error"} if active else {"completed", "error"})
                   for r in s["requests"])
        counts = Counter()
        for r in s["requests"]:
            assert r["context"]["run_identity"] == m["run_identity"] and r["context"]["variant_id"] in ops
            assert r["category"] == "solver"
            counts[r["context"]["variant_id"]] += 1
        assert all(n <= 4 for n in counts.values())
        states[name] = s
    if not active:
        assert len(states) == 2
        def identities(s):
            return Counter(pilot.sha_value([r["context"], r["status"], r.get("usage", {})]) for r in s["requests"])
        assert identities(states["requests"]) == identities(states["test_budget"])
    return states


def prepare():
    assert ROOT.is_dir() and not (ROOT / "retry_manifest.json").exists()
    with ExitStack() as stack:
        locks(stack)
        baseline = snapshot()
        m, _ = originals()
        failures = []
        mapping = []
        for job in m["jobs"]:
            old = pilot.read(ORIGINAL / "results" / job["arm"] / (job["operation_id"] + ".json"))
            if job["arm"] != "skill" or old["result"]["status"] != "service_error":
                continue
            assert old["result"]["request_count"] == 1
            op = pilot.sha_value({"original_operation_id": job["operation_id"], "authorized_scope": RETRY_SCOPE,
                                 "authorization": AUTHORIZATION, "retry_ordinal": 1})
            child = deepcopy(job)
            child["operation_id"] = op
            assert {k: v for k, v in child.items() if k != "operation_id"} == {k: v for k, v in job.items() if k != "operation_id"}
            response = pilot.read(ORIGINAL / "request_records" / job["operation_id"] / "001.response.json")
            failures.append(child)
            mapping.append({"original_operation_id": job["operation_id"], "retry_operation_id": op,
                            "question": job["ordinal"], "original_job_sha256": pilot.sha_value(job),
                            "retry_job_sha256": pilot.sha_value(child),
                            "original_result_sha256": pilot.sha(ORIGINAL / "results/skill" / (job["operation_id"] + ".json")),
                            "first_wire_sha256": response["request_identity"]["wire_payload_sha256"]})
        assert sorted(j["ordinal"] for j in failures) == ORDINALS and len(failures) == 12
        assert not {j["operation_id"] for j in failures} & {j["operation_id"] for j in m["jobs"]}
        pilot.write(ROOT / "retry_manifest.json", {"schema": "harness3d.skill-service-retry/1", "at": pilot.now(),
                    "run_identity": m["run_identity"], "jobs": failures, "mapping": mapping, "policy": m["shared_policy"],
                    "original_manifest_sha256": ORIGINAL_MANIFEST_SHA, "authorization": AUTHORIZATION,
                    "authorized_solver_retries": 12, "automatic_retries": 0, "maximum_model_requests": 48,
                    "reference_answer_visible": False, "online_train_requests": 0,
                    "selection_policy": "only original skill service_error; one fresh solver attempt per selected episode"})
        pilot.write(ROOT / "original_snapshot.json", baseline)
        inputs = {}
        offline = []
        for job in failures:
            problem = pilot.problem_for(job["problem"])
            assert pilot.image_identities(problem) == job["images"]
            prep = Path(job["prepared_root"]) / job["input_sha256"]
            prepared = pilot.load_prepared(prep, problem=problem)
            assert prepared.artifact_sha256 == job["artifact_sha256"]
            for path in [*map(Path, problem.image_paths), prep / "metadata.json", prep / "arrays.npz"]:
                inputs[str(path)] = pilot.sha(path)
            offroot = ROOT / "offline" / job["operation_id"]
            install_runtime(offroot, offline=True)
            try:
                entry.solve(job)
            except RuntimeError as exc:
                assert str(exc) == "operational_failure_no_retry"
            row = audit_result(job, offroot, offline=True)
            assert row["result"]["status"] == "service_error" and row["result"]["request_count"] == 1
            offline.append({"question": job["ordinal"], "first_wire_identity_equal": True,
                            "separate_root": True, "model_requests": 0, "network_requests": 0})
        pilot.ROOT = ORIGINAL
        verify_snapshot(baseline)
        frozen_source = {str(path): pilot.sha(path) for path in ROOT.glob("*.py")}
        pilot.write(ROOT / "preflight.json", {"status": "passed", "at": pilot.now(), "model_requests": 0,
                    "offline_checks": offline, "originals_unchanged": True, "input_hashes": inputs,
                    "retry_manifest_sha256": pilot.sha(ROOT / "retry_manifest.json"),
                    "original_snapshot_sha256": pilot.sha(ROOT / "original_snapshot.json"),
                    "code_hashes": frozen_source, "gateway": health(), "max_workers": 4})
    return {"status": "passed", "selected": len(failures), "ordinals": ORDINALS, "model_requests": 0,
            "offline_checks": len(offline), "original_combined_requests": baseline["original_combined_requests"],
            "maximum_additional_requests": 48, "root": str(ROOT)}


def verify_preflight():
    pre = pilot.read(ROOT / "preflight.json")
    assert pre["status"] == "passed" and pre["model_requests"] == 0
    assert pilot.sha(ROOT / "retry_manifest.json") == pre["retry_manifest_sha256"]
    assert pilot.sha(ROOT / "original_snapshot.json") == pre["original_snapshot_sha256"]
    for name in ("code_hashes", "input_hashes"):
        for path, digest in pre[name].items():
            assert pilot.sha(path) == digest
    verify_snapshot(pilot.read(ROOT / "original_snapshot.json"))
    return pilot.read(ROOT / "retry_manifest.json")


def run():
    with ExitStack() as stack:
        locks(stack)
        m = verify_preflight()
        health()
        assert not (ROOT / "startup.json").exists() and not (ROOT / "intents").exists()
        pilot.write(ROOT / "startup.json", {"status": "started", **process_receipt(), "max_workers": 4,
                    "automatic_retry": False, "selected_episodes": 12, "entry_sha256": pilot.sha(__file__)})
        pending = {}
        iterator = iter(m["jobs"])
        pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"))
        done = []
        def dispatch():
            job = next(iterator, None)
            if job is None:
                return
            pilot.write(ROOT / "intents" / (job["operation_id"] + ".json"),
                        {"job_sha256": pilot.sha_value(job), "at": pilot.now(), "solver_attempt": 1})
            pending[pool.submit(solve_once, job)] = job
        try:
            for _ in range(4):
                dispatch()
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for f in completed:
                    job = pending.pop(f)
                    row = f.result()
                    done.append({"question": job["ordinal"], "status": row["result"]["status"],
                                 "requests": row["result"]["request_count"], "seconds": round(row["seconds"], 2)})
                    ledger_audit(active=True)
                    pilot.update(ROOT / "progress.json", {"status": "running", "at": pilot.now(),
                                 "completed": len(done), "expected": 12, "outcomes": done})
                    print(json.dumps(done[-1]), flush=True)
                    dispatch()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        assert len(done) == 12
        states = ledger_audit()
        for job in m["jobs"]:
            audit_result(job)
        verify_preflight()
        pilot.write(ROOT / "solver_completion.json", {"status": "complete", "at": pilot.now(),
                    "episodes": 12, "solver_attempts": 12, "automatic_retries": 0,
                    "outcomes": done, "model_requests": len(states["requests"]["requests"]),
                    "originals_unchanged": True})


def finish_scores():
    verify_preflight()
    completion = pilot.read(ROOT / "solver_completion.json")
    assert completion["status"] == "complete" and completion["episodes"] == 12
    exit_receipt = pilot.read(ROOT / "exit.json")
    assert exit_receipt["returncode"] == 0
    assert not entry.watcher.process_state(pilot.read(ROOT / "startup.json"))["alive"]
    retry = pilot.read(ROOT / "retry_manifest.json")
    original_m, original_scores = originals()
    children = {r["original_operation_id"]: j for r, j in zip(retry["mapping"], retry["jobs"])}
    merged = [deepcopy(children.get(j["operation_id"], j)) for j in original_m["jobs"]]
    assert len(merged) == 730 and sum(j["arm"] == "direct" for j in merged) == 365
    view = ROOT / "statistics"
    assert not view.exists()
    view.mkdir()
    pilot.write(view / "manifest.json", {"jobs": merged, "original_manifest_sha256": ORIGINAL_MANIFEST_SHA,
                "retry_manifest_sha256": retry["content_sha256"], "label": "Skill服务失败重试后的统计"})
    frozen = {}
    for job in merged:
        srcroot = ROOT if job["operation_id"] in {j["operation_id"] for j in retry["jobs"]} else ORIGINAL
        path = srcroot / "results" / job["arm"] / (job["operation_id"] + ".json")
        if srcroot == ROOT:
            audit_result(job)
        relative = str(path.relative_to(srcroot))
        dest = view / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(path)
        frozen[relative] = pilot.sha(path)
        target = view / "request_records" / job["operation_id"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(srcroot / "request_records" / job["operation_id"], target_is_directory=True)
    pilot.write(view / "predictions_frozen.json", {"predictions": frozen, "questions": 365, "episodes": 730, "at": pilot.now()})
    (view / "host_evaluation_binding.json").write_bytes((ORIGINAL / "host_evaluation_binding.json").read_bytes())
    assert pilot.sha(ORIGINAL / "score.py") == SCORER_SHA
    (view / "score.py").write_bytes((ORIGINAL / "score.py").read_bytes())
    assert pilot.sha(view / "score.py") == SCORER_SHA
    env = {k: v for k, v in os.environ.items() if not any(t in k.upper() for t in ("KEY", "TOKEN", "ENDPOINT", "BASE_URL"))}
    env["PYTHONPATH"] = str(ORIGINAL)
    result = subprocess.run([sys.executable, "-B", str(view / "score.py"), "--final"],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, "isolated_original_scoring_failed"
    scores = pilot.read(view / "scores.json")
    assert scores["status"] == "complete" and len(scores["rows"]) == 730
    assert [r for r in scores["rows"] if r["arm"] == "direct"] == [r for r in original_scores["rows"] if r["arm"] == "direct"]
    before = {r["operation_id"]: r for r in original_scores["rows"]}
    selected = {j["operation_id"] for j in retry["jobs"]}
    for row in scores["rows"]:
        if row["operation_id"] not in selected:
            assert row == before[row["operation_id"]]
    retry_rows = [r for r in scores["rows"] if r["operation_id"] in selected]
    summary = {"status": "complete", "at": pilot.now(), "label": "Skill服务失败重试后的统计",
               "original": original_scores["summary"], "after_retry": scores["summary"],
               "retry_outcomes": retry_rows, "retry_count": 12, "retry_correct": sum(r["score"] for r in retry_rows),
               "retry_status_counts": dict(Counter(r["status"] for r in retry_rows)),
               "additional_model_requests": completion["model_requests"],
               "originals_unchanged": True, "direct_rows_unchanged": True,
               "unselected_rows_unchanged": True, "solver_attempts_per_selected_question": 1,
               "automatic_model_retries": 0, "denominator_per_arm": 365,
               "scorer_sha256": SCORER_SHA, "scores_sha256": pilot.sha(view / "scores.json"),
               "original_manifest_sha256": ORIGINAL_MANIFEST_SHA,
               "retry_manifest_sha256": retry["content_sha256"], "reference_answers_returned": False,
               "model_requests_during_scoring": 0}
    summary["skill_gain_percentage_points"] = (summary["after_retry"]["skill"]["correct"] - 192) / 365 * 100
    summary["skill_minus_direct_percentage_points"] = (summary["after_retry"]["skill"]["correct"] - 163) / 365 * 100
    verify_preflight()
    ledger_audit()
    pilot.write(ROOT / "completion.json", summary)
    return summary


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "prepare":
        print(json.dumps(prepare()))
    elif action == "score":
        print(json.dumps(finish_scores()))
    elif action == "run":
        code = 0
        try:
            run()
        except BaseException as exc:
            code = 2
            pilot.update(ROOT / "failure.json", {"status": "stopped", "error_type": type(exc).__name__,
                         "at": pilot.now(), "automatic_retry": False,
                         "locations": [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                                       for f in traceback.extract_tb(exc.__traceback__)]})
        pilot.update(ROOT / "exit.json", {"returncode": code, "at": pilot.now(), "automatic_retry": False})
        sys.exit(code)
