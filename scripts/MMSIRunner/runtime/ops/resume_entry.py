"""Continue only never-dispatched episodes with the original frozen solvers."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing
import sys
import traceback
from pathlib import Path

import resume_control as control
pilot = control.pilot
ROOT, OPS = control.ROOT, control.OPS


def solve(job):
    return control.entry.solve(job)


def run():
    manifest = control.manifest_current()
    with control.operator_locks(manifest):
        baseline = pilot.read(control.PREFLIGHT)
        assert baseline["status"] == "passed" and baseline["manifest_sha256"] == control.MANIFEST_SHA
        assert pilot.read(OPS / "resume02_preflight_exit.json")["returncode"] == 0
        assert not control.watcher.process_state(pilot.read(ROOT / "startup.json"))["alive"]
        assert not control.RECOVERY.exists()
        control.retained_audit(baseline, check_cache=True)
        control.ledger_audit(manifest, baseline)
        gateway = control.health()
        jobs = {j["operation_id"]: j for j in manifest["jobs"]}
        remaining = [jobs[op] for op in baseline["pending_never_dispatched"]]
        assert len(remaining) == 523
        for job in remaining:
            assert not (ROOT / "intents" / (job["operation_id"] + ".json")).exists()
            assert not (ROOT / "request_records" / job["operation_id"]).exists()
            assert not (ROOT / "results" / job["arm"] / (job["operation_id"] + ".json")).exists()
        history = control.archive_runtime()
        failure_path, failure_sha = control.materialize_timeout(manifest)
        control.retained_audit(baseline)
        control.ledger_audit(manifest, baseline)
        pilot.write(control.RECOVERY, {
            "status": "resumed_existing_run", "at": pilot.now(),
            "manifest_sha256": control.MANIFEST_SHA, "retained_predictions": 206,
            "timeout_result_path": failure_path, "timeout_result_sha256": failure_sha,
            "timeout_retried": False, "remaining_never_dispatched": 523,
            "request_prefix_count": 217, "historical_runtime_hashes": history,
            "preflight_sha256": pilot.sha(control.PREFLIGHT), "gateway": gateway,
            "max_workers": 4, "max_requests": 1900, "other_experiments_resumed": False,
            "source_frozen_modified": False, "reference_answers_read": False})
        pilot.write(ROOT / "startup.json", {
            **control.process_receipt(), "status": "started", "resume_attempt": 2,
            "source_sha256": pilot.sha(__file__), "manifest_sha256": control.MANIFEST_SHA,
            "max_workers": 4, "automatic_retry": False, "retained_predictions": 206,
            "retained_service_errors": 1, "remaining_episodes": 523})
        pilot.score()
        pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"))
        iterator, pending = iter(remaining), {}
        done = 0

        def progress(status="running"):
            counts = {arm: len(list((ROOT / "results" / arm).glob("*.json")))
                      for arm in ("skill", "direct")}
            pilot.update(ROOT / "progress.json", {
                "status": status, "phase": "test_only", "resume_attempt": 2,
                "completed_episodes": sum(counts.values()), "completed": counts,
                "expected_episodes": 730, "new_completed_episodes": done,
                "retained_service_errors": 1, "pending_workers": len(pending), "at": pilot.now(),
                "request_usage": pilot.governor(ROOT / "requests", manifest["shared_policy"]).summary(),
                "automatic_retry": False})

        def dispatch():
            control.ledger_audit(manifest, baseline, active_ok=True)
            job = next(iterator, None)
            if job is None:
                return
            assert not (ROOT / "request_records" / job["operation_id"]).exists()
            assert not (ROOT / "results" / job["arm"] / (job["operation_id"] + ".json")).exists()
            pilot.write(ROOT / "intents" / (job["operation_id"] + ".json"), {
                "job_sha256": pilot.sha_value(job), "at": pilot.now()})
            pending[pool.submit(solve, job)] = job

        try:
            for _ in range(4):
                dispatch()
            progress()
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                failures = []
                for future in completed:
                    job = pending.pop(future)
                    try:
                        row = future.result()
                        control.result_audit(
                            job, ROOT / "results" / job["arm"] / (job["operation_id"] + ".json"))
                        done += 1
                        print(json.dumps({"event": "resumed_episode", "arm": job["arm"],
                                          "question": job["ordinal"], "status": row["result"]["status"],
                                          "seconds": round(row["seconds"], 2)}), flush=True)
                    except BaseException as exc:
                        failures.append(exc)
                progress()
                if failures:
                    raise failures[0]
                control.retained_audit(baseline)
                control.ledger_audit(manifest, baseline, active_ok=True)
                if done % 4 == 0 or done == 1:
                    pilot.score()
                for _ in completed:
                    dispatch()
                progress()
        except BaseException:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            progress("stopped_requires_diagnosis")
            pilot.score()
            raise
        else:
            pool.shutdown(wait=True)
        assert done == 523
        control.final_audit()


if __name__ == "__main__":
    code = 0
    try:
        run()
    except BaseException as exc:
        code = 2
        # Preserve the first run's runtime receipts in history_run01.
        target = ROOT / "failure.json" if control.RECOVERY.exists() else OPS / "resume02_launch_failure.json"
        pilot.write(target, {
            "status": "stopped", "at": pilot.now(), "error_type": type(exc).__name__,
            "automatic_retry": False, "resume_attempt": 2,
            "locations": [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                          for f in traceback.extract_tb(exc.__traceback__)]})
    target = ROOT / "exit.json" if control.RECOVERY.exists() else OPS / "resume02_launch_exit.json"
    pilot.write(target, {"returncode": code, "at": pilot.now(), "automatic_retry": False, "resume_attempt": 2})
    sys.exit(code)
