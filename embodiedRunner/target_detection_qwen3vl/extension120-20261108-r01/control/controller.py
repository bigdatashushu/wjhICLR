"""A unique serial controller; continue never-started tasks after a recorded stop."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys

TMP = Path(__file__).parent
ROOT = Path("/Users/bytedance/wangjh/wjhICLR/embodiedRunner")
OUT = ROOT / "target_detection_qwen3vl/extension120-20261108-r01"
PYTHON = str(ROOT.parent / ".venv/bin/python")


def load(path):
    return json.loads(Path(path).read_text())


frozen = load(OUT / "frozen.json")
sys.path.insert(0, frozen["source_root"])
from target_detection_qwen3vl.config import load_config
from target_detection_qwen3vl.detector import Detector
from target_detection_qwen3vl.sources import Sources, canonical, file_digest, implementation_hashes
from target_detection_r05.__main__ import check
from target_detection_r05.lifecycle import remote
from target_detection_r05.protocol import Task
from target_detection_r05.transport import child_environment, preflight
import audit
import prepare


def save(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def verify_state(config, metadata, suffix="before"):
    state = remote(config, prepare.state_code(), timeout=60)
    assert not state["owned_trial_workers"], "owned_worker_cleanup_required"
    assert state["worker_implementation_hashes"] == {
        n.split("/", 1)[1]: h for n, h in frozen["implementation_hashes"].items()
        if n.startswith("target_detection_r05/")}
    assert state["disk_free_bytes"] >= 2 * 1024**3, "runtime_disk_headroom_low"
    save(metadata / ("server-" + suffix + ".json"), state)
    save(metadata / ("detector-" + suffix + ".json"), Detector(config).status())


def launch(metadata):
    receipt = frozen if metadata == OUT else load(metadata / "prepared.json")
    run_id = receipt["run_id"]
    audit.preservation()
    assert not (metadata / "startup.json").exists() and not (metadata / "run").exists()
    config = load(metadata / "config.json")
    assert all(Task(**t).directory not in audit.started() for t in config["episodes"])
    key_file = ROOT.parent / "qwenkey.md"
    assert key_file.stat().st_mode & 0o077 == 0
    keys = set(re.findall(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_.+/-]{16,}", key_file.read_text()))
    assert len(keys) == 1
    os.environ["DASHSCOPE_API_KEY"] = keys.pop()
    os.environ["DASHSCOPE_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    save(metadata / "startup.json", {
        "pid": os.getpid(), "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id, "selected": len(config["episodes"]), "automatic_retry": False,
        "reference_answer_visible": False, "credential_persistence": False})
    code = 2
    try:
        from target_detection_qwen3vl.__main__ import main
        sys.argv = ["target_detection_qwen3vl", "run", "--config", str(metadata / "config.json"),
                    "--run-id", run_id, "--source-root", frozen["local_input_source_root"],
                    "--manipulation-catalog", frozen["local_manipulation_catalog"]]
        code = main()
        return code
    finally:
        os.environ.pop("DASHSCOPE_API_KEY", None)
        os.environ.pop("DASHSCOPE_BASE_URL", None)
        save(metadata / "exit.json", {"status": "exited", "exit_code": code,
                                     "at": datetime.now(timezone.utc).isoformat(), "automatic_retry": False})


def prepare_remaining():
    assert all((metadata / "exit.json").exists() for metadata, _ in audit.roots())
    audit.preservation()
    started = audit.started()
    assert started == set(audit.folders()), "unfinished_started_episode_requires_resolution"
    config = load(OUT / "config.json")
    config["episodes"] = [t for t in config["episodes"] if Task(**t).directory not in started]
    assert config["episodes"]
    number = len(audit.roots())
    metadata = OUT / "continuations" / f"remaining{number:03d}"
    assert not metadata.exists()
    metadata.mkdir(parents=True)
    save(metadata / "config.json", config)
    (metadata / "config.json").chmod(0o444)
    config, tasks = load_config(metadata / "config.json")
    sources = Sources(frozen["local_input_source_root"], frozen["local_manipulation_catalog"])
    save(metadata / "offline.json", check(config, tasks, sources))
    verify_state(config, metadata)
    reports = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(preflight, config, t, sources): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            reports[task.identity] = future.result()
    save(metadata / "preflight.json", {"reports": reports, "model_requests": 0, "simulator_launches": 0})
    run_id = frozen["run_id"] + f"-remaining{number:03d}"
    (metadata / "run").symlink_to(Path(frozen["source_root"]) / "target_detection_qwen3vl/runs" / run_id,
                                target_is_directory=True)
    save(metadata / "prepared.json", {
        "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": file_digest(metadata / "config.json"),
        "selected": len(tasks), "never_started_tasks_only": True, "failed_task_repeated": False,
        "automatic_retry": False, "reference_answer_visible": False,
        "planning_model_request_cap": sum(config["max_requests"][t.environment] for t in tasks)})
    return metadata


def wait_segment(child, metadata, config):
    audited, retired = set(), set()
    retirement_number = 0
    run_id = frozen["run_id"] if metadata == OUT else load(metadata / "prepared.json")["run_id"]
    package = Path(frozen["source_root"]) / "target_detection_qwen3vl"
    while True:
        completed = {name for name, folder in audit.folders().items()
                     if str(folder).startswith(str(metadata / "run") + "/")}
        fresh = completed - audited
        if fresh:
            audit.completed_audits(fresh)
            audited.update(fresh)
            save(OUT / "progress.json", {
                "updated_at": datetime.now(timezone.utc).isoformat(), **audit.status()})
        caches = {n for n in audited - retired if n.startswith(("alfred-", "navigation-"))}
        if caches:
            receipt = remote(config, (TMP / "retire_cache_remote.py").read_text(),
                             canonical({"work_root": config["runtime"]["work_root"], "run_id": run_id,
                                        "completed_tasks": sorted(caches)}), timeout=45)
            retirement_number += 1
            save(metadata / f"cache-retirement-{retirement_number:03d}.json", receipt)
            retired.update(caches - set(receipt["deferred_active_tasks"]))
        code = child.poll()
        if code is not None:
            return code
        candidates = [metadata, package, package / "runs", metadata / "run"]
        if (metadata / "run").exists():
            candidates += [p for p in (metadata / "run").iterdir() if p.is_dir() and p.name not in completed]
        fds = [os.open(p, os.O_RDONLY) for p in candidates if p.is_dir()]
        queue = select.kqueue()
        try:
            queue.control([select.kevent(fd, filter=select.KQ_FILTER_VNODE,
                                        flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                                        fflags=select.KQ_NOTE_WRITE) for fd in fds], 0, 0)
            queue.control(None, len(fds), 50)
        finally:
            queue.close()
            for fd in fds:
                os.close(fd)


def control():
    with (OUT / "controller.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (OUT / "controller-startup.json").exists()
        save(OUT / "controller-startup.json", {
            "pid": os.getpid(), "created_at": datetime.now(timezone.utc).isoformat(),
            "selected": 120, "serial": True, "failed_task_retry": False, "credential_persistence": False})
        metadata = OUT
        while True:
            save(OUT / "controller-status.json", {
                "phase": "running", "metadata": str(metadata),
                "completed_before_segment": len(audit.folders()),
                "updated_at": datetime.now(timezone.utc).isoformat()})
            config = load(metadata / "config.json")
            verify_state(config, metadata / "verification" if metadata == OUT else metadata, "launch")
            with (metadata / "run.log").open("xb") as log:
                child = subprocess.Popen([PYTHON, "-B", __file__, "launch", str(metadata)],
                                         cwd=metadata, env=child_environment(), stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=log, close_fds=True)
            save(metadata / "controller-child.json", {"pid": child.pid, "run_id":
                frozen["run_id"] if metadata == OUT else load(metadata / "prepared.json")["run_id"]})
            code = wait_segment(child, metadata, config)
            assert (metadata / "exit.json").exists(), "launch_process_did_not_record_exit"
            assert (metadata / "run/summary.json").exists(), "suite_did_not_produce_summary"
            audit.completed_audits()
            if len(audit.folders()) == 120:
                verify_state(config, OUT / "verification", "after")
                result = audit.finish()
                save(OUT / "controller-status.json", {
                    "phase": "finished", "completed": 120,
                    "updated_at": datetime.now(timezone.utc).isoformat()})
                print(canonical(result).decode(), flush=True)
                return 0
            assert code == 2, "incomplete_normal_exit"
            save(OUT / "controller-status.json", {
                "phase": "preparing_never_started", "completed": len(audit.folders()),
                "updated_at": datetime.now(timezone.utc).isoformat()})
            metadata = prepare_remaining()


if __name__ == "__main__":
    try:
        if sys.argv[1] == "launch":
            raise SystemExit(launch(Path(sys.argv[2])))
        raise SystemExit(control())
    except Exception as exc:
        value = {"phase": "needs_attention", "exception_type": type(exc).__name__,
                 "error_code": getattr(exc, "code", None),
                 "assertion_code": str(exc) if isinstance(exc, AssertionError) else None,
                 "updated_at": datetime.now(timezone.utc).isoformat()}
        if sys.argv[1] != "launch":
            save(OUT / "controller-status.json", value)
        print(canonical(value).decode(), flush=True)
        raise SystemExit(2)
