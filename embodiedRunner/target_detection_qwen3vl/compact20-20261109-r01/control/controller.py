"""Single run; stop on operational failures, with persistent startup/exit receipts."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    assert not (OUT / "startup.json").exists(), "experiment_already_started"
    frozen = json.loads((OUT / "frozen.json").read_text())
    for name, pin in frozen["snapshot_files"].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == pin
    assert not Path(frozen["run_directory"]).exists(), "run_already_exists"
    sys.path.insert(0, frozen["source_root"])
    from target_detection_qwen3vl.config import load_config
    from target_detection_qwen3vl.detector import Detector
    from target_detection_qwen3vl.runner import suite
    from target_detection_qwen3vl.sources import Sources, implementation_hashes
    from target_detection_r05.__main__ import check
    from target_detection_r05.client import ModelClient
    from target_detection_r05.transport import Session, child_environment
    assert implementation_hashes() == frozen["implementation_hashes"]
    config, tasks = load_config(OUT / "config.json")
    assert [t.as_dict() for t in tasks] == frozen["selected_tasks"]
    sources = Sources(frozen["input_root"], OUT / "inputs/manipulation_assets.json")
    check(config, tasks, sources)
    assert sources.hashes == frozen["source_hashes"]
    assert json.loads((OUT / "verification/preflight.json").read_text())["status"] == "runtime_ready"
    save(OUT / "verification/detector-launch.json", Detector(config).status())
    key_file = ROOT.parent / "qwenkey.md"
    assert key_file.stat().st_mode & 0o077 == 0, "authorization_permissions"
    keys = set(re.findall(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_.+/-]{16,}", key_file.read_text()))
    assert len(keys) == 1, "authorization_identity"
    os.environ["DASHSCOPE_API_KEY"] = keys.pop()
    os.environ["DASHSCOPE_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    client = ModelClient(config["model"])
    client.assert_available()
    save(OUT / "startup.json", {
        "pid": os.getpid(), "created_at": now(),
        "process_created_at": subprocess.check_output(
            ["ps", "-p", str(os.getpid()), "-o", "lstart="], text=True).strip(),
        "command_sha256": hashlib.sha256(json.dumps(sys.argv).encode()).hexdigest(),
        "run_id": frozen["run_id"], "selected": len(tasks), "automatic_retry": False,
        "credential_persistence": False, "reference_answer_visible": False,
        "gpu_assignment": config["runtime"]["gpu"]})
    wake = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
                            env=child_environment(), stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    save(OUT / "controller-status.json", {"phase": "running", "at": now()})
    code = 2
    try:
        def factory(task, folder, journal):
            return Session(config, task, folder, journal, sources, frozen["run_id"])
        summary = suite(tasks, config, sources, client, factory, OUT / "run")
        save(OUT / "suite-exit.json", {"at": now(), "summary": summary})
        os.environ.pop("DASHSCOPE_API_KEY", None)
        os.environ.pop("DASHSCOPE_BASE_URL", None)
        client.key = ""
        save(OUT / "controller-status.json", {"phase": "verifying", "at": now()})
        sys.path.insert(0, str(OUT / "control"))
        from audit import finish
        report = finish()
        code = 0 if report["final_results_available"] else 2
        save(OUT / "controller-status.json",
             {"phase": "completed" if code == 0 else "needs_attention", "at": now(),
              "completed": report["completed"], "selected": len(tasks),
              "successes": report["current_successes"],
              "final_results_available": report["final_results_available"]})
        print(json.dumps({"status": "completed" if code == 0 else "interrupted",
                          "completed": report["completed"], "successes": report["current_successes"],
                          "selected": len(tasks)}), flush=True)
        return code
    except BaseException as exc:
        save(OUT / "controller-status.json", {"phase": "needs_attention", "at": now(),
             "exception_type": type(exc).__name__, "final_results_available": False})
        print(json.dumps({"status": "controller_failed", "exception_type": type(exc).__name__}), flush=True)
        return 2
    finally:
        os.environ.pop("DASHSCOPE_API_KEY", None)
        os.environ.pop("DASHSCOPE_BASE_URL", None)
        client.key = ""
        save(OUT / "exit.json", {"at": now(), "exit_code": code, "automatic_retry": False})
        if wake.poll() is None:
            wake.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
