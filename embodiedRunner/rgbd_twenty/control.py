"""Run each accepted package once, with no retry or repeated action delivery."""
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from rgbd_three_envs.sources import canonical, file_digest
from rgbd_three_envs.transport import child_environment


def remote_audit(phase):
    frozen = json.loads((OUT / "frozen.json").read_text())
    argv = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "UpdateHostKeys=no", "-o", "ConnectTimeout=10", "cvailab@10.10.0.198",
            shlex.join(["/home/cvailab/wangjh/embodiedRunner/venv-household/bin/python", "-B",
                        frozen["remote_root"] + "/rgbd_twenty/remote_audit.py", "--phase", phase])]
    completed = subprocess.run(argv, env=child_environment(), stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, timeout=90, check=False)
    assert completed.returncode == 0, "remote_audit_failed"
    report = json.loads(completed.stdout)
    assert report["status"] == "verified" and report["selected"] == 20
    (OUT / f"server-{phase}.json").write_bytes(canonical(report) + b"\n")
    return report


def main():
    frozen = json.loads((OUT / "frozen.json").read_text())
    assert not (OUT / "controller.json").exists(), "run_already_started"
    remote_audit("before")
    # Check all twenty bindings with their actual interpreter, without launching simulations.
    for package, pin in frozen["packages"].items():
        module = importlib.import_module(package + ".__main__")
        config, tasks = importlib.import_module(package + ".config").load_config(OUT / pin["config"])
        sources = importlib.import_module(package + ".sources").Sources(
            manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")
        module.check(config, tasks, sources)
        transport = importlib.import_module(package + ".transport")
        reports = {task.identity: transport.preflight(config, task, sources) for task in tasks}
        (OUT / f"preflight-{package}.json").write_bytes(canonical({
            "status": "verified", "reports": reports, "model_requests": 0, "simulator_launches": 0}) + b"\n")
    receipt = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
               "pid": os.getpid(), "automatic_retry": False, "selected": 20, "phases": []}
    (OUT / "controller.json").write_bytes(canonical(receipt) + b"\n")
    for package, pin in frozen["packages"].items():
        assert importlib.import_module(package + ".sources").implementation_hashes() == pin["implementation_hashes"]
        assert file_digest(OUT / pin["config"]) == pin["config_sha256"]
        run_dir = ROOT / package / "runs" / pin["run_id"]
        assert not run_dir.exists(), "run_directory_exists"
        argv = [sys.executable, "-B", "-m", package + ".launch_baseline",
                "--config", str(OUT / pin["config"]), "--run-id", pin["run_id"]]
        print(json.dumps({"phase": package, "status": "started", "episodes": len(pin["episodes"])}), flush=True)
        with (OUT / f"launch-{package}.json").open("xb") as output:
            process = subprocess.Popen(argv, cwd=ROOT, env=child_environment(), stdout=output,
                                       stderr=subprocess.DEVNULL)
            code = process.wait()
        summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else None
        receipt["phases"].append({"package": package, "exit_code": code, "summary": summary,
                                  "run_directory": str(run_dir)})
        (OUT / "controller.json").write_bytes(canonical(receipt) + b"\n")
        print(json.dumps(receipt["phases"][-1]), flush=True)
        if code != 0 or summary is None or summary["status"] != "finished":
            receipt["status"] = "stopped_on_operational_failure"
            (OUT / "controller.json").write_bytes(canonical(receipt) + b"\n")
            return 2
    remote_audit("after")
    receipt["status"] = "finished"
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    (OUT / "controller.json").write_bytes(canonical(receipt) + b"\n")
    print(json.dumps({"status": "finished", "selected": 20}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
