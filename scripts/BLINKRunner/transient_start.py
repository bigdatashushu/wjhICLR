"""Start a prepared Titan SR run with the existing local credential through SSH stdin."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trial-module", choices=("sr_trial", "sr_retest"), default="sr_trial")
    args = parser.parse_args()
    if not str(args.root).startswith("/var/tmp/harness3d-blink-sr"):
        raise ValueError("unexpected_sr_remote_root")
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project / "BLINK"))
    spec = importlib.util.spec_from_file_location("blink_existing_credential", project / "BLINK/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    credential = module.resolve_api_key(argparse.Namespace(
        api_key_env="DASHSCOPE_API_KEY", key_file=None))
    skill = Path("/Users/bytedance/wangjh/.trae/skills/harness3d-cluster-ops")
    command = [
        "ssh", "-F", str(skill / "assets/ssh_config"), "h3d-titan",
        "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python",
        "-B", str(args.root / "source/scripts/BLINKRunner/sr_start.py"),
        "--root", str(args.root), "--trial-module", args.trial_module]
    env = dict(os.environ)
    for k in list(env):
        if any(t in k.upper() for t in ("API_KEY", "TOKEN", "ENDPOINT", "BASE_URL", "PASSWORD", "SECRET")):
            env.pop(k, None)
    env.update(SSH_ASKPASS="/private/tmp/blink93_ssh_askpass.py",
               SSH_ASKPASS_REQUIRE="force", DISPLAY=":0", BLINK_SSH_CREDENTIAL_INDEX="1")
    child = subprocess.run(command, input=json.dumps({"api_key": credential}).encode(),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=600)
    if credential.encode() in child.stdout or credential.encode() in child.stderr:
        raise ValueError("launcher_output_contains_credentials")
    if child.returncode:
        print(json.dumps({"status": "failed", "returncode": child.returncode,
                          "stdout": child.stdout.decode(errors="replace")}))
        raise SystemExit(1)
    print(child.stdout.decode().strip())


if __name__ == "__main__":
    main()
