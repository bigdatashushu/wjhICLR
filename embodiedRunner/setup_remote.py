"""Install the simulator into this task's independent Linux virtual environment."""
from pathlib import Path
import os
import subprocess
import sys

sys.dont_write_bytecode = True
from bridge import simulator_environment
from protocol import ROOT


def main():
    env = simulator_environment(ROOT)
    for name in list(env):
        if "PROXY" in name.upper():
            env.pop(name)
    for name, folder in (
        ("PIP_CACHE_DIR", "cache/pip"), ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"), ("XDG_STATE_HOME", "state"),
    ):
        path = ROOT / folder
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    venv = ROOT / "venv-nav"
    if not (venv / "bin/python").exists():
        subprocess.run([sys.executable, "-B", "-m", "venv", str(venv)],
                       env=env, cwd=ROOT, check=True)
    subprocess.run([
        str(venv / "bin/python"), "-B", str(ROOT / "ipv4_pip.py"), "--isolated", "install",
        "--index-url", "https://pypi.org/simple",
        "--cache-dir", str(ROOT / "cache/pip"),
        "--timeout", "15", "--retries", "0",
        "-r", str(ROOT / "requirements-nav.txt"),
    ], env=env, cwd=ROOT, check=True)
    subprocess.run([str(venv / "bin/python"), "-B", "-m", "pip", "--isolated", "check"],
                   env=env, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
