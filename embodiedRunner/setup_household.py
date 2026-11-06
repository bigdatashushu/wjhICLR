"""Task-local Linux dependency installation; no model requests."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def clean():
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in
                      ("TOKEN", "API_KEY", "PASSWORD", "SECRET", "CREDENTIAL", "ENDPOINT", "PROXY"))
           and k not in ("PYTHONPATH", "PYTHONSTARTUP")}
    for name, relative in (("CONDA_PKGS_DIRS", "cache/conda-household"),
                           ("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache/xdg"),
                           ("HF_HOME", "cache/huggingface"), ("TORCH_HOME", "cache/torch"),
                           ("PIP_CACHE_DIR", "cache/pip-household")):
        folder = ROOT / relative
        folder.mkdir(parents=True, exist_ok=True)
        env[name] = str(folder)
    env.update(PYTHONDONTWRITEBYTECODE="1", CONDA_NO_PLUGINS="true",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    os.environ.clear()
    os.environ.update(env)
    addresses = socket.getaddrinfo
    socket.getaddrinfo = lambda host, port, family=0, *a, **kw: addresses(
        host, port, socket.AF_INET if family == 0 else family, *a, **kw)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("conda", "solver", "pip", "xvfb", "font", "habitat-lab",
                                         "assets", "torch", "torch-download"))
    args = parser.parse_args()
    env = clean()
    if args.stage == "conda":
        cpp = ROOT / "runtime/conda-solver/lib/libstdc++.so.6"
        if os.environ.get("LD_PRELOAD") != str(cpp):
            env["LD_PRELOAD"] = str(cpp)
            os.execve(sys.executable, [sys.executable, "-B", __file__, "conda"], env)
        os.environ.pop("CONDA_NO_PLUGINS", None)
        sys.path.insert(0, str(ROOT / "runtime/conda-solver/site-packages"))
        import conda_libmamba_solver.plugin
        from conda.plugins.manager import get_plugin_manager
        get_plugin_manager().register(conda_libmamba_solver.plugin)
        sys.argv = ["conda", "create", "--yes", "--solver", "libmamba", "--override-channels",
                    "-p", str(ROOT / "venv-household"),
                    "-c", "conda-forge", "-c", "aihabitat",
                    "python=3.9", "habitat-sim=0.3.0", "headless", "withbullet",
                    "numpy=1.23.5", "pip", "git-lfs"]
        runpy.run_module("conda", run_name="__main__")
    elif args.stage == "solver":
        import bz2
        import hashlib
        import urllib.request
        from conda_package_handling.api import extract
        url = "https://conda.anaconda.org/conda-forge/noarch/"
        data = None
        for path in (ROOT / "cache/conda-household/cache").glob("*.json"):
            candidate = json.loads(path.read_text())
            if any(r["name"] == "conda-libmamba-solver" and r["version"] == "24.1.0"
                   for r in candidate.get("packages.conda", {}).values()):
                data = candidate
                break
        if data is None:
            data = json.loads(bz2.decompress(urllib.request.urlopen(url + "repodata.json.bz2", timeout=90).read()))
        packages = {**data["packages"], **data["packages.conda"]}
        filename, row = next((n, r) for n, r in packages.items()
                             if r["name"] == "conda-libmamba-solver" and r["version"] == "24.1.0")
        content = urllib.request.urlopen(url + filename, timeout=90).read()
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError("solver_package_hash_mismatch")
        archive = ROOT / "cache" / filename
        archive.write_bytes(content)
        extract(str(archive), dest_dir=str(ROOT / "runtime/conda-solver"))
        cpp = Path("/home/cvailab/anaconda3/pkgs/libstdcxx-14.1.0-hc0a3c3a_1/lib/libstdc++.so.6")
        destination = ROOT / "runtime/conda-solver/lib"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cpp.resolve(), destination / "libstdc++.so.6")
        print(json.dumps({"solver_package": filename, "sha256": row["sha256"]}))
    elif args.stage == "pip":
        subprocess.run([str(ROOT / "venv-household/bin/python"), "-B",
                        str(ROOT / "ipv4_pip.py"), "--isolated", "install", "--retries", "0",
                        "--timeout", "60", "--index-url", "https://pypi.org/simple",
                        "-r", str(ROOT / "requirements-household.txt")], env=env, check=True)
    elif args.stage in ("xvfb", "font"):
        folder = ROOT / "runtime/xvfb-download"
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.run(["apt-get", "download", "xvfb" if args.stage == "xvfb" else "fonts-ubuntu"],
                       cwd=folder, env=env, check=True)
        for package in folder.glob("*.deb"):
            subprocess.run(["dpkg-deb", "-x", str(package), str(ROOT / "runtime/xvfb")],
                           env=env, check=True)
    elif args.stage == "torch-download":
        import hashlib
        import html
        import re
        import urllib.parse
        import urllib.request
        index = urllib.request.urlopen("https://download.pytorch.org/whl/cpu/torch/", timeout=60).read().decode()
        href = next(html.unescape(x) for x in re.findall(r'href="([^"]+)"', index)
                    if "torch-2.4.0%2Bcpu-cp39-cp39-linux_x86_64.whl#" in x)
        url, expected = href.split("#sha256=")
        url = urllib.parse.urljoin("https://download.pytorch.org/whl/cpu/torch/", url)
        archive = ROOT / "cache/torch-2.4.0+cpu-cp39-cp39-linux_x86_64.whl"
        if not archive.is_file():
            partial = archive.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as stream:
                while block := response.read(4 * 1024 * 1024):
                    stream.write(block)
            partial.replace(archive)
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("torch_wheel_identity_mismatch")
        print(json.dumps({"status": "ready", "torch_sha256": expected, "bytes": archive.stat().st_size}))
    elif args.stage == "torch":
        subprocess.run([str(ROOT / "venv-household/bin/python"), "-B", str(ROOT / "ipv4_pip.py"),
                        "--isolated", "install", "--retries", "0", "--timeout", "60",
                        "--index-url", "https://pypi.org/simple",
                        str(ROOT / "cache/torch-2.4.0+cpu-cp39-cp39-linux_x86_64.whl")],
                       env=env, check=True)
    elif args.stage == "habitat-lab":
        import tarfile
        folder = ROOT / "runtime/habitat-lab-v0.3.0"
        folder.mkdir(parents=True, exist_ok=True)
        with tarfile.open(ROOT / "habitat-lab-v0.3.0.tar.gz") as tar:
            for member in tar:
                path = member.name.partition("/")[2]
                if not path or not member.isfile():
                    continue
                target = (folder / path).resolve()
                if not target.is_relative_to(folder):
                    raise ValueError("archive_path_escaped")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
        subprocess.run([str(ROOT / "venv-household/bin/python"), "-B", str(ROOT / "ipv4_pip.py"),
                        "--isolated", "install", "--no-deps", "--retries", "0", "--timeout", "60",
                        "--index-url", "https://pypi.org/simple", "-e", str(folder / "habitat-lab")],
                       env=env, check=True)
    elif args.stage == "assets":
        import hashlib
        import tarfile
        archive = ROOT / "habitat-official-assets.tar"
        receipt = json.loads((ROOT / "habitat-official-assets-receipt.json").read_text())
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if archive.stat().st_size != receipt["archive_bytes"] or digest.hexdigest() != receipt["archive_sha256"]:
            raise ValueError("assets_archive_identity_mismatch")
        env_root = ROOT / "upstream-household/embodiedbench/envs/eb_habitat"
        with tarfile.open(archive) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                target = (env_root / member.name).resolve()
                if not target.is_relative_to(env_root):
                    raise ValueError("assets_archive_path_escaped")
                target.parent.mkdir(parents=True, exist_ok=True)
                content = tar.extractfile(member).read()
                if target.is_file() and target.read_bytes() != content:
                    raise ValueError("existing_asset_identity_changed")
                target.write_bytes(content)
        destination = ROOT / "upstream-household/embodiedbench/envs/eb_habitat/data"
        destination.mkdir(parents=True, exist_ok=True)
        env["PATH"] = str(ROOT / "venv-household/bin") + os.pathsep + env.get("PATH", "")
        os.environ.update(env)
        # The three HF repositories above retain the official requested tags.
        sys.argv = ["datasets_download", "--uids", "rearrange_pick_dataset_v0",
                    "rearrange_dataset_v1", "--no-prune",
                    "--no-replace", "--data-path", str(destination)]
        original_check_call = subprocess.check_call
        def bounded_download(command, *a, **kw):
            if command[0] == "wget":
                command = [command[0], "-4", "--timeout=60", "--tries=1", *command[1:]]
            return original_check_call(command, *a, **kw)
        subprocess.check_call = bounded_download
        runpy.run_path(str(ROOT / "runtime/habitat-sim-datasets_download-v0.3.0.py"), run_name="__main__")


if __name__ == "__main__":
    main()
