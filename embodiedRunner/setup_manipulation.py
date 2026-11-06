"""Stage and install a separate Linux manipulation runtime without service auth."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import urllib.request
import venv

BASE = Path("/home/cvailab/wangjh/embodiedRunner")
PYREP_REVISION = "8f420be8064b1970aae18a9cfbc978dfb15747ef"
YOLO_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
SIM_URL = ("https://downloads.coppeliarobotics.com/V4_1_0/"
           "CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz")
PACKAGES = {
    "numpy": "1.26.4", "scipy": "1.15.3", "Pillow": "11.3.0", "cffi": "1.17.1",
    "gymnasium": "1.1.1", "ultralytics": "8.3.253", "natsort": "8.4.0",
    "opencv-python-headless": "4.11.0.86",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def transfer():
    from action_deploy import ssh
    from manipulation_protocol import verify_sources
    from protocol import ROOT
    verify_sources()
    if digest(ROOT / "cache/yolo11n.pt") != YOLO_SHA256:
        raise ValueError("yolo_weight_changed")
    archive = ROOT / "tmp/manipulation-runtime-stage.tar"
    catalog = json.loads((ROOT / "manipulation_assets.json").read_bytes())
    manifest = json.loads((ROOT / "manipulation_sources.json").read_bytes())
    with tarfile.open(archive, "w") as tar:
        for row in manifest["files"]:
            tar.add(ROOT / "upstream-manipulation" / row["path"],
                    arcname="upstream-manipulation/" + row["path"], recursive=False)
        for row in catalog["files"]:
            p = ROOT / "cache/manipulation-assets" / row["path"]
            if digest(p) != row["sha256"]:
                raise ValueError("dataset_asset_changed")
            tar.add(p, arcname="manipulation-runtime/dataset/" + row["path"], recursive=False)
        for name, source in (
                ("setup_manipulation.py", ROOT / "setup_manipulation.py"),
                ("pyrep.tar.gz", ROOT / "cache/pyrep-8f420be.tar.gz"),
                ("yolo11n.pt", ROOT / "cache/yolo11n.pt"),
                ("manipulation_sources.json", ROOT / "manipulation_sources.json"),
                ("manipulation_assets.json", ROOT / "manipulation_assets.json")):
            tar.add(source, arcname="manipulation-runtime/" + name, recursive=False)
    bootstrap = """import sys,pathlib,tarfile
b=pathlib.Path(sys.argv[1])
if not b.is_dir(): raise ValueError('base_missing')
for name in ('upstream-manipulation','manipulation-runtime'):
    if (b/name).exists(): raise ValueError('destination_exists')
for name in ('upstream-manipulation','manipulation-runtime'):
    (b/name).mkdir(mode=0o700)
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as t:
    for m in t:
        p=b/m.name
        if not m.isfile() or not p.resolve().is_relative_to(b.resolve()):
            raise ValueError('invalid_archive')
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f: f.write(t.extractfile(m).read())
print('{"status":"staged","existing_runtime_modified":false}')
"""
    command = shlex.join(["/home/cvailab/anaconda3/envs/skill3d-exp/bin/python", "-B",
                          "-c", bootstrap, str(BASE)])
    with archive.open("rb") as stream:
        subprocess.run(ssh(command), stdin=stream, check=True, timeout=120)


def extract(archive, destination):
    destination.mkdir(exist_ok=False)
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter="data")


def install():
    runtime = BASE / "manipulation-runtime"
    if Path(__file__).resolve().parent != runtime or (runtime / "runtime.json").exists():
        raise ValueError("install_location_or_receipt")
    logs = runtime / "setup-logs"
    logs.mkdir(exist_ok=True)
    addresses = socket.getaddrinfo
    socket.getaddrinfo = lambda host, port, family=0, *a, **kw: addresses(
        host, port, socket.AF_INET if family == 0 else family, *a, **kw)
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG") if k in os.environ}
    env.update(COPPELIASIM_ROOT=str(runtime / "coppeliasim"),
               LD_LIBRARY_PATH=str(runtime / "coppeliasim"),
               QT_QPA_PLATFORM_PLUGIN_PATH=str(runtime / "coppeliasim"),
               PIP_CONFIG_FILE="/dev/null", PIP_DISABLE_PIP_VERSION_CHECK="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONNOUSERSITE="1")

    def run(name, command):
        log = logs / (name + ".log")
        if log.exists():
            number = next(n for n in range(2, 100) if not (logs / f"{name}-{n:02d}.log").exists())
            log = logs / f"{name}-{number:02d}.log"
        with log.open("xb") as output:
            result = subprocess.run(command, cwd=runtime, env=env,
                                    stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("install_failed:" + name)
        print(json.dumps({"stage": name, "status": "completed"}), flush=True)

    archive = runtime / "coppeliasim.tar.xz"
    if not archive.exists():
        partial = runtime / "coppeliasim.tar.xz.partial"
        request = urllib.request.Request(SIM_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, 8 * 1024**2)
        if partial.stat().st_size != 156384828:
            raise ValueError("simulator_archive_size_changed")
        partial.rename(archive)
    if not (runtime / "coppeliasim").exists():
        extract(archive, runtime / "sim-extract")
        (runtime / "sim-extract/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04").rename(runtime / "coppeliasim")
        (runtime / "sim-extract").rmdir()
    if not (runtime / "PyRep").exists():
        extract(runtime / "pyrep.tar.gz", runtime / "pyrep-extract")
        (runtime / "pyrep-extract" / ("PyRep-" + PYREP_REVISION)).rename(runtime / "PyRep")
        (runtime / "pyrep-extract").rmdir()
    if not (runtime / "venv").exists():
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(runtime / "venv")
    python = str(runtime / "venv/bin/python")
    run("dependencies", [python, "-B", str(BASE / "ipv4_pip.py"), "--isolated", "install",
                         "--retries", "0", "--timeout", "60", "--index-url", "https://pypi.org/simple",
                         *[name + "==" + version for name, version in PACKAGES.items()]])
    run("pyrep", [python, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
                  "-e", str(runtime / "PyRep")])
    official = BASE / "upstream-manipulation/embodiedbench/envs/eb_manipulation"
    run("amsolver", [python, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
                     "-e", str(official)])
    # Official custom-data Lua must be copied after PyRep's build overwrites it.
    shutil.copyfile(official / "simAddOnScript_PyRep.lua",
                    runtime / "coppeliasim/simAddOnScript_PyRep.lua")
    run("dependency-imports", [python, "-B", "-c",
        "import sys;sys.path.insert(0," + repr(str(official)) + ");"
        "import numpy,scipy,open3d,cv2,torch,torchvision,pyrep,amsolver,ultralytics,gymnasium;"
        "from scipy.spatial.transform import Rotation;"
        "assert numpy.__version__=='1.26.4';"
        "assert numpy.isfinite(Rotation.from_euler('xyz',[0,0,0]).as_quat()).all()"])
    versions = json.loads(subprocess.check_output([python, "-B", "-c",
        "import importlib.metadata as m,json;"
        "print(json.dumps({n:m.version(n) for n in " +
        repr(list(PACKAGES) + ["torch", "torchvision", "open3d", "pyquaternion", "PyRep", "amsolver"]) +
        "}))"], env=env))
    pinned = []
    roots = [runtime / name for name in ("coppeliasim", "PyRep", "venv/lib")]
    paths = [p for root in roots for p in root.rglob("*")
             if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts
             and p.suffix != ".pyc"]
    paths += [runtime / "yolo11n.pt", runtime / "pyrep.tar.gz", archive]
    for path in sorted(paths):
        pinned.append({"path": str(path.relative_to(runtime)), "sha256": digest(path),
                       "bytes": path.stat().st_size})
    receipt = {
        "schema": "embodied-manipulation-runtime/1", "python": sys.version.split()[0],
        "coppeliasim_version": "4.1.0", "coppeliasim_archive_sha256": digest(archive),
        "pyrep_revision": PYREP_REVISION, "yolo_sha256": YOLO_SHA256,
        "packages": versions, "pinned_files": pinned,
        "official_sources_sha256": hashlib.sha256(wire(json.loads(
            (runtime / "manipulation_sources.json").read_bytes()))).hexdigest(),
        "assets_sha256": hashlib.sha256(wire(json.loads(
            (runtime / "manipulation_assets.json").read_bytes()))).hexdigest(),
        "system_rc_modified": False, "model_requests": 0, "simulator_launches": 0,
    }
    with (runtime / "runtime.json").open("xb") as out:
        out.write(wire(receipt) + b"\n")
    print(json.dumps({"status": "installed", "pinned_files": len(pinned),
                      "runtime_sha256": hashlib.sha256(wire(receipt)).hexdigest()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if args.transfer == args.install:
        parser.error("select one operation")
    transfer() if args.transfer else install()
