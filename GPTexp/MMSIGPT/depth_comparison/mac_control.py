"""Mac-only continuation, exact RGB transfer, and independent post-freeze scoring."""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile

from ...traces import file_digest, read, write
from .continuation import untouched
from .control import SSH, PYTHON
from .pool import read_keys
from .runner import validate, validate_frozen

ROOT = Path(__file__).resolve().parents[3]
PARENT = Path("/Users/bytedance/wangjh/MMSI深度图对比-20261107-服务器失败记录/run")
DESTINATION = Path("/Users/bytedance/wangjh/MMSI深度图对比-20261107")
IMAGE_ROOT = Path("/tmp/h3d-four-smallimage-public-20261031/dataset/MMSI_images")
REFERENCE = "/home/wangy/wangjh/harness3d/four-datasets-20261031/smallimage/host_evaluation/answers.json"
REFERENCE_SHA = "cbd06fa24683a0739e67119ba8d318721bad9ca7c919927cb5a2caabe7144aed"


def transfer():
    parent, frozen, attempted, remaining = untouched(PARENT)
    expected = {}
    for item in remaining:
        for path, receipt in zip(item["problem"]["image_paths"], item["images"]):
            path = Path(path)
            if path.parent != IMAGE_ROOT or not re.fullmatch(r"[a-f0-9]{64}\.png", path.name):
                raise ValueError("unexpected_original_path")
            expected[path.name] = receipt["sha256"]
    if len(expected) != 48 or not DESTINATION.parent.is_dir() or DESTINATION.exists():
        raise ValueError("requires_48_unique_images_and_new_destination")
    command = "tar -czf - -C " + shlex.quote(str(IMAGE_ROOT)) + " " + " ".join(
        shlex.quote(name) for name in sorted(expected))
    result = subprocess.run(SSH + [command], capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError("rgb_transfer_failed")
    originals = {}
    with tarfile.open(fileobj=BytesIO(result.stdout)) as tar:
        for member in tar.getmembers():
            if (member.name not in expected or not member.isfile() or member.issym()
                    or member.islnk() or member.name in originals):
                raise ValueError("unexpected_rgb_archive_member")
            raw = tar.extractfile(member).read()
            if hashlib.sha256(raw).hexdigest() != expected[member.name]:
                raise ValueError("transferred_original_bytes_changed")
            originals[member.name] = raw
    if set(originals) != set(expected):
        raise ValueError("incomplete_original_transfer")
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    for name, raw in originals.items():
        path = IMAGE_ROOT / name
        if path.exists():
            if file_digest(path) != expected[name]:
                raise ValueError("existing_original_bytes_disagree")
        else:
            with path.open("xb") as stream:
                stream.write(raw)
    DESTINATION.mkdir()
    write(DESTINATION / "transfer.json", {
        "original_images": len(originals), "original_bytes": sum(map(len, originals.values())),
        "image_sha256": expected, "parent_run_identity": parent["run_identity"],
        "parent_freeze_identity": frozen["freeze_identity"],
        "excluded_attempted_sample_ids": attempted, "reference_files_copied": False,
        "credentials_copied": False, "model_requests": 0,
    }, exclusive=True)
    print(json.dumps({"images_verified": len(originals), "model_requests": 0}))


def clean_environment():
    env = dict(os.environ)
    for name in list(env):
        if any(part in name.upper() for part in (
                "KEY", "TOKEN", "SECRET", "PASSWORD", "REFERENCE", "GROUND_TRUTH")):
            env.pop(name, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def execute(directory=None):
    directory = Path(directory) if directory else DESTINATION / "run"
    manifest = validate(directory, verify_requests=True)
    available = read_keys(ROOT / "xcodeapi/credentials.local.env")
    keys = {api: available[api] for api in manifest["api_ids"]}
    command = [sys.executable, "-B", "-m", "GPTexp.MMSIGPT.depth_comparison",
               "execute", "--directory", str(directory), "--credentials-stdin"]
    result = subprocess.run(command, input=json.dumps(keys).encode(),
                            env=clean_environment(), cwd=ROOT)
    if result.returncode:
        raise RuntimeError("mac_execution_failed")


def score(directory=None):
    directory = Path(directory) if directory else DESTINATION / "run"
    manifest = validate(directory)
    validate_frozen(directory, manifest)
    if read(directory / "execution.json")["status"] != "complete":
        raise ValueError("do_not_read_reference_after_incomplete_comparison")
    # The bytes never enter online workers or terminal output.
    code = "from pathlib import Path; import sys; sys.stdout.buffer.write(Path(" + repr(REFERENCE) + ").read_bytes())"
    result = subprocess.run(SSH + [shlex.quote(PYTHON) + " -B -c " + shlex.quote(code)],
                            capture_output=True, timeout=60)
    if result.returncode or hashlib.sha256(result.stdout).hexdigest() != REFERENCE_SHA:
        raise ValueError("post_freeze_reference_transfer_or_hash_failed")
    with tempfile.TemporaryDirectory(prefix="mmsi-depth-host-score-") as temporary:
        path = Path(temporary) / "references.json"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(result.stdout)
        command = [sys.executable, "-B", "-m", "GPTexp.MMSIGPT.depth_comparison",
                   "score", "--directory", str(directory), "--references", str(path),
                   "--reference-sha256", REFERENCE_SHA]
        completed = subprocess.run(command, env=clean_environment(), cwd=ROOT)
        if completed.returncode:
            raise RuntimeError("independent_mac_score_failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("transfer", "execute", "score"))
    parser.add_argument("--run-directory")
    try:
        args = parser.parse_args()
        if args.phase == "transfer":
            if args.run_directory:
                raise ValueError("transfer_does_not_accept_run_directory")
            transfer()
        else:
            globals()[args.phase](args.run_directory)
    except Exception:
        print('{"status":"failed","reason_code":"mac_control_phase_failed"}', file=sys.stderr)
        raise SystemExit(2)
