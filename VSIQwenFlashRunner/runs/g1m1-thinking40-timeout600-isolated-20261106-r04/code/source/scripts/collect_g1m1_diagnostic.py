#!/usr/bin/env python3
"""Collect finished native JSON artifacts with a byte-verified transfer receipt."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile


REMOTE_PROGRAM = r'''
import ctypes,hashlib,json,os,select,shlex,shutil,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve()
partial=sys.argv[2].startswith("partial")
if sys.argv[2] in ("wait","partial_wait"):
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.inotify_init1(os.O_NONBLOCK)
    assert fd>=0
    assert libc.inotify_add_watch(fd,os.fsencode(root),0x80|0x8|0x100)>=0
    while not ((root/"recovery_receipt.json").exists() if partial else (root/"exit.json").exists()):
        select.select([fd],[],[])
        os.read(fd,65536)
    os.close(fd)
summary=json.loads((root/"system/summary.json").read_text()) if (root/"system/summary.json").exists() else {}
requests=json.loads((root/"requests/state.json").read_text())
completed=len(list((root/"system/results").glob("*.json")))
if partial:
    assert completed<=40
    assert all(row["status"] in ("completed","error") for row in requests["requests"])
    assert (root/"system_exit.json").exists() and (root/"recovery_receipt.json").exists()
    if len(sys.argv)>3 and sys.argv[3]:
        extra=Path(sys.argv[3]).resolve()
        assert json.loads((extra/"exit.json").read_text())["status"] in ("complete","stopped")
        destination=root/"direct_continuation"
        assert not destination.exists()
        destination.mkdir()
        for name in ("diagnostic.json","direct_start.json","direct_receipt.json","direct_health.json",
                     "launcher_start.json","exit.json","code/code_identity.json"):
            if (extra/name).is_file():
                (destination/name).parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(extra/name,destination/name)
        shutil.copytree(extra/"direct",destination/"direct")
else:
    assert summary["status"]=="complete" and summary["completed_variants"]==40
    assert completed==40
    assert all(row["status"]=="completed" for row in requests["requests"])
    assert (root/"solve_receipt.json").exists() and (root/"exit.json").exists()
comparison=json.loads((root/"diagnostic.json").read_text()).get("comparison")
if comparison and not partial:
    assert comparison["both_enable_thinking"] is True
    if (root/"pair_completion.json").exists():
        pair=json.loads((root/"pair_completion.json").read_text())
        assert pair["status"]=="complete" and pair["system_questions"]==pair["direct_questions"]==40
        assert len(list((root/"direct/responses").glob("*.json")))==pair["original_direct_questions"]
        assert len(list((root/"direct_continuation/direct/responses").glob("*.json")))==pair["continuation_direct_questions"]
    else:
        assert json.loads((root/"direct_receipt.json").read_text())["status"]=="complete"
        assert len(list((root/"direct/responses").glob("*.json")))==40
        assert json.loads((root/"exit.json").read_text())["status"]=="complete"
files=[]
for directory in ("system","direct","requests","seed_bundle","direct_continuation","startup_history"):
    files.extend(p for p in (root/directory).rglob("*")
                 if p.is_file() and p.suffix in (".json",".jsonl")
                 and not any(part in ("work","recon","offline_case_inputs")
                             for part in p.relative_to(root).parts))
for name in ("diagnostic.json","solve_start.json","solve_receipt.json",
             "solve_health.json","exit.json","execution_note.json","code/code_identity.json",
             "direct_start.json","direct_receipt.json","direct_health.json",
             "pair_start.json","system_exit.json","direct_exit.json","pair_completion.json",
             "deployment_receipt.json","thinking_startup_audit.json","recovery_start.json",
             "recovery_process.json","recovery_receipt.json"):
    if (root/name).is_file():
        files.append(root/name)
secrets=[]
env_file=Path("/home/wangy/wangjh/harness3d/precompute-20261022-r01/.transfer.env")
for line in env_file.read_text().splitlines():
    words=shlex.split(line)
    if words and words[0]=="export":
        words=words[1:]
    if len(words)==1 and "=" in words[0]:
        key,value=words[0].split("=",1)
        if any(marker in key.upper() for marker in ("TOKEN","PASSWORD","SECRET","KEY")) and len(value)>=8:
            secrets.append(value.encode())
records=[]
for path in sorted(set(files)):
    digest=hashlib.sha256()
    carry=b""
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
            assert not any(value in carry+block for value in secrets),"credential_in_artifact"
            carry=(carry+block)[-max([len(value) for value in secrets]+[1]):]
    records.append({"path":str(path.relative_to(root)),"bytes":path.stat().st_size,
                    "sha256":digest.hexdigest()})
receipt={"schema":"vsi-g1m1-native-transfer/1","observed_at":time.time(),
         "diagnostic_identity":json.loads((root/"diagnostic.json").read_text())["identity_sha256"],
         "status":"incomplete" if partial else "complete",
         "native_completed_questions":completed,
         "requests_completed":sum(row["status"]=="completed" for row in requests["requests"]),
         "request_errors":sum(row["status"]=="error" for row in requests["requests"]),
         "credential_value_scan":"passed","files":records}
(root/"native_transfer_manifest.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
names=[row["path"] for row in records]+["native_transfer_manifest.json"]
raise SystemExit(subprocess.run(["tar","-czf","-","-C",str(root),"--",*names]).returncode)
'''


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--ssh-helper", type=Path, required=True)
    parser.add_argument("--wait-for-completion", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--continuation-root")
    args = parser.parse_args()
    root = args.run_root.resolve()
    assert root.is_dir() and args.ssh_helper.is_file()
    archive = root / "native_artifacts.tar.gz"
    assert not archive.exists(), "retain previous collection and choose a separate archive"
    remote_command = (
        "python3 -c " + shlex.quote(REMOTE_PROGRAM) + " " + shlex.quote(args.remote_root)
        + " " + (("partial_wait" if args.wait_for_completion else "partial")
                 if args.allow_incomplete else ("wait" if args.wait_for_completion else "now"))
        + " " + shlex.quote(args.continuation_root or ""))
    with archive.open("xb") as stream:
        result = subprocess.run(
            [sys.executable, str(args.ssh_helper), remote_command], stdout=stream)
    assert result.returncode == 0, "native collection failed; partial archive retained"
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            path = root / member.name
            assert not Path(member.name).is_absolute() and ".." not in Path(member.name).parts
            assert path.resolve().is_relative_to(root) and member.isfile()
            with bundle.extractfile(member) as source:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as destination:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(block)
    manifest = json.loads((root / "native_transfer_manifest.json").read_text())
    experiment = json.loads((root / "diagnostic.json").read_text())
    assert manifest["diagnostic_identity"] == experiment["identity_sha256"]
    for row in manifest["files"]:
        path = root / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert digest_file(path) == row["sha256"], "native artifact content differs"
    receipt = {
        "schema": "vsi-g1m1-local-collection/1", "status": "passed",
        "diagnostic_identity": manifest["diagnostic_identity"],
        "experiment_status": manifest.get("status", "complete"),
        "native_completed_questions": manifest["native_completed_questions"],
        "request_errors": manifest.get("request_errors", 0),
        "verified_files": len(manifest["files"]),
        "uncompressed_bytes": sum(row["bytes"] for row in manifest["files"]),
        "archive": archive.name, "archive_bytes": archive.stat().st_size,
        "archive_sha256": digest_file(archive),
        "credential_value_scan": manifest["credential_value_scan"],
    }
    (root / "collection_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
