"""Task-scoped Titan deployment and stdin-only authentication for this paired run."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

from ...traces import file_digest, read, write
from .pool import read_keys
from .runner import code_identity

LOCAL = Path(__file__).resolve().parent
WORK = LOCAL / "work-20261107"
GPTEXP = LOCAL.parents[1]
RUNTIME = "/home/wangy/wangjh/harness3d/GPTexp/mmsi-depth-runtime-20261107-r01"
RUN = RUNTIME + "/run"
PYTHON = "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python"
TOKENIZER = (
    "/var/folders/ss/m6nf5tl91mq0d8wk7x_9ll4w0000gn/T/"
    "mmsigpt-prompt-tokenizer-9t47h_yb/o200k_base.tiktoken"
)
SSH = [
    "ssh", "-F", "/Users/bytedance/wangjh/.trae/skills/harness3d-cluster-ops/assets/ssh_config",
    "-S", "/Users/bytedance/.cache/harness3d-ssh/mmsi-depth-20261107",
    "-T", "-o", "BatchMode=yes", "h3d-titan",
]


def remote(code, *, timeout=600):
    result = subprocess.run(SSH + [PYTHON + " -B -"], input=code.encode(),
                            capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError("remote_operation_failed")
    return result.stdout


def deploy():
    # Parent existence is checked before creating this new isolated runtime.
    remote(
        "from pathlib import Path\n"
        f"root=Path({RUNTIME!r})\n"
        "assert root.parent.is_dir() and not root.exists()\n"
        "root.mkdir()\n"
        "(root/'code').mkdir()\n"
        "(root/'deps').mkdir()\n"
        "(root/'wheels').mkdir()\n")
    with tempfile.TemporaryDirectory(prefix="mmsigpt-depth-bundle-") as temporary:
        archive = Path(temporary) / "public.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for relative in code_identity():
                tar.add(GPTEXP / relative, arcname="code/GPTexp/" + relative)
            tar.add(WORK / "selection.json", arcname="selection.json")
            tar.add(TOKENIZER, arcname="o200k_base.tiktoken")
        for value in read_keys(GPTEXP.parent / "xcodeapi/credentials.local.env").values():
            with tarfile.open(archive) as tar:
                if any(value.encode() in tar.extractfile(member).read()
                       for member in tar.getmembers() if member.isfile()):
                    raise ValueError("credential_in_public_bundle")
        with archive.open("rb") as stream:
            result = subprocess.run(SSH + ["tar -xzf - -C " + shlex.quote(RUNTIME)],
                                    stdin=stream, capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError("public_bundle_transfer_failed")
        write(WORK / "deployment.json", {
            "runtime": RUNTIME, "run": RUN, "public_archive_sha256": file_digest(archive),
            "source_files": code_identity(), "credentials_copied": False,
            "reference_files_copied": False, "model_requests": 0,
        }, exclusive=True)
    repair_tokenizer()


def repair_tokenizer():
    # Titan uses glibc 2.27; this wheel supports glibc >=2.17. The o200k
    # vocabulary is pinned independently, so text counting remains fixed.
    url = ("https://files.pythonhosted.org/packages/01/c4/"
           "c4a4360de845217b6aa9709c15773484b50479f36bb50419c443204e5de9/"
           "tiktoken-0.8.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl")
    wheel_sha = "9fb0e352d1dbe15aba082883058b3cce9e48d33101bdaac1eccf66424feb5b47"
    with urllib.request.urlopen(url, timeout=40) as response:
        raw = response.read()
    import hashlib
    if hashlib.sha256(raw).hexdigest() != wheel_sha:
        raise ValueError("tokenizer_wheel_hash_changed")
    code = (
        "import hashlib,io,json,sys,zipfile\nfrom pathlib import Path\n"
        f"root=Path({RUNTIME!r})\n"
        "assert root.is_dir() and not (root/'run/manifest.json').exists()\n"
        "data=sys.stdin.buffer.read()\n"
        f"assert hashlib.sha256(data).hexdigest()=={wheel_sha!r}\n"
        "target=root/'deps'\n"
        "assert target.is_dir()\n"
        "with zipfile.ZipFile(io.BytesIO(data)) as wheel:\n"
        " for member in wheel.infolist():\n"
        "  assert (target/member.filename).resolve().is_relative_to(target.resolve())\n"
        "  assert (member.external_attr >> 16) & 0o170000 != 0o120000\n"
        " wheel.extractall(target)\n"
        "sys.path.insert(0,str(target))\n"
        "import tiktoken\n"
        "print(json.dumps({'status':'tokenizer_ready','version':tiktoken.__version__,"
        "'shared_environment_modified':False,'model_requests':0}))\n")
    command = PYTHON + " -B -c " + shlex.quote(code)
    result = subprocess.run(SSH + [command], input=raw, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError("compatible_tokenizer_install_failed")
    # Refresh this task's source before preflight; no frozen run exists yet.
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for relative in code_identity():
            tar.add(GPTEXP / relative, arcname="code/GPTexp/" + relative)
    result_sync = subprocess.run(SSH + ["tar -xzf - -C " + shlex.quote(RUNTIME)],
                                 input=archive.getvalue(), capture_output=True, timeout=60)
    if result_sync.returncode:
        raise RuntimeError("isolated_source_refresh_failed")
    print(result.stdout.decode(), end="")
    write(WORK / "dependency-repair.json", {
        "wheel_version": "0.8.0", "wheel_sha256": wheel_sha,
        "source_files": code_identity(), "shared_environment_modified": False,
        "model_requests": 0,
    }, exclusive=True)


def environment_code():
    return (
        "import os,subprocess,json,sys\nfrom pathlib import Path\n"
        f"root=Path({RUNTIME!r})\n"
        "env=dict(os.environ)\n"
        "for name in list(env):\n"
        " if any(part in name.upper() for part in ('KEY','TOKEN','SECRET','PASSWORD')):\n"
        "  env.pop(name,None)\n"
        "env.update(PYTHONPATH=str(root/'deps')+':'+str(root/'code'),"
        "OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')\n"
        f"command=[{PYTHON!r},'-B','-m','GPTexp.MMSIGPT.depth_comparison']\n"
    )


def prepare():
    code = environment_code() + (
        "command+=['prepare','--selection',str(root/'selection.json'),"
        "'--tokenizer',str(root/'o200k_base.tiktoken'),'--directory',str(root/'run')]\n"
        "result=subprocess.run(command,env=env,cwd=root/'code',capture_output=True)\n"
        "print(result.stdout.decode(),end='')\n"
        "if result.returncode: raise RuntimeError('depth_preparation_failed')\n")
    output = remote(code, timeout=600)
    print(output.decode(), end="")
    write(WORK / "server-preparation.json", json.loads(output), exclusive=True)


def execute():
    api_ids = read(WORK / "selection.json")["api_ids"]
    available = read_keys(GPTEXP.parent / "xcodeapi/credentials.local.env")
    keys = {api: available[api] for api in api_ids}
    wrapper = environment_code() + (
        "command+=['execute','--directory',str(root/'run'),'--credentials-stdin']\n"
        "data=sys.stdin.buffer.read()\n"
        "result=subprocess.run(command,input=data,env=env,cwd=root/'code')\n"
        "raise SystemExit(result.returncode)\n"
    )
    command = shlex.quote(PYTHON) + " -B -c " + shlex.quote(wrapper)
    # No key appears in command arguments, logs, env, source bundle or files.
    result = subprocess.run(SSH + [command], input=json.dumps(keys).encode(), capture_output=True)
    for key in keys.values():
        if key.encode() in result.stdout or key.encode() in result.stderr:
            raise ValueError("credential_in_execution_output")
    if result.returncode:
        raise RuntimeError("remote_execution_failed")
    print(result.stdout.decode(), end="")


def finish():
    code = environment_code() + (
        "for args in [['freeze','--directory',str(root/'run')],"
        "['score','--directory',str(root/'run'),'--references',"
        "'/home/wangy/wangjh/harness3d/four-datasets-20261031/smallimage/host_evaluation/answers.json',"
        "'--reference-sha256','cbd06fa24683a0739e67119ba8d318721bad9ca7c919927cb5a2caabe7144aed']]:\n"
        " result=subprocess.run(command+args,env=env,cwd=root/'code',capture_output=True)\n"
        " print(result.stdout.decode(),end='')\n"
        " if result.returncode: raise RuntimeError('freeze_or_score_failed')\n")
    print(remote(code, timeout=240).decode(), end="")


def collect(destination):
    destination = Path(destination)
    if not destination.parent.is_dir() or destination.exists():
        raise ValueError("requires_new_destination_with_existing_parent")
    result = subprocess.run(
        SSH + ["tar -czf - --exclude=operator.lock -C " + shlex.quote(RUNTIME) + " run"],
        capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError("collection_failed")
    destination.mkdir()
    with tarfile.open(fileobj=BytesIO(result.stdout)) as tar:
        for member in tar.getmembers():
            if (not member.name.startswith("run/") and member.name != "run"
                    or member.issym() or member.islnk()
                    or not (member.isfile() or member.isdir())
                    or not (destination / member.name).resolve().is_relative_to(destination.resolve())):
                raise ValueError("unexpected_result_archive_member")
        tar.extractall(destination)
    keys = read_keys(GPTEXP.parent / "xcodeapi/credentials.local.env")
    if any(key.encode() in path.read_bytes()
           for path in destination.rglob("*") if path.is_file() for key in keys.values()):
        raise ValueError("credential_in_collected_evidence")
    print(json.dumps({"collected": str(destination), "credentials_copied": False}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=[
        "deploy", "repair_tokenizer", "prepare", "execute", "finish", "collect"])
    parser.add_argument("--destination")
    args = parser.parse_args()
    if args.phase == "collect":
        collect(args.destination)
    else:
        globals()[args.phase]()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print('{"status":"failed","reason_code":"control_phase_failed"}', file=sys.stderr)
        raise SystemExit(2)
