#!/usr/bin/env python3
"""Deploy, observe and collect an authorized frozen VSI paired experiment."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import shutil
import tarfile

REPO = Path(__file__).resolve().parents[2]
TITAN_PYTHON = "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python"
API_PYTHON = "/home/cvailab/anaconda3/envs/skill3d-exp/bin/python"
SPEC = importlib.util.spec_from_file_location(
    "authorized_ssh", REPO / "VSIbench8Brunner/scripts/ssh_from_server.py")
ssh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ssh)


def remote(host, command, data=None):
    code, stdout, stderr = ssh.remote_run(host, command, input_bytes=data, return_output=True)
    if code:
        raise RuntimeError(f"Remote exit {code}: {stderr.decode('utf-8')[-1800:]}; "
                           f"{stdout.decode('utf-8')[-1800:]}")
    return stdout


def archive(root, names):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        for name in names:
            tar.add(root / name, arcname=name)
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("deploy", "gateway", "health", "preflight",
        "start", "status", "await", "collect", "cleanup"))
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--gateway-root", required=True)
    parser.add_argument("--gateway-config", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    manifest = json.loads((root / "diagnostic.json").read_text(encoding="utf-8"))
    native = manifest["native_root"]
    gateway = args.gateway_root
    endpoint = manifest["runtime"]["perception_endpoint"]
    port = int(endpoint.rsplit(":", 1)[1])
    capacity = manifest["runtime"]["request_concurrency"]
    session = root.name

    def titan_command(script, extra):
        return ("env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "
            "MKL_NUM_THREADS=2 "
            "LD_PRELOAD=/home/wangy/wangjh/harness3d/precompute-20261022-r01/cpu_sqlite/libsqlite3.so.0 "
            "PYTHONPATH=/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/cpu_compat "
            + TITAN_PYTHON+" -B "+native+"/"+script+" --run-root "+native+" "+extra)

    if args.action == "deploy":
        assert args.gateway_config and args.gateway_config.is_file()
        shutil.copyfile(args.gateway_config, root / "gateway_config.json")
        shutil.copyfile(root / "code/source/scripts/launch_pair_gateway.py",
                        root / "launch_pair_gateway.py")
        for host, target, names in (
            ("wangy@10.10.0.34", native, ["code", "inputs", "controller.py", "launcher.py", "diagnostic.json"]),
            ("cvailab@10.10.0.198", gateway, ["code", "gateway_config.json", "launch_pair_gateway.py"])):
            remote(host, "ls -ld "+shlex.quote(str(Path(target).parent)))
            remote(host, "test ! -e "+shlex.quote(target)+" && mkdir "+shlex.quote(target)
                   + " && tar -xzf - -C "+shlex.quote(target), archive(root, names))
        print(json.dumps({"status": "deployed", "identity": manifest["identity_sha256"]}))
    elif args.action == "gateway":
        check = f"import socket;s=socket.socket();s.bind(('0.0.0.0',{port}));s.close()"
        remote("cvailab@10.10.0.198", API_PYTHON+" -c "+shlex.quote(check))
        command = (API_PYTHON+" -B "+gateway+"/launch_pair_gateway.py --run-root "
                   +gateway+f" --port {port} --concurrency {capacity}")
        remote("cvailab@10.10.0.198", "tmux new-session -d -s "+shlex.quote(session+"-gateway")+" "
               +shlex.quote(command+" > "+gateway+"/gateway.log 2>&1"))
        print('{"status":"gateway_started"}')
    elif args.action == "health":
        program = ("import json,urllib.request;"
                   f"h=json.load(urllib.request.urlopen('http://127.0.0.1:{port}/health'));"
                   f"assert h['status']=='ready' and h['max_concurrency']=={capacity};"
                   "assert h['upstream']['started']==0;print(json.dumps(h))")
        health = remote("cvailab@10.10.0.198", API_PYTHON+" -c "+shlex.quote(program))
        (root / "gateway_initial_health.json").write_bytes(health)
        (root / "gateway_start4090.json").write_bytes(
            remote("cvailab@10.10.0.198", "cat "+gateway+"/gateway_start.json"))
        print(health.decode("utf-8"))
    elif args.action == "preflight":
        command = titan_command("controller.py", "--action preflight")
        remote("wangy@10.10.0.34", "test ! -e "+native+"/preflight_receipt.json && "
               "tmux new-session -d -s "+shlex.quote(session+"-preflight")+" "
               +shlex.quote(command+" > "+native+"/preflight.log 2>&1"))
        print('{"status":"preflight_started","model_requests":0}')
    elif args.action == "start":
        remote("wangy@10.10.0.34", "test ! -e "+native+"/pair_start.json && "
               "tmux new-session -d -s "+shlex.quote(session+"-pair")+" "
               +shlex.quote(titan_command("launcher.py", "")+" > "+native+"/launcher.log 2>&1"))
        print('{"status":"paired_arms_started"}')
    elif args.action in ("status", "await"):
        wait = "--wait-change 45" if args.action == "await" else ""
        print(remote("wangy@10.10.0.34",
            titan_command("code/source/scripts/observe_small_comparison.py", wait)).decode("utf-8"))
    elif args.action == "collect":
        command = titan_command("code/source/scripts/package_small_results.py", "--allow-stopped")
        data = remote("wangy@10.10.0.34", command)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for entry in tar.getmembers():
                target = root / entry.name
                assert entry.isfile() and target.resolve().is_relative_to(root)
                if target.exists():
                    assert target.read_bytes() == tar.extractfile(entry).read(), entry.name
            tar.extractall(root)
        receipt = json.loads((root / "collection_receipt.json").read_text(encoding="utf-8"))
        for name, identity in receipt["files"].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == identity["sha256"]
        print(json.dumps({"status": "collected", "run_status": receipt["run_status"],
                          "files": len(receipt["files"]), "archive_bytes": len(data)}))
    elif args.action == "cleanup":
        exit_record = json.loads((root / "exit.json").read_text(encoding="utf-8"))
        assert exit_record["status"] in ("complete", "stopped")
        expected = json.loads((root / "gateway_start4090.json").read_text(encoding="utf-8"))
        final = json.loads((root / "gateway_final_health.json").read_text(encoding="utf-8"))
        program = (
            "import hashlib,json,os,signal,urllib.request;from pathlib import Path;"
            f"r=Path({gateway!r});expected={expected!r};final={final!r};"
            "assert json.loads((r/'gateway_start.json').read_text())==expected;"
            "pid=expected['pid'];args=[x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\\0') if x];"
            "start=args.index(str(r/'launch_pair_gateway.py'));"
            "assert hashlib.sha256(json.dumps(args[start:]).encode()).hexdigest()==expected['command_sha256'];"
            f"h=json.load(urllib.request.urlopen('http://127.0.0.1:{port}/health'));"
            "assert h['identity']==final['identity'] and h['upstream']['active']==0;"
            "assert h['upstream']['started']==final['upstream']['started'];"
            "os.kill(pid,signal.SIGTERM);"
            f"print(json.dumps({{'status':'termination_sent','pid':pid,'port':{port},"
            "'process_identity_verified':True,'active_requests':0}))")
        receipt = json.loads(remote("cvailab@10.10.0.198", API_PYTHON+" -c "+shlex.quote(program)))
        (root / "operational_cleanup.json").write_text(
            json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(receipt))


if __name__ == "__main__":
    main()
