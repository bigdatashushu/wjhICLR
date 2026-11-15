#!/usr/bin/env python3
"""Deploy and collect the frozen ten-question fresh-round panorama comparison."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import shutil
import sys
import tarfile

REPO = Path(__file__).resolve().parents[2]
PRIOR = REPO / "VSIQwenFlashRunner/runs/counting-statefix10-c10-20261110-r08"
ROOT = PRIOR.parent / "count-panorama-multiround10-c32-20261110-r10"
NATIVE = "/home/wangy/wangjh/harness3d/VSIQwenFlashRunner/runs/" + ROOT.name
GATEWAY = "/home/cvailab/wangjh/harness3d/count-panorama-multiround10-gateway-20261110-r10"
PORT = 27051
TITAN_PYTHON = "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python"
API_PYTHON = "/home/cvailab/anaconda3/envs/skill3d-exp/bin/python"
SPEC = importlib.util.spec_from_file_location(
    "authorized_ssh", REPO / "VSIbench8Brunner/scripts/ssh_from_server.py")
ssh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ssh)


def remote(host, command, data=None):
    code, stdout, stderr = ssh.remote_run(host, command, input_bytes=data, return_output=True)
    if code:
        raise RuntimeError(f"Remote exit {code}: {stderr.decode('utf-8')[-2500:]}; "
                           f"{stdout.decode('utf-8')[-1000:]}")
    return stdout


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def archive(names):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        for name in names:
            tar.add(ROOT / name, arcname=name)
    return stream.getvalue()


def titan_command(action):
    return ("env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "
        "LD_PRELOAD=/home/wangy/wangjh/harness3d/precompute-20261022-r01/cpu_sqlite/libsqlite3.so.0 "
        "PYTHONPATH=/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/cpu_compat "
        + TITAN_PYTHON + " -B " + NATIVE + "/runner.py --run-root " + NATIVE + " --action " + action)


def main():
    action = sys.argv[1]
    if action == "deploy":
        assert not ROOT.exists()
        ROOT.mkdir()
        (ROOT / "inputs").mkdir()
        shutil.copytree(PRIOR / "code", ROOT / "code")
        for name in ("online.yaml", "published_inputs.json"):
            shutil.copyfile(PRIOR / "inputs" / name, ROOT / "inputs" / name)
        shutil.copyfile(REPO / "VSIQwenFlashRunner/scripts/run_counting_panorama_pair.py",
                        ROOT / "runner.py")
        for name in ("gateway_config.json", "launch_pair_gateway.py"):
            shutil.copyfile(PRIOR / name, ROOT / name)
        old = json.loads((PRIOR / "diagnostic.json").read_text(encoding="utf-8"))
        overviews = {}
        for p in (PRIOR / "system/variants").glob("*/plan.json"):
            plan = json.loads(p.read_text(encoding="utf-8"))
            overviews[plan["variant"]["qa_id"]] = plan["scene_overview"]
        manifest = {
            "protocol": "counting-direct-versus-fresh-round-panorama-v1",
            "native_root": NATIVE, "prior_identity": old["identity_sha256"],
            "source_sha256": old["source_sha256"],
            "runner_sha256": hashlib.sha256((ROOT / "runner.py").read_bytes()).hexdigest(),
            "variants": old["variants"], "overviews": overviews,
            "panorama_notice": "An additional panoramic image is provided.",
            "runtime": {"request_concurrency": 32, "max_requests": 60, "max_rounds": 5,
                "enable_thinking": True, "max_tokens": 8192, "seed": 137, "temperature": 0,
                "perception_endpoint": f"http://10.10.0.198:{PORT}"},
            "input_files": {name: hashlib.sha256((ROOT / "inputs" / name).read_bytes()).hexdigest()
                            for name in ("online.yaml", "published_inputs.json")},
            "comparison": {"arms": ["direct", "panorama"], "n_questions": 10,
                "fresh_calls_both_arms": True, "original_frame_count": 32,
                "extra_panorama_count": 1, "direct_max_rounds": 1,
                "panorama_max_rounds": 5, "history_accumulates": False,
                "summary_handoff": "only_previous_round_output",
                "early_answer_allowed": True, "last_round_answer_required": True,
                "tools": False, "reference_answer_visible": False}}
        manifest["identity_sha256"] = digest(manifest)
        (ROOT / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        for host, target, names in (
            ("wangy@10.10.0.34", NATIVE, ["code", "inputs", "runner.py", "manifest.json"]),
            ("cvailab@10.10.0.198", GATEWAY,
             ["code", "gateway_config.json", "launch_pair_gateway.py"])):
            parent = str(Path(target).parent)
            remote(host, "ls -ld " + shlex.quote(parent))
            remote(host, "test ! -e " + shlex.quote(target) + " && mkdir " + shlex.quote(target)
                   + " && tar -xzf - -C " + shlex.quote(target), archive(names))
        print(json.dumps({"status": "deployed", "run_root": str(ROOT),
                          "identity": manifest["identity_sha256"]}))
    elif action == "prepare":
        print(remote("wangy@10.10.0.34", titan_command("prepare")).decode("utf-8"))
    elif action == "gateway":
        check = f"import socket;s=socket.socket();s.bind(('0.0.0.0',{PORT}));s.close()"
        remote("cvailab@10.10.0.198", API_PYTHON + " -c " + shlex.quote(check))
        cmd = (API_PYTHON+" -B "+GATEWAY+"/launch_pair_gateway.py --run-root "+GATEWAY
               + f" --port {PORT} --concurrency 32")
        remote("cvailab@10.10.0.198", "tmux new-session -d -s vsi-panorama-r10-gateway "
               + shlex.quote(cmd + " > " + GATEWAY + "/gateway.log 2>&1"))
        print('{"status":"gateway_started"}')
    elif action == "health":
        program = ("import json,urllib.request;"
                   f"print(json.dumps(json.load(urllib.request.urlopen('http://127.0.0.1:{PORT}/health'))))")
        print(remote("cvailab@10.10.0.198", API_PYTHON + " -c " + shlex.quote(program)).decode("utf-8"))
    elif action == "start":
        remote("wangy@10.10.0.34", "test ! -e " + NATIVE + "/start.json && "
               "tmux new-session -d -s vsi-panorama-r10-runner "
               + shlex.quote(titan_command("run") + " > " + NATIVE + "/runner.log 2>&1"))
        print('{"status":"runner_started"}')
    elif action == "status":
        program = (
            "import json;from pathlib import Path;"
            f"r=Path({NATIVE!r});"
            "responses=[json.loads(p.read_text()) for p in r.glob('*/responses/*.json')];"
            "rounds=[json.loads(p.read_text()) for p in r.glob('*/rounds/QA*/round*.json') "
            "if not p.name.endswith('.intent.json')];"
            "print(json.dumps({'exit':json.loads((r/'exit.json').read_text()) if (r/'exit.json').exists() else None,"
            "'n_round_responses':len(rounds),"
            "'rounds':[{'arm':p.parent.parent.parent.name,'qa_id':p.parent.name,"
            "'round':json.loads(p.read_text())['round_index'],'status':json.loads(p.read_text())['status']}"
            " for p in r.glob('*/rounds/QA*/round*.json') if not p.name.endswith('.intent.json')],"
            "'responses':[{'arm':x['arm'],'qa_id':x['qa_id'],'status':x['status'],"
            "'answer':x['answer'],'n_rounds':x['n_rounds']} for x in responses]},ensure_ascii=False))")
        print(remote("wangy@10.10.0.34", TITAN_PYTHON + " -c " + shlex.quote(program)).decode("utf-8"))
    elif action == "await":
        program = (
            "import ctypes,json,os,select;from pathlib import Path;"
            f"r=Path({NATIVE!r});"
            "libc=ctypes.CDLL(None);fd=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC);"
            "assert fd>=0;assert libc.inotify_add_watch(fd,str(r).encode(),0x80|0x100)>=0;"
            "\nif not (r/'exit.json').exists():\n select.select([fd],[],[],45)\n"
            "os.close(fd);"
            "print(json.dumps({'exit':json.loads((r/'exit.json').read_text()) if (r/'exit.json').exists() else None}))")
        print(remote("wangy@10.10.0.34", TITAN_PYTHON+" -", program.encode("utf-8")).decode("utf-8"))
    elif action == "collect":
        data = remote("wangy@10.10.0.34", "tar -czf - -C " + NATIVE + " --exclude=code --exclude=inputs .")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for item in tar.getmembers():
                assert (ROOT / item.name).resolve().is_relative_to(ROOT.resolve())
                assert not item.issym() and not item.islnk()
                if item.isfile() and (ROOT / item.name).exists():
                    assert (ROOT / item.name).read_bytes() == tar.extractfile(item).read(), item.name
            tar.extractall(ROOT)
        (ROOT / "gateway_start.json").write_bytes(
            remote("cvailab@10.10.0.198", "cat "+GATEWAY+"/gateway_start.json"))
        print(json.dumps({"status": "collected", "bytes": len(data), "run_root": str(ROOT)}))
    elif action == "cleanup":
        exit_record = json.loads((ROOT / "exit.json").read_text(encoding="utf-8"))
        assert exit_record["n_results"] == 20
        expected = json.loads((ROOT / "gateway_start.json").read_text(encoding="utf-8"))
        program = (
            "import hashlib,json,os,signal,urllib.request;from pathlib import Path;"
            f"r=Path({GATEWAY!r});expected={expected!r};"
            "assert json.loads((r/'gateway_start.json').read_text())==expected;"
            "pid=expected['pid'];args=[x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\\0') if x];"
            "start=args.index(str(r/'launch_pair_gateway.py'));"
            "assert hashlib.sha256(json.dumps(args[start:]).encode()).hexdigest()==expected['command_sha256'];"
            f"h=json.load(urllib.request.urlopen('http://127.0.0.1:{PORT}/health'));"
            f"assert h['upstream']['active']==0 and h['upstream']['started']=={exit_record['model_requests']};"
            "os.kill(pid,signal.SIGTERM);"
            f"print(json.dumps({{'status':'termination_sent','pid':pid,'port':{PORT},"
            "'process_identity_verified':True,'active_requests':0}))")
        receipt = json.loads(remote("cvailab@10.10.0.198", API_PYTHON+" -c "+shlex.quote(program)))
        (ROOT / "cleanup.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(receipt))
    else:
        raise ValueError(action)


if __name__ == "__main__":
    main()
