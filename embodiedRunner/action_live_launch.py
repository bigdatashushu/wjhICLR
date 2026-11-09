"""Mac-to-Linux launcher. Auth crosses SSH stdin only and is never written."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess

from action_deploy import ssh, REMOTE_PARENT
from protocol import ROOT, canonical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--detach", action="store_true",
                        help="run in an owned tmux session using transient credential handoff")
    parser.add_argument("--config-name", default="action-live-4090.json")
    parser.add_argument("--output-name", default="live-smoke.json")
    parser.add_argument("--timeout-s", type=int, default=8400)
    args = parser.parse_args()
    if (not args.remote.startswith(REMOTE_PARENT + "/embodied-action-v2-") or ".." in args.remote
            or Path(args.output_name).name != args.output_name or not args.output_name.endswith(".json")
            or args.config_name not in {"action-live-4090.json", "action-compare-4090.json",
                                        "action-compare-extra-4090.json", "action-manipulation-4090.json",
                                        "action-task-card-15-4090.json", "action-task-card-pilot-r05.json",
                                        "action-task-card-pilot-r06.json"}
            or not 60 <= args.timeout_s <= 24000 or args.detach and args.compare):
        parser.error("use a deployed isolated workspace and a plain JSON output name")
    key = ""
    if not args.check:
        from bare_run import load_authorized_key
        load_authorized_key()
        key = os.environ.pop("DASHSCOPE_API_KEY")
    bootstrap = ("""import sys,json,pathlib
root=pathlib.Path(sys.argv[1])/'embodiedRunner'
sys.path.insert(0,str(root))
from action_detach import launch
receipt=launch(root,sys.argv[2],sys.argv[3],sys.argv[4],json.load(sys.stdin)['api_key'])
print(json.dumps(receipt))
""" if args.detach else """import sys,os,json,pathlib,importlib.util
root=pathlib.Path(sys.argv[1])/'embodiedRunner'
credential=json.load(sys.stdin)['api_key']
if credential:
    os.environ['DASHSCOPE_API_KEY']=credential
path='/home/cvailab/wangjh/harness3d/four-datasets-20261031/transport_auth.py'
spec=importlib.util.spec_from_file_location('authorized_transport_auth',path)
auth=importlib.util.module_from_spec(spec)
spec.loader.exec_module(auth)
auth.load_transport_token()
sys.path[:0]=[str(root),str(root.parent/'src')]
sys.argv=['action_run.py',sys.argv[2],'--live-config',str(root/'config'/sys.argv[4]),
          '--output',str(root/'runs'/sys.argv[3])]
from action_run import main
main()
""")
    command = shlex.join([
        "/home/cvailab/anaconda3/envs/skill3d-exp/bin/python", "-B", "-c", bootstrap,
        args.remote, "--live-check" if args.check else ("--compare" if args.compare else "--live"),
        args.output_name, args.config_name])
    result = subprocess.run(ssh(command), input=canonical({"api_key": key}), timeout=args.timeout_s)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
