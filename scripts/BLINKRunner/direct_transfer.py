"""Transfer prepared evidence directly between authorized nodes with transient SSH auth."""
import json
import os
from pathlib import Path
import shlex
import subprocess


def main():
    root = Path("/var/tmp/harness3d-blink-sr80-20261105-r01")
    config = "/Users/bytedance/wangjh/.trae/skills/harness3d-cluster-ops/assets/ssh_config"
    auth_env = {**os.environ, "BLINK_SSH_CREDENTIAL_INDEX": "1"}
    password = subprocess.check_output(
        ["/private/tmp/blink93_ssh_askpass.py"], env=auth_env).decode().strip()
    keys = subprocess.check_output([
        "ssh-keygen", "-F", "10.10.0.34",
        "-f", str(Path.home() / ".ssh/known_hosts")], text=True)
    known_hosts = "\n".join(line for line in keys.splitlines() if not line.startswith("#")) + "\n"
    if not password or not known_hosts.strip():
        raise ValueError("missing_existing_ssh_auth_or_trusted_host_key")
    script = '''
import json,os,pathlib,subprocess,sys,time
root=pathlib.Path(RUN_ROOT)
auth=json.loads(sys.stdin.read())
known=root/"transfer_known_hosts"
known.write_text(auth["known_hosts"])
helper=root/"transfer_askpass.py"
helper.write_text("#!/usr/bin/env python3\\nimport os\\nprint(os.environ['BLINK_TRANSFER_SSH_AUTH'])\\n")
helper.chmod(0o700)
env=dict(os.environ)
for k in list(env):
 if any(t in k.upper() for t in ("API_KEY","TOKEN","ENDPOINT","BASE_URL","PASSWORD","SECRET")):
  env.pop(k,None)
env.update(SSH_ASKPASS=str(helper),SSH_ASKPASS_REQUIRE="force",DISPLAY=":0",
           BLINK_TRANSFER_SSH_AUTH=auth["password"])
source="/home/zhouky/wangjh/harness3d/blink-sr80-20261105-r01/prepared_evidence.tar.gz"
command=["scp","-o","StrictHostKeyChecking=yes","-o","UserKnownHostsFile="+str(known),
         "-o","ConnectTimeout=10",source,
         "wangy@10.10.0.34:"+str(root/"prepared_evidence.direct.tar.gz")]
started=time.monotonic()
child=subprocess.run(command,input=b"",stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                     env=env,timeout=180)
if auth["password"].encode() in child.stdout or auth["password"].encode() in child.stderr:
 raise ValueError("transfer_output_contains_credentials")
print(json.dumps({"returncode":child.returncode,"elapsed_s":time.monotonic()-started,
                  "credential_written_to_disk":False}))
sys.exit(child.returncode)
'''.replace("RUN_ROOT", repr(str(root)))
    env = dict(os.environ)
    env.update(SSH_ASKPASS="/private/tmp/blink93_ssh_askpass.py",
               SSH_ASKPASS_REQUIRE="force", DISPLAY=":0", BLINK_SSH_CREDENTIAL_INDEX="0")
    child = subprocess.run([
        "ssh", "-F", config, "h3d-3090", "python3 -c " + shlex.quote(script)],
        input=json.dumps({"password": password, "known_hosts": known_hosts}).encode(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=210)
    if password.encode() in child.stdout or password.encode() in child.stderr:
        raise ValueError("transfer_output_contains_credentials")
    print(child.stdout.decode().strip())
    if child.returncode:
        raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
