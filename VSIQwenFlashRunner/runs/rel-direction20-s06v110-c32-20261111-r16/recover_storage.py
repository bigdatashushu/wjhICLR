"""Preserve request evidence and document the observed storage interruption."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sqlite3
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT.parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from control_small_comparison import remote, TITAN_PYTHON, API_PYTHON

manifest = json.loads((ROOT / "diagnostic.json").read_text())
NATIVE = manifest["native_root"]
GATEWAY = "/home/cvailab/wangjh/harness3d/vsi-direction20-s06v110-gateway-20261111-r16"


def main():
    remote("cvailab@10.10.0.198", "ls -ld " + shlex.quote(GATEWAY))
    program = f"""
import hashlib,json,sqlite3,urllib.request
from pathlib import Path
root=Path({GATEWAY!r})
health=json.load(urllib.request.urlopen('http://127.0.0.1:27058/health',timeout=10))
assert health['upstream']['active']==health['upstream']['unknown']==0
p=root/'gateway_cache/responses.sqlite3'
snapshot=root/'storage_recovery_snapshot.sqlite3'
if not snapshot.exists():
    source=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True)
    dest=sqlite3.connect(str(snapshot))
    source.backup(dest)
    dest.close()
    source.close()
data=snapshot.read_bytes()
print(json.dumps({{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),
                  'health':health}}))
"""
    info = json.loads(remote("cvailab@10.10.0.198",
                            API_PYTHON + " -B -c " + shlex.quote(program)))
    data = remote("cvailab@10.10.0.198",
                  "cat " + shlex.quote(GATEWAY + "/storage_recovery_snapshot.sqlite3"))
    assert len(data) == info["bytes"]
    assert hashlib.sha256(data).hexdigest() == info["sha256"]
    local = ROOT / "gateway_storage_snapshot.sqlite3"
    if local.exists():
        assert local.read_bytes() == data
    else:
        local.write_bytes(data)
    connection = sqlite3.connect("file:" + str(local) + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = []
    for row in connection.execute(
            "SELECT r.*,a.started_at_ns,a.ended_at_ns,a.success,a.error_type,"
            "a.usage_status,a.token_usage FROM requests r "
            "JOIN upstream_audit a USING(request_id) ORDER BY a.started_at_ns"):
        value = dict(row)
        reply = value.pop("reply")
        if isinstance(reply, str):
            reply = reply.encode()
        value["reply_sha256"] = hashlib.sha256(reply).hexdigest()
        value["usage"] = json.loads(value.pop("token_usage")) if value["token_usage"] else {}
        if value["success"]:
            parsed = json.loads(reply)
            assert all(parsed["last_usage"][k] == v for k, v in value["usage"].items())
            value["audit_usage"] = value["usage"]
            value["usage"] = parsed["last_usage"]
        rows.append(value)
    assert len(rows) == info["health"]["upstream"]["started"] == 36
    assert all(r["state"] == "complete" for r in rows)
    evidence = {"origin_identity": manifest["identity_sha256"],
                "database_sha256": info["sha256"], "database_bytes": len(data),
                "snapshot_method": "SQLite backup from read-only connection",
                "gateway_health": info["health"], "requests": rows}
    (ROOT / "gateway_request_evidence_receipt.json").write_text(
        json.dumps(evidence, indent=2) + "\n")
    program = f"""
import hashlib,json,os,time
from pathlib import Path
r=Path({NATIVE!r})
manifest=json.loads((r/'diagnostic.json').read_text())
assert manifest['identity_sha256']=={manifest['identity_sha256']!r}
targets={{str(r/'controller.py'),str(r/'launcher.py')}}
alive=[]
for p in Path('/proc').glob('[0-9]*'):
    try:
        argv=(p/'cmdline').read_bytes().split(b'\\0')
        if any(x.decode(errors='replace') in targets for x in argv):
            alive.append(int(p.name))
    except (OSError,ValueError):pass
assert not alive,alive
state=json.loads((r/'requests/state.json').read_text())
done=[json.loads(p.read_text())['variant_id'] for p in (r/'system/results').glob('*.json')]
expected={{v['variant_id'] for v in manifest['variants']}}
missing=sorted(expected-set(done))
assert len(done)==17 and len(missing)==3
log=(r/'launcher.log').read_bytes()
assert b'No space left on device' in log and b'Errno 28' in log
logs={{name:hashlib.sha256((r/name).read_bytes()).hexdigest()
      for name in ('launcher.log','system.log')}}
cache=r/'prepared_cache'
assert cache.is_symlink()
observed=time.time()
receipt={{'status':'observed_storage_interruption','observed_at':observed,
 'origin_identity':manifest['identity_sha256'],'online_processes_alive':alive,
 'failure':'ENOSPC while saving episode/summary/exit artifacts',
 'raw_requests_state_sha256':hashlib.sha256((r/'requests/state.json').read_bytes()).hexdigest(),
 'log_sha256':logs,'completed_variant_ids':sorted(done),'missing_variant_ids':missing,
 'failed_requests':[x for x in state['requests'] if x['status']=='error'],
 'unterminated_requests':[x for x in state['requests'] if x['status']=='started'],
 'prepared_cache_target':str(cache.resolve()),
 'home_free_bytes':os.statvfs(str(r)).f_bavail*os.statvfs(str(r)).f_frsize,
 'cache_volume_free_bytes':os.statvfs(str(cache.resolve())).f_bavail*os.statvfs(str(cache.resolve())).f_frsize,
 'selection_used_reference_answers':False,'selection_basis':'absence of complete system result',
 'actual_exit_code':None,'actual_ended_at':None,
 'note':'Observed postmortem; original exit metadata could not be written. Raw logs/state/summary retained.'}}
p=r/'storage_interruption_receipt.json'
assert not p.exists()
p.write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\\n')
exit={{'status':'stopped','postmortem':True,'observed_at':observed,'exit_code':None,
 'ended_at':None,'cause':'ENOSPC','storage_receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}
assert not (r/'exit.json').exists()
(r/'exit.json').write_text(json.dumps(exit,sort_keys=True,indent=2)+'\\n')
print(json.dumps(receipt))
"""
    receipt = json.loads(remote("wangy@10.10.0.34",
                               TITAN_PYTHON + " -B -c " + shlex.quote(program)))
    print(json.dumps({"status": receipt["status"],
                      "missing_variant_ids": receipt["missing_variant_ids"],
                      "gateway_known_completed": sum(r["success"] for r in rows),
                      "gateway_failed": sum(not r["success"] for r in rows),
                      "database_sha256": info["sha256"]}))


if __name__ == "__main__":
    main()
