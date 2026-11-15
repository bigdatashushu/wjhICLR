"""Record an observed timeout without resubmitting any model request."""
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT.parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import control_small_comparison as control
import score_small_comparison as score

sys.path.insert(0, str(ROOT / "code/source/src"))
from skill3d.infra.m5_identity import content_sha256

manifest = score.read(ROOT / "diagnostic.json")
assert manifest["comparison"]["arms"] == ["system"]
direct_root = ROOT.parent / "rel-direction20-g1m0-c32-20261111-r14"
direct = score.verify_root(direct_root, content_sha256)
score.verify_reused_direct_inputs(ROOT, manifest, direct_root, direct)
assert score.read(direct_root / "direct_receipt.json")["status"] == "complete"
assert score.read(direct_root / "direct_exit.json")["exit_code"] == 0
n_direct = len(list((direct_root / "direct/responses").glob("*.json")))
assert n_direct == manifest["n_questions"]
program = f"""
import json,time,urllib.request
from pathlib import Path
r=Path({manifest['native_root']!r})
m=json.loads((r/'diagnostic.json').read_text())
assert m['identity_sha256']=={manifest['identity_sha256']!r}
state=json.loads((r/'requests/state.json').read_text())
errors=[x for x in state['requests'] if x['status']=='error']
assert errors and all(x['exception_type']=='DiagnosticOperationalStop' for x in errors)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({{}}))
h=json.load(opener.open({manifest['runtime']['perception_endpoint']!r}+'/health',timeout=10))
assert h['upstream']['failed']==len(errors) and h['upstream']['unknown']==0
done=[json.loads(p.read_text())['variant_id'] for p in (r/'system/results').glob('*.json')]
assert not set(done)&{{x['context']['variant_id'] for x in errors}}
assert not list((r/'direct/responses').glob('*.json'))
print(json.dumps({{'observed_at':time.time(),'origin_identity':m['identity_sha256'],
 'failed_requests':errors,'completed_variant_ids':sorted(done),
 'active_variant_ids':[x['context']['variant_id'] for x in state['requests'] if x['status']=='started'],
 'direct_completed':0,'reused_direct_identity':{direct['identity_sha256']!r},
 'reused_direct_completed':{n_direct},'gateway_upstream':h['upstream']}}))
"""
data = control.remote("wangy@10.10.0.34",
                      control.TITAN_PYTHON + " -B -c " + shlex.quote(program))
path = ROOT / "transport_interruption_receipt.json"
assert not path.exists()
path.write_bytes(data)
print(json.dumps({"status": "observed_timeout_snapshot", "failed": len(json.loads(data)["failed_requests"]),
                  "direct_results_reused": True}))
