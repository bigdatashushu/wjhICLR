"""Host-only 365-test scoring; invalid answers remain in the denominator."""
import collections
import json
import os
from pathlib import Path
import sys
for k in list(os.environ):
    if any(t in k.upper() for t in ("KEY","TOKEN","ENDPOINT","BASE_URL")):os.environ.pop(k,None)
import pilot
from skill3d.small_image.runner import parse_small_image_answer
P=Path(__file__).resolve().parent
m=pilot.read(P/"manifest.json")
binding=pilot.read(P/"host_evaluation_binding.json")
assert pilot.sha(binding["path"])==binding["sha256"]
references={(r["sample_id"]):r["reference_answer"] for r in json.loads(Path(binding["path"]).read_text())
    if r["benchmark"]=="mmsi" and r["source_split"]=="test"}
assert len(references)==365
rows=[]
for job in m["jobs"]:
    path=P/"results"/job["arm"]/(job["operation_id"]+".json")
    if not path.exists():continue
    before=pilot.sha(path);d=pilot.read(path);r=d["result"]
    assert d["job_sha256"]==pilot.sha_value(job) and r["input_sha256"]==job["input_sha256"]
    assert r["source_split"]=="test"
    for relative,expected in d["request_records"].items():assert pilot.sha(P/relative)==expected
    expected=parse_small_image_answer(references[job["sample_id"]],job["problem"]["options"])
    correct=r["status"]=="answered" and r.get("answer")==expected
    assert pilot.sha(path)==before
    rows.append({"arm":job["arm"],"question":job["ordinal"],"operation_id":job["operation_id"],"sample_id":job["sample_id"],
        "task_type":r["task_type"],"prediction":r.get("answer"),"status":r["status"],"score":int(correct),
        "requests":r["request_count"],"tool_attempts":r["attempted_tool_calls"],"tool_successes":r["succeeded_tool_calls"],
        "program_error":any(x["status"] in {"program_parse_error","program_missing_action","program_rejected","run_error"}
            for x in r["rounds"]),"seconds":d["seconds"],"prediction_sha256":before})
def summary(rs,n):
    attempts=sum(r["tool_attempts"] for r in rs)
    return {"completed":len(rs),"expected":n,"correct":sum(r["score"] for r in rs),
        "accuracy_completed":sum(r["score"] for r in rs)/len(rs) if rs else None,
        "accuracy_full_denominator":sum(r["score"] for r in rs)/n if len(rs)==n else None,
        "legal":sum(r["status"]=="answered" for r in rs),"program_errors":sum(r["program_error"] for r in rs),
        "requests":sum(r["requests"] for r in rs),"tool_attempts":attempts,
        "tool_success_rate":sum(r["tool_successes"] for r in rs)/attempts if attempts else None,
        "mean_seconds":sum(r["seconds"] for r in rs)/len(rs) if rs else None}
final="--final" in sys.argv
if final:
    assert len(rows)==730
    frozen=pilot.read(P/"predictions_frozen.json")
    for relative,expected in frozen["predictions"].items():assert pilot.sha(P/relative)==expected
out={"status":"complete" if final else "phase","at":pilot.now(),"manifest_sha256":m["content_sha256"],
    "rows":rows,"summary":{arm:summary([r for r in rows if r["arm"]==arm],365) for arm in ("skill","direct")},
    "by_task":{arm:{task:summary([r for r in rows if r["arm"]==arm and r["task_type"]==task],
        sum(j["arm"]==arm and j["problem"]["task_type"]==task for j in m["jobs"]))
        for task in sorted({j["problem"]["task_type"] for j in m["jobs"]})} for arm in ("skill","direct")},
    "model_requests":0,"reference_answers_returned":False,"solver_feedback_written":False,
    "denominator_policy":"all 365 author-test questions; invalid/program errors score zero"}
pilot.update(P/"scores.json",out)
print(json.dumps({"event":"host_scores","at":out["at"],"summary":out["summary"]}),flush=True)
