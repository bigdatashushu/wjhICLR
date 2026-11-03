"""Freeze revised six slots and validate test inputs without model calls."""
import collections
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

P=Path(__file__).resolve().parent
LAB=P.parent;S=LAB/"smallimage";OLD=LAB/"mmsi-continuation-20261101"
CV=LAB/"cv3d-840-test-thinking-20261101-dispatch-fixed"
for k in list(os.environ):
    if any(t in k.upper() for t in ("API_KEY","TOKEN","ENDPOINT","BASE_URL","MODELS_KEY")):os.environ.pop(k,None)
os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
sys.path.insert(0,str(P))
import pilot
from skill3d.small_image.runner import SmallImageRunnerConfig,parse_options,load_problem_frames,_validate_prepared,build_solver_prompt
from skill3d.small_image.storage import load_prepared
from skill3d.synthesis.bailian_client import BailianClient
from provider import health,PLACEHOLDER_ENV
def skill_sha(s):
    return hashlib.sha256(json.dumps({k:v for k,v in s.items() if k!="content_sha256"},
        sort_keys=True,ensure_ascii=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def main():
    started=time.monotonic()
    assert not (P/"manifest.json").exists()
    assert not json.loads((OLD/"requests/state.json").read_text())["requests"]
    assert not (OLD/"startup.json").exists() and (OLD/"superseded.json").exists()
    prior=pilot.read(OLD/"preflight.json")
    assert prior["all_inputs_verified"]==prior["all_npz_hashes_verified"]==522
    rows=[r for r in json.loads((S/"manifest.json").read_text())["problems"]
        if r["benchmark"]=="mmsi" and r["source_split"]=="test"]
    assert len(rows)==365 and len({r["input_sha256"] for r in rows})==365
    skills=json.loads((P/"frozen_skills.json").read_text())
    assert len(skills)==6
    for slot,entry in skills.items():
        assert skill_sha(entry["skill"])==entry["skill"]["content_sha256"]
        parent=entry["parent"];path=Path(parent["origin"])
        assert pilot.sha(path)==parent["origin_sha256"]
        current=json.loads(path.read_text());current=current["active"] if path.name=="state.json" else current
        assert current["content_sha256"]==entry["skill"]["parent_content_sha256"]==parent["skill"]["content_sha256"]
        assert entry["revision"]["test_trajectories_used"] is False
        assert entry["revision"]["reference_answers_used"] is False
        assert entry["skill"]["allowed_tools"]==["pair_point_distance","point_depth","view_candidates","view_depth_summary"]
    cached={r["input_sha256"]:r for r in prior["all_cache"]}
    seen=[]
    for i,r in enumerate(rows,1):
        problem=pilot.problem_for(r);parse_options(problem.options)
        cache=cached[r["input_sha256"]];folder=S/"prepared"/r["input_sha256"]
        assert cache["source_split"]=="test"
        assert pilot.sha(folder/"metadata.json")==cache["metadata_sha256"]
        assert pilot.sha(folder/"arrays.npz")==cache["arrays_sha256"]
        assert pilot.image_identities(problem)==cache["images"]
        seen.append({"sample_id":r["sample_id"],"input_sha256":r["input_sha256"],"n_images":len(r["image_paths"]),
            "task_type":r["task_type"],"metadata_sha256":cache["metadata_sha256"],"arrays_sha256":cache["arrays_sha256"]})
        if i%10==0 or i==365:
            pilot.update(P/"preflight_progress.json",{"status":"running","stage":"all_test_identity",
                "completed":i,"expected":365,"at":pilot.now(),"live_model_requests":0})
    solver=SmallImageRunnerConfig(max_tokens=8192,enable_thinking=True,original_image_bytes=True,
        explicit_final_answer=True,capture_responses=True,expected_model="qwen3.8-flash")
    model={"model":"qwen3.8-flash","max_output_tokens":8192,"temperature":0,"timeout_s":600,"api_key_env":PLACEHOLDER_ENV}
    rng=random.Random(137);full=[]
    for slot,entry in sorted(skills.items()):
        task=entry["skill"]["task_type"];pool=[r for r in rows if r["task_type"]==task]
        assert pool
        for r in rng.sample(pool,5):
            problem=pilot.problem_for(r);prepared=load_prepared(S/"prepared"/r["input_sha256"],problem=problem)
            assert _validate_prepared(prepared,problem,load_problem_frames(problem)) is None
            registry,scene=pilot.registry_for(entry["skill"]["allowed_tools"])(prepared,problem)
            allowed=set(registry.names_for_scope(scene.question_tool_scope,scene.available_artifacts,
                evidence_profile=scene.evidence_profile,question_type=scene.question_type,local_facts=scene.local_facts))
            prompt=build_solver_prompt(problem,prepared.frames,registry,allowed,skill_text=entry["skill"]["text"])
            assert "For CV3D" not in prompt and "POINT markers" not in prompt
            assert "candidate_depth_summary" not in allowed and "ReadObservation" not in allowed
            full.append({"input_sha256":r["input_sha256"],"artifact_sha256":prepared.artifact_sha256,"task_type":task})
            del prepared,registry,scene;gc.collect()
    smoke=subprocess.run([sys.executable,"-B",str(P/"smoke.py")],capture_output=True)
    assert smoke.returncode==0
    smoke_report=json.loads(smoke.stdout);assert smoke_report["status"]=="passed" and smoke_report["live_model_requests"]==0
    pilot.write(P/"smoke.json",smoke_report)
    proxy=health()
    assert isinstance(proxy["schema_version"],str)
    cv=pilot.read(CV/"manifest.json")
    assert cv["content_sha256"]=="17fddd21857c33d7de42df09e67a44c2446712cee4bcfa08938e0e1bc12ed7f3"
    ledger=json.loads((S/"requests/state.json").read_text());used=len(ledger["requests"])
    remaining=max(0,4200-(used-5461))
    assert used+remaining+1900<=20000
    policy={"max_requests":1900,"max_inflight":4,"requests_per_second":0.0}
    pilot.governor(P/"requests",policy);pilot.governor(P/"test_budget",policy)
    source_files={str(q.relative_to(P)):pilot.sha(q) for q in (P/"source_frozen").rglob("*") if q.is_file()}
    driver_files={q.name:pilot.sha(q) for q in P.glob("*.py")}
    driver_files["frozen_skills.json"]=pilot.sha(P/"frozen_skills.json")
    identity=pilot.sha_value({"source_files":source_files,"driver_files":driver_files,"skills":skills,
        "inputs":[r["input_sha256"] for r in rows],"model":model,"solver":solver.__dict__,"policy":policy})
    jobs=[]
    for i,r in enumerate(rows,1):
        slot,entry=next((k,v) for k,v in skills.items() if v["skill"]["task_type"]==r["task_type"])
        cache=cached[r["input_sha256"]]
        for arm in ("skill","direct"):
            job={"benchmark":"mmsi","arm":arm,"ordinal":i,"slot":slot,"sample_id":r["sample_id"],"problem":r,
                "input_sha256":r["input_sha256"],"artifact_sha256":cache["artifact_sha256"],"images":cache["images"],
                "skill":entry["skill"] if arm=="skill" else None,
                "allowed_tools":entry["skill"]["allowed_tools"] if arm=="skill" else [],
                "model":model,"solver":solver.__dict__,"prepared_root":str(S/"prepared"),"shared_policy":policy,
                "test_policy":policy,"run_identity":identity,"proxy_schema":proxy["schema_version"]}
            job["operation_id"]=pilot.sha_value(job);jobs.append(job)
    manifest={"schema":"mmsi-author-test-only/1","at":pilot.now(),"run_identity":identity,
        "source_commit":"b13952cd4d044c868b3d2418ee2a2cb6f93edf91","source_files":source_files,"driver_files":driver_files,
        "public_manifest_path":str(S/"manifest.json"),"public_manifest_sha256":pilot.sha(S/"manifest.json"),
        "jobs":jobs,"source_split":"test","questions":365,"episodes":730,"execution_role":"frozen_test_evaluation",
        "shared_policy":policy,"test_policy":policy,"maximum_additional_requests":1900,
        "maximum_episode_request_demand":1825,"automatic_retry":False,"reference_answer_visible":False,
        "online_train_requests":0,"six_skills_frozen_before_test":True,"test_results_used_for_revision":False,
        "cv3d_root":str(CV),"cv3d_manifest_sha256":cv["content_sha256"],"cv3d_budget_prefix":5461,
        "cv3d_request_cap":4200,"original_total_cap":20000,"budget_used_at_preflight":used,
        "cv3d_reserved_remaining_at_preflight":remaining,"prior_preflight_sha256":pilot.sha(OLD/"preflight.json"),
        "superseded_train_launcher":str(OLD),"task_counts":dict(collections.Counter(r["task_type"] for r in rows))}
    pilot.write(P/"manifest.json",manifest)
    labels=S/"host_evaluation/answers.json"
    pilot.write(P/"host_evaluation_binding.json",{"path":str(labels),"sha256":pilot.sha(labels),
        "reference_answer_visible_to_online_worker":False})
    pilot.write(P/"preflight.json",{"status":"passed","at":pilot.now(),"all_test_identity_verified":365,
        "selected_full_content_verified":30,"all_test_records":seen,"selected_full_records":full,
        "previous_all522_preflight_sha256":pilot.sha(OLD/"preflight.json"),
        "live_model_requests":0,"gpu_requests":0,"reference_answers_read":False,
        "test_trajectories_read_for_revision":False,"seconds":time.monotonic()-started})
    pilot.write(P/"ready.json",{"status":"ready","at":pilot.now(),"manifest_sha256":pilot.read(P/"manifest.json")["content_sha256"],
        "online_status":"awaiting_cv3d_completion","questions":365,"episodes":730,"test_only":True,
        "live_model_requests":0,"gpu_requests":0})
    pilot.update(P/"preflight_progress.json",{"status":"complete","at":pilot.now(),"all_test_identity_verified":365})
    print(json.dumps({"status":"ready","questions":365,"episodes":730,"test_only":True,"live_model_requests":0}),flush=True)
if __name__=="__main__":
    code=0
    try:main()
    except BaseException as exc:
        code=2
        import traceback
        pilot.update(P/"preflight_failure.json",{"status":"failed","error_type":type(exc).__name__,
            "at":pilot.now(),"live_model_requests":0,"automatic_retry":False,
            "frames":[{"file":Path(f.filename).name,"line":f.lineno} for f in traceback.extract_tb(exc.__traceback__)]})
    pilot.update(P/"preflight_exit.json",{"returncode":code,"at":pilot.now()})
    sys.exit(code)
