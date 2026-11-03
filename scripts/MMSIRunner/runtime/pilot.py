"""Frozen MMSI 365-test evaluation with independent host scores."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
import datetime
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/"source_frozen/src"),"/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/cpu_compat"]
from skill3d.small_image.blink_protocol import explicit_answer, image_identities, original_messages, response_record, sha_value
from skill3d.small_image.datasets import SmallImageProblem
from skill3d.small_image.runner import SmallImageRunnerConfig, SmallImageSolver
from skill3d.small_image.storage import load_prepared
from skill3d.small_image.tools import build_registry
from skill3d.schemas.authorization import AuthorizationDecision, not_applicable_gate
from skill3d.schemas.tool_response import model_tool_response
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.request_governor import GovernedClient, RequestGovernor, RequestGovernorConfig

def now():return datetime.datetime.now().astimezone().isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024**2),b""):h.update(block)
    return h.hexdigest()
def seal(d):return {**d,"content_sha256":sha_value(d)}
def read(path):
    d=json.loads(Path(path).read_text())
    assert d["content_sha256"]==sha_value({k:v for k,v in d.items() if k!="content_sha256"})
    return d
def write(path,d):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as stream:json.dump(seal(d),stream,ensure_ascii=False,sort_keys=True);stream.write("\n")
def update(path,d):
    path=Path(path);temp=path.with_name("."+path.name+"-"+str(os.getpid())+".tmp")
    temp.write_text(json.dumps(seal(d),ensure_ascii=False,sort_keys=True)+"\n");os.replace(temp,path)
def governor(path,policy):return RequestGovernor(path,RequestGovernorConfig(**policy))
def problem_for(public):
    p=dict(public);p.pop("input_sha256",None)
    p["image_paths"]=tuple(p["image_paths"]);p["options"]=tuple(p["options"])
    problem=SmallImageProblem(**p)
    assert problem.input_sha256==public["input_sha256"]
    return problem
def registry_for(allowed):
    def make(prepared,problem):
        registry,scene=build_registry(prepared,problem);original=registry._authorizer
        def authorize(name,spec,**context):
            if name not in allowed:
                return AuthorizationDecision(allowed=False,reason_codes=["task_denied"],missing_refs=["task_visible_tool"],
                    dependency_refs=[],metric_gate_result=not_applicable_gate())
            return original(name,spec,**context)
        registry._authorizer=authorize;registry._model_tools=frozenset(allowed)
        return registry,scene
    return make
def direct_prompt(problem):
    return ("Answer this spatial multiple-choice question using the supplied original images in their exact order. "
        "Reason about the requested relationship and choose one option. Return JSON with "
        '"reasoning" (a concise explanation) and "answer" (one original option letter).\n'+
        json.dumps({"question":problem.question,"options":[{"label":chr(65+i),"text":v}
            for i,v in enumerate(problem.options)]},ensure_ascii=False))
class TraceClient(BailianClient):
    def __init__(self,job):
        from provider import ProxyOpener,configure_placeholder
        configure_placeholder()
        super().__init__(**job["model"],opener=ProxyOpener(job["operation_id"],job["run_identity"],job["proxy_schema"]))
        self.job=job;self.index=0
    def chat(self,messages,**kwargs):
        self.index+=1
        path=ROOT/"request_records"/self.job["operation_id"]/f"{self.index:03d}"
        write(str(path)+".intent.json",{"at":now(),"parameters":kwargs,
            "payload_sha256":sha_value(self.build_payload(messages,**kwargs)),
            "images":self.job["images"],"reference_answer_visible":False})
        started=time.monotonic()
        try:
            text=super().chat(messages,**kwargs)
            assert self.last_response_identity.get("model")=="qwen3.8-flash"
            assert self.last_response_identity.get("model_reported") is True
        except BaseException as exc:
            write(str(path)+".response.json",{"status":"error","error_type":type(exc).__name__,
                "seconds":time.monotonic()-started,**response_record(self,None)})
            raise
        write(str(path)+".response.json",{"status":"received","seconds":time.monotonic()-started,**response_record(self,text)})
        return text
def solve_worker(job):
    for k in list(os.environ):
        if any(t in k.upper() for t in ("API_KEY","TOKEN","ENDPOINT","BASE_URL","MODELS_KEY")):os.environ.pop(k,None)
    os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    problem=problem_for(job["problem"])
    assert problem.source_split=="test" and image_identities(problem)==job["images"]
    context={"variant_id":job["operation_id"],"slot_key":job["slot"],
        "campaign_id":"mmsi-365-test-multiagent-20261101","run_identity":job["run_identity"]}
    client=TraceClient(job)
    shared=GovernedClient(client,governor(ROOT/"requests",job["shared_policy"]),category="solver",context=context)
    bounded=GovernedClient(shared,governor(ROOT/"test_budget",job["test_policy"]),category="solver",context=context)
    observations=[];started=time.monotonic()
    if job["arm"]=="direct":
        # The direct arm never loads prepared geometry or candidates.
        text=bounded.chat(original_messages(problem,direct_prompt(problem)),max_tokens=8192,
            temperature=0,seed=137,enable_thinking=True)
        try:answer,method=explicit_answer(text,problem.options);status="answered"
        except (ValueError,TypeError):answer=None;method=None;status="invalid_answer"
        result={"benchmark":"mmsi","sample_id":problem.sample_id,"source_split":"test","task_type":problem.task_type,
            "input_sha256":problem.input_sha256,"answer":answer,"status":status,"request_count":1,
            "attempted_tool_calls":0,"succeeded_tool_calls":0,"denied_authorizations":0,
            "rounds":[],"answer_parse_method":method}
    else:
        prepared=load_prepared(Path(job["prepared_root"])/problem.input_sha256,problem=problem)
        assert prepared.artifact_sha256==job["artifact_sha256"]
        import skill3d.small_image.runner as runner
        original=runner.RestrictedNamespaceKernel
        class TracedKernel(original):
            def run_cell(self,source,*args,**kwargs):
                cell=super().run_cell(source,*args,**kwargs)
                observations.append({"program_source":source,"program_sha256":sha_value(source),
                    "terminated":cell.terminated,"error_code":cell.error_code,
                    "yielded_result_ids":list(cell.yielded_result_ids),
                    "tool_calls":[{"tool":getattr(c,"tool",None),"args":getattr(c,"args",None)} for c in self.cell_calls],
                    "tool_results":[model_tool_response(r) for r in self.cell_results]})
                return cell
        runner.RestrictedNamespaceKernel=TracedKernel
        try:
            result=SmallImageSolver(bounded,config=SmallImageRunnerConfig(**job["solver"]),
                skill_text=job["skill"]["text"],registry_builder=registry_for(job["allowed_tools"])).solve(
                    problem,prepared=prepared,categories=prepared.categories).to_dict()
        finally:runner.RestrictedNamespaceKernel=original
    operational=result["status"] in {"service_error","input_error","precompute_error","precompute_unavailable",
        "prepared_identity_mismatch","prepared_input_mismatch","precompute_contract_error","request_exhausted"}
    receipt={"status":"stopped" if operational else "complete","operation_id":job["operation_id"],
        "job_sha256":sha_value(job),"result":result,"observations":observations,"seconds":time.monotonic()-started,
        "request_records":{str(q.relative_to(ROOT)):sha(q) for q in (ROOT/"request_records"/job["operation_id"]).glob("*.json")},
        "reference_answer_visible":False}
    write(ROOT/"results"/job["arm"]/(job["operation_id"]+".json"),receipt)
    if operational:raise RuntimeError("operational_failure_no_retry")
    return receipt
def verify(manifest):
    for relative,expected in {**manifest["source_files"],**manifest["driver_files"]}.items():
        assert sha(ROOT/relative)==expected
    assert sha(manifest["public_manifest_path"])==manifest["public_manifest_sha256"]
    state=json.loads((ROOT/"requests/state.json").read_text())
    assert state["policy"]==manifest["shared_policy"]
    assert all(r["context"].get("run_identity")==manifest["run_identity"] for r in state["requests"])
    return state
def score(final=False):
    env={k:v for k,v in os.environ.items() if not any(t in k.upper() for t in ("KEY","TOKEN","ENDPOINT","BASE_URL"))}
    result=subprocess.run([sys.executable,"-B",str(ROOT/"score.py")]+(["--final"] if final else []),env=env)
    if result.returncode:raise RuntimeError("host_scoring_audit_failed")
def run():
    manifest=read(ROOT/"manifest.json")
    from watcher import audit_cv3d
    assert audit_cv3d(manifest)["status"]=="complete_verified"
    assert read(ROOT/"ready.json")["status"]=="ready"
    with (ROOT/"operator.lock").open("a+b") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=verify(manifest)
        assert not state["requests"] and not (ROOT/"intents").exists()
        pid=os.getpid();stat=Path(f"/proc/{pid}/stat").read_text().split(") ",1)[1].split()
        write(ROOT/"startup.json",{"status":"started","pid":pid,"start_ticks":stat[19],
            "command_sha256":sha(f"/proc/{pid}/cmdline"),"at":now(),"automatic_retry":False})
        done=0
        for phase in ("test_only",):
            jobs=iter(manifest["jobs"])
            pool=ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context("spawn"))
            pending={}
            def dispatch():
                job=next(jobs,None)
                if job is None:return
                write(ROOT/"intents"/(job["operation_id"]+".json"),{"job_sha256":sha_value(job),"at":now()})
                pending[pool.submit(solve_worker,job)]=job
            try:
                for _ in range(4):dispatch()
                while pending:
                    completed,_=wait(pending,return_when=FIRST_COMPLETED);failures=[]
                    for future in completed:
                        job=pending.pop(future)
                        try:row=future.result()
                        except BaseException as exc:failures.append(exc);continue
                        done+=1
                        update(ROOT/"progress.json",{"status":"running","phase":phase,"completed_episodes":done,
                            "expected_episodes":730,"last_arm":job["arm"],"last_question":job["ordinal"],
                            "last_status":row["result"]["status"],"at":now(),
                            "request_usage":governor(ROOT/"requests",manifest["shared_policy"]).summary()})
                        print(json.dumps({"arm":job["arm"],"question":job["ordinal"],"status":row["result"]["status"],
                            "seconds":round(row["seconds"],2),"tools":[row["result"]["succeeded_tool_calls"],
                                row["result"]["attempted_tool_calls"]]}),flush=True)
                    if failures:raise failures[0]
                    if done%4==0:score()
                    for _ in completed:dispatch()
            except BaseException:
                for f in pending:f.cancel()
                pool.shutdown(wait=True,cancel_futures=True);score();raise
            else:pool.shutdown(wait=True)
        state=verify(manifest)
        assert all(r["status"]=="completed" for r in state["requests"])
        frozen={};paired={};actual_requests=0
        for job in manifest["jobs"]:
            path=ROOT/"results"/job["arm"]/(job["operation_id"]+".json");row=read(path)
            assert row["job_sha256"]==sha_value(job)
            for relative,expected in row["request_records"].items():
                assert sha(ROOT/relative)==expected
                if not relative.endswith(".response.json"):continue
                response=read(ROOT/relative);actual_requests+=1
                assert response["status"]=="received"
                identity=response["request_identity"];params=identity["parameters"]
                assert params["model"]=="qwen3.8-flash" and params["seed"]==137 and params["temperature"]==0
                assert params["enable_thinking"] is True and params["max_tokens"]==8192
                assert [r["jpeg_sha256"] for r in identity["images_in_order"]]==[r["sha256"] for r in job["images"]]
                assert response["response_identity"]["model"]=="qwen3.8-flash"
                assert response["response_identity"]["model_reported"] is True
                if relative.endswith("001.response.json"):
                    paired.setdefault(job["sample_id"],{})[job["arm"]]=(job,identity)
            frozen[str(path.relative_to(ROOT))]=sha(path)
        assert len(frozen)==730 and len(paired)==365 and actual_requests==len(state["requests"])
        for arms in paired.values():
            assert set(arms)=={"skill","direct"}
            a,ai=arms["skill"];b,bi=arms["direct"]
            assert a["input_sha256"]==b["input_sha256"]
            assert ai["parameters"]==bi["parameters"] and ai["images_in_order"]==bi["images_in_order"]
        write(ROOT/"request_pairing_audit.json",{"status":"passed","pairs":365,"requests":actual_requests,
            "parameters_and_original_images_equal":True,"at":now()})
        write(ROOT/"predictions_frozen.json",{"predictions":frozen,"questions":365,"episodes":730,"at":now()})
        write(ROOT/"completion.json",{"status":"complete","episodes":730,"questions":365,"at":now(),
            "manifest_sha256":manifest["content_sha256"],"reference_answers_read":False,
            "request_usage":governor(ROOT/"requests",manifest["shared_policy"]).summary(),
            "online_train_requests":0,"test_only":True})
        score(final=True)
        update(ROOT/"progress.json",{"status":"complete","episodes":730,"at":now(),"test_only":True})
if __name__=="__main__":
    code=0
    try:run()
    except BaseException as exc:
        code=2;update(ROOT/"failure.json",{"status":"stopped","error_type":type(exc).__name__,"at":now(),"automatic_retry":False})
        print(json.dumps({"status":"stopped","error_type":type(exc).__name__}),flush=True)
    update(ROOT/"exit.json",{"returncode":code,"at":now(),"automatic_retry":False})
    sys.exit(code)
