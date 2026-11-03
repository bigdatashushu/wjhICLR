"""Synthetic verification of transport, ordered choices, sandbox and budgets."""
import io
import json
import os
from pathlib import Path
import tempfile
import urllib.error
import pilot
import numpy as np
from PIL import Image
from provider import ProxyOpener,PLACEHOLDER_ENV,configure_placeholder
from skill3d.small_image.datasets import SmallImageProblem
from skill3d.small_image.precompute import SmallImagePrecomputer
from skill3d.small_image.storage import save_prepared
from skill3d.small_image.runner import SmallImageRunnerConfig,parse_options,parse_small_image_answer,build_solver_prompt,load_problem_frames
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.request_audit import describe_request

counter=[]
def wire(request,timeout):
    envelope=json.loads(request.data);counter.append(envelope)
    assert set(envelope)=={"schema_version","request_id","payload","audit_identity"}
    assert request.get_header("Authorization")=="Bearer synthetic-local-auth"
    assert envelope["payload"]["enable_thinking"] is True and envelope["payload"]["max_tokens"]==8192
    assert "local-proxy-configuration" not in json.dumps(envelope)
    return io.BytesIO(json.dumps({"id":"synthetic","model":"qwen3.8-flash",
        "choices":[{"message":{"content":'{"answer":"A"}',"reasoning_content":"fixture"},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}).encode())
configure_placeholder()
client=BailianClient(model="qwen3.8-flash",api_key_env=PLACEHOLDER_ENV,max_output_tokens=8192,
    opener=ProxyOpener("fixture","fixture","synthetic-schema",secret="synthetic-local-auth",transport=wire))
assert client.chat([{"role":"user","content":"fixture"}],max_tokens=8192,seed=137,enable_thinking=True)=='{"answer":"A"}'
assert len(counter)==1 and client.last_request_identity["parameters"]["enable_thinking"] is True
error_count=[]
def broken(request,timeout):
    error_count.append(1)
    raise urllib.error.HTTPError(request.full_url,503,"synthetic",{},None)
bad=BailianClient(api_key_env=PLACEHOLDER_ENV,max_output_tokens=8192,
    opener=ProxyOpener("error-fixture","fixture","synthetic-schema",secret="synthetic-local-auth",transport=broken))
try:bad.chat([{"role":"user","content":"fixture"}],max_tokens=8192)
except Exception:pass
else:raise AssertionError("missing_service_error")
assert len(error_count)==1 and bad.last_service_error["http_status"]==503
assert parse_options(("same","same","other"))==(("A","same"),("B","same"),("C","other"))
assert parse_small_image_answer("B",("same","same","other"))=="B"
try:parse_small_image_answer("same",("same","same","other"))
except ValueError:pass
else:raise AssertionError("ambiguous_option_accepted")

class Fake:
    model="qwen3.8-flash";max_output_tokens=8192
    last_usage={"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}
    last_response_identity={"model":"qwen3.8-flash","model_reported":True}
    last_request_identity={};last_generation={};last_reasoning_content=None
    def __init__(self,job):self.count=0;self.job=job
    def audit_identity(self,**kwargs):return {"provider":"synthetic","model":self.model,"seed":137}
    def build_payload(self,messages,**kwargs):return {"model":self.model,"messages":messages,**kwargs}
    def chat(self,messages,**kwargs):
        self.count+=1
        assert kwargs["enable_thinking"] is True and kwargs["max_tokens"]==8192
        payload=self.build_payload(messages,**kwargs)
        actual=describe_request(payload,model_identity=self.audit_identity())
        assert [r["jpeg_sha256"] for r in actual["images_in_order"]]==[r["sha256"] for r in self.job["images"]]
        prompt=messages[0]["content"][0]["text"]
        assert "For CV3D" not in prompt and "POINT markers" not in prompt
        if self.job["arm"]=="direct":return '{"reasoning":"fixture","answer":"B"}'
        if self.count==1:
            return "def solve(ctx):\n    r=ctx.tools.point_depth(view_index=0,x=5,y=5)\n    d=ctx.tools.pair_point_distance(view_index=0,a_xy=[5,5],b_xy=[6,6])\n    YieldObservations([r['result_id'],d['result_id']])"
        assert "local median" in prompt
        return "def solve(ctx):\n    ReturnAnswer(AnswerPayload(value='B',unit='option',basis='mixed'))"
with tempfile.TemporaryDirectory(prefix="h3d-mmsi-offline-") as folder:
    t=Path(folder);pilot.ROOT=t
    images=[]
    for i in range(4):
        path=t/f"image-{i}.png";Image.fromarray(np.full((12,12,3),i,dtype=np.uint8)).save(path);images.append(str(path))
    p=SmallImageProblem("mmsi","synthetic","test","Which direction is the region?",
        ("same","same","other"),tuple(images),"Positional Relationship (Obj.–Reg.)")
    yy,xx=np.mgrid[:12,:12]
    prepared=SmallImagePrecomputer(depth_estimator=lambda rgb:{"depth":np.ones((12,12)),
        "valid_mask":np.ones((12,12),dtype=bool),"points":np.stack([xx,yy,np.ones((12,12))],axis=-1)},
        model_identities={"depth_estimator":{"model":"synthetic"}}).process(p,categories=())
    save_prepared(prepared,t/"prepared"/p.input_sha256)
    pilot.TraceClient=Fake
    policy={"max_requests":10,"max_inflight":4,"requests_per_second":0.0}
    base={"problem":p.public_dict(),"images":pilot.image_identities(p),"artifact_sha256":prepared.artifact_sha256,
        "prepared_root":str(t/"prepared"),"shared_policy":policy,"test_policy":policy,"model":{},
        "solver":SmallImageRunnerConfig(max_tokens=8192,enable_thinking=True,original_image_bytes=True,
            explicit_final_answer=True,capture_responses=True,expected_model="qwen3.8-flash").__dict__,
        "slot":"synthetic","run_identity":"synthetic","skill":{"text":"fixture"},
        "allowed_tools":["point_depth","pair_point_distance","view_depth_summary","view_candidates"]}
    a=pilot.solve_worker({**base,"operation_id":"synthetic-skill","arm":"skill"})
    # Break geometry to demonstrate that the direct arm never loads it.
    (t/"prepared"/p.input_sha256/"metadata.json").rename(t/"unused-metadata.json")
    b=pilot.solve_worker({**base,"operation_id":"synthetic-direct","arm":"direct"})
    assert a["result"]["answer"]==b["result"]["answer"]=="B"
    assert a["result"]["succeeded_tool_calls"]==2 and a["result"]["request_count"]==2
    assert a["observations"][0]["program_source"] and a["observations"][0]["yielded_result_ids"]
    assert pilot.governor(t/"requests",policy).summary()["reserved_requests"]==3
    assert pilot.governor(t/"test_budget",policy).summary()["reserved_requests"]==3
print(json.dumps({"status":"passed","live_model_requests":0,"gpu_requests":0,"reference_answers_read":False,
    "proxy_envelope_and_no_retry_verified":True,"four_original_image_order_verified":True,
    "duplicate_choices_verified":True,"two_tool_feedback_verified":True,"direct_geometry_independence_verified":True,
    "synthetic_governed_requests":3}))
