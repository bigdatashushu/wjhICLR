"""Continue a fenced native fullQA attempt from its verified original storage."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

TASK=Path("/home/zhouky/wangjh/harness3d/precompute-fullqa-20261024-r01")
CACHE=TASK.parent/"fullqa-network-scratch-mount-20261025-v1-cached"
HOME=TASK/"ops/recovery-20261026-v1"
WH=TASK.parent/"fullqa-network-scratch-ops-20261025-v1"
WORKER="fc0f4c7390c8e33fef0f27af2e1d15203e576f44c65d43b71ae57c9e80e3d716"
METHOD="b89bf2353210174360941afe95f64665a7349b4f26bdafd7033412b90121d22e"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value, *, replace=False):
    target=path.with_name(path.name+"."+str(time.time_ns())+".tmp") if replace else path
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"w") as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    if replace:
        os.replace(target,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def main():
    # The custody token arrives on a pipe and is never written by this adapter.
    envelope=json.load(sys.stdin)
    gpu=envelope["gpu"]
    assert gpu in [0,5,6] and os.environ["CUDA_VISIBLE_DEVICES"]==str(gpu)
    proof=json.loads((HOME/f"precheck-g{gpu}.json").read_bytes())
    assert proof["status"]=="verified" and proof["job_key"]==envelope["lease"]["spec"]["job_key"]
    assert sha((TASK/"method_config.json").read_bytes())==METHOD
    path=TASK/"source_frozen/scripts/cluster_worker.py"
    source=path.read_bytes()
    assert sha(source)==WORKER
    sys.path[:0]=[str(TASK/"source_frozen/src"),str(TASK/"source_frozen/scripts")]
    import cluster_worker as native
    lease=native.Lease.from_dict(envelope["lease"])
    assert lease.attempt==1 and lease.phase=="running"
    assert lease.worker.to_dict()=={"worker_id":f"3090-fullqa-r01-g{gpu}","host":"3090","gpu_id":str(gpu)}
    key=sha(lease.job_key.encode())
    assert not (CACHE/"jobs"/key/"attempt-1/pending.json").exists()
    assert sha((Path(proof["output_root"])/"compute_checkpoint.json").read_bytes())==proof["checkpoint_sha256"]
    assert sha((Path(proof["input_root"])/"scene.json").read_bytes())==proof["input_scene_sha256"]
    text=source.decode()
    module=native
    if gpu in [0,6]:
        local=Path(proof["input_root"]).parent
        for old,new in {
            'archive = staging / "input.tar"':'archive = local / "input.tar"',
            'inputs = staging / "input"':'inputs = local / "input"',
            'temp = staging / "input.unpacking"':'temp = local / "input.unpacking"',
            'shutil.rmtree(staging / "input.unpacking", ignore_errors=True)':
                'shutil.rmtree(local / "input.unpacking", ignore_errors=True)',
        }.items():
            assert text.count(old)==1
            text=text.replace(old,new)
        module=types.ModuleType("_retained_native_worker")
        module.__file__=str(path)
        module.__dict__["local"]=local
        exec(compile(text,str(path),"exec"),module.__dict__)
    proc=Path("/proc")/str(os.getpid())
    identity={"pid":os.getpid(),"start_ticks":int((proc/"stat").read_text().rsplit(")",1)[1].split()[19]),
              "cmdline_sha256":sha((proc/"cmdline").read_bytes())}
    state={"at":time.time(),"status":"running","process":identity,"gpu":gpu,
           "worker":lease.worker.to_dict(),"job_key":lease.job_key,"attempt":1,
           "lease_token_sha256":sha(lease.token.encode()),"native_worker_sha256":WORKER,
           "storage_locator_adapter_sha256":sha(text.encode())}
    current=HOME/f"retained-g{gpu}-current.json"
    save(current,state)
    save(WH/f"receiver-g{gpu}/current.json",state,replace=True)

    class FencedClient(native.Client):
        def request(self,route,body,**kwargs):
            if route in ["/claim","/fail","/transfer-failed"]:
                raise native.CoordinatorRejected("retained_custody_requires_operator")
            return super().request(route,body,**kwargs)

    client=FencedClient("http://10.10.0.34:27149",os.environ["HARNESS3D_CLUSTER_TOKEN"])
    try:
        result=module.run_job(client,CACHE,lease,700*1024**3)
        assert result.get("status")=="complete"
        raw=(CACHE/"acks"/(key+".json")).read_bytes()
        ack=json.loads(raw)
        assert ack["job_key"]==lease.job_key and ack["status"]=="complete"
        assert not (CACHE/"jobs"/key).exists()
        destination=TASK/"worker_cache/acks"/(key+".json")
        if destination.exists():
            assert destination.read_bytes()==raw
        else:
            fd=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,"wb") as stream:
                stream.write(raw);stream.flush();os.fsync(stream.fileno())
            fd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        state.update(status="complete",at=time.time(),ack_sha256=sha(raw),
                     native_cleanup_verified=True,original_ack_directory_updated=True)
        save(HOME/f"retained-g{gpu}-result.json",state)
        save(current,state,replace=True)
        save(WH/f"receiver-g{gpu}/current.json",state,replace=True)
        save(WH/f"transition-g{gpu}.json",
             {"status":"native_ack_released","gpu":gpu,"at":time.time(),
              "job_key":lease.job_key,"ack_sha256":sha(raw),"recovery_process":identity})
    except BaseException as error:
        state.update(status="operator_required",at=time.time(),error_type=type(error).__name__,
            frames=[{"function":f.name,"line":f.lineno} for f in traceback.extract_tb(error.__traceback__)])
        save(current,state,replace=True)
        save(WH/f"receiver-g{gpu}/current.json",state,replace=True)
        # Preserve the recovered custody on an operational fault; do not retry models.
        with native.Heartbeat(client,lease):
            while True:
                time.sleep(10)


if __name__=="__main__":
    main()
