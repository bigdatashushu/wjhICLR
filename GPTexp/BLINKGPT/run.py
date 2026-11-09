"""Unique-owner supervisor; fresh main-thread subprocess for every solver."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

from .common import COUNTS, ROOT, digest, file_digest, read, source_files, write


def validate_manifest(root):
    m = read(root / "manifest.json")
    body = {k: v for k, v in m.items() if k != "run_identity"}
    if digest(body) != m["run_identity"] or m["source_files"] != source_files():
        raise ValueError("frozen_manifest_or_sources_changed")
    if Counter(j["dataset"] + ":" + j["arm"] for j in m["jobs"]) != COUNTS:
        raise ValueError("denominator_changed")
    return m


def command(mode, root, dataset=None):
    result = [sys.executable, "-m", "BLINKGPT.run", mode, "--root", str(root)]
    if dataset:
        result.extend(["--dataset", dataset])
    return result


def process_identity(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        return {"pid": pid, "start_ticks": raw[raw.rfind(")") + 2:].split()[19]}
    except OSError:
        return {"pid": pid}


def preflight_dataset(root, dataset):
    from .adapter import initial
    from .transport import Client, Slots
    m = validate_manifest(root)
    client = Client(m["config"], root / "unused", Slots(root / "slots", {}),
                    "preflight", 4)
    records = []
    for job in m["jobs"]:
        if job["dataset"] != dataset:
            continue
        messages, tools = initial(job, client)
        if job["arm"] == "official":
            from PIL import Image
            for image in job["spec"]["row"]["images"]:
                with Image.open(image["path"]) as actual:
                    if (actual.width != image["width"] or actual.height != image["height"]
                            or Image.MIME.get(actual.format) != image["mime_type"]):
                        raise ValueError("official_image_metadata_changed")
        payload = client.build_payload(messages, max_tokens=8192, temperature=0,
                                       seed=137, enable_thinking=True)
        records.append({"job_id": job["job_id"], "job_sha256": digest(job),
                        "initial_payload_sha256": digest(payload), "tools": tools})
    write(root / f"preflight-{dataset}.json", {
        "run_identity": m["run_identity"], "jobs": records,
        "reference_answers_read": False, "model_requests": 0})


def preflight(root):
    m = validate_manifest(root)
    for dataset in ("mv", "sr"):
        p = subprocess.run(command("preflight-dataset", root, dataset), cwd=ROOT.parent,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode:
            # Input-only preflight errors contain no credentials.
            sys.stderr.write(p.stderr.decode())
            raise ValueError("dataset_preflight_failed")
    write(root / "preflight.json", {
        "run_identity": m["run_identity"], "counts": COUNTS, "jobs": len(m["jobs"]),
        "source_files_sha256": digest(m["source_files"]), "model_requests": 0,
        "reference_answers_read": False,
        "dataset_receipts": {d: file_digest(root / f"preflight-{d}.json") for d in ("mv", "sr")}})


def mock_solver(root, dataset):
    """Exercise real caches and sandbox with controlled answers, zero API calls."""
    from .adapter import solve
    from .transport import Client, Slots
    m = validate_manifest(root)
    rows = []
    for arm in ("direct", "harness", "official"):
        job = next(j for j in m["jobs"] if j["dataset"] == dataset and j["arm"] == arm)
        class Mock(Client):
            def chat(self, messages, **kwargs):
                self.index += 1
                self.last_response_identity = {"model": self.model, "model_reported": True}
                self.last_generation = {"status": "completed", "finish_reason": "stop"}
                if arm != "harness":
                    return "A" if arm == "official" else '{"reasoning":"offline check","answer":"A"}'
                if self.index == 1:
                    return ("def solve(ctx):\n"
                            "    r = ctx.tools.view_depth_summary()\n"
                            "    if r['status'] == 'ok':\n"
                            "        YieldObservations([r['result_id']])\n")
                return ('{"reasoning":"offline check","answer":"A"}' if dataset == "sr" else
                        "def solve(ctx):\n"
                        "    ReturnAnswer(AnswerPayload(value='A', unit='option', basis='visual_estimate'))")
        client = Mock(m["config"], root / "unused", Slots(root / "slots", {}), "mock", 4)
        result = solve(job, client)
        if (result.get("status") != "answered" or result.get("answer") != "A"
                or arm == "harness" and result.get("succeeded_tool_calls", 0) < 1):
            raise ValueError("cached_solver_mock_failed_" + dataset + "_" + arm + "_" + result.get("status", "unknown"))
        rows.append({"dataset": dataset, "arm": arm, "status": result["status"],
                     "logical_calls": client.index, "succeeded_tool_calls": result.get("succeeded_tool_calls", 0)})
    write(root / f"mock-{dataset}.json", {"run_identity": m["run_identity"],
                                        "results": rows, "model_requests": 0})
    print(json.dumps(rows))


def worker(root):
    from .adapter import solve
    from .transport import Client, Slots
    private = json.load(sys.stdin)
    job, keys = private["job"], private.pop("keys")
    m = validate_manifest(root)
    if private["run_identity"] != m["run_identity"] or job not in m["jobs"]:
        raise ValueError("worker_job_identity_changed")
    for name in tuple(os.environ):
        if any(term in name.upper() for term in ("API_KEY", "TOKEN", "PASSWORD", "ENDPOINT")):
            os.environ.pop(name)
    directory = root / "requests" / job["job_id"]
    client = Client(m["config"], directory, Slots(root / "slots", keys),
                    job["job_id"], job["logical_request_cap"])
    result, failure = None, None
    started = time.time()
    try:
        result = solve(job, client)
    except Exception as exc:
        failure = {"type": type(exc).__name__, "code": getattr(exc, "code", None),
                   "locations": [{"file": Path(f.filename).name, "line": f.lineno}
                                 for f in traceback.extract_tb(exc.__traceback__)]}
    record = {"job_id": job["job_id"], "job_sha256": digest(job),
              "run_identity": m["run_identity"], "dataset": job["dataset"], "arm": job["arm"],
              "sample_id": job["sample_id"], "official_id": job["official_id"],
              "result": result, "failure": failure, "started_at": started,
              "ended_at": time.time(), "logical_requests": client.index,
              "replayed_responses": client.replayed, "reference_answer_visible": False,
              "automatic_transport_retry": True,
              "requests_sha256": {str(p.relative_to(root)): file_digest(p)
                                  for p in sorted(directory.rglob("*.json"))},
              "transport_attempts": len(list(directory.rglob("attempt-*.intent.json")))}
    write(root / "episodes" / f"{job['job_id']}.json", record, once=True)


def snapshot(root):
    m = read(root / "manifest.json")
    files = sorted((root / "episodes").glob("*.json"))
    rows = [read(p) for p in files]
    slots = [read(p) for p in sorted((root / "slots").glob("pool-*.json"))]
    return {"at": time.time(), "run_identity": m["run_identity"], "total": len(m["jobs"]),
            "completed": len(rows), "remaining": len(m["jobs"]) - len(rows),
            "counts_completed": dict(Counter(r["dataset"] + ":" + r["arm"] for r in rows)),
            "logical_requests": sum(r["logical_requests"] for r in rows),
            "transport_attempts_completed_jobs": sum(r["transport_attempts"] for r in rows),
            "replayed_responses": sum(r["replayed_responses"] for r in rows),
            "failed_episodes": sum(bool(r["failure"]) for r in rows),
            "slots": slots, "episodes": rows,
            "owner": read(root / "owner.json") if (root / "owner.json").exists() else None,
            "completed_receipt": read(root / "completed.json")
                if (root / "completed.json").exists() else None,
            "final_collection": read(root / "final-collection.json")
                if (root / "final-collection.json").exists() else None}


def finalize(root, manifest):
    files = sorted((root / "episodes").glob("*.json"))
    if len(files) != len(manifest["jobs"]):
        raise ValueError("incomplete_episode_denominator")
    jobs = {j["job_id"]: j for j in manifest["jobs"]}
    for path in files:
        row = read(path)
        if (row["job_sha256"] != digest(jobs[row["job_id"]])
                or row["run_identity"] != manifest["run_identity"]):
            raise ValueError("episode_manifest_binding_changed")
        for name, sha in row["requests_sha256"].items():
            if file_digest(root / name) != sha:
                raise ValueError("request_record_changed")
    completed = {"run_identity": manifest["run_identity"], "counts": COUNTS,
                 "jobs": len(files), "at": time.time(), "manifest_sha256": file_digest(root / "manifest.json"),
                 "episodes_sha256": {p.name: file_digest(p) for p in files},
                 "reference_answers_read": False,
                 "logical_requests": sum(read(p)["logical_requests"] for p in files),
                 "transport_attempts": sum(read(p)["transport_attempts"] for p in files),
                 "failed_episodes": sum(bool(read(p)["failure"]) for p in files)}
    if completed["logical_requests"] > manifest["logical_request_cap"]:
        raise ValueError("logical_request_cap_exceeded")
    write(root / "completed.json", completed, once=True)
    archive = root / "results.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("manifest.json", "preflight.json", "preflight-mv.json", "preflight-sr.json",
                     "completed.json", "episodes", "requests"):
            tar.add(root / name, arcname=name)
    # Read back every archive member, compare to the actual member on disk.
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.isfile():
                import hashlib
                stream = tar.extractfile(member)
                sha = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    sha.update(block)
                if sha.hexdigest() != file_digest(root / member.name):
                    raise ValueError("archive_member_readback_failed")
    write(root / "final-collection.json", {
        "run_identity": manifest["run_identity"], "jobs": len(files),
        "archive": archive.name, "archive_sha256": file_digest(archive),
        "completed_sha256": file_digest(root / "completed.json"),
        "archive_members_verified": True, "at": time.time()}, once=True)


def supervise(root, keys):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "owner.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        m = validate_manifest(root)
        if read(root / "preflight.json")["run_identity"] != m["run_identity"]:
            raise ValueError("preflight_identity_changed")
        if (root / "completed.json").exists():
            raise ValueError("completed_run_cannot_be_relaunched")
        write(root / "owner.json", {**process_identity(os.getpid()), "started_at": time.time(),
                                   "run_identity": m["run_identity"], "state": "running",
                                   "credential_slots": len(keys)})
        for folder in ("episodes", "requests", "worker-history", "slots"):
            (root / folder).mkdir(exist_ok=True)

        def execute(job):
            episode = root / "episodes" / f"{job['job_id']}.json"
            if episode.exists():
                row = read(episode)
                if row["job_sha256"] != digest(job) or row["run_identity"] != m["run_identity"]:
                    raise ValueError("resume_episode_identity_changed")
                return
            history = root / "worker-history" / job["job_id"]
            history.mkdir(exist_ok=True)
            count = len(list(history.glob("*.json")))
            # Only worker crashes receive episode recovery; terminal predictions never do.
            while count < 3:
                count += 1
                payload = json.dumps({"job": job, "keys": keys, "run_identity": m["run_identity"]})
                p = subprocess.Popen(command("worker", root), cwd=ROOT.parent,
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                write(history / f"{count:02d}.json", {
                    **process_identity(p.pid), "at": time.time(), "job_sha256": digest(job)})
                p.communicate(payload.encode())
                if episode.exists():
                    return
            write(episode, {
                "job_id": job["job_id"], "job_sha256": digest(job), "run_identity": m["run_identity"],
                "dataset": job["dataset"], "arm": job["arm"], "sample_id": job["sample_id"],
                "official_id": job["official_id"], "result": None,
                "failure": {"code": "worker_crash_recovery_exhausted"}, "ended_at": time.time(),
                "logical_requests": len(list((root / "requests" / job["job_id"]).glob("*/intent.json"))),
                "replayed_responses": 0, "reference_answer_visible": False,
                "requests_sha256": {str(p.relative_to(root)): file_digest(p)
                    for p in (root / "requests" / job["job_id"]).rglob("*.json")},
                "transport_attempts": len(list((root / "requests" / job["job_id"]).rglob(
                    "attempt-*.intent.json")))}, once=True)

        with ThreadPoolExecutor(max_workers=min(len(keys), m["config"]["workers"])) as pool:
            list(pool.map(execute, m["jobs"]))
        finalize(root, m)
        write(root / "owner.json", {**read(root / "owner.json"), "state": "completed",
                                  "ended_at": time.time()})


def start(root, keys):
    p = subprocess.Popen(command("supervise", root), cwd=ROOT.parent,
        stdin=subprocess.PIPE, stdout=(root / "supervisor.log").open("ab"),
        stderr=subprocess.STDOUT, start_new_session=True)
    p.stdin.write(json.dumps({"keys": keys}).encode())
    p.stdin.close()
    write(root / "launch.json", {**process_identity(p.pid), "at": time.time(),
                               "command_sha256": digest(command("supervise", root))})
    print(json.dumps({"pid": p.pid, "state": "launched"}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("preflight", "preflight-dataset", "mock", "worker", "supervise",
                                   "start", "snapshot", "qualify"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--dataset", choices=("mv", "sr"))
    args = p.parse_args()
    root = args.root.resolve()
    if args.mode == "preflight":
        preflight(root)
        print(json.dumps(read(root / "preflight.json")))
    elif args.mode == "preflight-dataset":
        preflight_dataset(root, args.dataset)
    elif args.mode == "mock":
        mock_solver(root, args.dataset)
    elif args.mode == "worker":
        worker(root)
    elif args.mode in ("supervise", "start", "qualify"):
        keys = json.load(sys.stdin)["keys"]
        if args.mode == "qualify":
            from .transport import qualify
            result = qualify(keys)
            write(root / "api-readiness.json", {"at": time.time(), "slots": result, "model_requests": 0})
            print(json.dumps(result))
        elif args.mode == "start":
            start(root, keys)
        else:
            supervise(root, keys)
    elif args.mode == "snapshot":
        print(json.dumps(snapshot(root), ensure_ascii=False))


if __name__ == "__main__":
    main()
