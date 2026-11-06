"""Isolated preflight or one label-free episode; kernel runs on the main thread."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

from .adapters import COUNTS, ROOT, make_specs, setup, solve
from .MMSIGPT.packaging import preflight_identity
from .api_pool import ApiPool
from .traces import digest, file_digest, read, write
from .transport import ResponsesClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=COUNTS, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if args.preflight:
        specs = make_specs(args.dataset)
        write(output, {"dataset": args.dataset, "questions": COUNTS[args.dataset],
                       "specs": specs, "model_requests": 0, "reference_answers_read": False,
                       "identity": digest(specs),
                       "adapter_files": preflight_identity(ROOT, args.dataset)},
              exclusive=True)
        print(args.dataset, "preflight passed", len(specs), "arms")
        return 0
    job = __import__("json").load(sys.stdin)
    setup(args.dataset)
    if job["spec"]["arm"] == "direct" and any(k in job["spec"] for k in
            ("prepared_root", "prepared_directory", "artifact_sha256", "skill", "allowed_tools")):
        raise ValueError("direct_received_prepared_evidence")
    config = job["config"]
    api = next(a for a in config["apis"] if a["id"] == job["api_id"])
    pool = ApiPool(job["pool_root"], config)
    client = ResponsesClient(api, pool, output / "requests", job["operation_id"],
                             reasoning_effort=config["reasoning_effort"])
    # Key stays in the host client, outside the kernel context or input frames.
    from .api_pool import resolve_key
    client._key = resolve_key(api)
    for name in tuple(os.environ):
        if any(term in name.upper() for term in ("API_KEY", "TOKEN", "BASE_URL", "ENDPOINT", "MODELS_KEY")):
            os.environ.pop(name)
    started = time.monotonic()
    result = None
    failure = None
    try:
        result = solve(job["spec"], args.dataset, client, output, job["run_identity"])
    except BaseException as exc:
        failure = {"error_type": type(exc).__name__,
                   "locations": [{"file": Path(f.filename).name, "line": f.lineno}
                                 for f in traceback.extract_tb(exc.__traceback__)]}
    requests = sorted((output / "requests").glob("*.json"))
    record = {"schema": "gptexp-episode/1", "dataset": args.dataset, "arm": job["spec"]["arm"],
              "sample_id": job["sample_id"], "operation_id": job["operation_id"],
              "job_sha256": digest(job), "api_id": api["id"], "pair_identity": job["pair_identity"],
              "run_identity": job["run_identity"], "result": result, "failure": failure,
              "elapsed_seconds": time.monotonic() - started, "reference_answer_visible": False,
              "automatic_retry": False,
              "request_files": {str(p.relative_to(output)): file_digest(p) for p in requests}}
    write(output / "episode.json", record, exclusive=True)
    return 2 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
