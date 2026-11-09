"""Distinguish configured backends, returned evidence, program reads, and actions."""
import base64
from collections import Counter
import io
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "src")]
from action_trajectory import restore
from protocol import canonical, sha256

rows = []
for scope, stem in (("r05_pilot", "task-card-pilot-r05"), ("r06_pilot", "task-card-pilot-r06"),
                    ("r06_full", "task-card-full-r06")):
    index = json.loads((ROOT / "runs" / (stem + "-episodes/trajectories.json")).read_bytes())
    for item in index["episodes"]:
        path = ROOT / "runs" / (stem + "-episodes") / item["directory"] / "trajectory.json"
        events = [restore(e, path.parent) for e in json.loads(path.read_bytes())["events"]]
        backend_requests, backend_responses = Counter(), Counter()
        requests, image_rows, executions, preparations = 0, [], [], []
        png = None
        for e in events:
            if e["kind"] == "observation":
                png = e["png"]
            elif e["kind"] == "model_request":
                requests += 1
                images = [c for m in e["messages"] if isinstance(m["content"], list)
                          for c in m["content"] if c["type"] == "image_url"]
                assert len(images) == 1
                url = images[0]["image_url"]["url"]
                assert url.startswith("data:image/png;base64,")
                decoded = base64.b64decode(url.split(",", 1)[1], validate=True)
                assert decoded == png
                image = Image.open(io.BytesIO(decoded))
                assert image.size == (500, 500)
                image_rows.append({"event": e["index"], "sha256": sha256(decoded),
                                   "current_rgb_only": True, "size": list(image.size)})
                card = json.loads(e["messages"][1]["content"][0]["text"])
                if card["evidence"]["status"] == "prepared":
                    preparations.append({"event": e["index"], "evidence": card["evidence"]})
            elif e["kind"] == "perception_request":
                backend_requests[e["backend"]] += 1
            elif e["kind"] == "perception_response":
                backend_responses[e["backend"]] += 1
            elif e["kind"] == "execution":
                executions.append({"event": e["index"], "tool_calls": e["receipt"]["tool_calls"],
                                   "reader_calls": e["receipt"]["reader_calls"],
                                   "program_accesses_frames": "ctx.frames" in e["program_source"],
                                   "error_code": e["receipt"]["error_code"],
                                   "trace": e["receipt"]["receipt"].get("execution_trace"),
                                   "tool_receipts": e["receipt"]["receipt"]["tool_receipts"],
                                   "action": e["submission"]})
        assert backend_requests == backend_responses
        rows.append({"scope": scope, "episode_id": item["episode_id"], "qwen_requests": requests,
                     "backend_requested": dict(backend_requests), "backend_returned": dict(backend_responses),
                     "program_tool_calls": sum(e["tool_calls"] for e in executions),
                     "program_reader_calls": sum(e["reader_calls"] for e in executions),
                     "programs_accessing_frames": sum(e["program_accesses_frames"] for e in executions),
                     "images": image_rows, "prepared_model_inputs": preparations, "executions": executions})
totals = {}
for scope in ("r05_pilot", "r06_pilot", "r06_full"):
    selected = [r for r in rows if r["scope"] == scope]
    backends = Counter()
    for row in selected:
        backends.update(row["backend_requested"])
    totals[scope] = {"episodes": len(selected), "qwen": sum(r["qwen_requests"] for r in selected),
                     "backends_requested_and_returned": dict(backends),
                     "actual_program_tool_calls": sum(r["program_tool_calls"] for r in selected),
                     "actual_reader_calls": sum(r["program_reader_calls"] for r in selected),
                     "programs_accessing_frames": sum(r["programs_accessing_frames"] for r in selected)}
with (HERE / "consumption.json").open("xb") as f:
    f.write(canonical({"schema": "embodied-followup-consumption-audit/1", "status": "passed",
                       "totals": totals, "episodes": rows, "new_model_requests": 0,
                       "scope": "Actual calls and public RGB/evidence delivery. No depth precision claim."}) + b"\n")
print(json.dumps(totals, indent=2))
