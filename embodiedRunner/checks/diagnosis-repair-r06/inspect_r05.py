"""Read each complete new pilot trace and retain public evidence for follow-up."""
import io
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "src")]
from action_trajectory import restore
from protocol import canonical, sha256

index = json.loads((ROOT / "runs/task-card-pilot-r05-episodes/trajectories.json").read_bytes())
rows, images = [], []
for item in index["episodes"]:
    path = ROOT / "runs/task-card-pilot-r05-episodes" / item["directory"] / "trajectory.json"
    raw = json.loads(path.read_bytes())
    events = [restore(e, path.parent) for e in raw["events"]]
    request = None
    decisions, actions, feedback, tools = [], [], [], []
    for event in events:
        kind = event["kind"]
        if kind == "model_request":
            request = event
        elif kind == "model_response":
            card = json.loads(request["messages"][1]["content"][0]["text"])
            decisions.append({
                "request_event": request["index"], "response_event": event["index"],
                "decision_id": event["decision_id"],
                "input_card": card, "output": json.loads(event["text"]),
                "input_messages_sha256": sha256(canonical(request["messages"]))})
        elif kind == "action_submission":
            actions.append({k: v for k, v in event.items() if k != "kind"})
        elif kind == "execution":
            tools.append(event)
        elif kind == "observation":
            obs = event["observation"]
            feedback.append({"event": event["index"], "observation": obs})
            if item["episode_id"] in {"eb_alfred:base:000", "eb_alfred:base:004",
                                     "eb_navigation:base:002"}:
                picture = Image.open(io.BytesIO(event["png"])).convert("RGB")
                picture.thumbnail((380, 300))
                images.append((item["episode_id"], event["index"], obs["environment_step"], picture))
    row = {"episode_id": item["episode_id"], "trajectory_path": str(path),
           "trajectory_sha256": sha256(path.read_bytes()), "event_count": len(events),
           "decisions": decisions, "actions": actions, "observations": feedback,
           "execution_evidence": tools, "terminal": events[-1]}
    rows.append(row)
    print(json.dumps({"episode": item["episode_id"], "events": len(events),
                      "decisions": [{"event": d["response_event"], "goal": d["output"]["subgoal"],
                                     "transition": d["output"]["transition"],
                                     "output": {k: v for k, v in d["output"].items()
                                                if k not in {"subgoal", "transition"}},
                                     "repair": d["input_card"]["repair"]}
                                    for d in decisions],
                      "feedback": [{"event": f["event"], "step": f["observation"]["environment_step"],
                                    "public_feedback": f["observation"]["public_feedback"]}
                                   for f in feedback],
                      "actions": actions}, ensure_ascii=False))
with (HERE / "r05-followup-read.json").open("xb") as f:
    f.write(canonical({"schema": "embodied-r05-full-followup-read/1", "episodes": rows,
                       "model_requests": 0, "event_count": sum(r["event_count"] for r in rows)}) + b"\n")
cols, cell_w, cell_h = 4, 400, 335
canvas = Image.new("RGB", (cols * cell_w, ((len(images) + cols - 1) // cols) * cell_h), "white")
draw = ImageDraw.Draw(canvas)
for i, (ep, event, step, image) in enumerate(images):
    x, y = (i % cols) * cell_w, (i // cols) * cell_h
    draw.text((x + 5, y + 5), f"{ep} event={event} step={step}", fill="black")
    canvas.paste(image, (x + 5, y + 30))
canvas.save(HERE / "r05-followup-rgb.png")
