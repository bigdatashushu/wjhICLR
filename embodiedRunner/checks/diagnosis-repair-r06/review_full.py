"""Expose all completed raw decisions and RGB observations for manual review."""
from collections import Counter
import io
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent / "full"
sys.path[:0] = [str(ROOT), str(ROOT.parent / "src")]
from action_trajectory import restore
from protocol import canonical

verified = json.loads((HERE / "verification.json").read_bytes())
summaries = []
for item in verified["episodes"]:
    path = Path(item["trajectory_path"])
    events = [restore(e, path.parent) for e in json.loads(path.read_bytes())["events"]]
    request = None
    chain, pictures = [], []
    for e in events:
        if e["kind"] == "model_request":
            request = e
        elif e["kind"] == "model_response":
            card = json.loads(request["messages"][1]["content"][0]["text"])
            output = json.loads(e["text"])
            chain.append({
                "request_event": request["index"], "response_event": e["index"],
                "decision_id": e["decision_id"], "old_goal": card["subgoal"],
                "output": output, "facts": card["facts"], "latest": card["latest"],
                "remaining": card["remaining"], "repair": card["repair"],
                "evidence": card["evidence"], "loop": card["loop"]})
        elif e["kind"] == "observation":
            image = Image.open(io.BytesIO(e["png"])).convert("RGB")
            image.thumbnail((300, 300))
            pictures.append((e["index"], e["observation"]["environment_step"], image,
                             e["observation"]["public_feedback"]))
    report = {
        "episode_id": item["episode_id"], "success": item["success"], "progress": item["progress"],
        "terminal": item["terminal_reason"], "actions": item["actions"],
        "initial_comparisons": item["initial_comparisons"], "chain": chain,
        "feedback": [{"event": i, "step": s, "feedback": f} for i, s, _, f in pictures],
        "event_counts": dict(Counter(e["kind"] for e in events)),
        "execution_evidence": item["execution_evidence"]}
    name = item["episode_id"].replace(":", "-")
    with (HERE / (name + "-review.json")).open("xb") as f:
        f.write(canonical(report) + b"\n")
    w, h, cols = 320, 340, 4
    canvas = Image.new("RGB", (w * cols, h * ((len(pictures) + cols - 1) // cols)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (event, step, picture, feedback) in enumerate(pictures):
        x, y = (i % cols) * w, (i // cols) * h
        draw.text((x + 5, y + 5), f"{name} e{event} step={step}", fill="black")
        canvas.paste(picture, (x + 5, y + 30))
    canvas.save(HERE / (name + "-rgb.png"))
    summaries.append({"episode": item["episode_id"], "success": item["success"],
                      "actions": item["actions"], "requests": len(chain),
                      "event_counts": report["event_counts"], "feedback": report["feedback"],
                      "response_summary": [
                          {"event": r["response_event"], "old_goal": r["old_goal"],
                           "goal": r["output"]["subgoal"], "transition": r["output"]["transition"],
                           "program": r["output"].get("program_source"),
                           "perception": r["output"].get("perception_request"),
                           "progress": r["output"].get("progress"), "repair": r["repair"]}
                          for r in chain]})
with (HERE / "all-review.json").open("xb") as f:
    f.write(canonical(summaries) + b"\n")
print(json.dumps([{"episode": r["episode"], "success": r["success"],
                   "actions": len(r["actions"]), "requests": r["requests"], "event_counts": r["event_counts"]}
                  for r in summaries], indent=2))
