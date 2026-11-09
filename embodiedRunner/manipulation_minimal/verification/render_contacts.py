"""Contact sheets of actual recorded observations, without altering scene pixels."""
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "manipulation_minimal"
TRIAL = PACKAGE / "runs/single-step-visual-five-20261107-r01"
BASELINE = ROOT / "official_direct/runs/baseline-five-20261107-r03"


def trace(folder):
    return [json.loads(x) for x in (folder / "trajectory.jsonl").read_text().splitlines()]


def image(public):
    return Image.open(io.BytesIO(base64.b64decode(public["images"][0]["base64"]))).convert("RGB")


def main():
    final_sheet = Image.new("RGB", (1000, 2650), "white")
    final_draw = ImageDraw.Draw(final_sheet)
    for i in range(5):
        folder = TRIAL / f"manipulation-base-{i:03d}"
        events = trace(folder)
        decisions, public = [], None
        for event in events:
            if event["type"] == "observation":
                public = event["public"]
            elif event["type"] == "model_request":
                decisions.append((event["request_index"], public, None))
            elif event["type"] == "model_response":
                request, observation, _ = decisions[-1]
                decisions[-1] = (request, observation, event["plan"]["actions"])
        rows = (len(decisions) + 3) // 4
        sheet = Image.new("RGB", (2000, rows * 550), "white")
        draw = ImageDraw.Draw(sheet)
        for j, (request, observation, action) in enumerate(decisions):
            x, y = j % 4 * 500, j // 4 * 550
            sheet.paste(image(observation), (x, y))
            draw.text((x + 8, y + 505),
                      f"base:{i:03d} request {request} / step {observation['step']}", fill="black")
            draw.text((x + 8, y + 522), f"Next action: {action}", fill="black")
        sheet.save(PACKAGE / f"verification/trial-base-{i:03d}-decisions.png")
        for column, run in enumerate((BASELINE, TRIAL)):
            episode = run / f"manipulation-base-{i:03d}"
            last = [e for e in trace(episode) if e["type"] == "observation"][-1]
            result = json.loads((episode / "result.json").read_text())
            final_sheet.paste(image(last["public"]), (column * 500, i * 530))
            name = "baseline" if column == 0 else "single-step visual"
            final_draw.text((column * 500 + 8, i * 530 + 505),
                            f"base:{i:03d} {name} final; official success={result['official']['task_success']}",
                            fill="black")
    final_sheet.save(PACKAGE / "verification/final-comparison.png")
    print(json.dumps({"decision_contact_sheets": 5, "final_comparison": True}))


if __name__ == "__main__":
    main()
