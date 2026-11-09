"""Render recorded RGB/depth pairs without changing their 500x500 scene pixels."""
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "rgbd_trial"
RUN = PACKAGE / "runs/nav-rgbd-five-20261107-r01"


def image(record):
    return Image.open(io.BytesIO(base64.b64decode(record["base64"]))).convert("RGB")


def main():
    initial = Image.new("RGB", (1000, 5 * 530), "white")
    draw_initial = ImageDraw.Draw(initial)
    rendered = []
    for folder in sorted(RUN.glob("navigation-base-*")):
        trace = [json.loads(line) for line in (folder / "trajectory.jsonl").read_text().splitlines()]
        decisions, public = [], None
        for event in trace:
            if event["type"] == "observation":
                public = event["public"]
            elif event["type"] == "model_request":
                decisions.append((event["request_index"], public))
        if not decisions:
            continue
        index = int(folder.name.rsplit("-", 1)[1])
        first = decisions[0][1]
        initial.paste(image(first["images"][0]), (0, index * 530))
        initial.paste(image(first["depth"]["image"]), (500, index * 530))
        draw_initial.text((8, index * 530 + 505), f"base:{index:03d} current RGB", fill="black")
        draw_initial.text((508, index * 530 + 505), f"base:{index:03d} same-event depth", fill="black")
        chosen = decisions if len(decisions) <= 6 else decisions[:4] + decisions[-2:]
        sheet = Image.new("RGB", (1000, len(chosen) * 530), "white")
        draw = ImageDraw.Draw(sheet)
        for row, (request, public) in enumerate(chosen):
            sheet.paste(image(public["images"][0]), (0, row * 530))
            sheet.paste(image(public["depth"]["image"]), (500, row * 530))
            label = f"base:{index:03d} request {request} / step {public['step']}"
            draw.text((8, row * 530 + 505), label + " - RGB", fill="black")
            draw.text((508, row * 530 + 505), label + " - depth 0-10m", fill="black")
        sheet.save(PACKAGE / f"rgbd-base-{index:03d}-decisions.png")
        rendered.append(index)
    initial.save(PACKAGE / "rgbd-five-initial-pairs.png")
    print(json.dumps({"rendered_episodes": rendered}))


if __name__ == "__main__":
    main()
