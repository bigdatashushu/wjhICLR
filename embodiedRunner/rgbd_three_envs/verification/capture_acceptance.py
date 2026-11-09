"""Accept three native RGB-D captures before admitting any live model request."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rgbd_three_envs.config import load_config
from rgbd_three_envs.sources import canonical, file_digest
from audit_three import PACKAGE, capture_audit, image, load
from PIL import Image, ImageDraw


def main():
    config, tasks = load_config(PACKAGE / "experiment-20261107-r02.json")
    frozen = load(PACKAGE / "verification/frozen-manifest-r02.json")
    reports = []
    canvas = Image.new("RGB", (1000, 530 * len(tasks)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, task in enumerate(tasks):
        report, packet = capture_audit(task, frozen)
        reports.append(report)
        public = packet["public"]
        canvas.paste(image(public["images"][0]), (0, i * 530))
        canvas.paste(image(public["depth"]["image"]), (500, i * 530))
        draw.text((8, i * 530 + 505), task.identity + " RGB", fill="black")
        draw.text((508, i * 530 + 505), task.identity + " depth 0-10m", fill="black")
    canvas.save(PACKAGE / "rgbd-three-capture-preview.png")
    failure = load(PACKAGE / "verification/habitat-r01-initialization-failure.json")
    assert failure["operation"] == "open" and failure["automatic_retry"] is False
    receipt = {"status": "accepted", "captures": reports, "model_requests": 0,
               "all_rgb_depth_shapes": [500, 500], "raw_unit": "meters",
               "config_sha256": file_digest(PACKAGE / "experiment-20261107-r02.json"),
               "habitat_r01_initialization_failure_retained": True}
    (PACKAGE / "verification/capture-acceptance.json").write_bytes(canonical(receipt) + b"\n")
    print(json.dumps({"status": "accepted", "environments": [t.environment for t in tasks],
                     "model_requests": 0}))


if __name__ == "__main__":
    main()
