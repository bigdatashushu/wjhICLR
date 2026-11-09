"""Capture one unannotated Manipulation RGB/depth pair, with zero actions."""
import base64
import io
import json
import os
from pathlib import Path
import sys

CODE = Path("/home/cvailab/wangjh/embodied-rgbd-twenty-20261107-r01/embodiedRunner")
sys.path.insert(0, str(CODE))
from rgbd_three_envs.depth import make_depth
from rgbd_three_envs.protocol import Task
from rgbd_three_envs.sources import Sources, canonical, digest, implementation_hashes
from rgbd_three_envs.worker import Environment, native_log, probe

ROOT = Path("/home/cvailab/wangjh/embodied-groundingdino-probe-20261107-r01")
assert ROOT.parent.is_dir()
ROOT.mkdir(exist_ok=False)
work = ROOT / "manipulation-raw"
work.mkdir()
config = json.loads((CODE / "rgbd_twenty/config-rgbd_three_envs.json").read_text())
runtime = config["runtime"]
sources = Sources(runtime["source_root"], runtime["manipulation_catalog"], runtime["manipulation_dataset"])
task = Task("manipulation", "base", 1)
ready = probe(sources, task, None)
assert ready["runtime_ready"] and ready["model_requests"] == ready["simulator_launches"] == 0


class RawEnvironment(Environment):
    def packet(self, annotate=False):
        from PIL import Image
        stream = io.BytesIO()
        Image.fromarray(self.obs["front_rgb"]).save(stream, format="PNG")
        png = stream.getvalue()
        misc = self.obs["misc"]
        depth = make_depth(self.obs["front_depth"], digest(png), 0, "manipulation",
                           float(misc["front_camera_near"]), float(misc["front_camera_far"]))
        (work / "rgb.png").write_bytes(png)
        (work / "depth.png").write_bytes(base64.b64decode(depth["image"]["base64"]))
        return {"status": "ok", "task": task.as_dict(),
                "instruction": self.env.episode_language_instruction,
                "rgb_base64": base64.b64encode(png).decode(),
                "depth_base64": depth["image"]["base64"],
                "rgb_sha256": digest(png), "depth_sha256": depth["image"]["sha256"],
                "model_requests": 0, "environment_steps": 0, "simulator_launches": 1,
                "same_observation": True, "raw_unannotated_rgb": True,
                "implementation_hashes": implementation_hashes(), "source_hashes": sources.hashes,
                "remote_directory": str(work)}


session = RawEnvironment(sources, task, work, None, 137)
report = None
with (work / "runtime.log").open("x") as log, native_log(log):
    try:
        report = session.open()
    finally:
        session.close()
owned = []
for path in Path("/proc").iterdir():
    if not path.name.isdigit() or int(path.name) == os.getpid():
        continue
    try:
        command = (path / "cmdline").read_bytes()
        cwd = str((path / "cwd").resolve())
    except OSError:
        continue
    if command and (str(ROOT).encode() in command or cwd.startswith(str(ROOT))):
        owned.append(int(path.name))
assert not owned
report["active_owned_workers"] = owned
(work / "capture.json").write_bytes(canonical({k: v for k, v in report.items()
                                             if k not in ("rgb_base64", "depth_base64")}) + b"\n")
print(canonical(report).decode())
