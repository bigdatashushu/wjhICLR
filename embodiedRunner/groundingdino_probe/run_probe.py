"""Run twelve pinned screenshot detections and preserve native boxes unchanged."""
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(OUT))
from ops import remote


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def save(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def call_service(value):
    source = "INPUT=" + repr(value) + "\n" + (OUT / "remote_service.py").read_text()
    return remote(source, timeout=240)


def font(size):
    return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)


def prepare(resume_capture=False):
    assert not (OUT / "inputs.json").exists(), "inputs_already_prepared"
    folder = OUT / "inputs"
    folder.mkdir(exist_ok=resume_capture)
    suite = ROOT / "rgbd_three_envs/runs/rgbd-twenty-20261107-r01"
    selected = [
        ("navigation", ROOT / "rgbd_trial/runs/rgbd-twenty-20261107-r01/navigation-base-000", 7,
         "bread.", "bread. pot. toaster. microwave. sink. cabinet. counter. fridge. stove. window. kettle."),
        ("alfred", suite / "alfred-base-002", 13,
         "toilet paper roll. toilet.", "toilet paper roll. toilet. cabinet. towel. bathtub. soap. candle. plunger. sink. door."),
        ("habitat", suite / "habitat-base-000", 20,
         "toy airplane. counter.", "toy airplane. toy. robot arm. apple. orange. counter. bowl. bottle. sink. cabinet."),
    ]
    cases = []
    for kind, source, serial, target, broad in selected:
        packet = None
        with (source / "trajectory.jsonl").open() as stream:
            for line in stream:
                event = json.loads(line)
                if event["type"] == "observation" and event["serial"] == serial:
                    packet = event
                    break
        assert packet is not None
        public = packet["public"]
        png = (source / f"observation-{serial:03d}-0.png").read_bytes()
        depth = (source / f"observation-{serial:03d}-depth.png").read_bytes()
        assert sha(png) == public["images"][0]["sha256"]
        assert sha(depth) == public["depth"]["image"]["sha256"]
        assert public["depth"]["synchronization"]["same_event"]
        (folder / f"{kind}-rgb.png").write_bytes(png)
        (folder / f"{kind}-depth.png").write_bytes(depth)
        cases.append({"environment": kind, "task": packet["task"], "instruction": public["instruction"],
                      "serial": serial, "step": public["step"], "source_directory": str(source),
                      "raw_unannotated_rgb": True, "rgb_sha256": sha(png), "depth_sha256": sha(depth),
                      "target_prompt": target, "broad_prompt": broad})
    sim = "/home/cvailab/wangjh/embodied-action-v2-task-card-full-20261106-r06/embodiedRunner/manipulation-runtime/coppeliasim"
    if resume_capture:
        recovery = '''
import base64,hashlib,json,os
from pathlib import Path
base=Path("/home/cvailab/wangjh/embodied-groundingdino-probe-20261107-r01")
work=base/"manipulation-raw"
rgb=(work/"rgb.png").read_bytes();depth=(work/"depth.png").read_bytes()
owned=[]
for p in Path("/proc").iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:cmd=(p/"cmdline").read_bytes();cwd=str((p/"cwd").resolve())
 except OSError:continue
 if cmd and (str(base).encode() in cmd or cwd.startswith(str(base))):owned.append(int(p.name))
assert not owned
print(json.dumps({"status":"ok","task":{"environment":"manipulation","subset":"base","index":1},
 "instruction":"Pick up the star and place it into the blue container.",
 "rgb_base64":base64.b64encode(rgb).decode(),"depth_base64":base64.b64encode(depth).decode(),
 "rgb_sha256":hashlib.sha256(rgb).hexdigest(),"depth_sha256":hashlib.sha256(depth).hexdigest(),
 "model_requests":0,"environment_steps":0,"simulator_launches":1,"raw_unannotated_rgb":True,
 "same_observation":True,"remote_directory":str(work),"active_owned_workers":owned,
 "metadata_source":"executed capture script and frozen task identity",
 "capture_protocol_note":"First capture exited 0 with trailing native stdout; recovered saved images without recapture."}))
'''
        capture = remote(recovery, timeout=30)
    else:
        capture = remote((OUT / "capture_raw.py").read_text(),
                         python="/home/cvailab/wangjh/embodiedRunner/manipulation-runtime/venv/bin/python",
                         prefix=("env", "LD_LIBRARY_PATH=" + sim, "COPPELIASIM_ROOT=" + sim,
                                 "QT_QPA_PLATFORM_PLUGIN_PATH=" + sim), timeout=180)
    png, depth = base64.b64decode(capture.pop("rgb_base64")), base64.b64decode(capture.pop("depth_base64"))
    assert sha(png) == capture["rgb_sha256"] and sha(depth) == capture["depth_sha256"]
    assert capture["environment_steps"] == capture["model_requests"] == 0
    assert not capture["active_owned_workers"]
    (folder / "manipulation-rgb.png").write_bytes(png)
    (folder / "manipulation-depth.png").write_bytes(depth)
    save(OUT / "raw-capture-receipt.json", capture)
    cases.append({"environment": "manipulation", "task": capture["task"],
                  "instruction": capture["instruction"], "serial": 0, "step": 0,
                  "source_directory": capture["remote_directory"], "raw_unannotated_rgb": True,
                  "rgb_sha256": sha(png), "depth_sha256": sha(depth),
                  "target_prompt": "star shaped block. container.",
                  "broad_prompt": "star shaped block. triangular prism. cube. block. container. tray. box. robot arm. table."})
    for case in cases:
        image = Image.open(folder / f"{case['environment']}-rgb.png")
        assert image.size == (500, 500) and image.mode == "RGB"
    save(OUT / "inputs.json", cases)
    return cases


def iou(a, b):
    ax0, ay0, ax1, ay1 = a["bbox"]
    bx0, by0, bx1, by1 = b["bbox"]
    intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return intersection / union if union > 0 else 0


def filtered(rows):
    kept = []
    for row in sorted(rows, key=lambda r: -r["confidence"]):
        if row["confidence"] < 0.35:
            continue
        if any(row["label"] == old["label"] and iou(row, old) >= 0.8 for old in kept):
            continue
        kept.append(row)
    return kept


def annotate(kind, rows, name):
    original = Image.open(OUT / "inputs" / f"{kind}-rgb.png").convert("RGB")
    display = original.copy()
    draw = ImageDraw.Draw(display)
    f = font(14)
    for row in rows:
        x0, y0, x1, y1 = row["bbox"]
        assert x0 < x1 and y0 < y1
        draw.rectangle((x0, y0, x1, y1), outline="#E02875", width=2)
        label = str(row["id"])
        text_box = draw.textbbox((0, 0), label, font=f)
        width = text_box[2] + 6
        x, y = min(max(0, x0), 500 - width), min(max(0, y0 - 19), 480)
        draw.rectangle((x, y, x + width, y + 19), fill="#FFFFFF")
        draw.text((x + 3, y), label, font=f, fill="#A00044")
    path = OUT / "images" / f"{kind}-{name}.png"
    display.save(path)
    changed = float(np.mean(np.any(np.asarray(display) != np.asarray(original), axis=2)))
    overlaps = sum(iou(a, b) >= 0.75 for i, a in enumerate(rows) for b in rows[i + 1:])
    return {"image": str(path), "count": len(rows), "high_overlap_pairs_iou075": overlaps,
            "annotation_changed_pixel_ratio": changed,
            "candidate_legend": [{"id": r["id"], "label": r["label"], "confidence": r["confidence"],
                                  "bbox": r["bbox"]} for r in rows]}


def render(cases, results):
    overview = Image.new("RGB", (2000, 590 * len(cases)), "white")
    draw = ImageDraw.Draw(overview)
    for row, case in enumerate(cases):
        kind = case["environment"]
        configs = results[kind]
        items = [
            ("Original RGB", OUT / "inputs" / f"{kind}-rgb.png"),
            (f"Broad query / 0.15: {configs['broad_low']['count']} boxes", Path(configs["broad_low"]["image"])),
            (f"Task query / 0.15: {configs['target_low']['count']} boxes", Path(configs["target_low"]["image"])),
            (f"Task query / 0.35: {configs['target_high']['count']} boxes", Path(configs["target_high"]["image"])),
        ]
        draw.text((12, row * 590 + 7), kind.upper(), font=font(22), fill="black")
        for col, (label, image) in enumerate(items):
            overview.paste(Image.open(image).convert("RGB"), (col * 500, row * 590 + 42))
            draw.text((col * 500 + 10, row * 590 + 549), label, font=font(17), fill="black")
        overview.crop((0, row * 590, 2000, (row + 1) * 590)).save(OUT / "images" / f"{kind}-comparison.png")
        ab = Image.new("RGB", (1500, 550), "white")
        ad = ImageDraw.Draw(ab)
        for col, (label, image) in enumerate((
            ("Original RGB", OUT / "inputs" / f"{kind}-rgb.png"),
            (f"All broad boxes: {configs['broad_low']['count']}", Path(configs["broad_low"]["image"])),
            (f"Score >=0.35 + same-label NMS: {configs['broad_filtered']['count']}", Path(configs["broad_filtered"]["image"])),
        )):
            ab.paste(Image.open(image).convert("RGB"), (col * 500, 0))
            ad.text((col * 500 + 8, 510), label, font=font(16), fill="black")
        ab.save(OUT / "images" / f"{kind}-density-comparison.png")
    overview.save(OUT / "groundingdino-comparison.png")


def main():
    assert not (OUT / "detections.json").exists(), "detector_run_already_exists"
    before = call_service({"mode": "status"})
    save(OUT / "detector-preflight.json", before)
    cases = prepare(resume_capture="--resume-capture" in sys.argv)
    (OUT / "images").mkdir()
    (OUT / "responses").mkdir()
    results, requests = {}, 0
    for case in cases:
        kind = case["environment"]
        encoded = base64.b64encode((OUT / "inputs" / f"{kind}-rgb.png").read_bytes()).decode()
        results[kind] = {}
        for name, prompt, threshold in (
            ("broad_low", case["broad_prompt"], 0.15),
            ("target_low", case["target_prompt"], 0.15),
            ("target_high", case["target_prompt"], 0.35),
        ):
            payload = {"image": encoded, "text_prompt": prompt, "box_threshold": threshold}
            reply = call_service({"mode": "infer", "payload": payload})
            assert reply["model_sha256"] == before["model_sha256"]
            assert reply["request_sha256"] == sha(canonical(payload))
            requests += 1
            save(OUT / "responses" / f"{kind}-{name}.json", reply)
            rows = [{"id": i + 1, **r} for i, r in enumerate(reply["detections"])]
            result = annotate(kind, rows, name)
            result.update(prompt=prompt, box_threshold=threshold, elapsed_s=reply["elapsed_s"],
                          request_sha256=reply["request_sha256"], response_sha256=reply["response_sha256"])
            results[kind][name] = result
            if name == "broad_low":
                results[kind]["broad_filtered"] = annotate(kind, filtered(rows), "broad_filtered")
            print(json.dumps({"environment": kind, "setting": name, "boxes": len(rows),
                              "high_overlap_pairs": result["high_overlap_pairs_iou075"],
                              "elapsed_s": reply["elapsed_s"]}), flush=True)
    after = call_service({"mode": "status"})
    assert before["model_sha256"] == after["model_sha256"]
    save(OUT / "detector-final-identity.json", after)
    report = {"schema": "embodied-groundingdino-screenshot-probe/1", "status": "verified",
              "checked_at": datetime.now(timezone.utc).isoformat(), "detector_model": "GroundingDINO_SwinB",
              "model_sha256": before["model_sha256"], "physical_gpu": 1,
              "detector_inference_requests": requests, "qwen_requests": 0,
              "automatic_retry": False, "environment_actions": 0,
              "inputs": cases, "results": results}
    render(cases, results)
    save(OUT / "detections.json", report)
    lines = ["# 环境截图 GroundingDINO 实测\n",
             "实际运行 GroundingDINO-SwinB 12 次，使用已加载的物理 GPU 1 副本。"
             "四张 500×500 原始 RGB 图，输入 PNG 原始像素；三个已有运行观测和一张新采集的无标注 Manipulation 截图。"
             "没有执行环境动作，没有自动重试。\n",
             "| 环境 | 宽泛查询 0.15 | 问题查询 0.15 | 问题查询 0.35 | 宽泛结果本地过滤 |",
             "|---|---:|---:|---:|---:|"]
    for kind, value in results.items():
        lines.append(f"| {kind} | {value['broad_low']['count']} | {value['target_low']['count']} | "
                     f"{value['target_high']['count']} | {value['broad_filtered']['count']} |")
    lines += ["\n本地过滤使用置信度 ≥0.35、仅相同标签做 IoU≥0.8 的 NMS，无额外检测调用；"
              "高重叠框并不等于重复物体，框数量也不等于物体数量。\n"]
    for case in cases:
        kind = case["environment"]
        lines += [f"## {kind}\n", case["instruction"] + "\n",
                  f"问题查询：`{case['target_prompt']}`。\n",
                  f"[原图与三配置比较](computer://{OUT / 'images' / (kind + '-comparison.png')}) · "
                  f"[全框与过滤图](computer://{OUT / 'images' / (kind + '-density-comparison.png')})\n"]
    lines += [f"\n[四环境总览](computer://{OUT / 'groundingdino-comparison.png'}) · "
              f"[原始框、标签、分数及完整检测记录](computer://{OUT / 'detections.json'})\n",
              "这一阶段直接检查检测图。框数量对 Qwen 判断的影响需另读同一帧的密集/过滤图，"
              "不能仅根据框数量推出任务成功率变化。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "verified", "detector_requests": requests,
                      "counts": {k: {n: r["count"] for n, r in v.items()} for k, v in results.items()}}), flush=True)


if __name__ == "__main__":
    main()
