"""Archive twenty existing successes and embed their exact recorded contexts."""
import base64
import hashlib
import io
import json
import pathlib
import shutil
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
SELECTION = {
    "manipulation": [9, 22, 2, 8, 38],
    "navigation": [27, 6, 23, 46, 58],
    "alfred": [2, 15, 0, 1, 38],
    "habitat": [33, 25, 8, 43, 35],
}
LABELS = {"alfred": "EB-ALFRED", "habitat": "EB-Habitat",
          "manipulation": "EB-Manipulation", "navigation": "EB-Navigation"}
inventory = json.loads((ROOT / "review/embodied-candidate-inventory.json").read_text())
assets = {}
archive = ROOT / "source/embodied"
image_dir = ROOT / "assets/embodied"
archive.mkdir(exist_ok=True)
image_dir.mkdir(exist_ok=True)
shutil.copyfile("/Users/bytedance/wangjh/具身最小提示词剩余104题并发合并156题-20261109-r02/results.json",
                archive / "combined-results.json")

def asset(raw, expected=None):
    h = hashlib.sha256(raw).hexdigest()
    if expected:
        assert h == expected, ("image identity", h, expected)
    if h not in assets:
        im = Image.open(io.BytesIO(raw))
        extension = ".png" if im.format == "PNG" else ".jpg"
        out = image_dir / (h + extension)
        if not out.exists():
            out.write_bytes(raw)
        assets[h] = {"file": str(out.relative_to(ROOT)), "mime": Image.MIME[im.format],
                     "width": im.width, "height": im.height, "bytes": len(raw)}
    return h

def normalize(value, directory, depth_path=None):
    if isinstance(value, str) and value.startswith("data:image/"):
        prefix, encoded = value.split(",", 1)
        return {"asset": asset(base64.b64decode(encoded)), "original_data_url_prefix": prefix}
    if isinstance(value, list):
        return [normalize(v, directory, depth_path) for v in value]
    if not isinstance(value, dict):
        return value
    if "base64" in value:
        raw = base64.b64decode(value["base64"])
        h = hashlib.sha256(raw).hexdigest()
        assert not value.get("sha256") or h == value["sha256"]
        result = {k: normalize(v, directory, depth_path) for k, v in value.items() if k != "base64"}
        if raw.startswith((b"\x89PNG", b"\xff\xd8")):
            result["asset"] = asset(raw, value.get("sha256"))
        else:
            result["archived_numeric_payload"] = {
                "sha256": h, "bytes": len(raw), "file": depth_path,
                "display": "完整数值保存在制作资料的原始轨迹及 depth-meters.npy；页面显示同帧深度图、统计与来源。",
            }
        return result
    result = {k: normalize(v, directory, depth_path) for k, v in value.items()}
    if isinstance(value.get("file"), str) and value["file"].endswith((".png", ".jpg")):
        result["asset"] = asset((directory / value["file"]).read_bytes(), value.get("sha256"))
    return result

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

cases, audits = [], []
for env, indices in SELECTION.items():
    for index in indices:
        c = next(x for x in inventory if x["task"]["environment"] == env and x["task"]["index"] == index)
        source = pathlib.Path(c["directory"])
        target = archive / source.name
        shutil.copytree(source, target, dirs_exist_ok=True)
        assert c["result_hash_matches"]
        raw_rows = [json.loads(s) for s in (target / "trajectory.jsonl").read_text().splitlines()]
        assert hashlib.sha256((target / "trajectory.jsonl").read_bytes()).hexdigest() == c["audit"]["trajectory_sha256"]
        observations, requests, events = [], [], []
        sent, actions, last_obs = None, [], None
        for raw in raw_rows:
            depth_path = str((target / f'observation-{raw.get("serial", 0):03d}-depth-meters.npy').relative_to(ROOT))
            row = normalize(raw, target, depth_path)
            events.append(row)
            kind = raw["type"]
            if kind == "observation":
                obs = {"serial": raw["serial"], **row["public"], "official": row["official"],
                       "input_sha256": row["input_sha256"]}
                observations.append(obs)
                last_obs = obs
            elif kind == "model_request":
                req_file = raw["request_file"]
                payload = json.loads((target / req_file).read_text())
                payload_hash = hashlib.sha256(canonical(payload)).hexdigest()
                assert payload_hash == raw["payload_sha256"], (c["identity"], req_file, "payload")
                response = json.loads((target / f'response-{raw["request_index"]:03d}.json').read_text())
                requests.append({**row, "payload": normalize(payload, target),
                                 "response": normalize(response, target),
                                 "stage": raw.get("stage", "planning")})
            elif kind == "action_sent":
                assert sent is None
                old_obs = last_obs
                action = raw["action"]
                label = old_obs["actions"][action] if isinstance(action, int) else "机械臂位姿 " + json.dumps(action)
                sent = {**row, "name": label, "before_serial": old_obs["serial"]}
            elif kind == "action_ack":
                assert sent is not None and raw["action"] == sent["action"]
                actions.append({**sent, "number": len(actions) + 1,
                                "after_serial": last_obs["serial"], "feedback": last_obs["env_feedback"],
                                "succeeded": row["last_action_success"], "official": row["official"]})
                sent = None
        assert sent is None and len(actions) == c["audit"]["actions"] == c["audit"]["acks"]
        assert actions[-1]["official"]["task_success"] == 1 and actions[-1]["official"]["done"]
        assert observations[0]["official"]["task_success"] == 0
        # All environment PNGs, including secondary cameras and detector overlays, remain browseable.
        gallery = [{"asset": asset(p.read_bytes()), "label": p.name}
                   for p in sorted(target.glob("*.png"))]
        case = {"kind": "embodied", "dataset": LABELS[env], "benchmark": "EmbodiedBench",
                "environment": env, "id": c["identity"], "question": c["instruction"],
                "run": str(source.parent.parent.name), "observations": observations,
                "requests": requests, "actions": actions, "events": events, "gallery": gallery,
                "result": json.loads((target / "result.json").read_text()),
                "source": {"original_directory": str(source), "archived_directory": str(target.relative_to(ROOT)),
                           "origin": c["origin"], "result_sha256": c["audit"]["result_sha256"],
                           "trajectory_sha256": c["audit"]["trajectory_sha256"]},
                "audit": {"title": c["instruction"], "mechanism": "观察 → 规划 → 执行",
                          "why": "", "limits": "已有系统成功轨迹；本批无 Direct 对照。"}}
        cases.append(case)
        audits.append({"id": c["identity"], "environment": env, "instruction": c["instruction"],
                       "initial_success": observations[0]["official"]["task_success"],
                       "final_success": actions[-1]["official"]["task_success"],
                       "official_done": True, "actions": len(actions), "acks": len(actions),
                       "observations": len(observations), "exact_requests": len(requests),
                       "result_hash_verified": True, "trajectory_hash_verified": True,
                       "all_request_hashes_verified": True,
                       "failed_actions": [a["number"] for a in actions if not a["succeeded"]],
                       "final_distance": case["result"]["official"].get("distance")})
        print(c["identity"], "requests", len(requests), "observations", len(observations), flush=True)
(ROOT / "build/embodied-cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
(ROOT / "build/embodied-assets.json").write_text(json.dumps(assets, ensure_ascii=False, indent=2))
(ROOT / "review/embodied-selected-evidence.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2))

font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
for env in SELECTION:
    selected = [c for c in cases if c["environment"] == env]
    sheet = Image.new("RGB", (1000, 5 * 310), "#eff2f5")
    draw = ImageDraw.Draw(sheet)
    for i, c in enumerate(selected):
        first, last = c["observations"][0], c["observations"][-1]
        tool = next(e for e in c["events"] if e["type"] == "perception_tool_result")
        views = [first["images"][0]["asset"], tool["images"][0]["asset"], last["images"][0]["asset"]]
        y = i * 310
        draw.text((10, y + 5), c["id"] + " | " + c["question"][:88], font=font, fill="#102e43")
        for j, h in enumerate(views):
            im = Image.open(ROOT / assets[h]["file"]).convert("RGB")
            im.thumbnail((320, 250))
            sheet.paste(im, (j * 333 + (320 - im.width)//2, y + 35))
        draw.text((10, y + 286), "Initial RGB                    Initial detections                  Final RGB", font=font, fill="#102e43")
    sheet.save(ROOT / f"review/embodied-{env}-contact.png")
print("20 episodes extracted;", len(assets), "unique original images")
