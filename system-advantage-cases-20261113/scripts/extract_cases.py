"""Normalize selected historical contexts; preserve images as exact local assets."""
import base64
import hashlib
import io
import json
import pathlib
import re

from PIL import Image

TASK = pathlib.Path(__file__).resolve().parents[1]
BASE = pathlib.Path("/Users/bytedance/wangjh")
SELECT = json.loads((TASK / "review/selection.json").read_text())
ASSETS = {}
CASES = []

def read(p):
    return json.loads(pathlib.Path(p).read_text())

def save_json(p, obj):
    pathlib.Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def asset_bytes(data, mime=None):
    digest = hashlib.sha256(data).hexdigest()
    if digest not in ASSETS:
        image = Image.open(io.BytesIO(data))
        fmt = image.format
        ext = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}.get(fmt, "png")
        name = f"{digest}.{ext}"
        (TASK / "assets" / name).write_bytes(data)
        ASSETS[digest] = {"file": "assets/" + name, "mime": mime or Image.MIME.get(fmt, "image/png"),
                          "width": image.width, "height": image.height, "bytes": len(data)}
    return digest

def asset_file(p):
    return asset_bytes(pathlib.Path(p).read_bytes())

def norm(obj):
    if isinstance(obj, str):
        if obj.startswith("data:image/") and "," in obj:
            return {"asset": asset_bytes(base64.b64decode(obj.split(",", 1)[1]),
                                         obj.split(";", 1)[0].split(":", 1)[1])}
        return obj
    if isinstance(obj, list):
        return [norm(x) for x in obj]
    if isinstance(obj, dict):
        return {k: norm(v) for k, v in obj.items()}
    return obj

def find_messages(v):
    if isinstance(v, dict):
        if isinstance(v.get("messages"), list):
            return v["messages"]
        for x in v.values():
            got = find_messages(x)
            if got is not None:
                return got
    elif isinstance(v, list):
        for x in v:
            got = find_messages(x)
            if got is not None:
                return got
    return None

def decode_body(data):
    body = data.get("body", data)
    if isinstance(body, dict) and "base64" in body:
        return json.loads(base64.b64decode(body["base64"]))
    return body

def image_ids(obj):
    out = []
    def walk(v):
        if isinstance(v, dict):
            if "asset" in v:
                out.append(v["asset"])
            else:
                for x in v.values():
                    walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(obj)
    return list(dict.fromkeys(out))

def gallery(messages, label="输入图片"):
    return [{"asset": h, "label": f"{label} {i+1}", "kind": "input"} for i, h in enumerate(image_ids(messages))]

def item(ds, qid, question, options, reference, run):
    return {"dataset": ds, "id": qid, "question": question, "options": options,
            "reference": reference, "run": run, "gallery": [], "arms": {},
            "evidence": {}, "audit": {}}

def arm(answer, model, rounds, notes=None):
    return {"answer": answer, "model": model, "rounds": rounds, "notes": notes or []}

def round_record(messages=None, response="", reasoning="", tools=None, mode="exact", meta=None):
    return {"messages": norm(messages or []), "response": response or "", "reasoning": reasoning or "",
            "tools": norm(tools or []), "request_mode": mode, "metadata": meta or {}}

def extract_blink():
    root = BASE / "wjhICLR/GPTexp/BLINKGPT/runs/gpt56terra-blink-mv-sr-20261109-r01/final"
    manifest = read(root / "manifest.json")
    jobs = {j["job_id"]: j for j in manifest["jobs"]}
    eps = [read(p) for p in (root / "episodes").glob("*.json")]
    refs = {r["sample_id"]: r["reference"] for r in read(
        BASE / "outputs/BLINK_SR_v2_test100_20261105_r01/results_and_full_trajectories.json")["pairs"]}
    for qid in SELECT["BLINK"]:
        pair = {e["arm"]: e for e in eps if e["sample_id"] == qid and e["dataset"] == "sr"}
        job = jobs[pair["harness"]["job_id"]]
        problem = job["spec"]["problem"]
        c = item("BLINK", qid, problem["question"], problem["options"], refs[qid], root.parent.name)
        for label, name in [("system", "harness"), ("direct", "direct")]:
            e = pair[name]
            rounds = []
            intents = sorted((root / "requests" / e["job_id"]).glob("*/intent.json"))
            for i, intent in enumerate(intents):
                data = read(intent)
                r = e["result"]["rounds"][i]
                rounds.append(round_record(data["messages"], r.get("model_response", ""),
                                           r.get("reasoning_content", ""), mode="exact",
                                           meta={"phase": r.get("phase"), "request_parameters": r.get("request_parameters"),
                                                 "payload_sha256": data["payload_sha256"]}))
            c["arms"][label] = arm(e["result"]["answer"], "gpt-5.6-terra · high", rounds,
                                  ["模型在回答 JSON 中保留了简短推理；隐藏 thinking 未返回。"])
            if label == "direct":
                c["gallery"] = gallery(rounds[0]["messages"], "原始照片")
        c["evidence"] = {"input_sha256": problem["input_sha256"], "method": "SR Skill 引导的视觉参照关系判断",
                         "tools_used": pair["harness"]["result"].get("attempted_tool_calls", 0),
                         "original_input_images": job["images"]}
        CASES.append(c)

def extract_cv3d():
    root = BASE / "outputs/harness3d_four_datasets_20261031/CV3D精简Skill取证_20261101"
    for qid in SELECT["CV3D"]:
        path = next(root.glob(f"*/{qid}/case.json"))
        data = read(path)
        c = item("CV3D", qid, data["official_prompt"], [], data["offline_reference_answer"], root.name)
        rgb, depth = asset_file(path.parent / "original.png"), asset_file(path.parent / "depth.png")
        c["gallery"] = [{"asset": rgb, "label": "原始 RGB 与题目标注框", "kind": "rgb"},
                        {"asset": depth, "label": "模型所见预测深度图", "kind": "depth"}]
        for label, name in [("system", "tools"), ("direct", "direct")]:
            responses = sorted((path.parent / name).glob("*.response.json"))
            rounds = []
            history = [{"role": "user", "content": [{"type": "text", "text": data["official_prompt"]},
                        {"type": "image", "asset": rgb}]}]
            if name == "tools":
                history[0]["content"].append({"type": "image", "asset": depth})
            for p in responses:
                response = read(p)
                msg = response.get("message", {})
                tools_path = p.with_name(p.name.replace(".response.json", ".tools.json"))
                tools = read(tools_path) if tools_path.exists() else []
                rounds.append(round_record(history, msg.get("content", ""), msg.get("reasoning_content", ""),
                                           tools, mode="restored",
                                           meta={"source_response": p.name, "finish_reason": response.get("finish_reason")}))
                history = history + [{"role": "assistant", "content": msg.get("content", ""),
                                     "tool_calls": msg.get("tool_calls", [])}]
                for t in tools:
                    history.append({"role": "tool", "name": t["name"], "content": json.dumps(t["response"], ensure_ascii=False)})
            c["arms"][label] = arm(data["results"][name]["answer"], "qwen3.8-flash · thinking", rounds,
                                  ["原始响应、thinking 和工具返回均已保存。请求正文由原题、原图与逐轮工具记录恢复；原系统提示正文未保存在该取证包中。"])
        c["evidence"] = {"tool_calls": data["results"]["tools"]["attempted_tool_calls"],
                         "input_identity": data.get("input_identity"),
                         "final_system_explanation": data["results"]["tools"]["final_text"],
                         "final_direct_explanation": data["results"]["direct"]["final_text"]}
        CASES.append(c)

def extract_mmsi():
    sysroot = BASE / "MMSI六类四十题拍摄位置关系-20261109-r01"
    directroot = BASE / "MMSI六类四十题原图直答-20261109-r01"
    systems = {r["sample_id"]: r for r in read(sysroot / "score.json")["rows"]}
    directs = {r["sample_id"]: r for r in read(directroot / "score.json")["rows"]}
    siman = {r["problem"]["sample_id"]: r for r in read(sysroot / "manifest.json")["items"]}
    diman = {r["problem"]["sample_id"]: r for r in read(directroot / "manifest.json")["items"]}
    for qid in SELECT["MMSI"]:
        r = systems[qid]
        si, di = siman[qid], diman[qid]
        problem = si["problem"]
        c = item("MMSI", qid, r["question"], r["options"], r["reference_answer"], sysroot.name)
        images = []
        for v, im in enumerate(si["images"]):
            p = sysroot / "gpu/images" / f"{qid}-v{v+1}.jpg"
            assert hashlib.sha256(p.read_bytes()).hexdigest() == im["sha256"]
            images.append(asset_file(p))
        c["gallery"] = [{"asset": h, "label": f"Photo {i+1} · 原始输入", "kind": "rgb"} for i, h in enumerate(images)]
        options = "\n".join(f"({chr(65+i)}) {x}" for i, x in enumerate(problem["options"]))
        for label, source, meta in [("system", sysroot, si), ("direct", directroot, di)]:
            result = None
            for p in source.glob("**/*.result.json"):
                data = read(p)
                if data.get("sample_id") == qid and data.get("status") == "answered":
                    result = data
                    break
            assert result is not None
            text = problem["question"] + "\n" + options
            if label == "system":
                text += "\n\n" + si["relation_text"]
            content = [{"type": "text", "text": text}]
            for i, h in enumerate(images):
                content += [{"type": "text", "text": f"Photo {i+1}"}, {"type": "image", "asset": h}]
            rr = result["response"]
            rounds = [round_record([{"role": "user", "content": content}], rr["text"],
                                   rr.get("reasoning_summary", ""), mode="restored",
                                   meta={"input_sha256": problem["input_sha256"], "payload_sha256": meta["payload_sha256"],
                                         "reported_model": rr["reported"], "usage": rr["usage"]})]
            c["arms"][label] = arm(result["answer"], "gpt-5.6-terra · high", rounds,
                                  ["模型只返回选项字母，隐藏 thinking 未保存。题目、原图和关系提示按冻结 manifest 恢复；正文末尾的原始字母回答指令未逐字归档。"])
        c["evidence"] = {"relation_text": si["relation_text"], "relation_rows": si["relation_rows"],
                         "pose": read(sysroot / si["pose_path"]), "same_image_sha256": si["images"] == di["images"],
                         "system_correct": r["correct"], "direct_correct": directs[qid]["direct_correct"]}
        CASES.append(c)

def extract_omni():
    inspected = read(TASK / "review/omni-candidates.json")
    for qid in SELECT["Omni3D"]:
        info = next(x for x in inspected if x["qid"] == qid)
        root = pathlib.Path(info["root"])
        scores = info["scores"]
        c = item("Omni3D", qid, info["input"]["question"], info["input"].get("options", []), scores["skill"]["prediction"], root.name)
        for label, name in [("system", "skill"), ("direct", "direct")]:
            ep = root / "episodes" / f"{qid}-{name}"
            rounds = []
            for p in sorted(ep.glob("*.messages.json")):
                messages = read(p)
                if isinstance(messages, dict):
                    messages = find_messages(messages)
                rp = p.with_name(p.name.replace(".messages.json", ".response.json"))
                xp = p.with_name(p.name.replace(".messages.json", ".execution.json"))
                resp = read(rp) if rp.exists() else {}
                execution = read(xp) if xp.exists() else {}
                rounds.append(round_record(messages, resp.get("output", ""), resp.get("thinking", ""),
                                           [execution] if execution else [], mode="exact",
                                           meta={"phase": resp.get("phase"), "request_parameters": resp.get("parameters")}))
            c["arms"][label] = arm(scores[name]["prediction"], "qwen3.8-flash · thinking", rounds)
            if label == "direct":
                c["gallery"] = gallery(rounds[0]["messages"], "原始 RGB")
        for g in gallery(c["arms"]["system"]["rounds"][0]["messages"], "系统输入"):
            if g["asset"] not in {x["asset"] for x in c["gallery"]}:
                g["kind"] = "depth"
                c["gallery"].append(g)
        c["evidence"] = {"scores": scores, "input_identity": info["input"],
                         "tool_calls": scores["skill"].get("tool_calls", 0)}
        CASES.append(c)

def extract_mindcube():
    root = BASE / "outputs/harness3d_four_datasets_20261031/mindcube-tiny-test700-complete34-20261104-r01"
    qpath = BASE / "outputs/harness3d_four_datasets_20261031/mindcube-tiny-test700-paired-20261104-r01/test_questions.jsonl"
    questions = {r["sample_id"]: r for r in map(json.loads, qpath.read_text().splitlines())}
    comparisons = {r["sample_id"]: r for r in read(root / "comparison.json")["pairs"]}
    selected = {}
    for qid in SELECT["MindCube"]:
        q = questions[qid]
        ref = comparisons[qid]["reference_answer"]
        c = item("MindCube", qid, q["question"], q["options"], ref, root.name)
        c["evidence"] = {"input_sha256": q["input_sha256"], "scores": comparisons[qid]["arms"]}
        selected[qid] = c
    with (root / "experiment_trace.jsonl").open() as fp:
        for line in fp:
            match = re.search(r'"sample_id"\s*:\s*"([^"]+)"', line[:512])
            if not match or match.group(1) not in selected:
                continue
            raw = json.loads(line)
            c = selected[raw["sample_id"]]
            label = "system" if raw["arm"] == "skill" else "direct"
            rounds = []
            for rt in raw["round_traces"]:
                messages, response, reasoning, tools = [], "", "", []
                meta = {"phase": rt["identity"].get("round_index"), "config": rt["identity"].get("config")}
                for event in rt["events"]:
                    kind, data = event["kind"], event["data"]
                    if kind == "client_payload":
                        decoded = decode_body(data)
                        messages = find_messages(decoded) or []
                    elif kind == "response_body":
                        decoded = decode_body(data)
                        for choice in decoded.get("choices", []):
                            m = choice.get("message", {})
                            response += m.get("content", "") or ""
                            reasoning += m.get("reasoning_content", "") or ""
                    elif kind not in ("started", "state_before", "prepared", "admitted", "gateway_envelope",
                                     "request_sent", "assistant_content", "program"):
                        tools.append({"event": kind, "data": data})
                rounds.append(round_record(messages, response, reasoning, tools, meta=meta))
            c["arms"][label] = arm(raw["result"]["answer"], "qwen3.8-flash · thinking", rounds)
            c["evidence"][label + "_result"] = norm(raw["result"])
            if label == "direct":
                c["gallery"] = gallery(rounds[0]["messages"], "原始视角")
    for c in selected.values():
        assert set(c["arms"]) == {"direct", "system"}
        for g in gallery(c["arms"]["system"]["rounds"][0]["messages"], "系统增强输入"):
            if g["asset"] not in {x["asset"] for x in c["gallery"]}:
                g["kind"] = "depth"
                c["gallery"].append(g)
        CASES.append(c)

def extract_vsi():
    import cv2
    import numpy as np
    inspected = read(TASK / "review/vsi-candidates.json")
    for qid in SELECT["VSIbench"]:
        info = next(x for x in inspected if x["qid"] == qid)
        root = pathlib.Path(info["root"])
        outcome = read(info["outcome"])
        pair = info["pair"]
        scene = outcome["scene_name"]
        inputfile = pathlib.Path(info["outcome"]).parent / "trace/episode_input.jsonl"
        original = json.loads(inputfile.read_text().splitlines()[0])
        c = item("VSIbench", qid, original["question_text"], original["options"], pair["reference_answer"], root.name)
        records = [r for r in outcome["image_ledger"]["images"] if r["kind"] == "frame"]
        frame_dirs = list((TASK / "source").glob(scene + "*frames"))
        if frame_dirs:
            files = sorted(frame_dirs[0].glob("*.png"))
            arrays = [np.asarray(Image.open(p).convert("RGB")) for p in files]
        else:
            video = next((TASK / "source/videos").glob("*_" + scene + ".mp4"))
            wanted = {r["source_frame_index"] for r in records}
            cap = cv2.VideoCapture(str(video))
            arrays, i = [], 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if i in wanted:
                    arrays.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                i += 1
            cap.release()
        assert len(arrays) == len(records) == 32
        gallery_rows = []
        for a, r in zip(arrays, records):
            assert hashlib.sha256(a.tobytes()).hexdigest() == r["content_sha256"].split(":")[-1], (qid, r["image_id"])
            buffer = io.BytesIO()
            Image.fromarray(a).save(buffer, format="PNG")
            h = asset_bytes(buffer.getvalue())
            gallery_rows.append({"asset": h, "label": f"{r['image_id']} · 源帧 {r['source_frame_index']}", "kind": "rgb"})
        overview_path = root / "inputs/scene_overviews" / f"{scene}.png"
        if overview_path.exists():
            gallery_rows.append({"asset": asset_file(overview_path), "label": "overview · 系统场景总览", "kind": "overview"})
        c["gallery"] = gallery_rows
        base_content = [{"type": "text", "text": original["question_text"] + "\n" + "\n".join(
            f"({chr(65+i)}) {x}" for i, x in enumerate(original["options"]))}]
        for g in gallery_rows:
            if g["kind"] != "overview":
                base_content += [{"type": "text", "text": g["label"]}, {"type": "image", "asset": g["asset"]}]
        sr = []
        for r in outcome["rounds"]:
            content = list(base_content)
            for g in gallery_rows:
                if g["kind"] == "overview":
                    content += [{"type": "text", "text": g["label"]}, {"type": "image", "asset": g["asset"]}]
            messages = [{"role": "system", "content": outcome["m8_prompt"]}, {"role": "user", "content": content}]
            if r.get("feedback"):
                messages.append({"role": "user", "content": r["feedback"]})
            sr.append(round_record(messages, r.get("response_text", ""), tools=r.get("tool_calls", []),
                                   mode="restored", meta={"round": r["index"], "observation_reads": r.get("observation_reads"),
                                                          "results": norm(r.get("results", [])),
                                                          "generation": norm(r.get("generation", {}))}))
        direct = next(read(p) for p in (root / "direct/responses").glob("*.json") if str(read(p).get("qa_id")) == qid)
        c["arms"]["system"] = arm(pair["system_answer"], "deepseek-v4.1-flash · max", sr,
                                 ["原始程序、反馈、工具结果、系统提示和 32 张冻结帧均保留；请求消息按逐轮记录恢复，原始序列化正文未归档。隐藏 thinking 未保存在本次回收记录。"])
        c["arms"]["direct"] = arm(pair["direct_answer"], "deepseek-v4.1-flash · low", [
            round_record([{"role": "user", "content": base_content}], direct.get("response_text", ""), mode="restored",
                         meta={"generation": norm(direct.get("generation", {}))})],
            ["直答响应已保存，图片与系统的 32 张原始帧一致；完整直答提示末尾指令未逐字归档，隐藏 thinking 未保存在本次回收记录。",
             "官方文档说明 low 兼容映射为 high；本次两臂的 effort 标签和工具输入不同。"])
        c["evidence"] = {"tool_results": norm(pair["system_tool_results"]), "image_ledger": outcome["image_ledger"],
                         "attribution": outcome["attribution"], "derivation_replay": outcome["episode_trace"].get("derivation_replay"),
                         "frozen_frame_pixels_verified": 32, "same_frozen_rgb_identity": pair["same_frozen_rgb_identity"],
                         "outcome_source": str(pathlib.Path(info["outcome"]).relative_to(BASE))}
        CASES.append(c)

if __name__ == "__main__":
    extract_blink()
    extract_cv3d()
    extract_mmsi()
    extract_omni()
    extract_mindcube()
    extract_vsi()
    assert len(CASES) == 30
    for ds in SELECT:
        save_json(TASK / "build" / f"cases-{ds}.json", [c for c in CASES if c["dataset"] == ds])
    save_json(TASK / "build/assets.json", ASSETS)
    save_json(TASK / "build/cases.json", CASES)
    print(json.dumps({"cases": len(CASES), "assets": len(ASSETS),
                      "asset_bytes": sum(x["bytes"] for x in ASSETS.values()),
                      "by_dataset": {k: sum(c["dataset"] == k for c in CASES) for k in SELECT}}, ensure_ascii=False))
