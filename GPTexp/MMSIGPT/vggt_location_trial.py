"""Thirty saved VGGT cases, location-language input, bounded new requests."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import FunctionType

from ..traces import digest, file_digest, read, write
from . import vggt_pose_trial as base
from .depth_comparison import runner as shared
from .prototype.direct import operator_lock
from .prototype.inputs import load_images, public_problem
from .prototype.protocol import PROMPT_VERSION, PARSER_VERSION, messages, payload
from .vggt_location_relation import RELATION_VERSION, location_relations, relation_note

PARENT = base.ROOT.parent / "MMSI六类五题位姿实验-20261108-v2/continuation"
DESTINATION = base.ROOT.parent / "MMSI拍摄位置关系实验"
ARM = "relation"


def code_identity():
    files = (base.PACKAGE / name for name in (
        "vggt_location_relation.py", "vggt_location_trial.py", "depth_comparison/runner.py"))
    return {**base.code_identity(),
            **{str(path.relative_to(base.ROOT)): file_digest(path) for path in files}}


def source(parent):
    parent = Path(parent)
    manifest, frozen = read(parent / "manifest.json"), read(parent / "frozen.json")
    if (digest({k: v for k, v in manifest.items() if k != "run_identity"})
            != manifest["run_identity"]
            or digest({k: v for k, v in frozen.items() if k != "freeze_identity"})
            != frozen["freeze_identity"]
            or frozen["run_identity"] != manifest["run_identity"]
            or manifest["reference_answer_visible"] is not False
            or frozen["reference_answer_visible"] is not False
            or manifest["questions"] != 30 or len(manifest["items"]) != 30):
        raise ValueError("parent_identity_changed")
    if (len(frozen["predictions"]) != 60
            or any(row["status"] not in ("answered", "invalid_answer")
                   for row in frozen["predictions"])):
        raise ValueError("parent_not_complete")
    if any(file_digest(parent / name) != expected
           for name, expected in frozen["files"].items()):
        raise ValueError("parent_frozen_file_changed")
    tasks = Counter(item["problem"]["task_type"] for item in manifest["items"])
    if len(tasks) != 6 or set(tasks.values()) != {5}:
        raise ValueError("requires_six_types_five_each")
    if [item["index"] for item in manifest["items"]] != list(range(30)):
        raise ValueError("parent_order_changed")
    return manifest, frozen


def request(directory, item, arm):
    if arm != ARM:
        raise ValueError("only_location_relation_arm")
    content = messages(item["problem"], load_images(item["problem"], item["images"]))
    content[0]["content"].insert(-1, {"type": "text", "text": item["relation_text"]})
    return payload(content, base.settings(item["api_id"]))


def prepare(directory=DESTINATION, parent=PARENT):
    directory, parent = Path(directory), Path(parent)
    if directory.exists() or not directory.parent.is_dir():
        raise ValueError("requires_new_run_directory_with_existing_parent")
    old, old_frozen = source(parent)
    items = []
    for original in old["items"]:
        problem = public_problem(original["problem"])
        load_images(problem, original["images"])
        pose = read(parent / original["pose_path"])
        if (file_digest(parent / original["pose_path"]) != original["pose_sha256"]
                or pose["sample_id"] != problem["sample_id"]
                or pose["input_sha256"] != problem["input_sha256"]
                or pose["original_image_sha256"] != [i["sha256"] for i in original["images"]]
                or len(original["images"]) != 2 or pose["joint_forward"] is not True
                or pose["reference_answer_visible"] is not False
                or pose["weight_sha256"] != base.WEIGHT_SHA):
            raise ValueError("parent_camera_or_original_image_changed")
        # The original question and original image request remains exactly the same.
        if (digest(base.request(original, "direct")) != original["payload_sha256"]["direct"]
                or base.settings(original["api_id"]) != old["inference_by_api"][original["api_id"]]):
            raise ValueError("historical_direct_or_inference_changed")
        item = {
            "index": original["index"], "problem": problem, "images": original["images"],
            "api_id": original["api_id"], "arm_order": [ARM],
            "source_pose_path": original["pose_path"],
            "pose_path": original["pose_path"], "pose_sha256": original["pose_sha256"],
            "relation_rows": location_relations(pose["camera_evidence"]),
            "relation_text": relation_note(pose["camera_evidence"]),
            "historical_payload_sha256": original["payload_sha256"],
        }
        item["payload_sha256"] = {ARM: digest(request(None, item, ARM))}
        items.append(item)
    directory.mkdir()
    (directory / "poses").mkdir()
    for item in items:
        shutil.copyfile(parent / item["source_pose_path"], directory / item["pose_path"])
    apis = list(dict.fromkeys(item["api_id"] for item in items))
    manifest = {
        "schema": "mmsi-vggt-location-language-trial/1",
        "parent_directory": str(parent), "parent_run_identity": old["run_identity"],
        "parent_freeze_identity": old_frozen["freeze_identity"],
        "parent_files": {name: file_digest(parent / name) for name in ("manifest.json", "frozen.json")},
        "items": items, "questions": 30, "max_requests": 30, "max_inflight": len(apis),
        "api_ids": apis, "inference_by_api": {api: base.settings(api) for api in apis},
        "code_identity": code_identity(), "environment": base.environment(),
        "parent_environment": old["environment"],
        "prompt_version": PROMPT_VERSION, "parser_version": PARSER_VERSION,
        "relation_version": RELATION_VERSION,
        "sample_role": "previously_seen_development_diagnostic",
        "comparison": "one_new_relation_answer_against_saved_direct_and_numeric_pose_answers",
        "reference_answer_visible": False, "automatic_retry": False,
        "tools_enabled": False, "depth_images_sent": False,
        "numeric_pose_sent": False, "camera_orientation_relation_sent": False,
        "gpu_forwards": 0, "saved_vggt_estimates_reused": 30, "prepared_at": time.time(),
    }
    manifest["run_identity"] = digest(manifest)
    write(directory / "manifest.json", manifest, exclusive=True)
    write(directory / "preflight.json", {
        "run_identity": manifest["run_identity"], "model_requests": 0,
        "reference_answers_read": False, "original_images_verified": True,
        "historical_direct_payloads_verified": True, "numeric_pose_sent": False,
        "questions": 30, "gpu_forwards": 0,
    }, exclusive=True)
    preview = [
        "# 拍摄位置关系：实际附加给模型的文字",
        "图片仍按原题顺序发送；随后附加下面的关系文字，最后发送原题、原选项和官方单字母输出要求。",
        "每一段分别以对应照片的拍摄点为原点，以该照片相机拍摄时的观察方向为前方。"
        "描述左右和前后分量，并说明哪一分量更大；没有发送位置数值、旋转矩阵、朝向向量或相机夹角。",
    ]
    for item in items:
        preview += [f"## {item['problem']['sample_id']} · {item['problem']['task_type']}",
                    item["problem"]["question"], "```text", item["relation_text"], "```"]
    (directory / "提示预览.md").write_text("\n\n".join(preview) + "\n", encoding="utf-8")
    validate(directory, verify_requests=True)
    print(json.dumps({"status": "prepared", "questions": 30, "new_model_requests": 30,
                      "gpu_forwards": 0, "reference_answers_read": False,
                      "run_identity": manifest["run_identity"]}), flush=True)
    return manifest


def validate(directory, *, verify_requests=False):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    if (manifest["schema"] != "mmsi-vggt-location-language-trial/1"
            or digest({k: v for k, v in manifest.items() if k != "run_identity"})
            != manifest["run_identity"] or manifest["code_identity"] != code_identity()
            or manifest["environment"] != base.environment()
            or manifest["relation_version"] != RELATION_VERSION
            or manifest["prompt_version"] != PROMPT_VERSION
            or manifest["parser_version"] != PARSER_VERSION
            or manifest["questions"] != 30 or manifest["max_requests"] != 30
            or manifest["max_inflight"] != len(manifest["api_ids"])
            or manifest["inference_by_api"] != {
                api: base.settings(api) for api in manifest["api_ids"]}
            or any(manifest[k] is not False for k in (
                "reference_answer_visible", "automatic_retry", "tools_enabled",
                "depth_images_sent", "numeric_pose_sent", "camera_orientation_relation_sent"))):
        raise ValueError("new_trial_configuration_changed")
    parent = Path(manifest["parent_directory"])
    if any(file_digest(parent / name) != expected for name, expected in manifest["parent_files"].items()):
        raise ValueError("parent_identity_files_changed")
    old, frozen = source(parent)
    if (manifest["parent_run_identity"] != old["run_identity"]
            or manifest["parent_freeze_identity"] != frozen["freeze_identity"]
            or manifest["parent_environment"] != old["environment"]
            or len(manifest["items"]) != 30):
        raise ValueError("source_binding_changed")
    for item, original in zip(manifest["items"], old["items"]):
        pose = read(directory / item["pose_path"])
        if (any(item[key] != original[key] for key in ("index", "problem", "images", "api_id"))
                or item["source_pose_path"] != original["pose_path"]
                or item["pose_path"] != original["pose_path"]
                or item["pose_sha256"] != original["pose_sha256"]
                or file_digest(directory / item["pose_path"]) != item["pose_sha256"]
                or item["arm_order"] != [ARM]
                or item["relation_text"] != relation_note(pose["camera_evidence"])
                or item["relation_rows"] != location_relations(pose["camera_evidence"])
                or item["historical_payload_sha256"] != original["payload_sha256"]):
            raise ValueError("location_text_or_case_binding_changed")
        if verify_requests and digest(request(directory, item, ARM)) != item["payload_sha256"][ARM]:
            raise ValueError("prepared_location_request_changed")
    preflight = read(directory / "preflight.json")
    if (preflight["run_identity"] != manifest["run_identity"] or preflight["model_requests"] != 0
            or preflight["reference_answers_read"] is not False):
        raise ValueError("zero_request_preflight_changed")
    return manifest


_namespace = {**vars(shared), "validate": validate, "request": request, "inference": base.settings}
execute = FunctionType(shared.execute.__code__, _namespace, "execute")
execute.__kwdefaults__ = {"client_factory": base.BoundedClient}
_collect = FunctionType(shared.collect.__code__, {**vars(shared), "inference": base.settings}, "collect")


def collect(directory, manifest):
    adapter = {**manifest, "items": [
        {**item, "depth_images": [{"path": item["pose_path"], "sha256": item["pose_sha256"]}]}
        for item in manifest["items"]]}
    return _collect(directory, adapter)


def freeze(directory):
    directory = Path(directory)
    with operator_lock(directory):
        manifest = validate(directory)
        records, files = collect(directory, manifest)
        if (directory / "frozen.json").exists():
            return validate_frozen(directory, manifest)
        value = {
            "schema": "mmsi-location-prediction-freeze/1", "run_identity": manifest["run_identity"],
            "predictions": records, "files": files, "reference_answer_visible": False,
            "automatic_retry": False, "frozen_at": time.time(),
        }
        value["freeze_identity"] = digest(value)
        write(directory / "frozen.json", value, exclusive=True)
        return value


def validate_frozen(directory, manifest):
    frozen = read(Path(directory) / "frozen.json")
    records, files = collect(directory, manifest)
    if (digest({k: v for k, v in frozen.items() if k != "freeze_identity"}) != frozen["freeze_identity"]
            or frozen["run_identity"] != manifest["run_identity"]
            or frozen["predictions"] != records or frozen["files"] != files):
        raise ValueError("prediction_freeze_changed")
    return frozen


def controlled_execute(directory):
    from .depth_service_probe import all_keys
    manifest = validate(directory, verify_requests=True)
    available = all_keys()
    keys = {api: available[api] for api in manifest["api_ids"]}
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "GPTexp.MMSIGPT.vggt_location_trial",
         "execute", "--directory", str(directory)],
        input=json.dumps(keys).encode("utf-8"), env=base.clean_environment(), cwd=base.ROOT)
    if completed.returncode:
        raise ValueError("location_trial_worker_failed")


def score(directory):
    """Read the old GT-bearing score only after all new predictions are frozen."""
    directory = Path(directory)
    manifest = validate(directory)
    frozen = validate_frozen(directory, manifest)
    execution = read(directory / "execution.json")
    if execution["status"] != "complete" or execution["dispatched"] != 30:
        raise ValueError("incomplete_no_reference_access")
    parent = Path(manifest["parent_directory"])
    old_score = read(parent / "score.json")
    if (digest({k: v for k, v in old_score.items() if k != "score_identity"})
            != old_score["score_identity"]
            or old_score["run_identity"] != manifest["parent_run_identity"]
            or old_score["freeze_identity"] != manifest["parent_freeze_identity"]):
        raise ValueError("historical_score_identity_changed")
    old_frozen = read(parent / "frozen.json")
    old_predictions = {(p["sample_id"], p["arm"]): p for p in old_frozen["predictions"]}
    old_pairs = {p["sample_id"]: p for p in old_score["pairs"]}
    new_predictions = {p["sample_id"]: p for p in frozen["predictions"]}
    rows = []
    for item in manifest["items"]:
        sample = item["problem"]["sample_id"]
        old, new = old_pairs[sample], new_predictions[sample]
        if (old["input_sha256"] != item["problem"]["input_sha256"]
                or any(old[arm]["answer"] != old_predictions[(sample, arm)]["answer"]
                       for arm in ("direct", "pose"))):
            raise ValueError("historical_answers_changed")
        _, path = shared.paths(directory, item, ARM)
        result = read(path)
        reference = old["reference_answer"]
        rows.append({
            "sample_id": sample, "index": item["index"], "task_type": item["problem"]["task_type"],
            "input_sha256": item["problem"]["input_sha256"], "reference_answer": reference,
            "question": item["problem"]["question"], "options": item["problem"]["options"],
            "direct_answer": old["direct"]["answer"], "pose_answer": old["pose"]["answer"],
            "relation_answer": new["answer"], "relation_status": new["status"],
            "direct_correct": old["direct"]["answer"] == reference,
            "pose_correct": old["pose"]["answer"] == reference,
            "relation_correct": new["answer"] == reference,
            "relation_text": item["relation_text"], "relation_rows": item["relation_rows"],
            "reported": result["response"]["reported"],
            "usage": result["response"]["usage"],
        })

    def summarize(group):
        n = len(group)
        value = {"questions": n}
        for arm in ("direct", "pose", "relation"):
            value[f"{arm}_correct"] = sum(row[f"{arm}_correct"] for row in group)
            value[f"{arm}_accuracy"] = value[f"{arm}_correct"] / n
        value["relation_legal_answers"] = sum(row["relation_answer"] is not None for row in group)
        for other in ("direct", "pose"):
            value[f"gains_vs_{other}"] = sum(
                row["relation_correct"] and not row[f"{other}_correct"] for row in group)
            value[f"regressions_vs_{other}"] = sum(
                row[f"{other}_correct"] and not row["relation_correct"] for row in group)
            value[f"answers_changed_vs_{other}"] = sum(
                row["relation_answer"] != row[f"{other}_answer"] for row in group)
        return value

    value = {
        "schema": "mmsi-location-language-score/1", "run_identity": manifest["run_identity"],
        "freeze_identity": frozen["freeze_identity"], "source_score_identity": old_score["score_identity"],
        "new_model_requests": execution["dispatched"], "gpu_forwards": 0,
        "comparison": manifest["comparison"], "reference_answer_visible_during_solving": False,
        "metric": "extracted_letter_exact_match", "denominator": "all_30_selected_questions",
        "overall": summarize(rows),
        "by_task": {task: summarize([row for row in rows if row["task_type"] == task])
                    for task in dict.fromkeys(row["task_type"] for row in rows)},
        "rows": rows, "scored_at": time.time(),
    }
    value["score_identity"] = digest(value)
    write(directory / "score.json", value, exclusive=True)
    report(directory, manifest, value)
    print(json.dumps({"status": "scored", "overall": value["overall"], "by_task": value["by_task"]},
                     ensure_ascii=False), flush=True)
    return value


def report(directory, manifest, scored):
    directory = Path(directory)
    n = scored["overall"]["questions"]
    totals = scored["overall"]
    example = next((row for row in scored["rows"] if row["sample_id"] == "MMSI_845"),
                   scored["rows"][0])
    lines = [
        "# VGGT 拍摄位置关系实验",
        f"沿用上一次六类各 5 题，共 {n} 题。新方案只发送两处拍摄位置的双向文字关系，"
        "不发送位姿数值、旋转矩阵、前向向量或两台相机朝向的关系。原图、题目、选项、"
        "API 槽位、请求参数和答案解析方式保留。",
        f"新增 {scored['new_model_requests']} 次模型请求，复用 30 组冻结的 VGGT 预测，"
        "新增 GPU 前向 0 次，无自动重试。先冻结全部新答案，评分阶段才读取旧结果中的 GT。",
        "**对照方式：**Direct 和数值位姿答案是上次同一批题的已保存结果；"
        "文字关系方案为本次新调用。服务回报的采样参数可能与请求值不同，"
        "因此这次差值也可能包含模型输出波动。",
        "| 输入方案 | 答对 | 准确率 |",
        "|---|---:|---:|",
    ]
    for arm, label in (("direct", "原图（历史）"), ("pose", "原图＋数值位姿（历史）"),
                       ("relation", "原图＋拍摄位置关系（本次）")):
        lines.append(f"| {label} | {totals[f'{arm}_correct']}/{n} | {totals[f'{arm}_accuracy']:.1%} |")
    lines += [
        f"相对原图：进步 {totals['gains_vs_direct']} 题，退步 {totals['regressions_vs_direct']} 题；"
        f"相对数值位姿：进步 {totals['gains_vs_pose']} 题，退步 {totals['regressions_vs_pose']} 题。",
        "## 各题型",
        "| 题型 | 原图（历史） | 数值位姿（历史） | 位置关系（本次） |",
        "|---|---:|---:|---:|",
    ]
    for task, row in scored["by_task"].items():
        lines.append(f"| {task} | {row['direct_correct']}/5 | {row['pose_correct']}/5 | {row['relation_correct']}/5 |")
    lines += [
        "Cam–Cam 的 5 题中，258 问视角如何转动，348 问两台相机前向投影的夹角；"
        "本次给模型的拍摄位置关系没有提供这两类题所需的朝向关系。829、95、845 询问拍摄位置或位移。",
        "## 实际表述示例：" + example["sample_id"],
        "```text",
        example["relation_text"],
        "```",
        "这里两个方向各自使用对应照片的拍摄朝向；没有把相机的拍摄点当作图中物体或区域的位置。",
        "## 每题结果",
        "| 题目 | GT | 原图 | 数值位姿 | 拍摄位置关系 |",
        "|---|---|---|---|---|",
    ]
    for row in scored["rows"]:
        cells = [str(row[f"{arm}_answer"]) + (" ✓" if row[f"{arm}_correct"] else " ✗")
                 for arm in ("direct", "pose", "relation")]
        lines.append(f"| {row['sample_id']} | {row['reference_answer']} | " + " | ".join(cells) + " |")
    reported = Counter((row["reported"].get("model"), row["reported"].get("temperature"),
                        row["reported"].get("seed"), row["reported"].get("reasoning_effort"))
                       for row in scored["rows"])
    lines += ["## 模型与运行身份",
              "请求：gpt-5.6-terra，reasoning_effort=high，temperature=0，seed=137，"
              "max_output_tokens=8192；每个槽位至多 1 个在途请求，总并发至多 6。",
              "服务实际回报（模型、temperature、seed、reasoning_effort → 次数）：",
              "```json", json.dumps([{"reported": list(k), "count": v} for k, v in reported.items()],
                                    ensure_ascii=False, indent=2), "```",
              "本地环境：" + json.dumps(manifest["environment"], ensure_ascii=False),
              "历史环境：" + json.dumps(manifest["parent_environment"], ensure_ascii=False),
              "原图载荷与历史 Direct 的哈希核对一致；Pillow 版本差异没有改变发送的图片字节。",
              f"run_identity: `{manifest['run_identity']}`",
              f"freeze_identity: `{scored['freeze_identity']}`",
              f"score_identity: `{scored['score_identity']}`",
              "完整问题、关系文字、逐题答案、实际服务参数和 token 用量见 `score.json`；"
              "逐请求身份与回包记录见 `traces/`。"]
    output, in_code = [], False
    for line in lines:
        if line.startswith("```"):
            if not in_code:
                output.append("")
            output.append(line)
            in_code = not in_code
            if not in_code:
                output.append("")
        elif in_code:
            output.append(line)
        elif line.startswith("|"):
            if not output or not output[-1].startswith("|"):
                output.append("")
            output.append(line)
        else:
            output.extend(["", line, ""])
    (directory / "实验结果.md").write_text("\n".join(output).strip() + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "validate", "controlled-execute", "execute", "score"))
    parser.add_argument("--directory", default=str(DESTINATION))
    parser.add_argument("--parent", default=str(PARENT))
    args = parser.parse_args()
    try:
        if args.phase == "prepare":
            prepare(args.directory, args.parent)
        elif args.phase == "validate":
            value = validate(args.directory, verify_requests=True)
            print(json.dumps({"status": "validated", "run_identity": value["run_identity"]}), flush=True)
        elif args.phase == "execute":
            value = execute(args.directory, json.loads(sys.stdin.buffer.read().decode("utf-8")))
            freeze(args.directory)
            print(json.dumps(value), flush=True)
            if value["status"] != "complete":
                raise SystemExit(3)
        elif args.phase == "controlled-execute":
            controlled_execute(args.directory)
        else:
            score(args.directory)
    except Exception:
        print('{"status":"failed","reason_code":"location_relation_trial_failed"}', flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
