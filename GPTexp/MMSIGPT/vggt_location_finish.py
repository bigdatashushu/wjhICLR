"""Finish never-dispatched location cases; retain transport-unknown attempts."""
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
from . import vggt_location_trial as trial
from .prototype.direct import operator_lock

TERMINAL = {"answered", "invalid_answer"}
DESTINATION = trial.DESTINATION / "continuation"


def code_identity():
    return {**trial.code_identity(), str(Path(__file__).relative_to(trial.base.ROOT)): file_digest(__file__)}


def origin(directory):
    manifest = trial.validate(directory, verify_requests=True)
    frozen = trial.validate_frozen(directory, manifest)
    execution = read(Path(directory) / "execution.json")
    if execution["status"] != "stopped_service_error":
        raise ValueError("requires_observed_operational_stop")
    pending_ids = {row["sample_id"] for row in frozen["predictions"] if row["status"] == "not_dispatched"}
    items = [item for item in manifest["items"] if item["problem"]["sample_id"] in pending_ids]
    if not items or execution["dispatched"] + len(items) != manifest["max_requests"]:
        raise ValueError("no_unattempted_cases_within_original_budget")
    return manifest, frozen, execution, items


def prepare(directory=DESTINATION, parent=trial.DESTINATION):
    directory, parent = Path(directory), Path(parent)
    if directory.exists() or not directory.parent.is_dir():
        raise ValueError("requires_new_continuation_directory")
    old, frozen, execution, items = origin(parent)
    directory.mkdir()
    (directory / "poses").mkdir()
    for item in items:
        shutil.copyfile(parent / item["pose_path"], directory / item["pose_path"])
    apis = list(dict.fromkeys(item["api_id"] for item in items))
    manifest = {
        "schema": "mmsi-location-unattempted-continuation/1",
        "parent_directory": str(parent), "parent_run_identity": old["run_identity"],
        "parent_freeze_identity": frozen["freeze_identity"],
        "parent_files": {name: file_digest(parent / name) for name in (
            "manifest.json", "frozen.json", "execution.json")},
        "items": items, "questions": len(items), "max_requests": len(items),
        "previous_request_attempts": execution["dispatched"],
        "cumulative_request_limit": old["max_requests"],
        "api_ids": apis, "max_inflight": len(apis),
        "inference_by_api": {api: trial.base.settings(api) for api in apis},
        "code_identity": code_identity(), "environment": trial.base.environment(),
        "reference_answer_visible": False, "automatic_retry": False,
        "failed_cases_repeated": 0, "prepared_at": time.time(),
    }
    manifest["run_identity"] = digest(manifest)
    write(directory / "manifest.json", manifest, exclusive=True)
    write(directory / "preflight.json", {
        "run_identity": manifest["run_identity"], "model_requests": 0,
        "reference_answers_read": False, "failed_cases_repeated": 0,
    }, exclusive=True)
    validate(directory, verify_requests=True)
    print(json.dumps({"status": "prepared", "never_dispatched_cases": len(items),
                      "failed_cases_repeated": 0,
                      "cumulative_request_limit": old["max_requests"]}), flush=True)
    return manifest


def validate(directory, *, verify_requests=False):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    parent = Path(manifest["parent_directory"])
    old, frozen, execution, items = origin(parent)
    if (manifest["schema"] != "mmsi-location-unattempted-continuation/1"
            or digest({k: v for k, v in manifest.items() if k != "run_identity"}) != manifest["run_identity"]
            or manifest["code_identity"] != code_identity()
            or manifest["environment"] != trial.base.environment()
            or manifest["items"] != items or manifest["max_requests"] != len(items)
            or manifest["questions"] != len(items)
            or manifest["previous_request_attempts"] != execution["dispatched"]
            or manifest["cumulative_request_limit"] != old["max_requests"]
            or manifest["parent_run_identity"] != old["run_identity"]
            or manifest["parent_freeze_identity"] != frozen["freeze_identity"]
            or manifest["max_inflight"] != len(manifest["api_ids"])
            or manifest["api_ids"] != list(dict.fromkeys(item["api_id"] for item in items))
            or manifest["inference_by_api"] != {api: trial.base.settings(api) for api in manifest["api_ids"]}
            or manifest["failed_cases_repeated"] != 0
            or manifest["reference_answer_visible"] is not False
            or manifest["automatic_retry"] is not False
            or any(file_digest(parent / name) != expected
                   for name, expected in manifest["parent_files"].items())):
        raise ValueError("continuation_or_original_budget_changed")
    for item in items:
        if file_digest(directory / item["pose_path"]) != item["pose_sha256"]:
            raise ValueError("continuation_pose_changed")
        if verify_requests and digest(trial.request(directory, item, trial.ARM)) != item["payload_sha256"][trial.ARM]:
            raise ValueError("never_dispatched_payload_changed")
    preflight = read(directory / "preflight.json")
    if (preflight["run_identity"] != manifest["run_identity"] or preflight["model_requests"] != 0
            or preflight["reference_answers_read"] is not False):
        raise ValueError("continuation_preflight_changed")
    return manifest


execute = FunctionType(trial.shared.execute.__code__, {
    **vars(trial.shared), "validate": validate, "request": trial.request, "inference": trial.base.settings,
}, "execute")
execute.__kwdefaults__ = {"client_factory": trial.base.BoundedClient}


def freeze(directory):
    directory = Path(directory)
    with operator_lock(directory):
        manifest = validate(directory)
        predictions, files = trial.collect(directory, manifest)
        value = {
            "run_identity": manifest["run_identity"], "predictions": predictions, "files": files,
            "reference_answer_visible": False, "automatic_retry": False, "frozen_at": time.time(),
        }
        value["freeze_identity"] = digest(value)
        write(directory / "frozen.json", value, exclusive=True)
        return value


def validate_frozen(directory, manifest):
    frozen = read(Path(directory) / "frozen.json")
    predictions, files = trial.collect(directory, manifest)
    if (digest({k: v for k, v in frozen.items() if k != "freeze_identity"}) != frozen["freeze_identity"]
            or frozen["run_identity"] != manifest["run_identity"]
            or frozen["predictions"] != predictions or frozen["files"] != files):
        raise ValueError("continuation_predictions_changed")
    return frozen


def controlled_execute(directory):
    from .depth_service_probe import all_keys
    manifest = validate(directory, verify_requests=True)
    available = all_keys()
    keys = {api: available[api] for api in manifest["api_ids"]}
    value = subprocess.run(
        [sys.executable, "-B", "-m", "GPTexp.MMSIGPT.vggt_location_finish", "execute",
         "--directory", str(directory)], input=json.dumps(keys).encode("utf-8"),
        env=trial.base.clean_environment(), cwd=trial.base.ROOT)
    if value.returncode:
        raise ValueError("continuation_worker_failed")


def summarize(rows):
    valid = [row for row in rows if row["relation_status"] in TERMINAL]
    value = {"selected_questions": len(rows), "questions": len(valid),
             "operational_failures": len(rows) - len(valid)}
    for arm in ("direct", "pose", "relation"):
        value[f"{arm}_correct"] = sum(row[f"{arm}_correct"] for row in valid)
        value[f"{arm}_accuracy"] = value[f"{arm}_correct"] / len(valid) if valid else None
    value["relation_legal_answers"] = sum(row["relation_answer"] is not None for row in valid)
    for arm in ("direct", "pose"):
        value[f"gains_vs_{arm}"] = sum(row["relation_correct"] and not row[f"{arm}_correct"] for row in valid)
        value[f"regressions_vs_{arm}"] = sum(row[f"{arm}_correct"] and not row["relation_correct"] for row in valid)
        value[f"answers_changed_vs_{arm}"] = sum(row["relation_answer"] != row[f"{arm}_answer"] for row in valid)
    return value


def score(directory):
    directory = Path(directory)
    manifest = validate(directory)
    continuation = validate_frozen(directory, manifest)
    parent = Path(manifest["parent_directory"])
    original, frozen, execution, _ = origin(parent)
    latest = read(directory / "execution.json")
    if latest["inflight"] != 0 or latest["status"] not in ("complete", "stopped_service_error"):
        raise ValueError("continuation_not_terminal_no_reference_access")
    if latest["dispatched"] + execution["dispatched"] > manifest["cumulative_request_limit"]:
        raise ValueError("original_attempt_budget_exceeded")
    # Reference-bearing records are read only after both runs have ended and frozen.
    reference_root = Path(original["parent_directory"])
    old_score = read(reference_root / "score.json")
    if (digest({k: v for k, v in old_score.items() if k != "score_identity"}) != old_score["score_identity"]
            or old_score["run_identity"] != original["parent_run_identity"]
            or old_score["freeze_identity"] != original["parent_freeze_identity"]):
        raise ValueError("historical_reference_score_changed")
    old_pairs = {row["sample_id"]: row for row in old_score["pairs"]}
    old_frozen = read(reference_root / "frozen.json")
    old_answers = {(row["sample_id"], row["arm"]): row["answer"] for row in old_frozen["predictions"]}
    combined = {row["sample_id"]: row for row in frozen["predictions"]}
    child_ids = set()
    for row in continuation["predictions"]:
        if combined[row["sample_id"]]["status"] != "not_dispatched":
            raise ValueError("attempted_case_repeated")
        combined[row["sample_id"]] = row
        child_ids.add(row["sample_id"])
    rows = []
    for item in original["items"]:
        sample = item["problem"]["sample_id"]
        old, prediction = old_pairs[sample], combined[sample]
        if (old["input_sha256"] != item["problem"]["input_sha256"]
                or any(old[arm]["answer"] != old_answers[(sample, arm)] for arm in ("direct", "pose"))):
            raise ValueError("historical_pair_changed")
        record_root = directory if sample in child_ids else parent
        _, result_path = trial.shared.paths(record_root, item, trial.ARM)
        record = read(result_path) if result_path.exists() else {}
        reference = old["reference_answer"]
        rows.append({
            "sample_id": sample, "index": item["index"], "task_type": item["problem"]["task_type"],
            "question": item["problem"]["question"], "options": item["problem"]["options"],
            "reference_answer": reference, "input_sha256": item["problem"]["input_sha256"],
            "direct_answer": old["direct"]["answer"], "pose_answer": old["pose"]["answer"],
            "relation_answer": prediction["answer"], "relation_status": prediction["status"],
            "direct_correct": old["direct"]["answer"] == reference,
            "pose_correct": old["pose"]["answer"] == reference,
            "relation_correct": (prediction["answer"] == reference if prediction["status"] in TERMINAL else None),
            "relation_text": item["relation_text"], "relation_rows": item["relation_rows"],
            "reported": record.get("response", {}).get("reported"),
            "usage": record.get("response", {}).get("usage"),
            "error_code": record.get("error_code"), "http_status": record.get("http_status"),
        })
    value = {
        "schema": "mmsi-location-language-comparison-with-operational-exclusions/1",
        "run_identity": original["run_identity"], "continuation_run_identity": manifest["run_identity"],
        "freeze_identity": frozen["freeze_identity"],
        "continuation_freeze_identity": continuation["freeze_identity"],
        "source_score_identity": old_score["score_identity"],
        "new_model_request_attempts": latest["dispatched"] + execution["dispatched"],
        "failed_cases_repeated": 0, "gpu_forwards": 0,
        "comparison": original["comparison"],
        "denominator": "same_subset_with_new_answer_or_invalid_answer_excluding_operational_failures",
        "reference_answer_visible_during_solving": False,
        "overall": summarize(rows),
        "by_task": {task: summarize([row for row in rows if row["task_type"] == task])
                    for task in dict.fromkeys(row["task_type"] for row in rows)},
        "camera_location_questions": summarize([row for row in rows if row["sample_id"] in (
            "MMSI_829", "MMSI_95", "MMSI_845")]),
        "camera_orientation_questions": summarize([row for row in rows if row["sample_id"] in (
            "MMSI_258", "MMSI_348")]),
        "rows": rows, "scored_at": time.time(),
    }
    value["score_identity"] = digest(value)
    write(parent / "score.json", value, exclusive=True)
    make_report(parent, original, value)
    print(json.dumps({"status": "scored", "overall": value["overall"], "by_task": value["by_task"],
                      "camera_location_questions": value["camera_location_questions"]},
                     ensure_ascii=False), flush=True)
    return value


def make_report(parent, manifest, value):
    overall = value["overall"]
    n = overall["questions"]
    lines = [
        "# VGGT 拍摄位置关系实验", "",
        f"沿用上一轮六类各 5 题，共 30 题。本轮 {n} 题有模型回答，"
        f"{overall['operational_failures']} 题发生接口故障。下表三个方案都在同一组 {n} 题上比较，"
        "接口故障不计为模型答错。Direct 和数值位姿使用上一轮保存的答案；拍摄位置关系是本轮新调用。", "",
        "| 输入方案 | 答对 | 准确率 |", "|---|---:|---:|",
    ]
    for arm, label in (("direct", "原图（历史）"), ("pose", "原图＋数值位姿（历史）"),
                       ("relation", "原图＋拍摄位置关系（本次）")):
        lines.append(f"| {label} | {overall[f'{arm}_correct']}/{n} | {overall[f'{arm}_accuracy']:.1%} |")
    lines += ["",
              f"相对原图：进步 {overall['gains_vs_direct']} 题，退步 {overall['regressions_vs_direct']} 题；"
              f"相对数值位姿：进步 {overall['gains_vs_pose']} 题，退步 {overall['regressions_vs_pose']} 题。", "",
              "## 按题型比较", "",
              "| 题型 | 可比较 / 计划 | 原图 | 数值位姿 | 拍摄位置关系 |",
              "|---|---:|---:|---:|---:|"]
    for task, result in value["by_task"].items():
        k = result["questions"]
        lines.append(f"| {task} | {k}/{result['selected_questions']} | "
                     f"{result['direct_correct']}/{k} | {result['pose_correct']}/{k} | "
                     f"{result['relation_correct']}/{k} |")
    location = value["camera_location_questions"]
    orientation = value["camera_orientation_questions"]
    lines += ["",
              "Cam–Cam 中，829、95、845 问拍摄位置或位移；258、348 问视角转动或相机朝向夹角。"
              "本轮提供拍摄位置关系，朝向题依靠原图判断。", "",
              f"3 道位置题：原图 {location['direct_correct']}/3，数值位姿 {location['pose_correct']}/3，"
              f"拍摄位置关系 {location['relation_correct']}/3。"
              f"2 道朝向题：原图 {orientation['direct_correct']}/2，数值位姿 {orientation['pose_correct']}/2，"
              f"拍摄位置关系 {orientation['relation_correct']}/2。", "",
              "## 实际表述", "",
              "每条关系明确参考照片的拍摄点和拍摄朝向；所描述的目标也是另一张照片的拍摄点。"
              "左右与前后来自参考相机坐标中的位移分量，并说明哪一分量更大。"
              "不发送位姿数值、旋转矩阵、前向向量、夹角或额外图片。", ""]
    example = next((row for row in value["rows"] if row["sample_id"] == "MMSI_845"), value["rows"][0])
    lines += ["```text", example["relation_text"], "```", "",
              f"示例 {example['sample_id']}：GT={example['reference_answer']}，"
              f"本次答案={example['relation_answer']}。", "",
              "## 逐题结果", "", "| 题目 | GT | 原图 | 数值位姿 | 拍摄位置关系 |",
              "|---|---|---|---|---|"]
    for row in value["rows"]:
        cells = [str(row[f"{arm}_answer"]) + (" ✓" if row[f"{arm}_correct"] else " ✗")
                 for arm in ("direct", "pose")]
        cells.append(str(row["relation_answer"]) + (" ✓" if row["relation_correct"] else " ✗")
                     if row["relation_status"] in TERMINAL else "接口故障")
        lines.append(f"| {row['sample_id']} | {row['reference_answer']} | " + " | ".join(cells) + " |")
    failed = [row for row in value["rows"] if row["relation_status"] not in TERMINAL]
    lines += ["", "## 请求与比较范围", "",
              f"累计 {value['new_model_request_attempts']} 次请求尝试，失败题重复请求 0 次，"
              "新增 GPU 前向 0 次；复用已冻结的 30 组 VGGT 估计。"
              "首轮因传输失败停止，后续只派发从未派发的题。每次执行均无自动重试。"
              "两个执行阶段均结束并冻结后才读取 GT。", ""]
    for row in failed:
        lines.append(f"- {row['sample_id']}：{row['error_code']}，HTTP 状态 {row['http_status']}；"
                     "保留该次尝试，未取得可评分回答。")
    params = Counter((row["reported"].get("model"), row["reported"].get("temperature"),
                      row["reported"].get("seed"), row["reported"].get("reasoning_effort"))
                     for row in value["rows"] if row["reported"])
    lines += ["", "请求参数为 gpt-5.6-terra、reasoning_effort=high、temperature=0、seed=137、"
              "max_output_tokens=8192；实际服务回报如下。"
              "历史答案与本次新调用之间的差值可能包含采样波动。", "",
              "```json", json.dumps([{"reported": list(k), "count": count} for k, count in params.items()],
                                    ensure_ascii=False, indent=2), "```", "",
              "本地环境：" + json.dumps(manifest["environment"], ensure_ascii=False), "",
              "历史环境：" + json.dumps(manifest["parent_environment"], ensure_ascii=False), "",
              "原图载荷与历史 Direct 的哈希核对一致；Pillow 版本差异没有改变发送的图片字节。", "",
              f"run_identity: `{value['run_identity']}`", "",
              f"continuation_run_identity: `{value['continuation_run_identity']}`", "",
              f"score_identity: `{value['score_identity']}`", "",
              "逐题问题、实际附加文字、GT、答案和服务参数见 `score.json`。"
              "首轮与补发阶段的原始身份及结果分别保留在 `traces/` 和 `continuation/traces/`。"]
    (parent / "实验结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "controlled-execute", "execute", "score"))
    parser.add_argument("--directory", default=str(DESTINATION))
    parser.add_argument("--parent", default=str(trial.DESTINATION))
    args = parser.parse_args()
    try:
        if args.phase == "prepare":
            prepare(args.directory, args.parent)
        elif args.phase == "controlled-execute":
            controlled_execute(args.directory)
        elif args.phase == "execute":
            value = execute(args.directory, json.loads(sys.stdin.buffer.read().decode("utf-8")))
            freeze(args.directory)
            print(json.dumps(value), flush=True)
        else:
            score(args.directory)
    except Exception:
        print('{"status":"failed","reason_code":"location_unattempted_continuation_failed"}', flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
