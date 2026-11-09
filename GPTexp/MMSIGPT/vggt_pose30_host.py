"""Credential-isolated dispatch and independent host scoring after 60 frozen predictions."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

from ..traces import digest, read, write
from . import vggt_pose_trial as base, vggt_pose30_trial as trial
from .prototype.direct import operator_lock
from .prototype.evaluation import reference_answers
from .prototype.protocol import LETTERS
from .vggt_pose30_control import DESTINATION, SSH, REMOTE_PYTHON, REFERENCE, REFERENCE_SHA

TASK_NAMES = {
    "Positional Relationship (Cam.–Cam.)": "相机—相机",
    "Positional Relationship (Cam.–Obj.)": "相机—物体",
    "Positional Relationship (Cam.–Reg.)": "相机—区域",
    "Positional Relationship (Obj.–Obj.)": "物体—物体",
    "Positional Relationship (Obj.–Reg.)": "物体—区域",
    "Positional Relationship (Reg.–Reg.)": "区域—区域",
}


def execute(directory):
    from .depth_service_probe import all_keys
    manifest = trial.validate(directory, verify_requests=True)
    available = all_keys()
    keys = {api: available[api] for api in manifest["api_ids"]}
    result = subprocess.run(
        [sys.executable, "-B", "-m", "GPTexp.MMSIGPT.vggt_pose30_trial",
         "execute", "--directory", str(directory)],
        input=json.dumps(keys).encode(), env=base.clean_environment(), cwd=base.ROOT)
    if result.returncode:
        raise ValueError("online_worker_failed")
    trial.freeze(directory)
    status = read(Path(directory) / "execution.json")
    if status["status"] != "complete":
        raise ValueError("online_service_failure_no_retry")
    return status


def score(directory):
    directory = Path(directory)
    manifest = trial.validate(directory)
    if not (directory / "execution.json").exists() or not (directory / "frozen.json").exists():
        raise ValueError("incomplete_no_reference_access")
    trial.validate_frozen(directory, manifest)
    if read(directory / "execution.json")["status"] != "complete":
        raise ValueError("incomplete_no_reference_access")
    code = ("from pathlib import Path; import sys; sys.stdout.buffer.write(Path("
            + repr(REFERENCE) + ").read_bytes())")
    result = subprocess.run(SSH + [shlex.quote(REMOTE_PYTHON) + " -B -c " + shlex.quote(code)],
                            capture_output=True, timeout=60)
    if result.returncode or hashlib.sha256(result.stdout).hexdigest() != REFERENCE_SHA:
        raise ValueError("host_reference_transfer_or_hash_failed")
    with tempfile.TemporaryDirectory(prefix="mmsi-pose30-host-score-") as temporary:
        path = Path(temporary) / "references.json"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(result.stdout)
        completed = subprocess.run(
            [sys.executable, "-B", "-m", "GPTexp.MMSIGPT.vggt_pose30_host",
             "score-internal", "--directory", str(directory), "--references", str(path),
             "--reference-sha256", REFERENCE_SHA],
            env=base.clean_environment(), cwd=base.ROOT)
        if completed.returncode:
            raise ValueError("independent_host_scoring_failed")
    return read(directory / "score.json")


def summarize(pairs):
    result = base.summarize(pairs)
    total = len(pairs)
    for arm in ("direct", "pose"):
        legal = sum(p[arm]["status"] == "answered" for p in pairs)
        result[arm + "_legal_answers"] = legal
        result[arm + "_legal_answer_rate"] = legal / total if total else None
    return result


def score_internal(directory, references, expected_sha):
    directory = Path(directory)
    with operator_lock(directory):
        manifest = trial.validate(directory)
        if not (directory / "execution.json").exists() or not (directory / "frozen.json").exists():
            raise ValueError("incomplete_no_reference_access")
        frozen = trial.validate_frozen(directory, manifest)
        if read(directory / "execution.json")["status"] != "complete":
            raise ValueError("incomplete_no_reference_access")
        raw = Path(references).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError("reference_identity_changed")
        answers = reference_answers(raw, frozen["predictions"])
        predictions = {(p["sample_id"], p["arm"]): p for p in frozen["predictions"]}
        pairs = []
        for item in manifest["items"]:
            sample = item["problem"]["sample_id"]
            answer = answers[sample]
            if answer not in LETTERS[:len(item["problem"]["options"])]:
                raise ValueError("invalid_host_reference_option")
            pair = {
                "sample_id": sample, "task_type": item["problem"]["task_type"],
                "source_split": item["problem"]["source_split"],
                "input_sha256": item["problem"]["input_sha256"],
                "reference_answer": answer, "api_id": item["api_id"],
            }
            for arm in ("direct", "pose"):
                p = predictions[sample, arm]
                pair[arm] = {**p, "correct": p["status"] == "answered" and p["answer"] == answer}
            pairs.append(pair)
        result = {
            "schema": "mmsi-pose30-host-score/1", "run_identity": manifest["run_identity"],
            "freeze_identity": frozen["freeze_identity"], "reference_file_sha256": expected_sha,
            "reference_access": "separate_host_process_after_all_60_predictions_frozen",
            "sample_role": manifest["sample_role"], "metric": "extracted_letter_exact_match",
            "denominator": "all_30_selected_questions_including_invalid_answers",
            "judge_fallback": False, "overall": summarize(pairs), "pairs": pairs,
            "by_task": {task: summarize([p for p in pairs if p["task_type"] == task])
                        for task in sorted(TASK_NAMES)}, "scored_at": time.time(),
        }
        result["score_identity"] = digest(result)
        write(directory / "score.json", result, exclusive=True)
    return result


def report(directory):
    directory = Path(directory)
    manifest = trial.validate(directory)
    frozen = trial.validate_frozen(directory, manifest)
    scoring = read(directory / "score.json")
    if (scoring["run_identity"] != manifest["run_identity"]
            or scoring["freeze_identity"] != frozen["freeze_identity"]
            or digest({k: v for k, v in scoring.items() if k != "score_identity"})
            != scoring["score_identity"]):
        raise ValueError("scoring_identity_changed")
    actual, rows = Counter(), []
    usage, summaries, tool_calls = Counter(), 0, 0
    for item in manifest["items"]:
        row = {
            "sample_id": item["problem"]["sample_id"], "task_type": item["problem"]["task_type"],
            "question": item["problem"]["question"], "options": item["problem"]["options"],
            "original_image_paths": item["problem"]["image_paths"], "pose_text": item["pose_text"],
            "task_binding": item["task_binding"], "responses": {},
        }
        for arm in ("direct", "pose"):
            _, path = trial.paths(directory, item, arm)
            result = read(path)
            response = result.get("response", {})
            actual[digest(response.get("reported"))] += 1
            summaries += bool(response.get("reasoning_summary"))
            tool_calls += response.get("tool_call_count", 0)
            for key, value in response.get("usage", {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[key] += value
            row["responses"][arm] = {
                "answer": result["answer"], "status": result["status"],
                "reported": response.get("reported"), "usage": response.get("usage"),
                "text": response.get("text"), "reasoning_summary": response.get("reasoning_summary"),
                "elapsed_seconds": response.get("elapsed_seconds"),
            }
        rows.append(row)
    metadata = []
    for row in rows:
        for response in row["responses"].values():
            value = response["reported"]
            if value not in [r["reported"] for r in metadata]:
                metadata.append({"reported": value, "count": actual[digest(value)]})
    audit = {
        "schema": "mmsi-pose30-report/1", "run_identity": manifest["run_identity"],
        "freeze_identity": frozen["freeze_identity"], "score_identity": scoring["score_identity"],
        "overall": scoring["overall"], "by_task": scoring["by_task"],
        "execution": read(directory / "execution.json"), "reported_metadata": metadata,
        "numeric_usage": dict(usage), "responses_with_reasoning_summary": summaries,
        "model_tool_calls": tool_calls, "gpu_joint_forwards": 30, "depth_images_sent": False,
        "rows": rows,
    }
    write(directory / "结果与轨迹.json", audit)
    overall = scoring["overall"]
    lines = [
        "# MMSI 六类各五题：原图 vs 精简 VGGT 位姿 v2", "",
        f"原图 {overall['direct_correct']}/30（{overall['direct_accuracy']:.1%}），"
        f"位姿组 {overall['pose_correct']}/30（{overall['pose_accuracy']:.1%}）。"
        f"净变化 {overall['pose_correct']-overall['direct_correct']:+d} 题，"
        f"{overall['delta_percentage_points']:+.2f} 个百分点。", "",
        "| 题型 | 原图 | 位姿 v2 | 进步 | 退步 |",
        "|---|---:|---:|---:|---:|",
    ]
    for task, value in scoring["by_task"].items():
        lines.append(f"| {TASK_NAMES[task]} | {value['direct_correct']}/5 | "
                     f"{value['pose_correct']}/5 | {value['gains']} | {value['regressions']} |")
    lines += [
        "", f"共进步 {overall['gains']} 题、退步 {overall['regressions']} 题，"
        f"答案改变 {overall['answers_changed']} 题。合法答案："
        f"原图 {overall['direct_legal_answers']}/30，位姿 {overall['pose_legal_answers']}/30。", "",
        "抽样 seed=137，每类5题，排除此前GPT开发选择/已派发题目；原 source_split=test，"
        "本轮用途为分层开发诊断。两组均保留官方原图顺序、题目和选项说明；"
        "位姿组在题目之前加入精简 v2 文本，按题目绑定参照原点、朝向和目标。", "",
        "每题两组使用相同API槽，30题分配到6槽，每槽逐个请求、最多6并发；"
        "Direct先发与位姿先发各15题。60次预测全部冻结后才由独立宿主进程读取参考答案。"
        "没有自动重试、额外评分模型或模型工具调用。", "",
        "请求配置：gpt-5.6-terra，reasoning=high，temperature=0，seed=137，"
        "max_output_tokens=8192，每请求超时180秒。服务返回的配置：", "",
    ]
    for value in metadata:
        lines.append(f"- {json.dumps(value['reported'], ensure_ascii=False)}：{value['count']}次")
    lines += [
        "", "GPU位姿预计算使用4090物理GPU3，与此前3090版本相同的VGGT源码和权重哈希；"
        "官方crop预处理，联合双图，聚合器BF16、camera head FP32。"
        "未运行深度/点云/跟踪头，未向GPT发送深度图。", "",
        "每类只有5题，且服务返回的采样配置与请求可能有差异；"
        "本轮用于观察题型表现，单题变化仍可能包含模型输出波动。", "",
        "逐题预测与参考答案见 `score.json`；完整公开输入、位姿文本和响应见 `结果与轨迹.json`。", "",
        "进步题：" + "、".join(p["sample_id"] for p in scoring["pairs"]
                               if not p["direct"]["correct"] and p["pose"]["correct"]) + "。", "",
        "退步题：" + "、".join(p["sample_id"] for p in scoring["pairs"]
                               if p["direct"]["correct"] and not p["pose"]["correct"]) + "。", "",
        f"run_identity：`{manifest['run_identity']}`", "",
        f"freeze_identity：`{frozen['freeze_identity']}`", "",
    ]
    (directory / "实验结果.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"overall": scoring["overall"], "by_task": scoring["by_task"],
                      "reported_metadata": metadata, "tool_calls": tool_calls}), flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("execute", "score", "score-internal", "report"))
    parser.add_argument("--directory", default=str(DESTINATION))
    parser.add_argument("--references")
    parser.add_argument("--reference-sha256")
    args = parser.parse_args()
    try:
        if args.phase == "score-internal":
            value = score_internal(args.directory, args.references, args.reference_sha256)
            print(json.dumps({"status": "scored", "overall": value["overall"]}), flush=True)
        else:
            globals()[args.phase](args.directory)
    except Exception:
        print('{"status":"failed","reason_code":"pose30_host_operation_failed"}', flush=True)
        raise SystemExit(2)
