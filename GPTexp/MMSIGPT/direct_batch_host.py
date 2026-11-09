"""Public preparation, credential-isolated launch and post-freeze paired scoring."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ..traces import digest, file_digest, read, write
from . import direct_batch as batch
from .depth_comparison.pool import qualify
from .depth_service_probe import all_keys
from .prototype.direct import operator_lock
from .prototype.protocol import LETTERS

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "MMSI六类四十题拍摄位置关系-20261109-r01"
DESTINATION = ROOT.parent / "MMSI六类四十题原图直答-20261109-r01"
TASK_NAMES = {
    "Positional Relationship (Cam.–Cam.)": "相机—相机",
    "Positional Relationship (Cam.–Obj.)": "相机—物体",
    "Positional Relationship (Cam.–Reg.)": "相机—区域",
    "Positional Relationship (Obj.–Obj.)": "物体—物体",
    "Positional Relationship (Obj.–Reg.)": "物体—区域",
    "Positional Relationship (Reg.–Reg.)": "区域—区域",
}


def clean_environment(snapshot=None):
    env = {k: v for k, v in os.environ.items()
           if k in {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL"}}
    root = Path(snapshot) if snapshot else ROOT
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONPATH=str(root / "GPTexp/vendor") + os.pathsep + str(root))
    return env


def prepare(directory):
    available = all_keys()
    keys = {api: available[api] for api in batch.APIS}
    if len(set(keys.values())) != 10:
        raise ValueError("requires_ten_distinct_credentials")
    qualification = qualify(keys)
    manifest = batch.prepare(directory, SOURCE / "manifest.json", qualification)
    snapshot = Path(directory) / "source_frozen"
    for name, expected in manifest["source_identity"].items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if file_digest(target) != expected:
            raise ValueError("source_snapshot_changed")
    write(Path(directory) / "host-binding.json", {
        "run_identity": manifest["run_identity"],
        "comparison_score_path": str(SOURCE / "score.json"),
        "comparison_score_sha256": file_digest(SOURCE / "score.json"),
        "source_manifest_path": str(SOURCE / "manifest.json"),
        "host_source_sha256": file_digest(__file__),
        "reference_contents_read": False, "model_requests": 0,
    }, exclusive=True)
    (Path(directory) / "实验结果.md").write_text(
        "# MMSI 六类各四十题：原图直答与拍摄位置关系\n\n"
        "原图直答待运行。题单固定为六类各40题，共240题；10个API并发，每槽24题。\n\n"
        "请求模型gpt-5.6-terra，high，temperature=0，seed=137，"
        "max_output_tokens=8192，timeout=180秒。每题一次原图直答，"
        "服务失败保留；未派发题继续执行。评分在240个请求终态冻结后进行。\n",
        encoding="utf-8")
    return manifest


def run(directory):
    directory = Path(directory).resolve()
    manifest = batch.validate(directory, verify_requests=True)
    binding = read(directory / "host-binding.json")
    if (binding["run_identity"] != manifest["run_identity"]
            or binding["host_source_sha256"] != file_digest(__file__)):
        raise ValueError("host_binding_changed")
    available = all_keys()
    keys = {api: available[api] for api in batch.APIS}
    for key in available.values():
        __import__("GPTexp.traces", fromlist=["credential_free"]).credential_free(manifest, key)
    snapshot = directory / "source_frozen"
    child = subprocess.run(
        [sys.executable, "-B", "-m", "GPTexp.MMSIGPT.direct_batch",
         "execute", "--directory", str(directory)],
        input=json.dumps(keys).encode("utf-8"), env=clean_environment(snapshot), cwd=snapshot)
    del keys, available
    if child.returncode:
        raise ValueError("worker_failed_no_automatic_restart")
    frozen_child = subprocess.run(
        [sys.executable, "-B", "-m", "GPTexp.MMSIGPT.direct_batch",
         "freeze", "--directory", str(directory)],
        env=clean_environment(snapshot), cwd=snapshot)
    if frozen_child.returncode:
        raise ValueError("prediction_freeze_failed")
    scorer = subprocess.run(
        [sys.executable, "-B", "-m", "GPTexp.MMSIGPT.direct_batch_host",
         "score", "--directory", str(directory)],
        env=clean_environment(), cwd=ROOT)
    if scorer.returncode:
        raise ValueError("independent_host_score_failed")
    return read(directory / "score.json")["overall"]


def summarize(rows):
    n = len(rows)
    common = [r for r in rows if r["direct_status"] == "answered"
              and r["relation_status"] == "answered"]
    direct = sum(r["direct_correct"] for r in rows)
    relation = sum(r["relation_correct"] for r in rows)
    return {
        "questions": n, "direct_correct": direct, "relation_correct": relation,
        "direct_accuracy": direct / n if n else None,
        "relation_accuracy": relation / n if n else None,
        "delta_percentage_points": 100 * (relation - direct) / n if n else None,
        "direct_status_counts": dict(Counter(r["direct_status"] for r in rows)),
        "relation_status_counts": dict(Counter(r["relation_status"] for r in rows)),
        "direct_legal_answers": sum(r["direct_status"] == "answered" for r in rows),
        "both_legal_questions": len(common),
        "direct_correct_both_legal": sum(r["direct_correct"] for r in common),
        "relation_correct_both_legal": sum(r["relation_correct"] for r in common),
        "gains": sum(not r["direct_correct"] and r["relation_correct"] for r in rows),
        "regressions": sum(r["direct_correct"] and not r["relation_correct"] for r in rows),
    }


def score(directory):
    directory = Path(directory).resolve()
    with operator_lock(directory):
        manifest, frozen = batch.validate_frozen(directory)
        binding = read(directory / "host-binding.json")
        if (binding["run_identity"] != manifest["run_identity"]
                or binding["host_source_sha256"] != file_digest(__file__)
                or file_digest(binding["source_manifest_path"]) != manifest["source_manifest_sha256"]
                or file_digest(binding["comparison_score_path"])
                != binding["comparison_score_sha256"]):
            raise ValueError("host_comparison_source_changed")
        # First semantic reference access is after all public prediction checks above.
        prior = read(binding["comparison_score_path"])
        if (digest({k: v for k, v in prior.items() if k != "score_identity"})
                != prior["score_identity"] or prior["run_identity"] != manifest["source_run_identity"]
                or prior["reference_answer_visible_during_solving"] is not False):
            raise ValueError("historical_host_score_identity_changed")
        old_rows = {r["sample_id"]: r for r in prior["rows"]}
        if set(old_rows) != {p["sample_id"] for p in frozen["predictions"]}:
            raise ValueError("paired_question_set_changed")
        items = {i["problem"]["sample_id"]: i for i in manifest["items"]}
        rows, reported, usage = [], [], Counter()
        for prediction in frozen["predictions"]:
            sample = prediction["sample_id"]
            old = old_rows[sample]
            item = items[sample]
            reference = old["reference_answer"]
            if (reference not in LETTERS[:len(item["problem"]["options"])]
                    or old["input_sha256"] != prediction["input_sha256"]
                    or old["task_type"] != prediction["task_type"]
                    or old["correct"] != (old["status"] == "answered"
                                           and old["answer"] == reference)):
                raise ValueError("paired_reference_or_input_changed")
            response = read(batch.trace_paths(directory, item)[1]).get("response", {})
            if response:
                reported.append(response["reported"])
            for key, value in response.get("usage", {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[key] += value
            rows.append({
                "sample_id": sample, "input_sha256": prediction["input_sha256"],
                "task_type": prediction["task_type"], "api_id": prediction["api_id"],
                "direct_status": prediction["status"], "direct_answer": prediction["answer"],
                "direct_correct": prediction["status"] == "answered"
                                  and prediction["answer"] == reference,
                "relation_status": old["status"], "relation_answer": old["answer"],
                "relation_correct": old["correct"],
            })
        metadata = []
        for value in reported:
            if value not in [m["reported"] for m in metadata]:
                metadata.append({"reported": value, "count": reported.count(value)})
        value = {
            "schema": "mmsi-direct240-paired-host-score/1",
            "run_identity": manifest["run_identity"], "freeze_identity": frozen["freeze_identity"],
            "comparison_run_identity": prior["run_identity"],
            "comparison_score_identity": prior["score_identity"],
            "reference_file_sha256": prior["reference_file_sha256"],
            "reference_access": "independent_host_process_after_240_direct_predictions_frozen",
            "reference_answer_visible_during_solving": False,
            "overall": summarize(rows), "rows": rows,
            "by_task": {task: summarize([r for r in rows if r["task_type"] == task])
                        for task in sorted(manifest["task_counts"])},
            "reported_metadata": metadata, "numeric_usage": dict(usage),
            "new_model_requests": 240, "automatic_retry": False,
            "peak_inflight": read(directory / "execution.json")["peak_inflight"],
            "scored_at": time.time(),
        }
        value["score_identity"] = digest(value)
        write(directory / "score.json", value, exclusive=True)
        report(directory, value)
        write(directory / "completed.json", {
            "status": "complete", "run_identity": manifest["run_identity"],
            "freeze_identity": frozen["freeze_identity"], "score_identity": value["score_identity"],
            "score_sha256": file_digest(directory / "score.json"),
            "expected_attempts": 240, "terminal_attempts": 240, "unique_attempts": 240,
            "model_requests": 240, "automatic_retry": False, "completed_at": time.time(),
        }, exclusive=True)
    print(json.dumps({"status": "complete", "overall": value["overall"],
                      "by_task": value["by_task"]}, ensure_ascii=False), flush=True)
    return value


def report(directory, score):
    overall = score["overall"]
    legal = overall["direct_legal_answers"]
    accuracy_legal = f"{overall['direct_correct'] / legal:.2%}" if legal else "—"
    lines = [
        "# MMSI 六类各四十题：原图直答与拍摄位置关系", "",
        f"原图直答 **{overall['direct_correct']}/240（{overall['direct_accuracy']:.2%}）**；"
        f"同题历史拍摄位置关系 **{overall['relation_correct']}/240"
        f"（{overall['relation_accuracy']:.2%}）**；"
        f"关系方法减直答为 **{overall['delta_percentage_points']:+.2f} 个百分点**。", "",
        "| 题型 | 原图直答 | 拍摄位置关系 | 关系减直答／百分点 | 直答合法题数 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for task, row in score["by_task"].items():
        lines.append(
            f"| {TASK_NAMES[task]} | {row['direct_correct']}/40（{row['direct_accuracy']:.2%}） | "
            f"{row['relation_correct']}/40（{row['relation_accuracy']:.2%}） | "
            f"{row['delta_percentage_points']:+.2f} | {row['direct_legal_answers']} |")
    lines += [
        "", f"直答终态：{json.dumps(overall['direct_status_counts'], ensure_ascii=False)}。"
        f"合法回答口径为{overall['direct_correct']}/{legal}（{accuracy_legal}）。"
        "全量口径将服务失败和非法回答计错。", "",
        f"两臂均合法的同题子集共{overall['both_legal_questions']}题："
        f"直答正确{overall['direct_correct_both_legal']}题，"
        f"拍摄位置关系正确{overall['relation_correct_both_legal']}题。", "",
        "题目和原图与原拍摄位置关系240题逐题绑定，保留图片字节、图片顺序、"
        "问题与选项顺序。直答使用官方字母回答指令。每个API槽24题，"
        f"10个独立凭据槽并发，实测峰值{score['peak_inflight']}。"
        "本轮每题只派发一次，错误保留；本轮新增240次模型请求。", "",
        "请求设置：gpt-5.6-terra，Responses，high，temperature=0，seed=137，"
        "max_output_tokens=8192，timeout=180秒。实际服务回报：", "",
        "```json", json.dumps(score["reported_metadata"], ensure_ascii=False, indent=2), "```", "",
        "两臂来自不同轮次；服务回报的采样参数可能不同于请求值，"
        "同题差分同时受到轮次与服务输出波动影响。", "",
        "在线子进程只读取公开题目和原图，凭据通过标准输入传递。"
        "240个终态预测和480份派发／结果文件冻结并核验后，独立宿主进程"
        "才读取既有宿主评分中的参考答案。最终汇总不导出逐题参考答案。", "",
        f"run_identity：`{score['run_identity']}`", "",
        f"freeze_identity：`{score['freeze_identity']}`", "",
        f"score_identity：`{score['score_identity']}`", "",
    ]
    (Path(directory) / "实验结果.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "score"))
    parser.add_argument("--directory", default=str(DESTINATION))
    args = parser.parse_args()
    try:
        if args.phase == "prepare":
            value = prepare(args.directory)
            print(json.dumps({"status": "ready", "questions": 240, "model_requests": 0,
                              "api_ids": value["api_ids"], "run_identity": value["run_identity"]}))
        elif args.phase == "run":
            run(args.directory)
        else:
            score(args.directory)
    except Exception:
        print('{"status":"failed","reason_code":"direct_batch_host_operation_failed"}', flush=True)
        raise SystemExit(2)
