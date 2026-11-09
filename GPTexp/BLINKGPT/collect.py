"""Host-only scores, formal archive collection and readback validation."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import tarfile
import time

from .common import COUNTS, ROOT, digest, file_digest, read, write
from .deploy import remote_python, ssh


def host_references(local):
    manifest = read(local / "manifest.json")
    refs = {}
    for dataset, subroot in (("mv", ROOT.parents[1] / "BLINK"),
                             ("sr", ROOT.parents[1] / "BLINK/spatial_relation")):
        pin = read(local / "public" / f"{dataset}-metadata.json")
        path = subroot / "data/references.jsonl"
        if file_digest(path) != pin["reference_index_sha256"]:
            raise ValueError("host_reference_pin_changed")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["split"] == "val":
                if row["gt_available"] is not True or row["gt_answer"] not in {"A", "B"}:
                    raise ValueError("invalid_validation_reference")
                refs[dataset, row["sample_id"]] = row["gt_answer"]
    if not manifest["reference_answers_visible"] is False:
        raise ValueError("public_manifest_privacy_changed")
    return refs


def metrics(manifest, rows, refs):
    by_id = {r["job_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate_episodes")
    result = {}
    for key, expected in COUNTS.items():
        jobs = [j for j in manifest["jobs"] if j["dataset"] + ":" + j["arm"] == key]
        received = [by_id[j["job_id"]] for j in jobs if j["job_id"] in by_id]
        correct, answered, failures = 0, 0, Counter()
        for row in received:
            job = next(j for j in jobs if j["job_id"] == row["job_id"])
            if row["job_sha256"] != digest(job) or row["run_identity"] != manifest["run_identity"]:
                raise ValueError("score_episode_binding_changed")
            prediction = (row["result"] or {}).get("answer")
            if prediction in {"A", "B"}:
                answered += 1
                correct += int(prediction == refs[row["dataset"], row["official_id"]])
            else:
                failures[(row["failure"] or {}).get("code")
                         or (row["result"] or {}).get("status", "unknown")] += 1
        result[key] = {"expected": expected, "completed": len(received),
                       "remaining": expected - len(received), "answered": answered, "correct": correct,
                       "accuracy_percent": 100 * correct / expected,
                       "completed_accuracy_percent": 100 * correct / len(received) if received else None,
                       "failures": dict(failures)}
    return result


def report(local, snapshot):
    m = read(local / "manifest.json")
    if snapshot["run_identity"] != m["run_identity"]:
        raise ValueError("snapshot_identity_changed")
    scores = metrics(m, snapshot["episodes"], host_references(local))
    final = bool(snapshot["final_collection"])
    at = datetime.fromtimestamp(snapshot["at"], ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    lines = ["# BLINK GPT-5.6-terra 实验", "",
             f"北京时间：{at}。状态：{'已完成远端收集，等待本地验收' if final else '运行中；以下为已完成题目的阶段统计'}。",
             f"已完成 {snapshot['completed']}/662，剩余 {snapshot['remaining']}。", "",
             "| 数据集 / 流程 | 完成 / 总数 | 正确 | 已完成准确率 |",
             "|---|---:|---:|---:|"]
    labels = {"mv:direct": "MV 原流程直答", "mv:harness": "MV 方法",
              "sr:direct": "SR 原流程直答", "sr:harness": "SR v2 方法",
              "mv:official": "MV 官方 prompt 直答", "sr:official": "SR 官方 prompt 直答"}
    for key, score in scores.items():
        rate = score["completed_accuracy_percent"]
        lines.append(f"| {labels[key]} | {score['completed']}/{score['expected']} | "
                     f"{score['correct']} | {'—' if rate is None else f'{rate:.2f}%'} |")
    lines.extend(["", "官方 prompt 直答使用官方 val：MV 133 题、SR 143 题；原始图片与选项顺序不变，不加自定义回答后缀。",
                  "这是官方公开验证集的本地准确率；EvalAI 隐藏 test 的线上成绩需要官方提交。",
                  "方法对照固定为 MV 93 题、SR 100 题。MV 旧内部 test 已按图像哈希映射到官方 val；SR v2 曾使用该内部测试集合开发提示词。",
                  "最终准确率使用完整固定分母；错误或无效回答计为未答对。运行中的已完成准确率不代表最终成绩。",
                  "请求模型为 gpt-5.6-terra，reasoning=high，seed=137，temperature=0，输出上限8192。",
                  "每次请求最长36000秒；每个逻辑请求最多12次传输尝试；退避60至3600秒并遵守Retry-After。",
                  "缺失的 provider model 元数据保持缺失；若返回另一个模型名，拒收该响应。"] )
    lines.extend(["", "API 槽状态："])
    for slot in snapshot["slots"]:
        inflight = slot.get("inflight") or {}
        last_error = slot.get("last_error") or {}
        api = inflight.get("api_id") or last_error.get("api_id") or "空闲槽"
        lines.append(f"- {api}：在途 {int(bool(inflight))}，成功 {slot.get('successes', 0)}，"
                     f"错误 {slot.get('errors', 0)}，最近错误 {last_error.get('http_status') or last_error.get('code') or '无'}。")
    (local / "实验结果.md").write_text("\n".join(lines) + "\n")
    write(local / "scores.json", {"at": snapshot["at"], "final": final, "metrics": scores,
                                 "run_identity": m["run_identity"]})
    return scores


def collect_final(local, remote, snapshot):
    receipt = snapshot["final_collection"]
    archive = local / receipt["archive"]
    if not archive.exists():
        raw = ssh("cat " + __import__("shlex").quote(remote + "/" + receipt["archive"]))
        archive.write_bytes(raw)
    if file_digest(archive) != receipt["archive_sha256"]:
        raise ValueError("collected_archive_sha256_changed")
    dest = local / "final"
    dest.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar:
            target = dest / member.name
            if member.name.startswith("/") or ".." in Path(member.name).parts or member.issym() or member.islnk():
                raise ValueError("unsafe_archive_member")
            if member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open("wb") as out:
                    import shutil
                    shutil.copyfileobj(source, out)
    completed = read(dest / "completed.json")
    m = read(local / "manifest.json")
    if (file_digest(dest / "manifest.json") != file_digest(local / "manifest.json")
            or file_digest(dest / "completed.json") != receipt["completed_sha256"]
            or completed["run_identity"] != m["run_identity"] or completed["jobs"] != 662):
        raise ValueError("final_collection_manifest_binding_changed")
    rows = []
    for name, sha in completed["episodes_sha256"].items():
        path = dest / "episodes" / name
        if file_digest(path) != sha:
            raise ValueError("collected_episode_changed")
        row = read(path)
        for req, req_sha in row["requests_sha256"].items():
            if file_digest(dest / req) != req_sha:
                raise ValueError("collected_request_changed")
        rows.append(row)
    scores = metrics(m, rows, host_references(local))
    if any(s["completed"] != s["expected"] for s in scores.values()):
        raise ValueError("final_score_denominator_changed")
    write(local / "host-final-collection.json", {
        "run_identity": m["run_identity"], "archive_sha256": file_digest(archive),
        "completed_sha256": file_digest(dest / "completed.json"),
        "remote_receipt": receipt, "jobs": 662, "metrics": scores,
        "host_scored": True, "all_members_hash_verified": True, "at": time.time()}, once=True)
    (local / "实验结果.md").write_text(
        (local / "实验结果.md").read_text().replace("已完成远端收集，等待本地验收", "已完成；本地评分与整档哈希验收通过"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local-root", type=Path, required=True)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=float, default=60)
    args = p.parse_args()
    while True:
        snapshot = json.loads(ssh(remote_python(args.remote_root, "snapshot")))
        write(args.local_root / "snapshot.json", snapshot)
        scores = report(args.local_root, snapshot)
        print(json.dumps({"completed": snapshot["completed"], "remaining": snapshot["remaining"],
                          "metrics": scores}, ensure_ascii=False), flush=True)
        if snapshot["final_collection"]:
            collect_final(args.local_root, args.remote_root, snapshot)
            break
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
