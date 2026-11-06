#!/usr/bin/env python3
"""Collect the isolated Direct arm and score common legal answers locally."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from collect_score_pilot40 import validate
from skill3d.evaluation.accuracy import mca_correct
from skill3d.evaluation.mra import mra_from_text

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/home/wangy/wangjh/VSIbench8Brunner"
NAME = "vsi-direct40-thinking-20261105-r01"
HARNESS = ROOT / "runs/vsi8b-pilot40-20261105-r02"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def query(control):
    code = """
import json
from pathlib import Path
b=Path(%r); r=b/'runs'/%r
launch=json.loads((b/'pilot40-direct01.pid.json').read_text())
p=Path('/proc')/str(launch['pid']); alive=p.exists()
if alive:
    fields=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
    if fields[19]!=launch['process_identity']['start_ticks']:
        raise ValueError('Direct process identity changed')
    alive=fields[0]!='Z'
state=json.loads((r/'live_status.json').read_text()) if (r/'live_status.json').exists() else {'status':'preflight','completed':0}
print(json.dumps({'alive':alive,'state':state}))
""" % (REMOTE, NAME)
    result = subprocess.run([
        "ssh", "-S", control, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "wangy@10.10.0.34", "python3 -c " + shlex.quote(code)],
        capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError("Direct collection status query failed")
    return json.loads(result.stdout)


def verify_harness_snapshot(control, snapshot):
    expected = {row["qa_id"]: row["outcome_sha256"] for row in snapshot["rows"]}
    code = """
import hashlib,json
from pathlib import Path
r=Path(%r)/'runs/vsi8b-pilot40-20261105-r02'
print(json.dumps({p.stem:hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'outcomes').glob('*.json')}))
""" % REMOTE
    result = subprocess.run([
        "ssh", "-S", control, "-o", "BatchMode=yes", "wangy@10.10.0.34",
        "python3 -c " + shlex.quote(code)], capture_output=True, timeout=30)
    if result.returncode or json.loads(result.stdout) != expected:
        raise ValueError("completed Harness outcomes changed after the score snapshot")


def aggregate(root, summary, selected):
    target = root / "results_and_full_trajectories.json"
    with target.with_suffix(".json.tmp").open("w") as output:
        output.write('{"summary":')
        json.dump(summary, output, ensure_ascii=False, allow_nan=False)
        for kind in ("outcomes", "trajectories"):
            output.write(',\n"' + kind + '":{')
            for index, qa in enumerate(selected):
                if index:
                    output.write(",\n")
                json.dump(qa, output)
                output.write(":")
                with (root / kind / f"{qa}.json").open() as source:
                    shutil.copyfileobj(source, output, length=1024**2)
            output.write("}")
        output.write("}\n")
    target.with_suffix(".json.tmp").replace(target)


def score(root, metadata_path, control):
    manifest = read(root / "runtime_identity.json")
    sampling = read(root / "sampling_manifest.json")
    selected = {str(row["qa_id"]): row for row in sampling["questions"]}
    if len(selected) != 40 or set(manifest["selected_qa_ids"]) != set(selected):
        raise ValueError("Direct scoring panel changed")
    harness = read(HARNESS / "complete40_score_snapshot.json")
    verify_harness_snapshot(control, harness)
    hrows = {row["qa_id"]: row for row in harness["rows"]}
    if set(hrows) != set(selected):
        raise ValueError("Direct and Harness panels differ")
    metadata = {str(row["qa_id"]): row for row in
                map(json.loads, metadata_path.read_text().splitlines())}
    rows = []
    for qa, question in selected.items():
        result = read(root / "outcomes" / f"{qa}.json")
        outcome, reference = result["outcome"], metadata[qa]
        if (result["reference_answer_visible"] is not False
                or result["status"] != "finished"
                or result["direct_manifest_sha256"] != manifest["identity_sha256"]
                or reference["question"] != question["question"]
                or (reference["options"] or []) != (question["options"] or [])
                or reference["source_split"] != result["source_split"]
                or hrows[qa]["question"] != question["question"]
                or (hrows[qa]["options"] or []) != (question["options"] or [])):
            raise ValueError("Direct/Harness public question identity mismatch")
        request = read(root / "solver_requests" / qa / "001.json")
        payload = request["request"]
        if (request["status"] != "returned"
                or len(list((root / "solver_requests" / qa).glob("*.json"))) != 1
                or payload["model"] != manifest["model"]
                or payload["enable_thinking"] is not True
                or payload["seed"] != 137 or payload["temperature"] != 0.
                or payload["max_tokens"] != 8192
                or request["raw_response"]["model"] != manifest["model"]):
            raise ValueError("Direct native sampling/model audit failed")
        answer = outcome.get("predicted") or outcome.get("answer")
        value = float(mca_correct(answer or "", reference["ground_truth"])) if outcome["is_mca"] else (
            mra_from_text(answer, reference["ground_truth"]))
        rows.append({
            "qa_id": qa, "task": hrows[qa]["question_type"],
            "direct_answer": answer, "direct_score": value,
            "direct_legal": outcome["legal_answer"],
            "harness_answer": hrows[qa]["answer"], "harness_score": hrows[qa]["score"],
            "harness_legal": hrows[qa]["legal_answer"],
            "paired_mutually_legal": outcome["legal_answer"] and hrows[qa]["legal_answer"],
        })
    pairs = [row for row in rows if row["paired_mutually_legal"]]
    if not pairs:
        raise ValueError("no mutually legal paired answers")
    by_task = {}
    for task in sorted({row["task"] for row in rows}):
        group = [row for row in pairs if row["task"] == task]
        by_task[task] = {
            "n": len(group),
            "direct_mean_score": sum(row["direct_score"] for row in group) / len(group) if group else None,
            "harness_mean_score": sum(row["harness_score"] for row in group) / len(group) if group else None,
        }
    summary = {
        "schema": "vsi40-direct-versus-8b-harness/1",
        "comparison_scope": "Qwen Direct versus current 8B-description Harness",
        "completed_each_arm": 40, "paired_mutually_legal": len(pairs),
        "direct_legal_answers": sum(row["direct_legal"] for row in rows),
        "harness_legal_answers": sum(row["harness_legal"] for row in rows),
        "direct_mean_score": sum(row["direct_score"] for row in pairs) / len(pairs),
        "harness_mean_score": sum(row["harness_score"] for row in pairs) / len(pairs),
        "direct_all40_mean_score": sum(row["direct_score"] for row in rows) / 40,
        "harness_all40_mean_score": sum(row["harness_score"] for row in rows) / 40,
        "model": manifest["model"], "enable_thinking": True,
        "seed": 137, "temperature": 0., "max_tokens": 8192,
        "direct_frame_protocol": "Verified Qwen official fetch_video output; 32 original frames",
        "scoring": "MCA exact match and official-equivalent 10-threshold numeric MRA",
        "reference_answer_visible_during_solve": False,
        "direct_manifest_sha256": manifest["identity_sha256"],
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "harness_outcome_hashes_reverified": True, "by_task": by_task, "rows": rows,
    }
    summary["delta_percentage_points"] = 100 * (
        summary["harness_mean_score"] - summary["direct_mean_score"])
    write(root / "direct_vs_8b_scores.json", summary)
    lines = [
        "# Qwen Direct 与 8B 描述 Harness：40 题结果",
        "",
        f"两边各完成 40 题；双方答案均合法的共同题目为 {len(pairs)} 题。",
        "",
        "| 方法 | 共同合法题目平均分 | 全部 40 题平均分 | 合法答案 |",
        "|---|---:|---:|---:|",
        f"| Qwen Direct | {summary['direct_mean_score']:.2%} | {summary['direct_all40_mean_score']:.2%} | {summary['direct_legal_answers']}/40 |",
        f"| 8B 描述 Harness | {summary['harness_mean_score']:.2%} | {summary['harness_all40_mean_score']:.2%} | {summary['harness_legal_answers']}/40 |",
        "",
        f"Harness 在共同合法题目上的差值为 {summary['delta_percentage_points']:+.2f} 个百分点。",
        "",
        "直答使用 Qwen 官方工具已经抽取并逐帧核验的 32 帧，一题一次请求，"
        "没有 Skill、几何、候选、工具调用或额外概览图。"
        "两边主模型均为 Qwen3.8-Flash，thinking 开启，seed=137，temperature=0，max_tokens=8192。"
        "同题比较包含各自的输入流程；直答官方抽帧与 Harness 的冻结视觉输入应按各自回执解释。",
        "",
        "逐题分数和答案见 `direct_vs_8b_scores.json`；"
        "完整直答原生输入、响应和 reasoning_content 见 `results_and_full_trajectories.json`。",
        "参考答案只在完成后的本地评分阶段读取。",
    ]
    (root / "direct_vs_8b_report.md").write_text("\n".join(lines) + "\n")
    aggregate(root, summary, selected)
    print(json.dumps({key: summary[key] for key in (
        "completed_each_arm", "paired_mutually_legal",
        "direct_mean_score", "harness_mean_score", "delta_percentage_points")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-path", required=True)
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()
    root = ROOT / "runs" / NAME
    root.mkdir(exist_ok=True)
    with (root / ".collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline, previous = time.monotonic() + 3600, None
        while True:
            state = query(args.control_path)
            current = state["state"]
            counts = (current.get("completed", 0), current.get("active_requests", 0),
                      current["status"])
            if counts != previous:
                print(json.dumps(state), flush=True)
                previous = counts
            if current["status"] == "complete" and current["completed"] == 40:
                break
            if not state["alive"] or current["status"] == "stopped_on_operational_error":
                write(root / "collector_stopped.json", state)
                raise RuntimeError("Direct stopped; no automatic model retry")
            if time.monotonic() >= deadline:
                raise RuntimeError("Direct collection exceeded its time bound")
            time.sleep(10)
        subprocess.run([
            "rsync", "-a", "-e", "ssh -o BatchMode=yes -S " + shlex.quote(args.control_path),
            "wangy@10.10.0.34:" + REMOTE + "/runs/" + NAME + "/", str(root) + "/"], check=True)
        validate(root)
        for name in ("DASHSCOPE_API_KEY", "QWEN3VL8B_API_KEY", "VSI_SAM2_TOKEN",
                     "OPENROUTER_API_KEY", "MAKERS_MODELS_KEY"):
            os.environ.pop(name, None)
        score(root, args.metadata, args.control_path)


if __name__ == "__main__":
    main()
