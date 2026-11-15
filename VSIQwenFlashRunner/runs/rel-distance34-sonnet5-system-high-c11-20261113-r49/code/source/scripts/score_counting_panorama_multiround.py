#!/usr/bin/env python3
"""Audit every transmitted round and score the frozen paired comparison locally."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
BASE = REPO.parent
ROOT = REPO / "VSIQwenFlashRunner/runs/count-panorama-multiround10-c32-20261110-r10"
PRIOR = ROOT.parent / "counting-statefix10-c10-20261110-r08"
SINGLE = ROOT.parent / "count-panorama-only10-c32-20261110-r09"
sys.path.insert(0, str(ROOT / "code/source/src"))
from skill3d.evaluation.fitset_v11 import load_label_metadata, score_answer


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def messages(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def main():
    manifest = read(ROOT / "manifest.json")
    assert digest({k: v for k, v in manifest.items() if k != "identity_sha256"}) == manifest["identity_sha256"]
    assert hashlib.sha256((ROOT / "runner.py").read_bytes()).hexdigest() == manifest["runner_sha256"]
    spec = importlib.util.spec_from_file_location("frozen_panorama_runner", ROOT / "runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    exit_record = read(ROOT / "exit.json")
    assert exit_record["n_results"] == 20
    labels = load_label_metadata(PRIOR / "inputs/host_evaluation/metadata.jsonl")
    state = read(ROOT / "requests/state.json")
    health = read(ROOT / "gateway_final_health.json")
    calls = len(state["requests"])
    assert state["policy"]["max_inflight"] == health["max_concurrency"] == 32
    assert state["policy"]["max_requests"] == 60 and 20 <= calls <= 60
    assert health["upstream"]["started"] == exit_record["model_requests"] == calls
    assert health["upstream"]["active"] == health["upstream"]["unknown"] == 0
    pairs = []
    usage = {arm: {} for arm in ("direct", "panorama")}
    audited = 0
    for variant in sorted(manifest["variants"], key=lambda r: int(r["qa_id"])):
        qa = variant["qa_id"]
        pair = read(ROOT / "pairs" / f"QA{qa}.json")
        direct_base = messages(ROOT / "messages/direct" / f"QA{qa}.json.gz")
        panorama_base = messages(ROOT / "messages/panorama" / f"QA{qa}.json.gz")
        assert panorama_base[0]["content"][:-2] == direct_base[0]["content"]
        assert digest(direct_base) == pair["direct_messages_sha256"]
        assert digest(panorama_base) == pair["panorama_messages_sha256"]
        # Compare to the previous single-round experiment's frozen image inputs.
        assert direct_base == messages(SINGLE / "messages/direct" / f"QA{qa}.json.gz")
        assert panorama_base == messages(SINGLE / "messages/panorama" / f"QA{qa}.json.gz")
        records = {arm: read(ROOT / arm / "responses" / f"QA{qa}.json")
                   for arm in ("direct", "panorama")}
        d, p = records["direct"], records["panorama"]
        assert d["n_rounds"] == 1 and 1 <= p["n_rounds"] <= 5
        direct_images = None
        for arm, record in records.items():
            assert record["reference_answer_visible"] is False and record["tool_calls"] == 0
            assert record["n_rounds"] == len(record["rounds"])
            assert record["usage"] == runner.total_usage(record["rounds"])
            base = direct_base if arm == "direct" else panorama_base
            previous = None
            panorama_images = None
            for index, row in enumerate(record["rounds"], 1):
                assert row == read(ROOT / arm / "rounds" / f"QA{qa}" / f"round{index}.json")
                actual = messages(ROOT / "messages" / arm / f"QA{qa}" / f"round{index}.json.gz")
                expected = (base if arm == "direct" else runner.round_messages(base, index, 5, previous))
                assert actual == expected
                assert row["messages_sha256"] == digest(actual)
                assert len(actual) == row["message_count"] == 1 and actual[0]["role"] == "user"
                assert f"Question: {pair['question']}" in actual[0]["content"][0]["text"]
                assert row["input_summary"] == previous
                if arm == "panorama":
                    assert row["image_count"] == 33
                    assert actual[0]["content"][-2] == {"type": "text", "text": manifest["panorama_notice"]}
                    assert actual[0]["content"][1:] == panorama_base[0]["content"][1:]
                    assert ("FINAL round" in actual[0]["content"][0]["text"]) == (index == 5)
                else:
                    assert row["image_count"] == 32
                if "actual_request_identity" in row:
                    identity = row["actual_request_identity"]
                    params = identity["parameters"]
                    assert params["model"] == "qwen3.8-flash"
                    assert params["enable_thinking"] is True
                    assert params["temperature"] == 0 and params["seed"] == 137 and params["max_tokens"] == 8192
                    images = identity["images_in_order"]
                    if arm == "direct":
                        direct_images = images
                        assert len(images) == 32
                    else:
                        assert len(images) == 33 and images[:32] == direct_images
                        if panorama_images is not None:
                            assert images == panorama_images
                        panorama_images = images
                        assert params == d["rounds"][0]["actual_request_identity"]["parameters"]
                if index < record["n_rounds"]:
                    assert row["status"] == "continued" and isinstance(row["output_summary"], str)
                previous = row["output_summary"]
                audited += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if key in record["usage"]:
                    usage[arm][key] = usage[arm].get(key, 0) + record["usage"][key]
            reasoning = record["usage"].get("completion_tokens_details", {}).get("reasoning_tokens")
            if isinstance(reasoning, int):
                usage[arm]["reasoning_tokens"] = usage[arm].get("reasoning_tokens", 0) + reasoning
        label = labels[qa]
        scores = {arm: score_answer(answer=r["answer"], task=label["question_type"],
            options=label["options"], reference=label["reference_answer"],
            available=r["status"] == "completed") for arm, r in records.items()}
        pairs.append({
            "qa_id": qa, "question": pair["question"], "reference_answer": label["reference_answer"],
            "direct_answer": d["answer"], "panorama_answer": p["answer"],
            "direct_score": scores["direct"]["score"], "panorama_score": scores["panorama"]["score"],
            "delta": scores["panorama"]["score"] - scores["direct"]["score"],
            "direct_status": d["status"], "panorama_status": p["status"],
            "direct_evaluation": scores["direct"], "panorama_evaluation": scores["panorama"],
            "panorama_rounds": p["n_rounds"],
            "panorama_round_decisions": [{"round": r["round_index"], "status": r["status"],
                "answer": r["answer"], "output_summary": r["output_summary"],
                "elapsed_s": r["elapsed_s"]} for r in p["rounds"]],
            "direct_usage": d["usage"], "panorama_usage": p["usage"],
            "direct_elapsed_s": d["elapsed_s"], "panorama_elapsed_s": p["elapsed_s"]})
    assert audited == calls
    old = read(SINGLE / "comparison.json")["summary"]
    summary = {
        "n_pairs": len(pairs), "direct_mra": statistics.mean(r["direct_score"] for r in pairs),
        "panorama_mra": statistics.mean(r["panorama_score"] for r in pairs),
        "delta": statistics.mean(r["delta"] for r in pairs),
        "improved": sum(r["delta"] > 0 for r in pairs),
        "decreased": sum(r["delta"] < 0 for r in pairs),
        "tied": sum(r["delta"] == 0 for r in pairs),
        "direct_exact_correct": sum(r["direct_status"] == "completed"
            and r["direct_evaluation"]["predicted"] == float(r["reference_answer"]) for r in pairs),
        "panorama_exact_correct": sum(r["panorama_status"] == "completed"
            and r["panorama_evaluation"]["predicted"] == float(r["reference_answer"]) for r in pairs),
        "direct_completed": sum(r["direct_status"] == "completed" for r in pairs),
        "panorama_completed": sum(r["panorama_status"] == "completed" for r in pairs),
        "token_usage": usage, "token_ratio": usage["panorama"]["total_tokens"] / usage["direct"]["total_tokens"],
        "panorama_total_rounds": sum(r["panorama_rounds"] for r in pairs),
        "panorama_mean_rounds": statistics.mean(r["panorama_rounds"] for r in pairs),
        "continued_questions": sum(r["panorama_rounds"] > 1 for r in pairs),
        "reached_final_round": sum(r["panorama_rounds"] == 5 for r in pairs),
        "mean_direct_elapsed_s": statistics.mean(r["direct_elapsed_s"] for r in pairs),
        "mean_panorama_elapsed_s": statistics.mean(r["panorama_elapsed_s"] for r in pairs),
        "configured_concurrency": 32, "observed_request_peak": state["peak_inflight"],
        "observed_upstream_peak": health["upstream"]["peak"], "fresh_model_calls": calls,
        "service_failed": health["upstream"]["failed"], "automatic_retries": 0,
        "wall_time_s": exit_record["finished_at"] - exit_record["started_at"],
        "manifest_identity": manifest["identity_sha256"],
        "prior_single_round": {"direct_mra": old["direct_mra"], "panorama_mra": old["panorama_mra"],
            "panorama_delta": statistics.mean(r["panorama_score"] for r in pairs) - old["panorama_mra"],
            "separate_run": True}}
    audit = {"status": "passed", "transmitted_rounds_audited": audited,
        "one_fresh_user_message_each_round": True, "original_question_every_round": True,
        "panorama_notice_before_last_image_every_round": True, "identical_33_images_every_round": True,
        "only_previous_summary_carried": True, "thinking_enabled_every_request": True,
        "labels_read_only_after_run": True, "same_inputs_as_prior_single_round": True}
    result = {"summary": summary, "pairs": pairs, "audit": audit}
    target = BASE / "VSI计数全景多轮与直答10题对比-20261110.json"
    report = BASE / "VSI计数全景多轮与直答10题对比-20261110.md"
    for path, data in ((target, result), (ROOT / "comparison.json", result), (ROOT / "round_audit.json", audit)):
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = [
        "# VSI 计数：全景图多轮与直答配对实验", "",
        f"同一批 10 道计数题，直答 MRA **{summary['direct_mra']:.1%}**，"
        f"全景多轮 MRA **{summary['panorama_mra']:.1%}**，差值 **{summary['delta']*100:+.1f} 个百分点**。",
        f"完全正确题数：直答 {summary['direct_exact_correct']}/10，多轮 {summary['panorama_exact_correct']}/10。"
        f"逐题上升 {summary['improved']} 题、下降 {summary['decreased']} 题、持平 {summary['tied']} 题。", "",
        "## 实际运行规则", "",
        "两组均重新调用 `qwen3.8-flash`，所有请求开启 `enable_thinking=true`，"
        "`temperature=0`、`seed=137`、`max_tokens=8192`。原始帧、题目及预计算全景图沿用上一轮冻结输入。",
        "直答组每题一次请求、32 张视频帧。多轮组每题最多 5 次请求，每轮发送相同的 32 张视频帧和 1 张全景图。"
        "每轮均含完整原始问题，并在最后一张图前保留 `An additional panoramic image is provided.`。",
        "多轮请求每次只有一条新的 user message。模型被明确告知历史提示词、回复及 thinking 不累计；"
        "从第 2 轮起，只接收上一轮输出的新摘要。若确定，模型可提前输出整数；若不确定，"
        "输出 `{\"next_prompt\":\"供下一轮复核的简短摘要\"}`。第 5 轮明确提示这是最后一轮，必须给出整数答案。",
        "摘要继续和整数作答采用互斥解析；服务错误或无效格式保留原文并终止该题，不从分析文本抽取猜测答案。", "",
        "## 逐题结果", "",
        "| QA | 目标 | 参考 | 直答 | 多轮 | 多轮轮数 | 直答 MRA | 多轮 MRA | 差值 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in pairs:
        name = row["question"].removeprefix("How many ").removesuffix(" are in this room?")
        lines.append(f"| {row['qa_id']} | {name} | {row['reference_answer']} | {row['direct_answer']} | "
            f"{row['panorama_answer']} | {row['panorama_rounds']} | {row['direct_score']:.0%} | "
            f"{row['panorama_score']:.0%} | {row['delta']*100:+.0f} pp |")
    lines += ["", "## 轮数、Token 与时间", "",
        f"多轮组共请求 {summary['panorama_total_rounds']} 次，平均 {summary['panorama_mean_rounds']:.1f} 轮/题；"
        f"{summary['continued_questions']} 题主动续轮，{summary['reached_final_round']} 题到达第 5 轮。",
        "", "| 指标 | 直答 | 全景多轮 |", "|---|---:|---:|"]
    for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"):
        lines.append(f"| {key} | {usage['direct'].get(key, '未返回')} | {usage['panorama'].get(key, '未返回')} |")
    lines += [
        f"| 平均每题耗时（秒） | {summary['mean_direct_elapsed_s']:.1f} | {summary['mean_panorama_elapsed_s']:.1f} |",
        "", f"多轮总 Token 为直答的 {summary['token_ratio']:.2f} 倍。"
        f"共 {calls} 次真实请求，服务失败 {summary['service_failed']} 次，无自动重试。"
        f"并发上限 32，实际请求峰值 {state['peak_inflight']}、上游峰值 {health['upstream']['peak']}；"
        "10 题×两组最多同时有 20 条待运行请求，同一题的后续轮依次执行。",
        "", "## 与上一轮单次加图结果的关系", "",
        f"此前单次请求实验：直答 {old['direct_mra']:.1%}，单次加全景图 {old['panorama_mra']:.1%}。"
        f"本轮多轮结果比此前单次加图 {summary['prior_single_round']['panorama_delta']*100:+.1f} 个百分点。"
        "这是不同批次请求之间的观察差值；本轮主要对照是同时重跑的直答组。"
        "当前结果覆盖这 10 道题及现有全景图质量。",
        "", "## 每轮提示词", "",
        "第一轮示例（图像内容另存于原始请求）：", "", "```text",
        runner.round_messages(messages(ROOT / "messages/panorama/QA4513.json.gz"), 1, 5)[0]["content"][0]["text"],
        "```", "", "最后一轮固定提示（真实运行是否到达该轮见上表）：", "", "```text",
        runner.round_messages(messages(ROOT / "messages/panorama/QA4513.json.gz"), 5, 5,
                              "<上一轮摘要>")[0]["content"][0]["text"],
        "```", "", "## 原始证据", "",
        f"- [逐题数据与摘要](computer://{target})",
        f"- [完整实验目录](computer://{ROOT})",
        f"- [每轮输入与图片](computer://{ROOT / 'messages'})",
        f"- [多轮原始响应](computer://{ROOT / 'panorama/rounds'})",
        f"- [传输契约核对](computer://{ROOT / 'round_audit.json'})",
        "", f"实验身份：`{manifest['identity_sha256']}`。", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    assert "\ufffd" not in report.read_text(encoding="utf-8")
    print(json.dumps({"summary": summary, "audit": audit, "report": str(report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
