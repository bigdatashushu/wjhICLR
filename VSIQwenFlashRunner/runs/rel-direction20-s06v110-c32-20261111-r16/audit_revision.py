"""Audit actual revised-method delivery and compare the same twenty questions."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = Path("/Users/bytedance/wangjh/VSI相对方向20题S06新版复测-20261111.md")
PREVIOUS = Path("/Users/bytedance/wangjh/VSI相对方向新增20题g1m0与直答对比-20261111.json")
SKILL_SHA = "dc126016213649bc3a967f9d22637b0a7934e74105208ff3485826502b94b49a"
PROMPT_SHA = "4c679b59ef7de0ebf0129bbf03cdecc76791d08e80744cceffdbf07090b8537b"
RULE = ("- M5/Tool IDs identify candidates, not question referents or scene completeness. "
        "Match depicted regions and spatial qualifiers before selecting IDs; keep missing "
        "bindings visual. M11 checks technical validity/derivation, not intended identity.")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = read(ROOT / "comparison.json")
    old = read(PREVIOUS)
    summary = result["summary"]
    assert summary["n_pairs"] == 20
    assert summary["system_initial_skill_versions"] == ["S06-g1m0@1.1.0"]
    assert summary["system_evidence_states"] == ["g1m0"]
    assert summary["direct_results_reused"] is True
    old_rows = {row["qa_id"]: row for row in old["pairs"]}
    assert set(old_rows) == {row["qa_id"] for row in result["pairs"]}
    skill_path = ROOT / "code/source/skill_library/versions/S06-g1m0/1.1.0/judge-relative-direction-g1m0/SKILL.md"
    assert sha(skill_path) == SKILL_SHA
    assert sha(ROOT / "code/source/src/skill3d/synthesis/prompt_builder.py") == PROMPT_SHA
    skill = skill_path.read_text(encoding="utf-8")
    delivery = {}
    outcomes = list((ROOT / "system/variants").glob("*/outcome.json"))
    completion = summary.get("storage_completion") or summary.get("transport_completion")
    if completion:
        completion_root = Path(completion["completion_root"])
        outcomes += list((completion_root / "system/variants").glob("*/outcome.json"))
    expected = set(old_rows)
    for path in outcomes:
        outcome = read(path)
        qa = outcome["qa_id"]
        assert qa in expected and qa not in delivery
        selected = outcome["initial_skill_selection"]
        assert selected["skill_version"] == "S06-g1m0@1.1.0"
        assert selected["content_sha256"] == SKILL_SHA
        prompt = outcome["m8_prompt"]
        assert skill in prompt and RULE in prompt
        delivery[qa] = {
            "skill_version": selected["skill_version"],
            "skill_sha256": selected["content_sha256"],
            "full_skill_in_actual_prompt": True,
            "shared_rule_in_actual_prompt": True,
            "actual_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }
    assert set(delivery) == expected
    improved, decreased = [], []
    table = [
        "| QA | 修改前答案 | 新版答案 | 参考答案 | 修改前 | 新版 |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in result["pairs"]:
        previous = old_rows[row["qa_id"]]
        assert row["direct_answer"] == previous["direct_answer"]
        assert row["direct_score"] == previous["direct_score"]
        delta = row["system_score"] - previous["system_score"]
        row["previous_system_answer"] = previous["system_answer"]
        row["previous_system_score"] = previous["system_score"]
        row["delta_from_previous_system"] = delta
        if delta > 0:
            improved.append(row["qa_id"])
        elif delta < 0:
            decreased.append(row["qa_id"])
        table.append(
            f"| {row['qa_id']} | {previous['system_answer']} | {row['system_answer']} | "
            f"{row['reference_answer']} | {previous['system_score']:.0%} | {row['system_score']:.0%} |")
    old_tokens = old["summary"]["token_usage"]["system"]["total_tokens"]
    new_tokens = summary["token_usage"]["system"]["total_tokens"]
    final_episode_tokens = summary.get(
        "completed_episode_known_token_usage", summary["token_usage"]["system"])["total_tokens"]
    old_score = old["summary"]["system_score"]
    revision = {
        "previous_report_sha256": sha(PREVIOUS),
        "previous_diagnostic_identity": old["summary"]["diagnostic_identity"],
        "same_twenty_questions": True,
        "evidence_role": "same-question diagnostic retest after trajectory-informed revision",
        "previous_system_score": old_score,
        "score_delta": summary["system_score"] - old_score,
        "improved_qa_ids": improved,
        "decreased_qa_ids": decreased,
        "unchanged_score_count": 20 - len(improved) - len(decreased),
        "previous_known_total_tokens": old_tokens,
        "known_token_change_fraction": new_tokens / old_tokens - 1,
        "completed_twenty_episode_known_total_tokens": final_episode_tokens,
        "completed_twenty_episode_token_change_fraction": final_episode_tokens / old_tokens - 1,
        "discarded_episode_known_total_tokens": new_tokens - final_episode_tokens,
        "method_delivery_verified_questions": len(delivery),
        "method_delivery": delivery,
        "wrong_qa_ids": [r["qa_id"] for r in result["pairs"] if r["system_score"] == 0],
    }
    summary["revision_retest"] = revision
    for path in (ROOT / "comparison.json", OUTPUT.with_suffix(".json")):
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "method_delivery_audit.json").write_text(
        json.dumps(revision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    new_correct = sum(r["system_score"] == 1 for r in result["pairs"])
    delta_pp = 100 * revision["score_delta"]
    opening = [
        "# VSI 相对方向：S06 新版同二十题复测", "",
        f"新版工具组 **{new_correct}/20（{summary['system_score']:.0%}）**；"
        f"修改前 **17/20（85%）**，变化 **{delta_pp:+.1f} 个百分点**；"
        "已有 Qwen 直答 **16/20（80%）**。", "",
        f"错转对：{'、'.join(improved) or '无'}。"
        f"对转错：{'、'.join(decreased) or '无'}。"
        f"新版错题：{'、'.join(revision['wrong_qa_ids']) or '无'}。", "",
        f"已知 Token **{new_tokens:,}**，修改前 **{old_tokens:,}**，"
        f"变化 **{revision['known_token_change_fraction']:+.1%}**；"
        f"为复用直答的 **{summary['known_total_token_ratio']:.2f} 倍**。", "",
        f"其中最终 20 条完整轨迹的已知 Token 为 **{final_episode_tokens:,}**，"
        f"较修改前变化 **{revision['completed_twenty_episode_token_change_fraction']:+.1%}**；"
        f"另有 **{new_tokens - final_episode_tokens:,}** Token 来自两道存储故障题"
        "在原批次中的 4 次成功调用，已计入上面的实际总消耗。超时请求的 Token 仍未知。", "",
        "20/20 道完成轨迹均核对了 `S06-g1m0@1.1.0` 的版本和正文哈希，"
        "实际提示中包含完整新版 Skill 和替换后的共享规则。"
        "模型参数、原始 32 帧、base/g1m0 数据与求解预算沿用此前协议。", "",
        "这批题目曾用于轨迹诊断和方法修订，本轮是同题诊断复测；"
        "结果反映这 20 题上的变化，不能据此估计未见题的泛化收益。", "",
    ]
    prior_text = OUTPUT.read_text(encoding="utf-8").splitlines()
    OUTPUT.write_text(
        "\n".join(opening + table + ["", "## 运行、配对与离线评分记录", ""] + prior_text[2:]) + "\n",
        encoding="utf-8")
    print(json.dumps({k: v for k, v in revision.items() if k != "method_delivery"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
