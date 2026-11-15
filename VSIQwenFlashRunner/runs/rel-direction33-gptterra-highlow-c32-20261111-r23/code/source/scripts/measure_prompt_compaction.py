#!/usr/bin/env python3
"""Measure compact prompts offline using synthetic inputs and optional public text."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill3d.evolution.fitset_revision_v11 import load_active_parent_slots
from skill3d.infra.vsi_runner_source import seed_library_root
from skill3d.online import runner
from skill3d.online.vsi_execution_contract import finalization_instructions
from skill3d.schemas.answer import CANONICAL_UNIT_BY_QUESTION_TYPE
from skill3d.schemas.evidence import EvidenceProfile
from skill3d.skills.delivery import DeliveredSkill, SkillDeliveryPlan
from skill3d.synthesis import prompt_builder
from skill3d.synthesis.prompt_compaction import (
    COMPACTION_VERSION, compact_m5_coverage, compact_rendered_tool_docs,
)
from skill3d.tools import REGISTRY
from skill3d.tools.contract import KNOWN_ARTIFACTS

TASK_NAMES = {
    "object_counting": "物体计数", "object_abs_distance": "绝对距离",
    "object_rel_distance": "相对距离", "object_size_estimation": "物体尺寸",
    "room_size_estimation": "房间面积", "object_rel_direction": "相对方向",
    "obj_appearance_order": "出现顺序", "route_planning": "路径规划",
}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def baseline_module(ref, relative, name):
    text = subprocess.check_output(
        ["git", "show", f"{ref}:{relative}"], cwd=ROOT.parent, text=True)
    module = ModuleType(name)
    exec(compile(text, f"<{ref}:{relative}>", "exec"), module.__dict__)
    return module, sha(text)


def context(task, state):
    geometry = state != "g0m0"
    metric = state == "g1m1" and task in {
        "object_abs_distance", "object_size_estimation", "room_size_estimation"}
    return dict(
        scope="metric_enabled" if metric else "full_3d" if geometry else "fallback_2d_only",
        available=KNOWN_ARTIFACTS, gate_passed=metric, question_type=task,
        evidence_profile=EvidenceProfile(
            geometry_3d="available" if geometry else "unavailable",
            metric_scale="available" if state == "g1m1" else "unavailable"),
        local_facts={"world_frame": True})


def synthetic_lengths(old_builder, old_interfaces):
    parents, seed_identity = load_active_parent_slots(seed_library_root(ROOT))
    rows = []
    originals = [f"orig-{i}" for i in range(32)]
    image_plan = SimpleNamespace(
        image_ids=originals, original_image_ids=originals,
        derived_image_ids=[], omitted_originals=[])
    old_manifest = "\n\n## Images for this round (in order, matching the images in this request)\n"
    old_manifest += "\n".join(f"{i+1}. [Original frame] frame_id={i}" for i in range(32)) + "\n"
    new_manifest = runner._image_manifest_text(None, image_plan)
    for task in TASK_NAMES:
        for state in ("g0m0", "g1m0", "g1m1"):
            ctx = context(task, state)
            options = (["A. left", "B. right"] if task == "object_rel_direction"
                       else ["A. first", "B. second"] if CANONICAL_UNIT_BY_QUESTION_TYPE[task] == "option"
                       else [])
            ep = SimpleNamespace(
                question="Synthetic spatial question preserving object roles and the stated scope.",
                question_type=task, options=options)
            scene = SimpleNamespace(
                question_type=task, question_tool_scope=ctx["scope"], summary="Synthetic scene.",
                available_artifacts=ctx["available"], grounding_coverage={},
                local_facts=ctx["local_facts"], evidence_profile=ctx["evidence_profile"],
                metric_evidence_gate_result=SimpleNamespace(
                    gate_passed=ctx["gate_passed"], missing_subconditions=[]))
            skill = parents[(task, state)]
            new, plan = runner._build_prompt_ex(ep, scene, scene, [skill], None)
            old = old_builder.PromptBuilder().render(
                question=ep.question, scene_summary=scene.summary, scene_frame="world",
                tool_docs=REGISTRY.docs(**ctx), options=options, skill_plan=plan,
                scope=ctx["scope"], available_artifacts=sorted(ctx["available"]),
                question_type=task, evidence_profile=ctx["evidence_profile"],
                gate_passed=ctx["gate_passed"], gate_missing=[])
            old += old_interfaces.program_instructions(task) + old_manifest
            new += new_manifest
            assert skill.skill_md in old and skill.skill_md in new
            assert len(new) < len(old)
            old_final = old_builder.build_finalization_prompt(
                question=ep.question, options=options, question_type=task,
                observations="No historical observations.", remaining_requests=1,
                method_context_max_chars=plan.max_chars) + old_interfaces.finalization_instructions(task)
            new_final = prompt_builder.build_finalization_prompt(
                question=ep.question, options=options, question_type=task,
                observations="No historical observations.", remaining_requests=1,
                method_context_max_chars=plan.max_chars) + finalization_instructions(task)
            assert len(new_final) < len(old_final)
            rows.append(dict(
                task=task, state=state, before_chars=len(old), after_chars=len(new),
                reduction_pct=round(100 * (1 - len(new) / len(old)), 2),
                skill_chars=len(skill.skill_md), skill_sha256=sha(skill.skill_md),
                finalization_before_chars=len(old_final), finalization_after_chars=len(new_final)))
    return rows, seed_identity


def project_public_prompt(path, old_builder, output):
    """Project only the original public prompt, without answers/results/history."""
    original = path.read_text()
    question_section = original.split("## Current question and answer requirements\n", 1)[1].split(
        "## Current evidence state\n", 1)[0]
    question, details = question_section.split("\nQuestion type: ", 1)
    task = details.splitlines()[0].strip()
    options = (details.split("Options:\n", 1)[1].split("MCA:", 1)[0].strip().splitlines()
               if "Options:\n" in details else [])
    evidence = original.split("## Current evidence state\n", 1)[1].split(
        "## Authorized tools and interface documentation\n", 1)[0]
    route, scene = evidence.split("Scene summary: ", 1)
    scene = scene.split("\nCoordinate and scale availability", 1)[0]
    for line in scene.splitlines():
        if line.startswith("M5 candidate coverage"):
            prefix, raw = line.split(": ", 1)
            compact = compact_m5_coverage(json.loads(raw))
            scene = scene.replace(line, prefix + ": " + json.dumps(
                compact, ensure_ascii=False, sort_keys=True, separators=(",", ":")), 1)
    docs = original.split("## Authorized tools and interface documentation\n", 1)[1].split(
        "\nCall only tools listed in the current documentation", 1)[0]
    methods = original.split("## Reference method\n", 1)[1].split(
        "\n\nVSI executable output supplement", 1)[0]
    max_chars = int(re.search(r'"method_context_max_chars":\s*(\d+)', original).group(1))
    # The original complete method is used byte-for-byte, with its actual digest.
    plan = SkillDeliveryPlan(
        entries=[DeliveredSkill("public-frozen-method", sha(methods), len(methods), methods)],
        max_chars=max_chars, used_chars=len(methods))
    new = prompt_builder.PromptBuilder().render(
        question=question, scene_summary=scene, scene_frame="world",
        tool_docs=compact_rendered_tool_docs(docs), options=options, question_type=task,
        route_header=route.strip(), skill_plan=plan)
    authorized = set(re.findall(r"(?m)^- (\w+)\(", docs))
    from skill3d.online.vsi_execution_contract import program_instructions

    new += program_instructions(task, authorized_tools=authorized)
    manifest = original.split(
        "## Images for this round (in order, matching the images in this request)", 1)[1]
    slots = [int(value) for value in re.findall(r"\[Original frame\] frame_id=(\d+)", manifest)]
    derived = [line for line in manifest.splitlines()
               if "[Crop]" in line or "[Scene overview]" in line]
    lines = ["", "## Images for this round (in order, matching the images in this request)"]
    lines += [line.replace(old_builder.SCENE_OVERVIEW_GUIDANCE,
                           prompt_builder.SCENE_OVERVIEW_GUIDANCE) for line in derived]
    if slots:
        lines.append(f"{len(derived)+1}..{len(derived)+len(slots)}. [Original frames] "
                     "frozen frame slots in image order: " + json.dumps(slots, separators=(",", ":")))
    omitted_note = re.search(r"\(\d+ original frames are absent.*", manifest, re.S)
    if omitted_note:
        lines.append(omitted_note.group(0).strip())
    new += "\n".join(lines) + "\n"
    assert methods in new and question in new
    assert authorized == set(re.findall(r"(?m)^- (\w+)\(", compact_rendered_tool_docs(docs)))
    output.write_text(new)
    return dict(
        source=str(path.resolve()), source_sha256=sha(original), task=task,
        before_chars=len(original), after_chars=len(new),
        reduction_pct=round(100 * (1 - len(new) / len(original)), 2),
        complete_method_sha256=sha(methods), complete_method_chars=len(methods),
        images=len(slots) + len(derived), original_slots=slots,
        compact_prompt=str(output.resolve()), compact_sha256=sha(new),
        measurement="Offline projection of the frozen public initial prompt using current templates.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="HEAD")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--public-prompt", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    old, builder_hash = baseline_module(
        args.baseline_ref, "src/skill3d/synthesis/prompt_builder.py", "baseline_prompt")
    old_interfaces, interfaces_hash = baseline_module(
        args.baseline_ref, "VSIQwenFlashRunner/src/skill3d/online/vsi_execution_contract.py",
        "baseline_interfaces")
    rows, seed = synthetic_lengths(old, old_interfaces)
    metrics = dict(
        compaction_version=COMPACTION_VERSION, model_requests=0,
        baseline_commit=subprocess.check_output(
            ["git", "rev-parse", args.baseline_ref], cwd=ROOT.parent, text=True).strip(),
        baseline_builder_sha256=builder_hash, baseline_interfaces_sha256=interfaces_hash,
        current_builder_sha256=hashlib.sha256(Path(prompt_builder.__file__).read_bytes()).hexdigest(),
        seed_identity=seed, synthetic_rows=rows,
        synthetic_input="Same neutral question, exact active Skill, authorized tool upper bound, "
                        "32 ordered original frames, no M5 failures or model responses; 24 task/state pairs.")
    if args.public_prompt:
        metrics["public_projection"] = project_public_prompt(
            args.public_prompt, old, output / "public-prompt-compact.txt")
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    text = [
        "# 八类题型提示词精简结果", "",
        "已接入 VSIQwenFlashRunner 本地普通轮、恢复轮和最终 JSON 轮。合并重复执行规则，"
        "按题型显示推导说明，压缩授权工具描述和有序图像清单；M5 相同失败按原因累计次数。"
        "完整 Skill 正文、真实工具授权、证据检查和推导重放继续使用原契约。", "",
        "以下使用相同的合成公开问题、当前完整 Skill、G1M1 工具能力上界和 32 张原帧清单。"
        "不含 M5 失败列表或模型回复，用于衡量固定提示词开销；不是实际题目平均长度。", "",
        "| 题型 | 精简前字符 | 精简后字符 | 减少 |",
        "|---|---:|---:|---:|",
    ]
    text += [f"| {TASK_NAMES[row['task']]} | {row['before_chars']:,} | "
             f"{row['after_chars']:,} | {row['reduction_pct']}% |"
             for row in rows if row["state"] == "g1m1"]
    if args.public_prompt:
        public = metrics["public_projection"]
        text += [
            "", f"QA373 原始公开上下文的离线投影：**{public['before_chars']:,} → "
            f"{public['after_chars']:,} 字符，减少 {public['reduction_pct']}%**。"
            f"保留 {public['images']} 张图的顺序映射和完整方法正文。"
            "这份长度包含原题的 M5 诊断；相同失败原因聚合后仍保留阻断状态和扫描覆盖。",
            "", "[查看 QA373 精简后的完整提示词](public-prompt-compact.txt)",
        ]
    text += [
        "", "metrics.json 包含全部八题型 × 三证据状态的字符数、最终轮字符数和输入/源码身份。"
        "请求审计记录精简文档的实际哈希及源文档版本；最终轮的工具文档交付为空。",
        "", "未请求真实模型。字符减少不等同于 token、费用或准确率的同等变化。",
    ]
    (output / "README.md").write_text("\n".join(text) + "\n")
    print(json.dumps({
        "model_requests": 0, "synthetic_pairs": len(rows),
        "g1m1": [row for row in rows if row["state"] == "g1m1"],
        "public_projection": metrics.get("public_projection"), "report": str(output / "README.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
