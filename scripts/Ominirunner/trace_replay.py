"""Read-only contract replay of saved public submissions, with zero inference/tool execution."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

if not __package__:
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    __package__ = "Ominirunner"

from skill3d.schemas.tool import ToolResult
from skill3d.schemas.tool_response import model_tool_response
from .answer_contract import canonical, specification
from .common import VERSION, sha, write
from .evidence import EvidenceError, dependency_ids, feedback, validate
from .native import ProgramFormatError, parse_json, parse_program, program_feedback
from .provenance import program_inputs
from .tools import MEASUREMENTS
from .review import apply_updates, final_payload, validate as validate_reviews, validate_program


def replay(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("replay_output_must_be_outside_frozen_run")
    paths = [source / "inputs.json"]
    items = json.loads(paths[0].read_text())
    paths += [source / "episodes" / f"{item['sample_id']}-{arm}" / "result.json"
              for item in items for arm in ("skill", "direct")]
    before = {str(p.relative_to(source)): sha(p) for p in paths}
    cases, statuses = [], Counter()
    for public_item in items:
        item = {k: public_item[k] for k in ("sample_id", "question", "task_type", "options")}
        task = specification(item)
        saved = json.loads((source / "episodes" / f"{item['sample_id']}-skill" / "result.json").read_text())
        if saved.get("reference_answer_visible") is not False or saved.get("real_labels_available") is not False:
            raise ValueError("source_episode_not_label_free")
        observations, reviews, dependencies, rounds = [], {}, [], []
        for row in saved["rounds"]:
            record = {"round": row["round"], "original_status": row["status"], "phase": row["phase"]}
            if row["phase"] != "final":
                try:
                    program = parse_program(row["output"])
                except ProgramFormatError as exc:
                    record.update(status="program_parse_error", executed=False,
                                  errors=program_feedback(exc))
                    rounds.append(record)
                    statuses[record["status"]] += 1
                    continue
                accepted, review_errors = validate_program(
                    program, observations, row["request"]["visible_measurement_ids"])
            else:
                program = None
                _, submitted, _ = final_payload(row["output"])
                accepted, review_errors = validate_reviews(
                    submitted, observations, row["request"]["visible_measurement_ids"])
            apply_updates(reviews, accepted, observations)
            record["review_errors"] = review_errors
            fresh = []
            # Project already-saved successful receipts only. No tool is called.
            for raw in row.get("tool_receipts", []):
                tool = ToolResult.model_validate(raw)
                if tool.status == "ok" and not tool.error and not tool.invalidated_by:
                    response = model_tool_response(tool)
                    observations.append(response)
                    if tool.tool in MEASUREMENTS:
                        fresh.append(tool.result_id)
            apply_updates(reviews, {}, observations)
            if "evidence_declaration" in row:
                evidence, answer = row["evidence_declaration"], row.get("submitted_native_answer")
                if row["phase"] == "final":
                    answer, _, evidence = final_payload(row["output"])
                prior = list(dependencies)
                dependencies = dependency_ids(evidence, observations, dependencies)
                origin = program_inputs(program, observations, fresh)
                record["program_input_origin"] = origin
                try:
                    checked = validate(
                        answer, evidence, observations, reviews,
                        row["request"]["visible_measurement_ids"], task,
                        visible_result_ids=row["request"]["visible_result_ids"],
                        previous_dependency_ids=prior, program_dependency_ids=origin["result_ids"])
                except (TypeError, ValueError) as exc:
                    record.update(status="answer_rejected", errors=feedback(exc if isinstance(exc, EvidenceError) else str(exc)))
                else:
                    record.update(status="answer_contract_accepted", canonical_answer=canonical(answer, task),
                                  verified_arithmetic=checked["verified_arithmetic"],
                                  origin_validation=checked.get("origin_validation"))
                # Demonstrate the scope migration using original entities and
                # numeric submission; never alter the saved declaration.
                if (task["quantity"] == "count_ratio" and isinstance(evidence, dict)
                        and isinstance(evidence.get("object_scope"), dict)):
                    upgraded = deepcopy(evidence)
                    groups = {}
                    for role in ("numerator", "denominator"):
                        category = next(o["target"] for o in task["operands"] if o["role"] == role)
                        spellings = {category, category[:-1] if category.endswith("s") else category}
                        groups[role] = [e["entity_id"] for e in upgraded["object_scope"].get("entities", [])
                                        if isinstance(e, dict) and e.get("included")
                                        and isinstance(e.get("category"), str)
                                        and (e["category"].casefold() in spellings
                                             or category.startswith(e["category"].casefold() + " "))]
                    upgraded["count_groups"] = groups
                    try:
                        migrated = validate(answer, upgraded, observations, reviews, [], task)
                    except (TypeError, ValueError) as exc:
                        record["scope_migration"] = {"status": "rejected", "errors": feedback(
                            exc if isinstance(exc, EvidenceError) else str(exc))}
                    else:
                        record["scope_migration"] = {"status": "accepted", "original_answer_preserved": True,
                                                    "ratio": migrated["scope_validation"]["ratio"]}
            else:
                record["status"] = "no_answer_submission"
            rounds.append(record)
            statuses[record["status"]] += 1
        direct = json.loads((source / "episodes" / f"{item['sample_id']}-direct" / "result.json").read_text())
        direct_answers = [parse_json(row["output"]) for row in direct["rounds"]]
        cases.append({"sample_id": item["sample_id"], "public_task": task, "skill_rounds": rounds,
                      "direct_scalar_submissions": len(direct_answers),
                      "independent_visual_geometry_verified": False})
    after = {str(p.relative_to(source)): sha(p) for p in paths}
    if before != after:
        raise ValueError("frozen_source_changed_during_replay")
    summary = {"schema": "omni-read-only-contract-replay/1", "implementation": VERSION,
               "source": str(source), "cases": len(cases),
               "skill_rounds": sum(len(c["skill_rounds"]) for c in cases),
               "direct_rounds": sum(c["direct_scalar_submissions"] for c in cases),
               "validation_statuses": dict(statuses), "source_files_verified_unchanged": len(paths),
               "model_requests": 0, "tool_executions": 0, "score_reads": 0,
               "reference_answer_visible": False,
               "independent_visual_geometry_verified": False,
               "scope": "saved public declarations and receipts only; no new predictions or scores"}
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "replay.json"
    if artifact.exists():
        existing = json.loads(artifact.read_text()).get("summary", {})
        if existing.get("schema") != summary["schema"] or existing.get("source") != str(source):
            raise ValueError("replay_output_belongs_to_another_source")
    write(artifact, {"summary": summary, "source_hashes": before, "cases": cases})
    lines = ["# Omni3D 20 题契约修复离线回放", "",
             f"检查原架构 {summary['skill_rounds']} 轮及直跑 {summary['direct_rounds']} 轮。零模型请求、零工具执行、未读取评分或参考答案。",
             f"{len(paths)} 个输入/原结果文件在回放前后哈希一致；输出保存在独立目录。",
             "", "这是原提交在新契约下的诊断，不是新一轮评测。原视觉声明仍由同模型给出，未获得独立几何认证。",
             "", "| 题号 | 新题型 | 原提交的主要新诊断 |", "|---|---|---|"]
    for case in cases:
        codes = sorted({e["code"] for r in case["skill_rounds"] for e in r.get("errors", [])})
        lines.append(f"| {case['sample_id']} | {case['public_task']['quantity']} | {', '.join(codes) or '无答案契约拒绝'} |")
    (output / "replay.md").write_text("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    print(json.dumps(replay(args.source, args.output), ensure_ascii=False))
