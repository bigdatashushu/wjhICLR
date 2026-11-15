#!/usr/bin/env python3
"""Offline counting projection and exact comparison with a frozen source bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
IMAGE_HEADER = "\n## Images for this round (in order, matching the images in this request)"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def synthetic_contexts(source_root):
    sys.path.insert(0, str(source_root / "src"))
    from skill3d.evolution.fitset_revision_v11 import load_active_parent_slots
    from skill3d.online import runner
    from skill3d.online.vsi_execution_contract import finalization_instructions
    from skill3d.schemas.answer import CANONICAL_UNIT_BY_QUESTION_TYPE
    from skill3d.schemas.evidence import EvidenceProfile
    from skill3d.synthesis.prompt_builder import build_finalization_prompt
    from skill3d.tools import REGISTRY
    from skill3d.tools.contract import KNOWN_ARTIFACTS
    from skill3d.synthesis.prompt_compaction import compact_tool_docs

    parents, _ = load_active_parent_slots(source_root / "skill_library")
    rows = []
    for task, unit in sorted(CANONICAL_UNIT_BY_QUESTION_TYPE.items()):
        for state in ("g0m0", "g1m0", "g1m1"):
            geometry, metric = state != "g0m0", state == "g1m1"
            gate = metric and task in {
                "object_abs_distance", "object_size_estimation", "room_size_estimation"}
            scope = "metric_enabled" if gate else "full_3d" if geometry else "fallback_2d_only"
            ctx = dict(
                scope=scope, available=KNOWN_ARTIFACTS, gate_passed=gate,
                question_type=task, local_facts={"world_frame": True},
                evidence_profile=EvidenceProfile(
                    geometry_3d="available" if geometry else "unavailable",
                    metric_scale="available" if metric else "unavailable"))
            options = (["A. left", "B. right"] if task == "object_rel_direction"
                       else ["A. first", "B. second"] if unit == "option" else [])
            ep = SimpleNamespace(
                question="Synthetic spatial question preserving the specified object roles and scope.",
                question_type=task, options=options)
            scene = SimpleNamespace(
                summary="Synthetic scene.", question_type=task, question_tool_scope=scope,
                available_artifacts=KNOWN_ARTIFACTS, grounding_coverage={},
                evidence_profile=ctx["evidence_profile"], local_facts=ctx["local_facts"],
                metric_evidence_gate_result=SimpleNamespace(
                    gate_passed=gate, missing_subconditions=[]))
            skill = parents[(task, state)]
            prompt, plan = runner._build_prompt_ex(ep, scene, scene, [skill], None)
            final = build_finalization_prompt(
                question=ep.question, options=options, question_type=task,
                observations="No historical observations.", remaining_requests=1,
                method_context_max_chars=8000) + finalization_instructions(task)
            docs = compact_tool_docs(REGISTRY, **ctx)
            assert skill.skill_md in prompt and plan.entries[0].content_sha256 == skill.content_sha256
            rows.append({
                "task": task, "state": state, "prompt_sha256": sha(prompt),
                "prompt_chars": len(prompt), "final_sha256": sha(final), "final_chars": len(final),
                "docs_sha256": sha(docs), "skill_sha256": skill.content_sha256,
            })
    return rows


def final_text(public_input):
    from skill3d.online.recovery import ValidatedObservation
    from skill3d.online import runner
    from skill3d.online.vsi_execution_contract import finalization_instructions
    from skill3d.synthesis.prompt_builder import build_finalization_prompt

    observations = [ValidatedObservation(**row) for row in public_input["observations"]]
    feedback = runner.observation_feedback(
        observations, question_type="object_counting", include_reader=False)
    if hasattr(runner, "finalization_pages_text"):
        feedback += runner.finalization_pages_text(
            public_input["pages"], question_type="object_counting")
    else:
        feedback += "\nRecent requested valid pages (at most 3):\n" + json.dumps(
            public_input["pages"], ensure_ascii=False, sort_keys=True)
    return build_finalization_prompt(
        question=public_input["question"], options=[], question_type="object_counting",
        observations=feedback, remaining_requests=1, method_context_max_chars=8000,
    ) + finalization_instructions("object_counting") + public_input["image_manifest"]


def project_initial(outcome, parents):
    from skill3d.online.vsi_execution_contract import program_instructions
    from skill3d.skills.delivery import plan_delivery
    from skill3d.synthesis.prompt_builder import PromptBuilder
    from skill3d.synthesis.prompt_compaction import compact_rendered_tool_docs
    from skill3d.synthesis.counting_context import compact_counting_scene_summary

    original = outcome["m8_prompt"]
    question = original.split("## Question\n", 1)[1].split("\nQuestion type:", 1)[0]
    evidence = original.split("## Current evidence\n", 1)[1].split(
        "\nThe public profile", 1)[0]
    route, scene = evidence.split("\nscene=", 1)
    old_docs = original.split("## Authorized tools\n", 1)[1].split(
        "\nSignatures, dependencies", 1)[0]
    docs = compact_rendered_tool_docs(old_docs, question_type="object_counting")
    state = "g1m1"
    skill = parents[("object_counting", state)]
    prompt = PromptBuilder().render(
        question=question, scene_summary=compact_counting_scene_summary("scene=" + scene),
        scene_frame="world", route_header=route, tool_docs=docs,
        question_type="object_counting", skill_plan=plan_delivery([skill]))
    authorized = set(re.findall(r"(?m)^- (\w+)\(", old_docs))
    prompt += program_instructions("object_counting", authorized_tools=authorized)
    prompt += original[original.index(IMAGE_HEADER):]
    assert skill.skill_md in prompt
    assert original[original.index(IMAGE_HEADER):] in prompt
    assert authorized == set(re.findall(r"(?m)^- (\w+)\(", docs))
    return prompt, question


def public_final_input(outcome, question):
    rounds = outcome["rounds"][:-1]
    invalid = set(outcome.get("invalidated_result_ids", []))
    observations, pages = [], []
    for row in rounds:
        for result in row.get("results", []):
            if result["status"] != "ok" or result.get("error") is not None:
                continue
            if result.get("invalidated_by") or result["result_id"] in invalid:
                continue
            payload = result.get("payload")
            if payload is None:
                payload = result.get("value")
                if isinstance(payload, str):
                    payload = json.loads(payload)
            observations.append({
                "result_id": result["result_id"],
                "source_tool": result.get("source_tool") or result["tool"],
                "payload": payload, "args": result.get("args", {}),
                "evidence_version": result.get("evidence_version", ""),
                "degraded_evidence": result.get("degraded_evidence", []),
            })
        pages.extend(row.get("observation_reads", []))
    valid = {o["result_id"] for o in observations}
    return {
        "question": question, "observations": observations,
        "pages": [page for page in pages if page["result_id"] in valid][-3:],
        "image_manifest": outcome["m8_prompt"][outcome["m8_prompt"].index(IMAGE_HEADER):],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-run", type=Path,
                        default=ROOT / "runs/compact-relcount20-c32-20261110-r06")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs/counting-context-offline-20261110")
    parser.add_argument("--worker-source", type=Path)
    args = parser.parse_args()
    if args.worker_source:
        rows = synthetic_contexts(args.worker_source.resolve())
        request = json.loads(sys.stdin.read() or "{}")
        result = {"synthetic": rows}
        if request:
            result["final_text"] = final_text(request)
        print(json.dumps(result))
        return
    current = synthetic_contexts(ROOT)
    from skill3d.evolution.fitset_revision_v11 import load_active_parent_slots
    from skill3d.synthesis.counting_context import COUNTING_CONTEXT_VERSION
    parents, identity = load_active_parent_slots(ROOT / "skill_library")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    projections = []
    final_input = None
    for path in sorted((args.frozen_run / "system/variants").glob("*/outcome.json")):
        outcome = json.loads(path.read_text())
        if outcome["task"] != "object_counting":
            continue
        prompt, question = project_initial(outcome, parents)
        qa = str(outcome["qa_id"])
        target = output / f"QA{qa}-initial-specialized.txt"
        target.write_text(prompt)
        before = outcome["m8_prompt"]
        projections.append({
            "qa_id": qa, "source": str(path.resolve()), "source_sha256": sha(before),
            "before_chars": len(before), "after_chars": len(prompt),
            "reduction_pct": round(100 * (1 - len(prompt) / len(before)), 2),
            "prompt": str(target), "prompt_sha256": sha(prompt),
            "measurement": "Public initial-prompt projection; image manifest preserved verbatim.",
        })
        if qa == "11":
            final_input = public_final_input(outcome, question)
    assert len(projections) == 10 and final_input is not None
    baseline = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker-source",
         str((args.frozen_run / "code/source").resolve())],
        input=json.dumps(final_input), text=True, capture_output=True, check=True)
    old = json.loads(baseline.stdout)
    matches = []
    for before, after in zip(old["synthetic"], current):
        assert (before["task"], before["state"]) == (after["task"], after["state"])
        if after["task"] != "object_counting":
            assert before == after, (before, after)
            matches.append({"task": after["task"], "state": after["state"],
                            "ordinary_final_docs_skill_exact_match": True})
    assert len(matches) == 21
    old_final, new_final = old["final_text"], final_text(final_input)
    archive = json.loads((args.frozen_run / "system/request_seeds/"
                          "703d6f219dc92c504525-006.json").read_text())
    archived_sha = archive["request_identity"]["text_parts"][0]["text_sha256"]
    # Do not report a reconstruction as the actual wire prompt without a hash match.
    verified = sha(old_final) == archived_sha
    (output / "QA11-final-before-reconstructed.txt").write_text(old_final)
    (output / "QA11-final-specialized.txt").write_text(new_final)
    ordinary = [row for row in projections if row["qa_id"] != "11"]
    before_mean = sum(r["before_chars"] for r in ordinary) / len(ordinary)
    after_mean = sum(r["after_chars"] for r in ordinary) / len(ordinary)
    report = {
        "context_version": COUNTING_CONTEXT_VERSION, "model_requests": 0,
        "active_skill_identity": identity,
        "ordinary_nine": {"before_mean_chars": round(before_mean, 2),
                          "after_mean_chars": round(after_mean, 2),
                          "reduction_pct": round(100 * (1 - after_mean / before_mean), 2)},
        "initial_projections": projections, "other_task_state_checks": matches,
        "synthetic_counting": [
            {"before": a, "after": b} for a, b in zip(old["synthetic"], current)
            if b["task"] == "object_counting"],
        "qa11_final": {
            "before_chars": len(old_final), "after_chars": len(new_final),
            "reduction_pct": round(100 * (1 - len(new_final) / len(old_final)), 2),
            "archived_prompt_sha256": archived_sha,
            "reconstructed_sha256": sha(old_final), "archived_hash_match": verified,
            "measurement": "Same valid observation/page sources and image manifest; old template "
                           "reconstruction, verified against the archived request text hash.",
        },
    }
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "model_requests", "ordinary_nine", "qa11_final")}, ensure_ascii=False, indent=2))
    if not verified:
        raise RuntimeError("QA11 final reconstruction differs from archived text; investigate before reporting")


if __name__ == "__main__":
    main()
