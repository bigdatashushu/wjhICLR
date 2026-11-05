"""Export complete public traces and score in a separate host-only process."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from .traces import digest, file_digest, read, write


def trajectory_rows(directory, manifest):
    for pair in manifest["pairs"]:
        for spec in pair["arms"]:
            root = directory / "episodes" / pair["dataset"] / pair["pair_identity"] / spec["arm"]
            path = root / "episode.json"
            complete = path.exists()
            if not complete and not root.exists():
                continue
            episode = read(path) if complete else {
                "schema": "gptexp-partial-episode/1", "dataset": pair["dataset"],
                "arm": spec["arm"], "sample_id": pair["sample_id"],
                "pair_identity": pair["pair_identity"], "run_identity": manifest["run_identity"],
                "reference_answer_visible": False, "automatic_retry": False,
                "result": None, "failure": {"error_type": "episode_not_finalized"},
                "request_files": {}}
            if (episode["run_identity"] != manifest["run_identity"]
                    or episode["pair_identity"] != pair["pair_identity"]
                    or episode["sample_id"] != pair["sample_id"]
                    or episode["arm"] != spec["arm"]
                    or episode["reference_answer_visible"] is not False):
                raise ValueError("episode_pair_identity_changed")
            for name, expected in episode["request_files"].items():
                if file_digest(root / name) != expected:
                    raise ValueError("request_trace_changed")
            # Include MindCube's original full tool/authorization traces too.
            records = {str(p.relative_to(root)): read(p) for p in sorted(root.rglob("*.json"))
                       if p.name != "episode.json"}
            row = {**episode, "episode_complete": complete,
                   "public_input": spec.get("problem", spec.get("item")),
                   "full_trajectory": records}
            yield row


def export(directory, *, references=None):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    if digest({k: v for k, v in manifest.items() if k != "run_identity"}) != manifest["run_identity"]:
        raise ValueError("manifest_identity_changed")
    episodes = []
    combined = directory / "results_and_full_trajectories.json"
    jsonl = directory / "experiment_trace.jsonl"
    combined_temp = combined.with_suffix(".json.tmp")
    jsonl_temp = jsonl.with_suffix(".jsonl.tmp")
    header = {"schema": "gptexp-full-trajectories/1", "run_identity": manifest["run_identity"],
              "fixed_questions": manifest["counts"], "expected_arms": manifest["arms"],
              "reference_answer_visible": False}
    with combined_temp.open("w", encoding="utf-8") as full, jsonl_temp.open("w", encoding="utf-8") as stream:
        full.write(json.dumps(header, ensure_ascii=False)[:-1] + ', "episodes": [')
        for row in trajectory_rows(directory, manifest):
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
            if episodes:
                full.write(",\n")
            full.write(encoded)
            stream.write(encoded + "\n")
            # Keep small summaries in memory; images and tool trace trees stream.
            traces = row.pop("full_trajectory")
            result = row["result"] or {}
            row["result"] = {"status": result.get("status") or "worker_failure",
                             "answer": result.get("answer")}
            row.pop("public_input", None)
            row["request_intents"] = sum(name.startswith("requests/") and name.endswith(".intent.json")
                                          for name in traces)
            row["request_admissions"] = sum(
                name.startswith("requests/") and name.endswith(".admission.json") for name in traces)
            row["received_responses"] = sum(
                name.startswith("requests/") and name.endswith(".response.json")
                and value.get("status") == "received" for name, value in traces.items())
            row["transport_outcome_unknown"] = sum(
                name.startswith("requests/") and name.endswith(".response.json")
                and value.get("reason_code") == "transport_outcome_unknown"
                for name, value in traces.items())
            usage = Counter()
            for name, value in traces.items():
                if name.startswith("requests/") and name.endswith(".response.json"):
                    for key, number in value.get("usage", {}).items():
                        if type(number) is int:
                            usage[key] += number
            row["request_usage"] = dict(usage)
            episodes.append(row)
        completed = sum(row["episode_complete"] for row in episodes)
        full.write('], "observed_arms": ' + str(completed)
                   + ', "partial_arms": ' + str(len(episodes) - completed) + "}\n")
        for output in (full, stream):
            output.flush()
            os.fsync(output.fileno())
    os.replace(combined_temp, combined)
    os.replace(jsonl_temp, jsonl)
    # Scoring starts only after all predictions have frozen, in this host process.
    if references:
        if (completed != manifest["arms"] or not (directory / "execution.json").exists()
                or read(directory / "execution.json")["status"] != "complete"):
            raise ValueError("scoring_requires_full_prediction_freeze")
        rows = read(references)
        if isinstance(rows, dict) and "references" in rows:
            rows = rows["references"]
        mapping = {(r["dataset"], r["sample_id"]): r["reference_answer"] for r in rows}
        expected = {(p["dataset"], p["sample_id"]) for p in manifest["pairs"]}
        if set(mapping) != expected:
            raise ValueError("reference_denominator_changed")
    else:
        mapping = {}
    scored_pairs = []
    paired_statistics = []
    if mapping:
        index = {(row["dataset"], row["sample_id"], row["arm"]): row for row in episodes}
        for pair in manifest["pairs"]:
            reference = mapping[(pair["dataset"], pair["sample_id"])]
            row = {key: pair[key] for key in ("dataset", "sample_id", "pair_identity")}
            row["reference_answer"] = reference
            for arm in ("direct", "harness"):
                episode = index[(pair["dataset"], pair["sample_id"], arm)]
                result = episode["result"]
                row[arm] = {
                    "answer": result["answer"], "status": result["status"],
                    "correct": result["status"] == "answered" and result["answer"] == reference,
                    "api_id": episode.get("api_id")}
            scored_pairs.append(row)
        write(directory / "scored_pairs.json", {
            "run_identity": manifest["run_identity"], "pairs": scored_pairs,
            "reference_access": "host-only after complete prediction freeze",
            "reference_answer_visible_during_solving": False})
        for dataset, count in manifest["counts"].items():
            outcomes = Counter((row["direct"]["correct"], row["harness"]["correct"])
                               for row in scored_pairs if row["dataset"] == dataset)
            paired_statistics.append({
                "dataset": dataset, "questions": count, "both_correct": outcomes[(True, True)],
                "both_incorrect": outcomes[(False, False)],
                "harness_only_correct": outcomes[(False, True)],
                "direct_only_correct": outcomes[(True, False)],
                "accuracy_delta": (outcomes[(False, True)] - outcomes[(True, False)]) / count})
    statistics = []
    for dataset, count in manifest["counts"].items():
        for arm in ("direct", "harness"):
            selected = [row for row in episodes if row["dataset"] == dataset and row["arm"] == arm]
            completed_arms = sum(row["episode_complete"] for row in selected)
            legal = correct = requests = intents = received = unknown = 0
            status = Counter()
            usage = Counter()
            for row in selected:
                result = row["result"] or {}
                status[result.get("status", "worker_failure")] += 1
                answer = result.get("answer")
                legal += bool(result.get("status") == "answered" and answer in ("A", "B", "C", "D"))
                if mapping:
                    correct += bool(result.get("status") == "answered"
                                    and answer == mapping[(dataset, row["sample_id"])])
                requests += row["request_admissions"]
                intents += row["request_intents"]
                received += row["received_responses"]
                unknown += row["transport_outcome_unknown"]
                usage.update(row["request_usage"])
            statistics.append({"dataset": dataset, "arm": arm, "questions": count,
                "completed_arms": completed_arms, "partial_arms": len(selected) - completed_arms,
                "legal": legal, "status": dict(status),
                "missing": count - completed_arms, "requests": requests,
                "request_intents": intents, "received_responses": received,
                "transport_outcome_unknown": unknown, "usage": dict(usage),
                "correct": correct if mapping else None,
                "accuracy": correct / count if mapping else None})
    write(directory / "comparison.json", {
        "run_identity": manifest["run_identity"], "fixed_denominator": True,
        "request_accounting": "requests count admitted attempts; queued intents are separate",
        "host_scored": bool(mapping), "statistics": statistics,
        "paired_statistics": paired_statistics})
    lines = ["# GPT-6.1-sol 配对评测\n",
             f"固定 {manifest['questions']:,} 题；已保存 {completed}/{manifest['arms']:,} 个完整题臂，"
             f"{len(episodes) - completed} 个未完成题臂的部分轨迹。\n",
             "两臂使用原冻结题单、提示词、工具及预算；GPT reasoning effort 为 high。\n",
             "| 数据集 | 组别 | 完成/固定题量 | 合法回答 | 正确数 | 准确率 |\n",
             "| --- | --- | ---: | ---: | ---: | ---: |\n"]
    for row in statistics:
        accuracy = f"{row['accuracy']:.2%}" if mapping else "待宿主评分"
        lines.append(f"| {row['dataset']} | {row['arm']} | {row['completed_arms']}/{row['questions']} "
                     f"| {row['legal']} | {row['correct'] if mapping else '—'} | {accuracy} |\n")
    (directory / "report.md").write_text("".join(lines), encoding="utf-8")
    return {"episodes": completed, "partial_arms": len(episodes) - completed,
            "host_scored": bool(mapping)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--references", help="Host-only JSON; never passed to solver workers")
    args = parser.parse_args()
    print(json.dumps(export(args.run, references=args.references)))


if __name__ == "__main__":
    main()
