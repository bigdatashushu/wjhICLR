"""Freeze immutable completed records, then score only that host-only snapshot."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from .score import references_for
from .traces import digest, file_digest, read, write


def verify_episode(episode, manifest, pair, arm):
    expected = {"run_identity": manifest["run_identity"], "pair_identity": pair["pair_identity"],
                "sample_id": pair["sample_id"], "dataset": pair["dataset"], "arm": arm,
                "reference_answer_visible": False}
    if any(episode.get(key) != value for key, value in expected.items()):
        raise ValueError("partial_prediction_identity_changed")


def freeze(source, target):
    source, target = Path(source), Path(target)
    manifest = read(source / "manifest.json")
    if digest({k: v for k, v in manifest.items() if k != "run_identity"}) != manifest["run_identity"]:
        raise ValueError("partial_manifest_changed")
    paths = set((source / "episodes").rglob("episode.json"))
    target.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source / "manifest.json", target / "manifest.json")
    index = []
    for pair in manifest["pairs"]:
        endpoints = []
        for spec in pair["arms"]:
            relative = Path("episodes") / pair["dataset"] / pair["pair_identity"] / spec["arm"]
            root = source / relative
            path = root / "episode.json"
            if path not in paths:
                continue
            episode = read(path)
            verify_episode(episode, manifest, pair, spec["arm"])
            job = {"spec": spec, "sample_id": pair["sample_id"], "api_id": episode["api_id"],
                   "config": manifest["config"], "run_identity": manifest["run_identity"],
                   "pair_identity": pair["pair_identity"], "pool_root": str(source / "pool")}
            job["operation_id"] = digest({"pair": pair["pair_identity"], "arm": spec["arm"],
                                          "api": episode["api_id"], "run": manifest["run_identity"]})
            dispatch = read(root / "dispatch.json")
            if (episode["job_sha256"] != digest(job) or episode["operation_id"] != job["operation_id"]
                    or dispatch["job_sha256"] != digest(job)
                    or dispatch["operation_id"] != job["operation_id"]
                    or dispatch["api_id"] != episode["api_id"]):
                raise ValueError("partial_dispatch_identity_changed")
            endpoints.append(episode["api_id"])
            actual_requests = {str(p.relative_to(root)) for p in (root / "requests").glob("*.json")}
            if actual_requests != set(episode["request_files"]):
                raise ValueError("partial_request_inventory_changed")
            for name, expected in episode["request_files"].items():
                request_path = root / name
                if not request_path.resolve().is_relative_to(root.resolve()) or file_digest(request_path) != expected:
                    raise ValueError("partial_request_bytes_changed")
            copied = {}
            for public_file in sorted(root.rglob("*.json")):
                rel = public_file.relative_to(source)
                expected = file_digest(public_file)
                output = target / rel
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(public_file, output)
                if file_digest(output) != expected:
                    raise ValueError("partial_copy_changed")
                copied[str(rel)] = expected
            index.append({"dataset": pair["dataset"], "sample_id": pair["sample_id"],
                          "pair_identity": pair["pair_identity"], "arm": spec["arm"],
                          "episode": str(relative / "episode.json"), "files": copied})
        if len(set(endpoints)) > 1:
            raise ValueError("partial_endpoint_affinity_changed")
    if len(index) != len(paths):
        raise ValueError("partial_unknown_episode")
    with (source / "pool/state.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        ledger = read(source / "pool/state.json")
    if ledger["policy_identity"] != digest(manifest["config"]):
        raise ValueError("partial_pool_policy_changed")
    write(target / "request_ledger_snapshot.json", ledger, exclusive=True)
    requests = {row["request_id"]: row for row in ledger["requests"]}
    for row in index:
        episode = read(target / row["episode"])
        root = (target / row["episode"]).parent
        for admission_path in sorted((root / "requests").glob("*.admission.json")):
            prefix = str(admission_path).removesuffix(".admission.json")
            admission, intent = read(admission_path), read(prefix + ".intent.json")
            request = requests[admission["request_id"]]
            if (request["api"] != episode["api_id"] or request["operation"] != episode["operation_id"]
                    or request["ordinal"] != intent["ordinal"]
                    or request["payload_sha256"] != intent["payload_sha256"]
                    or digest(intent["native_payload"]) != intent["payload_sha256"]
                    or request["status"] not in ("completed", "error")):
                raise ValueError("partial_request_ledger_binding_changed")
    receipt = {"schema": "gptexp-partial-freeze/1", "source": str(source), "at": time.time(),
               "run_identity": manifest["run_identity"], "expected_arms": manifest["arms"],
               "completed_arms": len(index), "index": index,
               "manifest_sha256": file_digest(target / "manifest.json"),
               "request_ledger_sha256": file_digest(target / "request_ledger_snapshot.json"),
               "reference_answers_read": False, "scope": "available finalized records only"}
    receipt["snapshot_identity"] = digest(receipt)
    write(target / "snapshot.json", receipt, exclusive=True)
    return {"completed_arms": len(index), "snapshot_identity": receipt["snapshot_identity"]}


def summarize(rows):
    legal = [row for row in rows if row["legal"]]
    correct = sum(row["correct"] for row in rows)
    return {"records": len(rows), "legal": len(legal), "correct": correct,
            "accuracy": correct / len(rows) if rows else None,
            "legal_accuracy": correct / len(legal) if legal else None,
            "status": dict(Counter(row["status"] for row in rows))}


def score(target):
    target = Path(target)
    if any(any(term in key.upper() for term in ("API_KEY", "TOKEN", "ENDPOINT", "BASE_URL", "MODELS_KEY"))
           for key in os.environ):
        raise ValueError("host_scoring_requires_credential_stripped_environment")
    snapshot, manifest = read(target / "snapshot.json"), read(target / "manifest.json")
    if (digest({k: v for k, v in snapshot.items() if k != "snapshot_identity"}) != snapshot["snapshot_identity"]
            or file_digest(target / "manifest.json") != snapshot["manifest_sha256"]
            or file_digest(target / "request_ledger_snapshot.json") != snapshot["request_ledger_sha256"]):
        raise ValueError("partial_snapshot_changed")
    observed = set()
    for entry in snapshot["index"]:
        for name, expected in entry["files"].items():
            if file_digest(target / name) != expected:
                raise ValueError("partial_frozen_record_changed")
        observed.add(entry["pair_identity"])
    pairs = [p for p in manifest["pairs"] if p["pair_identity"] in observed]
    # No host reference is opened until all snapshot bytes above have verified.
    mapping = {(r["dataset"], r["sample_id"]): r["reference_answer"]
               for r in references_for(manifest, pairs)}
    pair_map = {p["pair_identity"]: p for p in pairs}
    rows = []
    full_path = target / "scored_full_trajectories.jsonl"
    with full_path.open("x", encoding="utf-8") as stream:
        for entry in snapshot["index"]:
            episode = read(target / entry["episode"])
            verify_episode(episode, manifest, pair_map[entry["pair_identity"]], entry["arm"])
            result = episode["result"] or {}
            answer = result.get("answer")
            legal = not episode["failure"] and result.get("status") == "answered" and answer in ("A", "B", "C", "D")
            reference = mapping[(episode["dataset"], episode["sample_id"])]
            row = {key: episode[key] for key in ("dataset", "sample_id", "pair_identity", "arm", "api_id")}
            row.update(answer=answer, reference_answer=reference,
                       status=result.get("status") or "worker_failure", legal=bool(legal),
                       correct=bool(legal and answer == reference))
            rows.append(row)
            root = (target / entry["episode"]).parent
            trajectory = {str(p.relative_to(root)): read(p) for p in root.rglob("*.json")
                          if p.name != "episode.json"}
            stream.write(json.dumps({"score": row, "episode": episode, "full_trajectory": trajectory},
                                    ensure_ascii=False, allow_nan=False) + "\n")
    grouped = {(r["pair_identity"], r["arm"]): r for r in rows}
    common = [p for p in pairs if all((p["pair_identity"], a) in grouped for a in ("direct", "harness"))]
    both_legal = [p for p in common if all(grouped[(p["pair_identity"], a)]["legal"] for a in ("direct", "harness"))]
    statistics = []
    for dataset in manifest["counts"]:
        item = {"dataset": dataset, "fixed_questions": manifest["counts"][dataset], "available": {}}
        for arm in ("direct", "harness"):
            item["available"][arm] = summarize([r for r in rows if r["dataset"] == dataset and r["arm"] == arm])
        for label, selected in (("common_pairs", common), ("both_legal_pairs", both_legal)):
            chosen = [p for p in selected if p["dataset"] == dataset]
            result = {"pairs": len(chosen)}
            for arm in ("direct", "harness"):
                result[arm] = summarize([grouped[(p["pair_identity"], arm)] for p in chosen])
            result["harness_only_correct"] = sum(not grouped[(p["pair_identity"], "direct")]["correct"]
                                                 and grouped[(p["pair_identity"], "harness")]["correct"] for p in chosen)
            result["direct_only_correct"] = sum(grouped[(p["pair_identity"], "direct")]["correct"]
                                                and not grouped[(p["pair_identity"], "harness")]["correct"] for p in chosen)
            item[label] = result
        statistics.append(item)
    output = {"schema": "gptexp-partial-scores/1", "snapshot_identity": snapshot["snapshot_identity"],
              "run_identity": manifest["run_identity"], "at": time.time(),
              "completed_arms": len(rows), "common_pairs": len(common), "both_legal_pairs": len(both_legal),
              "full_dataset_accuracy": None, "scoring_rule": "failures count as incorrect; unobserved excluded",
              "reference_access": "host-only after immutable partial prediction freeze",
              "reference_answer_visible_during_solving": False, "statistics": statistics}
    write(target / "partial_scores.json", output, exclusive=True)
    write(target / "scored_records.json", {"snapshot_identity": snapshot["snapshot_identity"], "records": rows}, exclusive=True)
    lines = ["# GPT-6.1-sol 已有结果阶段评分\n\n",
             f"冻结完整题臂 {len(rows)}/{manifest['arms']}；两臂均完成 {len(common)} 对，均有合法回答 {len(both_legal)} 对。\n\n",
             "以下主表使用同一批两臂均完成的题目；接口失败、程序拒绝和非法回答计错。未完成题目不进入分母。\n\n",
             "| 数据集 | 共同题数 | Direct | Harness | Harness − Direct |\n",
             "| --- | ---: | ---: | ---: | ---: |\n"]
    for item in statistics:
        result = item["common_pairs"]
        n = result["pairs"]
        d, h = result["direct"]["correct"], result["harness"]["correct"]
        lines.append(f"| {item['dataset']} | {n} | {d}/{n} ({d/n:.2%}) | {h}/{n} ({h/n:.2%}) "
                     f"| {(h-d)/n:+.2%} |\n" if n else f"| {item['dataset']} | 0 | — | — | — |\n")
    lines += ["\n各臂所有已完成记录（分母可能不同，不能直接视为严格配对比较）：\n\n",
              "| 数据集 | Direct | Harness |\n", "| --- | ---: | ---: |\n"]
    for item in statistics:
        values = []
        for arm in ("direct", "harness"):
            r = item["available"][arm]
            values.append(f"{r['correct']}/{r['records']} ({r['accuracy']:.2%})" if r["records"] else "—")
        lines.append(f"| {item['dataset']} | {values[0]} | {values[1]} |\n")
    lines.append("\n阶段快照不能代表全量成绩；均合法子集和逐题记录见 JSON。评分未送给在线模型。\n")
    (target / "report.md").write_text("".join(lines), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "score"])
    parser.add_argument("--source")
    parser.add_argument("--snapshot", required=True)
    args = parser.parse_args()
    result = freeze(args.source, args.snapshot) if args.action == "freeze" else score(args.snapshot)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
