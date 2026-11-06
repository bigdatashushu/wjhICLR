"""User-requested per-type subset with immutable reuse of every finalized arm."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from .adapters import ROOT
from .api_pool import ApiPool, resolve_key
from .model_binding import binding_record
from .response_policy import policy
from .partial_score import freeze, verify_episode
from .relaunch import reuse_validation
from .run import PAIRED_REQUESTS_PER_QUESTION, code_identity, environment, verify_manifest
from .traces import digest, file_digest, read, write


def identity_check(manifest):
    if digest({k: v for k, v in manifest.items() if k != "run_identity"}) != manifest["run_identity"]:
        raise ValueError("manifest_changed")


def stratum(pair):
    public = pair["arms"][0].get("problem", pair["arms"][0].get("item"))
    return pair["dataset"], public["task_type"]


def arm_cap(dataset, arm):
    return 1 if arm == "direct" else PAIRED_REQUESTS_PER_QUESTION[dataset] - 1


def select_pairs(previous, finalized):
    """Selection sees public type/identity and completion presence, never outcomes."""
    known = {(p["pair_identity"], s["arm"]) for p in previous["pairs"] for s in p["arms"]}
    if not finalized <= known:
        raise ValueError("unknown_finalized_arm")
    groups = defaultdict(list)
    for pair in previous["pairs"]:
        if sorted(s["arm"] for s in pair["arms"]) != ["direct", "harness"]:
            raise ValueError("invalid_source_pair")
        groups[stratum(pair)].append(pair)
    selected, rows = set(), []
    mandatory = {p for p, _ in finalized}
    seed = previous["config"]["seed"]
    for (dataset, task_type), pairs in groups.items():
        quota = (3 * len(pairs) + 9) // 10
        existing = [p for p in pairs if p["pair_identity"] in mandatory]
        extra = sorted((p for p in pairs if p["pair_identity"] not in mandatory),
                       key=lambda p: digest({"seed": seed, "dataset": dataset,
                                             "task_type": task_type, "pair": p["pair_identity"]}))
        chosen = existing + extra[:max(0, quota - len(existing))]
        selected.update(p["pair_identity"] for p in chosen)
        rows.append({"dataset": dataset, "task_type": task_type, "original_questions": len(pairs),
                     "minimum_questions": quota, "selected_questions": len(chosen),
                     "reused_question_pairs": len(existing),
                     "reused_arms": sum((p["pair_identity"], s["arm"]) in finalized
                                        for p in existing for s in p["arms"])})
    return [p for p in previous["pairs"] if p["pair_identity"] in selected], rows


def safe_file(root, name):
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("record_path_escape")
    return path


def verified_record(root, manifest, pair, spec, origin, ledger):
    episode = read(root / "episode.json")
    verify_episode(episode, manifest, pair, spec["arm"])
    job = {"spec": spec, "sample_id": pair["sample_id"], "api_id": episode["api_id"],
           "config": manifest["config"], "run_identity": manifest["run_identity"],
           "pair_identity": pair["pair_identity"], "pool_root": str(origin / "pool")}
    job["operation_id"] = digest({"pair": pair["pair_identity"], "arm": spec["arm"],
                                  "api": episode["api_id"], "run": manifest["run_identity"]})
    dispatch = read(root / "dispatch.json")
    if (episode["job_sha256"] != digest(job) or episode["operation_id"] != job["operation_id"]
            or dispatch["job_sha256"] != digest(job) or dispatch["operation_id"] != job["operation_id"]
            or dispatch["api_id"] != episode["api_id"]):
        raise ValueError("dispatch_identity_changed")
    requests = {str(p.relative_to(root)): file_digest(p) for p in (root / "requests").glob("*.json")}
    if requests != episode["request_files"]:
        raise ValueError("request_inventory_or_bytes_changed")
    binding = read(root / "model_binding.json")
    _, expected_binding = binding_record(spec, manifest["config"]["model"], manifest["run_identity"],
                                        policy(manifest["config"]))
    expected_binding.pop("schema")
    if any(binding.get(k) != v for k, v in expected_binding.items()):
        raise ValueError("runtime_model_binding_changed")
    if ledger["policy_identity"] != digest(manifest["config"]):
        raise ValueError("request_ledger_policy_changed")
    admissions = sorted((root / "requests").glob("*.admission.json"))
    if len(admissions) > arm_cap(pair["dataset"], spec["arm"]):
        raise ValueError("arm_request_cap_changed")
    for path in admissions:
        prefix = str(path).removesuffix(".admission.json")
        admission, intent = read(path), read(prefix + ".intent.json")
        request = ledger["requests"][admission["request_id"]]
        payload = intent["native_payload"]
        if (request["api"] != episode["api_id"] or request["operation"] != episode["operation_id"]
                or request["request_id"] != admission["request_id"]
                or request["ordinal"] != intent["ordinal"]
                or request["payload_sha256"] != intent["payload_sha256"]
                or digest(payload) != intent["payload_sha256"]
                or payload["model"] != manifest["config"]["model"]
                or payload["reasoning"] != {"effort": "high"}
                or request["status"] not in ("completed", "error")):
            raise ValueError("request_ledger_binding_changed")
    return episode


def verify_reuse(directory, manifest):
    if manifest.get("schema") == "gptexp-type30-paired-reuse/2":
        from .continuation import verify_continuation
        return verify_continuation(directory, manifest)
    source = directory / "reuse_source"
    snapshot = read(source / "snapshot.json")
    if (digest({k: v for k, v in snapshot.items() if k != "snapshot_identity"})
            != snapshot["snapshot_identity"] or snapshot["snapshot_identity"] != manifest["reuse_snapshot_identity"]
            or file_digest(source / "manifest.json") != snapshot["manifest_sha256"]
            or file_digest(source / "request_ledger_snapshot.json") != snapshot["request_ledger_sha256"]):
        raise ValueError("reuse_snapshot_changed")
    previous = read(source / "manifest.json")
    identity_check(previous)
    if previous["run_identity"] != manifest["source_run_identity"]:
        raise ValueError("reuse_origin_changed")
    selected, strata = select_pairs(previous, {
        (entry["pair_identity"], entry["arm"]) for entry in snapshot["index"]})
    if selected != manifest["pairs"] or strata != manifest["strata"]:
        raise ValueError("subset_selection_changed")
    if len(snapshot["index"]) != manifest["reused_arms"]:
        raise ValueError("reuse_denominator_changed")
    ledger = read(source / "request_ledger_snapshot.json")
    pairs = {p["pair_identity"]: p for p in manifest["pairs"]}
    index, endpoints = {}, defaultdict(set)
    for entry in snapshot["index"]:
        key = entry["pair_identity"], entry["arm"]
        if key in index:
            raise ValueError("duplicate_reuse_arm")
        for name, expected in entry["files"].items():
            if file_digest(safe_file(source, name)) != expected:
                raise ValueError("reused_trajectory_changed")
        pair = pairs[entry["pair_identity"]]
        spec = next(s for s in pair["arms"] if s["arm"] == entry["arm"])
        episode = verified_record(safe_file(source, entry["episode"]).parent,
                                  previous, pair, spec, Path(snapshot["source"]), ledger)
        endpoints[pair["pair_identity"]].add(episode["api_id"])
        index[key] = entry
    if any(len(ids) != 1 for ids in endpoints.values()):
        raise ValueError("reused_pair_api_affinity_changed")
    pins = {pair: next(iter(ids)) for pair, ids in endpoints.items()}
    if pins != manifest["reused_api_pins"]:
        raise ValueError("reused_api_pins_changed")
    reduced = deepcopy(previous["config"])
    reduced["max_requests"] = manifest["max_requests"]
    if reduced != manifest["config"]:
        raise ValueError("non_budget_configuration_changed")
    expected_budget = sum(arm_cap(p["dataset"], s["arm"]) for p in manifest["pairs"]
                          for s in p["arms"] if (p["pair_identity"], s["arm"]) not in index)
    if expected_budget != manifest["max_requests"]:
        raise ValueError("missing_arm_budget_changed")
    for name, expected in manifest["validation_receipt_sha256"].items():
        if file_digest(safe_file(directory, name)) != expected:
            raise ValueError("api_validation_receipt_changed")
    return index


def selection_report(manifest):
    lines = [
        "# 按题型取至少 30%，全部复用已有结果", "",
        f"固定 {manifest['questions']} 题、{manifest['arms']} 个题臂；复用 {manifest['reused_arms']} 个已完成题臂，"
        f"仅补跑 {manifest['missing_arms']} 个题臂。",
        "每型配额为原题量 × 30% 向上取整；所有已有结果优先纳入，补充题目按种子 137 的哈希顺序选择。",
        "失败和无效答案同样复用。旧 episode、请求、程序和工具回执保持原字节与原运行身份，外层记录复用来源。",
        "模型 gpt-5.6-terra/high、Prompt、Harness、每题预算及 API 参数保持原配置。",
        "选择过程只使用公开题型、题目身份和完成状态；不使用得分或参考答案。", "",
        "| 数据集 | 题型 | 原题量 | 30% 配额 | 选入题量 | 复用题臂 |",
        "|---|---|---:|---:|---:|---:|"]
    for row in manifest["strata"]:
        lines.append(f"| {row['dataset']} | {row['task_type']} | {row['original_questions']} | "
                     f"{row['minimum_questions']} | {row['selected_questions']} | {row['reused_arms']} |")
    lines.extend(["", f"新请求总上限：{manifest['max_requests']}；旧请求成本单独保留在复用账本。",
                  "原全量任务已停止；其未完成请求保留在旧目录，不计为已完成答案。"])
    return "\n".join(lines) + "\n"


def prepare(directory, source):
    directory, source = Path(directory).absolute(), Path(source).absolute()
    directory.mkdir(exist_ok=True)
    if (directory / "manifest.json").exists():
        raise ValueError("subset_already_prepared")
    previous = read(source / "manifest.json")
    identity_check(previous)
    if not (source / "stopped_by_user.json").exists():
        raise ValueError("source_not_stopped")
    stop = read(source / "stopped_by_user.json")
    actual = {str(p.relative_to(source)): file_digest(p)
              for p in (source / "episodes").rglob("episode.json")}
    if actual != stop["completed_episode_sha256"]:
        raise ValueError("source_changed_after_user_stop")
    with (source / "operator.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze(source, directory / "reuse_source")
    snapshot = read(directory / "reuse_source/snapshot.json")
    finalized = {(r["pair_identity"], r["arm"]) for r in snapshot["index"]}
    if len(finalized) != len(snapshot["index"]):
        raise ValueError("duplicate_source_finalized_arm")
    pairs, strata = select_pairs(previous, finalized)
    budget = sum(arm_cap(p["dataset"], s["arm"]) for p in pairs for s in p["arms"]
                 if (p["pair_identity"], s["arm"]) not in finalized)
    config = deepcopy(previous["config"])
    config["max_requests"] = budget
    final, validation_previous = reuse_validation(directory, source, config)
    if final != config or validation_previous != previous:
        raise ValueError("subset_api_configuration_changed")
    pins = {}
    for entry in snapshot["index"]:
        api = read(directory / "reuse_source" / entry["episode"])["api_id"]
        old = pins.setdefault(entry["pair_identity"], api)
        if old != api:
            raise ValueError("source_pair_api_affinity_changed")
    counts = dict(Counter(p["dataset"] for p in pairs))
    manifest = {
        "schema": "gptexp-type30-paired-reuse/1", "questions": len(pairs), "arms": 2 * len(pairs),
        "counts": counts, "config": final, "pairs": pairs, "strata": strata,
        "max_requests": budget, "reused_arms": len(finalized), "missing_arms": 2 * len(pairs) - len(finalized),
        "source_run_identity": previous["run_identity"], "source_directory": str(source),
        "reuse_snapshot_identity": snapshot["snapshot_identity"], "reused_api_pins": pins,
        "source_receipts": previous["source_receipts"], "preflight": previous["preflight"],
        "adapter_source": code_identity(),
        "validation_receipt_sha256": read(directory / "api_validation.json")["receipt_sha256"],
        "reference_answer_visible": False, "automatic_retry": False, "skill_evolution": False,
        "selection_policy": "all finalized outcomes first; per-type ceil(30%); seeded hash fill",
        "allowed_changes": ["user-requested per-type subset and new global budget",
                            "immutable prior-arm reuse with original identity",
                            "missing-arm scheduling", "counts-aware passive monitor and export"],
        "prepared_at": time.time()}
    manifest["run_identity"] = digest(manifest)
    write(directory / "manifest.json", manifest, exclusive=True)
    write(directory / "launch.config.json", config, exclusive=True)
    write(directory / "run.config.json", config, exclusive=True)
    verify_manifest(manifest)
    verify_reuse(directory, manifest)
    with tarfile.open(directory / "public_source_snapshot.tar.gz", "w:gz") as archive:
        for name in code_identity():
            archive.add(ROOT / name, arcname=name)
    receipt = {
        "status": "ready", "run_identity": manifest["run_identity"], "questions": len(pairs),
        "arms": manifest["arms"], "reused_arms": len(finalized), "missing_arms": manifest["missing_arms"],
        "max_requests": budget, "validated_apis": len(config["apis"]), "model_requests": 0,
        "new_api_probes": 0, "all_finalized_outcomes_reused": True,
        "reference_answers_read": False, "original_episode_bytes_preserved": True,
        "partial_pair_api_affinity_preserved": True,
        "only_global_budget_changed_in_config": True,
        "public_source_snapshot_sha256": file_digest(directory / "public_source_snapshot.tar.gz"),
        "at": time.time()}
    write(directory / "readiness_audit.json", receipt, exclusive=True)
    write(directory / "preparation.json", receipt, exclusive=True)
    (directory / "selection_report.md").write_text(selection_report(manifest), encoding="utf-8")
    return receipt


def missing_pairs(directory, manifest, reused):
    """Never dispatch a finalized or previously dispatched fresh arm again."""
    assignments = read(directory / "assignments.json") if (directory / "assignments.json").exists() else {}
    ledger_path = directory / "pool/state.json"
    ledger = read(ledger_path) if ledger_path.exists() else {
        "policy_identity": digest(manifest["config"]), "requests": []}
    pending, unresolved = [], []
    for pair in manifest["pairs"]:
        specs, endpoints = [], set()
        pinned = manifest["reused_api_pins"].get(pair["pair_identity"])
        if pinned:
            endpoints.add(pinned)
        for spec in pair["arms"]:
            key = pair["pair_identity"], spec["arm"]
            root = directory / "episodes" / pair["dataset"] / pair["pair_identity"] / spec["arm"]
            if key in reused:
                if root.exists():
                    raise ValueError("reused_arm_was_redispatched")
                continue
            if (root / "episode.json").exists():
                episode = verified_record(root, manifest, pair, spec, directory, ledger)
                endpoints.add(episode["api_id"])
            elif root.exists() and any(root.iterdir()):
                unresolved.append({"pair_identity": pair["pair_identity"], "arm": spec["arm"]})
            else:
                specs.append(spec)
        if len(endpoints) > 1:
            raise ValueError("fresh_pair_api_affinity_changed")
        prior = assignments.get(pair["pair_identity"])
        if endpoints and prior and prior not in endpoints:
            raise ValueError("assignment_api_affinity_changed")
        if specs:
            pending.append((pair, specs, pinned or prior or next(iter(endpoints), None)))
    return pending, unresolved, assignments


def execute(directory):
    directory = Path(directory).absolute()
    manifest = read(directory / "manifest.json")
    verify_manifest(manifest)
    reused = verify_reuse(directory, manifest)
    config = manifest["config"]
    if config["worker_module"] != "GPTexp.terra_worker" or config["model"] != "gpt-5.6-terra":
        raise ValueError("subset_worker_binding_changed")
    apis = [a for a in config["apis"] if a["validated"] and a["enabled"]]
    keys = [resolve_key(a) for a in apis]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate_api_credentials")
    del keys
    with (directory / "operator.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "execution.json").exists():
            raise ValueError("subset_execution_already_finalized")
        pool = ApiPool(directory / "pool", config)
        pairs, unresolved, assignments = missing_pairs(directory, manifest, reused)
        write(directory / "startup.json", {
            "pid": os.getpid(), "at": time.time(), "run_identity": manifest["run_identity"],
            "reused_arms": len(reused), "scheduled_missing_arms": sum(len(s) for _, s, _ in pairs),
            "automatic_retry": False})
        write(directory / "bootstrap.json", {
            "stage": "executing", "at": time.time(), "run_identity": manifest["run_identity"],
            "validated_apis": len(apis), "reference_answers_read": False})

        def arm_job(pair, spec, api):
            if (pair["pair_identity"], spec["arm"]) in reused:
                raise ValueError("cannot_execute_reused_arm")
            job = {"spec": spec, "sample_id": pair["sample_id"], "api_id": api["id"],
                   "config": config, "run_identity": manifest["run_identity"],
                   "pair_identity": pair["pair_identity"], "pool_root": str(directory / "pool")}
            job["operation_id"] = digest({"pair": pair["pair_identity"], "arm": spec["arm"],
                                          "api": api["id"], "run": manifest["run_identity"]})
            output = directory / "episodes" / pair["dataset"] / pair["pair_identity"] / spec["arm"]
            output.mkdir(parents=True, exist_ok=True)
            write(output / "dispatch.json", {
                "job_sha256": digest(job), "api_id": api["id"], "operation_id": job["operation_id"],
                "at": time.time()}, exclusive=True)
            with (output / "worker.log").open("xb") as log:
                process = subprocess.Popen(
                    [sys.executable, "-B", "-m", config["worker_module"], "--dataset", pair["dataset"],
                     "--output", str(output)], cwd=ROOT.parent, env=environment(),
                    stdin=subprocess.PIPE, stdout=log, stderr=log)
                try:
                    process.communicate(json.dumps(job, ensure_ascii=False).encode(),
                                        timeout=config["worker_timeout_seconds"])
                except subprocess.TimeoutExpired:
                    process.terminate()
                    write(output / "process_timeout.json", {
                        "pid": process.pid, "request_outcome": "unknown", "automatic_retry": False},
                          exclusive=True)
                    pool.pause(api, "worker_timeout")
                    return
            if not (output / "episode.json").exists():
                pool.pause(api, "worker_failure")
                return
            episode = read(output / "episode.json")
            if (episode["job_sha256"] != digest(job) or episode["run_identity"] != manifest["run_identity"]):
                raise ValueError("fresh_worker_identity_changed")
            result = episode["result"] or {}
            if process.returncode or episode["failure"] or result.get("status") in {
                    "service_error", "host_error", "input_error", "precompute_error",
                    "prepared_identity_mismatch", "prepared_input_mismatch", "prepared_contract_error",
                    "request_exhausted", "budget_exhausted"}:
                errors = [read(p) for p in (output / "requests").glob("*.response.json")
                          if read(p).get("status") == "error"]
                if not (errors and not pool.paused(api) and all(
                        pool.transient(reason_code=r.get("reason_code"), http_status=r.get("http_status"))
                        for r in errors)):
                    pool.pause(api, "operational_failure")

        waiting, active, finished_jobs = list(pairs), {}, 0
        next_progress = 0
        with ThreadPoolExecutor(max_workers=config["max_workers"]) as executor:
            while waiting or active:
                loads = Counter(api for _, api in active.values())
                for row in list(waiting):
                    pair, specs, pin = row
                    if len(active) + len(specs) > config["max_workers"]:
                        continue
                    available = [a for a in apis if not pool.paused(a) and not pool.cooling(a)
                                 and (not pin or a["id"] == pin)]
                    if not available:
                        continue
                    api = min(available, key=lambda a: loads[a["id"]] / a["max_inflight"])
                    assignments[pair["pair_identity"]] = api["id"]
                    write(directory / "assignments.json", assignments)
                    waiting.remove(row)
                    for spec in specs:
                        active[executor.submit(arm_job, pair, spec, api)] = (pair, api["id"])
                        loads[api["id"]] += 1
                if not active:
                    possible = any(any(not pool.paused(a) and (not pin or a["id"] == pin) for a in apis)
                                   for _, _, pin in waiting)
                    if possible:
                        time.sleep(1)
                        continue
                    break
                done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    _, api_id = active.pop(future)
                    try:
                        future.result()
                    except BaseException:
                        pool.pause(next(a for a in apis if a["id"] == api_id), "scheduler_worker_failure")
                    finished_jobs += 1
                if time.monotonic() >= next_progress:
                    fresh = sum(1 for _ in (directory / "episodes").rglob("episode.json"))
                    write(directory / "progress.json", {
                        "at": time.time(), "reused_arms": len(reused), "new_finalized_arms": fresh,
                        "completed_arms": len(reused) + fresh, "expected_arms": manifest["arms"],
                        "new_finished_jobs": finished_jobs, "pending_arms": len(active),
                        "remaining_undispatched_arms": sum(len(s) for _, s, _ in waiting),
                        "unresolved_arms": unresolved, "accounting": pool.summary()})
                    next_progress = time.monotonic() + 30
        fresh = sum(1 for _ in (directory / "episodes").rglob("episode.json"))
        complete = fresh + len(reused) == manifest["arms"]
        status = "complete" if complete else "paused_with_retained_evidence"
        write(directory / "execution.json", {
            "status": status, "episode_count": fresh + len(reused), "fresh_arms": fresh,
            "reused_arms": len(reused), "expected_arms": manifest["arms"],
            "accounting": pool.summary(), "at": time.time(),
            "automatic_retry": False, "reference_answers_read": False})
        write(directory / "bootstrap.json", {"stage": status, "at": time.time()})
    env = environment()
    for name in tuple(env):
        if any(t in name.upper() for t in ("API_KEY", "TOKEN", "BASE_URL", "ENDPOINT", "MODELS_KEY")):
            env.pop(name)
    command = [sys.executable, "-B", "-m", "GPTexp.subset_export", "--run", str(directory)]
    if complete:
        command.append("--score")
    subprocess.run(command, cwd=ROOT.parent, env=env, check=True, timeout=864000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "execute"])
    parser.add_argument("--run", required=True)
    parser.add_argument("--source")
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.run, args.source)))
    else:
        execute(args.run)


if __name__ == "__main__":
    main()
