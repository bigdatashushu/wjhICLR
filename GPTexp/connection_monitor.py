"""Passive Linux socket and request-ledger monitoring; never sends API traffic."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import time

from .traces import read, write

TCP_STATES = {
    "01": "ESTABLISHED", "02": "SYN_SENT", "03": "SYN_RECV",
    "04": "FIN_WAIT1", "05": "FIN_WAIT2", "06": "TIME_WAIT", "07": "CLOSE",
    "08": "CLOSE_WAIT", "09": "LAST_ACK", "0A": "LISTEN", "0B": "CLOSING"}
REASONS = {
    "http_error", "transport_outcome_unknown", "invalid_response_json", "actual_model_changed",
    "reported_reasoning_effort_changed", "reported_sampling_changed", "reported_output_limit_changed",
    "reported_instructions_changed", "api_paused", "queue_timeout", "global_budget_exhausted",
    "service_error", "validation_failed", "APIError", "operational_failure", "worker_failure",
    "scheduler_worker_failure"}
STAGES = {
    "reusing_frozen_preflight", "validating_apis", "executing", "blocked_no_validated_api",
    "complete", "paused_with_retained_evidence", "reusing_verified_receipts"}


def safe_read(path):
    try:
        value = read(path)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def numeric(value, default=0):
    return value if type(value) in (int, float) and abs(value) < 10**15 else default


def reason(value):
    return value if value in REASONS else "other"


def process_identity(pid, proc=Path("/proc")):
    try:
        fields = (proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return {"state": fields[0], "ppid": int(fields[1]), "start_ticks": int(fields[19])}
    except (OSError, ValueError, IndexError):
        return {}


def sockets(pid, table, proc=Path("/proc")):
    counts = Counter()
    readable = True
    try:
        fds = list((proc / str(pid) / "fd").iterdir())
    except OSError:
        return counts, False
    inodes = set()
    for fd in fds:
        try:
            target = os.readlink(fd)
            if target.startswith("socket:[") and target.endswith("]"):
                inodes.add(target[8:-1])
        except OSError:
            readable = False
    for inode in inodes:
        if inode in table:
            counts[table[inode]] += 1
    return counts, readable


def socket_table(proc=Path("/proc")):
    result = {}
    readable = False
    for name in ("tcp", "tcp6"):
        try:
            lines = (proc / "net" / name).read_text().splitlines()[1:]
            readable = True
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) > 9 and fields[2].rsplit(":", 1)[-1] == "01BB":
                result[fields[9]] = TCP_STATES.get(fields[3], "UNKNOWN")
    return result, readable


def processes(run, owner_pid, start_ticks, proc=Path("/proc")):
    owner = process_identity(owner_pid, proc)
    alive = bool(owner and owner["state"] not in {"Z", "X"}
                 and owner["start_ticks"] == start_ticks)
    found = []
    if alive:
        found.append({"pid": owner_pid, "scope": "bootstrap_owner", "api_id": None})
    # Worker attribution uses dispatch receipts, never credentials or request bodies.
    for path in proc.iterdir():
        if not path.name.isdigit() or int(path.name) == owner_pid:
            continue
        try:
            args = (path / "cmdline").read_bytes().decode().split("\0")
            if (not any(m in args for m in ("GPTexp.worker", "GPTexp.terra_worker"))
                    or "--output" not in args):
                continue
            output = Path(args[args.index("--output") + 1])
            if not output.is_relative_to(run / "episodes"):
                continue
            identity = process_identity(int(path.name), proc)
            if not identity or identity["state"] in {"Z", "X"}:
                continue
            dispatch = safe_read(output / "dispatch.json")
            api = dispatch.get("api_id")
            if not isinstance(api, str) or not re.fullmatch(r"xcode-(?:10|[1-9])", api):
                api = None
            directory = output / "requests"
            # File names alone identify intentions awaiting admission; no prompt reads.
            queued = sum(not p.with_name(p.name.replace(".intent.", ".admission.")).exists()
                         and not p.with_name(p.name.replace(".intent.", ".response.")).exists()
                         for p in directory.glob("*.intent.json"))
            found.append({"pid": int(path.name), "scope": "episode_worker",
                          "api_id": api, "queued_requests": queued})
        except (OSError, ValueError, IndexError, UnicodeError):
            continue
    return alive, found


def collect(run, owner_pid, start_ticks, proc=Path("/proc"), now=None):
    now = time.time() if now is None else now
    bootstrap = safe_read(run / "bootstrap.json")
    config = safe_read(run / "launch.config.json")
    table, tcp_readable = socket_table(proc)
    alive, workers = processes(run, owner_pid, start_ticks, proc)
    tcp_total = Counter()
    tcp_by_api = {}
    for worker in workers:
        counts, readable = sockets(worker["pid"], table, proc)
        worker.update(https_tcp_states=dict(counts), socket_fds_readable=readable)
        tcp_total.update(counts)
        if worker["api_id"]:
            tcp_by_api.setdefault(worker["api_id"], Counter()).update(counts)
    api_rows = []
    ledger_events = []
    totals = {"probe": Counter(), "benchmark": Counter()}
    for api in config.get("apis", []):
        api_id = api.get("id")
        if not isinstance(api_id, str) or not re.fullmatch(r"xcode-(?:10|[1-9])", api_id):
            continue
        probe_root = run / "api-probes" / api_id
        probe = safe_read(probe_root / "summary.json")
        row = {
            "api_id": api_id, "validated": probe.get("validated") is True,
            "validation_status": ("passed" if probe.get("validated") is True else
                                  "failed" if probe else "waiting"),
            "https_tcp_states": dict(tcp_by_api.get(api_id, {})),
            "tcp_attribution": "episode_workers_only",
            "queued_requests": sum(w.get("queued_requests", 0) for w in workers if w["api_id"] == api_id),
            "active_requests": [], "accounting": {}, "recent_http_errors": [],
        }
        if probe:
            identity = probe.get("response_identity", {})
            row["returned_requested_model"] = identity.get("model") == config.get("model")
            row["image_order_verified"] = probe.get("image_order_verified") is True
            row["validation_elapsed_seconds"] = numeric(probe.get("elapsed_seconds"))
            if not row["validated"]:
                row["validation_reason"] = reason(probe.get("reason_code"))
        for phase, root in (("probe", probe_root / "pool"), ("benchmark", run / "pool")):
            state = safe_read(root / "state.json")
            requests = [r for r in state.get("requests", []) if r.get("api") == api_id]
            counts = Counter()
            for request in requests:
                status = request.get("status")
                status = status if status in {"started", "completed", "error"} else "unknown"
                counts[status] += 1
                totals[phase][status] += 1
                if status == "started":
                    row["active_requests"].append({
                        "phase": phase, "request_id": numeric(request.get("request_id")),
                        "age_seconds": round(max(0, now - numeric(request.get("started"), now)), 1),
                        "timeout_seconds": numeric(api.get("timeout_seconds")),
                        "owner_or_worker_alive": alive or bool(workers),
                    })
                else:
                    code = reason(request.get("reason_code")) if status == "error" else None
                    http = request.get("http_status")
                    http = http if type(http) is int and 100 <= http <= 599 else None
                    event = {
                        "phase": phase, "api_id": api_id, "request_id": numeric(request.get("request_id")),
                        "status": status, "http_status": http, "reason_code": code,
                        "finished_at": numeric(request.get("finished")),
                        "elapsed_seconds": round(max(0, numeric(request.get("finished"))
                                                     - numeric(request.get("started"))), 1)}
                    ledger_events.append(event)
                    if status == "error":
                        row["recent_http_errors"].append(event)
            row["accounting"][phase] = dict(counts)
            cooldown = state.get("cooldowns", {}).get(api_id, {})
            row[phase + "_cooldown_remaining_seconds"] = round(
                max(0, numeric(cooldown.get("until")) - now), 1)
            pause = safe_read(root / ("paused-" + api_id + ".json"))
            row[phase + "_paused"] = bool(pause)
            if pause:
                row[phase + "_pause_reason"] = reason(pause.get("reason"))
        row["recent_http_errors"] = sorted(row["recent_http_errors"],
                                          key=lambda v: v["finished_at"])[-5:]
        api_rows.append(row)
    execution = safe_read(run / "execution.json")
    episode_count = sum(1 for _ in (run / "episodes").rglob("episode.json"))
    stage = bootstrap.get("stage")
    stage = stage if stage in STAGES else "starting"
    if (run / "stopped_by_user.json").exists():
        stage = "stopped_by_user"
    elif (run / "stopped_for_model_binding_repair.json").exists():
        stage = "stopped_for_model_binding_repair"
    elif (run / "bootstrap_failure.json").exists():
        stage = "bootstrap_failed"
    if not alive and not workers and stage in {"starting", "validating_apis", "executing"}:
        stage = "owner_exited"
    status = {
        "schema": "gptexp-passive-connections/1", "at": now, "interval_seconds": 5,
        "model": config.get("model") if config.get("model") == "gpt-5.6-terra" else "unknown",
        "reasoning_effort": "high" if config.get("reasoning_effort") == "high" else "unknown",
        "stage": stage, "owner_pid": owner_pid, "owner_alive": alive,
        "monitor_pid": os.getpid(), "tcp_table_readable": tcp_readable,
        "https_tcp_states": dict(tcp_total), "processes": workers, "apis": api_rows,
        "validated_apis": sum(r["validated"] for r in api_rows),
        "requests": {phase: dict(counts) for phase, counts in totals.items()},
        "completed_arms": episode_count, "expected_arms": 3996,
        "execution_complete": execution.get("status") == "complete",
        "monitor_requests": 0, "automatic_retry": False, "reference_answers_read": False,
        "socket_observation_scope": "owned_process_https_tcp; bootstrap sockets are aggregate",
        "connection_is_not_response_validation": True}
    return status, ledger_events


def report(status):
    tcp = ", ".join(f"{k}={v}" for k, v in sorted(status["https_tcp_states"].items())) or "无可见连接"
    lines = [
        "# gpt-5.6-terra 实时连接状态", "",
        f"更新时间：{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(status['at']))}",
        f"模型：{status['model']}；思考档位：{status['reasoning_effort']}；阶段：{status['stage']}",
        f"API 核验：{status['validated_apis']}/10；已保存题臂：{status['completed_arms']}/3996",
        f"HTTPS TCP：{tcp}；执行进程存活：{status['owner_alive']}",
        "", "每 5 秒被动读取请求账本和 Linux 进程 socket，不发额外模型请求。",
        "TCP ESTABLISHED 只表示连接建立；调用成功以响应、模型和 high 档位核验为准。",
        "API 核验期间同一进程持有多路连接，TCP 数量只在进程层面归属。",
        "", "| API | 核验 | 在途 | 最长等待（秒） | 排队 | 正式完成 | 正式失败 | 最近错误 |",
        "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in status["apis"]:
        benchmark = row["accounting"]["benchmark"]
        latest = row["recent_http_errors"][-1] if row["recent_http_errors"] else {}
        error = str(latest.get("http_status") or latest.get("reason_code") or "")
        lines.append(
            f"| {row['api_id']} | {row['validation_status']} | {len(row['active_requests'])} | "
            f"{max((r['age_seconds'] for r in row['active_requests']), default=0)} | "
            f"{row['queued_requests']} | {benchmark.get('completed', 0)} | "
            f"{benchmark.get('error', 0)} | {error} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--owner-pid", type=int, required=True)
    parser.add_argument("--owner-start-ticks", type=int, required=True)
    args = parser.parse_args()
    # This process never needs API credentials.
    for name in tuple(os.environ):
        if any(part in name.upper() for part in ("API_KEY", "TOKEN", "BASE_URL", "ENDPOINT", "MODELS_KEY")):
            os.environ.pop(name)
    run = Path(args.run).absolute()
    with (run / "connection_monitor.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seen = set()
        while True:
            status, events = collect(run, args.owner_pid, args.owner_start_ticks)
            with (run / "connection_events.jsonl").open("a", encoding="utf-8") as stream:
                for event in events:
                    identity = (event["phase"], event["api_id"], event["request_id"], event["status"])
                    if identity not in seen:
                        stream.write(json.dumps({"event": "request_finished", **event}) + "\n")
                        seen.add(identity)
                stream.write(json.dumps({
                    "event": "snapshot", "at": status["at"], "stage": status["stage"],
                    "owner_alive": status["owner_alive"], "https_tcp_states": status["https_tcp_states"],
                    "requests": status["requests"], "validated_apis": status["validated_apis"],
                    "completed_arms": status["completed_arms"]}) + "\n")
            terminal = not status["owner_alive"] and not status["processes"]
            status["monitor_status"] = "finished" if terminal else "running"
            write(run / "connection_status.json", status)
            target = run / "connection_status.md"
            temp = target.with_name(".connection_status.md.tmp")
            temp.write_text(report(status), encoding="utf-8")
            os.replace(temp, target)
            if terminal:
                break
            time.sleep(5)


if __name__ == "__main__":
    main()
