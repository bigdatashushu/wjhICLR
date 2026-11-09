"""Read-only paired run status; usable through the cluster's stdin transport."""
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import sys

DEFAULT_ROOT = Path("/home/wangy/wangjh/harness3d/four-datasets-20261031/cv3d-simple-paired-20261101-r02")


def status(root=DEFAULT_ROOT):
    root = Path(root)
    out = {"at": datetime.now().astimezone().isoformat(), "root": str(root),
           "inspection_model_requests": 0}
    for name in ("startup.json", "preflight_progress.json", "preflight.json",
                 "protocol_validation.json", "failure.json", "exit.json", "completion.json"):
        if (root / name).exists():
            out[name] = json.loads((root / name).read_text())
    revision = root / "ops/concurrency8"
    for name in ("drain-progress.json", "drained.json", "startup.json", "policy-change.json",
                 "launch.json", "switch-failure.json", "failure.json"):
        if (revision / name).exists():
            out.setdefault("concurrency_revision", {})[name] = json.loads(
                (revision / name).read_text())
    if "startup.json" in out:
        start = out.get("concurrency_revision", {}).get("startup.json", out["startup.json"])
        process = Path("/proc") / str(start["pid"])
        if process.exists():
            fields = (process / "stat").read_text().split(") ", 1)[1].split()
            same = (fields[19] == start.get("start_ticks") and
                    hashlib.sha256((process / "cmdline").read_bytes()).hexdigest() ==
                    start.get("command_sha256"))
            out["process"] = {"alive": same and fields[0] not in {"Z", "X"},
                              "identity_verified": same}
        else:
            out["process"] = {"alive": False}
    for phase in ("protocol_train", "test"):
        info = {}
        for arm in ("tools", "direct"):
            episodes = root / phase / "episodes" / arm
            paths = list(episodes.glob("*/result.json"))
            rows = [json.loads(p.read_text()) for p in paths]
            requests = list(episodes.glob("*/*.intent.json"))
            info[arm] = {
                "started": len(list(episodes.glob("*/started.json"))),
                "requests_issued": len(requests),
                "responses_received": len(list(episodes.glob("*/*.response.json"))),
                "first_request_at": (
                    datetime.fromtimestamp(
                        min(p.stat().st_mtime for p in requests)).astimezone().isoformat()
                    if requests else None),
                "recorded": len(rows), "statuses": dict(Counter(r["status"] for r in rows)),
                "tools_attempted": sum(r.get("attempted_tool_calls", 0) for r in rows),
                "tools_successful": sum(r.get("succeeded_tool_calls", 0) for r in rows),
            }
        if (root / phase / "scores.json").exists():
            value = json.loads((root / phase / "scores.json").read_text())
            info["complete"] = value["complete"]
            info["paired"] = value["paired"]
            info["accuracy"] = {
                arm: value["arms"][arm]["overall"] for arm in ("tools", "direct")}
        out[phase] = info
    if (root / "budget/state.json").exists():
        value = json.loads((root / "budget/state.json").read_text())
        out["budget"] = {
            "policy": value["policy"], "used": len(value["requests"]),
            "statuses": dict(Counter(r["status"] for r in value["requests"])),
        }
        active_ids = []
        for index in range(value["policy"]["max_inflight"]):
            with (root / "budget" / f"slot-{index:03d}.lock").open("a+b") as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.seek(0)
                    key = stream.read().decode("ascii")
                    if key.isdigit():
                        active_ids.append(int(key))
        out["cv3d_active_requests"] = len(active_ids)
        out["cv3d_inflight_by_arm"] = dict(Counter(
            value["requests"][key]["context"]["variant_id"] for key in active_ids
            if key < len(value["requests"])))
    config = json.loads((root / "run_config.json").read_text())
    if not config.get("shared_slots_root"):
        out["combined_active_request_slots"] = out.get("cv3d_active_requests", 0)
        out["evaluation_role"] = config.get("evaluation_role", "benchmark")
        return out
    slots = Path(config["shared_slots_root"])
    live = 0
    for index in range(4):
        with (slots / f"slot-{index:03d}.lock").open("a+b") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                live += 1
    out["mmsi_shared_slots_active"] = live
    out["combined_active_request_slots"] = (
        live + out.get("cv3d_active_requests", 0)
        if out.get("concurrency_revision", {}).get("startup.json") else live)
    return out


if __name__ == "__main__":
    print(json.dumps(status(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT),
                     ensure_ascii=False))
