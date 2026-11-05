"""Read sanitized SR progress without reading predictions, labels, or credentials."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def status(root):
    manifest = json.loads((root / "manifest.json").read_text())
    episodes, partial = [], 0
    for path in (root / "episodes").glob("*.json"):
        try:
            episodes.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            partial += 1
    state = json.loads((root / "budget/state.json").read_text())
    output = {
        "episodes_finished": len(episodes), "episodes_expected": len(manifest["jobs"]),
        "episode_files_currently_being_written": partial,
        "episode_statuses": dict(Counter(p["result"]["status"] for p in episodes)),
        "requests": len(state["requests"]),
        "request_statuses": dict(Counter(p["status"] for p in state["requests"])),
        "completed": (root / "completed.json").exists(), "stopped": (root / "stopped.json").exists(),
        "exit": json.loads((root / "exit.json").read_text()) if (root / "exit.json").exists() else None,
    }
    startup = json.loads((root / "startup.json").read_text())
    proc = Path("/proc") / str(startup["runner_pid"])
    if (proc / "stat").exists():
        output["runner_identity_matches"] = (
            (proc / "stat").read_text().split(") ", 1)[1].split()[19] ==
            startup["runner_start_ticks"] and
            hashlib.sha256((proc / "cmdline").read_bytes()).hexdigest() ==
            startup["proc_command_sha256"])
    else:
        output["runner_identity_matches"] = None
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(status(args.root.resolve())))
