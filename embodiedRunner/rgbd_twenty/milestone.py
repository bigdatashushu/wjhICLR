"""Wait for new completed episode receipts using filesystem notifications."""
import argparse
import json
import os
from pathlib import Path
import select

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    assert 1 <= args.count <= 20
    frozen = json.loads((OUT / "frozen.json").read_text())
    run_roots = [ROOT / package / "runs" / pin["run_id"] for package, pin in frozen["packages"].items()]
    queue = select.kqueue()
    opened = {}
    try:
        while True:
            watched = {OUT / "controller.json"}
            for run in run_roots:
                watched.add(run.parent)
                if run.exists():
                    watched.add(run)
                    for folder in run.glob("*-base-*"):
                        watched.add(folder)
                        watched.update(p for p in folder.glob("*.jsonl"))
                        watched.update(p for p in folder.glob("result.json"))
            for path in watched - opened.keys():
                fd = os.open(path, os.O_RDONLY)
                opened[path] = fd
                queue.control([select.kevent(
                    fd, filter=select.KQ_FILTER_VNODE, flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                    fflags=select.KQ_NOTE_WRITE | select.KQ_NOTE_EXTEND)], 0)
            rows = []
            for run in run_roots:
                for path in run.glob("*/result.json"):
                    try:
                        row = json.loads(path.read_text())
                        final = json.loads((path.parent / "trajectory.jsonl").read_text().splitlines()[-1])
                    except (json.JSONDecodeError, IndexError, FileNotFoundError):
                        continue
                    if final.get("type") != "episode_end" or final["result"] != row:
                        continue
                    rows.append({
                        "task": row["task"], "status": row["status"],
                        "success": row["official"].get("task_success") if row["official"] else None,
                        "steps": row["metrics"]["num_steps"], "requests": row["metrics"]["planner_steps"],
                    })
            try:
                controller = json.loads((OUT / "controller.json").read_text())
            except json.JSONDecodeError:
                controller = {"status": "running"}
            if len(rows) >= args.count or controller["status"] != "running":
                report = {"completed_receipts": len(rows), "target": args.count,
                          "controller_status": controller["status"], "episodes": rows}
                (OUT / f"milestone-{args.count:02d}.json").write_text(json.dumps(report) + "\n")
                print(json.dumps(report, ensure_ascii=False))
                return 0
            queue.control(None, 1)
    finally:
        queue.close()
        for fd in opened.values():
            os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
