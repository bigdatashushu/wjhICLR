"""Rehash the original explicit protection inventory; never parse old run data."""
import argparse
from pathlib import Path

from protocol import canonical, json_object, sha256

ROOT = Path(__file__).resolve().parent


def hash_file(path):
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_preservation():
    receipt = json_object((ROOT / "checks/architecture-preparation/local-readonly.json").read_bytes())
    result = {"schema": "embodied-original-preservation/1", "live_requests": 0,
              "simulator_launches": 0, "old_run_content_parsed": False}
    for key, base, label in (
        ("source_sha256", ROOT.parent, "source"),
        ("existing_baseline_files_sha256", ROOT, "baseline"),
    ):
        changes, missing = [], []
        for name, expected in receipt[key].items():
            path = (base / name).resolve()
            if not path.is_relative_to(base) or not path.is_file():
                missing.append(name)
            elif hash_file(path) != expected:
                changes.append(name)
        result[label] = {"files": len(receipt[key]), "changed": changes, "missing": missing}
    pointer = ROOT.parent / "skill_library/snapshots/active_snapshot.json"
    result["active_pointer_unchanged"] = (
        pointer.is_file() and hash_file(pointer) == receipt["active_pointer"]["sha256"])
    result["status"] = ("passed" if result["active_pointer_unchanged"] and all(
        not result[label]["changed"] and not result[label]["missing"] for label in ("source", "baseline"))
                        else "rejected")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_preservation()
    if args.output:
        output = args.output.resolve()
        if not output.is_relative_to(ROOT) or not output.parent.is_dir() or output.exists():
            parser.error("output must be a new file in an existing embodiedRunner directory")
        with output.open("xb") as stream:
            stream.write(canonical(result) + b"\n")
    print(canonical(result).decode())
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
