"""Merged trajectories retain original identities; labels stay in host scoring."""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path

from .export import export
from .subset import identity_check, safe_file, verified_record, verify_reuse
from .traces import read, write


def records(directory, manifest):
    reused = verify_reuse(directory, manifest)
    ledger_path = directory / "pool/state.json"
    if ledger_path.exists():
        with (directory / "pool/state.lock").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            ledger = read(ledger_path)
    else:
        ledger = None
    rows, endpoints = [], defaultdict(set)
    for pair in manifest["pairs"]:
        for spec in pair["arms"]:
            key = pair["pair_identity"], spec["arm"]
            current = directory / "episodes" / pair["dataset"] / pair["pair_identity"] / spec["arm"]
            if key in reused:
                if current.exists():
                    raise ValueError("duplicate_merged_arm")
                entry = reused[key]
                root = safe_file(directory / "reuse_source", entry["episode"]).parent
                episode = read(root / "episode.json")
                provenance = {
                    "reused": True, "selection_run_identity": manifest["run_identity"],
                    "source_run_identity": episode["run_identity"],
                    "source_snapshot_identity": entry.get(
                        "origin_snapshot_identity", manifest["reuse_snapshot_identity"]),
                    "original_episode": entry["episode"],
                    "original_episode_sha256": entry["files"][entry["episode"]],
                    "original_identity_and_bytes_preserved": True}
                complete = True
            else:
                root = current
                if (root / "episode.json").exists():
                    episode = verified_record(root, manifest, pair, spec, directory, ledger)
                    complete = True
                elif root.exists():
                    episode = {
                        "schema": "gptexp-partial-episode/1", "dataset": pair["dataset"],
                        "arm": spec["arm"], "sample_id": pair["sample_id"],
                        "pair_identity": pair["pair_identity"], "run_identity": manifest["run_identity"],
                        "reference_answer_visible": False, "automatic_retry": False,
                        "result": None, "failure": {"error_type": "episode_not_finalized"},
                        "request_files": {}}
                    complete = False
                else:
                    continue
                provenance = {"reused": False, "selection_run_identity": manifest["run_identity"]}
            if complete:
                endpoints[pair["pair_identity"]].add(episode["api_id"])
            rows.append((pair, spec, root, episode, complete, provenance))
    if any(len(ids) > 1 for ids in endpoints.values()):
        raise ValueError("merged_pair_api_affinity_changed")
    return rows


def merged_export(directory, *, score=False):
    directory = Path(directory).absolute()
    manifest = read(directory / "manifest.json")
    identity_check(manifest)
    rows = records(directory, manifest)
    completed = sum(row[4] for row in rows)
    references_path = None
    if score:
        if any(any(t in k.upper() for t in ("API_KEY", "TOKEN", "BASE_URL", "ENDPOINT", "MODELS_KEY"))
               for k in os.environ):
            raise ValueError("scoring_requires_credential_stripped_environment")
        if completed != manifest["arms"] or read(directory / "execution.json")["status"] != "complete":
            raise ValueError("subset_scoring_requires_full_prediction_freeze")
        # All original/fresh records, request ledgers and provenance verified above.
        from .score import references_for
        references_path = directory / "host_scoring_references.json"
        write(references_path, {
            "references": references_for(manifest), "run_identity": manifest["run_identity"],
            "reference_access": "host-only after merged immutable prediction verification"})

    def provider(_directory, _manifest):
        for pair, spec, root, episode, complete, provenance in rows:
            yield {
                **episode, "episode_complete": complete, "reuse_provenance": provenance,
                "public_input": spec.get("problem", spec.get("item")),
                "full_trajectory": {str(p.relative_to(root)): read(p)
                                    for p in sorted(root.rglob("*.json")) if p.name != "episode.json"}}

    result = export(directory, references=references_path, row_provider=provider)
    result["reused_arms"] = manifest["reused_arms"]
    result["new_arms"] = completed - manifest["reused_arms"]
    write(directory / "merged_export_receipt.json", {
        **result, "run_identity": manifest["run_identity"], "original_identity_preserved": True,
        "reference_answers_read": score})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    print(json.dumps(merged_export(args.run, score=args.score)))


if __name__ == "__main__":
    main()
