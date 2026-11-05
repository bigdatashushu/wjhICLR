"""Original host-reference lookups, permitted only after complete prediction freeze."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from .adapters import BASE, ROOT
from .export import export
from .traces import digest, file_digest, read, write

MINDCUBE_ANSWERS = "/home/wangy/wangjh/harness3d-external-benchmarks-20261023/MindCube/extracted/data/raw/MindCube_tinybench.jsonl"
MINDCUBE_SHA = "0289eb82d81ff9aa0201ae75f86da7fd1924cf23dc202856d0579a0effd22ac8"


def references_for(manifest, pairs=None):
    """Open pinned host references only after the caller has frozen predictions."""
    common_path = Path(BASE + "/smallimage/host_evaluation/answers.json")
    original = read(ROOT / "frozen/blink-multiview/manifest.json")
    if file_digest(common_path) != original["reference_file_sha256"]:
        raise ValueError("original_reference_bytes_changed")
    common = {(row["benchmark"], row["sample_id"]): row["reference_answer"]
              for row in read(common_path)}
    mindcube = {}
    if "mindcube-tiny" in manifest["counts"]:
        if file_digest(MINDCUBE_ANSWERS) != MINDCUBE_SHA:
            raise ValueError("original_mindcube_reference_bytes_changed")
        mindcube = {str(row["id"]): row for row in
                    (json.loads(line) for line in Path(MINDCUBE_ANSWERS).read_text().splitlines())}
    sr = {}
    if "blink-sr" in manifest["counts"]:
        sr_root = Path(BASE + "/blink-sr-paired20-20261104-r03/dataset/data")
        sr_manifest = read(sr_root / "manifest.json")
        if file_digest(sr_root / "references.jsonl") != sr_manifest["reference_index_sha256"]:
            raise ValueError("original_sr_reference_bytes_changed")
        sr = {row["sample_id"]: row for row in
              (json.loads(line) for line in (sr_root / "references.jsonl").read_text().splitlines())}
    references = []
    for pair in manifest["pairs"] if pairs is None else pairs:
        dataset, sample_id = pair["dataset"], pair["sample_id"]
        public = pair["arms"][0].get("problem", pair["arms"][0].get("item"))
        if dataset == "mindcube-tiny":
            row = mindcube[sample_id]
            if (row["question"] != public["question"] or str(row["type"]) != public["task_type"]
                    or len(row["images"]) != len(public["views"])):
                raise ValueError("mindcube_reference_question_changed")
            answer = row["gt_answer"]
        elif dataset == "blink-sr":
            row = sr[sample_id]
            if row["split"] != "val" or row["gt_available"] is not True:
                raise ValueError("sr_reference_role_changed")
            answer = row["gt_answer"]
        else:
            answer = common[("blink" if dataset == "blink-multiview" else dataset, sample_id)]
        if answer not in ("A", "B", "C", "D"):
            raise ValueError("invalid_reference_letter")
        references.append({"dataset": dataset, "sample_id": sample_id, "reference_answer": answer})
    return references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    directory = Path(args.run)
    with (directory / "operator.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read(directory / "manifest.json")
        if (digest({k: v for k, v in manifest.items() if k != "run_identity"}) != manifest["run_identity"]
                or read(directory / "execution.json")["status"] != "complete"):
            raise ValueError("full_execution_not_frozen")
        # Validate every result and request record before opening any reference.
        for pair in manifest["pairs"]:
            endpoints = []
            for spec in pair["arms"]:
                root = directory / "episodes" / pair["dataset"] / pair["pair_identity"] / spec["arm"]
                episode = read(root / "episode.json")
                if (episode["run_identity"] != manifest["run_identity"]
                        or episode["pair_identity"] != pair["pair_identity"]
                        or episode["reference_answer_visible"] is not False):
                    raise ValueError("prediction_identity_changed")
                endpoints.append(episode["api_id"])
                for name, expected in episode["request_files"].items():
                    if file_digest(root / name) != expected:
                        raise ValueError("request_record_changed")
            if len(set(endpoints)) != 1:
                raise ValueError("pair_endpoint_affinity_changed")
        references = references_for(manifest)
        path = directory / "host_scoring_references.json"
        write(path, {"references": references, "run_identity": manifest["run_identity"],
                     "reference_access": "host-only after complete prediction freeze"})
        print(json.dumps(export(directory, references=path)))


if __name__ == "__main__":
    main()
