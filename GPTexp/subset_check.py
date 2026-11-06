"""Offline checks for quota selection, immutable reuse and paired scheduling."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
from unittest.mock import patch

from .model_binding import bound_spec
from .subset import arm_cap, missing_pairs, select_pairs, verified_record, verify_reuse
from .subset_export import merged_export
from .partial_score import freeze
from .traces import digest, file_digest, read, write


def seal(value):
    value["run_identity"] = digest(value)
    return value


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("changed evidence was accepted")


def fixture(root, manifest, pair, arm, *, failure=None, answer="A"):
    spec = next(s for s in pair["arms"] if s["arm"] == arm)
    output = root / "episodes" / pair["dataset"] / pair["pair_identity"] / arm
    api = "xcode-1"
    job = {"spec": spec, "sample_id": pair["sample_id"], "api_id": api,
           "config": manifest["config"], "run_identity": manifest["run_identity"],
           "pair_identity": pair["pair_identity"], "pool_root": str(root / "pool")}
    job["operation_id"] = digest({"pair": pair["pair_identity"], "arm": arm, "api": api,
                                  "run": manifest["run_identity"]})
    write(output / "dispatch.json", {"job_sha256": digest(job), "api_id": api,
                                     "operation_id": job["operation_id"]})
    bound, fields = bound_spec(spec, manifest["config"]["model"])
    write(output / "model_binding.json", {
        "model": manifest["config"]["model"], "original_spec_sha256": digest(spec),
        "bound_spec_sha256": digest(bound), "changed_fields": fields,
        "non_model_fields_identical": True, "run_identity": manifest["run_identity"],
        "reference_answers_read": False, "adapter_bytes_unchanged": True, "worker_bytes_unchanged": True})
    write(output / "episode.json", {
        "dataset": pair["dataset"], "arm": arm, "sample_id": pair["sample_id"], "api_id": api,
        "run_identity": manifest["run_identity"], "pair_identity": pair["pair_identity"],
        "operation_id": job["operation_id"], "job_sha256": digest(job), "request_files": {},
        "reference_answer_visible": False, "automatic_retry": False, "failure": failure,
        "result": None if failure else {"status": "answered", "answer": answer}})
    return output


def main():
    pairs = []
    for i in range(10):
        public = {"sample_id": str(i), "task_type": "Multi-view_Reasoning"}
        pairs.append({"pair_identity": digest(i), "dataset": "blink-multiview", "sample_id": str(i),
                      "arms": [{"arm": a, "problem": public} for a in ("direct", "harness")]})
    previous = seal({"pairs": pairs, "counts": {"blink-multiview": 10}, "arms": 20,
                     "config": {"seed": 137, "model": "gpt-5.6-terra", "max_requests": 50}})
    mandatory = {(p["pair_identity"], "direct") for p in pairs[:4]}
    chosen, strata = select_pairs(previous, mandatory)
    assert len(chosen) == 4 and strata[0]["minimum_questions"] == 3
    assert chosen == select_pairs(previous, mandatory)[0]
    assert len(select_pairs(previous, set())[0]) == 3
    rejects(lambda: select_pairs(previous, {("unknown", "direct")}))
    with tempfile.TemporaryDirectory(prefix="gptexp-subset-check-") as temporary:
        base = Path(temporary)
        source, target = base / "old", base / "new"
        source.mkdir(); target.mkdir()
        write(source / "manifest.json", previous)
        ledger = {"policy_identity": digest(previous["config"]), "requests": []}
        write(source / "pool/state.json", ledger)
        (source / "pool/state.lock").touch()
        finalized = set()
        for i, pair in enumerate(pairs[:3]):
            for arm in ("direct", "harness"):
                if i == 2 and arm == "harness":
                    continue
                fixture(source, previous, pair, arm, failure={"error_type": "APIError"} if i == 0 else None,
                        answer="invalid" if i == 1 else "A")
                finalized.add((pair["pair_identity"], arm))
        freeze(source, target / "reuse_source")
        snapshot = read(target / "reuse_source/snapshot.json")
        selected, strata = select_pairs(previous, finalized)
        budget = sum(arm_cap(p["dataset"], s["arm"]) for p in selected for s in p["arms"]
                     if (p["pair_identity"], s["arm"]) not in finalized)
        manifest = seal({
            "pairs": selected, "strata": strata, "counts": {"blink-multiview": 3},
            "arms": 6, "questions": 3, "config": {**previous["config"], "max_requests": budget},
            "max_requests": budget, "reused_arms": 5, "missing_arms": 1,
            "source_run_identity": previous["run_identity"],
            "reuse_snapshot_identity": snapshot["snapshot_identity"],
            "reused_api_pins": {p["pair_identity"]: "xcode-1" for p in selected},
            "validation_receipt_sha256": {}, "selection_policy": "offline fixture"})
        write(target / "manifest.json", manifest)
        reused = verify_reuse(target, manifest)
        jobs, unresolved, _ = missing_pairs(target, manifest, reused)
        assert len(reused) == 5 and len(jobs) == 1 and not unresolved
        assert jobs[0][1][0]["arm"] == "harness" and jobs[0][2] == "xcode-1"
        entry = snapshot["index"][0]
        path = target / "reuse_source" / entry["episode"]
        original_bytes = path.read_bytes()
        path.write_bytes(original_bytes + b" ")
        rejects(lambda: verify_reuse(target, manifest))
        path.write_bytes(original_bytes)
        duplicate = target / "episodes" / selected[0]["dataset"] / selected[0]["pair_identity"] / "direct"
        duplicate.mkdir(parents=True)
        rejects(lambda: missing_pairs(target, manifest, reused))
        duplicate.rmdir()
        fresh_ledger = {"policy_identity": digest(manifest["config"]), "requests": []}
        write(target / "pool/state.json", fresh_ledger)
        (target / "pool/state.lock").touch()
        pair, specs, _ = jobs[0]
        output = fixture(target, manifest, pair, specs[0]["arm"])
        verified_record(output, manifest, pair, specs[0], target, fresh_ledger)
        wrong = deepcopy(read(output / "model_binding.json"))
        wrong["model"] = "another-model"
        write(output / "model_binding.json", wrong)
        rejects(lambda: verified_record(output, manifest, pair, specs[0], target, fresh_ledger))
        fixture(target, manifest, pair, specs[0]["arm"])
        write(target / "execution.json", {"status": "complete"})
        with patch("GPTexp.score.references_for", return_value=[
                {"dataset": p["dataset"], "sample_id": p["sample_id"], "reference_answer": "A"}
                for p in selected]) as references:
            result = merged_export(target, score=True)
            assert result["episodes"] == 6 and result["reused_arms"] == 5
            references.assert_called_once()
            full = read(target / "results_and_full_trajectories.json")
            old = [e for e in full["episodes"] if e["reuse_provenance"]["reused"]]
            assert len(old) == 5 and all(e["run_identity"] == previous["run_identity"] for e in old)
            assert sum(bool(e["failure"]) for e in old) == 2
            path.write_bytes(original_bytes + b" ")
            rejects(lambda: merged_export(target, score=True))
            references.assert_called_once()
    print("PASS: quotas; all failed/invalid arms reused; no redispatch; partial API pin; "
          "byte/model tamper rejection; merged provenance; verify before reference access.")


if __name__ == "__main__":
    main()
