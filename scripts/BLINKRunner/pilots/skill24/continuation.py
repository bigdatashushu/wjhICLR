"""Verify the exact carried request prefix and the revised Skill input identity."""
from pathlib import Path
import json


def verify_budget_prefix(current, baseline):
    count = len(baseline["requests"])
    if (current["policy"] != baseline["policy"]
            or current["requests"][:count] != baseline["requests"]
            or len(current["requests"]) < count
            or any(row["status"] not in ("completed", "error")
                   for row in baseline["requests"])):
        raise ValueError("carried_budget_prefix_changed")
    if any(row["status"] != "completed" for row in current["requests"][count:]):
        raise RuntimeError("new_failed_or_unresolved_request_no_retry")


def campaign_class(exp):
    class RevisedCampaign(exp.Campaign):
        def verify(self):
            lineage = self.manifest["skill_pilot_continuation"]
            if exp.runtime_identity() != self.inputs["runtime_identity"]:
                raise ValueError("runtime_changed")
            if exp.source_files(self.root/"code/source") != self.manifest["source_files"]:
                raise ValueError("frozen_solver_source_changed")
            for root, files in lineage["preserved_files"].items():
                for relative, digest in files.items():
                    if exp.file_sha(Path(root)/relative) != digest:
                        raise ValueError("prior_run_record_changed")
            for key, expected in self.inputs["inputs"].items():
                path = self.profile["inputs"][key]
                observed = exp.file_identity(path) if key == "host_answers" else exp.file_sha(path)
                if observed != expected:
                    raise ValueError("revised_input_identity_changed")
            baseline = json.loads((self.root/"budget_prefix.json").read_text(encoding="utf-8"))
            if exp.file_sha(self.root/"budget_prefix.json") != lineage["budget_prefix_sha256"]:
                raise ValueError("carried_budget_snapshot_changed")
            current = json.loads((self.root/"budget/state.json").read_text(encoding="utf-8"))
            verify_budget_prefix(current, baseline)
            usage = self.governor.summary()
            if usage["active_requests"] or usage["unknown_requests"]:
                raise RuntimeError("unresolved_active_requests")
    return RevisedCampaign
