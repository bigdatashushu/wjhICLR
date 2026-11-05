"""Observable changes and actionable unresolved evidence; notes remain hypotheses."""
from __future__ import annotations

from .tools import MEASUREMENTS
from .review import pending
from .common import digest

VERSION = "omni-evidence-progress/1"


def unresolved(observations, reviews):
    eligible = set(pending(observations, reviews))
    return [{"result_id": r["result_id"], "decision": reviews.get(
             r["result_id"], {}).get("decision", "unreviewed"),
             "points": [r["data"]["pixel_xy"]] if r["source_tool"] == "point_depth"
                       else [r["data"]["a_xy"], r["data"]["b_xy"]],
             "intended_targets": r["data"].get("intended_targets")}
            for r in observations if r["result_id"] in eligible]


def summary(row):
    """Prompt-sized counts; full IDs stay in host progress receipts."""
    return {"new_observations": len(row["new_observation_ids"]),
            "novel_observations": len(row["novel_observation_ids"]),
            "changed_reviews": len(row["changed_review_ids"]),
            "answer_contract_passed": row["answer_contract_passed"],
            "made_progress": row["made_progress"]}


def record(previous_observations, observations, previous_reviews, reviews, answered=False):
    old_ids = {r["result_id"] for r in previous_observations}
    def identity(row):
        return digest({"tool": row["source_tool"], "data": {
            k: v for k, v in row["data"].items()
            if k not in ("coordinate_transforms", "replaces_result_id")}})
    seen = {identity(r) for r in previous_observations}
    def judgment(row):
        return {k: v for k, v in (row or {}).items() if k != "replaced_by_result_id"}
    changed = [key for key, row in reviews.items()
               if judgment(previous_reviews.get(key)) != judgment(row)]
    new = [r["result_id"] for r in observations if r["result_id"] not in old_ids]
    novel = []
    for row in observations:
        signature = identity(row)
        if row["result_id"] in new and signature not in seen:
            novel.append(row["result_id"])
            seen.add(signature)
    return {"version": VERSION, "new_observation_ids": new,
            "novel_observation_ids": novel,
            "changed_review_ids": changed, "answer_contract_passed": answered,
            "made_progress": bool(novel or changed or answered)}


def activity(observations, reviews, progress):
    """Observable activity, with usable novel replacements counted separately."""
    novel = {key for row in progress for key in row["novel_observation_ids"]}
    replacements = [r for r in observations if r["source_tool"] in MEASUREMENTS
                    and r["data"].get("replaces_result_id")]
    return {"new_observations": sum(len(r["new_observation_ids"]) for r in progress),
            "novel_observations": len(novel),
            "changed_reviews": sum(len(r["changed_review_ids"]) for r in progress),
            "no_progress_rounds": sum(not r["made_progress"] for r in progress),
            "answer_contract_passes": sum(r["answer_contract_passed"] for r in progress),
            "replacement_measurements": len(replacements),
            "usable_novel_replacements": sum(
                r["result_id"] in novel and reviews.get(r["result_id"], {}).get("decision") == "usable"
                for r in replacements)}
