#!/usr/bin/env python3
"""Propose label-free, same-type/same-source replacements without changing data."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from audit_metric_coverage import checked, checked_identity, load_json


def propose(inventory, report, catalog):
    checked(report["status"] == "complete", "complete_native_report_required")
    checked(report["anchors"] == inventory["anchors"], "report_inventory_identity_changed")
    checked(catalog["status"] == "verified" and catalog["no_labels"] is True
            and catalog["identity_hash"] == inventory["anchors"]["catalog_identity_hash"],
            "catalog_identity_changed")
    scene_status = {r["scene_id"]: r["status"] for r in report["scene_checks"]}
    selected = inventory["selected"]
    selected_ids = {r["qa_id"] for r in selected}
    missing = [r for r in selected if scene_status[r["scene_id"]] != "pass"]
    needed = Counter((r["raw_type"], r["source_split"]) for r in missing)
    candidates = []
    for qa, entry in catalog["variants"]["metric"].items():
        if qa in selected_ids or scene_status.get(entry["scene_id"]) != "pass":
            continue
        cost = hashlib.sha256(f"{inventory['seed']}|metric-replacement|{qa}".encode()).hexdigest()
        candidates.append((cost, qa, entry))
    candidates.sort()
    validation_cache, found = {}, {}
    read_count = 0
    for _, qa, entry in candidates:
        if not any(n > 0 and split == entry["split"] for (_, split), n in needed.items()):
            continue
        path = entry["migration_receipt_file"]
        if path not in validation_cache:
            validation_cache[path], _ = load_json(path, entry["migration_receipt_file_sha256"])
        _, _, public = checked_identity(entry, validation_cache[path])
        read_count += 1
        cell = public["raw_type"], public["source_split"]
        if needed[cell] > 0:
            found.setdefault(cell, []).append(public)
            needed[cell] -= 1
        if not any(needed.values()):
            break
    replacements, proposal = [], []
    for old in selected:
        if scene_status[old["scene_id"]] == "pass":
            proposal.append(dict(old))
            continue
        cell = old["raw_type"], old["source_split"]
        if not found.get(cell):
            continue
        new = found[cell].pop(0)
        new["experiment_split"] = old["experiment_split"]
        proposal.append(new)
        replacements.append({"old": old, "new": new})
    complete = len(proposal) == len(selected)
    if complete:
        checked(len({r["qa_id"] for r in proposal}) == len(selected), "duplicate_replacement")
        for fields in (("raw_type",), ("raw_type", "source_split"),
                       ("raw_type", "source_split", "experiment_split")):
            checked(Counter(tuple(r[k] for k in fields) for r in proposal)
                    == Counter(tuple(r[k] for k in fields) for r in selected),
                    "replacement_quota_changed")
        checked(all(scene_status[r["scene_id"]] == "pass" for r in proposal),
                "replacement_scene_not_qualified")
    return {
        "schema": "vsi1000-metric-replacement-proposal/1",
        "status": "complete_proposal" if complete else "insufficient_candidates",
        "selection_id": inventory["selection_id"], "seed": inventory["seed"],
        "anchors": inventory["anchors"],
        "n_original_questions": len(selected),
        "n_retained_questions": len(selected) - len(missing),
        "n_questions_requiring_replacement": len(missing),
        "n_replacements_found": len(replacements), "n_proposed_questions": len(proposal),
        "original_scene_count": len({r["scene_id"] for r in selected}),
        "proposed_scene_count": len({r["scene_id"] for r in proposal}),
        "original_role_scene_counts": dict(
            (role, len({r["scene_id"] for r in selected if r["experiment_split"] == role}))
            for role in ("train", "test")),
        "proposed_role_scene_counts": dict(
            (role, len({r["scene_id"] for r in proposal if r["experiment_split"] == role}))
            for role in ("train", "test")),
        "exact_raw_type_quotas_preserved": complete,
        "exact_source_and_role_joint_quotas_preserved": complete,
        "reference_answers_read": False, "selection_uses_answers_or_scores": False,
        "model_requests": 0, "dataset_mutated": False,
        "public_candidate_identities_read": read_count,
        "replacement_pairs": replacements, "proposed_public_selection": proposal,
        "unfilled_cells": [
            {"raw_type": cell[0], "source_split": cell[1], "missing": n}
            for cell, n in needed.items() if n],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args()
    inventory = load_json(args.inventory)[0]
    report, report_hash = load_json(args.report)
    catalog, _ = load_json(args.catalog, inventory["anchors"]["catalog_sha256"])
    result = propose(inventory, report, catalog)
    result["native_report_file_sha256"] = report_hash
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
