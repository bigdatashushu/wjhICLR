#!/usr/bin/env python3
"""Publish the three user-authorized counting methods with immutable native receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill3d.evolution.fitset_revision_v11 import load_active_parent_slots
from skill3d.schemas import SkillCandidateV11, SkillSpecV11
from skill3d.skills.promote_atomic import read_active_snapshot
from skill3d.skills.v11_library import publish_v11_candidate
from skill3d.synthesis.counting_context import COUNTING_CONTEXT_VERSION

METHOD = """# Count whole instances across views

1. Fix the category, scope and exclusions. Before consulting candidate totals, enumerate whole instances from delivered images. For each keep a brief representative frame/location anchor and appearance or neighboring structure. Do not infer hidden instances without evidence.
2. Keep objects coexisting in one frame separate. Match reappearances using spatial continuity plus appearance/neighbors; similar appearance alone is insufficient. Check tracks and geometric overlap against images. Seams, legs, attachments and nested components alone do not establish additional whole objects.
3. Revisit missed regions, occlusions, ambiguous category membership and uncertain exclusions that could change the total. Explain inclusion/exclusion using the best view. Distinguish unseen coverage from confirmed absence; a technical scan or usable binding does not certify complete real-instance coverage.
4. If current images settle identity and coverage, answer with zero tools. Otherwise identify the exact doubt and choose the relevant authorized evidence: an existing candidate page, representative crop, or target-category detection on unresolved views. A full-frame scan is justified only by a coverage gap requiring it, not as a default.
5. When reading inventory, keep fields fixed (obj_id,category_name,visible_frames,track_id,duplicate_suspect as needed); follow returned cursors/cumulative coverage, not offset arithmetic. Read existing IDs instead of relisting all objects or removing fields after completion. If mapping the inventory matters, map each relevant row to a whole instance or an explained duplicate/exclusion.
6. count_objects groups tracks and consolidates geometry; count, list length, track totals and detection totals remain diagnostics until checked against whole instances. Do not sum per-frame boxes or add new boxes directly to an old count. New 2D boxes have no 3D cloud/track; mask visibility may be unavailable. Grounded geometry can aid identity when authorized; counting requires no metric scale.
7. A fault, unhealthy detector or partial frame error provides no negative evidence. It cannot resolve an omission or disagreement with an old count. Do not repeat a failed service/binding premise unchanged; use a different available view or valid evidence. If uncertainty remains, submit the best image-supported estimate with honest basis.

Before submission, leave brief English comments identifying instances, coverage checked and material exclusions. Count each whole once; recheck possible overmerging, fragmentation and whole/part splits. Use the host's actual evidence basis and valid IDs. MeasurementAnswer only reads/replays a count; it does not establish category, identity or completeness. The method stays fixed; current tool authorization and evidence withdrawals always apply.
"""


def main():
    library = ROOT / "skill_library"
    report_path = library / "validation" / "counting_context_20261110.json"
    parents, identity = load_active_parent_slots(library)
    if all(parents[("object_counting", s)].version == "1.1.0"
           for s in ("g0m0", "g1m0", "g1m1")):
        if not report_path.exists():
            raise RuntimeError("Published slots lack their validation receipt")
        print(report_path.read_text())
        return
    untouched = {slot: spec.content_sha256 for slot, spec in parents.items()
                 if slot[0] != "object_counting"}
    frozen = {
        str(p.relative_to(library)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("versions", "snapshots", "manifests")
        for p in (library / folder).rglob("*")
        if p.is_file() and p.name != "active_snapshot.json"
    }
    specs = []
    for state in ("g0m0", "g1m0", "g1m1"):
        parent = parents[("object_counting", state)]
        prefix = parent.skill_md.split("\n# ", 1)[0]
        spec = SkillSpecV11(
            skill_id=parent.skill_id, version="1.1.0", question_type=parent.question_type,
            evidence_state=state, skill_md=prefix + "\n" + METHOD)
        assert spec.name == parent.name and len(spec.skill_md) <= 8000
        specs.append((parent, spec))
    receipts = []
    for parent, spec in specs:
        current = read_active_snapshot(library / "snapshots")
        candidate = SkillCandidateV11(
            candidate_id=f"counting-context-20261110-{spec.evidence_state}",
            parent_snapshot_id=current["snapshot_id"],
            parent_skill_version=f"{parent.skill_id}@{parent.version}",
            full_skill_spec=spec,
            modification_reason=(
                "User-authorized counting-only behavioral specialization: image-first whole-instance "
                "identity/coverage, targeted evidence, stable pagination and count-helper semantics. "
                "No sample identities, labels or answers in candidate content."),
            source_run_ref="user-requested-counting-context-20261110")
        snapshot, receipt = publish_v11_candidate(
            library, candidate, expected_parent_snapshot_id=current["snapshot_id"])
        receipts.append({
            k: v for k, v in receipt.items() if k != "source_diff"})
    active, _ = load_active_parent_slots(library)
    assert len(active) == 24 and len(untouched) == 21
    assert all(active[slot].content_sha256 == sha for slot, sha in untouched.items())
    assert all(hashlib.sha256((library / ref).read_bytes()).hexdigest() == sha
               for ref, sha in frozen.items())
    report = {
        "context_version": COUNTING_CONTEXT_VERSION, "model_requests": 0,
        "snapshot_before": identity, "snapshot_after": snapshot["snapshot_id"],
        "active_slots": len(active), "unchanged_other_slots": len(untouched),
        "old_immutable_files_verified": len(frozen), "promotions": receipts,
        "counting_skills": [{
            "state": spec.evidence_state, "parent": f"{parent.skill_id}@{parent.version}",
            "active": f"{spec.skill_id}@{spec.version}",
            "before_chars": len(parent.skill_md), "after_chars": len(spec.skill_md),
            "sha256": spec.content_sha256,
        } for parent, spec in specs],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
