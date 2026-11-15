"""Apply the user-approved direction draft through the native immutable publisher."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from skill3d.evolution.fitset_revision_v11 import load_active_parent_slots
from skill3d.online import runner
from skill3d.schemas import SkillCandidateV11, SkillSpecV11
from skill3d.schemas.evidence import EvidenceProfile
from skill3d.skills.v11_library import publish_v11_candidate
from skill3d.synthesis import prompt_builder
from skill3d.tools.contract import KNOWN_ARTIFACTS


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    library = ROOT / "skill_library"
    draft_root = Path("/Users/bytedance/wangjh/vsi-direction-review-20261111/assets")
    report_path = library / "validation" / "relative_direction_context_20261111.json"
    parents, before = load_active_parent_slots(library)
    parent = parents[("object_rel_direction", "g1m0")]
    if parent.version == "1.1.0" and report_path.is_file():
        assert parent.content_sha256 == json.loads(report_path.read_text())["skill"]["sha256"]
        print(report_path.read_text())
        return
    assert parent.version == "1.0.0"
    assert parent.skill_md == (draft_root / "S06-g1m0-original.md").read_text()
    content = (draft_root / "S06-g1m0-candidate.md").read_text()
    assert "\r" not in content and content.endswith("\n")
    assert not re.search(r"\bQA\d+\b|reference_answer|ground_truth", content)
    spec = SkillSpecV11(
        skill_id=parent.skill_id,
        version="1.1.0",
        question_type=parent.question_type,
        evidence_state=parent.evidence_state,
        skill_md=content,
    )
    assert spec.name == parent.name and len(content) <= 8000
    untouched = {
        slot: (old.version, old.content_sha256)
        for slot, old in parents.items()
        if slot != ("object_rel_direction", "g1m0")
    }
    immutable = {
        str(path.relative_to(library)): sha(path)
        for folder in ("versions", "snapshots", "manifests")
        for path in (library / folder).rglob("*")
        if path.is_file() and path.name != "active_snapshot.json"
    }
    preserved_paths = [
        ROOT / "configs/vsi_m4_1000.json",
        ROOT / "configs/datasets.json",
        ROOT / "src/skill3d/sandbox/ast_guard.py",
        ROOT / "src/skill3d/online/vsi_execution_contract.py",
    ]
    preserved = {str(p.relative_to(ROOT)): sha(p) for p in preserved_paths}
    candidate = SkillCandidateV11(
        candidate_id="relative-direction-context-20261111-g1m0",
        parent_snapshot_id=before["snapshot_id"],
        parent_skill_version=f"{parent.skill_id}@{parent.version}",
        full_skill_spec=spec,
        modification_reason=(
            "User-approved diagnostic revision: image-region evidence for the three referents; "
            "confirmed usable instances before geometry; preserve visual referents when bindings "
            "are missing; compare horizontal observer rays and the existing 135-degree threshold. "
            "Preserve direction mapping and authorization. No sample identities or answers in method."
        ),
        source_run_ref="ccfa-workfiles/trajectory-review/rel-direction30-20261111",
    )
    snapshot, receipt = publish_v11_candidate(
        library, candidate, expected_parent_snapshot_id=before["snapshot_id"])
    active, after = load_active_parent_slots(library)
    selected = active[("object_rel_direction", "g1m0")]
    assert len(active) == 24 and len(untouched) == 23
    assert selected.version == "1.1.0" and selected.content_sha256 == spec.content_sha256
    assert all((active[slot].version, active[slot].content_sha256) == identity
               for slot, identity in untouched.items())
    assert all(sha(library / ref) == digest for ref, digest in immutable.items())
    assert all(sha(ROOT / ref) == digest for ref, digest in preserved.items())
    profile = EvidenceProfile(geometry_3d="available", metric_scale="unavailable")
    scene = SimpleNamespace(
        question_type="object_rel_direction", question_tool_scope="full_3d",
        scene_route="full_3d", summary="Synthetic direction scene; no reference labels.",
        available_artifacts=KNOWN_ARTIFACTS, grounding_coverage={},
        local_facts={"world_frame": True}, evidence_profile=profile,
        metric_evidence_gate_result=SimpleNamespace(gate_passed=False, missing_subconditions=[]))
    episode = SimpleNamespace(
        qa_id="synthetic", scene_name="synthetic", frame_set=None,
        question="Standing by a shelf and facing a window, is the table to my left, right, or back?",
        question_type="object_rel_direction",
        options=["A. left", "B. right", "C. back"])
    prompt, delivery = runner._build_prompt_ex(episode, scene, scene, [selected], None)
    new_shared_line = next(line for line in prompt_builder._PROGRAM_SYNTH.splitlines()
                           if line.startswith("- M5/Tool IDs"))
    assert selected.skill_md in prompt
    assert delivery.entries[0].text == content
    assert delivery.entries[0].content_sha256 == spec.content_sha256
    assert new_shared_line in prompt
    assert Path(prompt_builder.__file__).is_relative_to(ROOT / "src")
    report = {
        "status": "published_and_offline_delivery_verified",
        "model_requests": 0,
        "snapshot_before": before,
        "snapshot_after": after,
        "active_slots": len(active),
        "unchanged_other_slots": len(untouched),
        "old_immutable_files_verified": len(immutable),
        "skill": {
            "parent": f"{parent.skill_id}@{parent.version}",
            "active": f"{selected.skill_id}@{selected.version}",
            "before_chars": len(parent.skill_md),
            "after_chars": len(content),
            "sha256": spec.content_sha256,
            "source_ref": snapshot["entries"][candidate.candidate_skill_version]["source_ref"],
            "full_method_delivery_verified": True,
        },
        "shared_prompt": {
            "source": str(Path(prompt_builder.__file__).relative_to(ROOT)),
            "sha256": sha(Path(prompt_builder.__file__)),
            "old_rule_chars": 250,
            "new_rule_chars": len(new_shared_line),
            "new_rule": new_shared_line,
            "actual_rendered_prompt_verified": True,
        },
        "preserved_config_and_contract_hashes": preserved,
        "promotion": {key: value for key, value in receipt.items() if key != "source_diff"},
        "revision_evidence_role": "user-authorized diagnostic feedback; no held-out score claimed",
        "accuracy_retested": False,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
