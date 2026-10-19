"""Outer holdout is report-only and cannot enter the promotion path."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from skill3d.evolution import outer_v11
from skill3d.evolution.live_v11 import (
    REACHABLE_V11_SLOTS,
    V11ValidationPanel,
    slot_key,
)
from skill3d.online.runner import OnlineRunConfig


def _panels(tmp_path):
    panels = {}
    for question_type, evidence_state in REACHABLE_V11_SLOTS:
        items = tuple(
            SimpleNamespace(
                episode=SimpleNamespace(
                    qa_id=(
                        f"{question_type}-{evidence_state}-outer-{index}"
                    ),
                    question_type=question_type,
                    split="outer_holdout",
                    scene_name=(
                        f"{question_type}-{evidence_state}-scene-{index}"
                    ),
                ),
                input_error=None,
            )
            for index in range(8)
        )
        artifacts = {
            item.episode.qa_id: str(
                tmp_path / f"{item.episode.qa_id}.json"
            )
            for item in items
        }
        key = slot_key(question_type, evidence_state)
        panels[key] = V11ValidationPanel(
            question_type=question_type,
            evidence_state=evidence_state,
            items=items,
            artifact_paths=artifacts,
            source_split="outer_holdout",
        )
    return panels


def test_outer_runner_only_writes_report_receipts(tmp_path, monkeypatch):
    panels = _panels(tmp_path)
    skills = {
        key: SimpleNamespace(
            question_type=panel.question_type,
            evidence_state=panel.evidence_state,
        )
        for key, panel in panels.items()
    }
    monkeypatch.setattr(
        outer_v11,
        "active_snapshot_provenance",
        lambda _root: ("snapshot-final", "f" * 64),
    )
    monkeypatch.setattr(
        outer_v11.V11OuterHoldoutEvaluator,
        "_snapshot",
        lambda _self, snapshot_id: (
            {
                "snapshot_id": snapshot_id,
                "manifest_hash": (
                    "a" * 64
                    if snapshot_id == "S0-v11-task-tools"
                    else "f" * 64
                ),
            },
            skills,
        ),
    )
    calls = []

    def fake_pair(items, **kwargs):
        assert all(
            item.episode.split == "outer_holdout" for item in items
        )
        calls.append(kwargs)
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True)
        (output / "manifest.json").write_text("{}", encoding="utf-8")
        (output / "paired_results.jsonl").write_text(
            "{}\n",
            encoding="utf-8",
        )
        (output / "summary.json").write_text(
            json.dumps({
                "status": "completed",
                "n_pairs": 8,
                "formal_result_eligible": False,
                "groups": {"all": {}},
            }),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        outer_v11,
        "run_skill_pair_evaluation_v11",
        fake_pair,
    )
    report = outer_v11.V11OuterHoldoutEvaluator(
        panels=panels,
        base_cfg=OnlineRunConfig(mode="real", seed=137),
        output_root=tmp_path / "outer",
        library_root=tmp_path / "library",
        final_snapshot_id="snapshot-final",
    ).run()

    assert len(calls) == 19
    assert report["n_slots"] == 19
    assert report["purpose"] == "report_only"
    assert report["feeds_revision"] is False
    assert report["can_promote"] is False
    assert (tmp_path / "outer" / "outer_report.json").is_file()
