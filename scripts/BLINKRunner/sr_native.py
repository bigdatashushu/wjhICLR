"""Bound SR detection fairly across subject and target before shared preparation."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class BalancedDetector:
    def __init__(self, detector, limit=2):
        if limit != 2:
            raise ValueError("sr_per_category_candidate_budget_must_be_two")
        self.detector, self.limit, self.trace = detector, limit, []
        self.identity = {
            **detector.identity,
            "sr_selection": {"policy": "top_confidence_per_category",
                             "max_per_category": 2, "max_total": 4,
                             "tie_break": "box_xyxy, source_label, original_index"},
        }

    def __call__(self, rgb, categories):
        rows = list(self.detector(rgb.copy(), categories))
        selected = []
        for category in categories:
            ranked = sorted(
                [(i, row) for i, row in enumerate(rows) if row["category"] == category],
                key=lambda pair: (-float(pair[1]["confidence"]),
                                  tuple(pair[1]["box_xyxy"]),
                                  pair[1].get("source_label", ""), pair[0]))
            selected.extend(i for i, _ in ranked[:self.limit])
        if len(set(selected)) != len(selected) or len(selected) > 4:
            raise ValueError("sr_bounded_candidate_selection_failed")
        self.trace.append({
            "categories": list(categories), "raw_detections": copy.deepcopy(rows),
            "selected_indices": selected,
            "discarded_indices": [i for i in range(len(rows)) if i not in selected],
            "policy": self.identity["sr_selection"],
        })
        return [rows[i] for i in selected]


def build_precomputer(config, *, seed=137):
    from run_small_image import build_precomputer as shared_builder
    from skill3d.small_image.precompute import _prepared, PreparedSmallImage
    if config.get("candidate_selection") != "top2_per_category":
        raise ValueError("sr_requires_balanced_frozen_candidate_policy")
    native = shared_builder(config, seed=seed)
    detector = BalancedDetector(native.detector)
    native.detector = detector
    native.model_identities["detector"] = detector.identity

    class Prepared:
        def process(self, problem, *, categories):
            detector.trace = []
            value = native.process(problem, categories=categories)
            fields = {k: getattr(value, k) for k in PreparedSmallImage.__dataclass_fields__
                      if k != "artifact_sha256"}
            fields["provenance"] = {
                **value.provenance, "detection_selection": copy.deepcopy(detector.trace)}
            return _prepared(**fields)
    return Prepared()
