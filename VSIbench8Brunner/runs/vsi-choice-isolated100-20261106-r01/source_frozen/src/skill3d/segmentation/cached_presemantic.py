"""Replay the sealed parser and detector ledgers before new SAM2 inference."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

from skill3d.infra.m5_identity import content_sha256


class CachedPresemantic:
    def __init__(self, prepared, reference, *, allow_unavailable_parser=False):
        self.stats = copy.deepcopy(prepared.stats)
        self.reference = copy.deepcopy(reference)
        self.allow_unavailable_parser = allow_unavailable_parser
        if allow_unavailable_parser:
            parser = self.stats.get("object_parse") or {}
            if (parser.get("status") != "unavailable" or parser.get("objects")
                    or getattr(prepared, "objects", None)
                    or getattr(prepared, "materialized", False)
                    or getattr(prepared, "prepared_measurements", None)
                    or self.stats.get("materialized")
                    or self.stats.get("candidate_receipts")
                    or self.stats.get("detection_receipts")
                    or (self.stats.get("review_selection") or {}).get("selected_candidate_ids")
                    or self.stats.get("binding_error") or self.stats.get("detector_fault")):
                raise ValueError("unavailable cached parser contradicts published evidence")
        self.receipt = {
            "schema": "vsi-cached-presemantic/1",
            "source_reference_sha256": content_sha256(reference),
            "source_generation_sha256": self.stats["generation"]["sha256"],
            "reference_answer_visible": False,
            "new_parser_requests": 0, "new_detector_requests": 0,
        }

    def validate(self, generation):
        old = self.stats["generation"]
        if old["sha256"] != content_sha256(old["identity"]):
            raise ValueError("cached generation digest differs")
        current = generation["identity"]
        for name in ("question", "frames", "geometry", "frame_set_hash",
                     "reconstruction_identity", "grid_transform"):
            if old["identity"][name] != current[name]:
                raise ValueError(f"cached presemantic {name} differs")
        sam = current.get("sam2") or {}
        original_sam = old["identity"].get("sam2") or {}
        for name in ("checkpoint", "config", "settings"):
            if sam.get(name) != original_sam.get(name):
                raise ValueError(f"SAM2 {name} differs from original preparation")
        self.receipt["sam2_runtime_identical"] = sam == original_sam
        self.receipt["original_sam2_sha256"] = content_sha256(original_sam)
        self.receipt["current_sam2_sha256"] = content_sha256(sam)

    def parser(self, inputs):
        from .question_binding import validate_parse
        original = self.stats["object_parse"]
        if original["status"] == "unavailable" and self.allow_unavailable_parser:
            if inputs != self.stats["generation"]["identity"]["question"]:
                raise ValueError("cached unavailable parser question differs")
            receipt = copy.deepcopy(original)
            receipt.update(cache_hit=True, replay_source=self.receipt,
                           source_unavailable_preserved=True)
            return [], receipt
        if original["status"] != "parsed":
            raise ValueError("cached parser was not successful")
        rows = validate_parse(json.dumps({"objects": original["objects"]}), inputs)
        receipt = copy.deepcopy(original)
        receipt.update(cache_hit=True, replay_source=self.receipt)
        return rows, receipt

    def detections(self, categories, readable, requirements):
        from .question_binding import _nms
        selected, receipts, covered = [], [], set()
        for receipt in self.stats.get("detection_receipts", []):
            category, slot = receipt["category"], receipt["frame_slot"]
            if category not in categories or slot not in readable or receipt.get("error"):
                raise ValueError("cached detector scan is unavailable or out of scope")
            if (category, slot) in covered:
                raise ValueError("duplicate detector ledger")
            covered.add((category, slot))
            detections = [SimpleNamespace(
                label=row["source_label"], bbox_xyxy=row["bbox"],
                confidence=row["score"]) for row in receipt["candidates"]]
            rows, kept = _nms(detections, category, slot, receipt["request_sha256"])
            if rows != receipt["candidates"]:
                raise ValueError("cached NMS ledger does not replay")
            selected.extend(kept)
            receipts.append(copy.deepcopy(receipt))
        if covered != {(cat, slot) for cat in categories for slot in readable}:
            raise ValueError("cached detector scan is incomplete")
        old_ids = [row["candidate_id"] for row in self.stats["candidate_receipts"]]
        if [row["candidate_id"] for row in selected] != old_ids:
            raise ValueError("cached post-NMS candidate order differs")
        for requirement in requirements:
            requirement["scanned_slots"] = list(readable)
            requirement["scan_complete"] = True
        self.receipt.update(
            nms_verified=True, nms_iou=0.7, candidate_count=len(selected),
            candidate_sha256=content_sha256(selected))
        return selected, receipts

    def select(self, observations):
        by_id = {row["candidate"]["candidate_id"]: row for row in observations}
        ids = self.stats["review_selection"]["selected_candidate_ids"]
        if len(ids) != len(set(ids)) or any(cid not in by_id for cid in ids):
            raise ValueError("regenerated masks omit an original review candidate")
        self.receipt.update(review_candidate_count=len(ids),
                            review_candidate_sha256=content_sha256(ids))
        return [by_id[cid] for cid in ids]

    def compare_images(self, candidate_id, visual):
        old = next(row for row in self.stats["candidate_receipts"]
                   if row["candidate_id"] == candidate_id)["semantic_visual"]
        expected = old["request_identity"]["images"]
        actual = visual.get("request_identity", {}).get("images", [])
        return {
            "candidate_id": candidate_id, "original": expected, "regenerated": actual,
            "all_three_image_hashes_equal": len(actual) == 3 and actual == expected,
            "original_p_yes": old.get("binary_review", {}).get("p_yes"),
            "original_pass": old.get("binary_review", {}).get("semantic_pass"),
        }
