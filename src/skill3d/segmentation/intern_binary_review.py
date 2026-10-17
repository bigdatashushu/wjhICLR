"""Opt-in target-conditioned binary review; frozen independent quality evidence.

Only the target and the unchanged three images enter the semantic request.
No observation name is synthesized from a yes/no answer.
"""
from __future__ import annotations

import math
import re
import time
import json
from pathlib import Path

from skill3d.infra.m5_identity import content_sha256, file_identity
from .intern_decision_client import InternDecisionClient

BINARY_VERSION = "m5-target-binary-v1"
QUALITY_FIELDS = ("visibility", "mask_purity", "mask_completeness")
STATE = """The three images depict one candidate region, in this exact order:
1. full scene with an orange bounding rectangle;
2. original unmasked crop;
3. the same crop with a translucent turquoise segmentation overlay."""
P0 = 'Does the physical object primarily covered by the turquoise mask belong to the category "{target_category}"?'
P1 = P0 + """
Judge the object under the turquoise mask, not another object inside the orange box, crop, or scene. Use the unmasked crop to inspect appearance and the overlay to locate the region.
A partly visible object can still belong to the category.
Judge category membership only; mask completeness and purity are evaluated separately."""
P2 = P1 + """
Consider visual evidence both for and against the proposed category.
Do not favor yes because the category was supplied.
Preserve visible material and form-factor distinctions, while allowing ordinary synonyms and subtypes.
When identity is unclear, reflect the ambiguity in the probabilities rather than assuming the proposed category."""
PROMPTS = {"P0": P0, "P1": P1, "P2": P2}
PROTOCOL = {"version": BINARY_VERSION, "state": STATE, "prompts": PROMPTS,
            "type": "noul", "options": ["no", "yes"],
            "yes": "The physical object primarily covered by the turquoise mask belongs to the target category.",
            "no": "The physical object primarily covered by the turquoise mask does not belong to the target category."}
PROTOCOL_SHA256 = content_sha256(PROTOCOL)


def binary_request(images, target_category, prompt_version):
    if prompt_version not in PROMPTS:
        raise ValueError("unknown binary prompt version")
    if not isinstance(target_category, str) or not re.fullmatch(r"[a-z]+(?:[ -][a-z]+)*", target_category) or len(target_category) > 80:
        raise ValueError("target category must be a plain lower-case category name")
    if len(images) != 3:
        raise ValueError("binary review requires exactly three images")
    return {"state": STATE, "questions": {"target_match": {
        "type": "noul", "instructions": PROMPTS[prompt_version].format(target_category=target_category),
        "criteria": {"no": PROTOCOL["no"], "yes": PROTOCOL["yes"]}}}, "images": list(images)}


def validate_threshold(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not .5 <= value <= .99:
        raise ValueError("binary yes_threshold must be finite and between 0.50 and 0.99")
    return float(value)


def _binary_answer(result):
    if result.get("model") != "Intern-Decision-4B" or result.get("backend") != "hf":
        raise ValueError("binary model/backend mismatch")
    answers = result.get("answers", {})
    if set(answers) != {"target_match"}:
        raise ValueError("binary answer field mismatch")
    answer = answers["target_match"]
    probabilities = answer.get("probabilities", {})
    if set(probabilities) != {"no", "yes"} or any(
            type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
            for p in probabilities.values()) or abs(sum(probabilities.values()) - 1) > 1e-5:
        raise ValueError("invalid binary probabilities")
    best = min(probabilities, key=lambda key: (-probabilities[key], key))
    if (answer.get("type") != "noul" or answer.get("decision") != best
            or type(answer.get("noul")) not in (int, float)
            or not math.isclose(answer["noul"], probabilities["yes"], abs_tol=1e-7)
            or type(answer.get("confidence")) not in (int, float)
            or not math.isclose(answer["confidence"], probabilities[best], abs_tol=1e-7)):
        raise ValueError("binary decision/noul/confidence mismatch")
    return probabilities


def quality_passes(quality):
    return (quality.get("visibility") in {"clear", "partial"}
            and quality.get("mask_purity") == "clean"
            and quality.get("mask_completeness") in {"whole", "major_part"})


def quality_from_receipt(receipt, images, *, expected_model_sha256=None):
    """Validate an explicitly frozen old visual receipt before reusing its quality."""
    if receipt.get("status") != "observed":
        raise ValueError("frozen quality was not observed")
    body = receipt["response"]
    if not body.get("identity_sha256"):
        raise ValueError("frozen quality identity missing")
    if expected_model_sha256 and body["identity_sha256"] != expected_model_sha256:
        raise ValueError("frozen quality model identity changed")
    actual = [row["identity"]["sha256"] for row in images]
    if receipt["image_sha256"] != actual:
        raise ValueError("frozen quality image identity mismatch")
    observation = receipt["observation"]
    for field, allowed in [("visibility", {"clear", "partial", "insufficient"}),
                           ("mask_purity", {"clean", "mixed", "background", "unknown"}),
                           ("mask_completeness", {"whole", "major_part", "fragment", "unknown"})]:
        if observation.get(field) not in allowed or body["result"]["answers"][field]["decision"] != observation[field]:
            raise ValueError("frozen quality field mismatch")
    return {field: observation[field] for field in QUALITY_FIELDS}


class InternBinaryClient(InternDecisionClient):
    provider = "intern-decision-binary-local"
    target_binary = True

    def __init__(self, endpoint, *, prompt_version="P1", yes_threshold=None,
                 quality_manifest=None, quality_manifest_sha256="", **kwargs):
        super().__init__(endpoint, **kwargs)
        if prompt_version not in PROMPTS:
            raise ValueError("unknown binary prompt version")
        self.prompt_version = prompt_version
        self.yes_threshold = None if yes_threshold is None else validate_threshold(yes_threshold)
        self.quality_manifest = Path(quality_manifest) if quality_manifest else None
        self.quality_manifest_sha256 = quality_manifest_sha256
        self._quality = None

    def audit_identity(self, *, seed=None):
        return {**super().audit_identity(seed=seed), "provider": self.provider,
                "protocol_sha256": PROTOCOL_SHA256, "prompt_version": self.prompt_version,
                "yes_threshold": self.yes_threshold, "quality_manifest_sha256": self.quality_manifest_sha256}

    def review_binary(self, images, target_category):
        identity = self.identity()
        request = binary_request([r["url"] for r in images], target_category, self.prompt_version)
        sha = content_sha256(request)
        body = self._call("/decisions", {"request_sha256": sha, "request": request})
        if body.get("request_sha256") != sha or body.get("identity_sha256") != content_sha256(identity):
            raise ValueError("binary request/model identity mismatch")
        if body["result"].get("calibration") != identity["calibration"]:
            raise ValueError("binary calibration mismatch")
        probs = _binary_answer(body["result"])
        raw_probs = _binary_answer(body["uncalibrated_result"])
        if body["uncalibrated_result"].get("calibration", {}).get("temperature") != 1:
            raise ValueError("binary uncalibrated result is not T=1")
        vision = body.get("vision_evidence", {})
        if (vision.get("image_count") != 3 or vision.get("image_token_count", 0) <= 0
                or not vision.get("pixel_values_shape") or len(vision.get("image_grid_thw", [])) != 3
                or len(vision.get("decision_logit_positions", [])) != 1):
            raise ValueError("binary image or decision token evidence missing")
        return {"p_yes": probs["yes"], "p_no": probs["no"],
                "raw_p_yes": raw_probs["yes"], "raw_p_no": raw_probs["no"],
                "decision": body["result"]["answers"]["target_match"]["decision"],
                "target_category": target_category, "prompt_version": self.prompt_version,
                "protocol_sha256": PROTOCOL_SHA256}, body, sha

    def _frozen_quality(self, candidate_id, images):
        if not self.quality_manifest or not self.quality_manifest_sha256:
            raise ValueError("binary M5 requires an explicitly pinned frozen quality manifest")
        if self._quality is None:
            if file_identity(self.quality_manifest)["sha256"] != self.quality_manifest_sha256:
                raise ValueError("quality manifest SHA256 mismatch")
            self._quality = json.loads(self.quality_manifest.read_text())
            if self._quality.get("version") != "m5-frozen-quality-manifest-v1":
                raise ValueError("quality manifest version mismatch")
        entry = self._quality["candidates"][candidate_id]
        if entry.get("candidate_id") != candidate_id:
            raise ValueError("quality candidate identity mismatch")
        source_identity = self._quality.get("source_model_identity")
        source_sha = self._quality.get("source_model_identity_sha256")
        if not source_identity or content_sha256(source_identity) != source_sha:
            raise ValueError("frozen quality source identity mismatch")
        quality = quality_from_receipt(entry, images, expected_model_sha256=source_sha)
        return quality, {"manifest_sha256": self.quality_manifest_sha256,
                         "entry_sha256": content_sha256(entry), "entry": entry}

    def observe_candidate(self, candidate, frame, mask, *, seed, min_pixels, max_pixels, out_dir=None):
        from .mask_semantic_review import candidate_images, _write
        cid = str(candidate["candidate_id"])
        receipt = {"version": BINARY_VERSION, "candidate_id": cid,
                   "provider": self.provider, "status": "unavailable"}
        start = time.perf_counter()
        try:
            threshold = validate_threshold(self.yes_threshold)
            images = candidate_images(frame, candidate["bbox"], mask)
            quality, quality_source = self._frozen_quality(cid, images)
            receipt["request_sent"] = True
            binary, body, sha = self.review_binary(images, candidate["category"])
            decision = {**binary, "threshold": threshold, "semantic_pass": binary["p_yes"] >= threshold}
            receipt.update(status="observed", observation=quality, binary_review=decision,
                frozen_quality=quality_source, raw_response=body, request_sha256=sha,
                request_identity={"protocol_sha256": PROTOCOL_SHA256, "model": self.identity(),
                    "prompt_version": self.prompt_version, "target_category": candidate["category"],
                    "images": [r["identity"] for r in images],
                    "quality_manifest_sha256": self.quality_manifest_sha256},
                observation_sha256=content_sha256(quality), model_identity_sha256=content_sha256(self.identity()),
                binary_review_sha256=content_sha256(decision),
                normalization_sources={"observation": "only quality from exact-image frozen receipt",
                    "observed_object": "absent; binary answer does not name an independently observed class"})
        except Exception as exc:
            receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["elapsed_s"] = time.perf_counter() - start
        if out_dir is not None:
            _write(Path(out_dir) / "visual" / f"{cid}.json", receipt)
        return receipt


def binary_relation(receipt):
    binary = receipt.get("binary_review", {})
    if receipt.get("status") != "observed" or not binary:
        return {"status": "unavailable", "relation": "uncertain"}
    # Below-threshold means not established, never a certified physical absence.
    return {"status": "reviewed", "relation": "match" if binary["semantic_pass"] else "uncertain",
            "request_sha256": receipt["request_sha256"], "provider": "binary-threshold",
            "p_yes": binary["p_yes"], "threshold": binary["threshold"], "version": BINARY_VERSION}


def threshold_search(samples):
    """Coarse brackets then bisection; all calculations use already saved probabilities.

Rows: id, label=true/false/uncertain, p_yes. Repeats are reduced pessimistically.
No threshold below .50 or above .99 is considered. A useless all-reject gate is
reported separately from a useful zero-false-positive operating point.
"""
    groups = {}
    for row in samples:
        if row["label"] not in {"true", "false", "uncertain"}:
            raise ValueError("invalid threshold reference label")
        p = row["p_yes"]
        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("invalid threshold input probability")
        old = groups.setdefault(row["id"], {"label": row["label"], "values": []})
        if old["label"] != row["label"]:
            raise ValueError("inconsistent repeat labels")
        old["values"].append(p)
    negatives = [max(r["values"]) for r in groups.values() if r["label"] == "false"]
    positives = [min(r["values"]) for r in groups.values() if r["label"] == "true"]
    if not negatives or not positives:
        raise ValueError("threshold selection needs positive and negative references")
    trace = []

    def measure(t, stage):
        fp, tp = sum(p >= t for p in negatives), sum(p >= t for p in positives)
        trace.append({"threshold": t, "stage": stage, "false_admitted": fp,
                      "true_retained": tp, "n_negative": len(negatives), "n_positive": len(positives)})
        return fp == 0

    lower, upper = .5, None
    for t in [.5, .7, .9, .95, .99]:
        if measure(t, "coarse"):
            upper = t
            break
        lower = t
    if upper is None:
        return {"threshold": None, "useful": False, "reason": "false positives remain at .99", "trace": trace}
    if upper != .5:
        while upper - lower > .01:
            middle = (lower + upper) / 2
            if measure(middle, "bisection"):
                upper = middle
            else:
                lower = middle
    selected = min(.99, math.ceil((upper - 1e-12) * 100) / 100)
    measure(selected, "selected")
    retained = trace[-1]["true_retained"]
    return {"threshold": selected, "useful": retained > 0,
            "reason": "zero development false positives" if retained else "all positives rejected",
            "trace": trace, "positive_retained": retained,
            "n_positive": len(positives), "n_negative": len(negatives)}
