"""Standard-library client for the opt-in native Intern decision service."""
from __future__ import annotations
import json
import math
import re
import time
import urllib.request
from urllib.parse import urlsplit
from skill3d.infra.m5_identity import content_sha256
from .intern_decision_protocol import (
    PROTOCOL_SHA256, PROTOCOL_VERSION, QUESTIONS, visual_request,
    fixed_relation, RELATION_IDENTITY, RELATION_VERSION,
)

class InternDecisionClient:
    provider = "intern-decision-local"
    def __init__(self, endpoint, *, timeout_s=60, expected_identity_sha256="", opener=None):
        parsed = urlsplit(endpoint)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Intern endpoint must be a credential-free loopback HTTP URL")
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 180:
            raise ValueError("Intern timeout must be finite and <=180 seconds")
        if expected_identity_sha256 and not re.fullmatch(r"[0-9a-f]{64}", expected_identity_sha256):
            raise ValueError("expected Intern identity must be a SHA256")
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = timeout_s
        self.expected_identity_sha256 = expected_identity_sha256
        self._opener = opener or urllib.request.urlopen
        self._identity = None

    def _call(self, path, body=None):
        request = urllib.request.Request(self.endpoint + path,
            data=None if body is None else json.dumps(body, allow_nan=False).encode(),
            headers={"Content-Type": "application/json"})
        with self._opener(request, timeout=self.timeout_s) as response:
            payload = response.read(4 * 1024 * 1024 + 1)
        if len(payload) > 4 * 1024 * 1024:
            raise ValueError("Intern response exceeds limit")
        return json.loads(payload)

    def identity(self):
        if self._identity is None:
            health = self._call("/health")
            identity = health["identity"]
            digest = content_sha256(identity)
            if (health.get("identity_sha256") != digest
                    or self.expected_identity_sha256 and digest != self.expected_identity_sha256
                    or identity.get("model") != "internlm/Intern-Decision-4B"
                    or identity.get("backend") != "native-hf-model-package"
                    or identity.get("dtype") != "bfloat16"
                    or not identity.get("model_revision") or not identity.get("files")
                    or not identity.get("implementation_sha256")
                    or type(identity.get("calibration", {}).get("temperature")) not in (int, float)
                    or not math.isfinite(identity["calibration"]["temperature"])
                    or identity["calibration"]["temperature"] <= 0):
                raise ValueError("Intern health/model identity mismatch")
            self._identity = identity
        return self._identity

    def audit_identity(self, *, seed=None):
        return {"provider": self.provider, "protocol_sha256": PROTOCOL_SHA256,
                "endpoint_sha256": content_sha256(self.endpoint), "cache_eligible": False,
                "expected_identity_sha256": self.expected_identity_sha256}

    def observe(self, images):
        identity = self.identity()
        request = visual_request([row["url"] for row in images])
        request_sha = content_sha256(request)
        body = self._call("/decisions", {"request_sha256": request_sha, "request": request})
        if (body.get("request_sha256") != request_sha
                or body.get("identity_sha256") != content_sha256(identity)):
            raise ValueError("Intern response request/model ID mismatch")
        result = body["result"]
        if result.get("model") != "Intern-Decision-4B" or result.get("backend") != "hf":
            raise ValueError("Intern native result model/backend mismatch")
        if result.get("calibration") != identity["calibration"]:
            raise ValueError("Intern calibration changed")
        answers = result["answers"]
        if set(answers) != set(QUESTIONS):
            raise ValueError("Intern decision fields mismatch")
        observation = {}
        for field, question in QUESTIONS.items():
            answer = answers[field]
            probs = answer["probabilities"]
            if set(probs) != set(question["criteria"]):
                raise ValueError("Intern option table mismatch")
            if any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1
                   for p in probs.values()) or abs(sum(probs.values()) - 1) > 1e-5:
                raise ValueError("Intern probability distribution invalid")
            best = min(probs, key=lambda key: (-probs[key], key))
            if (answer.get("type") != "choice" or answer.get("choice") != best
                    or answer.get("decision") != best or not math.isclose(
                        answer.get("confidence", -1), probs[best], abs_tol=1e-6)):
                raise ValueError("Intern selected decision/confidence mismatch")
            observation[field] = best
        vision = body["vision_evidence"]
        if (vision.get("image_count") != 3 or vision.get("image_token_count", 0) <= 0
                or not vision.get("pixel_values_shape") or len(vision.get("image_grid_thw", [])) != 3):
            raise ValueError("Intern did not encode all three images")
        return observation, body, request_sha

    def observe_candidate(self, candidate, frame, mask, *, seed, min_pixels, max_pixels, out_dir=None):
        from .mask_semantic_review import candidate_images, _write
        from pathlib import Path
        receipt = {"version": PROTOCOL_VERSION, "candidate_id": str(candidate["candidate_id"]),
                   "provider": self.provider, "status": "unavailable"}
        start = time.perf_counter()
        try:
            images = candidate_images(frame, candidate["bbox"], mask)
            receipt["request_sent"] = True
            observation, body, request_sha = self.observe(images)
            receipt.update(status="observed", observation=observation,
                raw_response=body, request_sha256=request_sha,
                request_identity={"protocol_sha256": PROTOCOL_SHA256,
                    "model": self.identity(), "images": [r["identity"] for r in images],
                    "rendering": "candidate_images:jpeg95:margin0.20:turquoise0.58"},
                observation_sha256=content_sha256(observation),
                model_identity_sha256=content_sha256(self.identity()),
                normalization_sources={"observed_object": "fixed option label",
                    "alternate_object": "absent; no free-form alternate generated"})
        except Exception as exc:
            receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["elapsed_s"] = time.perf_counter() - start
        if out_dir is not None:
            _write(Path(out_dir) / "visual" / f"{candidate['candidate_id']}.json", receipt)
        return receipt


def compare_fixed_categories(rows, *, out_dir=None):
    from .mask_semantic_review import _write
    from pathlib import Path
    pairs = [{"id": str(r["candidate_id"]), "observed_object": r["observation"]["observed_object"],
              "target_category": r["category"]} for r in rows]
    identity = {"rules": RELATION_IDENTITY, "pairs": pairs}
    sha = content_sha256(identity)
    results = {r["id"]: {"status": "reviewed", "relation": fixed_relation(
        r["observed_object"], r["target_category"]), "request_sha256": sha,
        "provider": "frozen-domain-rules", "version": RELATION_VERSION} for r in pairs}
    if out_dir is not None:
        _write(Path(out_dir) / "relation" / "fixed.json", {"identity": identity,
            "request_sha256": sha, "results": results, "actual_requests": 0})
    return results, {"actual_requests": 0, "n_batches": 0}


def review_config(config=None):
    config = dict(config or {})
    if config.get("provider", "qwen") not in {"qwen", "intern-decision"}:
        raise ValueError("unsupported M5 visual provider")
    if config.get("relation", "qwen") not in {"qwen", "frozen-rules"}:
        raise ValueError("unsupported M5 relation provider")
    allowed = {"provider", "endpoint", "timeout_s", "expected_identity_sha256", "relation",
               "mode", "prompt_version", "yes_threshold", "quality_manifest", "quality_manifest_sha256"}
    if config.get("mode", "observation") not in {"observation", "binary"}:
        raise ValueError("unsupported Intern visual mode")
    if config.get("mode") == "binary" and config.get("provider") != "intern-decision":
        raise ValueError("binary mode requires Intern provider")
    if set(config) - allowed:
        raise ValueError("unknown M5 review configuration")
    return config


def make_review_client(config=None):
    config = review_config(config)
    if config.get("provider", "qwen") == "qwen":
        return None
    if config.get("mode") == "binary":
        from .intern_binary_review import InternBinaryClient, validate_threshold
        validate_threshold(config.get("yes_threshold"))
        return InternBinaryClient(config.get("endpoint", "http://127.0.0.1:18445"),
            timeout_s=config.get("timeout_s", 60),
            expected_identity_sha256=config.get("expected_identity_sha256", ""),
            prompt_version=config.get("prompt_version"), yes_threshold=config["yes_threshold"],
            quality_manifest=config.get("quality_manifest"),
            quality_manifest_sha256=config.get("quality_manifest_sha256", ""))
    return InternDecisionClient(config.get("endpoint", "http://127.0.0.1:18445"),
        timeout_s=config.get("timeout_s", 60),
        expected_identity_sha256=config.get("expected_identity_sha256", ""))
