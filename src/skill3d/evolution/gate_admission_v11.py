"""FitSet admission conditions; program errors remain descriptive metrics."""
from __future__ import annotations

from collections.abc import Mapping

from skill3d.evaluation.fitset_v11 import content_sha256

GATE_ADMISSION_VERSION = "fitset-gate-admission/2"
PROGRAM_ERROR_METRIC_RULE = "any_round_error_code_episode_count"


def gate_arm_conditions(parent: Mapping, candidate: Mapping) -> dict[str, bool]:
    return {
        "strict_score_improvement": candidate["mean_score"] > parent["mean_score"],
        "runtime_errors_nonincreasing": candidate["errors"] <= parent["errors"],
        "legal_rate_nondecreasing": candidate["legal_rate"] >= parent["legal_rate"],
        "paired_input_identity": parent["shared_input_hashes"] == candidate["shared_input_hashes"],
        "returned_model_identity": parent["returned_models"] == candidate["returned_models"],
        "returned_backend_identity": (
            parent["backend_fingerprints"] == candidate["backend_fingerprints"]),
        **{key: all(arm[key] for arm in (parent, candidate))
           for key in ("trusted", "delivered", "service_ok", "seed_observed")},
    }


def gate_admission_decision(scored: Mapping) -> dict:
    """Apply the current rule to scored arms and seal its own decision identity."""
    body = {key: value for key, value in scored.items()
            if key not in ("identity_sha256", "admission_program_error_rule")}
    if body.get("arms"):
        body["conditions"] = {
            str(pair["seed"]): all(gate_arm_conditions(
                pair["parent"], pair["candidate"]).values())
            for pair in body["arms"]
        }
        body["passed"] = all(body["conditions"].values())
    else:
        body["passed"] = False
    body["admission_rule_version"] = GATE_ADMISSION_VERSION
    body["program_error_metric_rule"] = PROGRAM_ERROR_METRIC_RULE
    return {**body, "identity_sha256": content_sha256(body)}
