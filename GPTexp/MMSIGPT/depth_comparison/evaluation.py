"""Host evaluation after both arms freeze; no model or credential imports are used."""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
import random
import time

from ...traces import digest, read, write
from ..prototype.direct import operator_lock
from ..prototype.evaluation import reference_answers
from ..prototype.protocol import LETTERS
from .runner import validate, validate_frozen


def summarize(pairs):
    total = len(pairs)
    direct = sum(pair["direct"]["correct"] for pair in pairs)
    depth = sum(pair["depth"]["correct"] for pair in pairs)
    direct_legal = sum(pair["direct"]["status"] == "answered" for pair in pairs)
    depth_legal = sum(pair["depth"]["status"] == "answered" for pair in pairs)
    gain = sum(not p["direct"]["correct"] and p["depth"]["correct"] for p in pairs)
    loss = sum(p["direct"]["correct"] and not p["depth"]["correct"] for p in pairs)
    discordant = gain + loss
    pvalue = (min(1.0, 2 * sum(math.comb(discordant, i)
                              for i in range(min(gain, loss) + 1)) / 2**discordant)
              if discordant else 1.0)
    deltas = [int(p["depth"]["correct"]) - int(p["direct"]["correct"]) for p in pairs]
    rng = random.Random(137)
    boot = sorted(sum(rng.choices(deltas, k=total)) / total for _ in range(10000)) if total else []
    complete = [p for p in pairs if all(
        p[arm]["status"] in ("answered", "invalid_answer") for arm in ("direct", "depth"))]
    return {
        "questions": total, "complete_pairs": len(complete),
        "direct_correct": direct, "depth_correct": depth,
        "direct_accuracy": direct / total if total else None,
        "depth_accuracy": depth / total if total else None,
        "direct_legal_answers": direct_legal, "depth_legal_answers": depth_legal,
        "direct_legal_answer_rate": direct_legal / total if total else None,
        "depth_legal_answer_rate": depth_legal / total if total else None,
        "delta": (depth - direct) / total if total else None,
        "wrong_to_correct": gain, "correct_to_wrong": loss,
        "paired_bootstrap_95_ci": [boot[249], boot[9749]] if boot else None,
        "mcnemar_exact_two_sided_p": pvalue,
        "direct_status_counts": dict(Counter(p["direct"]["status"] for p in pairs)),
        "depth_status_counts": dict(Counter(p["depth"]["status"] for p in pairs)),
        "complete_pair_delta": (
            sum(int(p["depth"]["correct"]) - int(p["direct"]["correct"]) for p in complete)
            / len(complete) if complete else None),
    }


def score(directory, references, expected_sha256):
    directory = Path(directory)
    with operator_lock(directory):
        # The reference file is opened only after all frozen evidence is verified.
        manifest = validate(directory)
        frozen = validate_frozen(directory, manifest)
        raw = Path(references).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("host_reference_hash_changed")
        unique = [{"sample_id": item["problem"]["sample_id"]} for item in manifest["items"]]
        answers = reference_answers(raw, unique)
        predictions = {(row["sample_id"], row["arm"]): row for row in frozen["predictions"]}
        pairs = []
        for item in manifest["items"]:
            sample = item["problem"]["sample_id"]
            answer = answers[sample]
            if answer not in LETTERS[:len(item["problem"]["options"])]:
                raise ValueError("invalid_reference_letter")
            pair = {
                "sample_id": sample, "task_type": item["problem"]["task_type"],
                "api_id": item["api_id"], "reference_answer": answer,
                "input_sha256": item["problem"]["input_sha256"],
            }
            for arm in ("direct", "depth"):
                prediction = predictions[sample, arm]
                pair[arm] = {**prediction, "correct":
                             prediction["status"] == "answered" and prediction["answer"] == answer}
            pairs.append(pair)
        tasks = sorted({pair["task_type"] for pair in pairs})
        result = {
            "schema": "mmsi-depth-paired-score/1", "run_identity": manifest["run_identity"],
            "freeze_identity": frozen["freeze_identity"], "reference_file_sha256": expected_sha256,
            "reference_access": "host_only_after_prediction_freeze",
            "denominator": f"all_{manifest['questions']}_selected_pairs_including_invalid_service_errors_and_unrun",
            "comparison": "original_rgb_vs_original_rgb_plus_predicted_depth_and_bounded_annotation",
            "overall": summarize(pairs),
            "by_task": {task: summarize([p for p in pairs if p["task_type"] == task]) for task in tasks},
            "by_api": {api: summarize([p for p in pairs if p["api_id"] == api])
                       for api in manifest["api_ids"]},
            "pairs": pairs, "scored_at": time.time(),
        }
        result["score_identity"] = digest(result)
        path = directory / "score.json"
        if path.exists():
            previous = read(path)
            if (digest({k: v for k, v in previous.items() if k != "score_identity"})
                    != previous["score_identity"] or any(
                        previous[key] != result[key] for key in result
                        if key not in ("scored_at", "score_identity"))):
                raise ValueError("existing_score_changed")
            return previous
        write(path, result, exclusive=True)
        return result
