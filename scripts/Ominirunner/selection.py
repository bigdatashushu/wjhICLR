"""Fresh question selection from the pinned public bindings, without labels."""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

from .common import BINDINGS_SHA, IDS, read, sha
from .data import PUBLIC

SCHEMA = "omni-fresh-public-selection/1"
CONFIGURABLE_SCHEMA = "omni-fresh-public-selection/2"
SEED = "omni-fresh15:20261103:137"


def from_bindings(document, count=15, *, seed=SEED, excluded_ids=IDS):
    if type(count) is not int or count not in (15, 20):
        raise ValueError("fresh_selection_requires_fifteen_or_twenty")
    if not isinstance(seed, str) or not 1 <= len(seed) <= 100:
        raise ValueError("selection_seed_invalid")
    if (not isinstance(excluded_ids, (list, tuple)) or len(set(excluded_ids)) != len(excluded_ids)
            or any(not isinstance(i, str) or not i.isdecimal() for i in excluded_ids)
            or not set(IDS) <= set(excluded_ids)):
        raise ValueError("selection_exclusions_invalid")
    rows = document["questions"]
    if any(set(row) != PUBLIC for row in rows):
        raise ValueError("selection_public_fields_changed")
    bindings = {row["sample_id"]: row for row in rows}
    if len(bindings) != 501 or len(rows) != 501 or not set(IDS) <= set(bindings):
        raise ValueError("selection_catalog_changed")
    if not set(excluded_ids) <= set(bindings):
        raise ValueError("selection_exclusions_invalid")
    old_images = {key for i in excluded_ids for key in bindings[i]["image_keys"]}
    eligible = [row for row in rows if row["sample_id"] not in excluded_ids
                and not old_images.intersection(row["image_keys"])]
    eligible.sort(key=lambda row: (
        hashlib.sha256((seed + ":" + row["sample_id"]).encode()).hexdigest(),
        row["sample_id"]))
    chosen = eligible[:count]
    if len(chosen) != count:
        raise ValueError("insufficient_fresh_questions")
    return {
        "schema": SCHEMA if count == 15 and seed == SEED and tuple(excluded_ids) == IDS else CONFIGURABLE_SCHEMA,
        "seed": seed, "count": count,
        "bindings_sha256": BINDINGS_SHA, "excluded_sample_ids": list(excluded_ids),
        "excluded_image_keys": sorted(old_images), "eligible_questions": len(eligible),
        "sample_ids": [row["sample_id"] for row in chosen],
        "task_types": dict(sorted(Counter(row["task_type"] for row in chosen).items())),
        "method": "sha256_seed_and_sample_id_ascending",
        "reference_answer_visible": False, "real_labels_available": False,
        "selection_uses_outcomes": False, "direct_reuse": False,
    }


def choose(cache_root, count=15, *, seed=SEED, excluded_ids=IDS):
    path = Path(cache_root) / "bindings.json"
    if sha(path) != BINDINGS_SHA:
        raise ValueError("selection_bindings_identity")
    return from_bindings(read(path), count, seed=seed, excluded_ids=excluded_ids)


def validate(cache_root, receipt):
    if receipt.get("schema") == SCHEMA:
        expected = choose(cache_root)
    elif receipt.get("schema") == CONFIGURABLE_SCHEMA:
        expected = choose(cache_root, receipt["count"], seed=receipt["seed"],
                          excluded_ids=receipt["excluded_sample_ids"])
    else:
        raise ValueError("selection_schema_changed")
    if expected != receipt:
        raise ValueError("frozen_selection_changed")
    return tuple(receipt["sample_ids"])
