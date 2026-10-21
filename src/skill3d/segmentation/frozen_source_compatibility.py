"""Explicit replay of reviewed r10 bindings; never reseal generation provenance."""

from __future__ import annotations

from skill3d.infra.m5_identity import SOURCE_REFS, content_sha256

R10_TRANSPORT_REVIEW = "r10-frozen-binding-transport-review/1.0"
# Reviewed diffs: Bailian request audit/build_payload with unchanged M5 defaults;
# question_binding adds only the opt-in validation path below.
REVIEWED_SOURCE_PAIRS = {
    "synthesis/bailian_client.py": (
        {"size": 8233, "sha256":
         "92e586d681bb987c61a54bcee9c9f646290d8970d4f04f50744f851b55cd3a58"},
        {"size": 9311, "sha256":
         "00f8dee9450a844d9b17f09122bf429f481d1e9222ce511273a129c5830ab143"},
    ),
    "segmentation/question_binding.py": (
        {"size": 52591, "sha256":
         "894baaf8cd3447fa19160f597c781db08c1a72197f5b32b072b68de6c458c57b"},
        {"size": 52991, "sha256":
         "8d163a595a0bb584b3a907c4fd51d8e6fc8b3b74962a9a335c04ce8bfb5c5bf6"},
    ),
}


def reviewed_source_compatibility(frozen, current, review_id: str) -> dict:
    """Accept exactly the reviewed source pairs, with every other source equal."""
    if review_id != R10_TRANSPORT_REVIEW:
        raise ValueError("unknown frozen M5 source compatibility review")
    if (not isinstance(frozen, dict) or not isinstance(current, dict)
            or set(frozen) != set(SOURCE_REFS) or set(current) != set(SOURCE_REFS)):
        raise ValueError("frozen M5 source compatibility requires complete sources")
    for name in SOURCE_REFS:
        old, new = frozen[name], current[name]
        pair = REVIEWED_SOURCE_PAIRS.get(name)
        if pair:
            if (old, new) != pair:
                raise ValueError(f"unreviewed frozen M5 implementation: {name}")
        elif old != new or (old is None and name != "synthesis/vllm_client.py"):
            raise ValueError(f"unreviewed frozen M5 implementation: {name}")
    return {
        "review_id": review_id,
        "frozen_sources_sha256": content_sha256(frozen),
        "runtime_sources_sha256": content_sha256(current),
        "reviewed_pairs": REVIEWED_SOURCE_PAIRS,
        "generation_resealed": False,
        "binding_recomputed": False,
    }
