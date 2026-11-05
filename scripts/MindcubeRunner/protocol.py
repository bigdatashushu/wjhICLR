"""Fresh requests: exact question, original RGB and paired Skill depth previews."""
from __future__ import annotations

import base64
import json

from .common import digest, skill_text
from .context_stage import format_feedback, guidance, response_reminder, stage_actions
from .data import digest_bytes
from .depth_input import DEPTH_LEGEND, depth_preview
from .observation_context import build_observation_context
from .tools import empty_counts, phase_for

INTERFACES = """Evidence interfaces:
Every successful call returns a dict r with result_id, source_tool, status, data,
schema_version and evidence_version. Use r["result_id"] in YieldObservations;
the values described below are inside r["data"], not top-level fields or attributes.
r["status"] == "ok" means execution success; inspect the inner data's reliability.
ctx.tools.view_candidates(view_index=-1, category="") -> r["data"] contains candidates, count, coverage.
view_index=-1 selects all views; category="" retains exposed categories, otherwise exact spelling.
Use r["data"]["candidates"] for the returned hints; inspect unfiltered labels if an exact category is unknown.
Queries read a fixed exposed catalog. An empty unfiltered all-view result is
limited coverage; repeating it or adding a category filter cannot reveal new hints.
The EXECUTED CANDIDATE QUERIES section records successful calls' actual view_index
and category, linked by result_id to TOOL RESULTS. Proposed or rejected code is not a call.
ctx.tools.view_depth_summary() -> r["data"]["views"] contains per-view availability, n_valid and quantiles.
ctx.tools.point_depth(view_index, x, y) -> r["data"] contains camera Z as depth, pixel_xy, n_valid and sample_stability.
ctx.tools.pair_point_distance(view_index, a_xy, b_xy) -> r["data"] contains distance, endpoints and sample_stability.
Unreliable measurements omit usable depth/distance; check data.status and reason_code.
Use actual original-image coordinates for measurement arguments. The host adds yielded
results to TOOL RESULTS in the next request; new hints are marked HINTS TO INSPECT
and new measurements are marked REVIEW REQUIRED NOW."""


def candidate_query_context(observations, queries, *, view_count):
    """Link only explicit host query records to successful delivered results."""
    candidates = {row["result_id"] for row in observations
                  if row["source_tool"] == "view_candidates" and row["status"] == "ok"}
    result, seen = [], set()
    for row in queries:
        if (type(row) is not dict
                or set(row) != {"result_id", "view_index", "category"}
                or type(row["result_id"]) is not str
                or row["result_id"] not in candidates or row["result_id"] in seen
                or type(row["view_index"]) is not int
                or not -1 <= row["view_index"] < view_count
                or type(row["category"]) is not str):
            raise ValueError("invalid_candidate_query_record")
        seen.add(row["result_id"])
        result.append(dict(row))
    return result


def messages(item, cache, *, arm, config, round_index=0, observations=(),
             errors=(), review_ids=(), hint_ids=(), counters=None, candidate_queries=()):
    counters = empty_counts() if counters is None else dict(counters)
    if not 0 <= round_index < config.request_limit(arm):
        raise ValueError("round_outside_episode")
    if len(cache.rgb) != len(item["image_keys"]) or not 2 <= len(cache.rgb) <= 4:
        raise ValueError("original_image_count_changed")
    if len(cache.mime) != len(cache.rgb) or len(cache.frames) != len(cache.rgb):
        raise ValueError("original_image_grid_changed")
    if arm == "skill" and set(cache.geometries) != set(range(len(cache.rgb))):
        raise ValueError("depth_preview_view_binding_changed")
    phase = "final" if arm == "direct" else phase_for(config, round_index)
    final = phase == "final"
    actions = stage_actions(final=final, errors=errors, review_ids=review_ids, hint_ids=hint_ids)
    observation_receipt = None
    text = f"Valid option letters: {', '.join(item['labels'])}."
    if arm == "skill":
        packet, observation_receipt = build_observation_context(
            observations, review_ids=review_ids, hint_ids=hint_ids)
        queries = candidate_query_context(
            observations, candidate_queries, view_count=len(cache.rgb))
        text += f"\nPhase {phase}; request {round_index + 1}/{config.request_limit(arm)}; "
        text += ("this is the final JSON slot; no program slots remain." if final else
                 f"program slots remaining including this one: "
                 f"{config.max_program_rounds - round_index}; one reserved final JSON slot.")
        text += "\nCURRENT TASKS " + json.dumps(actions, separators=(",", ":"))
        text += "\n" + guidance(actions, final=final)
        text += "\nSTATIC SKILL\n" + skill_text()
        text += "\nCumulative tool use: " + json.dumps(counters, separators=(",", ":"))
        if not final:
            text += "\n" + INTERFACES
        text += "\nDEPTH LEGEND " + DEPTH_LEGEND
        if observations:
            text += "\nTOOL RESULTS " + json.dumps(
                packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if queries:
            text += "\nEXECUTED CANDIDATE QUERIES " + json.dumps(
                queries, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if review_ids:
            text += "\nREVIEW REQUIRED NOW " + json.dumps(list(review_ids))
            text += "; locate every pixel in its original view and check target/surface binding."
        if hint_ids:
            text += "\nHINTS TO INSPECT " + json.dumps(list(hint_ids))
        if errors:
            text += "\nRECENT FEEDBACK " + json.dumps(
                format_feedback(errors, final=final), sort_keys=True, separators=(",", ":"))
    else:
        text += '\nTools are closed. Output only {"answer":"A"} with a valid current letter.'
    parts = [{"type": "text", "text": item["question"]}, {"type": "text", "text": text}]
    rgb_hashes = [digest_bytes(raw) for raw in cache.rgb]
    image_hashes, previews = [], []
    for index, (raw, mime, frame) in enumerate(zip(cache.rgb, cache.mime, cache.frames)):
        h, w = frame.shape[:2]
        parts.extend([
            {"type": "text", "text": f"Original Image {index + 1}: view_index={index}, "
             f"grid {w}x{h}, x right and y down."},
            {"type": "image_url", "image_url": {
                "url": f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")}},
        ])
        image_hashes.append(rgb_hashes[index])
        if arm == "skill":
            depth_raw, caption, metadata = depth_preview(
                cache.geometries[index], view_index=index, width=w, height=h)
            parts.extend([
                {"type": "text", "text": caption},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(depth_raw).decode("ascii")}},
            ])
            image_hashes.append(metadata["image_sha256"])
            previews.append(metadata)
    if arm == "skill":
        parts.append({"type": "text", "text": response_reminder(final=final)})
    text_chars = sum(len(p["text"]) for p in parts if p["type"] == "text")
    if text_chars > config.max_context_chars:
        raise ValueError("context_limit")
    result = [{"role": "user", "content": parts}]
    receipt = {
        "trace_schema": "mindcube-full-json-trace/1",
        "messages_sha256": digest(result), "phase": phase, "round_index": round_index,
        "text_chars": text_chars, "image_count": len(image_hashes),
        "image_sha256": image_hashes,
        "rgb_image_count": len(cache.rgb), "rgb_image_sha256": rgb_hashes,
        "depth_image_count": len(previews), "depth_previews": previews,
        "reviewed_measurement_ids": list(review_ids), "tool_counts": counters,
        "context_actions": actions, "hint_result_ids": list(hint_ids),
        "observation_context": observation_receipt,
        "tools_enabled": arm == "skill" and not final,
        "skill_sha256": digest_bytes(skill_text().encode()) if arm == "skill" else None,
    }
    if arm == "skill":
        receipt["candidate_query_history_sha256"] = digest(queries)
        receipt["candidate_query_count"] = len(queries)
    return result, receipt
