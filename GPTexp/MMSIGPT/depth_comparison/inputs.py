"""Freeze label-free selection and build paired requests with bounded depth text."""
from __future__ import annotations

import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from ...traces import digest, write
from ..depth_prompt import build_annotation
from ..prototype.inputs import load_images, read_public_source, select
from ..prototype.protocol import messages, payload, prompt
from .depth import read_depth
from .pool import inference

TOKENIZER_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"


def tokenizer(path):
    import tiktoken
    from tiktoken_ext import openai_public
    from unittest.mock import patch

    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("tokenizer_identity_changed")
    ranks = {base64.b64decode(token): int(rank)
             for token, rank in (line.split() for line in raw.splitlines())}
    with patch.object(openai_public, "load_tiktoken_bpe", return_value=ranks):
        encoder = tiktoken.Encoding(**openai_public.o200k_base())
    return lambda text: len(encoder.encode(text))


def reviewed_ids(paths):
    ids = set()
    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get("sample_id"), str):
                ids.add(value["sample_id"])
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    for path in paths:
        visit(json.loads(Path(path).read_bytes()))
    return sorted(ids)


def selection(source, index_path, excluded, api_ids):
    rows, source_sha = read_public_source(source)
    index_raw = Path(index_path).read_bytes()
    index = json.loads(index_raw)
    if index["manifest_sha256"] != source_sha:
        raise ValueError("public_manifest_and_server_cache_index_disagree")
    records = {row["sample_id"]: row for row in index["records"]}
    if not api_ids or len(set(api_ids)) != len(api_ids):
        raise ValueError("requires_unique_api_slots")
    selected = select(rows, excluded=excluded, per_type=5, seed=137)
    # Hash order makes the 15/15 arm-order balance reproducible.
    ranked = sorted(range(len(selected)), key=lambda i: digest({
        "seed": 137, "purpose": "arm-order",
        "sample": selected[i]["problem"]["sample_id"],
    }))
    direct_first = set(ranked[:len(selected) // 2])
    items = []
    for i, row in enumerate(selected):
        problem = row["problem"]
        record = records[problem["sample_id"]]
        if (record["input_sha256"] != problem["input_sha256"]
                or record["images"] != problem["image_paths"]):
            raise ValueError("cache_index_public_input_disagrees")
        items.append({
            "index": i, "problem": problem, "images": row["images"],
            "prepared_directory": record["prepared_directory"],
            "artifact_sha256": record["artifact_sha256"],
            # Rotate slots across sampling rounds rather than binding one task
            # stratum permanently to one API when six slots are available.
            "api_id": api_ids[(i + i // 6) % len(api_ids)],
            "arm_order": ["direct", "depth"] if i in direct_first else ["depth", "direct"],
        })
    result = {
        "schema": "mmsi-depth-selection/1", "source_manifest_sha256": source_sha,
        "cache_index_sha256": hashlib.sha256(index_raw).hexdigest(),
        "excluded_sample_ids": sorted(set(excluded)), "seed": 137, "per_type": 5,
        "questions": len(items), "max_requests": 2 * len(items),
        "api_ids": list(api_ids), "items": items, "reference_answer_visible": False,
    }
    result["selection_identity"] = digest(result)
    return result


def depth_messages(problem, images, depths, annotation):
    content = []
    for label, (receipt, raw) in zip(annotation.rgb_labels, images):
        content.extend([
            {"type": "text", "text": label},
            {"type": "image_url", "image_url": {
                "url": "data:" + receipt["mime_type"] + ";base64,"
                       + base64.b64encode(raw).decode("ascii")}},
        ])
    for label, (raw, _) in zip(annotation.depth_labels, depths):
        content.extend([
            {"type": "text", "text": label},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(raw).decode("ascii")}},
        ])
    content.append({"type": "text", "text": annotation.note + "\n" + prompt(problem)})
    return [{"role": "user", "content": content}]


def prepare_item(directory, selected, count):
    images = load_images(selected["problem"], selected["images"])
    receipts = [row[0] for row in images]
    depths, provenance = read_depth(
        selected["prepared_directory"], selected["problem"], receipts,
        selected["artifact_sha256"])
    annotation = build_annotation([(row["width"], row["height"]) for row in receipts], count)
    original_prompt = prompt(selected["problem"])
    added_text_tokens = (
        sum(count(label) for label in (*annotation.rgb_labels, *annotation.depth_labels))
        + count(annotation.note + "\n" + original_prompt) - count(original_prompt)
    )
    if added_text_tokens > 160:
        raise ValueError("actual_transmitted_depth_text_exceeds_budget")
    depth_receipts = []
    for i, (raw, receipt) in enumerate(depths):
        relative = f"depth/{selected['index']:04d}-{i:02d}.png"
        path = Path(directory) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(raw)
        depth_receipts.append({**receipt, "path": relative, "view_index": i})
    bodies = {
        "direct": payload(messages(selected["problem"], images), inference(selected["api_id"])),
        "depth": payload(depth_messages(selected["problem"], images, depths, annotation),
                         inference(selected["api_id"])),
    }
    return {
        **selected, "images": receipts, "depth_images": depth_receipts,
        "depth_receipt": provenance, "annotation": asdict(annotation),
        "added_text_tokens": added_text_tokens,
        "prompt_sha256": digest(prompt(selected["problem"])),
        "payload_sha256": {arm: digest(body) for arm, body in bodies.items()},
    }


def request(directory, item, arm):
    images = load_images(item["problem"], item["images"])
    if arm == "direct":
        # Deliberately do not open prepared evidence, depth files or annotations.
        return payload(messages(item["problem"], images), inference(item["api_id"]))
    if arm != "depth":
        raise ValueError("unknown_arm")
    from ..depth_prompt import DepthAnnotation
    annotation = DepthAnnotation(**item["annotation"])
    depths = []
    for receipt in item["depth_images"]:
        raw = (Path(directory) / receipt["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
            raise ValueError("rendered_depth_bytes_changed")
        depths.append((raw, receipt))
    return payload(depth_messages(item["problem"], images, depths, annotation),
                   inference(item["api_id"]))
