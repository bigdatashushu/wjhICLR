"""Count saved context and billed tokens offline, without exposing payloads."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from target_detection_twenty.control import OUT, VERIFY, attempted, load, save
from target_detection_r05.perception import normalized
from target_detection_r05.sources import canonical, digest, file_digest

TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def summarize(folder):
    rows = [json.loads(line) for line in (folder / "trajectory.jsonl").read_text(
        encoding="utf-8").splitlines()]
    requests = [row for row in rows if row["type"] == "model_request"]
    calls, stages = [], {}
    totals = {key: 0 for key in TOKEN_KEYS}
    models, finish_reasons = Counter(), Counter()
    for row in requests:
        payload = load(folder / row["request_file"])
        assert digest(canonical(payload)) == row["payload_sha256"]
        stage = "official_plan"
        system = "\n".join(message["content"] for message in payload["messages"]
                           if message["role"] == "system" and isinstance(message["content"], str))
        for marker, name in (
            ("EXTRACT TARGETS AND DETECT", "extract_and_detect"),
            ("DETECT THE CURRENT VIEW", "detect"),
            ("INSPECT CROPS", "inspect"),
            ("REVIEW CROPS", "review"),
            ("FULL ACTION PLAN", "action_plan"),
        ):
            if "CURRENT PHASE: " + marker in system:
                stage = name
                break
        image_count, text_characters = 0, 0
        for message in payload["messages"]:
            if isinstance(message["content"], str):
                text_characters += len(message["content"])
            else:
                for part in message["content"]:
                    image_count += int(part["type"] == "image_url")
                    text_characters += len(part.get("text", ""))
        response_path = folder / f'response-{row["request_index"]:03d}.json'
        response = load(response_path) if response_path.exists() else {}
        usage = response.get("usage", {})
        for key in TOKEN_KEYS:
            totals[key] += usage.get(key, 0)
        if response:
            models[response.get("model")] += 1
            finish_reasons[response.get("finish_reason")] += 1
        detail = {
            "request_index": row["request_index"],
            "observation_serial": row.get("observation_serial"),
            "environment_step": row.get("environment_step"),
            "phase": stage, "messages": len(payload["messages"]),
            "images": image_count, "text_characters": text_characters,
            "usage": usage, "response_available": bool(response),
        }
        calls.append(detail)
        aggregate = stages.setdefault(stage, {
            "requests": 0, **{key: 0 for key in TOKEN_KEYS}})
        aggregate["requests"] += 1
        for key in TOKEN_KEYS:
            aggregate[key] += usage.get(key, 0)
    result = load(folder / "result.json")
    errors = [{"type": row["type"], "code": row.get("code")}
              for row in rows if row["type"] in ("planner_failure", "failure")]
    first_response_path = folder / "response-001.json"
    first_targets = []
    if first_response_path.exists():
        try:
            value = json.loads(load(first_response_path)["text"])
            first_targets = value.get("targets", []) if isinstance(value, dict) else []
        except (ValueError, TypeError):
            pass
    instruction = result["metrics"].get("instruction") or ""
    bindings = [{"id": target.get("id"), "category": target.get("category"),
                 "source_phrase": target.get("source_phrase"),
                 "matches_public_instruction": isinstance(target.get("source_phrase"), str)
                 and normalized(target["source_phrase"]) in normalized(instruction)}
                for target in first_targets if isinstance(target, dict)]
    last_value = {}
    if calls:
        last_path = folder / f'response-{calls[-1]["request_index"]:03d}.json'
        if last_path.exists():
            try:
                parsed = json.loads(load(last_path)["text"])
                last_value = parsed if isinstance(parsed, dict) else {}
            except (ValueError, TypeError):
                pass
    detections = [row for row in rows if row["type"] == "perception_tool_result"
                  and row["tool_request"]["name"] == "detect_targets"]
    reviews = [row for row in rows if row["type"] == "perception_action_review"]
    return {
        "requests": len(calls), "responses_with_usage": sum(bool(call["usage"]) for call in calls),
        "total_usage": totals, "by_phase": stages,
        "maximum_prompt_tokens": max((call["usage"].get("prompt_tokens", 0) for call in calls), default=0),
        "maximum_images_in_request": max((call["images"] for call in calls), default=0),
        "maximum_text_characters": max((call["text_characters"] for call in calls), default=0),
        "returned_models": dict(models), "finish_reasons": dict(finish_reasons),
        "host_elapsed_seconds": result["host_elapsed_seconds"],
        "num_invalid_actions": result["metrics"]["num_invalid_actions"],
        "task_progress": result["metrics"].get("task_progress"),
        "error": result["error"], "failure_events": errors,
        "public_instruction": instruction, "first_target_source_bindings": bindings,
        "last_response_root_keys": sorted(last_value),
        "last_response_contains_action_plan": "executable_plan" in last_value
        or "executable_plan" in last_value.get("properties", {})
        if isinstance(last_value.get("properties", {}), dict) else "executable_plan" in last_value,
        "last_tool_name": last_value.get("tool_request", {}).get("name")
        if isinstance(last_value.get("tool_request"), dict) else None,
        "detector_query_counts": dict(Counter(row["query"] for row in rows
                                              if row["type"] == "detector_result")),
        "zero_group_detections": sum(row["result"]["group_count"] == 0 for row in detections),
        "review_verdict_counts": dict(Counter(review["verdict"] for row in reviews
                                             for review in row["perception_review"])),
        "calls": calls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    historical = load(VERIFY / "historical-index.json")
    folders = attempted()
    pairs = {}
    for identity, prior in historical.items():
        folder = folders.get(identity)
        if folder is None or not (folder / "result.json").exists():
            continue
        assert file_digest(prior["result"]) == prior["result_sha256"]
        assert file_digest(prior["trace"]) == prior["trace_sha256"]
        pairs[identity] = {
            "task": prior["task"],
            "rgbd": summarize(Path(prior["trace"]).parent),
            "current": summarize(folder),
        }
    if not args.partial:
        assert len(pairs) == 20, "context_accounting_requires_all_twenty"
    aggregate = {}
    for arm in ("rgbd", "current"):
        rows = [pair[arm] for pair in pairs.values()]
        aggregate[arm] = {
            "requests": sum(row["requests"] for row in rows),
            "responses_with_usage": sum(row["responses_with_usage"] for row in rows),
            "total_usage": {key: sum(row["total_usage"][key] for row in rows) for key in TOKEN_KEYS},
            "host_elapsed_seconds": sum(row["host_elapsed_seconds"] for row in rows),
            "maximum_prompt_tokens": max((row["maximum_prompt_tokens"] for row in rows), default=0),
        }
    save(OUT / "context-costs.json", {
        "status": "partial" if len(pairs) != 20 else "verified", "completed": len(pairs),
        "live_requests_in_accounting": 0,
        "token_meaning": "Provider-reported billed tokens, including repeated context and images.",
        "aggregate": aggregate, "pairs": pairs,
    })
    print(json.dumps({"completed": len(pairs), "aggregate": aggregate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
