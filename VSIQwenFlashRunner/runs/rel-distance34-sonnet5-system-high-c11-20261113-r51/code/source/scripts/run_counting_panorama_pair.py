#!/usr/bin/env python3
"""Compare direct answers with a frozen panorama, optionally across fresh rounds."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
import urllib.request


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(raw, encoding="utf-8")
    temp.replace(path)


def save_messages(path, messages):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(messages, stream, ensure_ascii=False)


def round_messages(base, round_index, max_rounds, previous_summary=None):
    """Send one fresh message, unchanged images and only the latest summary."""
    assert 1 <= round_index <= max_rounds == 5
    assert len(base) == 1 and base[0]["role"] == "user"
    content = base[0]["content"]
    assert sum(item["type"] == "image_url" for item in content) == 33
    assert content[-2]["type"] == "text" and content[-1]["type"] == "image_url"
    question_prompt = content[0]["text"].split("\nOutput only", 1)[0]
    prompt = (
        question_prompt + f"\nRound {round_index}/{max_rounds}. "
        "Every round receives the same 33 images: 32 chronological video frames "
        "and one additional panoramic image.\n"
        "Conversation history does not accumulate. Earlier prompts, replies and "
        "thinking are not available; only the previous round's summary below is carried forward.\n"
    )
    if round_index == max_rounds:
        prompt += (
            "This is the FINAL round. You MUST give your best answer now. "
            "Output only a non-negative integer count; no summary or explanation."
        )
    else:
        prompt += (
            "If confident, output only a non-negative integer count. "
            'If uncertain, output only {"next_prompt":"brief summary for the next round"}. '
            "In that summary, preserve useful observations, count hypotheses and what to recheck. "
            "The next round receives only this new summary, replacing the previous one."
        )
    if previous_summary is not None:
        assert round_index > 1 and isinstance(previous_summary, str) and previous_summary.strip()
        prompt += "\nPrevious round's summary:\n" + previous_summary
    return [{"role": "user", "content": [
        {"type": "text", "text": prompt}, *content[1:]]}]


def parse_round_reply(text, *, final_round, parse_answer):
    """Continuation and final answers are mutually exclusive."""
    raw = text.strip()
    if raw.startswith("{"):
        payload = json.loads(raw)
        if (not final_round and isinstance(payload, dict)
                and set(payload) == {"next_prompt"}
                and isinstance(payload["next_prompt"], str)
                and payload["next_prompt"].strip()):
            return None, payload["next_prompt"].strip()
        raise ValueError("Expected a count, or next_prompt before the final round")
    answer, _ = parse_answer(raw, question_type="object_counting", options=[])
    return answer.value, None


def total_usage(rounds):
    result = {}
    for row in rounds:
        usage = row.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if isinstance(usage.get(key), int):
                result[key] = result.get(key, 0) + usage[key]
        reasoning = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
        if isinstance(reasoning, int):
            details = result.setdefault("completion_tokens_details", {})
            details["reasoning_tokens"] = details.get("reasoning_tokens", 0) + reasoning
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--action", choices=("prepare", "run"), required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    manifest = read(root / "manifest.json")
    assert manifest["identity_sha256"] == digest({k: v for k, v in manifest.items()
                                                  if k != "identity_sha256"})
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == manifest["runner_sha256"]
    source = root / "code/source"
    sys.path.insert(0, str(source / "src"))
    for row in read(root / "code/code_identity.json")["files"]:
        assert hashlib.sha256((source / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    from skill3d.evaluation.vsi_rgb_inputs import FrozenRGBInputProvider
    from skill3d.online.config import load_config, online_model_from, resolve_runtime
    from skill3d.online.scene_overview import read_overview_image
    from skill3d.synthesis.eval_visual_fallback import build_visual_request, encode_visual_frames, parse_visual_answer
    from skill3d.infra.experiment_stages_v11 import baseline_input_receipt
    from skill3d.cluster.perception_client import RemoteBailianClient
    from skill3d.synthesis.request_governor import (
        GovernedClient, RequestGovernor, RequestGovernorConfig)
    from skill3d.synthesis.request_context import request_phase

    model = online_model_from(resolve_runtime(load_config(root / "inputs/online.yaml"), seed=137))
    assert model.model == "qwen3.8-flash" and model.n_frames == 32
    assert model.temperature == 0 and model.max_tokens == 8192
    assert model.enabled and model.provider == "bailian"
    runtime = manifest["runtime"]
    max_rounds = runtime.get("max_rounds", 1)
    assert max_rounds in (1, 5) and runtime["enable_thinking"] is True
    max_requests = 10 * (1 + max_rounds)
    assert runtime["max_requests"] == max_requests
    provider = FrozenRGBInputProvider(root / "inputs/published_inputs.json", max_scenes=10)
    variants = manifest["variants"]
    assert len(variants) == 10 and all(v["question_type"] == "object_counting" for v in variants)

    if args.action == "prepare":
        assert not (root / "preflight.json").exists()
        receipts = []
        for variant in variants:
            item = provider(variant)
            assert len(item.pixels) == 32 and not item.episode.ground_truth and item.geometry is None
            prompt, direct, _ = build_visual_request(
                question=item.episode.question, options=item.episode.options,
                question_type="object_counting", frames=item.pixels, max_images=32)
            panorama = read_overview_image(manifest["overviews"][variant["qa_id"]],
                                           scene_id=variant["scene_id"])
            extra = list(encode_visual_frames([panorama], max_images=1))
            notice = {"type": "text", "text": manifest["panorama_notice"]}
            augmented = [{"role": "user", "content": [*direct[0]["content"], notice, *extra]}]
            assert augmented[0]["content"][:-2] == direct[0]["content"]
            assert len(direct[0]["content"]) == 33 and len(augmented[0]["content"]) == 35
            input_receipt = baseline_input_receipt(provider, variant, item)
            qa = variant["qa_id"]
            pair = {"qa_id": qa, "question": item.episode.question, "prompt": prompt,
                "notice": manifest["panorama_notice"], "input_receipt": input_receipt,
                "direct_messages_sha256": digest(direct),
                "panorama_messages_sha256": digest(augmented),
                "identical_original_content": True,
                "direct_image_count": 32, "panorama_image_count": 33,
                "panorama_rgb_sha256": manifest["overviews"][qa]["rgb_sha256"]}
            write(root / "pairs" / f"QA{qa}.json", pair)
            for arm, messages in (("direct", direct), ("panorama", augmented)):
                save_messages(root / "messages" / arm / f"QA{qa}.json.gz", messages)
            if max_rounds > 1:
                initial = round_messages(augmented, 1, max_rounds)
                assert initial[0]["content"][1:] == augmented[0]["content"][1:]
                for index in range(1, max_rounds + 1):
                    previous = None if index == 1 else f"contract-summary-{index - 1}"
                    check = round_messages(augmented, index, max_rounds, previous)
                    assert f"Question: {item.episode.question}" in check[0]["content"][0]["text"]
                    assert check[0]["content"][-2] == notice
                    assert check[0]["content"][1:] == augmented[0]["content"][1:]
                    assert len(check) == 1 and check[0]["role"] == "user"
                    assert ("FINAL round" in check[0]["content"][0]["text"]) == (index == max_rounds)
                pair["panorama_round1_messages_sha256"] = digest(initial)
                pair["max_rounds"] = max_rounds
                pair["round_contract_checked"] = {
                    "cases": max_rounds, "original_question_every_round": True,
                    "panorama_notice_before_last_image_every_round": True,
                    "identical_33_images_every_round": True, "single_fresh_message": True}
                write(root / "pairs" / f"QA{qa}.json", pair)
            receipts.append(pair)
        write(root / "preflight.json", {
            "status": "complete", "model_requests": 0, "n_pairs": len(receipts),
            "pairs": receipts, "model": asdict(model), "manifest_identity": manifest["identity_sha256"]})
        print(json.dumps({"status": "prepared", "n_pairs": 10, "model_requests": 0}), flush=True)
        return

    assert read(root / "preflight.json")["status"] == "complete"
    assert not (root / "start.json").exists(), "Existing run cannot be silently repeated"
    # Use the existing node credential file only for gateway authentication.
    credential_path = Path("/home/wangy/wangjh/harness3d/precompute-20261022-r01/.transfer.env")
    allowed = {"HARNESS3D_PERCEPTION_TOKEN", "HARNESS3D_TRANSFER_TOKEN"}
    values = {}
    for line in credential_path.read_text(encoding="utf-8").splitlines():
        words = shlex.split(line)
        if words and words[0] == "export":
            words = words[1:]
        if len(words) == 1 and "=" in words[0]:
            name, value = words[0].split("=", 1)
            if name in allowed:
                values[name] = value
    os.environ["HARNESS3D_PERCEPTION_TOKEN"] = (
        values.get("HARNESS3D_PERCEPTION_TOKEN") or values["HARNESS3D_TRANSFER_TOKEN"])
    endpoint = runtime["perception_endpoint"]
    with urllib.request.urlopen(endpoint+"/health", timeout=15) as response:
        health = json.load(response)
    assert health["status"] == "ready" and health["max_concurrency"] == 32
    assert health["upstream"]["started"] == 0
    write(root / "gateway_initial_health.json", health)
    governor = RequestGovernor(root / "requests", RequestGovernorConfig(max_requests, 32, 0))
    start = time.time()
    write(root / "start.json", {"pid": os.getpid(), "started_at": start,
        "manifest_identity": manifest["identity_sha256"], "max_requested_calls": max_requests,
        "no_automatic_retry": True})

    def one(arm, variant):
        qa = variant["qa_id"]
        pair = read(root / "pairs" / f"QA{qa}.json")
        with gzip.open(root / "messages" / arm / f"QA{qa}.json.gz", "rt", encoding="utf-8") as stream:
            messages = json.load(stream)
        assert digest(messages) == pair[f"{arm}_messages_sha256"]
        client = RemoteBailianClient(
            endpoint=endpoint, model=model.model, model_revision=model.model_revision,
            base_url=model.base_url, api_key_env=model.api_key_env,
            timeout_s=model.timeout_s, temperature=0.0, max_output_tokens=model.max_tokens,
            transport_retries=0)
        assert client.configured and client.audit_identity() == health["identity"]
        intent = root / arm / "intents" / f"QA{qa}.json"
        assert not intent.exists()
        write(intent, {"qa_id": qa, "arm": arm, "messages_sha256": digest(messages),
            "started_at": time.time(), "manifest_identity": manifest["identity_sha256"]})
        governed = GovernedClient(client, governor, category="solver",
            context={"variant_id": f"{arm}:QA{qa}", "campaign_id": manifest["identity_sha256"]})
        row = {"qa_id": qa, "scene_id": variant["scene_id"], "arm": arm,
            "messages_sha256": digest(messages), "answer": None, "status": "service_error",
            "reference_answer_visible": False, "tool_calls": 0}
        began = time.time()
        rounds = []
        previous_summary = None
        limit = max_rounds if arm == "panorama" else 1
        for round_index in range(1, limit + 1):
            current = (round_messages(messages, round_index, limit, previous_summary)
                       if limit > 1 else messages)
            trace = root / arm / "rounds" / f"QA{qa}" / f"round{round_index}.json"
            message_path = root / "messages" / arm / f"QA{qa}" / f"round{round_index}.json.gz"
            save_messages(message_path, current)
            record = {"round_index": round_index, "messages_sha256": digest(current),
                "input_summary": previous_summary, "output_summary": None,
                "message_count": len(current),
                "image_count": sum(i["type"] == "image_url" for i in current[0]["content"]),
                "final_round": round_index == limit, "status": "service_error",
                "answer": None, "started_at": time.time()}
            write(trace.with_name(f"round{round_index}.intent.json"), record)
            round_start = time.time()
            try:
                with request_phase("direct_panorama_ablation"):
                    text = governed.chat(current, max_tokens=8192, seed=137,
                                         temperature=0.0, enable_thinking=True)
                params = client.last_request_identity["parameters"]
                assert params["model"] == model.model and params["seed"] == 137
                assert params["temperature"] == 0 and params["max_tokens"] == 8192
                assert params["enable_thinking"] is True
                assert client.last_response_identity["model"] == model.model
                record.update(response_text=text, usage=client.last_usage,
                    generation=client.last_generation,
                    actual_request_identity=client.last_request_identity,
                    response_identity=client.last_response_identity, status="invalid_answer")
                try:
                    if limit > 1:
                        answer, summary = parse_round_reply(
                            text, final_round=round_index == limit, parse_answer=parse_visual_answer)
                    else:
                        payload, _ = parse_visual_answer(
                            text, question_type="object_counting", options=[])
                        answer, summary = payload.value, None
                except (ValueError, TypeError, AttributeError) as exc:
                    record["parse_error"] = str(exc)
                else:
                    record.update(status="continued" if summary is not None else "completed",
                                  answer=answer, output_summary=summary)
            except Exception as exc:
                record["exception_type"] = type(exc).__name__
                # Existing client transport errors omit secret values.
                record["error"] = str(exc)[:600]
            record["elapsed_s"] = time.time() - round_start
            write(trace, record)
            rounds.append(record)
            print(json.dumps({"qa_id": qa, "arm": arm, "round": round_index,
                "status": record["status"], "answer": record["answer"],
                "elapsed_s": round(record["elapsed_s"], 2)}, ensure_ascii=False), flush=True)
            if record["status"] != "continued":
                row.update(status=record["status"], answer=record["answer"])
                for key in ("response_text", "generation", "actual_request_identity",
                            "response_identity", "parse_error", "exception_type", "error"):
                    if key in record:
                        row[key] = record[key]
                break
            previous_summary = record["output_summary"]
        row.update(rounds=rounds, n_rounds=len(rounds), usage=total_usage(rounds))
        row["elapsed_s"] = time.time()-began
        write(root / arm / "responses" / f"QA{qa}.json", row)
        print(json.dumps({"qa_id": qa, "arm": arm, "status": row["status"],
            "answer": row["answer"], "elapsed_s": round(row["elapsed_s"], 2)}, ensure_ascii=False), flush=True)
        return row

    tasks = []
    for index, variant in enumerate(variants):
        order = ("direct", "panorama") if index % 2 == 0 else ("panorama", "direct")
        tasks.extend((arm, variant) for arm in order)
    with ThreadPoolExecutor(max_workers=32) as pool:
        results = [future.result() for future in as_completed([pool.submit(one, *task) for task in tasks])]
    with urllib.request.urlopen(endpoint+"/health", timeout=15) as response:
        final_health = json.load(response)
    write(root / "gateway_final_health.json", final_health)
    assert final_health["upstream"]["active"] == 0
    write(root / "exit.json", {
        "status": "complete" if all(r["status"] == "completed" for r in results) else "incomplete",
        "n_results": len(results), "n_completed": sum(r["status"] == "completed" for r in results),
        "started_at": start, "finished_at": time.time(), "model_requests": final_health["upstream"]["started"]})
    print(json.dumps(read(root / "exit.json")), flush=True)


if __name__ == "__main__":
    main()
