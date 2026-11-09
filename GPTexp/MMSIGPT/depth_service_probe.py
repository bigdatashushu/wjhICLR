"""One explicitly requested synthetic vision check; no benchmark answers or retries."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import re
import time

from PIL import Image
from xcodeapi.config import Profile

from ..traces import credential_free, digest, file_digest, write
from .depth_comparison.pool import BASE_URL, MODEL, Client, read_keys
from .prototype.client import ServiceError
from .prototype.protocol import payload


KEY_PATH = Path(__file__).resolve().parents[2] / "xcodeapi/credentials.local.env"


def all_keys():
    # Preserve the original GPT slot mapping, then include other distinct configured keys.
    credentials = read_keys(KEY_PATH)
    names = {"XCODE_API_KEY", "XCODE_GPT_API_KEY",
             "XCODE_CLAUDE_API_KEY", "XCODE_GEMINI_API_KEY"}
    for line in KEY_PATH.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*",
                             line.strip())
        if match and match.group(1) in names:
            value = match.group(2).strip("'\"")
            if value and value not in credentials.values():
                if any(char.isspace() for char in value):
                    raise ValueError("invalid_configured_credential")
                credentials[f"extra-{len(credentials) + 1:02d}"] = value
    return credentials


def check(api_id, credentials):
    client = Client(api_id, credentials[api_id])
    client.profile = Profile("responses", BASE_URL, MODEL, "XCODE_GPT_API_KEY",
                             timeout_seconds=90)
    image = BytesIO()
    Image.new("RGB", (32, 32), (255, 0, 0)).save(image, format="PNG")
    import base64
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(image.getvalue()).decode("ascii")}},
        {"type": "text", "text": "Name the dominant color in this image in one word."},
    ]}]
    body = payload(messages, client.inference)
    for key in credentials.values():
        credential_free(body, key)
    started = time.monotonic()
    receipt = {
        "schema": "mmsi-depth-service-recheck/1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "api_id": api_id, "available_credential_slots": len(credentials),
        "requested": client.inference, "timeout_seconds": 90,
        "model_requests": 1, "max_requests": 1, "automatic_retry": False,
        "input": "synthetic_32x32_red_image", "payload_sha256": digest(body),
        "benchmark_questions_sent": False, "reference_answers_read": False,
        "source_sha256": file_digest(__file__),
    }
    try:
        response = client.call(body)
        receipt.update(status="success", response=response,
                       synthetic_vision_check=response["text"].strip().lower().rstrip(".") == "red")
    except ServiceError as exc:
        receipt.update(status="service_error", error_code=exc.code, http_status=exc.http_status)
    except Exception:
        receipt.update(status="service_error", error_code="client_outcome_unknown", http_status=None)
    receipt["elapsed_seconds"] = time.monotonic() - started
    for key in credentials.values():
        credential_free(receipt, key)
    return receipt


def probe(output, api_id=None, *, every_key=False):
    output = Path(output)
    if not output.parent.is_dir() or output.exists():
        raise ValueError("requires_new_receipt_with_existing_parent")
    credentials = all_keys() if every_key else read_keys(KEY_PATH)
    if every_key:
        with ThreadPoolExecutor(max_workers=min(10, len(credentials))) as pool:
            results = list(pool.map(lambda api: check(api, credentials), credentials))
        receipt = {
            "schema": "mmsi-depth-all-api-recheck/1", "model": MODEL,
            "model_requests": len(results), "max_requests": len(credentials),
            "distinct_credential_slots": len(credentials), "automatic_retry": False,
            "reference_answers_read": False, "all_keys_checked_against_same_gpt_model": True,
            "successful_slots": [r["api_id"] for r in results
                                 if r["status"] == "success" and r["synthetic_vision_check"]],
            "results": results,
        }
        for key in credentials.values():
            credential_free(receipt, key)
        write(output, receipt, exclusive=True)
        print(json.dumps({
            "model_requests": len(results), "successful_slots": receipt["successful_slots"],
            "results": [{k: row[k] for k in ("api_id", "status", "elapsed_seconds")}
                        | {k: row[k] for k in ("http_status", "synthetic_vision_check") if k in row}
                        for row in results],
        }))
        return
    api_id = api_id or next(iter(credentials))
    if api_id not in credentials:
        raise ValueError("requested_api_slot_not_available")
    receipt = check(api_id, credentials)
    write(output, receipt, exclusive=True)
    print(json.dumps({key: receipt[key] for key in (
        "status", "model_requests", "elapsed_seconds", "automatic_retry")}
        | {key: receipt[key] for key in ("http_status", "synthetic_vision_check") if key in receipt}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    slots = parser.add_mutually_exclusive_group()
    slots.add_argument("--api-id")
    slots.add_argument("--all", action="store_true")
    try:
        args = parser.parse_args()
        probe(args.output, args.api_id, every_key=args.all)
    except Exception:
        print('{"status":"failed","reason_code":"service_probe_failed"}')
        raise SystemExit(2)
