"""Bounded transport diagnostics, deliberately separate from benchmark evidence."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import struct
import time
import zlib

from .api_pool import ApiPool
from .traces import read, write
from .transport import ResponsesClient


def png(red, green, blue):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    scan = b"".join(b"\0" + bytes((red, green, blue)) * 256 for _ in range(256))
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 256, 256, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scan)) + chunk(b"IEND", b""))
    return {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(data).decode()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=["text", "images"], required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--api-id", help="Select an API by its public configuration ID")
    args = parser.parse_args()
    config = read(args.config)
    config["upstream_policy"] = {}
    if args.concurrency not in (1, 2, 4):
        raise ValueError("bounded_probe_concurrency")
    config["max_requests"] = args.concurrency
    config["max_inflight"] = args.concurrency
    api = (next(a for a in config["apis"] if a["id"] == args.api_id)
           if args.api_id else config["apis"][0])
    api["max_inflight"] = args.concurrency
    pool = ApiPool(args.output + "/pool", config)
    started = time.monotonic()

    def one(index):
        client = ResponsesClient(api, pool, args.output + f"/requests/{index}", f"probe-{index}")
        content = "Return exactly the text OK."
        if args.phase == "images":
            content = [{"type": "text", "text": "Name the dominant color of image 1, then image 2. Return JSON with keys first and second."},
                       png(240, 10, 10), png(10, 10, 240)]
        try:
            text = client.chat([{"role": "user", "content": content}], max_tokens=8192,
                               seed=config["seed"], temperature=config["temperature"], enable_thinking=True)
            return {"index": index, "status": "received", "text": text,
                    "usage": client.last_usage, "generation": client.last_generation,
                    "response_identity": client.last_response_identity}
        except BaseException as exc:
            return {"index": index, "status": "error", "error_type": type(exc).__name__,
                    "service_error": client.last_service_error}

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        rows = list(executor.map(one, range(args.concurrency)))
    write(args.output + "/summary.json", {
        "schema": "gptexp-api-probe/1", "phase": args.phase, "concurrency": args.concurrency,
        "api_id": api["id"], "reasoning_effort": config["reasoning_effort"],
        "elapsed_seconds": time.monotonic() - started, "results": rows,
        "accounting": pool.summary(), "benchmark_evidence": False, "automatic_retry": False})
    print("Probe finished:", len(rows), "requests,", sum(r["status"] == "received" for r in rows), "received")


if __name__ == "__main__":
    main()
