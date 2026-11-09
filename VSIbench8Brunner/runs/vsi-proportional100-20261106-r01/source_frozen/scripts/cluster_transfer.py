#!/usr/bin/env python3
"""Authenticated, resumable file transfer with byte verification and atomic ACKs.

The export manifest is an explicit allowlist. Authentication is supplied through
HARNESS3D_TRANSFER_TOKEN; credentials and URLs are never written to receipts.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def checked_rows(manifest: dict) -> list[dict]:
    rows = manifest["files"]
    seen = set()
    for row in rows:
        relative = PurePosixPath(row["relative"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("unsafe relative transfer path")
        if str(relative) in seen:
            raise ValueError("duplicate transfer path")
        seen.add(str(relative))
        if not isinstance(row["size"], int) or row["size"] < 0:
            raise ValueError("invalid file size")
        digest = row["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid SHA256")
    return rows


def serve(manifest: dict, host: str, port: int, startup: Path) -> None:
    token = os.environ["HARNESS3D_TRANSFER_TOKEN"]
    rows = checked_rows(manifest)
    mapping = {}
    for index, row in enumerate(rows):
        source = Path(row["source"]).resolve(strict=True)
        if source.stat().st_size != row["size"] or not source.is_file():
            raise ValueError("export file identity changed")
        mapping[str(index)] = (source, row)
    counters = {"downloads": 0, "bytes": 0}
    lock = threading.Lock()
    slots = threading.BoundedSemaphore(4)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(45)

        def log_message(self, *args):
            pass

        def do_GET(self):
            if not hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + token
            ):
                self.send_error(403, "unauthorized")
                return
            if self.path == "/manifest":
                public = {
                    **manifest,
                    "files": [{k: v for k, v in r.items() if k != "source"} for r in rows],
                }
                data = json.dumps(public, sort_keys=True).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            item = mapping.get(self.path.removeprefix("/files/"))
            if not self.path.startswith("/files/") or item is None:
                self.send_error(404, "unknown file")
                return
            if not slots.acquire(timeout=1):
                self.send_error(429, "transfer capacity reached")
                return
            try:
                path, row = item
                if path.stat().st_size != row["size"]:
                    self.send_error(409, "export identity changed")
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(row["size"]))
                self.send_header("X-Content-SHA256", row["sha256"])
                self.end_headers()
                with path.open("rb") as source:
                    shutil.copyfileobj(source, self.wfile, length=4 * 1024 * 1024)
                with lock:
                    counters["downloads"] += 1
                    counters["bytes"] += row["size"]
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            finally:
                slots.release()

    server = ThreadingHTTPServer((host, port), Handler)
    process_stat = Path("/proc/self/stat")
    atomic_json(startup, {
        "schema": "harness3d-transfer-service/1",
        "pid": os.getpid(),
        "start_ticks": process_stat.read_text().split()[21] if process_stat.is_file() else None,
        "started_at": time.time(),
        "host": host, "port": port, "files": len(rows),
        "bytes": sum(row["size"] for row in rows),
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
    })
    server.serve_forever(poll_interval=0.5)


def fetch(endpoint: str, destination: Path, receipt: Path, *, workers: int, reserve: int,
          prefixes: tuple[str, ...] = ()) -> dict:
    token = os.environ["HARNESS3D_TRANSFER_TOKEN"]
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    def request(route):
        return urllib.request.Request(endpoint.rstrip("/") + route,
                                      headers={"Authorization": "Bearer " + token})

    with urllib.request.urlopen(request("/manifest"), timeout=30) as response:
        manifest = json.load(response)
    indexed_rows = [(index, row) for index, row in enumerate(checked_rows(manifest))
                    if not prefixes or any(row["relative"].startswith(prefix) for prefix in prefixes)]
    rows = [row for _, row in indexed_rows]
    if not rows:
        raise ValueError("transfer selection is empty")
    public_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    missing_bytes = sum(row["size"] for row in rows
                        if not (destination / row["relative"]).is_file())
    if shutil.disk_usage(destination).free < missing_bytes + reserve:
        raise RuntimeError("insufficient destination space including reserve")
    lock = threading.Lock()
    progress = {"schema": "harness3d-transfer-receipt/1", "manifest_sha256": public_hash,
                "planned_files": len(rows), "planned_bytes": sum(r["size"] for r in rows),
                "verified_files": 0, "verified_bytes": 0, "errors": [], "status": "running"}
    began = time.monotonic()

    def copy(index, row):
        target = destination / row["relative"]
        if not target.resolve().is_relative_to(destination):
            raise ValueError("destination escapes transfer root")
        target.parent.mkdir(parents=True, exist_ok=True)
        reused = target.is_file() and target.stat().st_size == row["size"] and file_hash(target) == row["sha256"]
        if not reused:
            if target.exists():
                raise ValueError("published destination conflicts with frozen identity")
            temp = target.with_name(target.name + "." + uuid.uuid4().hex + ".part")
            try:
                for attempt in range(3):
                    digest = hashlib.sha256()
                    size = 0
                    try:
                        with urllib.request.urlopen(request("/files/" + str(index)), timeout=45) as response, temp.open("wb") as output:
                            if response.headers.get("X-Content-SHA256") != row["sha256"]:
                                raise ValueError("sender digest mismatch")
                            while block := response.read(4 * 1024 * 1024):
                                output.write(block)
                                digest.update(block)
                                size += len(block)
                            output.flush()
                            os.fsync(output.fileno())
                        if size != row["size"] or digest.hexdigest() != row["sha256"]:
                            raise ValueError("received bytes do not match frozen identity")
                        break
                    except (OSError, urllib.error.URLError, TimeoutError):
                        if attempt == 2:
                            raise
                        time.sleep(0.5 * (attempt + 1))
                os.replace(temp, target)
                directory_fd = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                temp.unlink(missing_ok=True)
        with lock:
            progress["verified_files"] += 1
            progress["verified_bytes"] += row["size"]
            progress["elapsed_s"] = time.monotonic() - began
            atomic_json(receipt, progress)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(copy, i, row): row["relative"] for i, row in indexed_rows}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                with lock:
                    progress["errors"].append({"relative": futures[future],
                                               "type": type(exc).__name__})
    progress["status"] = "verified" if not progress["errors"] else "incomplete"
    progress["elapsed_s"] = time.monotonic() - began
    atomic_json(receipt, progress)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    source = sub.add_parser("serve")
    source.add_argument("--manifest", type=Path, required=True)
    source.add_argument("--host", required=True)
    source.add_argument("--port", type=int, required=True)
    source.add_argument("--startup", type=Path, required=True)
    target = sub.add_parser("fetch")
    target.add_argument("--endpoint", required=True)
    target.add_argument("--destination", type=Path, required=True)
    target.add_argument("--receipt", type=Path, required=True)
    target.add_argument("--workers", type=int, default=2, choices=range(1, 5))
    target.add_argument("--reserve-gib", type=int, default=160)
    target.add_argument("--prefix", action="append", default=[])
    args = parser.parse_args()
    if args.action == "serve":
        serve(json.loads(args.manifest.read_text()), args.host, args.port, args.startup)
    else:
        result = fetch(args.endpoint, args.destination, args.receipt, workers=args.workers,
                       reserve=args.reserve_gib * 1024 ** 3, prefixes=tuple(args.prefix))
        print(json.dumps(result, sort_keys=True))
        if result["status"] != "verified":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
