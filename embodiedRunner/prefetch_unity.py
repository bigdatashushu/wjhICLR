"""Prefetch the fixed public Unity build in ranges and verify its complete hash."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import re
import socket
import sys
import time

sys.dont_write_bytecode = True
from bridge import simulator_environment
from protocol import ROOT, canonical

environment = simulator_environment(ROOT)
for name in tuple(environment):
    if "PROXY" in name.upper():
        environment.pop(name)
os.environ.clear()
os.environ.update(environment)
original_addresses = socket.getaddrinfo
socket.getaddrinfo = lambda host, port, family=0, *a, **kw: original_addresses(
    host, port, socket.AF_INET if family == 0 else family, *a, **kw)

import requests

NAME = "thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917"
URL = "https://s3-us-west-2.amazonaws.com/ai2-thor-public/builds/" + NAME + ".zip"
destination = ROOT / "cache/unity-download"
destination.mkdir(parents=True, exist_ok=True)
archive = destination / (NAME + ".zip")
with requests.get(URL[:-4] + ".sha256", timeout=(10, 30)) as response:
    response.raise_for_status()
    expected = response.text.strip()
assert re.fullmatch(r"[0-9a-f]{64}", expected)
with requests.head(URL, timeout=(10, 30)) as response:
    response.raise_for_status()
    length = int(response.headers["Content-Length"])
assert 0 < length <= 1_600_000_000
partial = archive.with_suffix(".partial")
descriptor = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
os.ftruncate(descriptor, length)
chunk_size = 16 * 1024 * 1024
parts = [(offset, min(offset + chunk_size, length) - 1)
         for offset in range(0, length, chunk_size)]


def fetch(bounds):
    first, last = bounds
    offset = first
    with requests.get(URL, headers={"Range": f"bytes={first}-{last}"},
                      timeout=(10, 40), stream=True) as response:
        assert response.status_code == 206
        assert response.headers["Content-Range"] == f"bytes {first}-{last}/{length}"
        for block in response.iter_content(1024 * 1024):
            if not block:
                continue
            assert offset + len(block) <= last + 1
            assert os.pwrite(descriptor, block, offset) == len(block)
            offset += len(block)
    assert offset == last + 1
    return last - first + 1


started = time.monotonic()
try:
    finished = 0
    with ThreadPoolExecutor(max_workers=12) as pool:
        for future in as_completed([pool.submit(fetch, bounds) for bounds in parts]):
            finished += future.result()
            print(canonical({"downloaded_bytes": finished, "total_bytes": length}).decode(), flush=True)
    os.fsync(descriptor)
finally:
    os.close(descriptor)
digest = hashlib.sha256()
with partial.open("rb") as stream:
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        digest.update(block)
assert digest.hexdigest() == expected
os.replace(partial, archive)
receipt = {"status": "ready", "build_name": NAME, "bytes": length,
           "sha256": expected, "complete_readback_verified": True,
           "elapsed_s": round(time.monotonic() - started, 3),
           "model_requests": 0, "automatic_retry": False, "connections": 12}
(destination / "receipt.json").write_bytes(canonical(receipt) + b"\n")
print(canonical(receipt).decode())
