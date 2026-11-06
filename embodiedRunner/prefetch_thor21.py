"""Download and verify the AI2-THOR 2.1 Linux build in task-local storage."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
import socket
import zipfile

from bridge import simulator_environment
from protocol import ROOT, canonical

NAME = "thor-201909061227-Linux64"
URL = "https://s3-us-west-2.amazonaws.com/ai2-thor/builds/" + NAME + ".zip"
EXPECTED = "cc61de810ddb8d681e9f6a7e342dd2d2c8454db413dd681de6bc85d2b2f70459"


def main():
    environment = simulator_environment(ROOT)
    for key in tuple(environment):
        if "PROXY" in key.upper():
            environment.pop(key)
    os.environ.clear()
    os.environ.update(environment)
    addresses = socket.getaddrinfo
    socket.getaddrinfo = lambda host, port, family=0, *a, **kw: addresses(
        host, port, socket.AF_INET if family == 0 else family, *a, **kw)
    import requests
    folder = ROOT / "cache/unity-2.1-download"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / (NAME + ".zip")
    if not archive.is_file():
        with requests.head(URL, timeout=(10, 30)) as response:
            response.raise_for_status()
            length = int(response.headers["Content-Length"])
        if not 0 < length < 1_000_000_000:
            raise ValueError("unexpected_build_size")
        partial = archive.with_suffix(".partial")
        fd = os.open(partial, os.O_EXCL | os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(fd, length)
        def fetch(first):
            last = min(first + 16 * 1024 * 1024, length) - 1
            offset = first
            with requests.get(URL, headers={"Range": f"bytes={first}-{last}"}, stream=True, timeout=(10, 60)) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers["Content-Range"] != f"bytes {first}-{last}/{length}":
                    raise ValueError("range_identity_mismatch")
                for block in response.iter_content(1024 * 1024):
                    if block:
                        if offset + len(block) > last + 1:
                            raise ValueError("range_overflow")
                        os.pwrite(fd, block, offset)
                        offset += len(block)
            if offset != last + 1:
                raise ValueError("range_incomplete")
        try:
            with ThreadPoolExecutor(max_workers=12) as pool:
                for future in as_completed([pool.submit(fetch, start) for start in range(0, length, 16 * 1024 * 1024)]):
                    future.result()
            os.fsync(fd)
        finally:
            os.close(fd)
        archive = partial
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != EXPECTED:
        raise ValueError("build_hash_mismatch")
    final_archive = folder / (NAME + ".zip")
    if archive != final_archive:
        os.replace(archive, final_archive)
    destination = ROOT / "cache/ai2thor-2.1/releases" / NAME
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(final_archive) as z:
        for member in z.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("zip_path_escaped")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(member)
            if target.is_file() and target.read_bytes() != data:
                raise ValueError("existing_build_changed")
            target.write_bytes(data)
            mode = member.external_attr >> 16
            if mode:
                target.chmod(mode & 0o777)
    binary = destination / NAME
    binary.chmod(0o755)
    receipt = {"status": "ready", "sha256": EXPECTED, "bytes": final_archive.stat().st_size,
               "build": NAME, "model_requests": 0, "complete_archive_hash_verified": True,
               "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    (folder / "receipt.json").write_bytes(canonical(receipt) + b"\n")
    print(canonical(receipt).decode())


if __name__ == "__main__":
    main()
