"""Verify downloaded SR archives and extract regular trace files for host auditing."""
import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr


def receive(output):
    output = Path(output).resolve()
    receipt = sr.read(output / "delivery_receipt.json")
    for name, digest in receipt["archives"].items():
        if sr.file_sha(output / name) != digest:
            raise ValueError("sr_download_archive_hash_mismatch")
    raw = output / "raw"
    raw.mkdir(exist_ok=False)
    members = {}
    with tarfile.open(output / "completed_traces.tar.gz", "r:gz") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if (name.is_absolute() or ".." in name.parts or not member.isfile()
                    or member.name in members):
                raise ValueError("sr_archive_unsafe_or_duplicate_member")
            data = archive.extractfile(member).read()
            target = raw.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
            members[member.name] = hashlib.sha256(data).hexdigest()
    inventory = sr.read(raw / "delivery_files.json")
    if (inventory["content_sha256"] != receipt["inventory_sha256"]
            or inventory["files"] != {k: v for k, v in members.items()
                                      if k != "delivery_files.json"}):
        raise ValueError("sr_download_inventory_mismatch")
    manifest = sr.read(raw / "manifest.json")
    if any(members["source/" + name] != digest
           for name, digest in manifest["inspection"]["sources"].items()):
        raise ValueError("sr_download_frozen_source_mismatch")
    value = sr.write_once(output / "local_verified.json", {
        "status": "passed", "archive_hashes": receipt["archives"],
        "fully_read_and_checked_members": len(members),
        "manifest_sha256": manifest["content_sha256"],
        "reference_answers_read": False, "model_requests": 0,
        "receiver_sha256": sr.file_sha(Path(__file__)),
    })
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in list(os.environ):
        if any(t in name.upper() for t in (
                "API_KEY", "TOKEN", "ENDPOINT", "BASE_URL", "PASSWORD", "SECRET")):
            os.environ.pop(name, None)
    print(__import__("json").dumps(receive(args.output)))


if __name__ == "__main__":
    main()
