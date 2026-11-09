"""Create a public-only frozen SR runtime bundle; never includes host references."""
from pathlib import Path
import argparse
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr
from sr_trial import split_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    split = output / "split.json"
    sr.write_once(split, split_manifest(sr.DEFAULT_ROOT))
    source = sr.source_identity()
    source["scripts/run_small_image.py"] = sr.file_sha(sr.ROOT / "scripts/run_small_image.py")
    with tarfile.open(output / "public_runtime.tar.gz", "w:gz") as archive:
        archive.add(split, arcname="split.json")
        for name in source:
            archive.add(sr.ROOT / name, arcname="source/" + name, recursive=False)
        for name in ("manifest.json", "inputs.jsonl"):
            archive.add(sr.DEFAULT_ROOT / "data" / name, arcname="dataset/data/" + name)
        archive.add(sr.DEFAULT_ROOT / "data/images", arcname="dataset/data/images")
    sr.write_once(output / "bundle_receipt.json", {
        "sources": source, "split_sha256": sr.read(split)["content_sha256"],
        "bundle_sha256": sr.file_sha(output / "public_runtime.tar.gz"),
        "reference_files_included": False, "model_requests": 0,
    })
    print(output)


if __name__ == "__main__":
    main()
