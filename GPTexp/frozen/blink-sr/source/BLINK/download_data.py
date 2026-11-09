#!/usr/bin/env python3
"""Download only BLINK/Multi-view_Reasoning through hf-mirror.com.

Keep original Parquet files, extract unchanged image bytes, and produce a
label-free input index plus a separate reference index for local evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import quote

from PIL import Image
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parent
REPO = "BLINK-Benchmark/BLINK"
SUBSET = "Multi-view_Reasoning"
MIRROR = "https://hf-mirror.com"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".write-", delete=False,
    ) as f:
        temporary = Path(f.name)
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def fetch_file(session: requests.Session, url: str, target: Path,
               expected_size: int, expected_sha: str | None) -> None:
    if (target.exists() and target.stat().st_size == expected_size
            and (not expected_sha or sha256_file(target) == expected_sha)):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    # Interrupted downloads stay outside the workspace.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temporary = Path(f.name)
        try:
            with session.get(url, timeout=(15, 120), stream=True) as response:
                response.raise_for_status()
                for block in response.iter_content(1024 * 1024):
                    if block:
                        f.write(block)
            f.flush()
            if temporary.stat().st_size != expected_size:
                raise ValueError("download_size_mismatch")
            if expected_sha and sha256_file(temporary) != expected_sha:
                raise ValueError("download_sha256_mismatch")
            shutil.copyfile(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def normalize_choices(choices: list[str]) -> list[dict]:
    result = []
    for i, choice in enumerate(choices):
        label = chr(65 + i)
        text = str(choice).strip()
        # Dataset strings can already contain '(A)' or 'A.' prefixes.
        text = re.sub(rf"^\(?{label}\)?[.:)]?\s+", "", text).strip()
        result.append({"label": label, "text": text, "original": str(choice)})
    return result


def normalize_gt(answer, options: list[dict]) -> str | None:
    if answer is None:
        return None
    text = str(answer).strip()
    if not text or text.lower() in {"none", "null", "nan", "hidden", "n/a", "?"}:
        return None
    label = text.strip("().: ").upper()
    if label in {option["label"] for option in options}:
        return label
    matches = [option["label"] for option in options
               if text.casefold() in {option["text"].casefold(),
                                     option["original"].strip().casefold()}]
    if len(matches) == 1:
        return matches[0]
    raise ValueError("unrecognized_reference_answer")


def prepare(root: Path, revision: str = "main") -> dict:
    session = requests.Session()
    metadata = session.get(f"{MIRROR}/api/datasets/{REPO}/revision/{revision}",
                           timeout=(15, 60))
    metadata.raise_for_status()
    commit = metadata.json()["sha"]
    listing = session.get(
        f"{MIRROR}/api/datasets/{REPO}/tree/{commit}/{SUBSET}",
        params={"recursive": "true", "expand": "false"}, timeout=(15, 60))
    listing.raise_for_status()
    files = [entry for entry in listing.json()
             if entry["type"] == "file" and entry["path"].endswith(".parquet")]
    if not files:
        raise ValueError("no_subset_parquet_files")
    manifest = {
        "schema_version": "blink-data/1",
        "repository": REPO, "subset": SUBSET, "mirror": MIRROR,
        "requested_revision": revision, "resolved_revision": commit,
        "files": [], "splits": {},
    }
    inputs, references, samples = [], [], []
    for entry in sorted(files, key=lambda item: item["path"]):
        filename = Path(entry["path"]).name
        split = filename.split("-")[0]
        local = root / "data" / entry["path"]
        expected_sha = (entry.get("lfs") or {}).get("oid")
        url = f"{MIRROR}/datasets/{REPO}/resolve/{commit}/{quote(entry['path'])}"
        fetch_file(session, url, local, entry["size"], expected_sha)
        manifest["files"].append({
            "path": str(local.relative_to(root)), "source_path": entry["path"],
            "size_bytes": local.stat().st_size, "sha256": sha256_file(local),
        })
        rows = pq.read_table(local).to_pylist()
        image_counts = Counter()
        gt_counts = Counter()
        for row_index, row in enumerate(rows):
            sample_id = f"{split}:{row['idx']}"
            options = normalize_choices(row["choices"])
            gt = normalize_gt(row.get("answer"), options)
            images = []
            for field in ("image_1", "image_2", "image_3", "image_4"):
                image_value = row.get(field)
                if image_value is None:
                    continue
                raw = image_value.get("bytes")
                if not isinstance(raw, bytes) or not raw:
                    raise ValueError("expected_embedded_parquet_image_bytes")
                with Image.open(BytesIO(raw)) as image:
                    image.verify()
                with Image.open(BytesIO(raw)) as image:
                    image_format, width, height = image.format, image.width, image.height
                extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(
                    image_format, "." + str(image_format).lower())
                mime = Image.MIME.get(image_format)
                if not mime:
                    raise ValueError("unknown_image_mime")
                relative = (Path("data") / "images" / split
                            / f"{row_index:04d}" / (field + extension))
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or target.read_bytes() != raw:
                    target.write_bytes(raw)
                images.append({
                    "field": field, "path": str(relative), "width": width,
                    "height": height, "mime_type": mime, "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                })
            if not images or len(options) < 2:
                raise ValueError("invalid_multimodal_question")
            public = {
                "sample_id": sample_id, "idx": str(row["idx"]),
                "split": split, "subset": SUBSET, "row_index": row_index,
                "sub_task": row.get("sub_task"), "question": row["question"],
                "choices": list(row["choices"]), "options": options,
                "benchmark_prompt": row.get("prompt"), "images": images,
                "reference_answer_visible": False,
            }
            reference = {
                "sample_id": sample_id, "split": split, "gt_available": gt is not None,
                "gt_answer": gt, "gt_raw": row.get("answer"),
                "gt_explanation": row.get("explanation"),
                "gt_answer_text": next(
                    (o["text"] for o in options if o["label"] == gt), None),
            }
            inputs.append(public)
            references.append(reference)
            if row_index == 0:
                samples.append({**public, **reference})
            image_counts[len(images)] += 1
            gt_counts["available" if gt is not None else "hidden"] += 1
        stats = manifest["splits"].setdefault(split, {
            "questions": 0, "gt_available": 0, "gt_hidden": 0,
            "images_per_question": {},
        })
        stats["questions"] += len(rows)
        stats["gt_available"] += gt_counts["available"]
        stats["gt_hidden"] += gt_counts["hidden"]
        for count, frequency in image_counts.items():
            key = str(count)
            stats["images_per_question"][key] = (
                stats["images_per_question"].get(key, 0) + frequency)
        print(json.dumps({"split": split, **stats}, ensure_ascii=False), flush=True)
    if len({r["sample_id"] for r in inputs}) != len(inputs):
        raise ValueError("duplicate_sample_ids")
    write_jsonl(root / "data" / "inputs.jsonl", inputs)
    write_jsonl(root / "data" / "references.jsonl", references)
    manifest["input_index_sha256"] = sha256_file(root / "data" / "inputs.jsonl")
    manifest["reference_index_sha256"] = sha256_file(root / "data" / "references.jsonl")
    write_json(root / "data" / "manifest.json", manifest)
    write_json(root / "data" / "sample_preview.json", samples)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    manifest = prepare(args.root.resolve(), args.revision)
    print(json.dumps({"status": "downloaded", "splits": manifest["splits"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
