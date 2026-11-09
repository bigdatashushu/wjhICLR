"""Official BLINK SR adapter: public original inputs and separate host references."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "BLINK" / "spatial_relation"
sys.path.insert(0, str(ROOT / "src"))

from skill3d.small_image.blink_protocol import image_identities, strict_json
from sr_adapter import SRProblem as SmallImageProblem
from sr_protocol import SUBSET, relation_for, validate_official_prompt

PUBLIC_KEYS = {
    "sample_id", "idx", "split", "subset", "row_index", "sub_task", "question",
    "choices", "options", "benchmark_prompt", "images", "reference_answer_visible",
}
IMAGE_KEYS = {
    "field", "path", "width", "height", "mime_type", "size_bytes", "sha256",
}


def file_sha(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path) -> list[dict]:
    return [strict_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def download(root=DEFAULT_ROOT, revision="main") -> dict:
    # Parquet and requests dependencies are needed only for the download step.
    sys.path.insert(0, str(ROOT / "BLINK"))
    import download_data
    root = Path(root).resolve()
    existing = root / "data" / "manifest.json"
    if existing.exists() and strict_json(existing.read_text())["subset"] != SUBSET:
        raise ValueError("dataset_root_already_contains_another_subset")
    previous = download_data.SUBSET
    try:
        download_data.SUBSET = SUBSET
        return download_data.prepare(root, revision)
    finally:
        download_data.SUBSET = previous


def load_inputs(root=DEFAULT_ROOT, splits=("val", "test")) -> tuple[dict, list[tuple]]:
    """No reference file is opened, including during denominator verification."""
    root = Path(root).resolve()
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or set(splits) - {"val", "test"}:
        raise ValueError("sr_splits_must_be_val_or_test")
    manifest = strict_json((root / "data" / "manifest.json").read_text())
    if (manifest.get("subset") != SUBSET
            or manifest.get("repository") != "BLINK-Benchmark/BLINK"
            or file_sha(root / "data" / "inputs.jsonl") != manifest["input_index_sha256"]):
        raise ValueError("sr_dataset_manifest_mismatch")
    all_rows = read_rows(root / "data" / "inputs.jsonl")
    if (len(all_rows) != 286
            or Counter(row.get("split") for row in all_rows) != {"val": 143, "test": 143}
            or len({row.get("sample_id") for row in all_rows}) != 286):
        raise ValueError("sr_official_denominator_mismatch")
    result = []
    for row in sorted(all_rows, key=lambda r: (r["split"], r["row_index"])):
        if set(row) != PUBLIC_KEYS or row["reference_answer_visible"] is not False:
            raise ValueError("sr_public_input_contains_nonpublic_fields")
        if (row["subset"] != SUBSET or row["sub_task"] != "Spatial Relation"
                or row["choices"] != ["yes", "no"]
                or row["options"] != [
                    {"label": "A", "text": "yes", "original": "yes"},
                    {"label": "B", "text": "no", "original": "no"}]
                or len(row["images"]) != 1):
            raise ValueError("unexpected_sr_dataset_format")
        if (row["idx"] != f"{row['split']}_{SUBSET}_{row['row_index'] + 1}"
                or row["sample_id"] != f"{row['split']}:{row['idx']}"):
            raise ValueError("sr_row_identity_mismatch")
        image = row["images"][0]
        if set(image) != IMAGE_KEYS or image["field"] != "image_1":
            raise ValueError("unexpected_sr_image_fields")
        path = (root / image["path"]).resolve()
        if not path.is_relative_to((root / "data" / "images").resolve()):
            raise ValueError("sr_image_path_outside_dataset")
        if file_sha(path) != image["sha256"] or path.stat().st_size != image["size_bytes"]:
            raise ValueError("sr_original_image_changed")
        problem = SmallImageProblem(
            benchmark="blink", sample_id=row["sample_id"], source_split=row["split"],
            question=row["question"], options=("yes", "no"),
            image_paths=(str(path),), task_type=SUBSET)
        validate_official_prompt(problem, row["benchmark_prompt"])
        actual = image_identities(problem)[0]
        if any(actual[key] != image[key] for key in
               ("width", "height", "mime_type", "sha256")):
            raise ValueError("sr_original_image_metadata_changed")
        if problem.source_split in splits:
            result.append((problem, row["benchmark_prompt"]))
    result.sort(key=lambda item: (splits.index(item[0].source_split),
                                 int(item[0].sample_id.rsplit("_", 1)[1])))
    return manifest, result


def input_summary(manifest, items) -> dict:
    families = {}
    for split in ("val", "test"):
        selected = [p for p, _ in items if p.source_split == split]
        if selected:
            families[split] = dict(sorted(Counter(
                relation_for(p.question)["family"] for p in selected).items()))
    return {
        "subset": SUBSET, "dataset_revision": manifest["resolved_revision"],
        "input_index_sha256": manifest["input_index_sha256"],
        "questions": len(items), "splits": dict(Counter(p.source_split for p, _ in items)),
        "relation_families": families, "images_per_question": 1,
        "options": {"A": "yes", "B": "no"},
        "official_labels": {s: {"available": v["gt_available"], "hidden": v["gt_hidden"]}
                            for s, v in manifest["splits"].items()},
        "reference_answers_read": False, "model_requests": 0,
    }


def load_references(root, manifest) -> dict:
    """Host-only scoring step; hidden test references remain None."""
    path = Path(root) / "data" / "references.jsonl"
    if file_sha(path) != manifest["reference_index_sha256"]:
        raise ValueError("sr_reference_index_changed")
    rows = read_rows(path)
    if len(rows) != 286 or len({r["sample_id"] for r in rows}) != 286:
        raise ValueError("sr_reference_denominator_mismatch")
    for row in rows:
        if row["split"] == "test":
            if row["gt_available"] is not False or row["gt_answer"] is not None:
                raise ValueError("sr_official_test_labels_must_be_hidden")
        elif row["split"] == "val":
            if row["gt_available"] is not True or row["gt_answer"] not in {"A", "B"}:
                raise ValueError("sr_invalid_val_reference")
        else:
            raise ValueError("sr_invalid_reference_split")
    return {row["sample_id"]: row["gt_answer"] for row in rows}
