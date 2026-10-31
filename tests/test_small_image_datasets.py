"""Independent small-image loaders: public custody and exact byte-sensitive replay."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import pytest

from skill3d.small_image.datasets import (
    SmallImageProblem,
    canonical_benchmark,
    freeze_manifest,
    load_evaluation_answers,
    load_manifest,
    load_problems,
    normalize_answer,
    verify_manifest,
)

REAL_ROOT = Path("/Users/bytedance/wangjh/skill3d-paper-benchmarks-20261030/dataset")
COUNTS = {
    "blink": (40, 93), "cv3d": (360, 840), "mmsi": (157, 365),
}
SOURCE_NAMES = {"blink": "blink_multiview", "cv3d": "cv3d", "mmsi": "mmsi_pr"}
PUBLIC_KEYS = {
    "benchmark", "sample_id", "source_split", "question", "options",
    "image_paths", "task_type", "input_sha256",
}


def _row(sample_id="sample-1", *, benchmark="blink"):
    prompt = "Which direction?\n(A) left\n(B) right"
    image = "BLINK_images/a.png"
    if benchmark == "mmsi":
        prompt = "Which direction?\n(A: Front)\n(B: Back)\n(C: Left)\n(D: Right)\n"
        image = "MMSI_images/a.png"
    elif benchmark == "cv3d":
        image = "CVBench/3D/a.png"
    return {
        "id": sample_id,
        "task": "spatial relationship",
        "image": [image],
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": "B"},
        ],
        "others": {
            "thought": "HOST_ONLY_THOUGHT",
            "original_answer": "HOST_ONLY_ANSWER",
            "bbox": [[1, 2, 3, 4]],
        },
        "ground_truth": "HOST_ONLY_GT",
        "labels": {"hidden": "HOST_ONLY_LABEL"},
    }


def _write_split(root, rows, *, benchmark="blink", split="train"):
    directory = root / "skill3d_splits"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{SOURCE_NAMES[benchmark]}_{split}.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    for name in ("BLINK_images", "CVBench/3D", "MMSI_images"):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "a.png").write_bytes(b"independent-image-bytes")
        (directory / "b.png").write_bytes(b"second-image-bytes")
    _write_split(root, [_row(), _row("sample-2")])
    return root


@pytest.fixture(scope="module")
def real_problems():
    if not (REAL_ROOT / "skill3d_splits").is_dir():
        pytest.skip("real small-image data is not mounted on this host")
    return {
        (benchmark, split): load_problems(REAL_ROOT, benchmark, split)
        for benchmark in COUNTS for split in ("train", "test")
    }


def test_public_fields_are_frozen_and_explicitly_whitelisted(dataset):
    problem = load_problems(dataset, "blink")[0]
    assert isinstance(problem, SmallImageProblem)
    assert set(f.name for f in fields(problem)) == PUBLIC_KEYS
    assert set(problem.public_dict()) == PUBLIC_KEYS
    assert problem.options == ("left", "right")
    assert problem.question == "Which direction?"
    assert isinstance(problem.image_paths, tuple)
    assert all(Path(p).is_absolute() for p in problem.image_paths)
    public = problem.public_dict()
    assert "HOST_ONLY" not in json.dumps(public)
    public["options"][0] = "mutated externally"
    assert problem.options == ("left", "right")
    with pytest.raises(FrozenInstanceError):
        problem.question = "changed"
    object.__setattr__(problem, "ground_truth", "HOST_ONLY_GT")
    assert "ground_truth" not in problem.public_dict()


def test_public_loading_does_not_require_or_consult_labels(dataset):
    row = _row()
    row["conversations"] = row["conversations"][:1]
    del row["others"]
    del row["ground_truth"]
    del row["labels"]
    _write_split(dataset, [row])
    original = load_problems(dataset, "blink")[0]
    row["conversations"].append({"from": "gpt", "value": {"labels": "HOST_ONLY_GT"}})
    row["others"] = {"thought": "HOST_ONLY_THOUGHT"}
    _write_split(dataset, [row])
    assert load_problems(dataset, "blink")[0] == original
    with pytest.raises(ValueError):
        load_evaluation_answers(dataset, "blink")


@pytest.mark.parametrize("alias,canonical", [
    ("BLINK", "blink"), ("blink_multiview", "blink"),
    ("CV-3D", "cv3d"), ("CVBench", "cv3d"),
    ("mmsi_pr", "mmsi"), ("MMSI-Bench", "mmsi"),
])
def test_benchmark_aliases(alias, canonical):
    assert canonical_benchmark(alias) == canonical


def test_source_order_limit_and_real_split_not_padding(dataset):
    assert [p.sample_id for p in load_problems(dataset, "blink")] == ["sample-1", "sample-2"]
    assert len(load_problems(dataset, "blink", limit=1)) == 1
    assert load_problems(dataset, "blink", limit=0) == []
    assert len(load_problems(dataset, "blink", limit=32)) == 2
    with pytest.raises(ValueError, match="missing source split"):
        load_problems(dataset, "blink", "test")
    _write_split(dataset, [_row("test-only")], split="test")
    problem = load_problems(dataset, "blink", "test")[0]
    assert problem.sample_id == "test-only"
    assert problem.source_split == "test"


@pytest.mark.parametrize("limit", [-1, 1.5, True, "2"])
def test_invalid_limits(dataset, limit):
    with pytest.raises(ValueError, match="limit"):
        load_problems(dataset, "blink", limit=limit)


@pytest.mark.parametrize("field", ["id", "task", "image", "conversations"])
def test_missing_required_source_fields_fail(dataset, field):
    row = _row()
    del row[field]
    _write_split(dataset, [row])
    with pytest.raises(ValueError):
        load_problems(dataset, "blink")


@pytest.mark.parametrize("images", [
    [], ["BLINK_images/a.png"] * 7, ["BLINK_images/missing.png"],
    ["../BLINK_images/a.png"], ["MMSI_images/a.png"], "BLINK_images/a.png",
])
def test_invalid_image_paths_or_count_fail(dataset, images):
    row = _row()
    row["image"] = images
    _write_split(dataset, [row])
    with pytest.raises(ValueError):
        load_problems(dataset, "blink")


def test_absolute_source_path_is_rejected(dataset):
    row = _row()
    row["image"] = [str(dataset / "BLINK_images/a.png")]
    _write_split(dataset, [row])
    with pytest.raises(ValueError, match="containment"):
        load_problems(dataset, "blink")


def test_named_symlink_image_root_allowed_but_nested_escape_rejected(dataset, tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (external / "a.png").write_bytes(b"external-root-image")
    root = tmp_path / "symlink-dataset"
    root.mkdir()
    (root / "BLINK_images").symlink_to(external, target_is_directory=True)
    _write_split(root, [_row()])
    problem = load_problems(root, "blink")[0]
    assert problem.image_paths == (str(external / "a.png"),)
    assert load_manifest(freeze_manifest([problem]), root) == [problem]
    (external / "escape.png").symlink_to(dataset / "MMSI_images/a.png")
    row = _row()
    row["image"] = ["BLINK_images/escape.png"]
    _write_split(root, [row])
    with pytest.raises(ValueError, match="outside"):
        load_problems(root, "blink")


@pytest.mark.parametrize("prompt", [
    "No choices", "Question\n(A) one", "Question\n(A) one\n(C) two",
    "Question\n(B) one\n(A) two", "\n(A) one\n(B) two",
    "Question\n(A) one\n(B) ",
])
def test_malformed_prompt_fails(dataset, prompt):
    row = _row()
    row["conversations"][0]["value"] = prompt
    _write_split(dataset, [row])
    with pytest.raises(ValueError):
        load_problems(dataset, "blink")


def test_mmsi_multiline_semicolon_encoding_and_duplicate_options(dataset):
    row = _row(benchmark="mmsi")
    row["conversations"][0]["value"] = (
        "Which direction?\n(A: Farther)\n(on the left; B: Closer)\n"
        "(on the right; C: Farther)\n(on the right; D: Closer)\n(on the left)\n"
    )
    _write_split(dataset, [row], benchmark="mmsi")
    problem = load_problems(dataset, "mmsi")[0]
    assert problem.options == (
        "Farther, on the left", "Closer, on the right",
        "Farther, on the right", "Closer, on the left",
    )
    row["conversations"][0]["value"] = "Where?\n(A: Left)\n(B: Front)\n(C: Back)\n(D: Left)\n"
    _write_split(dataset, [row], benchmark="mmsi")
    duplicate = load_problems(dataset, "mmsi")[0]
    assert duplicate.options == ("Left", "Front", "Back", "Left")
    assert normalize_answer("D", duplicate.options) == "D"
    with pytest.raises(ValueError, match="exactly one"):
        normalize_answer("Left", duplicate.options)


@pytest.mark.parametrize("answer,expected", [
    ("A", "A"), ("(b)", "B"), (" B. ", "B"), ("A)", "A"),
    ("right", "B"), ("LEFT", "A"),
])
def test_answer_normalization(answer, expected):
    assert normalize_answer(answer, ("left", "right")) == expected


@pytest.mark.parametrize("answer", ["C", "1", 1, "", "Answer: A", "A or B", "left because..."])
def test_answer_normalization_is_not_guessing(answer):
    with pytest.raises(ValueError):
        normalize_answer(answer, ("left", "right"))


def test_answer_letter_text_conflict_is_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        normalize_answer("A", ("left", "A"))


def test_host_evaluation_is_separate(dataset):
    assert load_evaluation_answers(dataset, "blink") == {"sample-1": "B", "sample-2": "B"}
    assert all("answer" not in p.public_dict() for p in load_problems(dataset, "blink"))


def test_duplicate_source_ids_fail(dataset):
    _write_split(dataset, [_row(), _row()])
    with pytest.raises(ValueError, match="duplicate"):
        load_problems(dataset, "blink")
    with pytest.raises(ValueError, match="duplicate"):
        load_evaluation_answers(dataset, "blink")


def test_manifest_roundtrip_order_digest_and_no_overwrite(dataset, tmp_path):
    problems = load_problems(dataset, "blink")
    path = tmp_path / "manifest.json"
    manifest = freeze_manifest(problems, path)
    assert set(manifest) == {"schema", "problems", "manifest_sha256"}
    assert "HOST_ONLY" not in path.read_text()
    assert all(set(r) == PUBLIC_KEYS for r in manifest["problems"])
    assert load_manifest(path, dataset, expected_sha256=manifest["manifest_sha256"]) == problems
    verify_manifest(manifest, problems, dataset)
    with pytest.raises(FileExistsError):
        freeze_manifest(problems, path)
    with pytest.raises(ValueError, match="expected"):
        load_manifest(manifest, dataset, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="ordered"):
        verify_manifest(manifest, reversed(problems), dataset)
    assert freeze_manifest(reversed(problems))["manifest_sha256"] != manifest["manifest_sha256"]


def test_image_byte_mutation_changes_identity_and_rejects_stale_manifest(dataset):
    problem = load_problems(dataset, "blink")[0]
    manifest = freeze_manifest([problem])
    image = Path(problem.image_paths[0])
    original_bytes = image.read_bytes()
    image.write_bytes(b"changed-image-bytes")
    changed = load_problems(dataset, "blink")[0]
    assert changed.image_paths == problem.image_paths
    assert changed.input_sha256 != problem.input_sha256
    with pytest.raises(ValueError, match="identity"):
        load_manifest(manifest, dataset)
    with pytest.raises(ValueError, match="identity"):
        freeze_manifest([problem])
    with pytest.raises(ValueError, match="identity"):
        verify_manifest(manifest, [problem], dataset)
    image.write_bytes(original_bytes)
    assert load_manifest(manifest, dataset) == [problem]


@pytest.mark.parametrize("field,value", [
    ("sample_id", "changed-id"), ("source_split", "test"),
    ("question", "Different question?"), ("options", ("other", "choices")),
    ("task_type", "different task"),
])
def test_public_input_mutations_change_identity(dataset, field, value):
    problem = load_problems(dataset, "blink")[0]
    assert replace(problem, **{field: value}).input_sha256 != problem.input_sha256


def _resign(manifest):
    payload = {k: manifest[k] for k in ("schema", "problems")}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return manifest


@pytest.mark.parametrize("key", [
    "ground_truth", "answer", "labels", "others", "conversations", "source_row",
    "coordinates", "metadata",
])
def test_manifest_rejects_all_non_whitelisted_fields_even_when_resigned(dataset, key):
    manifest = freeze_manifest(load_problems(dataset, "blink"))
    manifest["problems"][0][key] = {"hidden": "HOST_ONLY"}
    with pytest.raises(ValueError, match="whitelisted"):
        load_manifest(_resign(manifest), dataset)
    manifest = freeze_manifest(load_problems(dataset, "blink"))
    manifest[key] = "HOST_ONLY"
    with pytest.raises(ValueError, match="whitelisted"):
        load_manifest(manifest, dataset)


def test_manifest_tampering_and_duplicate_identity_fail(dataset):
    manifest = freeze_manifest(load_problems(dataset, "blink"))
    altered = copy.deepcopy(manifest)
    altered["problems"][0]["question"] = "tampered"
    with pytest.raises(ValueError, match="identity"):
        load_manifest(altered, dataset)
    with pytest.raises(ValueError, match="identity"):
        load_manifest(_resign(altered), dataset)
    duplicate = copy.deepcopy(manifest)
    duplicate["problems"].append(duplicate["problems"][0])
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(_resign(duplicate), dataset)
    with pytest.raises(ValueError, match="duplicate"):
        freeze_manifest([load_problems(dataset, "blink")[0]] * 2)
    with pytest.raises(ValueError, match="only SmallImageProblem"):
        freeze_manifest([_row()])


def test_manifest_rejects_duplicate_json_keys(dataset, tmp_path):
    manifest = freeze_manifest(load_problems(dataset, "blink"))
    encoded = json.dumps(manifest)
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"ignored",' + encoded[1:], encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON"):
        load_manifest(path, dataset)


def test_manifest_cannot_replay_foreign_absolute_images(dataset, tmp_path):
    problem = load_problems(dataset, "blink")[0]
    external = tmp_path / "unapproved.png"
    external.write_bytes(Path(problem.image_paths[0]).read_bytes())
    manifest = freeze_manifest([replace(problem, image_paths=(str(external),))])
    with pytest.raises(ValueError, match="outside"):
        load_manifest(manifest, dataset)


def test_all_real_1855_rows_counts_order_splits_and_public_custody(real_problems):
    assert sum(map(len, real_problems.values())) == 1855
    image_counts = set()
    for benchmark, expected in COUNTS.items():
        train, test = (real_problems[(benchmark, split)] for split in ("train", "test"))
        assert (len(train), len(test)) == expected
        assert not ({p.sample_id for p in train} & {p.sample_id for p in test})
        for split, problems in (("train", train), ("test", test)):
            source = REAL_ROOT / "skill3d_splits" / f"{SOURCE_NAMES[benchmark]}_{split}.jsonl"
            with source.open() as stream:
                ids = [json.loads(line)["id"] for line in stream if line.strip()]
            assert [p.sample_id for p in problems] == ids
            for problem in problems:
                assert problem.benchmark == benchmark
                assert problem.source_split == split
                assert 1 <= len(problem.image_paths) <= 6
                assert set(problem.public_dict()) == PUBLIC_KEYS
                assert all(Path(p).is_absolute() and Path(p).is_file() for p in problem.image_paths)
                assert len(problem.input_sha256) == 64
                image_counts.add(len(problem.image_paths))
    assert image_counts == {1, 2, 3, 4, 6}


def test_all_real_answers_normalize_host_only(real_problems):
    for (benchmark, split), problems in real_problems.items():
        answers = load_evaluation_answers(REAL_ROOT, benchmark, split)
        assert list(answers) == [p.sample_id for p in problems]
        assert all(answers[p.sample_id] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(p.options)] for p in problems)


def test_all_real_records_manifest_replay(real_problems):
    problems = [p for split_problems in real_problems.values() for p in split_problems]
    manifest = freeze_manifest(problems)
    assert len(manifest["problems"]) == 1855
    assert load_manifest(manifest, REAL_ROOT, expected_sha256=manifest["manifest_sha256"]) == problems
    verify_manifest(manifest, problems, REAL_ROOT)
