"""Label-free BLINK/CV3D/MMSI inputs and content-verified replay manifests.

This module is independent of VSI. Only ``load_evaluation_answers`` reads
assistant turns; its return value belongs to the host evaluator, never prompts.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "SmallImageProblem", "canonical_benchmark", "load_problems",
    "load_evaluation_answers", "normalize_answer", "freeze_manifest",
    "load_manifest", "verify_manifest",
]

_SOURCE_NAMES = {
    "blink": "blink_multiview", "cv3d": "cv3d", "mmsi": "mmsi_pr",
}
_ALIASES = {
    "blink": "blink", "blink_multiview": "blink",
    "cv3d": "cv3d", "cv-3d": "cv3d", "cvbench": "cv3d",
    "mmsi": "mmsi", "mmsi_pr": "mmsi", "mmsi-bench": "mmsi",
}
_IMAGE_ROOTS = {
    "blink": ("BLINK_images",),
    "cv3d": ("CVBench/3D",),
    "mmsi": ("MMSI_images",),
}
_PUBLIC_KEYS = frozenset({
    "benchmark", "sample_id", "source_split", "question", "options",
    "image_paths", "task_type", "input_sha256",
})
_SCHEMA = "skill3d.small_image.manifest.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PLAIN_OPTION = re.compile(r"^\(([A-Z])\)[ \t]+", re.MULTILINE)
_COLON_OPTION = re.compile(r"(?:^\(|;[ \t]*)([A-Z]):[ \t]*", re.MULTILINE)
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def canonical_benchmark(benchmark: str) -> str:
    if not isinstance(benchmark, str) or benchmark.lower() not in _ALIASES:
        raise ValueError("benchmark must be blink, cv3d, or mmsi")
    return _ALIASES[benchmark.lower()]


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _split(split: str) -> str:
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    return split


def _limit(limit: int | None) -> None:
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("limit must be a nonnegative integer or None")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _image_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_sha256(record: Mapping[str, Any]) -> str:
    return _sha256({
        "schema": "skill3d.small_image.input.v1",
        "problem": {k: record[k] for k in sorted(_PUBLIC_KEYS - {"input_sha256"})},
        "image_sha256": [_image_sha256(p) for p in record["image_paths"]],
    })


@dataclass(frozen=True)
class SmallImageProblem:
    benchmark: str
    sample_id: str
    source_split: str
    question: str
    options: tuple[str, ...]
    image_paths: tuple[str, ...]
    task_type: str
    input_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark", canonical_benchmark(self.benchmark))
        _split(self.source_split)
        for name in ("sample_id", "question", "task_type"):
            _text(getattr(self, name), name)
        if not isinstance(self.options, tuple) or not 2 <= len(self.options) <= 26:
            raise ValueError("options must be a tuple of 2..26 option texts")
        for option in self.options:
            _text(option, "option")
        if not isinstance(self.image_paths, tuple) or not 1 <= len(self.image_paths) <= 6:
            raise ValueError("image_paths must be a tuple of 1..6 absolute paths")
        for path in self.image_paths:
            _text(path, "image path")
            if not Path(path).is_absolute() or not Path(path).is_file():
                raise ValueError("image path must be an existing absolute file")
        object.__setattr__(self, "input_sha256", _input_sha256(self._input_dict()))

    def _input_dict(self) -> dict[str, Any]:
        # Explicit whitelist: never serialize __dict__ or source-row metadata.
        return {
            "benchmark": self.benchmark,
            "sample_id": self.sample_id,
            "source_split": self.source_split,
            "question": self.question,
            "options": list(self.options),
            "image_paths": list(self.image_paths),
            "task_type": self.task_type,
        }

    def public_dict(self) -> dict[str, Any]:
        return {**self._input_dict(), "input_sha256": self.input_sha256}


def _dataset_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).expanduser().absolute()
    if not root.is_dir():
        raise ValueError("dataset_root must be an existing directory")
    return root


def _resolve_images(
    dataset_root: Path, benchmark: str, paths: Sequence[str], *, absolute: bool = False,
) -> tuple[str, ...]:
    if not isinstance(paths, (list, tuple)) or not 1 <= len(paths) <= 6:
        raise ValueError("image list must contain 1..6 paths")
    resolved: list[str] = []
    for raw in paths:
        _text(raw, "image path")
        path = Path(raw)
        if ".." in path.parts or path.is_absolute() != absolute:
            raise ValueError("image path has invalid containment or form")
        candidate = path if absolute else dataset_root / path
        allowed = False
        for relative_root in _IMAGE_ROOTS[benchmark]:
            image_root = dataset_root / relative_root
            # Trust only the named image-root symlinks, not symlinks below them.
            if not image_root.is_dir():
                continue
            lexical = candidate.is_relative_to(image_root)
            physical_root = image_root.resolve(strict=True)
            physical = candidate.resolve(strict=True) if candidate.is_file() else None
            if physical is not None and physical.is_relative_to(physical_root):
                if lexical or (absolute and candidate.is_relative_to(physical_root)):
                    allowed = True
                    break
        if not allowed or not candidate.is_file():
            raise ValueError("image is missing or outside its benchmark image root")
        resolved.append(str(candidate.resolve(strict=True)))
    return tuple(resolved)


def _source_rows(
    dataset_root: str | Path, benchmark: str, split: str, limit: int | None,
) -> Iterable[tuple[Path, dict[str, Any]]]:
    _limit(limit)
    root = _dataset_root(dataset_root)
    stem = _SOURCE_NAMES[canonical_benchmark(benchmark)]
    filename = f"{stem}_{_split(split)}.jsonl"
    candidates = (
        root / "skill3d_splits" / filename,
        root.parent / "skill3d_splits" / "skill3d_splits" / filename,
    )
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise ValueError(f"missing source split: {filename}")
    with source.open(encoding="utf-8") as stream:
        count = 0
        for line_number, line in enumerate(stream, 1):
            if limit is not None and count >= limit:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"invalid JSON in {filename}:{line_number}") from None
            if not isinstance(row, dict):
                raise ValueError(f"source row must be an object: {filename}:{line_number}")
            yield root, row
            count += 1


def _human_prompt(row: Mapping[str, Any]) -> str:
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("missing conversations list")
    human = [
        turn for turn in conversations
        if isinstance(turn, dict) and turn.get("from") == "human"
    ]
    if len(human) != 1:
        raise ValueError("source must have exactly one human prompt")
    return _text(human[0].get("value"), "human prompt")


def _parse_options(prompt: str, benchmark: str) -> tuple[str, tuple[str, ...]]:
    colon = benchmark == "mmsi"
    markers = list((_COLON_OPTION if colon else _PLAIN_OPTION).finditer(prompt))
    if not 2 <= len(markers) <= 26:
        raise ValueError("prompt must contain 2..26 encoded options")
    if [m.group(1) for m in markers] != list(_LETTERS[:len(markers)]):
        raise ValueError("option labels must be consecutive and ordered from A")
    question = _text(prompt[:markers[0].start()].strip(), "question")
    options = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(prompt)
        option = prompt[marker.end():end].strip()
        if colon:
            semicolon_boundary = index + 1 < len(markers) and prompt[end] == ";"
            if not semicolon_boundary:
                if not option.endswith(")"):
                    raise ValueError("unterminated colon option")
                option = option[:-1].strip()
            # MMSI encodes multi-part choices as adjacent parenthesized lines.
            option = re.sub(r"\)[ \t]*\n[ \t]*\(", ", ", option)
        options.append(_text(option, "option"))
    return question, tuple(options)


def _problem(root: Path, benchmark: str, split: str, row: Mapping[str, Any]) -> SmallImageProblem:
    question, options = _parse_options(_human_prompt(row), benchmark)
    return SmallImageProblem(
        benchmark=benchmark,
        sample_id=_text(row.get("id"), "id"),
        source_split=split,
        question=question,
        options=options,
        image_paths=_resolve_images(root, benchmark, row.get("image")),
        task_type=_text(row.get("task"), "task"),
    )


def load_problems(
    dataset_root: str | Path, benchmark: str, split: str = "train", limit: int | None = None,
) -> list[SmallImageProblem]:
    """Load the actual split in source order, with no labels or synthetic padding."""
    benchmark = canonical_benchmark(benchmark)
    problems = []
    seen: set[str] = set()
    for root, row in _source_rows(dataset_root, benchmark, split, limit):
        problem = _problem(root, benchmark, split, row)
        if problem.sample_id in seen:
            raise ValueError("duplicate sample_id in source split")
        seen.add(problem.sample_id)
        problems.append(problem)
    return problems


def normalize_answer(answer: str, options: Sequence[str]) -> str:
    """Return a choice letter for an exact encoding; reject ambiguous option text."""
    if not isinstance(options, (list, tuple)) or not 2 <= len(options) <= 26:
        raise ValueError("options must contain 2..26 texts")
    for option in options:
        _text(option, "option")
    answer = _text(answer, "answer").strip()
    letters = _LETTERS[:len(options)]
    match = re.fullmatch(r"([A-Za-z])|\(([A-Za-z])\)|([A-Za-z])[.)]", answer)
    indices = {
        index for index, option in enumerate(options)
        if answer.casefold() == option.strip().casefold()
    }
    if match:
        letter = next(group for group in match.groups() if group).upper()
        if letter not in letters:
            raise ValueError("answer letter is out of range")
        indices.add(letters.index(letter))
    if len(indices) != 1:
        raise ValueError("answer must identify exactly one option")
    return letters[indices.pop()]


def load_evaluation_answers(
    dataset_root: str | Path, benchmark: str, split: str = "train", limit: int | None = None,
) -> dict[str, str]:
    """HOST ONLY: load assistant answers separately, keyed by sample_id."""
    benchmark = canonical_benchmark(benchmark)
    answers = {}
    for root, row in _source_rows(dataset_root, benchmark, split, limit):
        problem = _problem(root, benchmark, split, row)
        turns = [
            turn for turn in row["conversations"]
            if isinstance(turn, dict) and turn.get("from") == "gpt"
        ]
        if len(turns) != 1:
            raise ValueError("evaluator requires exactly one assistant answer")
        if problem.sample_id in answers:
            raise ValueError("duplicate sample_id in source split")
        answers[problem.sample_id] = normalize_answer(turns[0].get("value"), problem.options)
    return answers


def _checked_records(problems: Iterable[SmallImageProblem]) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for problem in problems:
        if not isinstance(problem, SmallImageProblem):
            raise ValueError("manifest accepts only SmallImageProblem records")
        record = problem.public_dict()
        if set(record) != _PUBLIC_KEYS:
            raise ValueError("public record contains missing or non-whitelisted fields")
        if _input_sha256(record) != problem.input_sha256:
            raise ValueError("input identity mismatch: image bytes changed")
        key = (problem.benchmark, problem.source_split, problem.sample_id)
        if key in seen:
            raise ValueError("duplicate problem identity in manifest")
        seen.add(key)
        records.append(record)
    return records


def freeze_manifest(
    problems: Iterable[SmallImageProblem], path: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze ordered public records; image-byte hashes are part of input identity."""
    payload = {"schema": _SCHEMA, "problems": _checked_records(problems)}
    manifest = {**payload, "manifest_sha256": _sha256(payload)}
    if path is not None:
        with Path(path).open("x", encoding="ascii") as stream:
            stream.write(_canonical(manifest).decode("ascii") + "\n")
    return manifest


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON keys in manifest")
        result[key] = value
    return result


def _read_manifest(path_or_mapping: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_mapping, Mapping):
        manifest = dict(path_or_mapping)
    else:
        try:
            with Path(path_or_mapping).open(encoding="utf-8") as stream:
                manifest = json.load(stream, object_pairs_hook=_unique_object)
        except json.JSONDecodeError:
            raise ValueError("invalid manifest JSON") from None
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema", "problems", "manifest_sha256",
    }:
        raise ValueError("manifest has missing or non-whitelisted fields")
    if manifest["schema"] != _SCHEMA or not isinstance(manifest["problems"], list):
        raise ValueError("invalid manifest schema or problem list")
    for record in manifest["problems"]:
        if not isinstance(record, dict) or set(record) != _PUBLIC_KEYS:
            raise ValueError("manifest record has missing or non-whitelisted fields")
        if not isinstance(record["options"], list) or not isinstance(record["image_paths"], list):
            raise ValueError("manifest options and image_paths must be lists")
        if not isinstance(record["input_sha256"], str) or not _DIGEST.fullmatch(record["input_sha256"]):
            raise ValueError("invalid input_sha256")
    digest = manifest["manifest_sha256"]
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise ValueError("invalid manifest_sha256")
    if digest != _sha256({"schema": manifest["schema"], "problems": manifest["problems"]}):
        raise ValueError("manifest identity mismatch")
    return manifest


def load_manifest(
    path_or_mapping: str | Path | Mapping[str, Any],
    dataset_root: str | Path,
    *,
    expected_sha256: str | None = None,
) -> list[SmallImageProblem]:
    """Replay only exact whitelisted records, validating containment and image bytes."""
    manifest = _read_manifest(path_or_mapping)
    if expected_sha256 is not None and manifest["manifest_sha256"] != expected_sha256:
        raise ValueError("expected manifest identity mismatch")
    root = _dataset_root(dataset_root)
    problems = []
    for record in manifest["problems"]:
        benchmark = canonical_benchmark(record["benchmark"])
        problem = SmallImageProblem(
            benchmark=benchmark,
            sample_id=record["sample_id"],
            source_split=record["source_split"],
            question=record["question"],
            options=tuple(record["options"]),
            image_paths=_resolve_images(root, benchmark, record["image_paths"], absolute=True),
            task_type=record["task_type"],
        )
        if problem.public_dict() != record:
            raise ValueError("replay input identity mismatch")
        problems.append(problem)
    _checked_records(problems)
    return problems


def verify_manifest(
    path_or_mapping: str | Path | Mapping[str, Any],
    problems: Iterable[SmallImageProblem],
    dataset_root: str | Path,
) -> None:
    """Check a frozen manifest against current ordered inputs and actual image bytes."""
    replay = load_manifest(path_or_mapping, dataset_root)
    if _checked_records(replay) != _checked_records(problems):
        raise ValueError("replay inputs differ from current ordered problems")
