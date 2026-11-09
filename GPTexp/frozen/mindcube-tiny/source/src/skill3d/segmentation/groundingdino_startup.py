"""Resolve and load GroundingDINO's BERT dependency without network fallback."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping

BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
BERT_MODEL_ID = "bert-base-uncased"


class DetectorStartupError(RuntimeError):
    """A required local detector asset is missing or unusable."""


def validate_bert_snapshot(path: Path) -> Path:
    path = path.expanduser().resolve()
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DetectorStartupError(f"Cannot read local BERT config at {path}: {exc}") from exc
    if not isinstance(config, dict) or config.get("model_type") != "bert":
        raise DetectorStartupError(f"Local text encoder must be a BERT model: {path}")
    if not any((path / name).is_file() and (path / name).stat().st_size
               for name in ("vocab.txt", "tokenizer.json")):
        raise DetectorStartupError(f"Local BERT tokenizer files are missing or empty: {path}")
    weights = ("model.safetensors", "pytorch_model.bin",
               "model.safetensors.index.json", "pytorch_model.bin.index.json")
    if not any((path / name).is_file() and (path / name).stat().st_size for name in weights):
        raise DetectorStartupError(f"Local BERT weights are missing or empty: {path}")
    return path


def resolve_bert_snapshot(
    explicit_path: str | Path | None = None,
    *,
    revision: str = BERT_REVISION,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Use an explicit directory or the exact pinned snapshot in known caches."""
    env = os.environ if environ is None else environ
    requested = explicit_path or env.get("SKILL3D_BERT_PATH")
    if requested:
        # An invalid explicit choice must not silently switch to another model.
        return validate_bert_snapshot(Path(requested))
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise DetectorStartupError("BERT revision must be a 40-character Git commit")
    hf_home = Path(env.get("HF_HOME") or Path.home() / ".cache/huggingface").expanduser()
    roots = [Path(env[key]).expanduser() for key in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE")
             if env.get(key)]
    roots.extend([hf_home / "hub", hf_home / "transformers"])
    checked, errors = [], []
    for root in dict.fromkeys(roots):
        for name in ("models--bert-base-uncased", "models--google-bert--bert-base-uncased"):
            candidate = root / name / "snapshots" / revision
            checked.append(str(candidate))
            if candidate.is_dir():
                try:
                    return validate_bert_snapshot(candidate)
                except DetectorStartupError as exc:
                    errors.append(str(exc))
    raise DetectorStartupError(
        "No complete local BERT snapshot was found. Set SKILL3D_BERT_PATH or "
        "--bert-path to an existing snapshot directory. Automatic downloads are disabled.\n"
        + "\n".join(errors + checked))


def load_local_bert(snapshot: Path):
    """Actually deserialize both assets before starting the detector service."""
    from transformers import AutoTokenizer, BertModel

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    encoder = BertModel.from_pretrained(str(snapshot), local_files_only=True)
    return tokenizer, encoder


def bind_local_bert(loaders, snapshot: Path, tokenizer, encoder) -> None:
    """Override the external package's two model-ID loaders in this process only."""
    def check_requested(value):
        if str(value) == BERT_MODEL_ID or str(value) == str(snapshot):
            return
        raise DetectorStartupError(
            f"Detector requested unexpected text encoder {value!r}; "
            f"only the verified local BERT snapshot {snapshot} is allowed")

    def get_tokenizer(value):
        check_requested(value)
        return tokenizer

    def get_encoder(value):
        check_requested(value)
        return encoder

    loaders.get_tokenlizer = get_tokenizer
    loaders.get_pretrained_language_model = get_encoder


def snapshot_identity(snapshot: Path) -> dict:
    files = []
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(snapshot).parts:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        files.append({"path": path.relative_to(snapshot).as_posix(),
                      "size": path.stat().st_size, "sha256": digest.hexdigest()})
    return {"path": str(snapshot), "files": files}
