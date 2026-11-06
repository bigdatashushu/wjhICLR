"""Guard immutable dataset creation and host-only integrity anchors."""

import importlib.util
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "metric_dataset_update", ROOT / "scripts/update_metric_dataset.py")
update = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(update)


def test_existing_package_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(update, "ROOT", tmp_path)
    package = tmp_path / "prepared_cache" / update.NAME
    package.mkdir(parents=True)
    (package / "questions.jsonl").write_text('{"qa_id":"synthetic"}\n')
    before = update.digest(package / "questions.jsonl")
    with pytest.raises(ValueError, match="immutable destination already exists"):
        update.build(Path("/unused"), Path("/unused"), package)
    assert update.digest(package / "questions.jsonl") == before


def test_build_rejects_destination_outside_runner(tmp_path):
    with pytest.raises(ValueError, match="destination must be runner cache"):
        update.build(Path("/unused"), Path("/unused"), tmp_path / "package")
    assert not (tmp_path / "package").exists()


@pytest.mark.skipif(
    not (ROOT / "prepared_cache" / update.NAME / "sampling_manifest.json").is_file(),
    reason="requires external frozen metric dataset integrity anchors",
)
def test_host_label_anchor_rejects_altered_scoring_file(tmp_path):
    package = ROOT / "prepared_cache" / update.NAME
    for name in ("sampling_manifest.json", "eligibility/replacement_proposal.json",
                 "eligibility/native_report.json", "questions.jsonl",
                 "train.jsonl", "test.jsonl", "host_evaluation/labels.jsonl"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / name, target)
    with (tmp_path / "host_evaluation/labels.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="manifest file anchor changed"):
        update.validate(tmp_path)
