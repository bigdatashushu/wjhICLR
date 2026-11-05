"""Unused reservations can be released only against immutable host receipts."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from Ominirunner.budget import completed_mmsi_release, MMSI_COMPLETIONS, MMSI_LEDGERS, RELEASE_PATH
from Ominirunner.common import digest, sha, write


def release_fixture(root):
    rows = {name: [{"status": "completed", "request_id": name}] for name in MMSI_LEDGERS}
    for name in MMSI_COMPLETIONS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, {"status": "complete"})
    receipt = {"schema": "omni-completed-reservation-release/1", "status": "released",
               "active_budget_slots": 0, "model_requests": 0, "reference_answers_returned": False,
               "released_unused_requests": 1143,
               "completion_files": {name: sha(root / name) for name in MMSI_COMPLETIONS},
               "ledger_snapshots": {name: {"count": len(value), "requests_sha256": digest(value)}
                                    for name, value in rows.items()}}
    path = root / RELEASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, receipt)
    return rows, receipt


def test_release_preserves_charges_and_requires_unchanged_request_and_completion_identities(tmp_path):
    assert completed_mmsi_release(tmp_path, {}) is None
    rows, receipt = release_fixture(tmp_path)
    checked = completed_mmsi_release(tmp_path, rows)
    assert checked["released_unused_requests"] == 1143 and checked["historical_charges_preserved"]
    rows[MMSI_LEDGERS[0]].append({"status": "started", "request_id": "new"})
    with pytest.raises(ValueError, match="completed_reservation_ledger_changed"):
        completed_mmsi_release(tmp_path, rows)
    rows[MMSI_LEDGERS[0]].pop()
    write(tmp_path / MMSI_COMPLETIONS[0], {"status": "changed"})
    with pytest.raises(ValueError, match="completed_reservation_evidence_changed"):
        completed_mmsi_release(tmp_path, rows)


@pytest.mark.parametrize("field,value", [("status", "pending"), ("active_budget_slots", 1),
                                       ("model_requests", 1), ("reference_answers_returned", True)])
def test_release_rejects_unverified_or_active_work(tmp_path, field, value):
    rows, receipt = release_fixture(tmp_path)
    receipt[field] = value
    write(tmp_path / RELEASE_PATH, receipt)
    with pytest.raises(ValueError, match="completed_reservation_release_invalid"):
        completed_mmsi_release(tmp_path, rows)
