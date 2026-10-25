"""Execution adapter must preserve the original receipt and exact backup bytes."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import fullqa_custody as c
import fullqa_native_acceptance as adapter


@pytest.mark.parametrize("final", [False, True])
def test_original_main_execution_and_receipt_preservation(tmp_path, monkeypatch, final):
    root, home = tmp_path / "run", tmp_path / "ops"
    backup = root / "backups/queue-online.sqlite3"
    backup.parent.mkdir(parents=True)
    with sqlite3.connect(backup) as db:
        db.execute("CREATE TABLE fixture(value TEXT)")
    original = types.SimpleNamespace(RUN="fixture", WORKER_PROBE="worker probe",
                                     TITAN_PROBE="titan probe", OUTPUT=None)
    called = []

    def main():
        called.append("original.main")
        original.remote("3090", original.WORKER_PROBE, original.RUN, c.METHOD, final)
        original.remote("titan", original.TITAN_PROBE)
        value = {"collector_sha256": c.COLLECTOR, "status": "verified" if final else "partial"}
        if final:
            raw = original.subprocess.run(["ssh", "fixture", "cat " + str(backup)]).stdout
            copy = original.OUTPUT / "backups/exact.sqlite3"
            copy.parent.mkdir()
            copy.write_bytes(raw)
            value["backup"] = str(copy)
        name = ("fullqa_final_publication_ack_verification.json" if final else
                "fullqa_partial_publication_ack_latest.json")
        (original.OUTPUT / "receipts" / name).write_text(json.dumps(value, indent=2) + "\n")

    original.main = main
    monkeypatch.setattr(c, "load_pinned", lambda *args: original)
    monkeypatch.setattr(c, "probe_source", lambda *args: {})
    monkeypatch.setattr(c, "http", lambda *args, **kwargs: {
        "probe_source_sha256": c.digest(original.WORKER_PROBE.encode()),
        "cache_staging": {"fixture": []}, "native": {}})
    value = adapter.collect(root, home, "http://fixture", final)
    assert called == ["original.main"]
    if final:
        source = home / "acceptance/receipts/fullqa_final_publication_ack_verification.json"
        assert (root / "receipts" / source.name).read_bytes() == source.read_bytes()
        assert Path(value["backup"]).read_bytes() == backup.read_bytes()
    receipt = c.read(home / ("final_acceptance_adapter.json" if final else
                             "partial_acceptance_adapter.json"))
    assert receipt["original_collector_main_executed"] is True
    assert receipt["original_receipt_bytes_preserved"] is True
