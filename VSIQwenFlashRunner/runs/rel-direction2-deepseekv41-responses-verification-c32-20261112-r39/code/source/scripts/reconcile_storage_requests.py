#!/usr/bin/env python3
"""Derive a sidecar ledger from immutable local and gateway request evidence."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(root):
    manifest = read(root / "diagnostic.json")
    storage = read(root / "storage_interruption_receipt.json")
    collection = read(root / "collection_receipt.json")
    assert storage["origin_identity"] == manifest["identity_sha256"]
    assert storage["online_processes_alive"] == []
    assert storage["selection_used_reference_answers"] is False
    assert sha(root / "requests/state.json") == storage["raw_requests_state_sha256"]
    for name, value in storage["log_sha256"].items():
        assert sha(root / name) == value
    assert b"Errno 28" in (root / "launcher.log").read_bytes()
    exit_status = read(root / "exit.json")
    assert exit_status["status"] == "stopped" and exit_status["postmortem"] is True
    assert exit_status["exit_code"] is exit_status["ended_at"] is None
    assert exit_status["storage_receipt_sha256"] == sha(root / "storage_interruption_receipt.json")
    assert collection["diagnostic_identity"] == manifest["identity_sha256"]
    for name, value in collection["files"].items():
        assert sha(root / name) == value["sha256"]
    evidence = read(root / "gateway_request_evidence_receipt.json")
    assert evidence["origin_identity"] == manifest["identity_sha256"]
    assert sha(root / "gateway_storage_snapshot.sqlite3") == evidence["database_sha256"]
    db = sqlite3.connect("file:" + str(root / "gateway_storage_snapshot.sqlite3") + "?mode=ro",
                         uri=True)
    db.row_factory = sqlite3.Row
    gateway = {r["payload_sha256"]: dict(r) for r in db.execute(
        "SELECT r.*,a.started_at_ns,a.ended_at_ns,a.success,a.error_type,"
        "a.usage_status,a.token_usage FROM requests r JOIN upstream_audit a USING(request_id)")}
    assert len(gateway) == len(evidence["requests"]) == 36
    saved_gateway = {r["payload_sha256"]: r for r in evidence["requests"]}
    for key, row in gateway.items():
        saved = saved_gateway[key]
        reply = row["reply"].encode() if isinstance(row["reply"], str) else row["reply"]
        assert hashlib.sha256(reply).hexdigest() == saved["reply_sha256"]
        for field in ("request_id", "state", "http_status", "started_at_ns", "ended_at_ns",
                      "success", "error_type", "usage_status"):
            assert row[field] == saved[field]
        audit_usage = json.loads(row["token_usage"]) if row["token_usage"] else {}
        assert audit_usage == saved.get("audit_usage", saved["usage"])
        if row["success"]:
            assert json.loads(reply)["last_usage"] == saved["usage"]
    state = read(root / "requests/state.json")
    original = state["requests"]
    assert Counter(r["status"] for r in original) == {
        "completed": 33, "error": 1, "started": 2}
    sys.path.insert(0, str(root / "code/source/src"))
    from skill3d.evaluation.fitset_v11 import _variant_key
    normalized = []
    mapping = []
    consumed = set()
    for variant in manifest["variants"]:
        variant_id = variant["variant_id"]
        rows = sorted([r for r in original if r["context"]["variant_id"] == variant_id],
                      key=lambda r: r["request_id"])
        seed_paths = list((root / "system/request_seeds").glob(_variant_key(variant_id) + "*.json"))
        seeds = sorted([(read(p), p) for p in seed_paths
                        if "request_identity" in read(p)], key=lambda x: x[0]["request_index"])
        assert len(rows) == len(seeds)
        for row, (seed, path) in zip(rows, seeds):
            key = seed["request_identity"]["wire_payload_sha256"]
            assert key not in consumed
            consumed.add(key)
            response = saved_gateway[key]
            result = copy.deepcopy(row)
            if row["status"] == "completed":
                assert response["success"] and response["http_status"] == 200
                assert row["usage"] == response["audit_usage"]
            elif row["status"] == "error":
                assert not response["success"]
                assert row["exception_type"] == "DiagnosticOperationalStop"
                assert row["usage"] == response["usage"] == {}
            else:
                assert response["success"] and response["http_status"] == 200
                result.update(status="completed", completed_at=response["ended_at_ns"] / 1e9,
                              usage=response["audit_usage"], exception_type=None)
            result["gateway_reconciliation"] = {
                "original_status": row["status"], "gateway_request_id": response["request_id"],
                "wire_payload_sha256": key, "source_seed_sha256": sha(path),
                "terminal_status_source": ("gateway" if row["status"] == "started"
                                           else "original_ledger_and_gateway"),
            }
            normalized.append(result)
            mapping.append({"request_id": row["request_id"], "variant_id": variant_id,
                            "request_index": seed["request_index"], "seed_path": str(path.relative_to(root)),
                            **result["gateway_reconciliation"]})
    assert consumed == set(gateway)
    normalized.sort(key=lambda r: r["request_id"])
    done = {read(p)["variant_id"] for p in (root / "system/results").glob("*.json")}
    expected = {r["variant_id"] for r in manifest["variants"]}
    assert sorted(done) == storage["completed_variant_ids"]
    assert sorted(expected - done) == storage["missing_variant_ids"]
    assert {r["context"]["variant_id"] for r in original if r["status"] == "started"} == {
        "1558@g1m0", "3588@g1m0"}
    usage = Counter()
    for row in normalized:
        usage.update({k: v for k, v in row["usage"].items()
                      if isinstance(v, (int, float)) and not isinstance(v, bool)})
    discarded = [r for r in normalized if r["context"]["variant_id"] in expected - done
                 and r["status"] == "completed"]
    return {
        "schema_version": "storage-request-reconciliation/v2",
        "status": "verified_sidecar_reconciliation",
        "origin_identity": manifest["identity_sha256"],
        "original_state_sha256": sha(root / "requests/state.json"),
        "gateway_database_sha256": evidence["database_sha256"],
        "gateway_evidence_sha256": sha(root / "gateway_request_evidence_receipt.json"),
        "storage_receipt_sha256": sha(root / "storage_interruption_receipt.json"),
        "collection_receipt_sha256": sha(root / "collection_receipt.json"),
        "original_status_counts": dict(Counter(r["status"] for r in original)),
        "reconciled_status_counts": dict(Counter(r["status"] for r in normalized)),
        "original_state_rewritten": False,
        "reconciled_unterminated_successful_requests": 2,
        "discarded_episode_successful_requests": len(discarded),
        "discarded_episode_known_total_tokens": sum(r["usage"]["total_tokens"] for r in discarded),
        "discarded_episode_known_tokens_included": True,
        "known_usage": dict(usage), "requests": normalized, "mapping": mapping,
    }


def verify(root):
    result = derive(root)
    assert read(root / "reconciled_requests_receipt.json") == result
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    root = parser.parse_args().run_root.resolve()
    value = derive(root)
    path = root / "reconciled_requests_receipt.json"
    if path.exists() and read(path) != value:
        previous = read(path)
        assert "schema_version" not in previous
        assert previous["original_state_sha256"] == value["original_state_sha256"]
        assert previous["requests"] == value["requests"]
        archived = root / "reconciled_requests_receipt.v1.json"
        assert not archived.exists()
        path.rename(archived)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in value.items() if k not in {"requests", "mapping"}},
                     ensure_ascii=False))
