"""Run the original collector bytes using fixed authenticated server transports."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import time
import types

import fullqa_custody as c


def collect(root, home, endpoint, final):
    original = c.load_pinned("fullqa_server_original_collector",
                            home / "collect_fullqa_acceptance_v1.py", c.COLLECTOR)
    cleanup = {}

    def remote(node, source, *arguments):
        if node == "titan":
            c.require(source == original.TITAN_PROBE, "unregistered_titan_probe")
            return c.canon(c.probe_source(source, arguments))
        c.require(node == "3090" and source == original.WORKER_PROBE
                  and arguments == (original.RUN, c.METHOD, final), "unregistered_worker_probe")
        observed = c.http(endpoint, "/probe", {"final": final}, timeout=150)
        c.require(observed["probe_source_sha256"] == c.digest(source.encode()),
                  "worker_probe_source_changed")
        cleanup.update(observed["cache_staging"])
        return c.canon(observed["native"])

    def exact_backup_copy(command, **kwargs):
        # This is the original collector's only subprocess call after replacing
        # its remote(). Restrict the adapter to that exact SQLite file.
        expected = str(root / "backups/queue-online.sqlite3")
        c.require(command[-1] == "cat " + expected
                  and command[0] == "ssh", "unregistered_backup_transport")
        return types.SimpleNamespace(stdout=Path(expected).read_bytes(), returncode=0)

    output = home / "acceptance"
    (output / "receipts").mkdir(parents=True, exist_ok=True, mode=0o700)
    original.OUTPUT = output
    original.remote = remote
    original.subprocess = types.SimpleNamespace(
        PIPE=-1, run=exact_backup_copy)
    previous_argv = sys.argv
    sys.argv = [str(home / "collect_fullqa_acceptance_v1.py"), *(["--final"] if final else [])]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            original.main()
    finally:
        sys.argv = previous_argv
    name = ("fullqa_final_publication_ack_verification.json" if final
            else "fullqa_partial_publication_ack_latest.json")
    path = output / "receipts" / name
    raw = path.read_bytes()
    value = json.loads(raw)
    c.require(value["collector_sha256"] == c.COLLECTOR, "collector_execution_pin_changed")
    if final:
        c.durable_bytes(root / "receipts" / name, raw, immutable=True)
    c.durable(home / ("final_acceptance_adapter.json" if final else "partial_acceptance_adapter.json"), {
        "schema": "harness3d-fullqa-server-collector-adapter/1",
        "at": time.time(), "original_collector_main_executed": True,
        "collector_sha256": c.COLLECTOR, "collector_receipt_sha256": c.digest(raw),
        "collector_receipt_file": str(path), "cache_staging": cleanup,
        "execution_host": "Titan", "original_receipt_bytes_preserved": True,
        "adapter_sha256": c.digest(Path(__file__).read_bytes()),
        "model_requests": 0, "queue_mutated": False, "label_access": False})
    return value
