"""Native materialized members retain byte identity and the host memory cap."""
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[2] / "scripts/run_reviser_successor_v11.py"
spec = importlib.util.spec_from_file_location("successor_learning_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_large_member_uses_native_materialization_and_rejects_changed_bytes(tmp_path, monkeypatch):
    psutil = ModuleType("psutil")
    psutil.virtual_memory = lambda: SimpleNamespace(available=32 * 1024 ** 3)
    monkeypatch.setitem(sys.modules, "psutil", psutil)
    payload = b"frozen learning bytes"
    path = tmp_path / "member.json"
    path.write_bytes(payload)
    record = SimpleNamespace(size=len(payload), ref="retained-member",
                             sha256=hashlib.sha256(payload).hexdigest())

    @contextmanager
    def materialize(ref):
        assert ref == record.ref
        yield path

    history = SimpleNamespace(
        _cold_history=SimpleNamespace(max_read_bytes=1, materialize=materialize),
        cold_reference=lambda path: record,
        read_history_bytes=lambda path: pytest.fail("large member used bounded memory reader"))
    assert module.learning_bytes(history, path) == payload
    path.write_bytes(b"changed bytes")
    with pytest.raises(RuntimeError, match="identity changed"):
        module.learning_bytes(history, path)
    record.size = 513 * 1024 * 1024
    with pytest.raises(RuntimeError, match="memory allowance"):
        module.learning_bytes(history, path)
    record.size = 300 * 1024 * 1024
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=1024 ** 3))
    with pytest.raises(RuntimeError, match="headroom"):
        module.learning_bytes(history, path)
