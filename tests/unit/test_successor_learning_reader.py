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


def isolated_history(payload):
    record = SimpleNamespace(
        size=len(payload), ref="retained-member",
        sha256=hashlib.sha256(payload).hexdigest(),
        ack=object(), member_key="verified-archive-member")
    events = []

    @contextmanager
    def lease(refs):
        assert refs == (record.ref,)
        events.append("lease")
        yield
        events.append("released")

    def fetch(ack, key, writer):
        assert ack is record.ack and key == record.member_key
        events.append("fetch")
        writer.write(payload)

    cold = SimpleNamespace(
        lease=lease, validate_reference=lambda ref: record, fetch_member=fetch)
    history = SimpleNamespace(
        _cold_history=cold, cold_reference=lambda path: record,
        read_history_bytes=lambda path: pytest.fail("used shared-cache materialization"))
    return history, record, events


def test_isolated_reader_preserves_native_transport_identity_and_cleans_up(tmp_path):
    payload = b"the exact original learning member"
    history, _, events = isolated_history(payload)
    root = tmp_path / "independent"
    assert module.isolated_learning_bytes(history, tmp_path / "original", root) == payload
    assert events == ["lease", "fetch", "released"]
    assert not list(root.glob("member-*"))
    assert root.stat().st_mode & 0o777 == 0o700


def test_isolated_reader_rejects_corruption_and_leaves_no_member(tmp_path):
    from skill3d.storage.hot_cache import IntegrityError
    history, record, _ = isolated_history(b"original")
    history._cold_history.fetch_member = lambda ack, key, writer: writer.write(b"tampered")
    root = tmp_path / "independent"
    with pytest.raises(IntegrityError):
        module.isolated_learning_bytes(history, tmp_path / "original", root)
    assert not list(root.glob("member-*"))
    record.size = 513 * 1024 * 1024
    with pytest.raises(RuntimeError, match="memory allowance"):
        module.isolated_learning_bytes(history, tmp_path / "original", root)


def test_isolated_reader_rejects_changed_reference_and_unfinished_read(tmp_path):
    history, record, events = isolated_history(b"original")
    history._cold_history.validate_reference = lambda ref: SimpleNamespace(
        ref=ref, size=record.size, sha256="0" * 64)
    root = tmp_path / "independent"
    with pytest.raises(RuntimeError, match="reference changed"):
        module.isolated_learning_bytes(history, tmp_path / "original", root)
    assert "fetch" not in events
    (root / "member-unfinished").mkdir()
    with pytest.raises(RuntimeError, match="requires inspection"):
        module.isolated_learning_bytes(history, tmp_path / "original", root)


def test_isolated_reader_respects_disk_floor_before_fetch(tmp_path, monkeypatch):
    import psutil
    history, _, events = isolated_history(b"original")
    monkeypatch.setattr(psutil, "disk_usage", lambda path: SimpleNamespace(free=25_000_000_000))
    with pytest.raises(RuntimeError, match="free-space floor"):
        module.isolated_learning_bytes(history, tmp_path / "original", tmp_path / "cache")
    assert "fetch" not in events
