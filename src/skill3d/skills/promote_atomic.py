"""Snapshot reads, publication lock and atomic rollback for v11."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class PublishLockedError(RuntimeError):
    """发布锁被占用（§10.1-1 读取并锁定父快照）。"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def publish_lock(store_dir: Path):
    """§10.1-1 的"锁定父快照"：以 O_EXCL 独占锁文件阻止并发发布。

    锁只是**发布动作**的互斥（防两次 promote 交错写指针）；它不阻塞在线读取 ——
    在线链读的是原子替换后的指针，见 `_write_pointer_atomic`。
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    lock = store_dir / ".publish.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise PublishLockedError(
            f"发布锁已被占用（{lock}）；确认没有并发 publish 后手动清理") from exc
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "at": _now_iso()}))
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _snapshot_path(store_dir: Path, snapshot_id: str) -> Path:
    return store_dir / f"snapshot_{snapshot_id}.json"


def read_snapshot(store_dir: str | Path, snapshot_id: str) -> dict:
    """读**指定**快照（发布时读父快照，§10.1-1）。"""
    path = _snapshot_path(Path(store_dir), snapshot_id)
    if not path.exists():
        raise FileNotFoundError(f"快照不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_active_snapshot(store_dir: str | Path) -> dict:
    """读 active 指针指向的 snapshot；无 active 时返回空初始 snapshot。"""
    store_dir = Path(store_dir)
    pointer = store_dir / "active_snapshot.json"
    if not pointer.exists():
        return {"snapshot_id": "genesis", "entries": {}, "parent_snapshot_id": None,
                "created_at": _now_iso()}
    ref = json.loads(pointer.read_text(encoding="utf-8"))
    return read_snapshot(store_dir, ref["snapshot_id"])


def rollback(store_dir: str | Path, snapshot_before: str,
             promotion_log: list | None = None) -> dict:
    """Validate the complete v11 parent before atomically restoring its pointer."""
    from .v11_library import resolve_source_like_ref, write_v11_snapshot

    store_dir = Path(store_dir)
    with publish_lock(store_dir):
        snap = read_snapshot(store_dir, snapshot_before)
        if snap.get("schema_version") != "runtime-skill-snapshot/2.0":
            raise ValueError("回滚目标必须是 runtime-skill-snapshot/2.0")
        manifest_path = resolve_source_like_ref(store_dir.parent, snap["manifest_ref"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        write_v11_snapshot(store_dir.parent, snap, manifest, activate=True)
    if promotion_log is not None:
        promotion_log.append({"rollback_to": snapshot_before, "timestamp": _now_iso()})
    return snap
