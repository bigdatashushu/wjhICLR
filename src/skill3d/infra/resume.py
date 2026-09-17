"""M21 断点恢复幂等：

- 实验单元完成即落盘（内容寻址 key=(snapshot, branch, episode, seed)）；
- 重跑只补未完成单元；
- 同 key 重放结果必须字节级一致，否则审计告警（M17/M21 验收）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def unit_key(snapshot_id: str, branch_id: str, episode_id: str, seed: int) -> str:
    """内容寻址单元 key。"""
    raw = json.dumps([snapshot_id, branch_id, episode_id, seed]).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ReplayMismatchError(AssertionError):
    """同 key 重放结果与已落盘结果字节级不一致。"""


class ResumeStore:
    """单元结果落盘与断点恢复。"""

    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.store_dir / f"unit_{key}.json"

    def is_done(self, key: str) -> bool:
        return self._path(key).exists()

    def record(self, key: str, result_bytes: bytes) -> None:
        """单元完成即落盘；同 key 已存在时必须字节级一致（幂等重放校验）。"""
        path = self._path(key)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))["result_hex"]
            if existing != result_bytes.hex():
                raise ReplayMismatchError(
                    f"单元 {key} 重放结果与已落盘不一致（字节级校验失败）"
                )
            return
        path.write_text(json.dumps(
            {"key": key, "result_hex": result_bytes.hex()}, ensure_ascii=False),
            encoding="utf-8")

    def load(self, key: str) -> bytes:
        return bytes.fromhex(json.loads(self._path(key).read_text(encoding="utf-8"))["result_hex"])

    def pending_units(self, keys: list[str]) -> list[str]:
        """重跑只补未完成单元。"""
        return [k for k in keys if not self.is_done(k)]
