"""TraceStore（§4 M13）：JSONL 流式追加 + Parquet 聚合。在线只写不读。

pyarrow lazy import，不可用时跳过聚合并警告。
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TraceStore:
    """按 topic 分文件的 JSONL 追加写；线程安全。"""

    def __init__(self, root_dir: Union[str, Path]) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _lock_for(self, topic: str) -> threading.Lock:
        with self._global_lock:
            return self._locks.setdefault(topic, threading.Lock())

    def _path(self, topic: str) -> Path:
        # topic 只允许安全文件名字符
        safe = "".join(c for c in topic if c.isalnum() or c in "-_")
        return self.root / f"{safe}.jsonl"

    def append(self, topic: str, record: Union[BaseModel, dict]) -> None:
        """流式追加一行（model_dump_json）；在线只写不读。"""
        line = record.model_dump_json() if isinstance(record, BaseModel) else json.dumps(
            record, default=str
        )
        with self._lock_for(topic):
            with open(self._path(topic), "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def append_episode(self, episode_trace: BaseModel) -> None:
        self.append("episode_trace", episode_trace)

    def build_parquet(self, topic: str, out_path: Union[str, Path, None] = None):
        """离线聚合 JSONL → Parquet；pyarrow 不可用时跳过并警告。"""
        try:
            import pyarrow as pa  # lazy import
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning("pyarrow 未安装，跳过 Parquet 聚合（topic=%s）", topic)
            return None

        src = self._path(topic)
        if not src.exists():
            logger.warning("topic=%s 无 JSONL，跳过聚合", topic)
            return None
        rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            return None
        table = pa.Table.from_pylist(rows)
        out = Path(out_path) if out_path else src.with_suffix(".parquet")
        pq.write_table(table, out)
        return out
