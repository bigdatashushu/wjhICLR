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

    def topics(self) -> list[str]:
        """已有 JSONL 的 topic 列表（聚合入口用）。"""
        return sorted(p.stem for p in self.root.glob("*.jsonl"))

    def build_all_parquet(self, out_dir: Union[str, Path, None] = None) -> dict[str, Path]:
        """把全部 topic 的 JSONL 聚合为 Parquet（§4 M13）+ manifest 记录。

        `out_dir` 给定则写到该目录下同名 `.parquet`；否则写到各 JSONL 同目录。
        返回 `{topic: parquet_path}`；pyarrow 缺失时返回空 dict 并已警告（不抛）。
        """
        base = Path(out_dir) if out_dir else None
        if base is not None:
            base.mkdir(parents=True, exist_ok=True)
        out: dict[str, Path] = {}
        for topic in self.topics():
            target = (base / f"{topic}.parquet") if base is not None else None
            p = self.build_parquet(topic, target)
            if p is not None:
                out[topic] = p
        if out:
            manifest = {
                "topics": {k: str(v) for k, v in sorted(out.items())},
                "source_dir": str(self.root),
                "n_topics": len(out),
            }
            (base or self.root).joinpath("parquet_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return out


def main(argv: list[str] | None = None) -> int:
    """CLI：JSONL → Parquet 聚合（§4 M13）。

    ```bash
    python -m skill3d.trace.store --trace-dir data/traces
    python -m skill3d.trace.store --trace-dir data/traces --out-dir data/parquet
    ```
    """
    import argparse

    ap = argparse.ArgumentParser(description="M13 Trace Store：JSONL → Parquet 聚合")
    ap.add_argument("--trace-dir", required=True, help="含 <topic>.jsonl 的 trace 目录")
    ap.add_argument("--out-dir", default=None, help="Parquet 输出目录（默认与 JSONL 同目录）")
    args = ap.parse_args(argv)

    store = TraceStore(args.trace_dir)
    made = store.build_all_parquet(args.out_dir)
    if not made:
        print("无可聚合 topic（或 pyarrow 未安装）")
        return 1
    for topic, path in sorted(made.items()):
        print(f"{topic} → {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
