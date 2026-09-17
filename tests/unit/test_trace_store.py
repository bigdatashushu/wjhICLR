"""M13 TraceStore 单测：JSONL 追加不损坏、行数一致。"""

import json
import threading

from skill3d.schemas import EpisodeTrace
from skill3d.trace.store import TraceStore


def _trace(i: int) -> EpisodeTrace:
    return EpisodeTrace(
        episode_id=f"ep-{i}",
        qa_id=f"qa-{i}",
        final_state="EVALUATE",
        program_trace_ref="ref://p",
        geometry_check_ref="ref://g",
        evaluation_ref="ref://e",
        failure=None,
        active_snapshot_ref="snap://v0",
    )


def test_jsonl_append_and_count(tmp_path):
    store = TraceStore(tmp_path)
    n = 10
    for i in range(n):
        store.append_episode(_trace(i))
    path = tmp_path / "episode_trace.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert rec["episode_id"] == f"ep-{i}"  # 行级不损坏且保序


def test_append_continues_across_instances(tmp_path):
    TraceStore(tmp_path).append_episode(_trace(0))
    TraceStore(tmp_path).append_episode(_trace(1))  # 新实例继续追加
    lines = (tmp_path / "episode_trace.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_thread_safe_append(tmp_path):
    store = TraceStore(tmp_path)
    n = 50

    def worker(base):
        for i in range(n):
            store.append("mt", {"i": base + i})

    threads = [threading.Thread(target=worker, args=(t * 1000,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = (tmp_path / "mt.jsonl").read_text().splitlines()
    assert len(lines) == 4 * n
    for line in lines:
        json.loads(line)  # 每行都是合法 JSON（无交错损坏）


def test_build_parquet_skips_without_pyarrow(tmp_path):
    store = TraceStore(tmp_path)
    store.append("t", {"a": 1})
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        assert store.build_parquet("t") is None  # 不可用则跳过并警告
    else:
        out = store.build_parquet("t")
        assert out is not None and out.exists()
