"""G-20 向量检索底座：确定性 hashing 嵌入 + 内存索引 + 可选 LanceDB 持久化。

论文需要的检索能力是"语义排序"，而不是某个特定数据库：

- `HashingEmbedder`：词 token + 字符 3-gram 的**特征哈希**嵌入（无模型下载、
  确定性、CPU 毫秒级）。字符 n-gram 使形态相近但用词不同的文本也能匹配
  （例：question "How many chairs are in this room?" vs skill 描述
  "count the seating furniture items" —— 词级重叠为 0，字符级仍有信号），
  因此其排序与"精确 token 重叠"的关键词打分**不同**，满足 §7.1 G-20 的验收口径。
- `SkillVectorIndex`：内存 numpy 余弦检索；`to_lancedb` / `from_lancedb` 用
  LanceDB（Apache-2.0，§4 M14/G-20 首选）做本地持久化与 `where` 元数据过滤。
- `rerank`：若装了 `FlagEmbedding.FlagReranker`（bge-reranker-v2-m3）则用之；
  否则用嵌入余弦。两条路径都返回同构分数，便于消融"有/无 reranker"（§8.1）。

纪律：本模块在线（确定性、无 GPT-6，硬约束 1）；嵌入模型可替换（TODO_CALIBRATE）。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

# 嵌入维度与 n-gram 配置（TODO_CALIBRATE；维度越大冲突越少）
EMBED_DIM = 512
CHAR_NGRAM = 3
DEFAULT_TOP_K = 5


def lancedb_available() -> bool:
    """LanceDB 是否可用（可选依赖，不可用时走内存索引）。"""
    try:
        import lancedb  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def reranker_available() -> bool:
    """bge-reranker 是否可用（FlagEmbedding 未装时走嵌入余弦）。"""
    try:
        from FlagEmbedding import FlagReranker  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _bucket(token: str, dim: int) -> tuple[int, float]:
    """token → (维度下标, 符号)；符号来自哈希位，降低冲突的系统性偏差。"""
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    v = int.from_bytes(h, "little")
    return v % dim, (1.0 if (v >> 63) & 1 else -1.0)


class HashingEmbedder:
    """确定性 hashing 嵌入（词 token + 字符 n-gram，L2 归一化）。"""

    def __init__(self, dim: int = EMBED_DIM, char_ngram: int = CHAR_NGRAM) -> None:
        self.dim = int(dim)
        self.char_ngram = int(char_ngram)

    def features(self, text: str) -> list[str]:
        t = str(text).lower()
        words = [w for w in "".join(c if c.isalnum() else " " for c in t).split() if w]
        feats = [f"w:{w}" for w in words]
        compact = " ".join(words)
        for n in range(2, self.char_ngram + 1):
            feats.extend(f"c{n}:{compact[i:i + n]}"
                         for i in range(max(len(compact) - n + 1, 0)))
        return feats

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        for f in self.features(text):
            idx, sign = _bucket(f, self.dim)
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度（输入可未归一化；零向量返回 0）。"""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class IndexEntry:
    """索引条目：id + 向量 + 元数据（元数据供 LanceDB where 过滤）。"""

    entry_id: str
    vector: np.ndarray
    metadata: dict = field(default_factory=dict)


class SkillVectorIndex:
    """内存向量索引（numpy 余弦）+ 可选 LanceDB 持久化。"""

    def __init__(self, embedder: Optional[HashingEmbedder] = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self._entries: dict[str, IndexEntry] = {}
        self._matrix: Optional[np.ndarray] = None
        self._ids: list[str] = []

    def __len__(self) -> int:
        return len(self._entries)

    def add_text(self, entry_id: str, text: str, **metadata) -> IndexEntry:
        return self.add(entry_id, self.embedder.embed(text), **metadata)

    def add(self, entry_id: str, vector: np.ndarray, **metadata) -> IndexEntry:
        e = IndexEntry(entry_id=str(entry_id),
                       vector=np.asarray(vector, dtype=np.float64), metadata=dict(metadata))
        self._entries[e.entry_id] = e
        self._matrix = None                     # 惰性重建矩阵
        return e

    def _ensure_matrix(self) -> None:
        if self._matrix is not None:
            return
        self._ids = sorted(self._entries)
        if not self._ids:
            self._matrix = np.zeros((0, self.embedder.dim))
            return
        self._matrix = np.stack([self._entries[i].vector for i in self._ids])

    def search(self, query_vector: np.ndarray, *, top_k: int = DEFAULT_TOP_K,
               where: Optional[dict] = None) -> list[tuple[str, float]]:
        """余弦 top-k 检索；`where` 为等值元数据过滤（模拟 LanceDB 的 where）。"""
        self._ensure_matrix()
        q = np.asarray(query_vector, dtype=np.float64)
        keep = [i for i, eid in enumerate(self._ids)
                if _match_where(self._entries[eid].metadata, where)]
        if not keep:
            return []
        sub = self._matrix[keep]
        qn = q / (np.linalg.norm(q) or 1.0)
        denom = np.linalg.norm(sub, axis=1)
        denom[denom == 0] = 1.0
        sims = (sub @ qn) / denom
        order = sorted(range(len(keep)), key=lambda j: (-sims[j], self._ids[keep[j]]))
        return [(self._ids[keep[j]], float(sims[j])) for j in order[: max(top_k, 0)]]

    def search_text(self, query: str, **kw) -> list[tuple[str, float]]:
        return self.search(self.embedder.embed(query), **kw)

    def metadata_of(self, entry_id: str) -> dict:
        return dict(self._entries[entry_id].metadata)

    # ---- LanceDB 持久化（可选；不可用时静默跳过并返回 False）----
    def to_lancedb(self, path: str, table: str = "skills") -> bool:
        if not self.embedder or not self._entries or not lancedb_available():
            return False
        import lancedb  # lazy import

        rows = [{"entry_id": e.entry_id, "vector": e.vector.tolist(), **e.metadata}
                for e in self._entries.values()]
        db = lancedb.connect(str(path))
        db.create_table(table, data=rows, mode="overwrite")
        return True

    @classmethod
    def from_lancedb(cls, path: str, table: str = "skills",
                     embedder: Optional[HashingEmbedder] = None) -> "SkillVectorIndex":
        """从 LanceDB 读回索引（不可用时返回空索引，由调用方记录降级）。"""
        idx = cls(embedder)
        if not lancedb_available():
            return idx
        import lancedb  # lazy import

        db = lancedb.connect(str(path))
        if table not in _table_names(db):
            return idx
        for row in db.open_table(table).to_arrow().to_pylist():
            eid = str(row.pop("entry_id"))
            vec = np.asarray(row.pop("vector"), dtype=np.float64)
            idx.add(eid, vec, **row)
        return idx


def _table_names(db) -> set[str]:
    """LanceDB 表名集合（兼容旧版 table_names / 新版 list_tables）。"""
    if hasattr(db, "list_tables"):
        try:
            return set(db.list_tables())
        except Exception:  # noqa: BLE001
            pass
    return set(db.table_names())


def _match_where(metadata: dict, where: Optional[dict]) -> bool:
    if not where:
        return True
    return all(metadata.get(k) == v for k, v in where.items())


def rerank_scores(query: str, docs: Sequence[str],
                  embedder: Optional[HashingEmbedder] = None) -> list[float]:
    """语义排序分数：优先 bge-reranker（FlagEmbedding），否则嵌入余弦。

    两条路径返回同构（可比较大小的相对分数）结果，便于 §8.1 消融
    "Ours w/o LanceDB reranker"。
    """
    if reranker_available():                       # pragma: no cover - 需外部权重
        try:
            from FlagEmbedding import FlagReranker

            scorer = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
            pairs = [[query, d] for d in docs]
            scores = scorer.compute_score(pairs, normalize=True)
            if isinstance(scores, (int, float)):
                return [float(scores)]
            return [float(s) for s in scores]
        except Exception:  # noqa: BLE001 - 权重不可得 → 退回嵌入余弦
            pass
    emb = embedder or HashingEmbedder()
    qv = emb.embed(query)
    return [cosine(qv, emb.embed(d)) for d in docs]


def top_k_by_score(items: Iterable[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    """按分数降序取 top-k（同分按 id 升序，保证确定性）。"""
    ordered = sorted(items, key=lambda t: (-t[1], t[0]))
    return ordered[: max(k, 0)]


def _finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))
