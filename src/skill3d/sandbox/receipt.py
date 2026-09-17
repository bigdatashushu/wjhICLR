"""SandboxReceipt 哈希链（§4 M10 / §5）。

每条 receipt 含 prev_hash，构成 sha256 链；任何篡改都会被 verify_chain 检出。
注：§5 schemas 中未定义 SandboxReceipt，故在此本地定义（不改 schemas/）。
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

GENESIS_HASH = "0" * 64


class SandboxReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    receipt_id: str
    episode_id: str
    prev_hash: str  # 哈希链前驱
    event: str  # e.g. "container_start" / "cell_run" / "tool_call" / "container_destroy"
    payload_digest: str  # sha256(规范化 payload)
    source: str = "real"  # Tool 调用相关 receipt 的 source 元数据（T4 校验用）
    error_code: Optional[str] = None  # timeout/oom/disk_full/violation_*
    timestamp: float
    receipt_hash: str


def _digest_payload(payload: Any) -> str:
    import json

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _compute_hash(
    episode_id: str, prev_hash: str, event: str, payload_digest: str,
    source: str, error_code: Optional[str], timestamp: float,
) -> str:
    return hashlib.sha256(
        f"{episode_id}|{prev_hash}|{event}|{payload_digest}|{source}|{error_code}|{timestamp}".encode()
    ).hexdigest()


class ReceiptChain:
    """per-episode 收据链。"""

    def __init__(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self._receipts: list[SandboxReceipt] = []

    @property
    def receipts(self) -> list[SandboxReceipt]:
        return list(self._receipts)

    @property
    def head_hash(self) -> str:
        return self._receipts[-1].receipt_hash if self._receipts else GENESIS_HASH

    def append(
        self,
        event: str,
        payload: Any,
        source: str = "real",
        error_code: Optional[str] = None,
    ) -> SandboxReceipt:
        ts = time.time()
        pd = _digest_payload(payload)
        prev = self.head_hash
        h = _compute_hash(self.episode_id, prev, event, pd, source, error_code, ts)
        r = SandboxReceipt(
            receipt_id=uuid.uuid4().hex,
            episode_id=self.episode_id,
            prev_hash=prev,
            event=event,
            payload_digest=pd,
            source=source,
            error_code=error_code,
            timestamp=ts,
            receipt_hash=h,
        )
        self._receipts.append(r)
        return r


def verify_chain(receipts: list[SandboxReceipt]) -> bool:
    """重算哈希链，任一环节不匹配即 False（篡改检测）。"""
    prev = GENESIS_HASH
    for r in receipts:
        if r.prev_hash != prev:
            return False
        expected = _compute_hash(
            r.episode_id, r.prev_hash, r.event, r.payload_digest,
            r.source, r.error_code, r.timestamp,
        )
        if expected != r.receipt_hash:
            return False
        prev = r.receipt_hash
    return True
