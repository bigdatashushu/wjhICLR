"""本地 vLLM OpenAI 兼容客户端（§4 M8）。

temperature=0.0 greedy；支持 DP×N 多 endpoint 轮询。
仅调本地 Qwen3-VL-8B（硬约束 1：在线链绝对无 GPT-6）。
"""

from __future__ import annotations

import itertools
from typing import Sequence

from openai import OpenAI


class VLLMClient:
    """DP 多 endpoint 轮询的 OpenAI 兼容 chat 客户端。"""

    def __init__(
        self,
        endpoints: Sequence[str],
        model: str,
        timeout_s: float = 120.0,
    ) -> None:
        if not endpoints:
            raise ValueError("endpoints 不能为空")
        self._clients = [
            OpenAI(base_url=ep.rstrip("/") + "/v1", api_key="EMPTY", timeout=timeout_s)
            for ep in endpoints
        ]
        self._rr = itertools.cycle(range(len(self._clients)))
        self.model = model

    def chat(self, messages: list[dict], max_tokens: int = 4096) -> str:
        """greedy（temperature=0.0）生成，返回文本内容。"""
        client = self._clients[next(self._rr)]
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content
        if content is None:
            raise RuntimeError("vLLM 返回空 content")
        return content
