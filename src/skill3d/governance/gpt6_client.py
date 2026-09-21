"""M16 GPT-6 OpenAI 兼容客户端（仅离线，硬约束 1/2/4）。

- endpoint / model_id / auth 全部从环境变量读取，默认值 "TODO_USER_INPUT"（硬约束 4，不虚构）；
- 未配置时抛 GPT6NotConfiguredError；
- timeout/max_retries 从环境变量读取，缺省按**第三方中转站实测延迟**设定；
- 累计 token 预算检查 max_gpt6_tokens=500000（TODO_CALIBRATE），超预算抛 GPT6BudgetExhaustedError。

**实测延迟（2026-09-20，本机 → api.xcode.best 中转站，model=gpt-5.6-sol）**：
单次极短 prompt 调用约 243 s（17 tokens），远高于常规直连；因此

1. 默认 `GPT6_TIMEOUT_S=600`（沿用旧的 120 s 必然超时，属"配置即失效"陷阱）；
2. 离线演进链必须**批量化 prompt**（一次调用尽量覆盖多个 trace/candidate）并用
   `chat_many()` 并发掩盖延迟，而不是逐条串行。
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# TODO_USER_INPUT：GPT-6 接入参数，一律从环境变量读取，不虚构默认值
ENV_ENDPOINT = "SKILL3D_GPT6_ENDPOINT"
ENV_MODEL_ID = "SKILL3D_GPT6_MODEL_ID"
ENV_API_KEY = "SKILL3D_GPT6_API_KEY"
ENV_TIMEOUT = "SKILL3D_GPT6_TIMEOUT"
ENV_MAX_RETRIES = "SKILL3D_GPT6_MAX_RETRIES"
# 可选：中转站/模型支持的推理强度（"low"/"medium"/"high"）；空 = 不发送该参数
ENV_REASONING_EFFORT = "SKILL3D_GPT6_REASONING_EFFORT"
TODO_USER_INPUT = "TODO_USER_INPUT"

# TODO_CALIBRATE：超时/重试/预算为起始参考值。
# 600 s 的依据是中转站实测 ~243 s/次（含排队），留 2.5× 余量；120 s 旧值会误超时。
GPT6_TIMEOUT_S = 600
GPT6_MAX_RETRIES = 3
MAX_GPT6_TOKENS = 500000  # TODO_CALIBRATE：累计 token 预算
DEFAULT_CONCURRENCY = 4   # TODO_CALIBRATE：并发掩盖中转站延迟（受服务端限流约束）


class GPT6NotConfiguredError(RuntimeError):
    """GPT-6 未配置（endpoint/model_id/auth 仍为 TODO_USER_INPUT）。"""


class GPT6BudgetExhaustedError(RuntimeError):
    """累计 token 预算耗尽。"""


@dataclass
class GPT6CallStats:
    """调用统计（进 RunManifest / 报告：延迟与 token 都要留档）。"""

    n_calls: int = 0
    n_failures: int = 0
    total_tokens: int = 0
    latency_s: list[float] = field(default_factory=list)

    @property
    def median_latency_s(self) -> float:
        if not self.latency_s:
            return 0.0
        xs = sorted(self.latency_s)
        n = len(xs)
        return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

    def as_dict(self) -> dict:
        return {"n_calls": self.n_calls, "n_failures": self.n_failures,
                "total_tokens": self.total_tokens,
                "median_latency_s": round(self.median_latency_s, 2),
                "max_latency_s": round(max(self.latency_s), 2) if self.latency_s else 0.0}


class GPT6Client:
    """GPT-6 OpenAI 兼容客户端。仅离线治理链使用；在线链禁止 import 本模块。"""

    def __init__(self, endpoint: str | None = None, model_id: str | None = None,
                 api_key: str | None = None,
                 timeout_s: int | None = None,
                 max_retries: int | None = None,
                 max_tokens_budget: int = MAX_GPT6_TOKENS,
                 reasoning_effort: str | None = None) -> None:
        self.endpoint = endpoint if endpoint is not None else os.environ.get(
            ENV_ENDPOINT, TODO_USER_INPUT)
        self.model_id = model_id if model_id is not None else os.environ.get(
            ENV_MODEL_ID, TODO_USER_INPUT)
        self.api_key = api_key if api_key is not None else os.environ.get(
            ENV_API_KEY, TODO_USER_INPUT)
        self.timeout_s = int(timeout_s if timeout_s is not None
                             else os.environ.get(ENV_TIMEOUT, GPT6_TIMEOUT_S))
        self.max_retries = int(max_retries if max_retries is not None
                               else os.environ.get(ENV_MAX_RETRIES, GPT6_MAX_RETRIES))
        self.max_tokens_budget = max_tokens_budget
        # 空字符串 = 不发送该参数（部分中转站/模型不接受 reasoning_effort）
        self.reasoning_effort = (reasoning_effort if reasoning_effort is not None
                                 else os.environ.get(ENV_REASONING_EFFORT, ""))
        self.tokens_used = 0
        self.stats = GPT6CallStats()

    @property
    def configured(self) -> bool:
        return TODO_USER_INPUT not in (self.endpoint, self.model_id, self.api_key)

    def _require_configured(self) -> None:
        if not self.configured:
            raise GPT6NotConfiguredError(
                "GPT-6 未配置：endpoint/model_id/auth 需经环境变量 "
                f"{ENV_ENDPOINT}/{ENV_MODEL_ID}/{ENV_API_KEY} 提供（TODO_USER_INPUT）"
            )

    def check_budget(self, additional_tokens: int = 0) -> None:
        """累计 token 预算检查（TODO_CALIBRATE）。"""
        if self.tokens_used + additional_tokens > self.max_tokens_budget:
            raise GPT6BudgetExhaustedError(
                f"GPT-6 累计 token 预算耗尽: {self.tokens_used}+{additional_tokens}"
                f" > {self.max_tokens_budget}"
            )

    def chat(self, prompt: str, system: str | None = None) -> str:
        """调用 GPT-6（OpenAI 兼容 chat completions），返回文本。

        openai 库已安装；函数内构造 client 以便测试 mock。
        """
        self._require_configured()
        self.check_budget()
        from openai import OpenAI  # 已安装库；延迟构造便于 mock

        client = OpenAI(base_url=self.endpoint, api_key=self.api_key,
                        timeout=self.timeout_s, max_retries=self.max_retries)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict = {"model": self.model_id, "messages": messages}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        t0 = time.time()
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception:
            self.stats.n_failures += 1
            self.stats.latency_s.append(time.time() - t0)
            raise
        usage = getattr(resp, "usage", None)
        total = getattr(usage, "total_tokens", 0) if usage else 0
        self.check_budget(total)
        self.tokens_used += total
        self.stats.n_calls += 1
        self.stats.total_tokens += int(total or 0)
        self.stats.latency_s.append(time.time() - t0)
        return resp.choices[0].message.content or ""

    def chat_many(self, prompts: list[str], *, system: str | None = None,
                  concurrency: int = DEFAULT_CONCURRENCY,
                  return_exceptions: bool = False) -> list[str | Exception]:
        """并发调用（掩盖中转站高延迟；离线链专用）。

        **为什么必须并发**：中转站实测单次 ~4 分钟，逐个串行会让一次演进生成
        变成数小时。并发不会改变单次调用语义（同 prompt 同结果），只是并行等待。
        失败默认**抛出**（fail-closed）；`return_exceptions=True` 时把异常放在
        对应位置返回，交给调用方按 §8 降级策略处理（如 QUARANTINE）。
        """
        if not prompts:
            return []
        n = max(1, min(int(concurrency), len(prompts)))
        out: list[str | Exception] = [None] * len(prompts)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = {pool.submit(self.chat, p, system): i for i, p in enumerate(prompts)}
            for fut, idx in futs.items():
                try:
                    out[idx] = fut.result()
                except Exception as exc:  # noqa: BLE001 - 由调用方决定降级方式
                    if not return_exceptions:
                        raise
                    out[idx] = exc
        return out
