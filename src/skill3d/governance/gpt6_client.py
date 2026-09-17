"""M16 GPT-6 OpenAI 兼容客户端（仅离线，硬约束 1/2/4）。

- endpoint / model_id / auth 全部从环境变量读取，默认值 "TODO_USER_INPUT"（硬约束 4，不虚构）；
- 未配置时抛 GPT6NotConfiguredError；
- timeout=120s / max_retries=2（TODO_CALIBRATE）；
- 累计 token 预算检查 max_gpt6_tokens=500000（TODO_CALIBRATE），超预算抛 GPT6BudgetExhaustedError。
"""

from __future__ import annotations

import os

# TODO_USER_INPUT：GPT-6 接入参数，一律从环境变量读取，不虚构默认值
ENV_ENDPOINT = "SKILL3D_GPT6_ENDPOINT"
ENV_MODEL_ID = "SKILL3D_GPT6_MODEL_ID"
ENV_API_KEY = "SKILL3D_GPT6_API_KEY"
TODO_USER_INPUT = "TODO_USER_INPUT"

# TODO_CALIBRATE：超时/重试/预算为起始参考值
GPT6_TIMEOUT_S = 120
GPT6_MAX_RETRIES = 2
MAX_GPT6_TOKENS = 500000  # TODO_CALIBRATE：累计 token 预算


class GPT6NotConfiguredError(RuntimeError):
    """GPT-6 未配置（endpoint/model_id/auth 仍为 TODO_USER_INPUT）。"""


class GPT6BudgetExhaustedError(RuntimeError):
    """累计 token 预算耗尽。"""


class GPT6Client:
    """GPT-6 OpenAI 兼容客户端。仅离线治理链使用；在线链禁止 import 本模块。"""

    def __init__(self, endpoint: str | None = None, model_id: str | None = None,
                 api_key: str | None = None,
                 timeout_s: int = GPT6_TIMEOUT_S,
                 max_retries: int = GPT6_MAX_RETRIES,
                 max_tokens_budget: int = MAX_GPT6_TOKENS) -> None:
        self.endpoint = endpoint if endpoint is not None else os.environ.get(
            ENV_ENDPOINT, TODO_USER_INPUT)
        self.model_id = model_id if model_id is not None else os.environ.get(
            ENV_MODEL_ID, TODO_USER_INPUT)
        self.api_key = api_key if api_key is not None else os.environ.get(
            ENV_API_KEY, TODO_USER_INPUT)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_tokens_budget = max_tokens_budget
        self.tokens_used = 0

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
        resp = client.chat.completions.create(model=self.model_id, messages=messages)
        usage = getattr(resp, "usage", None)
        total = getattr(usage, "total_tokens", 0) if usage else 0
        self.check_budget(total)
        self.tokens_used += total
        return resp.choices[0].message.content or ""
