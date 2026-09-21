"""GPT-6 客户端离线纪律测试（超时/并发/预算），**不触网**。

为什么需要这组测试：中转站实测单次调用约 243 s，若沿用旧的 120 s 默认超时，
第一次真实离线演进必然以超时失败（"配置即失效"陷阱）；且逐个串行调用会让一次
演进生成变成数小时，故离线链必须能用 `chat_many` 并发掩盖延迟。
"""

from __future__ import annotations

import sys
import types

import pytest

from skill3d.governance import gpt6_client as g


class _FakeCompletions:
    def __init__(self, recorder: list[dict], fail_on: set[int]) -> None:
        self.recorder = recorder
        self.fail_on = fail_on

    def create(self, **kwargs):
        idx = len(self.recorder)
        self.recorder.append(kwargs)
        if idx in self.fail_on:
            raise RuntimeError(f"boom-{idx}")
        usage = types.SimpleNamespace(total_tokens=10)
        msg = types.SimpleNamespace(content=f"reply-{idx}")
        return types.SimpleNamespace(usage=usage, choices=[types.SimpleNamespace(message=msg)])


class _FakeOpenAI:
    recorder: list[dict] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.chat = types.SimpleNamespace(
            completions=_FakeCompletions(_FakeOpenAI.recorder, _FakeOpenAI.fail_on))


@pytest.fixture(autouse=True)
def _fake_openai(monkeypatch):
    _FakeOpenAI.recorder = []
    _FakeOpenAI.fail_on = set()
    mod = types.ModuleType("openai")
    mod.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mod)
    yield


def _client(**kw) -> g.GPT6Client:
    return g.GPT6Client(endpoint="https://relay.example/v1", model_id="gpt-5.6-sol",
                        api_key="sk-test", **kw)


def test_timeout_default_beats_relay_latency():
    """默认超时必须显著高于中转站实测 ~243 s，否则第一次真实调用就超时。"""
    c = _client()
    assert c.timeout_s >= 600 and c.max_retries >= 3
    assert g.GPT6_TIMEOUT_S >= 600


def test_env_overrides_timeout_and_counts_stats():
    c = _client(timeout_s=900, max_retries=5)
    assert (c.timeout_s, c.max_retries) == (900, 5)
    assert c.chat("hi") == "reply-0"
    assert c.tokens_used == 10
    assert c.stats.as_dict()["n_calls"] == 1


def test_reasoning_effort_passthrough_optional():
    _client(reasoning_effort="low").chat("hi")
    assert _FakeOpenAI.recorder[0].get("reasoning_effort") == "low"
    _client(reasoning_effort="").chat("hi")
    assert "reasoning_effort" not in _FakeOpenAI.recorder[1]


def test_chat_many_runs_concurrently_and_preserves_order():
    c = _client()
    out = c.chat_many(["a", "b", "c"], concurrency=3)
    assert out == ["reply-0", "reply-1", "reply-2"]
    assert c.stats.n_calls == 3


def test_chat_many_fail_closed_by_default():
    _FakeOpenAI.fail_on = {1}
    c = _client()
    with pytest.raises(RuntimeError):
        c.chat_many(["a", "b"], concurrency=1)
    assert c.stats.n_failures >= 1


def test_chat_many_can_return_exceptions_for_quarantine_flow():
    _FakeOpenAI.fail_on = {0}
    c = _client()
    out = c.chat_many(["a", "b"], concurrency=1, return_exceptions=True)
    assert isinstance(out[0], Exception) and out[1] == "reply-1"


def test_budget_exhaustion_is_fail_closed():
    c = _client(max_tokens_budget=5)
    with pytest.raises(g.GPT6BudgetExhaustedError):
        c.chat("hi")


def test_unconfigured_client_refuses_to_call(monkeypatch):
    for k in (g.ENV_ENDPOINT, g.ENV_MODEL_ID, g.ENV_API_KEY):
        monkeypatch.delenv(k, raising=False)
    c = g.GPT6Client()
    assert not c.configured
    with pytest.raises(g.GPT6NotConfiguredError):
        c.chat("hi")
