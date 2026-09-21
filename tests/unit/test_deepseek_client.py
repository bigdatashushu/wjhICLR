"""DeepSeek-V4.1-Flash 离线客户端测试（§3.4）：断网 + 假 transport，不触网。

为什么需要这组测试：§3.4 对离线治理调用有四条硬纪律，任何一条被放松都会让
"离线演进"这条论文创新点变成不可审计的黑盒：

1. 密钥只从 ``DEEPSEEK_API_KEY`` 注入，**绝不**进日志/manifest（本文件断言
   ``json.dumps(manifest_fields())`` 里找不到密钥本体）；
2. 健康检查失败必须 fail-closed（可以记 service_unavailable / quarantine，但绝不能
   用 mock 假成功去推进 Readiness，红线 8）；
3. 只重试超时/限流/可恢复 5xx；401/403 立即失败**不重试**，其余 4xx 不盲重试；
4. thinking / reasoning_effort 的具体值必须落到 manifest（冻结配置可审计）。

本文件用假 ``openai`` 模块替换真实 transport（仓库既有做法，见
``tests/unit/test_gpt6_client_latency.py``），并 monkeypatch ``_sleep`` 让退避零等待。
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from skill3d.governance import deepseek_client as d

KEY = "sk-deepseek-secret-must-never-leak-0123456789"
USER_TEXT = "hi"


# --------------------------------------------------------------------------- #
# 假 transport：openai 模块替身
# --------------------------------------------------------------------------- #
class FakeAPIError(Exception):
    """带 HTTP 状态码的假 openai 异常（类名与真实 SDK 一致，便于归族断言）。"""

    def __init__(self, message: str = "boom", *, status_code: int | None = None,
                 headers: dict | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = types.SimpleNamespace(status_code=status_code,
                                              headers=dict(headers or {}))


def _err(name: str, status_code: int | None) -> type:
    """造一个"类名与真实 openai 一致、默认带该状态码"的假异常类。"""

    def __init__(self, message: str = "boom", *, status_code: int | None = status_code,
                 headers: dict | None = None) -> None:
        FakeAPIError.__init__(self, message, status_code=status_code, headers=headers)

    return type(name, (FakeAPIError,), {"__init__": __init__})


AuthenticationError = _err("AuthenticationError", 401)
PermissionDeniedError = _err("PermissionDeniedError", 403)
RateLimitError = _err("RateLimitError", 429)
BadRequestError = _err("BadRequestError", 400)
NotFoundError = _err("NotFoundError", 404)
InternalServerError = _err("InternalServerError", 500)
APITimeoutError = _err("APITimeoutError", None)
APIConnectionError = _err("APIConnectionError", None)


class FakeAPI:
    """记录调用 + 可编排失败序列的假服务端（线程安全，供 chat_many 并发测试）。"""

    def __init__(self, *, models: tuple[str, ...] = ("deepseek-flash", "deepseek-v4-pro"),
                 text_fn=None, usage: tuple[int, int, int] | None = (11, 22, 33),
                 delay_s: float = 0.0, finish_reason: str = "stop",
                 content_none_for: tuple[str, ...] = (),
                 models_error: BaseException | None = None,
                 request_id: str = "req-0001") -> None:
        self.models = list(models)
        self.usage = usage
        self.delay_s = float(delay_s)
        self.finish_reason = finish_reason
        self.content_none_for = set(content_none_for)
        self.models_error = models_error
        self.request_id = request_id
        self.text_fn = text_fn or (lambda key: f"echo:{key}")
        self.calls: list[dict] = []
        self.clients: list[FakeOpenAI] = []
        self.models_calls = 0
        self.max_inflight = 0
        self._inflight = 0
        self._seen: dict[str, int] = {}
        self._lock = threading.Lock()
        # fail_plan: user 内容 → (前 n 次抛的异常, 异常实例)
        self.fail_plan: dict[str, tuple[int, BaseException]] = {}

    def fail(self, key: str, n: int, exc: BaseException) -> None:
        self.fail_plan[key] = (int(n), exc)

    # --- transport 接口 ---
    def create(self, **kwargs):
        with self._lock:
            self.calls.append(dict(kwargs))
            self._inflight += 1
            self.max_inflight = max(self.max_inflight, self._inflight)
        try:
            key = _user_text(kwargs)
            if self.delay_s:
                time.sleep(self.delay_s)
            n_fail, exc = self.fail_plan.get(key, (0, None))
            with self._lock:
                seen = self._seen.get(key, 0)
                self._seen[key] = seen + 1
            if exc is not None and seen < n_fail:
                raise exc
            content = None if key in self.content_none_for else self.text_fn(key)
            message = types.SimpleNamespace(content=content)
            choice = types.SimpleNamespace(message=message,
                                           finish_reason=self.finish_reason)
            usage = None
            if self.usage is not None:
                p, c, t = self.usage
                usage = types.SimpleNamespace(prompt_tokens=p, completion_tokens=c,
                                              total_tokens=t)
            return types.SimpleNamespace(id=self.request_id, choices=[choice],
                                         usage=usage, model=kwargs["model"])
        finally:
            with self._lock:
                self._inflight -= 1

    def list_models(self):
        self.models_calls += 1
        if self.models_error is not None:
            raise self.models_error
        return types.SimpleNamespace(
            object="list",
            data=[types.SimpleNamespace(id=n, object="model", owned_by="deepseek")
                  for n in self.models],
        )


class FakeOpenAI:
    """``openai.OpenAI`` 替身：只暴露 chat.completions.create 与 models.list。"""

    def __init__(self, api: FakeAPI, init_kwargs: dict) -> None:
        self.api = api
        self.init_kwargs = dict(init_kwargs)
        api.clients.append(self)
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=api.create))
        self.models = types.SimpleNamespace(list=api.list_models)


def _user_text(kwargs: dict) -> str:
    for m in reversed(list(kwargs.get("messages", []))):
        if m.get("role") == "user":
            return str(m.get("content"))
    return ""


def _install_fake_openai(monkeypatch, api: FakeAPI) -> None:
    mod = types.ModuleType("openai")
    mod.OpenAI = lambda **kw: FakeOpenAI(api, kw)
    mod.AuthenticationError = AuthenticationError
    mod.RateLimitError = RateLimitError
    monkeypatch.setitem(sys.modules, "openai", mod)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _env_key(monkeypatch):
    """默认注入密钥（缺失场景由用例自己 delenv）。"""
    monkeypatch.setenv(d.ENV_API_KEY, KEY)


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """退避零等待，并记录每次退避时长。"""
    recorded: list[float] = []
    monkeypatch.setattr(d, "_sleep", lambda s: recorded.append(float(s)))
    return recorded


@pytest.fixture
def api(monkeypatch) -> FakeAPI:
    fake = FakeAPI()
    _install_fake_openai(monkeypatch, fake)
    return fake


def _client(**kw) -> d.DeepSeekClient:
    return d.DeepSeekClient(**kw)


def _client_kwargs(api: FakeAPI, *, read_timeout: float) -> dict:
    """取按读取超时区分的那个假 transport 的构造参数。"""
    for client in api.clients:
        timeout = client.init_kwargs.get("timeout")
        if getattr(timeout, "read", timeout) == read_timeout:
            return client.init_kwargs
    raise AssertionError(f"没有 read={read_timeout} 的 transport：{api.clients}")


# --------------------------------------------------------------------------- #
# 固定配置与请求形态（§3.4 表 + Python 片段）
# --------------------------------------------------------------------------- #
def test_fixed_config_matches_spec():
    assert (d.PROVIDER, d.MODEL_ID, d.BASE_URL) == (
        "deepseek", "deepseek-flash", "https://api.deepseek.com")
    assert d.OFFLINE_MODEL_NAME == "DeepSeek-V4.1-Flash"
    assert d.ENV_API_KEY == "DEEPSEEK_API_KEY"
    c = _client()
    assert c.chat_endpoint == "https://api.deepseek.com/chat/completions"
    assert c.models_endpoint == "https://api.deepseek.com/models"
    # 阈值/超时必须是模块级常量且带 TODO_CALIBRATE 标注
    src = Path(d.__file__).read_text(encoding="utf-8")
    for const in ("DEFAULT_TIMEOUT_S", "DEFAULT_CONNECT_TIMEOUT_S", "HEALTH_TIMEOUT_S",
                  "DEFAULT_MAX_RETRIES", "BACKOFF_BASE_S", "BACKOFF_FACTOR",
                  "BACKOFF_MAX_S", "DEFAULT_MAX_TOKENS", "DEFAULT_CONCURRENCY"):
        assert f"{const} =" in src, f"{const} 缺失"
    assert src.count("# TODO_CALIBRATE") >= 9


def test_transport_uses_official_base_url_and_our_own_retry_loop(api):
    c = _client()
    assert c.chat(USER_TEXT) == f"echo:{USER_TEXT}"
    kwargs = _client_kwargs(api, read_timeout=d.DEFAULT_TIMEOUT_S)
    assert kwargs["base_url"] == "https://api.deepseek.com"
    assert kwargs["api_key"] == KEY                     # 密钥只在本进程内传给 SDK
    assert kwargs["max_retries"] == 0                   # 重试自管，避免隐藏重试不可审计
    timeout = kwargs["timeout"]
    if hasattr(timeout, "read"):                        # httpx.Timeout：连接 + 读取双超时
        assert timeout.connect == d.DEFAULT_CONNECT_TIMEOUT_S
        assert timeout.read == d.DEFAULT_TIMEOUT_S
    else:                                               # 无 httpx 时退化为单值超时
        assert timeout == d.DEFAULT_TIMEOUT_S


def test_request_payload_matches_spec_snippet(api):
    c = _client()
    c.chat([{"role": "system", "content": "You are the offline inducer."},
            {"role": "user", "content": USER_TEXT}])
    sent = api.calls[0]
    assert sent["model"] == "deepseek-flash"
    assert sent["stream"] is False
    assert sent["extra_body"] == {"thinking": {"type": "enabled"}}
    assert sent["reasoning_effort"] == "high"
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert "api_key" not in json.dumps(sent, default=str)
    assert "Authorization" not in json.dumps(sent, default=str)


def test_thinking_disabled_is_explicit_and_effort_not_sent(api):
    """官方默认 thinking=enabled，故关闭必须显式发 disabled；此时不发 reasoning_effort。"""
    c = _client(thinking=False, reasoning_effort="high")
    c.chat(USER_TEXT)
    sent = api.calls[0]
    assert sent["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in sent
    manifest = c.manifest_fields()
    assert manifest["thinking"] == "disabled"
    assert manifest["request_params"]["thinking_extra_body"] == {
        "thinking": {"type": "disabled"}}
    assert manifest["reasoning_effort"] is None


def test_invalid_reasoning_effort_rejected():
    with pytest.raises(ValueError):
        _client(reasoning_effort="medium")  # 官方只支持 none/low/high/max


def test_forbidden_overrides_rejected(api):
    c = _client()
    with pytest.raises(ValueError):
        c.chat(USER_TEXT, model="deepseek-v4-pro")   # 覆盖模型会让 manifest 失真
    with pytest.raises(ValueError):
        c.chat(USER_TEXT, stream=True)               # 默认非流式是冻结配置
    with pytest.raises(ValueError):
        c.chat([])                                   # 空消息 fail-closed
    assert api.calls == []


def test_seed_recorded_but_not_sent_by_default(api, monkeypatch):
    """官方 request body 未列 seed：默认不发（避免 400/虚假确定性），但要记录声明值。"""
    c = _client()
    meta = c.chat_with_meta(USER_TEXT, seed=7)
    assert "seed" not in api.calls[0]
    assert meta.seed == 7
    assert meta.seed_sent_to_server is False
    assert c.manifest_fields()["request_params"]["seed_sent_to_server"] is False
    # 若将来官方支持，可用模块开关打开（届时必须重新标定）
    monkeypatch.setattr(d, "SEND_SEED_TO_SERVER", True)
    c2 = _client()
    assert c2.chat_with_meta(USER_TEXT, seed=7).seed_sent_to_server is True
    assert api.calls[1]["seed"] == 7


# --------------------------------------------------------------------------- #
# 重试纪律（§3.4：只重试超时/限流/5xx；401/403 不重试）
# --------------------------------------------------------------------------- #
def test_auth_error_401_fails_immediately_without_retry(api):
    api.fail(USER_TEXT, 99, AuthenticationError("invalid api key", status_code=401))
    c = _client(max_retries=3)
    with pytest.raises(d.OfflineAuthError):
        c.chat(USER_TEXT)
    assert len(api.calls) == 1                # 一次即止，绝不盲重试
    assert c.stats.n_retries == 0


def test_auth_error_403_also_fails_immediately(api):
    api.fail(USER_TEXT, 99, PermissionDeniedError("forbidden", status_code=403))
    c = _client(max_retries=3)
    with pytest.raises(d.OfflineAuthError):
        c.chat(USER_TEXT)
    assert len(api.calls) == 1


def test_timeout_retried_up_to_max_retries_then_service_unavailable(api, sleeps):
    api.fail(USER_TEXT, 99, APITimeoutError("read timeout"))
    c = _client(max_retries=3)
    with pytest.raises(d.OfflineServiceUnavailable):
        c.chat(USER_TEXT)
    assert len(api.calls) == 4                # 总尝试 = max_retries + 1
    assert c.stats.n_retries == 3 and c.stats.n_failures == 1
    assert sleeps == [1.0, 2.0, 4.0]          # 指数退避


def test_transient_timeout_then_success(api, sleeps):
    api.fail(USER_TEXT, 2, APITimeoutError("read timeout"))
    c = _client(max_retries=3)
    assert c.chat(USER_TEXT) == f"echo:{USER_TEXT}"
    assert len(api.calls) == 3
    assert sleeps == [1.0, 2.0]
    assert c.stats.n_calls == 1 and c.stats.n_failures == 0


def test_builtin_timeout_is_retryable(api):
    api.fail(USER_TEXT, 1, TimeoutError("socket read timed out"))
    c = _client(max_retries=1)
    assert c.chat(USER_TEXT) == f"echo:{USER_TEXT}"
    assert len(api.calls) == 2


def test_rate_limit_429_retried_and_honours_retry_after(api, sleeps):
    api.fail(USER_TEXT, 2, RateLimitError("rate limited", status_code=429))
    c = _client(max_retries=3)
    assert c.chat(USER_TEXT) == f"echo:{USER_TEXT}"
    assert len(api.calls) == 3
    assert sleeps == [1.0, 2.0]

    api.fail("second", 1, RateLimitError("slow down", status_code=429,
                                         headers={"retry-after": "5"}))
    c.chat("second")
    assert sleeps[-1] == 5.0


def test_server_error_500_retried_but_400_not(api):
    api.fail(USER_TEXT, 1, InternalServerError("upstream"))
    c = _client(max_retries=2)
    assert c.chat(USER_TEXT) == f"echo:{USER_TEXT}"
    assert len(api.calls) == 2

    api.fail("bad", 99, BadRequestError("bad parameters"))
    c2 = _client(max_retries=3)
    n_before = len(api.calls)
    with pytest.raises(d.OfflineRequestError):
        c2.chat("bad")
    assert len(api.calls) - n_before == 1     # 400 只尝试一次，不盲重试
    assert c2.stats.n_retries == 0


def test_404_model_not_found_is_not_retried(api):
    api.fail(USER_TEXT, 99, NotFoundError("model not found", status_code=404))
    c = _client(max_retries=3)
    with pytest.raises(d.OfflineRequestError):
        c.chat(USER_TEXT)
    assert len(api.calls) == 1


def test_connection_error_after_retries_reports_service_unavailable(api):
    api.fail(USER_TEXT, 99, APIConnectionError("connection refused"))
    c = _client(max_retries=1)
    with pytest.raises(d.OfflineServiceUnavailable) as excinfo:
        c.chat(USER_TEXT)
    assert "service_unavailable" in str(excinfo.value)
    assert KEY not in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 密钥纪律（§3.4 / §19.2：绝不落盘、绝不回显）
# --------------------------------------------------------------------------- #
def test_manifest_fields_never_contains_api_key(api):
    c = _client(prompt_version="governance-v1")
    c.chat(USER_TEXT)
    manifest = c.manifest_fields()
    dumped = json.dumps(manifest, ensure_ascii=False)
    assert KEY not in dumped
    assert "api_key" not in dumped.lower()
    assert "authorization" not in dumped.lower()
    assert "bearer" not in dumped.lower()
    # endpoint_hash 是哈希，不是 URL 本体（§19.2）
    assert "api.deepseek.com" not in dumped
    assert manifest["endpoint_hash"] == hashlib.sha256(
        d.BASE_URL.encode("utf-8")).hexdigest()
    assert manifest["endpoint_hash"] == d.endpoint_hash(d.BASE_URL)
    assert manifest["provider"] == "deepseek"
    assert manifest["model_id"] == "deepseek-flash"
    assert manifest["offline_model"] == "DeepSeek-V4.1-Flash"
    assert manifest["prompt_version"] == "governance-v1"
    for field in ("latency", "token_usage", "request_params"):
        assert field in manifest, f"manifest 缺 {field}"


def test_repr_and_stats_never_leak_key(api):
    c = _client()
    c.chat(USER_TEXT)
    assert KEY not in repr(c)
    assert KEY not in str(c)
    assert KEY not in json.dumps(c.stats.as_dict(), ensure_ascii=False)
    assert c.api_key_configured is True


def test_missing_api_key_refuses_to_call(api, monkeypatch):
    monkeypatch.delenv(d.ENV_API_KEY, raising=False)
    c = d.DeepSeekClient()
    assert c.api_key_configured is False
    assert c.configured is False               # 兼容既有 client.configured 预检写法
    with pytest.raises(d.OfflineAuthError):
        c.chat(USER_TEXT)
    assert api.calls == []                    # 未配置时一次网络调用都不发生
    assert c.health_check() is False          # 缺密钥也判服务不可用（fail-closed）
    assert c.last_health["reason"].startswith("api_key_missing")


def test_model_id_alias_matches_frozen_config(api):
    c = _client()
    assert c.model_id == "deepseek-flash" == c.model == d.MODEL_ID
    assert c.configured is True
    assert c.manifest_fields()["model_id"] == c.model_id


def test_explicit_api_key_argument_is_accepted(api):
    c = d.DeepSeekClient(api_key=KEY)
    assert c.chat(USER_TEXT) == f"echo:{USER_TEXT}"
    assert _client_kwargs(api, read_timeout=d.DEFAULT_TIMEOUT_S)["api_key"] == KEY


# --------------------------------------------------------------------------- #
# 健康检查（§3.4：GET /models + Flash 在列；失败不得 mock 顶替）
# --------------------------------------------------------------------------- #
def test_health_check_true_only_when_flash_model_listed(api):
    c = _client()
    assert c.health_check() is True
    assert c.last_health["models"] == ["deepseek-flash", "deepseek-v4-pro"]
    assert c.last_health["reason"] == "ok"
    c.require_service()                        # 不抛


def test_health_check_false_when_no_flash_model(api):
    api.models = ["deepseek-chat", "deepseek-v4-pro"]   # 无 flash
    c = _client()
    assert c.health_check() is False
    assert c.last_health["flash_available"] is False
    assert c.last_health["reason"] == "flash_model_not_listed"
    with pytest.raises(d.OfflineServiceUnavailable):
        c.require_service()
    # manifest 必须如实记录健康检查未通过，供 service_unavailable / quarantine 归因
    assert c.manifest_fields()["health"]["ok"] is False


def test_health_check_false_on_transport_error_and_never_fakes_success(api):
    api.models_error = APIConnectionError("network down")
    c = _client(max_retries=0)
    assert c.health_check() is False
    assert "APIConnectionError" in c.last_health["reason"]
    with pytest.raises(d.OfflineServiceUnavailable):
        c.require_service()
    # "不得用 mock 结果推进 Readiness"：服务不可用时 chat 必须抛错，不得返回假文本
    api.fail(USER_TEXT, 99, APIConnectionError("network down"))
    with pytest.raises(d.OfflineServiceUnavailable):
        c.chat(USER_TEXT)
    assert c.stats.n_calls == 0
    assert c.manifest_fields()["token_usage"]["total_tokens"] == 0


def test_health_check_uses_health_timeout_and_same_auth(api):
    c = _client()
    c.health_check()
    kwargs = _client_kwargs(api, read_timeout=d.HEALTH_TIMEOUT_S)
    assert kwargs["base_url"] == "https://api.deepseek.com"
    assert kwargs["api_key"] == KEY
    assert api.models_calls == 1


# --------------------------------------------------------------------------- #
# 响应校验与元数据（返回原始文本 + 元数据，无副作用）
# --------------------------------------------------------------------------- #
def test_empty_content_fails_closed_without_retry(api):
    api.content_none_for = {USER_TEXT}
    c = _client(max_retries=3)
    with pytest.raises(d.OfflineResponseError):
        c.chat(USER_TEXT)
    assert len(api.calls) == 1                # 空文本不是"可重试的服务故障"
    assert c.stats.n_calls == 0


def test_truncated_response_is_flagged(api):
    api.finish_reason = "length"
    c = _client()
    meta = c.chat_with_meta(USER_TEXT)
    assert meta.text == f"echo:{USER_TEXT}"
    assert meta.truncated is True
    assert meta.manifest_fields()["truncated"] is True


def test_token_usage_latency_and_request_id_recorded(api):
    c = _client()
    c.chat(USER_TEXT)
    usage = c.manifest_fields()["token_usage"]
    assert (usage["prompt_tokens"], usage["completion_tokens"],
            usage["total_tokens"]) == (11, 22, 33)
    assert usage["n_calls_with_usage"] == 1
    latency = c.manifest_fields()["latency"]
    assert latency["n_calls"] == 1 and latency["n_attempts"] == 1
    assert latency["max_s"] >= latency["median_s"] >= 0.0
    assert len(c.stats.latency_s) == 1 and c.stats.latency_s[0] >= 0.0
    assert c.stats.request_ids == ["req-0001"]


def test_chat_with_meta_records_frozen_config(api):
    c = _client(prompt_version="revise-v1")
    meta = c.chat_with_meta(USER_TEXT)
    fields = meta.manifest_fields()
    assert KEY not in json.dumps(fields, ensure_ascii=False)
    assert fields["thinking"] == "enabled"
    assert fields["reasoning_effort"] == "high"
    assert fields["prompt_version"] == "revise-v1"
    assert fields["text_sha256"] == hashlib.sha256(
        f"echo:{USER_TEXT}".encode("utf-8")).hexdigest()
    assert fields["n_chars"] == len(f"echo:{USER_TEXT}")
    assert "text" not in fields               # 元数据不含内容本体


def test_manifest_records_thinking_and_reasoning_effort(api):
    c = _client(thinking=True, reasoning_effort="high")
    c.chat(USER_TEXT)
    manifest = c.manifest_fields()
    assert manifest["thinking"] == "enabled"
    assert manifest["reasoning_effort"] == "high"
    params = manifest["request_params"]
    assert params["thinking"] == "enabled"
    assert params["reasoning_effort"] == "high"
    assert params["thinking_extra_body"] == {"thinking": {"type": "enabled"}}
    assert params["max_retries"] == d.DEFAULT_MAX_RETRIES
    assert params["backoff"]["retry_on_status"] == [429, 500, 502, 503, 504]


# --------------------------------------------------------------------------- #
# chat_many：批量 + 并发（离线调用慢，必须并发掩盖延迟）
# --------------------------------------------------------------------------- #
def test_chat_many_empty_returns_empty(api):
    assert _client().chat_many([]) == []
    assert api.calls == []


def test_chat_many_preserves_order_and_bounds_concurrency(api):
    api.delay_s = 0.05
    c = _client()
    prompts = [f"p{i}" for i in range(6)]
    out = c.chat_many(prompts, concurrency=3)
    assert out == [f"echo:{p}" for p in prompts]   # 与输入同序
    assert 2 <= api.max_inflight <= 3              # 确实并发，但不超过上界
    assert c.stats.n_calls == 6


def test_chat_many_concurrency_one_is_serial(api):
    api.delay_s = 0.01
    c = _client()
    assert c.chat_many(["a", "b", "c"], concurrency=1) == ["echo:a", "echo:b", "echo:c"]
    assert api.max_inflight == 1


def test_chat_many_fail_closed_by_default_and_can_return_exceptions(api):
    api.fail("b", 99, BadRequestError("bad prompt", status_code=400))
    c = _client(max_retries=0)
    with pytest.raises(d.OfflineRequestError):
        c.chat_many(["a", "b"], concurrency=1)

    c2 = _client(max_retries=0)
    out = c2.chat_many(["a", "b"], concurrency=1, return_exceptions=True)
    assert out[0] == "echo:a"
    assert isinstance(out[1], d.OfflineRequestError)
    assert c2.stats.n_failures == 1


def test_chat_many_accepts_messages_and_system(api):
    c = _client()
    out = c.chat_many([[{"role": "user", "content": "m1"}], "m2"], concurrency=2,
                      system="sys")
    assert out == ["echo:m1", "echo:m2"]
    assert [m["role"] for m in api.calls[0]["messages"]] == ["system", "user"]


# --------------------------------------------------------------------------- #
# 静态纪律：不 import skills/memory（模型文本不能直接改 active Skill/Memory）
# --------------------------------------------------------------------------- #
def test_module_does_not_import_skills_or_memory():
    """离线客户端只返回文本 + 元数据，不得 import 或改动 active Skill/Memory。"""
    src = Path(d.__file__).read_text(encoding="utf-8")
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            low = stripped.lower()
            for forbidden in ("skill3d.skills", "skill3d.memory", "skill3d.evolution",
                              "registry", "admission"):
                assert forbidden not in low, f"离线客户端不得 import {forbidden}: {line}"


def test_classify_error_maps_spec_clauses():
    assert d.classify_error(AuthenticationError("x", status_code=401)) == "auth"
    assert d.classify_error(PermissionDeniedError("x", status_code=403)) == "auth"
    assert d.classify_error(RateLimitError("x", status_code=429)) == "retry"
    assert d.classify_error(InternalServerError("x", status_code=503)) == "retry"
    assert d.classify_error(BadRequestError("x", status_code=400)) == "fatal"
    assert d.classify_error(APITimeoutError("x")) == "retry"
    assert d.classify_error(APIConnectionError("x")) == "retry"
    assert d.classify_error(TimeoutError("x")) == "retry"
    assert d.classify_error(ValueError("x")) == "fatal"


def test_scrub_secrets_removes_key_from_messages():
    text = f"Authorization: Bearer {KEY} -> {KEY}"
    scrubbed = d._scrub_secrets(text, KEY)
    assert KEY not in scrubbed
    assert "***" in scrubbed


def test_message_summary_has_no_content():
    summary = d._summarize_messages([{"role": "user", "content": "secret prompt"}])
    assert "secret prompt" not in json.dumps(summary)
    assert summary[0]["n_chars"] == len("secret prompt")


def test_endpoint_hash_normalizes_slash_and_never_leaks_url(monkeypatch):
    assert d.endpoint_hash("https://api.deepseek.com/") == d.endpoint_hash(d.BASE_URL)
    c = d.DeepSeekClient(base_url="https://relay.internal.example/v1/")
    assert c.base_url == "https://relay.internal.example/v1"
    dumped = json.dumps(c.manifest_fields(), ensure_ascii=False)
    assert "relay.internal.example" not in dumped
    assert c.manifest_fields()["endpoint_hash"] == d.endpoint_hash(c.base_url)


def test_client_factory_injection_is_used(monkeypatch):
    """DI 钩子：离线 driver 可注入自己的传输，且完全不 import openai。"""
    monkeypatch.setitem(sys.modules, "openai", None)   # 一旦被 import 就会报错
    seen: dict = {}

    def factory(*, api_key, base_url, timeout, max_retries):
        seen.update(api_key=api_key, base_url=base_url, max_retries=max_retries)
        transport = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(
                create=lambda **kw: types.SimpleNamespace(
                    id="req-x", usage=None,
                    choices=[types.SimpleNamespace(
                        finish_reason="stop",
                        message=types.SimpleNamespace(content="from-factory"))]))),
            models=types.SimpleNamespace(list=lambda: types.SimpleNamespace(
                data=[types.SimpleNamespace(id="deepseek-flash")])),
        )
        return transport

    c = d.DeepSeekClient(client_factory=factory)
    assert c.chat(USER_TEXT) == "from-factory"
    assert c.health_check() is True
    assert seen["base_url"] == "https://api.deepseek.com"
    assert seen["api_key"] == KEY and seen["max_retries"] == 0
    assert c.manifest_fields()["token_usage"]["n_calls_without_usage"] == 1
