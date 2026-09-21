"""DeepSeek-V4.1-Flash 官方 API 客户端（§3.3 / §3.4；v6 取代 GPT-6，见 §20）。

离线治理链（Offline Inducer/Governor）唯一允许的强模型客户端。**仅离线**：在线链
绝对禁止任何离线强模型介入（红线 1），故本模块只允许被 `evolution/` 与
`governance/` 的离线模块 import。

固定配置（§3.4，不得改动）：

| 项 | 值 |
|---|---|
| Provider | ``deepseek`` |
| OpenAI-compatible base URL | ``https://api.deepseek.com`` |
| Chat endpoint | ``POST https://api.deepseek.com/chat/completions`` |
| API model ID | ``deepseek-flash`` |
| 实际模型版本 | DeepSeek-V4.1-Flash |
| Authentication | ``Authorization: Bearer ${DEEPSEEK_API_KEY}`` |
| 默认模式 | 非流式；离线归纳启用 thinking，``reasoning_effort`` 由冻结配置记录 |

官方依据：https://api-docs.deepseek.com/zh-cn/index.html 、
`POST /chat/completions`（`thinking` / `reasoning_effort` / `max_tokens` 参数）、
`GET /models`（返回 ``{"object": "list", "data": [{"id": ...}]}``）。

纪律（§3.4 / §19.2 / 红线 8）：

1. **密钥**只从环境变量 ``DEEPSEEK_API_KEY``（或显式构造参数）注入，**绝不**写进
   源码、配置文件、文档、日志、Trace 或 RunManifest；异常信息经 `_scrub_secrets`
   脱敏，不回显密钥。
2. **健康检查**：`health_check()` 调 ``GET {base}/models``，确认返回列表中当前可用
   的 Flash 模型存在；失败时 `require_service()` 抛 `OfflineServiceUnavailable`，
   离线任务据此记 ``service_unavailable`` / quarantine——**不切在线链、不用 mock
   结果推进 Readiness**。
3. **重试**：自管指数退避（SDK ``max_retries=0``，避免隐藏重试使次数不可审计），
   只重试超时/限流（429）/可恢复 5xx；401/403 立即抛 `OfflineAuthError`，其余 4xx
   抛 `OfflineRequestError`，**不盲目重试**。
4. **无副作用**：本模块只返回**原始文本 + 元数据**，不解析、不落盘、不改 active
   Skill/Memory、不决定 promote/reject。Schema / 泄漏检查 / 确定性准入门在
   `evolution/firewall.py`、`evolution/admission.py` 侧执行；promote 由确定性门 +
   预注册规则决定（§23.1 Phase 6 / §3.3）。本模块**不 import** skills / memory。
5. **脱敏**：日志只记形状与计数（消息条数、字符数、token、延迟），不记请求体内容
   与密钥。

延迟参考（v5 交接记录 §3）：旧中转站单次调用实测约 243 s，故超时默认值必须显著高于
该量级（"配置即失效"陷阱），且离线调用必须批量 + 并发（`chat_many`）。
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# §3.4 固定配置（已核验；改动即违反 §3.4）
# --------------------------------------------------------------------------- #
PROVIDER = "deepseek"
MODEL_ID = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
OFFLINE_MODEL_NAME = "DeepSeek-V4.1-Flash"
ENV_API_KEY = "DEEPSEEK_API_KEY"
CHAT_PATH = "/chat/completions"
MODELS_PATH = "/models"

# --------------------------------------------------------------------------- #
# 阈值/超时/重试：起始参考值，均待标定（§3.4「具体数值 [TODO_CALIBRATE]」）
# --------------------------------------------------------------------------- #
DEFAULT_TIMEOUT_S = 300.0       # TODO_CALIBRATE：读取超时（秒）。thinking=high 下
# 单次批量化 prompt 可能到分钟级；v5 中转站实测 ~243 s，故不得低于该量级。
DEFAULT_CONNECT_TIMEOUT_S = 10.0  # TODO_CALIBRATE：连接超时（秒）
HEALTH_TIMEOUT_S = 30.0         # TODO_CALIBRATE：启动前健康检查超时（秒）
DEFAULT_MAX_RETRIES = 3         # TODO_CALIBRATE：重试次数（总尝试 = max_retries + 1）
BACKOFF_BASE_S = 1.0            # TODO_CALIBRATE：指数退避首项（秒）
BACKOFF_FACTOR = 2.0            # TODO_CALIBRATE：指数退避倍率
BACKOFF_MAX_S = 30.0            # TODO_CALIBRATE：单次退避上限（秒）
DEFAULT_MAX_TOKENS = 32768      # TODO_CALIBRATE：官方上限 393216；非 thinking 默认 8K、
# thinking 默认 64K。thinking 下 8K 极易被思维链吃满 → 截断/空 content，故起始值取 32K。
DEFAULT_CONCURRENCY = 4         # TODO_CALIBRATE：并发数（受服务端限流约束；B2 批量化）
MAX_RECORDED_REQUEST_IDS = 16   # TODO_CALIBRATE：manifest 里保留的 response request id 条数

# 仅重试这些状态码：限流 + 可恢复 5xx（§3.4）
RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# 立即失败（不重试）的认证类状态码（§3.4）
AUTH_STATUS_CODES = frozenset({401, 403})

# 官方 `reasoning_effort` 取值（none = 关闭 thinking）；None 表示不发送该字段
REASONING_EFFORT_CHOICES = ("none", "low", "high", "max")
# extra_body 里 thinking 的两种形态（官方默认 enabled，故关闭必须显式发 disabled）
THINKING_ENABLED_BODY: dict = {"thinking": {"type": "enabled"}}
THINKING_DISABLED_BODY: dict = {"thinking": {"type": "disabled"}}

# 健康检查判定用的 Flash 模型名标记（官方 /models 返回 id，如 "deepseek-flash"）
FLASH_NAME_MARKER = "flash"

# endpoint_hash 摘要算法（§19.2 要求 hash 而非原始 URL）
ENDPOINT_HASH_ALG = "sha256"

# [未核验] 官方 `POST /chat/completions` 的 request body 未列 `seed`；为避免 400 与
# "虚假的可确定性"，默认**不发送**该参数，仅在元数据里记录调用方声明的值。
SEND_SEED_TO_SERVER = False

# prompt 模板版本（冻结配置项，进 RunManifest）。调用方应传自己的模板版本号，
# 例如 governance 侧 `governance-v1`、revise 侧 `revise-v1`。
DEFAULT_PROMPT_VERSION = "deepseek-offline-v1"

__all__ = [
    "PROVIDER", "MODEL_ID", "BASE_URL", "OFFLINE_MODEL_NAME", "ENV_API_KEY",
    "DEFAULT_TIMEOUT_S", "DEFAULT_CONNECT_TIMEOUT_S", "HEALTH_TIMEOUT_S",
    "DEFAULT_MAX_RETRIES", "BACKOFF_BASE_S", "BACKOFF_FACTOR", "BACKOFF_MAX_S",
    "DEFAULT_MAX_TOKENS", "DEFAULT_CONCURRENCY", "RETRY_STATUS_CODES",
    "AUTH_STATUS_CODES", "REASONING_EFFORT_CHOICES", "DEFAULT_PROMPT_VERSION",
    "OfflineServiceUnavailable", "OfflineAuthError", "OfflineNotConfiguredError",
    "OfflineRequestError", "OfflineResponseError",
    "DeepSeekCallStats", "ChatMeta", "DeepSeekClient",
    "classify_error", "endpoint_hash",
]


# --------------------------------------------------------------------------- #
# 异常族（fail-closed：任何一类都不得静默降级为 mock / 在线链）
# --------------------------------------------------------------------------- #
class OfflineServiceUnavailable(RuntimeError):
    """离线治理模型服务不可用（超时/限流/5xx 重试耗尽、连接失败、健康检查失败）。

    调用方（离线 driver）必须据此记 ``service_unavailable`` 或 quarantine（§3.4），
    **不得**切在线链、不得用 mock 结果推进 Readiness（红线 8）。
    """


class OfflineAuthError(RuntimeError):
    """认证失败（401/403）或缺少密钥：立即失败，**不重试**。"""


class OfflineNotConfiguredError(OfflineAuthError):
    """``DEEPSEEK_API_KEY`` 未注入（既无显式参数也无环境变量）。

    继承 `OfflineAuthError`，使"密钥缺失"与"认证被拒"共用同一捕获点，
    同时保留可区分的类型（便于 driver 记 ``gpt6_not_configured`` 同类原因）。
    """


class OfflineRequestError(RuntimeError):
    """请求参数错误等**不可重试**的 4xx（如 400），或非法调用参数。"""


class OfflineResponseError(RuntimeError):
    """响应不可用（空 content / 结构异常）。

    属于"模型没给出可用文本"，与"服务不可用"区分开；同样 fail-closed，
    不重试、不返回空串给调用方。
    """


# --------------------------------------------------------------------------- #
# 脱敏与分类工具
# --------------------------------------------------------------------------- #
_BEARER_RE = re.compile(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?\S+")
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9_\-]{6,}")


def _scrub_secrets(text: Any, *secrets: str | None) -> str:
    """去掉文本中的密钥 / Authorization 头 / ``sk-`` 形态串（异常信息绝不回显密钥）。"""
    out = str(text)
    for secret in secrets:
        if secret:
            out = out.replace(str(secret), "***")
    out = _BEARER_RE.sub(r"\1***", out)
    out = _SK_RE.sub("***", out)
    return out


def _summarize_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict]:
    """把 messages 压成"形状"摘要（角色 + 字符数），绝不带内容。"""
    out: list[dict] = []
    for m in messages:
        content = m.get("content") if isinstance(m, Mapping) else None
        if isinstance(content, str):
            n_chars, kind = len(content), "str"
        elif isinstance(content, (list, tuple)):
            n_chars, kind = sum(len(str(p)) for p in content), "parts"
        else:
            n_chars, kind = 0, type(content).__name__
        out.append({"role": str(m.get("role", "?")) if isinstance(m, Mapping) else "?",
                    "content_kind": kind, "n_chars": n_chars})
    return out


def _status_code_of(exc: BaseException) -> int | None:
    """从异常里取出 HTTP 状态码（openai SDK / httpx / 假 transport 都兼容）。"""
    for attr in ("status_code", "status", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def classify_error(exc: BaseException) -> str:
    """把异常归族为 ``"auth"`` / ``"retry"`` / ``"fatal"``（§3.4 重试纪律）。

    - 401/403（含 openai ``AuthenticationError`` / ``PermissionDeniedError``）→ ``auth``；
    - 429 与 5xx → ``retry``；
    - 其余 4xx（参数错误等）→ ``fatal``（不盲重试）；
    - 无状态码的传输层错误（超时/连接）→ ``retry``。
    """
    status = _status_code_of(exc)
    if status in AUTH_STATUS_CODES:
        return "auth"
    if status is not None:
        if status in RETRY_STATUS_CODES or 500 <= status < 600:
            return "retry"
        if 400 <= status < 500:
            return "fatal"
    name = type(exc).__name__
    if isinstance(exc, (PermissionError, FileNotFoundError)):
        return "fatal"  # 本机文件/权限类错误不属于"服务可恢复"，不重试
    if isinstance(exc, (TimeoutError, ConnectionError, BrokenPipeError, OSError)):
        # 传输层失败（TimeoutError/ConnectionError 均为 OSError 子类）：可恢复
        return "retry"
    if any(tok in name for tok in ("Timeout", "Connection", "ConnectError",
                                   "ReadError", "ReadTimeout", "RateLimit",
                                   "InternalServer", "ServiceUnavailable")):
        return "retry"
    return "fatal"


def _retry_after_s(exc: BaseException) -> float | None:
    """读 ``Retry-After`` 头（429 常用）；缺失或非法返回 None。"""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:  # noqa: BLE001 - 假 headers 对象
        return None
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _sleep(seconds: float) -> None:
    """退避等待；测试可 monkeypatch 本函数实现零等待断言。"""
    if seconds and seconds > 0:
        time.sleep(float(seconds))


def endpoint_hash(base_url: str) -> str:
    """``sha256`` 全摘要（十六进制）——RunManifest 只落哈希，不落 base URL 本体（§19.2）。

    对 base URL 做 ``strip().rstrip("/")`` 归一化后摘要，保证 ``.../`` 与 ``...``
    得到同一哈希。
    """
    normalized = str(base_url).strip().rstrip("/")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _content_sha256(text: str) -> str:
    """文本摘要（元数据里用它替代内容本体）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_flash_model(name: Any, model_id: str = MODEL_ID) -> bool:
    """返回模型名是否为当前可用的 Flash 模型（官方 id 如 ``deepseek-flash``）。"""
    token = str(name or "").strip().lower()
    if not token:
        return False
    if token == str(model_id).strip().lower():
        return True
    return FLASH_NAME_MARKER in token


# --------------------------------------------------------------------------- #
# 统计与元数据（进 RunManifest / receipt；不含任何内容与密钥）
# --------------------------------------------------------------------------- #
@dataclass
class DeepSeekCallStats:
    """调用统计（§19.2 的 ``latency`` / ``token_usage`` 字段来源）。"""

    n_calls: int = 0                 # 成功返回文本的调用数
    n_failures: int = 0              # 最终失败的调用数（不含"重试后成功"的调用）
    n_attempts: int = 0              # HTTP 尝试次数（含重试）
    n_retries: int = 0               # 实际发生的重试次数
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    n_calls_with_usage: int = 0      # 返回了 usage 的调用数
    n_calls_without_usage: int = 0   # 未返回 usage（诚实标注，不推断）
    latency_s: list[float] = field(default_factory=list)  # 成功调用延迟（秒）
    request_ids: list[str] = field(default_factory=list)  # 响应的 request id（若返回）

    @property
    def median_latency_s(self) -> float:
        if not self.latency_s:
            return 0.0
        xs = sorted(self.latency_s)
        n = len(xs)
        return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

    @property
    def max_latency_s(self) -> float:
        return max(self.latency_s) if self.latency_s else 0.0

    @property
    def p95_latency_s(self) -> float:
        if not self.latency_s:
            return 0.0
        xs = sorted(self.latency_s)
        idx = min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))
        return xs[idx]

    @property
    def total_latency_s(self) -> float:
        return float(sum(self.latency_s))

    def latency_fields(self) -> dict:
        """§19.2 ``latency``：只给聚合量（原始列表留在实例属性供 receipt 使用）。"""
        return {
            "n_calls": self.n_calls,
            "n_failures": self.n_failures,
            "n_attempts": self.n_attempts,
            "n_retries": self.n_retries,
            "median_s": round(self.median_latency_s, 3),
            "p95_s": round(self.p95_latency_s, 3),
            "max_s": round(self.max_latency_s, 3),
            "total_s": round(self.total_latency_s, 3),
        }

    def token_fields(self) -> dict:
        """§19.2 ``token_usage``。"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "n_calls_with_usage": self.n_calls_with_usage,
            "n_calls_without_usage": self.n_calls_without_usage,
        }

    def as_dict(self) -> dict:
        """合并视图（latency + token_usage + request id 尾部）。"""
        out = self.latency_fields()
        out.update(self.token_fields())
        out["request_ids"] = list(self.request_ids[-MAX_RECORDED_REQUEST_IDS:])
        return out


@dataclass
class ChatMeta:
    """单次 chat 调用的元数据（原始文本在 ``text``；本对象不含 prompt 内容）。"""

    text: str
    model: str
    request_id: str | None
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    usage_returned: bool
    latency_s: float
    n_attempts: int
    prompt_version: str
    thinking: bool
    reasoning_effort: str | None
    max_tokens: int
    seed: int | None = None
    seed_sent_to_server: bool = False

    @property
    def truncated(self) -> bool:
        """``finish_reason == "length"``：max_tokens 不足（thinking 下常见）——必须可见。"""
        return (self.finish_reason or "").lower() == "length"

    def manifest_fields(self) -> dict:
        """元数据落盘形态：**文本只给 sha256/长度**，不含内容、不含密钥。"""
        return {
            "model": self.model,
            "request_id": self.request_id,
            "finish_reason": self.finish_reason,
            "truncated": self.truncated,
            "text_sha256": _content_sha256(self.text),
            "n_chars": len(self.text),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "usage_returned": self.usage_returned,
            "latency_s": round(self.latency_s, 3),
            "n_attempts": self.n_attempts,
            "prompt_version": self.prompt_version,
            "thinking": "enabled" if self.thinking else "disabled",
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "seed_sent_to_server": self.seed_sent_to_server,
        }


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #
class DeepSeekClient:
    """DeepSeek-V4.1-Flash（``deepseek-flash``）OpenAI 兼容客户端。仅离线治理链使用。

    设计要点：

    - 密钥**只在调用瞬间**从 ``DEEPSEEK_API_KEY``（或构造参数）解析，不进 manifest、
      不进日志、不进异常文本；
    - 重试由本类自管（SDK ``max_retries=0``），以便审计"总尝试次数"并严格遵守"只重试
      超时/限流/5xx，4xx 不盲重试"；
    - `chat` / `chat_with_meta` 只返回文本 + 元数据，**无副作用**：不解析、不落盘、
      不改 active Skill/Memory、不决定 promote/reject（§3.3 / §23.1 Phase 6）。
    """

    PROVIDER = PROVIDER
    MODEL_ID = MODEL_ID
    BASE_URL = BASE_URL
    OFFLINE_MODEL_NAME = OFFLINE_MODEL_NAME

    def __init__(self, *, api_key: str | None = None, base_url: str = BASE_URL,
                 model: str = MODEL_ID, timeout_s: float = DEFAULT_TIMEOUT_S,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 reasoning_effort: str | None = "high", thinking: bool = True,
                 prompt_version: str = DEFAULT_PROMPT_VERSION,
                 connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
                 default_max_tokens: int = DEFAULT_MAX_TOKENS,
                 client_factory: Callable[..., Any] | None = None) -> None:
        """构造客户端。

        :param api_key: 显式密钥；为 None 时（默认）在每次调用时读 ``DEEPSEEK_API_KEY``。
        :param base_url: 官方 base URL（默认 §3.4 固定值，无 ``/v1`` 后缀）。
        :param model: 官方模型名（默认 ``deepseek-flash``）。
        :param timeout_s: 读取超时（秒）。
        :param max_retries: 最大重试次数（总尝试 = ``max_retries + 1``）。
        :param reasoning_effort: 官方取值 ``none/low/high/max`` 或 None（不发送）。
        :param thinking: 是否启用 thinking；关闭时**显式发** ``{"type": "disabled"}``
            （官方默认 enabled，不显式发就会继续 thinking）。
        :param prompt_version: prompt 模板版本（冻结配置项，进 RunManifest）。
        :param connect_timeout_s: 连接超时（秒）。
        :param default_max_tokens: `chat` 未显式指定时的 ``max_tokens``。
        :param client_factory: 可选传输注入钩子，签名
            ``factory(*, api_key, base_url, timeout, max_retries) -> client``；默认用
            ``openai.OpenAI``。测试可据此或 monkeypatch ``sys.modules["openai"]`` 断网。
        """
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORT_CHOICES:
            raise ValueError(
                f"reasoning_effort={reasoning_effort!r} 非法；官方取值为 "
                f"{list(REASONING_EFFORT_CHOICES)} 或 None"
            )
        self._api_key_explicit = api_key
        self.base_url = str(base_url).strip().rstrip("/")
        self.model = str(model)
        self.timeout_s = float(timeout_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.reasoning_effort = reasoning_effort
        self.thinking = bool(thinking)
        self.prompt_version = str(prompt_version)
        self.default_max_tokens = int(default_max_tokens)
        self.stats = DeepSeekCallStats()
        self.last_health: dict | None = None
        self._client_factory = client_factory
        self._clients: dict[tuple[str, float], Any] = {}
        self._client_lock = threading.Lock()

    # ---- 配置视图（全部不含密钥）----
    @property
    def provider(self) -> str:
        return PROVIDER

    @property
    def endpoint_hash(self) -> str:
        """base URL 的哈希（§19.2：manifest 只落哈希，不落 URL 本体）。"""
        return endpoint_hash(self.base_url)

    @property
    def chat_endpoint(self) -> str:
        return f"{self.base_url}{CHAT_PATH}"

    @property
    def models_endpoint(self) -> str:
        return f"{self.base_url}{MODELS_PATH}"

    @property
    def api_key_configured(self) -> bool:
        """密钥是否可用（**只返回布尔，不返回密钥**）。"""
        return bool(self._resolve_api_key_or_none())

    @property
    def model_id(self) -> str:
        """§19.2 的 ``model_id`` 视图（= 构造参数 ``model``；兼容既有
        ``getattr(client, "model_id", ...)`` 调用方）。"""
        return self.model

    @property
    def configured(self) -> bool:
        """是否可用：base_url / model 为 §3.4 固定值，故只取决于密钥是否注入。

        与 `api_key_configured` 同义，用于兼容既有 ``client.configured`` 预检写法。
        """
        return bool(self._resolve_api_key_or_none())

    @property
    def effective_reasoning_effort(self) -> str | None:
        """实际会发送的 ``reasoning_effort``（thinking 关闭时不发送，故为 None）。"""
        return self.reasoning_effort if self.thinking else None

    def __repr__(self) -> str:  # 绝不打印密钥
        return (f"DeepSeekClient(provider={PROVIDER!r}, model={self.model!r}, "
                f"base_url_hash={self.endpoint_hash!r}, thinking={self.thinking!r}, "
                f"reasoning_effort={self.reasoning_effort!r}, "
                f"prompt_version={self.prompt_version!r}, "
                f"api_key_configured={self.api_key_configured!r}, "
                f"timeout_s={self.timeout_s!r}, max_retries={self.max_retries!r})")

    __str__ = __repr__

    # ---- 密钥解析（只在调用瞬间，绝不落盘/落日志）----
    def _resolve_api_key_or_none(self) -> str | None:
        key = self._api_key_explicit
        if key is None:
            key = os.environ.get(ENV_API_KEY)
        key = (key or "").strip()
        return key or None

    def _require_api_key(self) -> str:
        key = self._resolve_api_key_or_none()
        if not key:
            raise OfflineNotConfiguredError(
                f"{ENV_API_KEY} 未设置：离线治理调用被拒绝（密钥只从环境变量或显式参数"
                "注入，绝不落盘；§3.4）"
            )
        return key

    # ---- 传输构造 ----
    def _make_client(self, api_key: str, timeout_s: float) -> Any:
        if self._client_factory is not None:
            return self._client_factory(api_key=api_key, base_url=self.base_url,
                                        timeout=self._timeout_arg(timeout_s),
                                        max_retries=0)
        import openai  # 延迟 import：便于测试注入假 transport，且不做模块级硬依赖

        return openai.OpenAI(base_url=self.base_url, api_key=api_key,
                             timeout=self._timeout_arg(timeout_s), max_retries=0)

    def _timeout_arg(self, timeout_s: float) -> Any:
        """连接 + 读取双超时（httpx.Timeout）；httpx 不可用时退化为单值。"""
        try:
            import httpx
        except Exception:  # noqa: BLE001 - 极端环境降级为单值超时
            return float(timeout_s)
        return httpx.Timeout(float(timeout_s), connect=self.connect_timeout_s)

    def _client_for(self, api_key: str, timeout_s: float) -> Any:
        """按 (密钥指纹, 超时) 缓存传输对象，令并发调用复用连接池。"""
        fingerprint = _content_sha256(api_key)[:12]
        cache_key = (fingerprint, float(timeout_s))
        with self._client_lock:
            client = self._clients.get(cache_key)
            if client is None:
                # 密钥轮换时丢弃旧传输对象，避免在内存里长期保留旧密钥
                for stale in [k for k in self._clients if k[0] != fingerprint]:
                    self._clients.pop(stale, None)
                client = self._make_client(api_key, timeout_s)
                self._clients[cache_key] = client
            return client

    # ---- 请求构造 ----
    def _thinking_body(self) -> dict:
        return dict(THINKING_ENABLED_BODY if self.thinking else THINKING_DISABLED_BODY)

    def _request_kwargs(self, messages: Sequence[Mapping[str, Any]], *,
                        max_tokens: int | None, seed: int | None,
                        extra: Mapping[str, Any]) -> dict:
        """组装 ``chat.completions.create`` 参数（冻结配置在此固化）。"""
        for forbidden in ("model", "messages", "stream"):
            if forbidden in extra:
                raise ValueError(
                    f"不允许覆盖 {forbidden!r}：模型/消息/非流式为 §3.4 冻结配置，"
                    f"覆盖会让 RunManifest 记录失真"
                )
        if not messages:
            raise ValueError("messages 不能为空")
        kwargs: dict = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "max_tokens": int(max_tokens if max_tokens is not None
                              else self.default_max_tokens),
        }
        extra_body = dict(extra.get("extra_body") or {})
        extra_body.update(self._thinking_body())  # 冻结配置优先：thinking 必须如实发送
        kwargs["extra_body"] = extra_body
        if self.thinking and self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if seed is not None and SEND_SEED_TO_SERVER:
            kwargs["seed"] = int(seed)
        for k, v in extra.items():
            if k != "extra_body":
                kwargs[k] = v
        return kwargs

    # ---- 调用 ----
    def chat(self, messages: Sequence[Mapping[str, Any]], *,
             max_tokens: int | None = None, seed: int | None = None,
             **kw: Any) -> str:
        """调用 ``POST /chat/completions``（非流式），返回**原始文本**。

        ``messages`` 为 OpenAI 风格消息列表（``[{"role": ..., "content": ...}]``）。
        ``seed`` 仅作记录（官方 request body 未列该字段，默认不发送，见
        ``SEND_SEED_TO_SERVER``）；``**kw`` 透传额外请求参数（不得覆盖 model /
        messages / stream）。

        本方法**无副作用**：不解析文本、不落盘、不改 active Skill/Memory、不决定
        promote/reject；Schema/泄漏/确定性准入门由调用方执行（§3.3 / §3.4）。
        """
        return self.chat_with_meta(messages, max_tokens=max_tokens, seed=seed, **kw).text

    def chat_with_meta(self, messages: Sequence[Mapping[str, Any]], *,
                       max_tokens: int | None = None, seed: int | None = None,
                       **kw: Any) -> ChatMeta:
        """同 `chat`，但返回 `ChatMeta`（原始文本 + 可进 manifest 的元数据）。"""
        api_key = self._require_api_key()
        payload = self._request_kwargs(_to_messages(messages), max_tokens=max_tokens,
                                       seed=seed, extra=kw)
        client = self._client_for(api_key, self.timeout_s)
        logger.debug(
            "DeepSeek 请求: model=%s endpoint_hash=%s thinking=%s reasoning_effort=%s "
            "max_tokens=%s n_messages=%d messages=%s",
            self.model, self.endpoint_hash,
            "enabled" if self.thinking else "disabled",
            self.effective_reasoning_effort,
            payload.get("max_tokens"), len(payload["messages"]),
            _summarize_messages(payload["messages"]),
        )
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            self.stats.n_attempts += 1
            try:
                resp = client.chat.completions.create(**payload)
            except Exception as exc:  # noqa: BLE001 - 按 §3.4 归族后决定重试/失败
                kind = classify_error(exc)
                reason = _scrub_secrets(f"{type(exc).__name__}: {exc}", api_key)[:400]
                status = _status_code_of(exc)
                if kind == "auth":
                    self.stats.n_failures += 1
                    raise OfflineAuthError(
                        f"DeepSeek 认证失败（HTTP {status}），不重试（§3.4）：{reason}"
                    ) from exc
                if kind == "fatal":
                    self.stats.n_failures += 1
                    raise OfflineRequestError(
                        f"DeepSeek 请求不可重试（HTTP {status}）：{reason}"
                    ) from exc
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                delay = self._backoff_delay(attempt, exc)
                self.stats.n_retries += 1
                logger.warning(
                    "DeepSeek 调用失败（第 %d/%d 次尝试，status=%s）：%s → %.1fs 后重试",
                    attempt + 1, self.max_retries + 1, status, reason, delay,
                )
                _sleep(delay)
                continue
            return self._parse_response(resp, t0, attempts=attempt + 1, seed=seed,
                                        max_tokens=payload.get("max_tokens"))
        self.stats.n_failures += 1
        raise OfflineServiceUnavailable(
            "DeepSeek 服务不可用（重试 %d 次后仍失败）：%s；离线任务须记 "
            "service_unavailable 或 quarantine，禁止切在线链 / mock（§3.4）"
            % (self.max_retries,
               _scrub_secrets(f"{type(last_exc).__name__}: {last_exc}", api_key)[:400])
        ) from last_exc

    def _backoff_delay(self, attempt: int, exc: BaseException) -> float:
        """指数退避（``base * factor**attempt``，上限 ``BACKOFF_MAX_S``）；优先 Retry-After。"""
        retry_after = _retry_after_s(exc)
        if retry_after is not None:
            return min(retry_after, BACKOFF_MAX_S)
        return min(BACKOFF_BASE_S * (BACKOFF_FACTOR ** attempt), BACKOFF_MAX_S)

    def _parse_response(self, resp: Any, t0: float, *, attempts: int,
                        seed: int | None, max_tokens: int | None) -> ChatMeta:
        """解析响应为 `ChatMeta`；空/畸形 content 一律 fail-closed。"""
        latency = time.perf_counter() - t0
        try:
            choice = resp.choices[0]
            content = getattr(getattr(choice, "message", None), "content", None)
        except Exception as exc:  # noqa: BLE001 - 响应结构异常
            self.stats.n_failures += 1
            raise OfflineResponseError(
                f"DeepSeek 响应结构异常（无 choices/message）：{type(exc).__name__}"
            ) from exc
        finish_reason = getattr(choice, "finish_reason", None)
        if not isinstance(content, str) or not content.strip():
            self.stats.n_failures += 1
            raise OfflineResponseError(
                "DeepSeek 返回空 content（finish_reason=%s）：拒绝把空文本交给下游，"
                "请检查 max_tokens 是否被 thinking 吃满（§3.4）" % finish_reason
            )
        usage = getattr(resp, "usage", None)
        usage_returned = usage is not None
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage_returned else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) \
            if usage_returned else 0
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage_returned else 0
        if usage_returned and not total_tokens:
            total_tokens = prompt_tokens + completion_tokens
        request_id = getattr(resp, "id", None)

        self.stats.n_calls += 1
        self.stats.latency_s.append(latency)
        self.stats.prompt_tokens += prompt_tokens
        self.stats.completion_tokens += completion_tokens
        self.stats.total_tokens += total_tokens
        if usage_returned:
            self.stats.n_calls_with_usage += 1
        else:
            self.stats.n_calls_without_usage += 1
        if isinstance(request_id, str) and request_id:
            self.stats.request_ids.append(request_id)
        if (finish_reason or "").lower() == "length":
            logger.warning(
                "DeepSeek 响应被 max_tokens 截断（finish_reason=length, n_chars=%d）",
                len(content),
            )
        return ChatMeta(
            text=content, model=self.model,
            request_id=request_id if isinstance(request_id, str) else None,
            finish_reason=finish_reason, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, total_tokens=total_tokens,
            usage_returned=usage_returned, latency_s=latency, n_attempts=attempts,
            prompt_version=self.prompt_version, thinking=self.thinking,
            reasoning_effort=self.reasoning_effort if self.thinking else None,
            max_tokens=int(max_tokens if max_tokens is not None else self.default_max_tokens),
            seed=seed, seed_sent_to_server=bool(seed is not None and SEND_SEED_TO_SERVER),
        )

    # ---- 批量并发 ----
    def chat_many(self, prompts: Sequence[Any], *,
                  concurrency: int = DEFAULT_CONCURRENCY,
                  system: str | None = None,
                  max_tokens: int | None = None,
                  seed: int | None = None,
                  return_exceptions: bool = False,
                  **kw: Any) -> list:
        """并发调用（掩盖单次延迟；离线链必须批量 + 并发，见交接记录 §3）。

        ``prompts`` 每项可为字符串（当作单条 user 消息）或 OpenAI 风格 messages 列表。
        返回列表与输入**同序**；默认 fail-closed（首个异常直接抛），
        ``return_exceptions=True`` 时把异常放在对应位置返回，供调用方按 §3.4 记
        ``service_unavailable`` / quarantine，而不误判成"模型答得不好"。
        """
        if not prompts:
            return []
        n_workers = max(1, min(int(concurrency), len(prompts)))
        out: list = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for idx, item in enumerate(prompts):
                futures[pool.submit(self.chat, _to_messages(item, system),
                                    max_tokens=max_tokens, seed=seed, **kw)] = idx
            for fut, idx in futures.items():
                try:
                    out[idx] = fut.result()
                except Exception as exc:  # noqa: BLE001 - fail-closed，由调用方决定降级
                    if not return_exceptions:
                        raise
                    out[idx] = exc
        return out

    # ---- 服务健康（§3.4 启动前健康检查）----
    def list_models(self) -> list[str]:
        """``GET {base}/models`` → 可用模型 id 列表；失败按其类型抛错（不吞）。"""
        api_key = self._require_api_key()
        client = self._client_for(api_key, HEALTH_TIMEOUT_S)
        resp = client.models.list()
        data = getattr(resp, "data", None)
        if data is None and isinstance(resp, Mapping):
            data = resp.get("data")
        names: list[str] = []
        for item in data or []:
            if isinstance(item, Mapping):
                name = item.get("id")
            else:
                name = getattr(item, "id", None)
            if name:
                names.append(str(name))
        return names

    def health_report(self) -> dict:
        """健康检查明细（判定 + 原因 + 计数）；**只含形状，不含密钥**。"""
        report: dict = {"ok": False, "reason": "", "n_models": 0,
                        "flash_available": False, "models": []}
        try:
            api_key = self._require_api_key()
        except OfflineAuthError as exc:
            report["reason"] = f"api_key_missing: {type(exc).__name__}"
            self.last_health = report
            return report
        try:
            names = self.list_models()
        except Exception as exc:  # noqa: BLE001 - 健康检查失败即 fail-closed
            report["reason"] = _scrub_secrets(
                f"{type(exc).__name__}: {exc}", api_key)[:300]
            self.last_health = report
            return report
        report["models"] = names
        report["n_models"] = len(names)
        report["flash_available"] = any(_is_flash_model(n, self.model) for n in names)
        report["ok"] = bool(report["flash_available"])
        report["reason"] = "ok" if report["ok"] else "flash_model_not_listed"
        self.last_health = report
        return report

    def health_check(self) -> bool:
        """``GET /models`` 确认当前可用的 Flash 模型存在；失败返回 False。

        失败**不等于**可以用 mock 顶替：调用方必须走 `require_service()` 抛错 →
        ``service_unavailable`` / quarantine（§3.4、红线 8）。
        """
        return bool(self.health_report()["ok"])

    def require_service(self) -> None:
        """健康检查不通过则抛 `OfflineServiceUnavailable`（启动前硬门）。"""
        report = self.health_report()
        if not report["ok"]:
            raise OfflineServiceUnavailable(
                f"DeepSeek 健康检查失败（{report['reason']}；endpoint_hash="
                f"{self.endpoint_hash}）：离线任务须记 service_unavailable 或 "
                "quarantine，禁止切在线链、禁止用 mock 结果推进 Readiness（§3.4）"
            )

    # ---- §19.2 RunManifest 字段 ----
    def request_params(self) -> dict:
        """冻结请求参数（可审计；thinking / reasoning_effort 的具体值在此固化）。"""
        return {
            "stream": False,
            "thinking": "enabled" if self.thinking else "disabled",
            "thinking_extra_body": self._thinking_body(),
            "reasoning_effort": self.effective_reasoning_effort,
            "max_tokens": self.default_max_tokens,
            "temperature": "server_default",  # thinking 模式下 temperature 无效，不发送
            "seed_sent_to_server": SEND_SEED_TO_SERVER,
            "timeout_s": self.timeout_s,
            "connect_timeout_s": self.connect_timeout_s,
            "max_retries": self.max_retries,
            "backoff": {"base_s": BACKOFF_BASE_S, "factor": BACKOFF_FACTOR,
                        "max_s": BACKOFF_MAX_S,
                        "retry_on_status": sorted(RETRY_STATUS_CODES)},
            "endpoint": {"hash": self.endpoint_hash, "hash_alg": ENDPOINT_HASH_ALG,
                         "chat_path": CHAT_PATH, "models_path": MODELS_PATH},
        }

    def manifest_fields(self) -> dict:
        """§19.2「离线治理模型字段」。

        含 ``offline_model`` / ``provider`` / ``model_id`` / ``endpoint_hash`` /
        ``prompt_version`` / ``latency`` / ``token_usage``，外加冻结请求参数与健康检查
        计数（thinking / reasoning_effort 的具体值在此可审计）。

        **绝不含**：API key、Authorization 头、base URL 本体（只给哈希）、完整环境变量、
        任何 prompt/响应内容。
        """
        health = self.last_health or {}
        return {
            "offline_model": OFFLINE_MODEL_NAME,
            "provider": PROVIDER,
            "model_id": self.model,
            "endpoint_hash": self.endpoint_hash,
            "endpoint_hash_alg": ENDPOINT_HASH_ALG,
            # §3.4 称 base_url_hash、§19.2 称 endpoint_hash：同值双写，任一侧消费都能读到
            "base_url_hash": self.endpoint_hash,
            "prompt_version": self.prompt_version,
            "latency": self.stats.latency_fields(),
            "token_usage": self.stats.token_fields(),
            "request_params": self.request_params(),
            # 扁平冗余：便于审计脚本直接读 thinking / reasoning_effort（§3.4 要求冻结值可审计）
            "thinking": "enabled" if self.thinking else "disabled",
            "reasoning_effort": self.effective_reasoning_effort,
            "health": {"checked": bool(health),
                       "ok": bool(health.get("ok", False)),
                       "reason": str(health.get("reason", ""))[:200],
                       "n_models_listed": int(health.get("n_models", 0) or 0),
                       "flash_available": bool(health.get("flash_available", False))},
        }


def _to_messages(item: Any, system: str | None = None) -> list[dict]:
    """把 ``str`` 或 messages 列表统一成 messages（可选前置 system）。"""
    if isinstance(item, str):
        messages: list[dict] = [{"role": "user", "content": item}]
    elif isinstance(item, Mapping):
        messages = [dict(item)]
    else:
        messages = [dict(m) for m in item]
    if system:
        messages = [{"role": "system", "content": system}, *messages]
    return messages
