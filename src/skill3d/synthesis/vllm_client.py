"""本地 vLLM OpenAI 兼容客户端（§4 M8 / §6.1）。

temperature=0.0 greedy；支持 DP×N 多 endpoint 轮询。
仅调本地 Qwen3-VL-8B（硬约束 1：在线链绝对无 GPT-6）。

§6.1 服务纪律：
- endpoint 归一化为 base_url（避免 `/v1/v1` 重复拼接的 404，C-1）；
- 启动前**健康检查 + 模型名检查**（`check_service` / `probe_service`）；
- 服务故障记 `service_unavailable`（不是 crash、不是静默降级）：`chat` 把
  连接/超时/5xx 归族为 `ServiceUnavailable`，由 runner 记该 episode `unavailable`。
"""

from __future__ import annotations

import itertools
import json
import urllib.error
import urllib.request
from typing import Optional, Sequence

from skill3d.synthesis.request_context import current_request_phase

def _normalize_base_url(endpoint: str) -> str:
    """把 endpoint 归一化为 OpenAI 兼容 base_url（保留末尾 /v1，去掉多余斜杠）。"""
    ep = str(endpoint).strip().rstrip("/")
    while ep.endswith("/v1/v1"):
        ep = ep[: -len("/v1")]
    if not ep.endswith("/v1"):
        ep += "/v1"
    return ep


def _root_url(endpoint: str) -> str:
    """去掉 `/v1` 的根地址（vLLM 的 /health 与 /v1/models 前缀不同）。"""
    ep = _normalize_base_url(endpoint)
    return ep[: -len("/v1")]


class ServiceUnavailable(RuntimeError):
    """本地推理服务不可用（健康检查失败/连接失败/模型名不匹配）。"""


def health_check(endpoint: str, timeout_s: float = 5.0) -> tuple[bool, str]:
    """vLLM `GET /health` 探活；返回 `(是否就绪, 原因)`。"""
    url = _root_url(endpoint) + "/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310 - 本地服务
            code = int(getattr(resp, "status", 200))
        return (code == 200), f"HTTP {code}"
    except urllib.error.HTTPError as exc:  # 服务在但未就绪
        return False, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - 未起服务/网络不可达
        return False, f"{type(exc).__name__}: {exc}"


def list_models(endpoint: str, timeout_s: float = 5.0) -> list[str]:
    """`GET /v1/models` → served model 名列表；失败返回空列表。"""
    url = _normalize_base_url(endpoint) + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310 - 本地服务
            payload = json.loads(resp.read().decode("utf-8"))
        return [str(m.get("id", "")) for m in payload.get("data", []) if m.get("id")]
    except Exception:  # noqa: BLE001
        return []


class VLLMClient:
    """DP 多 endpoint 轮询的 OpenAI 兼容 chat 客户端。

    endpoint 可写 `http://host:8100` 或 `http://host:8100/v1`（两种都接受，
    重复的 `/v1` 会被归一化，避免 `/v1/v1/chat/completions` 的 404）。
    """

    def __init__(
        self,
        endpoints: Sequence[str],
        model: str,
        timeout_s: float = 120.0,
        *,
        check_on_init: bool = False,
    ) -> None:
        from openai import OpenAI

        if not endpoints:
            raise ValueError("endpoints 不能为空")
        # v9 §9.4：最近一次请求的**真实** token 用量（含图像 token）；runner 逐轮落盘
        self.last_usage: dict = {}
        self._clients = [
            OpenAI(base_url=_normalize_base_url(ep), api_key="EMPTY", timeout=timeout_s)
            for ep in endpoints
        ]
        self._endpoints = list(endpoints)
        self._rr = itertools.cycle(range(len(self._clients)))
        self.model = model
        self.service_status: dict[str, str] = {}
        if check_on_init:
            ok, reason = self.check_service()
            if not ok:
                raise ServiceUnavailable(f"本地推理服务不可用: {reason}")

    # ---- §6.1 启动前检查 ----
    def check_service(self) -> tuple[bool, str]:
        """逐 endpoint 健康检查 + 模型名检查；返回 `(全部就绪, 原因摘要)`。"""
        problems: list[str] = []
        for ep in self._endpoints:
            ok, reason = health_check(ep)
            self.service_status[ep] = reason
            if not ok:
                problems.append(f"{ep} 健康检查失败（{reason}）")
                continue
            names = list_models(ep)
            if names and self.model not in names:
                problems.append(f"{ep} 未提供模型 {self.model}（在服务: {names}）")
        if problems:
            return False, "；".join(problems)
        return True, f"{len(self._endpoints)} 个 endpoint 就绪（model={self.model}）"

    def chat(self, messages: list[dict], max_tokens: int = 4096,
             seed: Optional[int] = None) -> str:
        """greedy（temperature=0.0）生成，返回文本内容。

        **复现性注意（实测）**：即使 temperature=0，vLLM 在连续批处理 + 多副本
        轮询下的数值路径会随批组成变化，长生成里偶发 token 翻转。实测同一份 32 题
        样本连跑两次，per-task 结果出现 20.94 → 14.38 的摆动。故：
        - 这里显式传请求级 `seed`（vLLM 支持），把随机性来源钉住；
        - 复现性实验/paired A/B 应使用**单一 endpoint**（`--vllm-endpoint` 只给一个），
          避免多副本带来的批组成差异；
        - 报告里必须给出"同配置重复运行"的噪声底，小于噪声底的增益不算增益。

        服务不可用/模型名错误 → 抛 `ServiceUnavailable`（记 `service_unavailable`），
        与"模型答得不对"严格区分开。
        """
        client = self._clients[next(self._rr)]
        # 评测增量请求最多发一次；避免 OpenAI SDK 隐式重试突破该上限。
        if current_request_phase() == "eval_visual_fallback":
            client = client.with_options(max_retries=0)
        self.last_usage = {}
        kwargs: dict = {"model": self.model, "messages": messages,
                        "temperature": 0.0, "max_tokens": max_tokens}
        if seed is not None:
            kwargs["seed"] = int(seed)
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - 归族为服务不可用
            name = type(exc).__name__
            if any(k in name for k in ("Connection", "Timeout", "APIStatus",
                                      "InternalServer", "NotFound")):
                raise ServiceUnavailable(f"{name}: {exc}") from exc
            raise
        content = resp.choices[0].message.content
        # v9 §9.4："实际传入图像、缩放和 token 成本完整记录" —— 服务返回的 usage 是
        # **真实** token 数（含图像 token），逐请求留在客户端上供 runner 落盘。
        try:
            usage = getattr(resp, "usage", None)
            self.last_usage = (usage.model_dump() if usage is not None
                               and hasattr(usage, "model_dump") else dict(usage or {}))
        except Exception:  # noqa: BLE001 - usage 缺失不得影响作答
            self.last_usage = {}
        if content is None:
            raise RuntimeError("vLLM 返回空 content")
        return content


def probe_service(endpoint: str, model: str = "") -> dict:
    """服务探活汇总（CLI 预检与审计用）：不抛异常，返回状态字典。"""
    ok, reason = health_check(endpoint)
    names = list_models(endpoint) if ok else []
    model_ok: Optional[bool] = None
    if ok and model and names:
        model_ok = model in names
    return {"endpoint": endpoint, "healthy": ok, "health": reason,
            "models": names, "model_checked": model_ok}
