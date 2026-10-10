"""开放词表检测器客户端（GroundingDINO 服务；**确定性 Tool**，非 LLM）。

**为什么需要它**（2026-09-20 真实链实测）：

M5 的对象绑定正解来源是"在线 Qwen3-VL-8B 给对象名 + bbox 提示"（§4 M5 字段 12）。
实测该 8B 模型在**定位**上不可靠：对同一帧，纯文本列举能说出"电脑主机"，但一旦要求
输出 0–1000 归一化 JSON bbox 就返回 `[]` 或 `NONE`（连"不确定也给最可能的框"都不行）。
后果不是"少检出一个对象"，而是 M5 抛错 → `materialized=False` → **整个 scene 的对象类
Tool 被 fail-closed 收回**（计数、相对方向、路线规划、外观顺序全线退化）。

因此加一层**确定性检测器兜底**：VLM 提示为空时，用本地 GroundingDINO 服务按问题名词
做开放词表检测，拿到真实框再做 SAM2 传播。检测器不是 LLM，不进"在线链无 GPT-6"的
约束范围（硬约束 1 只禁 GPT-6）；它的输出仍然只作为 SAM2 的 box prompt，
不参与任何答案生成。

服务契约（组内 `grounding_dino_server.py`）：`POST /infer`
`{"image": <base64 jpg>, "text_prompt": "a. b. c.", "box_threshold": 0.25}`
→ `{"success": bool, "detections": [{"label","bbox":[x0,y0,x1,y1] 像素,"confidence"}]}`
"""

from __future__ import annotations

import base64
import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

DEFAULT_ENDPOINT = os.environ.get("SKILL3D_DETECTOR_ENDPOINT", "")
DEFAULT_BOX_THRESHOLD = 0.25   # TODO_CALIBRATE
DEFAULT_TIMEOUT_S = 120
DEFAULT_TOTAL_TIMEOUT_S = 180
_failure_sink: ContextVar[Optional[list[str]]] = ContextVar("detector_failure_sink", default=None)
_attempt_sinks: ContextVar[tuple] = ContextVar("detector_attempt_sinks", default=())


@contextmanager
def capture_detector_attempts():
    """Nested captures retain attempts without treating recovered errors as outages."""
    attempts: list[dict] = []
    token = _attempt_sinks.set((*_attempt_sinks.get(), attempts))
    try:
        yield attempts
    finally:
        _attempt_sinks.reset(token)


def current_detector_attempts() -> list[dict]:
    sinks = _attempt_sinks.get()
    return list(sinks[-1]) if sinks else []


def _record_attempt(row: dict) -> None:
    for sink in _attempt_sinks.get():
        sink.append(dict(row))


@contextmanager
def capture_detector_failures():
    """Observe actual failures per evaluation run, including M5's caught errors."""
    failures: list[str] = []
    parent = _failure_sink.get()
    token = _failure_sink.set(failures)
    try:
        yield failures
    finally:
        _failure_sink.reset(token)
        if parent is not None:
            parent.extend(failures)


def current_detector_failures() -> list[str]:
    return list(_failure_sink.get() or [])


def _failed(message: str) -> list:
    detect.last_error = message
    sink = _failure_sink.get()
    if sink is not None:
        sink.append(message)
    return []


@dataclass
class Detection:
    label: str
    bbox_xyxy: tuple[float, float, float, float]   # 像素坐标（左上原点）
    confidence: float


def detector_endpoint() -> str:
    """检测器 endpoint（空 = 未配置 → 不启用兜底）。"""
    return str(os.environ.get("SKILL3D_DETECTOR_ENDPOINT", DEFAULT_ENDPOINT) or "").strip()


def available() -> bool:
    return bool(detector_endpoint())


def detect(frame: np.ndarray, prompt: str, *,
           endpoint: Optional[str] = None,
           box_threshold: float = DEFAULT_BOX_THRESHOLD,
           timeout_s: float = DEFAULT_TIMEOUT_S,
           max_attempts: int = 3,
           total_timeout_s: float = DEFAULT_TOTAL_TIMEOUT_S) -> list[Detection]:
    """对单帧做开放词表检测；服务不可用/解析失败 → 返回空列表（由调用方降级）。

    **原因记在 `detect.last_error`**（2026-09-21 修）：旧实现静默返回 `[]`，调用方
    无法区分"检测服务挂了"与"这一帧确实没有该类物体"——实测检测器故障时对象清单
    悄悄退化成"只有 VLM 框"（同一 scene 只剩 2 个对象），日志里毫无痕迹。返回值
    语义不变（仍返回空列表，不抛错、不阻断 episode），只是把失败原因暴露出来供
    调用方记录/重试。
    """
    ep = (endpoint or detector_endpoint()).rstrip("/")
    detect.last_error = ""
    if not ep or not prompt.strip() or frame is None:
        return _failed("endpoint 未配置 / prompt 为空 / frame 为 None")
    import cv2
    import requests

    img = np.asarray(frame)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                          [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return _failed("JPEG 编码失败")
    if timeout_s <= 0 or total_timeout_s <= 0 or not 1 <= max_attempts <= 5:
        return _failed("invalid detector retry budget")
    body = {"image": base64.b64encode(buf.tobytes()).decode("ascii"),
            "text_prompt": prompt, "box_threshold": float(box_threshold)}
    digest = hashlib.sha256(buf.tobytes() + prompt.encode()).hexdigest()
    deadline = time.monotonic() + total_timeout_s
    data = None
    for attempt in range(1, max_attempts + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _failed("detector total timeout budget exhausted")
        started = time.monotonic()
        status_code = None
        try:
            resp = requests.post(
                f"{ep}/infer", json=body,
                # 健康检测曾实测耗时 42–59s，不能为预留重试把首次窗口缩成 40s。
                timeout=min(timeout_s, remaining))
            status_code = resp.status_code
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError("detector response must be an object")
            if data.get("success") is False:
                raise ValueError(str(data.get("error") or "detector returned success=false"))
            if not isinstance(data.get("detections"), list):
                raise ValueError("detector response must include detections list")
        except Exception as exc:  # noqa: BLE001 - explicit non-retryable default
            retryable = (isinstance(exc, (requests.Timeout, requests.ConnectionError))
                         or (isinstance(exc, requests.HTTPError)
                             and status_code is not None
                             and (status_code == 429 or 500 <= status_code < 600)))
            delay = 0.2 * attempt
            retry = (retryable and attempt < max_attempts
                     and deadline - time.monotonic() > delay)
            message = f"{type(exc).__name__}: {exc}"
            _record_attempt({
                "request_sha256": digest, "prompt": prompt, "attempt": attempt,
                "status": "error", "http_status": status_code, "error": message,
                "retryable": retryable, "will_retry": retry,
                "elapsed_s": time.monotonic() - started,
            })
            if not retry:
                return _failed(message)
            time.sleep(delay)
            continue
        _record_attempt({
            "request_sha256": digest, "prompt": prompt, "attempt": attempt,
            "status": "ok" if data["detections"] else "empty",
            "http_status": status_code, "recovered": attempt > 1,
            "n_detections": len(data["detections"]),
            "elapsed_s": time.monotonic() - started,
        })
        break
    out: list[Detection] = []
    h, w = img.shape[0], img.shape[1]
    for d in (data.get("detections") or []):
        try:
            x0, y0, x1, y1 = [float(v) for v in d["bbox"]]
            confidence = float(d.get("confidence", 0.0))
            if not np.all(np.isfinite([x0, y0, x1, y1, confidence])):
                continue
        except Exception:  # noqa: BLE001
            continue
        # 裁到图像范围内，保证 SAM2 box prompt 合法
        x0, x1 = sorted((max(0.0, min(x0, w - 1)), max(0.0, min(x1, w - 1))))
        y0, y1 = sorted((max(0.0, min(y0, h - 1)), max(0.0, min(y1, h - 1))))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        out.append(Detection(label=str(d.get("label", "")).strip(),
                             bbox_xyxy=(x0, y0, x1, y1),
                             confidence=confidence))
    return out


def to_normalized_1000(dets: Sequence[Detection], frame_hw: tuple[int, int]
                       ) -> list[tuple[str, list[float]]]:
    """像素框 → VLM-normalized-1000（§9 坐标层口径；供统一处理与审计）。"""
    h, w = int(frame_hw[0]), int(frame_hw[1])
    out: list[tuple[str, list[float]]] = []
    for d in dets:
        x0, y0, x1, y1 = d.bbox_xyxy
        out.append((d.label, [round(x0 / max(w, 1) * 1000, 1),
                              round(y0 / max(h, 1) * 1000, 1),
                              round(x1 / max(w, 1) * 1000, 1),
                              round(y1 / max(h, 1) * 1000, 1)]))
    return out


def prompt_from_nouns(nouns: Sequence[str]) -> str:
    """问题名词 → 检测器文本提示（GroundingDINO 用句点分隔的类目串）。"""
    parts = [str(n).strip() for n in nouns if str(n).strip()]
    return ". ".join(parts) + ("." if parts else "")
