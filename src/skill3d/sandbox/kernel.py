"""持久 kernel（§4 M10）：code-as-action，跨 cell 变量存活。

使用 in-process RestrictedNamespaceKernel，在题级受控 namespace 中执行；
结合 AST 白名单和 Tool 授权实现 v11 的程序执行与分支隔离。

注入保留名：frames / scene / tools / show / ReturnAnswer。
提供 cell 超时、错误捕获和可撤销的答案提交。
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import linecache
import signal
import time
import traceback
import types
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from skill3d.schemas import AnswerPayload, ToolCall, ToolResult, parse_answer_payload
from skill3d.schemas.tool import FailureCause
from skill3d.schemas.tool_response import TOOL_RESPONSE_VERSION, model_tool_response
from skill3d.tools.contract import (
    AnswerAlreadyGiven,
    ArtifactUnavailableError,
    ConfidenceGateError,
    DomainValueError,
    ToolContractError,
)
from skill3d.sandbox.ast_guard import normalize_program_source
from skill3d.sandbox.safe_runtime import frozen_frames, safe_builtins
from skill3d.tools.contract import authorize_tool_call
from skill3d.tools.mock_switch import request_key
from skill3d.tools.registry import ToolRegistry
from skill3d.tools.scene_handle import SceneHandle

DEFAULT_CELL_TIMEOUT_S = 30  # TODO_CALIBRATE


class ControlTerminate(BaseException):
    """控制终结信号（v7 §10.2）：host 实现为**不可被生成程序吞掉**的终结操作。

    刻意继承 `BaseException` 而非 `Exception`：生成程序里常见的
    `try: ... except Exception: pass` 防御式写法**吞不掉**它，因此
    "提交答案后不得再调工具/再提交"由运行层确定性保证，不依赖模型的写法自律
    （§10.2"AST 检查与运行时封锁共同保证"）。
    """

    kind = "terminate"

    def __init__(self, payload: Any = None) -> None:
        super().__init__(f"control terminate: {self.kind}")
        self.payload = payload


class AnswerTerminate(ControlTerminate):
    """`ReturnAnswer(...)` 立即结束当前片段；host 验收后才结束本题。"""
    kind = "answer"


class YieldTerminate(ControlTerminate):
    """`YieldObservations(...)` 结束当前程序片段，等待同一 agent 的下一轮。"""
    kind = "yield"


class _AnswerSlot:
    """`ReturnAnswer(answer)` 保留名实现：校验并**立即终结**（v7 §10.2）。

    这明确替换 v6 的"只记录但不中止"语义。旧语义下模型必须把所有计算写在
    ReturnAnswer 之前，且答后调 Tool 由 `AnswerAlreadyGiven` 兜住；新语义下
    `return ReturnAnswer(...)` 是推荐写法，终结由 `AnswerTerminate` 保证。

    答案槽本身仍然记录值：即使生成程序用 `except BaseException` 强行吞掉终结
    信号，上层仍能从 `answer_slot.given` 判定"确实提交过答案"，不会丢答案。

    v9 §10.1/§12：入参可以是完整 `AnswerPayload`。原始对象与适配问题一并保留，
    供 §12 的框架核验使用（`payload` / `adapter_problems`）；`answer` 仍是归一化
    后的提交字符串，评分链路不变。**二次提交被硬拦**（§10.2：提交后不能再提交
    第二次）——此前只有"答后调工具"被拦，重复提交本身会静默覆盖首个答案。
    """

    def __init__(self) -> None:
        self.answer: Optional[str] = None
        self.given: bool = False
        self.payload: Optional[AnswerPayload] = None
        self.adapter_problems: list[str] = []

    def __call__(self, value: Any) -> None:
        if self.given:
            raise AnswerAlreadyGiven(
                "ReturnAnswer",
                f"This cell already submitted an answer ({self.answer!r}); a control call ends the cell. "
                "Do not submit again (§10.2).",
                route="", args={"value": value})
        payload, problems = parse_answer_payload(
            value, question_type=self._question_type)
        self.payload = payload
        self.adapter_problems = problems
        self.answer = _normalize_answer_value(payload.value)
        self.given = True
        raise AnswerTerminate(self.answer)

    # 题型由 kernel 在构造时注入（§4 单位合同查表用）；缺省时空串 → 适配器记问题
    _question_type: str = ""


class _YieldSlot:
    """`YieldObservations(result_ids, reason)` 保留名实现（v7 §10.2/§11.1）。

    结束当前片段并把指定结果（及其必要图像）反馈给**同一个**在线 agent。
    不提交答案；`reason` 用于 trace 归因（"需要判断/需要恢复"）。
    """

    def __init__(self) -> None:
        self.result_ids: list[str] = []
        self.reason: str = ""
        self.given: bool = False

    def __call__(self, result_ids: Any = None, reason: str = "") -> None:
        if result_ids is None:
            ids: list[str] = []
        elif isinstance(result_ids, (str, bytes)):
            ids = [str(result_ids)]
        else:
            ids = [str(x) for x in result_ids]
        self.result_ids = ids
        self.reason = str(reason or "")
        self.given = True
        raise YieldTerminate({"result_ids": ids, "reason": self.reason})


def _normalize_answer_value(value: Any) -> str:
    """把 ReturnAnswer 的入参归一成提交字符串。

    v7 §10.1：`AnswerPayload.value` 可以是 str/int/float。模型常把 numpy 标量
    或 `np.float64` 交进来（`str(np.float64(4.0))` == `"4.0"`，可解析），
    但 `array([2.])` 这类**只含一个元素**的数组也应接受 —— 直接 `str()` 会得到
    `"[2.]"`，评分器解析不出来就变成 0 分，属于纯粹的接口损失。
    """
    if value is None:
        return ""
    try:                                    # numpy 标量/0 维数组友好
        import numpy as _np

        if isinstance(value, _np.generic):
            value = value.item()
        elif isinstance(value, _np.ndarray) and value.size == 1:
            value = value.reshape(-1)[0].item()
    except Exception:                       # noqa: BLE001 - 归一化失败不改变行为
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class _CellTimeout(BaseException):
    """Generated `except Exception` must not suppress the host time limit."""


class _Timeout:
    """Protect the complete cell, including the host-invoked solve entry."""

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self._active = False

    def __enter__(self):
        try:
            self._previous_handler = signal.getsignal(signal.SIGALRM)
            self._started = time.monotonic()
            signal.signal(signal.SIGALRM, self._raise)
            self._previous_timer = signal.setitimer(signal.ITIMER_REAL, self.seconds)
            self._active = True
        except (ValueError, AttributeError) as exc:
            raise RuntimeError("cell timeout requires a main-thread SIGALRM runtime") from exc
        return self

    def __exit__(self, *exc):
        if self._active:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._previous_handler)
            remaining, interval = self._previous_timer
            if remaining:
                remaining = max(0.000001, remaining - (time.monotonic() - self._started))
            signal.setitimer(signal.ITIMER_REAL, remaining, interval)
        return False

    @staticmethod
    def _raise(signum, frame):
        raise _CellTimeout("Cell execution timed out")


@dataclass
class CellResult:
    stdout_tail: str
    error: Optional[str] = None
    # timeout / oom / violation_syntax / violation_runtime / violation_policy / tool_contract
    error_code: Optional[str] = None
    answer: Optional[str] = None
    new_vars: list[str] = field(default_factory=list)
    # D-3：命中契约违规后，ReturnAnswer 的答案不得采纳（上层据此 abstain）
    answer_untrusted: bool = False
    contract_violations: list[dict] = field(default_factory=list)
    # v7 §10.2：本轮如何结束 —— "answer" | "yield" | None（片段跑完但未终结）
    terminated: Optional[str] = None
    # v7 §11.1：YieldObservations 指定的结果与理由（terminated == "yield" 时有意义）
    yielded_result_ids: list[str] = field(default_factory=list)
    yielded_reason: str = ""
    # M3: complete local evidence; model feedback keeps its concise summary.
    stdout: str = ""
    traceback_text: str = ""
    error_type: str = ""
    error_message: str = ""
    error_line: Optional[int] = None
    error_source_line: str = ""
    program_sha256: str = ""
    source_filename: str = ""


def _compile_program(code: str, filename: str = "<episode_program>"):
    """编译生成程序；对**顶层 `return`** 做函数包装（v7 §10.2）。

    与 `ast_guard.normalize_program_source` 共用同一套口径：推荐写法是
    `def solve(ctx): ...`，但模型经常直接写顶层 `return ReturnAnswer(...)` ——
    那在 Python 里是 SyntaxError，会让本来正确的程序整题作废。
    """
    normalized = normalize_program_source(code)
    linecache.cache[filename] = (len(normalized), None, normalized.splitlines(True), filename)
    return compile(normalized, filename, "exec")


def _denial_receipt(kernel: Any, name: str, call_args: dict,
                    exc: ToolContractError) -> Any:
    """执行前被拒绝的调用 → §6.4 授权收据（`allowed=False` + 原因码）。

    走与 registry 同一个 `authorize_tool_call`，因此"拒绝收据"和"放行收据"出自
    同一判定函数，不会出现两套口径。
    """
    from skill3d.schemas.authorization import ToolAuthorizationReceipt

    scene = kernel._scene  # noqa: SLF001 - 同一模块内的审计构造
    spec = kernel._registry.get(name).spec  # noqa: SLF001
    profile = getattr(scene, "evidence_profile", None)
    try:
        decision = authorize_tool_call(
            name, spec, scope=str(getattr(scene, "question_tool_scope", "")),
            profile=profile,
            available_artifacts=getattr(scene, "available_artifacts", ()) or (),
            gate=getattr(scene, "metric_evidence_gate_result", None),
            supported_metric_tasks=spec.supported_metric_tasks,
            allowed_metric_tasks=getattr(scene, "allowed_metric_tasks", ()) or (),
            local_facts=getattr(scene, "local_facts", {}),
            gate_passed=getattr(scene, "metric_gate_passed", None),
            question_type=str(getattr(scene, "question_type", "") or ""),
            args=call_args, tools_enabled=kernel._tools_enabled)  # noqa: SLF001
        reason_codes = list(decision.reason_codes)
        deps = list(decision.dependency_refs)
        gate_payload = dict(decision.metric_gate_result)
    except Exception:  # noqa: BLE001 - 收据构造失败不得掩盖真正的拒绝
        reason_codes, deps, gate_payload = [], [], {}

    code = str(getattr(exc, "error_code", "") or "tool_contract")
    if code and code not in reason_codes:
        reason_codes = sorted(set(reason_codes) | {code})
    if not kernel._tools_enabled and "tools_disabled" not in reason_codes:  # noqa: SLF001
        reason_codes = sorted(set(reason_codes) | {"tools_disabled"})
    return ToolAuthorizationReceipt(
        episode_id=str(getattr(kernel, "episode_id", "") or ""),
        tool_id=name,
        argument_digest=_request_digest(scene, name, call_args),
        evidence_version=_evidence_version_of(scene),
        dependency_refs=deps, allowed=False, reason_codes=reason_codes,
        metric_gate_result=gate_payload, args=dict(call_args),
        scene_route=str(getattr(scene, "scene_route", "") or ""),
        question_tool_scope=str(getattr(scene, "question_tool_scope", "") or ""))


def _request_digest(scene: Any, name: str, call_args: dict) -> str:
    """执行前被拒绝的调用也用成功路径同一口径的请求摘要。

    与 `ToolRegistry.call_tool` 相同的构造（`request_key(args)` + 场景状态指纹），
    因此"同一个请求被拒/被放行"可以直接按 digest 对比，而不是两套 id 空间。
    """
    return hashlib.sha256(
        f"{request_key(name, call_args)}:{scene.state_digest()}".encode()
    ).hexdigest()


def _failed_result_id(request_digest: str, ordinal: int) -> str:
    """确定性 `result_id`：同参数重复拒绝也彼此可区分，同 seed 重放可复现（§4 M17）。"""
    return hashlib.sha256(f"{request_digest}:{ordinal}".encode()).hexdigest()[:16]


def _evidence_version_of(scene: Any) -> str:
    """当前 EvidenceProfile 版本；缺失记空串，不猜。"""
    profile = getattr(scene, "evidence_profile", None)
    return str(getattr(profile, "profile_version", "") or "")


@dataclass(frozen=True)
class QuestionContextView:
    """Generated-code view of question state; never exposes geometry or mutators."""

    frame: str
    scene_route: str
    question_tool_scope: str
    question_type: str
    summary: str
    available_artifacts: tuple[str, ...]


@dataclass(frozen=True)
class SolverContext:
    scene: QuestionContextView
    tools: object
    frames: tuple


class ToolNamespace:
    """No writable attributes shared across solver rounds."""

    __slots__ = ("_functions",)

    def __init__(self, functions):
        object.__setattr__(self, "_functions", functions)

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, "_functions")[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        raise TypeError("tool namespace is read-only")


def _question_context_view(scene: SceneHandle) -> QuestionContextView:
    return QuestionContextView(
        frame=str(scene.frame),
        scene_route=str(scene.scene_route),
        question_tool_scope=str(scene.question_tool_scope),
        question_type=str(scene.question_type),
        summary=str(scene.summary),
        available_artifacts=tuple(sorted(scene.available_artifacts)),
    )


class RestrictedNamespaceKernel:
    """in-process 持久 kernel：跨 cell 变量存活。"""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        scene: SceneHandle,
        frames: Optional[list] = None,
        mode: str = "real",
        cell_timeout_s: int = DEFAULT_CELL_TIMEOUT_S,
        mock_switch=None,
        # 两级兜底钩子（termination node：no-tool CoT → 正则抽取），由上层注入
        fallback_cot_fn: Optional[Callable[[], str]] = None,
        fallback_regex_fn: Optional[Callable[[str], Optional[str]]] = None,
        # tool call_id 生成器：重放确定性模式由上层注入计数器（§4 M17 同 seed 字节级一致）
        call_id_factory: Optional[Callable[[], str]] = None,
        tools_enabled: bool = True,
        episode_id: str = "",
    ) -> None:
        self._registry = tool_registry
        self._scene = scene
        self._mode = mode
        self._mock_switch = mock_switch
        self._tools_enabled = bool(tools_enabled)
        self._call_id_factory = call_id_factory or (lambda: uuid.uuid4().hex[:12])
        self.episode_id = str(episode_id or "")
        self.cell_timeout_s = cell_timeout_s
        self.answer_slot = _AnswerSlot()
        # §4：单位合同按规范题型查表（不是猜单位）
        self.answer_slot._question_type = str(  # noqa: SLF001
            getattr(scene, "question_type", "") or "")
        self.yield_slot = _YieldSlot()
        self.show_log: list[Any] = []
        self.fallback_cot_fn = fallback_cot_fn
        self.fallback_regex_fn = fallback_regex_fn
        # M10 产物：ProgramExecutionTrace 的 calls/results（§5.4）。这两个列表是
        # **episode 级账本**（跨轮只增不减，§17.1），逐轮视图见 cell_calls/cell_results。
        self.tool_calls: list[ToolCall] = []
        self.tool_results: list[ToolResult] = []
        self._cell_call_mark: int = 0
        self._cell_result_mark: int = 0
        # D-3：契约违规记录；一旦命中，ReturnAnswer 的答案不得采纳
        self.contract_violations: list[ToolResult] = []
        # v9 §6.4：每次实际调用的统一授权收据（含执行前被拒绝的调用）
        self.authorization_receipts: list[dict] = []
        self.observation_reads: list[dict] = []
        self.unresolved_result_ids: set[str] = set()
        self._cell_read_mark = 0
        self.answer_untrusted: bool = False

        # tools 命名空间：tools.<name>(**args) → 经 REGISTRY.call_tool
        self._tool_functions = {
            name: self._make_tool_fn(name) for name in tool_registry.names()
        }
        tools_ns = ToolNamespace(self._tool_functions)

        public_scene = _question_context_view(scene)
        self._ns: dict[str, Any] = {
            "__builtins__": safe_builtins(),
            "frames": frozen_frames(frames or []),
            "scene": public_scene,
            "tools": tools_ns,
            "show": self._show,
            "ReturnAnswer": lambda value: self.answer_slot(value),
            # v9 §10.1：完整答案载荷构造器（`ReturnAnswer(AnswerPayload(...))`）
            "AnswerPayload": lambda *args, **kwargs: AnswerPayload(*args, **kwargs),
            "YieldObservations": lambda result_ids=None, reason="": self.yield_slot(
                result_ids, reason),
            "ReadObservation": self._read_observation,
        }
        # 同步注入 Tool 顶层名（program 可直接写 euclidean_distance(...)）
        for name in tool_registry.names():
            self._ns[name] = self._tool_functions[name]

    def _invoke_entry(self) -> Optional[str]:
        """调用生成程序的入口 `def solve(ctx)`（v7 §10.2：**host 负责调用**）。

        实测根因（2026-09-22 真实 smoke）：prompt 推荐 `def solve(ctx): ...` 写法，
        模型照做，但只**定义**了函数、没调用它 —— 而执行器只是 `exec` 源码，
        于是函数体从未运行：零工具调用、没有答案、episode 记 `unanswerable`。
        8 题的 smoke 里 4 题（全部 MCA）栽在这里，`run_error` 都不是，纯粹是接口损失。

        v7 §10.2 把这件事定成 host 的职责（"生成程序统一采用 `def solve(ctx): ...`
        入口"），所以这里在 exec 之后主动调用一次入口；若程序自己在顶层已经调用过
        或已经提交/让出，则跳过（不重复执行）。

        返回终结类型 `"answer"`/`"yield"` 或 None。
        """
        if self.answer_slot.given or self.yield_slot.given:
            return None
        entry = self._ns.get("solve")
        if not callable(entry):
            return None
        ctx = SolverContext(
            scene=self._ns["scene"],
            tools=self._ns.get("tools"),
            frames=self._ns.get("frames", []),
        )
        try:
            params = [p for p in inspect.signature(entry).parameters.values()
                      if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                    inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            required = [p for p in params if p.default is inspect.Parameter.empty]
            if required:
                entry(ctx)
            elif params:
                entry(ctx)
            else:
                entry()                      # 模型写成 def solve(): 也接受
        except AnswerTerminate:
            return "answer"
        except YieldTerminate:
            return "yield"
        except ControlTerminate:
            return "yield"
        return None

    # ---- 注入 ----
    def inject(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in ("show", "ReturnAnswer", "YieldObservations", "AnswerPayload", "ReadObservation",
                 "tools", "scene",
                     "frames"):
                raise ValueError(f"Overwriting reserved names is forbidden: {k}")
            self._ns[k] = v

    def _read_observation(self, result_id, fields=None, offset=0, limit=8):
        from skill3d.online.observation_reader import (
            MAX_READS_PER_ROUND, read_observation, OBSERVATION_READER_VERSION,
        )
        from skill3d.online.recovery import collect_validated
        args = {"result_id": result_id, "fields": fields, "offset": offset, "limit": limit}
        receipt = {
            "name": "ReadObservation", "schema_version": OBSERVATION_READER_VERSION,
            "episode_id": self.episode_id, "args": args, "allowed": False,
        }
        try:
            if not self._tools_enabled or self.answer_slot.given or self.yield_slot.given:
                raise ValueError("ReadObservation is only available before ordinary-phase termination")
            if len(self.observation_reads) - self._cell_read_mark >= MAX_READS_PER_ROUND:
                raise ValueError("ReadObservation per-round budget exhausted")
            valid = [o for o in collect_validated(self) if o.result_id not in self.unresolved_result_ids]
            page = read_observation(valid, result_id, fields, offset, limit)
        except ValueError as exc:
            receipt["reason_code"] = "observation_read_rejected"
            self.authorization_receipts.append(receipt)
            raise ValueError(str(exc)) from exc
        receipt.update(allowed=True, reason_code="valid_episode_observation",
                       payload_hash=page["payload_hash"])
        self.authorization_receipts.append(receipt)
        # The model receives a separate copy: local edits cannot rewrite host feedback.
        self.observation_reads.append(json.loads(json.dumps(page)))
        return page

    def _record_failed_call(self, name: str, call_args: dict, exc: ToolContractError,
                            result: Optional[ToolResult] = None) -> None:
        """契约违规也要进 calls/results（§5.4 审计可回放），再抛给 run_cell 归因。

        v9 §17.1：失败的调用同样必须有可追踪 `result_id`，并且**不得**以
        `status="ok"` 混进 validated observations —— 执行前被拒绝的调用（scope、
        证据门、finalization 封锁、重复提交）若记成成功，`collect_validated`
        会把它们当作有效观测回灌给模型（§14.1）。
        """
        from skill3d.tools.failure_guidance import failure_guidance

        if result is None:
            request_digest = _request_digest(self._scene, name, call_args)
            result = ToolResult(
                tool=name, args=call_args, value="null", source=self._mode,  # type: ignore[arg-type]
                result_id=_failed_result_id(request_digest, len(self.tool_results)),
                source_tool=name,
                status="failed",
                model_response_schema_version=TOOL_RESPONSE_VERSION,
                evidence_version=_evidence_version_of(self._scene),
                request_digest=request_digest, latency_ms=0.0,
                error=f"{type(exc).__name__}: {exc}", error_code=exc.error_code,  # type: ignore[arg-type]
                failure_cause=exc.failure_cause,
                failure_guidance=failure_guidance(
                    self._registry, self._scene, name=name, cause=exc.failure_cause,
                    missing=exc.missing, tools_enabled=self._tools_enabled),
                missing_artifacts=list(exc.missing),
                available_artifacts=list(exc.available))
            self.tool_calls.append(
                ToolCall(tool=name, args=call_args, call_id=self._call_id_factory()))
            self.tool_results.append(result)
        if not result.authorization:
            # §6.4：执行前被拒绝的调用也要有同一判定函数产出的收据，且
            # `allowed=False` 必须与"确实没有执行"一致。
            result = result.model_copy(update={
                "authorization": _denial_receipt(
                    self, name, call_args, exc).model_dump()})
            self.tool_results[-1] = result
        self.authorization_receipts.append(dict(result.authorization))
        self.contract_violations.append(result)
        self.answer_untrusted = True

    def _make_tool_fn(self, name: str) -> Callable[..., Any]:
        """Tool 包装：位置/关键字传参均可（program 里 `euclidean_distance(a, b)` 与
        `euclidean_distance(point_a=a, point_b=b)` 等价）。"""
        sig = inspect.signature(self._registry.get(name).fn)

        def _fn(*args, **kwargs):
            # §15.1 运行层：控制接口已终结后再调 Tool → 受控 AnswerAlreadyGiven。
            # v7 里 ReturnAnswer/YieldObservations 都立即抛 ControlTerminate，
            # 所以走到这里说明生成程序**吞掉了**终结信号（如 except BaseException）
            # 或用了显式 return 之外的路径越过控制流；仍然硬拦，不让程序继续跑。
            if not self._tools_enabled:
                exc = ToolContractError(
                    name, f"Tool {name} cannot be called during finalization",
                    route=self._scene.scene_route, args=kwargs if kwargs else {"args": list(args)},
                    failure_cause=FailureCause(code="finalization_denied", scope="local_call"))
                self._record_failed_call(name, dict(kwargs), exc)
                raise exc
            if self.answer_slot.given or self.yield_slot.given:
                what = ("ReturnAnswer" if self.answer_slot.given else "YieldObservations")
                exc = AnswerAlreadyGiven(
                    name,
                    f"Tool {name} was called after {what}"
                    f" (submitted answer={self.answer_slot.answer!r}); "
                    "a control call ends the cell. Do not call tools afterward (§10.2).",
                    route=self._scene.scene_route,
                    args=kwargs if kwargs else {"args": list(args)},
                    failure_cause=FailureCause(code="answer_already_given", scope="local_call"))
                self._record_failed_call(name, dict(kwargs), exc)
                raise exc
            try:
                bound = sig.bind(self._scene, *args, **kwargs)
            except TypeError as error:
                exc = DomainValueError(
                    name, str(error), args=kwargs if kwargs else {"args": list(args)},
                    failure_cause=FailureCause(code="invalid_tool_arguments", scope="local_call"))
                self._record_failed_call(name, dict(kwargs), exc)
                raise exc from error
            call_args = {k: v for k, v in bound.arguments.items()
                         if k not in ("handle", "scene")}
            try:
                result = self._registry.call_tool(
                    name, call_args, self._scene, mode=self._mode,
                    mock_switch=self._mock_switch, episode_id=self.episode_id,
                )
            except ToolContractError as exc:
                # 执行期 fail-closed：产物缺失在调用实现前就抛（硬约束 23）
                self._record_failed_call(name, call_args, exc)
                raise
            # 请求摘要标识输入；结果 ID 标识本 episode 中的一次发生。
            result_id = _failed_result_id(result.request_digest, len(self.tool_results))
            authorization = dict(result.authorization or {})
            if authorization:
                authorization["result_id"] = result_id
            result = result.model_copy(update={
                "result_id": result_id, "authorization": authorization,
                "model_response_schema_version": TOOL_RESPONSE_VERSION,
            })
            # 记录到 ProgramExecutionTrace（§5.4）：调用与结果成对入库
            self.tool_calls.append(
                ToolCall(tool=name, args=call_args, call_id=self._call_id_factory())
            )
            self.tool_results.append(result)
            if result.authorization:
                self.authorization_receipts.append(dict(result.authorization))
            if result.error_code == "tool_contract":
                # D-3/硬约束 23：产物缺失等契约违规 → 确定性异常，绝不静默返回假值
                exc = ArtifactUnavailableError(
                    name, result.missing_artifacts or ["(unspecified)"],
                    route=self._scene.scene_route,
                    available=result.available_artifacts,
                    args=call_args, failure_cause=result.failure_cause)
                self._record_failed_call(name, call_args, exc, result=result)
                raise exc
            if result.error_code == "confidence_gate":
                exc = ConfidenceGateError(
                    name, result.error or "Local quality gate failed",
                    route=self._scene.scene_route,
                    available=result.available_artifacts, args=call_args,
                    failure_cause=result.failure_cause)
                self._record_failed_call(name, call_args, exc, result=result)
                raise exc
            if result.error_code == "domain_value":
                exc = DomainValueError(
                    name, result.error or "Domain value error", route=self._scene.scene_route,
                    available=result.available_artifacts, args=call_args,
                    failure_cause=result.failure_cause)
                self._record_failed_call(name, call_args, exc, result=result)
                raise exc
            if result.error is not None:
                raise RuntimeError(f"Tool {name} execution failed: {result.error}")
            return model_tool_response(result)

        return _fn

    def _show(self, obj: Any) -> None:
        """show() 视觉反馈通道：MVP 仅记录，不渲染。"""
        self.show_log.append(obj)

    # ---- 执行 ----
    def run_cell(self, code: str) -> CellResult:
        """执行一个 cell；捕获 stdout / 错误；答案与 yield 经控制接口终结（v7 §10.2）。

        `ToolContractError` 家族（tool_contract / confidence_gate / domain_value）
        由调用方归因：不静默吞掉，也不当作普通运行时错误（§4 M10，硬约束 23）。

        `AnswerTerminate` / `YieldTerminate` 是**正常终结**，不是错误：前者等价于
        "答完就返回"，后者等价于"这一片先算到这，把结果交回模型"。
        """
        buf = io.StringIO()
        before = set(self._ns.keys())
        self.mark_cell_start()
        error: Optional[str] = None
        error_code: Optional[str] = None
        terminated: Optional[str] = None
        diagnostics = {}
        program_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
        filename = f"<episode_program:{program_sha256}>"

        def capture(exc):
            frames = traceback.extract_tb(exc.__traceback__)
            own = [frame for frame in frames if frame.filename == filename]
            line = getattr(exc, "lineno", None) if isinstance(exc, SyntaxError) else (
                own[-1].lineno if own else None)
            normalized = normalize_program_source(code)
            # The top-level-return adapter adds one wrapper line to executable code.
            original_line = line - 1 if line and normalized != code else line
            lines = code.splitlines()
            diagnostics.update(
                traceback_text="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                error_type=type(exc).__name__, error_message=str(exc),
                error_line=original_line,
                error_source_line=(lines[original_line - 1]
                                   if original_line and 0 < original_line <= len(lines) else ""),
            )

        try:
            with contextlib.redirect_stdout(buf), _Timeout(self.cell_timeout_s):
                exec(_compile_program(code, filename), self._ns)
                if self.answer_slot.given:
                    terminated = "answer"
                elif self.yield_slot.given:
                    terminated = "yield"
                else:
                    terminated = self._invoke_entry()
        except AnswerTerminate:
            terminated = "answer"
        except YieldTerminate:
            terminated = "yield"
        except (_CellTimeout, TimeoutError) as exc:
            error, error_code = "Cell execution timed out", "timeout"
            capture(exc)
        except SyntaxError as exc:
            error, error_code = "Syntax error", "violation_syntax"
            capture(exc)
        except ToolContractError as exc:
            # 契约违规一律记 tool_contract 桶（§5 FailureTaxonomy 已有该桶）；
            # 具体子类（含 ConfidenceGateError / AnswerAlreadyGiven / 基类
            # ToolContractError 的 scope 违规）在 contract_violations 里逐条留痕。
            error = f"{type(exc).__name__}: {exc}"
            error_code = "tool_contract"
            self.answer_untrusted = True
            capture(exc)
        except Exception as exc:
            error, error_code = f"{type(exc).__name__}: {exc}", "violation_runtime"
            capture(exc)
        new_vars = sorted(set(self._ns.keys()) - before)
        return CellResult(
            stdout_tail=buf.getvalue()[-4096:],
            error=error,
            error_code=error_code,
            answer=self.answer_slot.answer,
            new_vars=new_vars,
            answer_untrusted=self.answer_untrusted,
            contract_violations=[r.model_dump() for r in self.contract_violations],
            terminated=terminated,
            yielded_result_ids=list(self.yield_slot.result_ids),
            yielded_reason=self.yield_slot.reason,
            stdout=buf.getvalue(), program_sha256=program_sha256,
            source_filename=filename, **diagnostics,
        )

    # ---- 状态重置（回灌重执行前必须重注入，§4 M6 字段 9）----
    def set_tools_enabled(self, enabled: bool) -> None:
        self._tools_enabled = bool(enabled)

    def retarget_scene(self, scene: SceneHandle) -> None:
        """恢复后使用更新的题级授权；保留本 episode 的工具账本。"""
        self._scene = scene
        self._ns["scene"] = _question_context_view(scene)
        self.answer_slot._question_type = str(scene.question_type or "")

    def reset_user_namespace(self) -> None:
        """清空用户命名空间并重注入保留名（保留 Tool/帧/答案槽）。

        回灌重生成后重执行前必须调用：per-episode 状态不得跨次执行泄漏
        （SpatialClaw §E.3 先例）。

        **不清空 episode 级的 host 侧账本**（`tool_calls` / `tool_results` /
        `contract_violations`）：账本属于 §17.1 的"Tool 层必须落盘"事实，不是用户
        变量。此前这里连账本一起清空，导致多轮 episode 里早先轮次的 result_id
        无法追溯（`collect_validated` / `used_result_ids` 只剩最后一轮），而
        runner 在 yield 路径上试图保留账本的写法被下一次 reset 立刻抹掉。
        逐轮视图由 `cell_calls` / `cell_results` 提供。
        """
        keep = ("frames", "scene", "tools", "show", "ReturnAnswer",
                "YieldObservations", "AnswerPayload", "ReadObservation", "__builtins__")
        preserved = {k: self._ns[k] for k in keep if k in self._ns}
        self._ns.clear()
        self._ns.update(preserved)
        for name in self._registry.names():
            self._ns[name] = self._tool_functions[name]
        self.answer_slot.answer = None
        self.answer_slot.given = False
        self.answer_slot.payload = None
        self.answer_slot.adapter_problems = []
        self.yield_slot.result_ids = []
        self.yield_slot.reason = ""
        self.yield_slot.given = False
        self.answer_untrusted = False

    def mark_cell_start(self) -> None:
        """记录本轮起点，供 `cell_calls` / `cell_results` 切出逐轮视图。"""
        self._cell_call_mark = len(self.tool_calls)
        self._cell_result_mark = len(self.tool_results)
        self._cell_read_mark = len(self.observation_reads)

    @property
    def cell_calls(self) -> list[ToolCall]:
        """本轮（当前 cell）产生的调用；episode 账本只增不减。"""
        return self.tool_calls[self._cell_call_mark:]

    @property
    def cell_results(self) -> list[ToolResult]:
        """本轮（当前 cell）产生的结果；与 `ProgramExecutionTrace` 的逐轮口径一致。"""
        return self.tool_results[self._cell_result_mark:]

    def reset_episode_ledger(self) -> None:
        """清空 episode 级账本（仅供确实要复用同一 kernel 开新 episode 的调用方）。"""
        self.tool_calls = []
        self.tool_results = []
        self.contract_violations = []
        self.authorization_receipts = []
        self.observation_reads = []
        self.unresolved_result_ids = set()
        self.mark_cell_start()

    def run_program(self, program_source: str) -> CellResult:
        return self.run_cell(program_source)

    # ---- 两级兜底（no-tool CoT → 正则抽取）----
    def run_with_fallback(self, program_source: str) -> CellResult:
        result = self.run_cell(program_source)
        if result.answer is not None:
            return result
        # 第一级：no-tool CoT 钩子
        if self.fallback_cot_fn is not None:
            cot_text = self.fallback_cot_fn()
            # 第二级：正则抽取钩子
            if self.fallback_regex_fn is not None:
                extracted = self.fallback_regex_fn(cot_text)
                if extracted is not None:
                    result.answer = extracted
        return result
