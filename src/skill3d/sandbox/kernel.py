"""持久 kernel（§4 M10）：code-as-action，跨 cell 变量存活。

MVP 为 in-process RestrictedNamespaceKernel（exec 于受控 namespace）；
Docker 隔离由 docker_manager 负责（AST 检查不能替代容器隔离，§4 M9 字段 12）。

注入保留名：frames / scene / tools / show / ReturnAnswer。
cell 超时与错误捕获；两级兜底（no-tool CoT → 正则抽取）留钩子。
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import signal
import types
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from skill3d.schemas import AnswerPayload, ToolCall, ToolResult, parse_answer_payload
from skill3d.tools.contract import (
    AnswerAlreadyGiven,
    ArtifactUnavailableError,
    ConfidenceGateError,
    DomainValueError,
    ToolContractError,
)
from skill3d.sandbox.ast_guard import normalize_program_source
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
    """`ReturnAnswer(...)` 立即结束本轮与 episode（v7 §10.2）。"""
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
                f"本片段已提交答案（{self.answer!r}）；控制接口一经调用即结束本片段，"
                "不得再次提交（§10.2）",
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


class _Timeout:
    """SIGALRM cell 超时（仅主线程可用；非主线程静默退化为无超时）。"""

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self._active = False

    def __enter__(self):
        try:
            signal.signal(signal.SIGALRM, self._raise)
            signal.alarm(self.seconds)
            self._active = True
        except (ValueError, AttributeError):
            self._active = False  # 非主线程或无 SIGALRM 平台
        return self

    def __exit__(self, *exc):
        if self._active:
            signal.alarm(0)
        return False

    @staticmethod
    def _raise(signum, frame):
        raise TimeoutError("cell 执行超时")


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


def _compile_program(code: str):
    """编译生成程序；对**顶层 `return`** 做函数包装（v7 §10.2）。

    与 `ast_guard.normalize_program_source` 共用同一套口径：推荐写法是
    `def solve(ctx): ...`，但模型经常直接写顶层 `return ReturnAnswer(...)` ——
    那在 Python 里是 SyntaxError，会让本来正确的程序整题作废。
    """
    return compile(normalize_program_source(code), "<episode_program>", "exec")


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
        self.answer_untrusted: bool = False

        # tools 命名空间：tools.<name>(**args) → 经 REGISTRY.call_tool
        tools_ns = types.SimpleNamespace()
        for name in tool_registry.names():
            tools_ns.__dict__[name] = self._make_tool_fn(name)

        self._ns: dict[str, Any] = {
            "frames": frames or [],
            "scene": scene,
            "tools": tools_ns,
            "show": self._show,
            "ReturnAnswer": self.answer_slot,
            # v9 §10.1：完整答案载荷构造器（`ReturnAnswer(AnswerPayload(...))`）
            "AnswerPayload": AnswerPayload,
            "YieldObservations": self.yield_slot,
        }
        # 同步注入 Tool 顶层名（program 可直接写 euclidean_distance(...)）
        for name in tool_registry.names():
            self._ns[name] = tools_ns.__dict__[name]

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
        ctx = types.SimpleNamespace(
            scene=self._scene,
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
            if k in ("show", "ReturnAnswer", "YieldObservations", "AnswerPayload",
                 "tools", "scene",
                     "frames"):
                raise ValueError(f"禁止覆盖保留名: {k}")
            self._ns[k] = v

    def _record_failed_call(self, name: str, call_args: dict, exc: ToolContractError,
                            result: Optional[ToolResult] = None) -> None:
        """契约违规也要进 calls/results（§5.4 审计可回放），再抛给 run_cell 归因。

        v9 §17.1：失败的调用同样必须有可追踪 `result_id`，并且**不得**以
        `status="ok"` 混进 validated observations —— 执行前被拒绝的调用（scope、
        证据门、finalization 封锁、重复提交）若记成成功，`collect_validated`
        会把它们当作有效观测回灌给模型（§14.1）。
        """
        if result is None:
            request_digest = _request_digest(self._scene, name, call_args)
            result = ToolResult(
                tool=name, args=call_args, value="null", source=self._mode,  # type: ignore[arg-type]
                result_id=_failed_result_id(request_digest, len(self.tool_results)),
                source_tool=name,
                status="failed",
                evidence_version=_evidence_version_of(self._scene),
                request_digest=request_digest, latency_ms=0.0,
                error=f"{type(exc).__name__}: {exc}", error_code=exc.error_code,  # type: ignore[arg-type]
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
                    name, f"Tool {name} 在 finalization 阶段不可调用",
                    route=self._scene.scene_route, args=kwargs if kwargs else {"args": list(args)})
                self._record_failed_call(name, dict(kwargs), exc)
                raise exc
            if self.answer_slot.given or self.yield_slot.given:
                what = ("ReturnAnswer" if self.answer_slot.given else "YieldObservations")
                exc = AnswerAlreadyGiven(
                    name,
                    f"Tool {name} 在 {what} 之后被调用"
                    f"（已给出的答案={self.answer_slot.answer!r}）；"
                    "控制接口一经调用即结束本片段，之后不得再调用工具（§10.2）",
                    route=self._scene.scene_route,
                    args=kwargs if kwargs else {"args": list(args)})
                self._record_failed_call(name, dict(kwargs), exc)
                raise exc
            bound = sig.bind(self._scene, *args, **kwargs)
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
            # 记录到 ProgramExecutionTrace（§5.4）：调用与结果成对入库
            self.tool_calls.append(
                ToolCall(tool=name, args=call_args, call_id=self._call_id_factory())
            )
            self.tool_results.append(result)
            if result.authorization:
                self.authorization_receipts.append(dict(result.authorization))
            if result.error_code == "tool_contract":
                # D-3/硬约束 23：产物缺失等契约违规 → 确定性异常，绝不静默返回假值
                self._record_failed_call(name, call_args, ArtifactUnavailableError(
                    name, result.missing_artifacts or ["(未标注)"],
                    route=self._scene.scene_route,
                    available=result.available_artifacts,
                    args=call_args), result=result)
                raise ArtifactUnavailableError(
                    name, result.missing_artifacts or ["(未标注)"],
                    route=self._scene.scene_route,
                    available=result.available_artifacts,
                    args=call_args)
            if result.error_code == "confidence_gate":
                exc = ConfidenceGateError(
                    name, result.error or "局部质量门未过",
                    route=self._scene.scene_route,
                    available=result.available_artifacts, args=call_args)
                self._record_failed_call(name, call_args, exc, result=result)
                raise exc
            if result.error_code == "domain_value":
                exc = DomainValueError(
                    name, result.error or "域值错误", route=self._scene.scene_route,
                    available=result.available_artifacts, args=call_args)
                self._record_failed_call(name, call_args, exc, result=result)
                raise exc
            if result.error is not None:
                raise RuntimeError(f"Tool {name} 执行失败: {result.error}")
            import json

            payload = json.loads(result.value)
            # v7 §10.2：把本结果的 `result_id` 一并交给程序。`YieldObservations`
            # 的参数就是 result_id 列表，而 id 由 host 生成 —— 程序拿不到它就无法
            # 点名要回灌哪条结果。dict 载荷额外挂一个键（不改动原有键），
            # 列表/标量载荷则保持原样（可用空列表让出，语义是"回灌本轮全部结果"）。
            if isinstance(payload, dict) and result.result_id:
                payload.setdefault("result_id", result.result_id)
            return payload

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
        try:
            with contextlib.redirect_stdout(buf), _Timeout(self.cell_timeout_s):
                exec(_compile_program(code), self._ns)
        except AnswerTerminate:
            terminated = "answer"
        except YieldTerminate:
            terminated = "yield"
        except TimeoutError:
            error, error_code = "cell 执行超时", "timeout"
        except SyntaxError:
            error, error_code = "语法错误", "violation_syntax"
        except ToolContractError as exc:
            # 契约违规一律记 tool_contract 桶（§5 FailureTaxonomy 已有该桶）；
            # 具体子类（含 ConfidenceGateError / AnswerAlreadyGiven / 基类
            # ToolContractError 的 scope 违规）在 contract_violations 里逐条留痕。
            error = f"{type(exc).__name__}: {exc}"
            error_code = "tool_contract"
            self.answer_untrusted = True
        except Exception as exc:
            error, error_code = f"{type(exc).__name__}: {exc}", "violation_runtime"
        else:
            # 未抛终结信号但槽已写入（生成程序吞掉了 BaseException）→ 仍按已终结处理，
            # 保证"提交过的答案不会丢"，也保证不会重复提交第二次。
            if self.answer_slot.given:
                terminated = "answer"
            elif self.yield_slot.given:
                terminated = "yield"
            else:
                # v7 §10.2：host 负责调用 `def solve(ctx)` 入口（模型常只定义不调用）
                try:
                    terminated = self._invoke_entry()
                except ToolContractError as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    error_code = "tool_contract"
                    self.answer_untrusted = True
                except Exception as exc:      # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                    error_code = "violation_runtime"
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
        )

    # ---- 状态重置（回灌重执行前必须重注入，§4 M6 字段 9）----
    def set_tools_enabled(self, enabled: bool) -> None:
        self._tools_enabled = bool(enabled)

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
        tools_ns = self._ns["tools"]
        keep = ("frames", "scene", "tools", "show", "ReturnAnswer",
                "YieldObservations", "AnswerPayload")
        preserved = {k: self._ns[k] for k in keep if k in self._ns}
        self._ns.clear()
        self._ns.update(preserved)
        for name in self._registry.names():
            self._ns[name] = tools_ns.__dict__[name]
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
