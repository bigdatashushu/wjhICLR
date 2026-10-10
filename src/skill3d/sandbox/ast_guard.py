"""AST 白名单静态检查（§4 M9 / §9.1 / v6 §15.1）。

仅允许：白名单库（numpy/scipy/math/statistics）+ REGISTRY 内 Tool 调用
+ 变量赋值/循环/print。
禁止：import os/socket/subprocess/requests、eval/exec/__import__、
文件写（.save/.to_csv/open(w)）、访问宿主路径、show/ReturnAnswer 重赋值。

**v6 新增（D7 静态层）**：`ReturnAnswer(...)` 之后再出现任何 Tool 调用 →
静态拒绝。运行层还有一道（kernel 抛 `AnswerAlreadyGiven`），两层一起保证
"答后调 Tool"不会退化成 IndexError → 假的"服务故障"。

> 背景 [已实测]：v5 模型自然写出 `if not ids: ReturnAnswer("abstain")` 然后继续
> `object_centroid(ids[0])` → IndexError → `violation_runtime`，内测方向题全栽在这里。
> AST 层拦不住的写法（例如把 Tool 存进变量再调用）由运行层兜住。

v11 将 AST 检查、受控 namespace 和 Tool 授权共同用于类沙箱执行。
"""

from __future__ import annotations

import ast
import re
from typing import Optional, Set

from skill3d.schemas import ASTCheckResult

from skill3d.tools.registry import REGISTRY
from skill3d.sandbox.safe_runtime import NUMERICAL_EXPORTS

# 白名单 import 模块（§9.1）
ALLOWED_MODULES = frozenset({"numpy", "scipy", "math", "statistics"})

# 保留名：禁止 program 重赋值（§4 M10 字段 5）
RESERVED_NAMES = frozenset(
    {"show", "ReturnAnswer", "YieldObservations", "AnswerPayload", "tools", "scene",
     "frames", "ctx"})

# 控制接口与答案合同构造器（§10.1/§10.2）：不属 Tool 面，不受证据门过滤。
# `AnswerPayload` 必须在这里 —— 否则 §10.2 的规范示例
# `ReturnAnswer(AnswerPayload(...))` 会被静态检查判为"未定义函数调用"而拒掉。
CONTROL_INTERFACE_NAMES = frozenset({"ReturnAnswer", "YieldObservations", "AnswerPayload"})

# 禁止调用的内建/危险函数名
FORBIDDEN_CALLS = frozenset(
    {
        "eval", "exec", "__import__", "compile", "open", "input",
        "getattr", "setattr", "delattr", "globals", "locals", "vars",
        "exit", "quit", "breakpoint", "help", "dir",
    }
)

# 允许的安全内建
SAFE_BUILTINS = frozenset(
    {
        "print", "len", "range", "float", "int", "str", "bool", "abs",
        "min", "max", "sum", "sorted", "list", "dict", "tuple", "set",
        "enumerate", "zip", "round", "isinstance", "format", "repr",
        "all", "any", "pow", "reversed", "next", "iter", "map", "filter", "slice",
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "RuntimeError", "ArithmeticError", "ZeroDivisionError", "AssertionError",
    }
)

# Generated code may inspect only these non-sensitive question-state fields.
# Geometry, scale, object bindings, and all mutations must go through REGISTRY Tools.
SAFE_SCENE_ATTRS = frozenset({
    "frame",
    "scene_route",
    "question_tool_scope",
    "question_type",
    "summary",
    "available_artifacts",
})

# 危险方法/属性（文件写、网络、进程、自省）
FORBIDDEN_ATTRS = frozenset(
    {
        "save", "to_csv", "to_pickle", "write", "writelines", "dump", "dumps_pickle",
        "remove", "unlink", "rmtree", "mkdir", "rename", "replace_file",
        "system", "popen", "spawnl", "spawnv", "execv", "fork",
        "connect", "request", "urlopen", "urlretrieve", "geturl",
        "loadtxt_from_url", "socket",
        "load", "loadtxt", "genfromtxt", "fromfile", "tofile", "memmap", "read",
        "read_bytes", "read_text", "savetxt", "savez", "savez_compressed", "dumps",
        "ctypes", "ctypeslib", "data", "format", "format_map", "mro",
    }
)

# 正则二次扫描（AST 之外兜底，SpatialClaw 同款机制）
_REGEX_BLACKLIST = [
    re.compile(p)
    for p in [
        r"\b__import__\b",
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"\bopen\s*\(",
        r"\bcompile\s*\(",
        r"\bos\s*\.",
        r"\bsys\s*\.",
        r"\bsocket\b",
        r"\bsubprocess\b",
        r"\brequests\b",
        r"\burllib\b",
        r"\bshutil\b",
        r"\bpathlib\b",
        r"\.save\s*\(",
        r"\.to_csv\s*\(",
        r"\.to_pickle\s*\(",
        r"/etc/", r"/proc/", r"/root/", r"~/",  # 宿主路径
    ]
]


def _attribute_path(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        base = _attribute_path(node.value)
        return (*base, node.attr) if base else ()
    return ()


def _scene_attribute(path: tuple[str, ...]) -> tuple[str, ...] | None:
    if path[:1] == ("scene",):
        return path[1:]
    if path[:2] == ("ctx", "scene"):
        return path[2:]
    return None


def _tool_attribute(path: tuple[str, ...]) -> str | None:
    if len(path) == 2 and path[0] == "tools":
        return path[1]
    if len(path) == 3 and path[:2] == ("ctx", "tools"):
        return path[2]
    return None


class _WhiteListVisitor(ast.NodeVisitor):
    def __init__(self, allowed_tools: Set[str]) -> None:
        self.allowed_tools = allowed_tools
        self.violations: list[str] = []
        self.tool_calls: list[str] = []
        self.local_defs: Set[str] = set()
        self.imported_names: Set[str] = set()
        # 块嵌套深度（仅用于把"函数体内定义"与模块顶层区分开）
        self._block_depth: int = 0

    # ---- 块深度维护（判断调用是否在分支/循环/函数体内）----
    def _visit_block(self, node: ast.AST) -> None:
        self._block_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._block_depth -= 1

    def visit_If(self, node: ast.If) -> None:
        self._visit_block(node)

    def visit_For(self, node: ast.For) -> None:
        self._check_target(node.target)
        self._visit_block(node)

    def visit_While(self, node: ast.While) -> None:
        self._visit_block(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._visit_block(node)

    # ---- import 白名单 ----
    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            root = a.name.split(".")[0]
            if a.name not in NUMERICAL_EXPORTS:
                self.violations.append(f"禁止 import 模块: {a.name}")
            self._check_target(ast.Name(id=a.asname or root))
            self.imported_names.add((a.asname or root).split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level or module not in NUMERICAL_EXPORTS:
            self.violations.append(f"禁止 from-import 模块: {node.module}")
        for a in node.names:
            if (a.name not in NUMERICAL_EXPORTS.get(module, ())
                    and f"{module}.{a.name}" not in NUMERICAL_EXPORTS):
                self.violations.append(f"禁止导入非计算接口: {module}.{a.name}")
            self._check_target(ast.Name(id=a.asname or a.name))
            self.imported_names.add(a.asname or a.name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__") and node.id != "__episode_entry__":
            self.violations.append(f"禁止访问内部名称: {node.id}")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.violations.append("生成程序不得定义类")
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.violations.append("禁止裸 except 吞掉控制终结或超时信号")
        self.generic_visit(node)

    # ---- 函数调用白名单 ----
    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
            if name in FORBIDDEN_CALLS:
                self.violations.append(f"禁止调用危险函数: {name}")
            elif name in self.allowed_tools:
                self.tool_calls.append(name)
            elif name in CONTROL_INTERFACE_NAMES:
                # v7 §10.2：控制接口是 host 实现的终结操作，运行层（kernel 抛
                # ControlTerminate）确定性保证"提交/让出后不再继续"，因此静态层
                # 不再需要"答后调 Tool"的行号顺序检查。
                # v9 §10.1：`AnswerPayload` 是答案合同构造器（纯数据），同样放行 ——
                # 否则规范自己的 `ReturnAnswer(AnswerPayload(...))` 示例会被拒。
                pass
            elif name == "show":
                pass  # 保留名回调允许调用（仅禁止重赋值）
            elif name in SAFE_BUILTINS or name in self.local_defs or name in self.imported_names:
                pass
            else:
                self.violations.append(f"REGISTRY 外未定义函数调用: {name}")
        elif isinstance(func, ast.Attribute):
            path = _attribute_path(func)
            scene_attr = _scene_attribute(path)
            tool_name = _tool_attribute(path)
            if scene_attr is not None:
                self.violations.append(
                    "禁止调用 scene/ctx.scene 接口；几何、尺度和对象数据只能经 REGISTRY Tool")
            elif func.attr.startswith("_"):
                self.violations.append(f"禁止访问下划线属性: {func.attr}")
            elif func.attr in FORBIDDEN_ATTRS:
                self.violations.append(f"禁止调用危险方法: .{func.attr}")
            elif tool_name is not None:
                if tool_name in self.allowed_tools:
                    self.tool_calls.append(tool_name)
                else:
                    self.violations.append(f"tools 命名空间内未知 Tool: {tool_name}")
        self.generic_visit(node)

    # ---- 属性访问（非调用）----
    def visit_Attribute(self, node: ast.Attribute) -> None:
        path = _attribute_path(node)
        scene_attr = _scene_attribute(path)
        if scene_attr is not None and scene_attr and (
                len(scene_attr) != 1 or scene_attr[0] not in SAFE_SCENE_ATTRS):
            self.violations.append(
                f"禁止访问 scene 属性: {'.'.join(path)}；只允许 {sorted(SAFE_SCENE_ATTRS)}")
        elif node.attr.startswith("_"):
            self.violations.append(f"禁止访问下划线属性: {node.attr}")
        elif node.attr in FORBIDDEN_ATTRS:
            self.violations.append(f"禁止访问危险属性: {node.attr}")
        self.generic_visit(node)

    # ---- 保留名重赋值 ----
    def _check_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name) and (
                target.id in RESERVED_NAMES or target.id in self.allowed_tools):
            self.violations.append(f"禁止重赋值保留名: {target.id}")
        elif isinstance(target, ast.Attribute) and \
                _scene_attribute(_attribute_path(target)) is not None:
            self.violations.append("禁止修改 scene/ctx.scene 只读题级状态")
        elif isinstance(target, ast.Attribute) and (
                _attribute_path(target)[:1]
                and _attribute_path(target)[0] in RESERVED_NAMES):
            self.violations.append("禁止修改框架接口或上下文")
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._check_target(elt)

    def visit_Assign(self, node: ast.Assign) -> None:
        for t in node.targets:
            self._check_target(t)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_target(node.target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_target(node.target)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name in RESERVED_NAMES:
            self.violations.append(f"禁止定义与保留名同名的函数: {node.name}")
        self.local_defs.add(node.name)
        self._visit_block(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._check_target(node.target)
        self.generic_visit(node)


class _TopLevelReturnFinder(ast.NodeVisitor):
    """在**函数体之外**寻找 `return`（不进入函数/lambda 体）。

    注意：`ast.parse` 在 Python 3.11 上**接受**模块顶层的 `return`
    （"return outside function" 是 `compile()` 阶段才报的 SyntaxError），
    所以不能靠 `except SyntaxError` 判断，必须走 AST。
    """

    def __init__(self) -> None:
        self.found = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:      # 不深入函数体
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Return(self, node: ast.Return) -> None:
        self.found = True


def normalize_program_source(code: str) -> str:
    """把**顶层 `return`** 包进函数体，使两种推荐写法都能通过（v7 §10.2）。

    v7 推荐 `def solve(ctx): ... return ReturnAnswer(...)`，但实测模型也经常直接
    写**顶层** `return ReturnAnswer(...)`。那在 Python 里无法 `compile`
    （"return outside function"），会让一个本来正确的程序整题作废 —— 纯属接口损失。

    仅当模块顶层（函数体之外）真的出现 `return` 时才包装；包装后的函数体与模块体
    局部作用域一致，对程序语义无影响（生成的程序不依赖跨 cell 的全局重绑定）。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    finder = _TopLevelReturnFinder()
    for node in tree.body:
        finder.visit(node)
    if not finder.found:
        return code
    indented = "\n".join(("    " + ln) if ln.strip() else ln
                         for ln in code.splitlines())
    return f"def __episode_entry__():\n{indented}\n\n__episode_entry__()\n"


def ast_guard(
    program: str,
    allowed_tools: Optional[Set[str]] = None,
) -> ASTCheckResult:
    """AST 白名单 + 正则二次扫描，返回 ASTCheckResult。"""
    tools = allowed_tools if allowed_tools is not None else set(REGISTRY.names())

    # 正则二次扫描（在**原文**上做：包装只是语法适配，不改变文本里出现的调用）
    regex_hits = [p.pattern for p in _REGEX_BLACKLIST if p.search(program)]

    program = normalize_program_source(program)
    try:
        tree = ast.parse(program)
    except SyntaxError as exc:
        return ASTCheckResult(ok=False, violations=[f"语法错误: {exc}"], allowed_tool_calls=[])

    # 先收集全部 local def，再正式检查（允许先定义后调用之外的任意顺序）
    collector = _WhiteListVisitor(tools)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            collector.local_defs.add(node.name)
    collector.visit(tree)

    violations = collector.violations + [f"正则二次扫描命中: {p}" for p in regex_hits]
    return ASTCheckResult(
        ok=not violations,
        violations=violations,
        allowed_tool_calls=collector.tool_calls,
    )
