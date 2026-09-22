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

AST 检查不能替代容器隔离（L2/L3/L4 由 M10 docker 承担）。
"""

from __future__ import annotations

import ast
import re
from typing import Optional, Set

from skill3d.schemas import ASTCheckResult

from skill3d.tools.registry import REGISTRY

# 白名单 import 模块（§9.1）
ALLOWED_MODULES = frozenset({"numpy", "scipy", "math", "statistics"})

# 保留名：禁止 program 重赋值（§4 M10 字段 5）
RESERVED_NAMES = frozenset({"show", "ReturnAnswer", "tools", "scene", "frames"})

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
    }
)

# 危险方法/属性（文件写、网络、进程、自省）
FORBIDDEN_ATTRS = frozenset(
    {
        "save", "to_csv", "to_pickle", "write", "writelines", "dump", "dumps_pickle",
        "remove", "unlink", "rmtree", "mkdir", "rename", "replace_file",
        "system", "popen", "spawnl", "spawnv", "execv", "fork",
        "connect", "request", "urlopen", "urlretrieve", "geturl",
        "loadtxt_from_url", "socket",
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


class _WhiteListVisitor(ast.NodeVisitor):
    def __init__(self, allowed_tools: Set[str]) -> None:
        self.allowed_tools = allowed_tools
        self.violations: list[str] = []
        self.tool_calls: list[str] = []
        self.local_defs: Set[str] = set()
        self.imported_names: Set[str] = set()
        # §15.1 静态层：**顶层** ReturnAnswer 之后不得再出现 Tool 调用
        self._answer_given_at: Optional[int] = None
        # 块嵌套深度：ReturnAnswer 在分支/循环里给出**不算**"答后再算"
        self._block_depth: int = 0

    def _check_answer_ordering(self, node: ast.AST, tool_name: str) -> None:
        """`ReturnAnswer` 之后再调 Tool → 违规（**仅顶层同层判定**）。

        2026-09-21 真实实测（32 题 inner_validation，arm3）：原先按"行号先后"
        无条件判定，把模型最自然的**防御式写法**整片拒掉：

        ```python
        objs = list_objects('telephone')
        if not objs:
            ReturnAnswer("abstain")     # 在 if 分支里
        phone_id = objs[0]['obj_id']
        objs = list_objects('trash can')  # 被判成"答后调 Tool"
        ```

        实测后果：16/32 episode 在 M9 被拒 → 三次重生成都写同一风格 → FSM 转
        `unanswerable` → 4 个方向题、计数题等**全部零分**，而这正是 D7 要修的
        "内测方向题全栽在这里"。把 §15.1 的原意（防止 ReturnAnswer 之后继续跑
        Tool 导致 IndexError 崩成假服务故障）实现成"无条件按行号拒绝"是过度收窄。

        现在的判据（保守但不过度）：
        - 只有 **模块顶层同一层**（`_block_depth == 0`）先 `ReturnAnswer` 后调
          Tool 才报违规 —— 那才是"答完还继续算"的真反例；
        - ReturnAnswer 在 `if`/`for`/`while`/`try`/函数体内时，是否真的"答后调
          Tool"由**运行层** `AnswerAlreadyGiven` 精确判定（§15.1 双层设计的本意）。
        """
        if self._answer_given_at is None or self._block_depth > 0:
            return
        line = getattr(node, "lineno", 0)
        if line and line > self._answer_given_at:
            self.violations.append(
                f"禁止在 ReturnAnswer 之后再调用 Tool: {tool_name}"
                f"（第 {line} 行 > 顶层 ReturnAnswer 第 {self._answer_given_at} 行；"
                "§15.1 静态层。若本意是「某个前提不成立才 abstain」，"
                "请把 ReturnAnswer 放进 if 分支里）")

    # ---- 块深度维护（判断 ReturnAnswer 是否在分支/循环/函数体内）----
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
            if root not in ALLOWED_MODULES:
                self.violations.append(f"禁止 import 模块: {a.name}")
            self.imported_names.add((a.asname or root).split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root not in ALLOWED_MODULES:
            self.violations.append(f"禁止 from-import 模块: {node.module}")
        for a in node.names:
            self.imported_names.add(a.asname or a.name)
        self.generic_visit(node)

    # ---- 函数调用白名单 ----
    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
            if name in FORBIDDEN_CALLS:
                self.violations.append(f"禁止调用危险函数: {name}")
            elif name in self.allowed_tools:
                self._check_answer_ordering(node, name)
                self.tool_calls.append(name)
            elif name == "ReturnAnswer":
                # 记录**顶层**"答案已给"的行号；分支里的 ReturnAnswer 不参与静态判定
                # （§15.1：保留记录/反作弊语义，不做中止语义）
                if self._answer_given_at is None and self._block_depth == 0:
                    self._answer_given_at = getattr(node, "lineno", 0)
            elif name == "show":
                pass  # 保留名回调允许调用（仅禁止重赋值）
            elif name in SAFE_BUILTINS or name in self.local_defs or name in self.imported_names:
                pass
            else:
                self.violations.append(f"REGISTRY 外未定义函数调用: {name}")
        elif isinstance(func, ast.Attribute):
            if func.attr.startswith("_"):
                self.violations.append(f"禁止访问下划线属性: {func.attr}")
            elif func.attr in FORBIDDEN_ATTRS:
                self.violations.append(f"禁止调用危险方法: .{func.attr}")
            elif isinstance(func.value, ast.Name) and func.value.id == "tools":
                if func.attr in self.allowed_tools:
                    self._check_answer_ordering(node, func.attr)
                    self.tool_calls.append(func.attr)
                else:
                    self.violations.append(f"tools 命名空间内未知 Tool: {func.attr}")
        self.generic_visit(node)

    # ---- 属性访问（非调用）----
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            self.violations.append(f"禁止访问下划线属性: {node.attr}")
        self.generic_visit(node)

    # ---- 保留名重赋值 ----
    def _check_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name) and target.id in RESERVED_NAMES:
            self.violations.append(f"禁止重赋值保留名: {target.id}")
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


def ast_guard(
    program: str,
    allowed_tools: Optional[Set[str]] = None,
) -> ASTCheckResult:
    """AST 白名单 + 正则二次扫描，返回 ASTCheckResult。"""
    tools = allowed_tools if allowed_tools is not None else set(REGISTRY.names())

    # 正则二次扫描
    regex_hits = [p.pattern for p in _REGEX_BLACKLIST if p.search(program)]

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
