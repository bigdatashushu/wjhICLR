"""VSI copy of the shared whitelist; adds only reserved host interface names.

Numerical exports, forbidden calls/attributes, source normalization, regular
expression scan and tool accounting retain the shared implementation's rules.
"""

from __future__ import annotations

import ast
import re
from typing import Optional, Set

from skill3d.schemas import ASTCheckResult
from skill3d.tools.registry import REGISTRY
from skill3d.sandbox.safe_runtime import NUMERICAL_EXPORTS
from skill3d.online.vsi_execution_contract import VSI_INTERFACE_NAMES

RESERVED_NAMES = frozenset(
    {"show", "ReturnAnswer", "YieldObservations", "AnswerPayload", "ReadObservation", "tools", "scene",
     "frames", "ctx"}) | VSI_INTERFACE_NAMES
CONTROL_INTERFACE_NAMES = frozenset(
    {"ReturnAnswer", "YieldObservations", "AnswerPayload", "ReadObservation"}) | VSI_INTERFACE_NAMES

FORBIDDEN_CALLS = frozenset(
    {
        "eval", "exec", "__import__", "compile", "open", "input",
        "getattr", "setattr", "delattr", "globals", "locals", "vars",
        "exit", "quit", "breakpoint", "help", "dir",
    }
)
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
SAFE_SCENE_ATTRS = frozenset({
    "frame", "scene_route", "question_tool_scope", "question_type", "summary", "available_artifacts",
})
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
_REGEX_BLACKLIST = [
    re.compile(p)
    for p in [
        r"\b__import__\b", r"\beval\s*\(", r"\bexec\s*\(", r"\bopen\s*\(",
        r"\bcompile\s*\(", r"\bos\s*\.", r"\bsys\s*\.", r"\bsocket\b",
        r"\bsubprocess\b", r"\brequests\b", r"\burllib\b", r"\bshutil\b", r"\bpathlib\b",
        r"\.save\s*\(", r"\.to_csv\s*\(", r"\.to_pickle\s*\(",
        r"/etc/", r"/proc/", r"/root/", r"~/",
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
        self._block_depth: int = 0

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

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            root = a.name.split(".")[0]
            if a.name not in NUMERICAL_EXPORTS:
                self.violations.append(f"Module import is forbidden: {a.name}")
            self._check_target(ast.Name(id=a.asname or root))
            self.imported_names.add((a.asname or root).split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level or module not in NUMERICAL_EXPORTS:
            self.violations.append(f"Module from-import is forbidden: {node.module}")
        for a in node.names:
            if (a.name not in NUMERICAL_EXPORTS.get(module, ())
                    and f"{module}.{a.name}" not in NUMERICAL_EXPORTS):
                self.violations.append(f"Importing non-numerical interfaces is forbidden: {module}.{a.name}")
            self._check_target(ast.Name(id=a.asname or a.name))
            self.imported_names.add(a.asname or a.name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__") and node.id != "__episode_entry__":
            self.violations.append(f"Access to internal names is forbidden: {node.id}")
        if isinstance(node.ctx, ast.Store) and node.id in VSI_INTERFACE_NAMES:
            self.violations.append(f"Reassigning reserved names is forbidden: {node.id}")

    def visit_arg(self, node: ast.arg) -> None:
        if node.arg in VSI_INTERFACE_NAMES:
            self.violations.append(f"Shadowing reserved VSI interfaces is forbidden: {node.arg}")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.violations.append("Generated programs must not define classes")
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.violations.append("Bare except must not swallow control termination or timeout signals")
        if node.name in VSI_INTERFACE_NAMES:
            self.violations.append(f"Shadowing reserved VSI interfaces is forbidden: {node.name}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
            if name in FORBIDDEN_CALLS:
                self.violations.append(f"Calling unsafe functions is forbidden: {name}")
            elif name in self.allowed_tools:
                self.tool_calls.append(name)
            elif name in CONTROL_INTERFACE_NAMES:
                pass
            elif name == "show":
                pass
            elif name in SAFE_BUILTINS or name in self.local_defs or name in self.imported_names:
                pass
            else:
                self.violations.append(f"Undefined function call outside REGISTRY: {name}")
        elif isinstance(func, ast.Attribute):
            path = _attribute_path(func)
            scene_attr = _scene_attribute(path)
            tool_name = _tool_attribute(path)
            if scene_attr is not None:
                self.violations.append(
                    "Calling scene/ctx.scene interfaces is forbidden; access geometry, scale, and object data only through REGISTRY tools")
            elif func.attr.startswith("_"):
                self.violations.append(f"Access to underscore attributes is forbidden: {func.attr}")
            elif func.attr in FORBIDDEN_ATTRS:
                self.violations.append(f"Calling unsafe methods is forbidden: .{func.attr}")
            elif tool_name is not None:
                if tool_name in self.allowed_tools:
                    self.tool_calls.append(tool_name)
                else:
                    self.violations.append(f"Unknown tool in the tools namespace: {tool_name}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        path = _attribute_path(node)
        scene_attr = _scene_attribute(path)
        if scene_attr is not None and scene_attr and (
                len(scene_attr) != 1 or scene_attr[0] not in SAFE_SCENE_ATTRS):
            self.violations.append(
                f"Access to scene attribute is forbidden: {'.'.join(path)}; only {sorted(SAFE_SCENE_ATTRS)} are allowed")
        elif node.attr.startswith("_"):
            self.violations.append(f"Access to underscore attributes is forbidden: {node.attr}")
        elif node.attr in FORBIDDEN_ATTRS:
            self.violations.append(f"Access to unsafe attributes is forbidden: {node.attr}")
        self.generic_visit(node)

    def _check_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name) and (
                target.id in RESERVED_NAMES or target.id in self.allowed_tools):
            self.violations.append(f"Reassigning reserved names is forbidden: {target.id}")
        elif isinstance(target, ast.Attribute) and \
                _scene_attribute(_attribute_path(target)) is not None:
            self.violations.append("Modifying read-only question state in scene/ctx.scene is forbidden")
        elif isinstance(target, ast.Attribute) and (
                _attribute_path(target)[:1]
                and _attribute_path(target)[0] in RESERVED_NAMES):
            self.violations.append("Modifying framework interfaces or context is forbidden")
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
            self.violations.append(f"Defining a function with a reserved name is forbidden: {node.name}")
        self.local_defs.add(node.name)
        self._visit_block(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._check_target(node.target)
        self.generic_visit(node)


class _TopLevelReturnFinder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.found = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Return(self, node: ast.Return) -> None:
        self.found = True


def normalize_program_source(code: str) -> str:
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


def ast_guard(program: str, allowed_tools: Optional[Set[str]] = None) -> ASTCheckResult:
    tools = allowed_tools if allowed_tools is not None else set(REGISTRY.names())
    regex_hits = [p.pattern for p in _REGEX_BLACKLIST if p.search(program)]
    program = normalize_program_source(program)
    try:
        tree = ast.parse(program)
    except SyntaxError as exc:
        return ASTCheckResult(ok=False, violations=[f"Syntax error: {exc}"], allowed_tool_calls=[])
    collector = _WhiteListVisitor(tools)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            collector.local_defs.add(node.name)
    collector.visit(tree)
    violations = collector.violations + [f"Secondary regex scan matched: {p}" for p in regex_hits]
    return ASTCheckResult(ok=not violations, violations=violations,
                          allowed_tool_calls=collector.tool_calls)
