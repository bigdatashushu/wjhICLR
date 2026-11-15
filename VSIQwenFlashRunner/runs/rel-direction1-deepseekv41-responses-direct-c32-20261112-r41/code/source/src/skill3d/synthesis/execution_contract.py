"""Shared, immutable method/execution boundary and conservative contract review."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import re

from skill3d.schemas.answer import CANONICAL_UNIT_BY_QUESTION_TYPE, derivation_contract
from skill3d.schemas.tool_response import TOOL_RESPONSE_VERSION, TOOL_RESPONSE_FIELDS

CONTRACT_VERSION = "execution-contract/5.1"

_RULES = (
    "Ordinary requests output exactly one Python fenced code block and no outside explanation.",
    "ReturnAnswer(AnswerPayload(...)) submits; YieldObservations(result_ids, reason) requests observation feedback.",
    "Finalization outputs one strict JSON object: action=answer, value, basis, used_result_ids, derivation; "
    "optional unit must match the task. No code, evidence tools or YieldObservations. "
    "The host constructs AnswerPayload and submits through the same M11 checks within the original request budget.",
    "AnswerPayload fields are value, unit, basis, used_result_ids, derivation. "
    "basis is visual_estimate, mixed, or tool_derived. tool_derived requires valid result IDs and replayable derivation.",
    "All input episodes start with the frozen 32 original frames. Additional tools are optional; zero calls do not mean no images.",
    "Each request contains only its declared image pack; a previous request is not persistent visual memory.",
    "User variables are cleared between rounds. print is diagnostic output, not an observation action.",
    "An ordinary program with evidence calls but no control can return successful current observations "
    "through the host observation-fallthrough adapter. It never infers or submits an answer, never "
    "uses historical results to qualify, and cannot return caught failures as successful observations.",
    "Generating a crop is not observing it: YieldObservations must deliver pixels in a completed subsequent request.",
    "selected_object_count and resolved_tool_object_count are read-only telemetry, never answer fields or a required nonzero selection.",
    "Candidate records and resolved tool IDs do not establish the intended referent or scene completeness.",
    "With readable images give the best legal answer; uncertainty permits a visual estimate, not abstain. Input failures remain in the denominator.",
    "A Skill specifies a method only; it cannot grant tools, increase budget, change scoring, or bypass M11.",
    "Failed, unresolved, or invalidated evidence cannot support a submission. Changing basis alone does not repair evidence.",
    "Caught tool failures reject submissions in ordinary and finalization phases alike. Predictions are "
    "saved for audit, then cleared. Rejected prediction programs are not reinserted into recovery requests. "
    "Independent valid evidence may support a subsequent submission.",
    "Relative-distance geometry must use a complete relative_distance_rank covering all current options; "
    "missing or unmeasurable options remain unresolved, never infinity or absent.",
    "Scene geometry does not imply valid object binding or collection coverage. Normalized coordinates do not supply metric scale.",
    "Geometry results depend on every object actually consumed. A connectivity graph requires usable "
    "bindings for every object node. Optional metric coordinates also depend on current metric scale; "
    "withdrawing that scale invalidates the old result, while a new normalized-only call remains legal.",
)


@dataclass(frozen=True)
class ExecutionContract:
    method_context_max_chars: int
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> dict:
        from skill3d.online.observation_reader import reader_contract
        from skill3d.sandbox.action_check import OBSERVATION_FALLTHROUGH_VERSION
        from skill3d.online.progress import PROGRESS_VERSION, NO_PROGRESS_LIMIT
        from skill3d.schemas.image_ledger import LAYOUT_DERIVED_PLUS_ORIGINALS
        from skill3d.online.submission import (
            FINALIZATION_SCHEMA_VERSION, FINALIZATION_DELIVERY_STRATEGY,
        )
        if self.method_context_max_chars <= 0:
            raise ValueError("method_context_max_chars must be positive")
        return {
            **asdict(self),
            "rules": list(_RULES),
            "answer_units": dict(CANONICAL_UNIT_BY_QUESTION_TYPE),
            "derivation_contract": derivation_contract(),
            "finalization_schema_version": FINALIZATION_SCHEMA_VERSION,
            "finalization_method_delivery": FINALIZATION_DELIVERY_STRATEGY,
            "tool_response_schema_version": TOOL_RESPONSE_VERSION,
            "observation_fallthrough_version": OBSERVATION_FALLTHROUGH_VERSION,
            "host_observation_reader": reader_contract(),
            "progress_policy": {"version": PROGRESS_VERSION, "no_progress_limit": NO_PROGRESS_LIMIT,
                                "status": "frozen_pilot_policy"},
            "image_pack_version": LAYOUT_DERIVED_PLUS_ORIGINALS,
            "tool_response": (
                "Every tool returns {result_id, source_tool, status, data, schema_version, evidence_version}. "
                "Read the unchanged list/dict/scalar via result['data']; use result['result_id'] for references. "
                "The outer status=ok means call success only; inspect inner data status and coverage. "
                "Errors raise, never fabricate successful evidence. M11 derivation fields address the raw data "
                "root, without a data prefix. Old trace storage remains raw."),
            "length_measurement": "len(normalized complete SKILL.md), including frontmatter; LF and one final newline",
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def render(self) -> str:
        return (f"Execution contract {self.contract_version}; sha256={self.sha256}\n"
                + json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True))


@dataclass(frozen=True)
class ContractIssue:
    code: str
    line: int
    rule: str
    severity: str = "error"

    def to_dict(self) -> dict:
        # No candidate excerpts: safe even if the source contains sample labels.
        return asdict(self)


def _response_shape_issues(tree: ast.AST, registry, offset: int) -> list[ContractIssue]:
    """Follow unambiguous bindings only; never infer types after reassignment."""
    writes: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            writes[node.id] = writes.get(node.id, 0) + 1
        elif isinstance(node, ast.arg):
            writes[node.arg] = writes.get(node.arg, 0) + 1
    bindings = {
        node.targets[0].id: node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name) and writes.get(node.targets[0].id) == 1
    }

    def shape(node, visited=frozenset()):
        if isinstance(node, ast.Name) and node.id in bindings and node.id not in visited:
            value = bindings[node.id]
            if value.lineno < node.lineno:
                return shape(value, visited | {node.id})
        if isinstance(node, ast.Call):
            name = (node.func.id if isinstance(node.func, ast.Name) else
                    node.func.attr if isinstance(node.func, ast.Attribute)
                    and ast.unparse(node.func.value) in {"tools", "ctx.tools"} else "")
            if name in registry:
                return ("envelope", str(inspect.signature(registry.get(name).fn).return_annotation))
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            parent = shape(node.value, visited)
            if parent and parent[0] == "envelope" and node.slice.value == "data":
                return ("data", parent[1])
        return None

    issues = []
    for node in ast.walk(tree):
        parent, key = None, None
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            parent, key = shape(node.value), node.slice.value
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "get" and node.args and isinstance(node.args[0], ast.Constant)):
            parent, key = shape(node.func.value), node.args[0].value
        if not parent:
            continue
        reason = ""
        if parent[0] == "envelope" and key not in TOOL_RESPONSE_FIELDS:
            reason = "tool_data_requires_envelope_data"
        elif parent[0] == "data" and parent[1].startswith(("list", "float", "bool", "int")):
            if isinstance(key, str):
                reason = "non_dict_data_string_index"
        if reason:
            issues.append(ContractIssue("return_schema_mismatch", offset + node.lineno, reason))
    return issues


def review_method_contract(skill_md: str, *, allowed_tools: set[str] | None = None) -> list[ContractIssue]:
    """Reject concrete contradictions; ambiguous prose is a review item.

    This does not claim to prove arbitrary natural-language methods correct.
    Code fragments are parsed but never executed against live tools.
    """
    from skill3d.tools import REGISTRY

    issues: list[ContractIssue] = []
    all_tools = set(REGISTRY.names())
    for number, line in enumerate(skill_md.splitlines(), 1):
        plain = line.lower().replace("`", "")
        # Explicit negative claims remain legal, e.g. "do not write telemetry".
        telemetry = re.search(r"\b(selected_object_count|resolved_tool_object_count)\b", plain)
        if telemetry and (
            re.search(r"(?:must|ensure|set|populate).*(?:final integer answer|final.*count|> ?0|non.?zero|verified identities)", plain)
            or re.search(r"(?:selected_object_count|resolved_tool_object_count).*(?:must be|must reflect)", plain)
        ) and not re.search(r"(?:never|do not|must not).*(?:set|write|populate|require)", plain):
            issues.append(ContractIssue("telemetry_as_api", number, "telemetry_is_not_submission"))
        if ("code block" in plain and re.search(r"(?:without|do not|never|no)\b.*(?:markdown|python|code block)", plain)
                and not re.search(r"(?:outside|extraneous.*outside)", plain)):
            issues.append(ContractIssue("execution_contract_conflict", number, "python_fence_required"))
        if re.search(r"\babstain(?: rather than| instead of| if| when|\.)", plain) and not re.search(
                r"(?:do not|never|must not|not permitted|no)\s+(?:submit\s+)?abstain", plain):
            issues.append(ContractIssue("execution_contract_conflict", number, "readable_images_require_answer"))
        if ("connectivity_graph" in plain and re.search(r"(?:determine|count|number|compute|calculate).*free[- ](?:space |)cells", plain)
                and not re.search(r"(?:does not|cannot|do not|never)", plain)):
            issues.append(ContractIssue("return_schema_mismatch", number, "connectivity_has_no_free_cell_mask"))
        if allowed_tools is not None:
            references = set(re.findall(r"\b[A-Za-z_]\w*\b", line)) & all_tools
            # A warning against using a tool is not a dependency.
            if not re.search(r"(?:do not|never|unavailable|not available|not authorized|cannot use)", plain):
                for name in sorted(references - allowed_tools):
                    issues.append(ContractIssue("unavailable_tool_reference", number, name))

    for match in re.finditer(r"```python[^\n]*\n(.*?)```", skill_md, re.DOTALL):
        offset = skill_md[:match.start(1)].count("\n")
        try:
            tree = ast.parse(match.group(1))
            # Compile without executing; top-level returns are ordinary protocol syntax.
            wrapped = "def _example(ctx):\n" + "\n".join("    " + line for line in match.group(1).splitlines())
            compile(wrapped, "<skill-example>", "exec")
        except (SyntaxError, ValueError):
            issues.append(ContractIssue("example_syntax_error", offset + 1, "python_example_must_compile"))
            continue
        issues.extend(_response_shape_issues(tree, REGISTRY, offset))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = node.func.id
                if name in all_tools and not any(isinstance(v, ast.Starred) for v in node.args) and all(
                        keyword.arg is not None for keyword in node.keywords):
                    signature = inspect.signature(REGISTRY.get(name).fn)
                    try:
                        signature.bind(None, *[None for _ in node.args],
                                       **{k.arg: None for k in node.keywords})
                    except TypeError:
                        issues.append(ContractIssue("tool_argument_schema_mismatch", offset + node.lineno, name))
                if name == "AnswerPayload":
                    allowed = {"value", "unit", "basis", "used_result_ids", "derivation"}
                    if any(k.arg not in allowed for k in node.keywords):
                        issues.append(ContractIssue("answer_schema_mismatch", offset + node.lineno, "answer_payload_fields"))
                if name == "ReturnAnswer" and (len(node.args) != 1 or node.keywords):
                    issues.append(ContractIssue("answer_schema_mismatch", offset + node.lineno, "return_answer_one_argument"))
    return issues
