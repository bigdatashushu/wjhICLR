"""M16 离线归纳（v10 §7）：**单个父 Skill 的 ExperienceBundle** → 完整合法候选。

规范原文（§7.1）——离线归纳器每次只收到：

    「父 Skill 完整 `SkillSpec`；父版本的 ExperienceBundle；允许使用的失败摘要和行为
    摘要；当前 ToolSpec 摘要；候选输出 Schema；禁止泄漏和禁止修改项。」

规范原文（§7.2）：候选是 `SkillCandidate`：完整 `full_skill_spec` + 结构化 diff +
父快照 / 父版本声明 + 假设与预期效果。

与 v9 的差别（对应 v10 §1.2 阻断点 EV-01/EV-02/EV-04）：

- **EV-01**：输入不再是"某目录下所有 trace"，而是框架按 `skill_id@version` 构建的
  不可变经验包 `ExperienceBundle`（§6.3）；
- **EV-02**：不再有 `parent_version=None` 的路径 —— 首期 `operation` 只有 `revise`，
  候选必须声明父快照与父版本，无父版本的"从零模板"被静态检查拒绝（§5.2）；
- **EV-04**：产出的是**完整 `SkillSpec`**（离线模型输出完整候选 JSON，框架做校验与
  父 / 子差分），禁止把自由文本拼到序列化 JSON 后面（§5.4 推荐实现方式二）。

纪律（保留 v9 的硬约束）：

- 离线模型**不**执行实验、**不**评分、**不**决定 promote/reject（§3.3）；
- prompt 发送前做泄漏扫描（硬约束 13/19）；
- 服务不可用 → quarantine，不用 mock 顶替（§3.4）；
- 只允许修改 §12.2 列出的项；§12.3 的项（检测模型/阈值、SAM2、FrameSet、Qwen 权重、
  工具实现、评分器、检索权重与 top-k、数据划分、求解轮数）在静态检查里被拒。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Sequence

from skill3d.evolution.firewall import scan_prompt_for_leakage
from skill3d.memory.consolidation import leakage_scan_text
from skill3d.schemas import (
    ExperienceBundle,
    EpisodeTrace,
    SkillCandidate,
    SkillSpec,
)
from skill3d.schemas.evolution import StaticValidationReceipt
from skill3d.skills.library import static_check_skill_spec
from skill3d.skills.registry import required_bump_level, validate_version_bump

# prompt 模板版本（冻结配置项，进 RunManifest 的 prompt_version；§3.4）
INDUCE_PROMPT_VERSION = "induce-v10.0"
# v5 名称别名（过渡期只读；新代码一律用 INDUCE_PROMPT_VERSION）
GPT6_INDUCE_PROMPT_VERSION = INDUCE_PROMPT_VERSION

# `CandidateRevision.created_by` 的受控枚举定义在 `schemas/evolution.py`（本模块不拥有
# 该文件，其字面量仍是 v5 取值）。集中在这里，避免 v5 名称散落在治理链各处。
CREATED_BY_OFFLINE_INDUCTION = "gpt6_induction"

# 跨 scene 最小样本数（与 configs/admission_thresholds.yaml 对齐；TODO_CALIBRATE）
N_MIN_CROSS_SCENE = 3

# §12.3 首期禁止同时修改的项：这些**不是** Skill 正文能表达的东西，但离线模型可能
# 试图在候选里"顺手"改掉它们（例如把阈值写进模板、把检索 top-k 写进描述）。
# 静态检查用确定性关键词 + 结构化字段比对拦截，而不是靠人读。
FORBIDDEN_TEMPLATE_TOKENS: tuple[str, ...] = (
    "box_threshold", "groundingdino", "max_pixels", "top_k", "rank_weights",
    "n_frames", "frame_sampling", "final_test", "outer_holdout",
)

# 方法正文里允许出现的 Tool 名集合由调用方注入（当前 ToolSpec 摘要），本模块不猜。


class InsufficientEvidenceError(ValueError):
    """跨 scene 样本不足 N_min，不归纳。"""


class CandidateStaticCheckError(ValueError):
    """候选静态检查失败（§5.4）：携带逐项检查结果。"""

    def __init__(self, message: str, receipt: StaticValidationReceipt) -> None:
        super().__init__(message)
        self.receipt = receipt


def _canonical(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def sha256_of_spec(spec: SkillSpec) -> str:
    """SkillSpec 的规范内容 hash（交付身份口径与 `skills.delivery` 一致）。"""
    from skill3d.skills.delivery import skill_content_sha256

    return skill_content_sha256(spec)


def structured_diff(parent: SkillSpec, child: SkillSpec) -> list[dict]:
    """父 → 子的**结构化**字段差分（§5.4：diff 必须非空且与候选完整内容一致）。

    只做字段级替换语义（`field` + `from` + `to`）：把它重新作用到父对象上必须还原出
    候选完整内容，这一条由 `validate_candidate_against_parent` 复核。
    """
    parent_dump = parent.model_dump(mode="json")
    child_dump = child.model_dump(mode="json")
    out: list[dict] = []
    for key in sorted(set(parent_dump) | set(child_dump)):
        before = parent_dump.get(key)
        after = child_dump.get(key)
        if before != after:
            out.append({"field": key, "from": before, "to": after})
    return out


def apply_structured_diff(parent: SkillSpec, diff: Sequence[dict]) -> SkillSpec:
    """把结构化 diff 作用到父对象上（§5.4 一致性复核用；不用于产出候选内容）。"""
    payload = parent.model_dump(mode="json")
    for entry in diff:
        field = str(entry.get("field", ""))
        if not field:
            raise CandidateStaticCheckError(
                f"diff 条目缺少 field: {entry!r}",
                StaticValidationReceipt(candidate_id="", problems=["diff_missing_field"]))
        payload[field] = entry.get("to")
    return SkillSpec.model_validate(payload)


# --------------------------------------------------------------------------- #
# prompt 构造（只含允许的摘要；绝不含答案 / sample id）
# --------------------------------------------------------------------------- #

def build_induction_prompt(parent_spec: SkillSpec,
                           bundle: ExperienceBundle,
                           tool_names: Sequence[str],
                           *,
                           allowed_tools_note: str = "",
                           feedback: str = "") -> str:
    """§7.1 的输入清单 → 归纳 prompt。

    输入**只有**：父完整 SkillSpec、父版本经验包摘要、当前 ToolSpec 工具名、输出
    Schema、禁止项。失败 / 行为摘要是**计数与类别**，不含题面、答案或 sample id
    （§6.4 标签边界：learning 的答对/答错可用于离线归纳，但具体答案不写入候选）。
    """
    lines = [
        "你是 harness3D 的离线 Skill 修订器（Offline Inducer）。",
        "任务：基于**同一个父 Skill** 的真实执行经验，产出一个完整的候选 SkillSpec（JSON），",
        "而不是片段或补丁文本。你不是执行者：不执行实验、不评分、不决定 promote/reject。",
        "",
        f"## 父 Skill 完整 SkillSpec（skill_id={parent_spec.skill_id}@{parent_spec.version}）",
        _canonical(parent_spec.model_dump(mode="json")),
        "",
        "## 父版本的真实执行经验包（不含题面与答案）",
        _canonical({
            "bundle_id": bundle.bundle_id,
            "generation": bundle.generation,
            "canonical_question_type": bundle.canonical_question_type,
            "scene_count": bundle.scene_count,
            "success_count": bundle.success_count,
            "failure_count": bundle.failure_count,
            "behavior_summary": bundle.behavior_summary,
            "failure_summary": bundle.failure_summary,
            "exclusion_summary": bundle.exclusion_summary,
        }),
        "",
        "## 当前 ToolSpec 可用工具名（候选正文只能引用这些名字）",
        ", ".join(sorted(str(t) for t in tool_names)) or "（无）",
    ]
    if allowed_tools_note:
        lines += ["", allowed_tools_note]
    if feedback:
        # §7.4："Schema 不合法：生成结构化错误反馈，产生新 revision" —— 反馈只含
        # **静态检查的问题清单**（不含答案、不含题目内容）。
        lines += ["", "## 上一版候选未通过静态检查（请修正后重新输出完整候选）", feedback]
    lines += [
        "",
        "## 输出（严格 JSON，不要 Markdown 围栏）",
        _canonical({
            "spec": {k: v for k, v in _SPEC_SCHEMA_HINT.items()},
            "hypothesis": "你预期这次修订解决什么可观察问题",
            "expected_effect": "预期在哪些行为指标上体现",
            "known_risks": ["已知风险（字符串列表）"],
            "diff_summary": "一句话说明相对父版本改了什么",
        }),
        "",
        "## 硬约束（违反即被静态检查拒绝）",
        f"- skill_id 必须等于 {parent_spec.skill_id!r}；version 必须**严格大于** "
        f"{parent_spec.version!r} 且只允许 MINOR 级修订（§7.3：MAJOR/PATCH 超出首期）；",
        "- 只允许修改 §12.2 的项：目标类别识别步骤、跨帧覆盖策略、重复实例与遮挡重现"
        "处理、检测故障/截断/空检出分支、工具结果与视觉证据冲突处理、提交前检查与失败教训；",
        "- **禁止**修改 §12.3 的项：检测模型与阈值、SAM2 参数、FrameSet 采样、Qwen 权重与"
        "量化、工具实现、评分器、检索权重与 top-k、数据划分、求解轮数与重试次数；",
        "- 不得出现任何具体题目答案、选项字母、sample id、scene id 或题号；",
        "- 不得引用上面工具清单之外的工具名；",
        f"- applicable_question_types 必须仍为 [{bundle.canonical_question_type!r}]"
        "（首期单题型）；",
    ]
    prompt = "\n".join(lines)
    scan_prompt_for_leakage(prompt)   # 发送前泄漏扫描（硬约束 13/19）
    return prompt


_SPEC_SCHEMA_HINT: dict = {
    "skill_id": "string（= 父 skill_id）",
    "version": "string MAJOR.MINOR.PATCH（> 父版本，MINOR 级）",
    "applicable_question_types": ["string（单题型）"],
    "required_evidence_signature": {"能力名": "available|degraded|unavailable"},
    "requires_metric_evidence": False,
    "applicable_gate_version": None,
    "skill_family": "counting|metric|relative_geometry|route|appearance",
    "source": "real",
    "description": "string",
    "call_graph_template": "string（步骤 / 检查 / 失败教训 / 通用代码示例）",
    "supported_coordinate_frames": ["world"],
    "validation_assertions": ["string"],
}


# --------------------------------------------------------------------------- #
# 静态检查（§5.4）
# --------------------------------------------------------------------------- #

def validate_candidate_against_parent(candidate_spec: SkillSpec,
                                      parent_spec: SkillSpec,
                                      *,
                                      parent_content_sha256: str,
                                      claimed_parent_content_sha256: str,
                                      canonical_question_type: str,
                                      tool_names: Sequence[str],
                                      forbidden_sample_ids: Optional[set[str]] = None,
                                      method_context_max_chars: int = 8000,
                                      raw_text: str = "") -> StaticValidationReceipt:
    """§5.4 静态检查的**唯一实现**（归纳与修订两条路径共用）。

    `raw_text` 是离线模型原始输出文本。给了就复核"它确实是完整 JSON 且与解析结果
    一致"（`json_parseable`）；不给（例如从收据重放对象）则该检查按"调用方已从 JSON
    构造对象"记为通过，但**不**把任何拼接文本路径当作候选来源。
    """
    checks: dict[str, bool] = {}
    problems: list[str] = []

    checks["json_parseable"] = True
    if raw_text:
        try:
            reparsed, _meta = parse_full_spec_response(raw_text)
            if _canonical(reparsed.model_dump(mode="json")) != \
                    _canonical(candidate_spec.model_dump(mode="json")):
                checks["json_parseable"] = False
                problems.append("离线模型原始输出与解析后的候选不一致（§5.4）")
        except Exception as exc:  # noqa: BLE001 - 静态检查 fail-closed
            checks["json_parseable"] = False
            problems.append(f"离线模型原始输出不是完整合法 JSON 候选: {exc}")

    # schema_extra_forbid：SkillSpec 构造期即 extra=forbid；这里复核题型单一与正文长度。
    spec_problems = static_check_skill_spec(
        candidate_spec, method_context_max_chars=int(method_context_max_chars))
    checks["schema_extra_forbid"] = not spec_problems
    checks["method_body_deliverable"] = not [
        p for p in spec_problems if "服务限制" in p or "方法正文" in p]
    for p in spec_problems:
        problems.append(p)
    checks["single_canonical_question_type"] = (
        list(candidate_spec.applicable_question_types) == [canonical_question_type])
    if not checks["single_canonical_question_type"]:
        problems.append(
            "候选声明题型 "
            f"{candidate_spec.applicable_question_types} ≠ 父唯一规范题型 "
            f"{canonical_question_type!r}（§8.5 三处必须一致）")

    checks["skill_id_matches_parent"] = candidate_spec.skill_id == parent_spec.skill_id
    if not checks["skill_id_matches_parent"]:
        problems.append(
            f"候选 skill_id={candidate_spec.skill_id!r} ≠ 父 {parent_spec.skill_id!r}"
            "（§5.2：首期只允许修订已有方法，禁止从零创建）")

    # 版本：必须严格递增，且首期只接受 MINOR（§7.3）。
    #
    # §7.3 的映射：**修改描述、步骤、检查、示例或失败教训 → MINOR**（`required_bump_level`
    # 把"只改 description/template 之外"判为 PATCH，那是 v6 的措辞级别，不是行为差异）；
    # **修改题型、硬证据条件或权限语义 → MAJOR 且超出首期范围**。
    # 至于"只修错字、不改变行为"这一类，机制上无法与"改进了描述"区分 —— 因此不靠
    # 文本差异猜测，而是由 §7.4 的 `no_effective_diff`（内容 diff 为空）与下游的配对
    # 效果评测把关。
    bump_ok = True
    try:
        validate_version_bump(parent_spec, candidate_spec)
    except Exception as exc:  # noqa: BLE001 - 静态检查必须 fail-closed
        bump_ok = False
        problems.append(f"版本递增非法: {exc}")
    level = required_bump_level(parent_spec, candidate_spec)
    if bump_ok and level == "MAJOR":
        bump_ok = False
        problems.append(
            "候选改动了检索前提（题型/证据签名/米制 gate/族/来源）→ MAJOR，"
            "超出首期范围（§7.3）")
    checks["version_bump_legal"] = bump_ok

    checks["method_body_deliverable"] = not [
        p for p in problems if "服务限制" in p or "方法正文" in p]

    # 工具名必须来自当前 ToolSpec（§5.4）。
    #
    # 只把**裸调用** `name(...)` 当作工具引用：`x.append(...)` 是属性访问（Python
    # 方法），不是 Tool 名 —— 本轮真实运行里候选写了 `tracks.append(...)` 被误判成
    # "未知工具 append"（假阳性），所以这里要求名字**前面不是 `.`**。
    known = {str(t) for t in tool_names}
    import re as _re
    template = str(candidate_spec.call_graph_template or "")
    mentioned = set(_re.findall(r"(?<![.\w])([a-z][a-z0-9_]{2,})\s*\(", template))
    unknown_tools = sorted(t for t in mentioned if t not in known
                           and t not in _NON_TOOL_CALL_WORDS)
    checks["tools_known"] = not unknown_tools
    if unknown_tools:
        problems.append(f"候选正文引用了未知工具 {unknown_tools}（§5.4）")

    # 泄漏：答案模式 / qa_id / 题号 / scene id（硬约束 13/19）。
    leaks = leakage_scan_text(_canonical(candidate_spec.model_dump(mode="json")),
                              forbidden_sample_ids=forbidden_sample_ids)
    for token in FORBIDDEN_TEMPLATE_TOKENS:
        if token in _canonical(candidate_spec.model_dump(mode="json")).lower():
            leaks.append(f"forbidden_knob:{token}")
    checks["no_leakage"] = not leaks
    if leaks:
        problems.append(f"候选内容命中泄漏 / 禁止修改项: {sorted(set(leaks))}")

    checks["parent_hash_matches"] = bool(
        parent_content_sha256 and claimed_parent_content_sha256
        and parent_content_sha256 == claimed_parent_content_sha256)
    if not checks["parent_hash_matches"]:
        problems.append(
            "父版本 hash 与候选记录不一致（§5.4）："
            f"snapshot={parent_content_sha256!r} claimed={claimed_parent_content_sha256!r}")

    # diff 非空且与候选完整内容一致（§5.4 最后一条）。
    #
    # "非空"指**内容**非空：只 bump 版本号、其余字段一字未改的候选属于
    # §7.4 的 `no_effective_diff`（reject），不能因为 version 字段不同就算"有改动"。
    diff_valid = True
    try:
        diff = structured_diff(parent_spec, candidate_spec)
        effective = [e for e in diff if str(e.get("field", "")) != "version"]
        if not effective:
            diff_valid = False
            problems.append(
                "候选与父版本除版本号外无任何字段变化 → no_effective_diff（§7.4）")
        else:
            rebuilt = apply_structured_diff(parent_spec, effective)
            rebuilt_payload = rebuilt.model_dump(mode="json")
            rebuilt_payload["version"] = candidate_spec.version
            if _canonical(rebuilt_payload) != _canonical(
                    candidate_spec.model_dump(mode="json")):
                diff_valid = False
                problems.append("diff 与候选完整内容不一致（§5.4）")
    except Exception as exc:  # noqa: BLE001
        diff_valid = False
        problems.append(f"diff 复核失败: {type(exc).__name__}: {exc}")
    checks["diff_nonempty_and_consistent"] = diff_valid

    return StaticValidationReceipt(
        candidate_id="", checks=checks, problems=problems,
        passed=not problems and all(checks.values()),
        created_at=datetime.now(timezone.utc).isoformat())


# 模板里"名字后面带括号但不是工具"的常见词（避免把 `for x in ...` / 语言构造误判成工具）。
_NON_TOOL_CALL_WORDS: frozenset[str] = frozenset({
    "if", "for", "while", "print", "len", "range", "int", "float", "str", "list",
    "dict", "set", "tuple", "sorted", "sum", "min", "max", "abs", "round", "enumerate",
    "zip", "map", "filter", "any", "all", "isinstance", "getattr", "format", "json",
    "def", "return", "assert", "except", "with", "open", "type", "bool", "nan",
    # Python 关键字（`else` 这种"名字后面带括号"的写法在本轮真实候选里出现过）
    "else", "elif", "try", "finally", "raise", "lambda", "yield", "del", "import",
    "from", "as", "global", "nonlocal", "pass", "break", "continue", "in", "not",
    "and", "or", "is", "None", "True", "False",
    # 常见内建/容器方法（裸调用形态；`x.append(...)` 已由"前导点"规则排除）
    "append", "extend", "insert", "remove", "pop", "keys", "values", "items", "get",
    "update", "add", "discard", "join", "split", "strip", "lower", "upper", "replace",
    "startswith", "endswith", "count", "index", "copy", "clear", "reverse", "sort",
    "mean", "median", "std", "argmax", "argmin", "shape", "reshape", "astype",
    "tolist", "sqrt", "exp", "log", "pow", "ceil", "floor", "zeros", "ones", "array",
})


# --------------------------------------------------------------------------- #
# 归纳主入口
# --------------------------------------------------------------------------- #

def parse_full_spec_response(response_text: str) -> tuple[SkillSpec, str]:
    """离线模型输出 → `(SkillSpec, raw_json_text)`；解析失败抛 `ValueError`。

    只接受完整候选对象：`{"spec": {...}, ...}` 或直接给 SkillSpec 对象两种形态都支持，
    但**绝不**接受"父 spec + 文本补丁"这种 v9 形态（§5.4 禁止方式一与方式二的混淆）。
    """
    text = str(response_text or "").strip()
    if not text:
        raise ValueError("离线模型返回空响应")
    if text.startswith("```"):
        # 首期纪律：不把围栏当作正常格式（§11.3 的口径对候选同样适用）——
        # 显式剥离并记录，但**不**作为"解析恢复"。
        text = text.strip("`")
        if "\n" in text:
            first, _, rest = text.partition("\n")
            if first.strip().lower() in ("json", ""):
                text = rest
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"候选必须是 JSON 对象，收到 {type(payload).__name__}")
    spec_payload = payload.get("spec") if "spec" in payload else payload
    if not isinstance(spec_payload, dict):
        raise ValueError("候选的 spec 字段不是 JSON 对象")
    if "# PATCH" in text:
        raise ValueError(
            "候选内容含 '# PATCH' 文本补丁形态（v10 §5.4 禁止把自由文本拼接到 JSON）")
    return SkillSpec.model_validate(spec_payload), text


def induce_candidate_from_bundle(
    *,
    bundle: ExperienceBundle,
    parent_spec: SkillSpec,
    parent_snapshot_id: str,
    campaign_id: str,
    generation: int,
    candidate_skill_version: str,
    offline_client,
    tool_names: Sequence[str],
    parent_content_sha256: str = "",
    method_context_max_chars: int = 8000,
    qa_ids_to_avoid: Optional[set[str]] = None,
    n_min: int = N_MIN_CROSS_SCENE,
    static_feedback: str = "",
) -> SkillCandidate:
    """§7 主入口：经验包 → 完整合法候选 `SkillCandidate`。

    - `offline_client` 为 None → 抛 `RuntimeError`（**不**返回 None 冒充候选；
      服务不可用由调用方记 quarantine，§3.4）；
    - 经验包不足 `n_min` 或跨题不足 → `InsufficientEvidenceError`；
    - 静态检查失败 → `CandidateStaticCheckError`（携带逐项收据，供 quarantine 落盘）。
    """
    if offline_client is None:
        raise RuntimeError(
            "离线模型不可用：不得用 mock / 空候选顶替（§3.4）→ 调用方应记 quarantine")
    if bundle.parent_skill_id != parent_spec.skill_id \
            or bundle.parent_skill_version != parent_spec.version:
        raise ValueError(
            "经验包不属于该父版本（§6.3 禁止混桶）："
            f"bundle={bundle.parent_skill_key} parent={parent_spec.skill_id}@{parent_spec.version}")
    if bundle.scene_count < n_min or bundle.n_eligible < n_min:
        raise InsufficientEvidenceError(
            f"合格经验不足: scenes={bundle.scene_count} eligible={bundle.n_eligible}, "
            f"N_min={n_min}（TODO_CALIBRATE）")

    prompt = build_induction_prompt(parent_spec, bundle, tool_names,
                                    feedback=static_feedback)
    response = offline_client.chat(prompt)
    spec, raw_text = parse_full_spec_response(response)

    parent_hash = parent_content_sha256 or sha256_of_spec(parent_spec)
    receipt = validate_candidate_against_parent(
        spec, parent_spec,
        parent_content_sha256=parent_hash,
        claimed_parent_content_sha256=parent_hash,
        canonical_question_type=bundle.canonical_question_type,
        tool_names=tool_names,
        forbidden_sample_ids=qa_ids_to_avoid,
        method_context_max_chars=method_context_max_chars,
        raw_text=raw_text)
    receipt.candidate_id = f"cand-{uuid.uuid4().hex[:12]}"
    receipt.campaign_id = campaign_id
    receipt.generation = int(generation)
    if not receipt.passed:
        raise CandidateStaticCheckError(
            f"候选未通过静态检查（§5.4）: {receipt.problems}", receipt)

    meta = parse_candidate_metadata(response)
    diff = structured_diff(parent_spec, spec)
    return SkillCandidate(
        candidate_id=receipt.candidate_id,
        campaign_id=campaign_id,
        generation=int(generation),
        operation="revise",
        parent_snapshot_id=parent_snapshot_id,
        parent_skill_version=f"{parent_spec.skill_id}@{parent_spec.version}",
        candidate_skill_version=candidate_skill_version,
        canonical_question_type=bundle.canonical_question_type,
        hypothesis=meta.get("hypothesis", ""),
        expected_effect=meta.get("expected_effect", ""),
        known_risks=list(meta.get("known_risks") or []),
        full_skill_spec=spec,
        structured_diff=diff,
        source_experience_bundle_ref=bundle.bundle_id,
        inducer_receipt_ref=f"induce:{INDUCE_PROMPT_VERSION}",
    )


def parse_candidate_metadata(response_text: str) -> dict:
    """从离线模型响应里取候选的假设 / 预期效果 / 风险（缺字段返回空值，不编造）。"""
    try:
        payload = json.loads(str(response_text or "").strip().strip("`"))
    except Exception:  # noqa: BLE001 - 元数据缺失不影响候选本体
        return {}
    if not isinstance(payload, dict):
        return {}
    risks = payload.get("known_risks")
    if isinstance(risks, str):
        risks = [risks]
    return {
        "hypothesis": str(payload.get("hypothesis", "") or ""),
        "expected_effect": str(payload.get("expected_effect", "") or ""),
        "known_risks": [str(r) for r in (risks or [])],
        "diff_summary": str(payload.get("diff_summary", "") or ""),
    }


# --------------------------------------------------------------------------- #
# v9 兼容路径（**不得**用于 v10 campaign）
# --------------------------------------------------------------------------- #

def induce_candidate_legacy_v9(traces: list[EpisodeTrace],
                               task_type_of: dict[str, str],
                               scene_of: dict[str, str],
                               offline_client=None,
                               n_min: int = N_MIN_CROSS_SCENE,
                               *,
                               gpt6_client=None):
    """v9 形状的归纳（tracks → 无父版本 draft）。

    **不得用于 v10 演化链**：它按题型聚合并产出 `parent_version=None` 的候选，
    正是 v10 §1.2 的 EV-01/EV-02 阻断点。保留本函数只为让 v9 的 `offline_driver`
    仍在旧口径下可运行；新代码一律用 `induce_candidate_from_bundle`。
    """
    from skill3d.schemas import CandidateRevision

    if offline_client is None and gpt6_client is not None:
        offline_client = gpt6_client
    clusters = cluster_traces(traces, task_type_of, scene_of)
    covered_scenes = {key.split("|", 1)[1] for key in clusters}
    if len(covered_scenes) < n_min or len(traces) < n_min:
        raise InsufficientEvidenceError(
            f"跨 scene 样本不足: scenes={len(covered_scenes)}, traces={len(traces)}, "
            f"N_min={n_min}（TODO_CALIBRATE）"
        )
    failure_summaries = sorted({
        ",".join(t.failure.categories) for t in traces if t.failure is not None
    })
    input_features = sorted({task_type_of.get(t.episode_id, "unknown") for t in traces})
    prompt = _legacy_induction_prompt(failure_summaries, input_features)
    if offline_client is None:
        return None
    spec_content = offline_client.chat(prompt)
    return CandidateRevision(
        revision_id=f"rev-{uuid.uuid4().hex[:12]}",
        root_candidate_id=f"cand-{uuid.uuid4().hex[:12]}",
        parent_version=None,  # v9：无父版本（v10 禁止，见函数 docstring）
        candidate_type="skill",
        spec_content=spec_content,
        status="draft",
        induction_trace_refs=[t.episode_id for t in traces],
        evidence_lineage_ref="",
        created_by=CREATED_BY_OFFLINE_INDUCTION,  # type: ignore[arg-type]
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def cluster_traces(traces: list[EpisodeTrace],
                   task_type_of: dict[str, str],
                   scene_of: dict[str, str]) -> dict[str, list[EpisodeTrace]]:
    """按 (题型, scene) 聚类轨迹。key = "task_type|scene"（v9 兼容路径用）。"""
    clusters: dict[str, list[EpisodeTrace]] = {}
    for t in traces:
        key = f"{task_type_of.get(t.episode_id, 'unknown')}|{scene_of.get(t.episode_id, 'unknown')}"
        clusters.setdefault(key, []).append(t)
    return clusters


def _legacy_induction_prompt(failure_summaries: list[str],
                             input_features: list[str]) -> str:
    lines = [
        "你是 harness3D 的离线 Skill 归纳器（Offline Inducer）。",
        "基于以下失败类型摘要与输入特征，归纳一个跨场景可泛化的题型级程序合成模板",
        "（不得包含任何具体题目答案）。",
        "你**不**执行实验、不评分、不决定 promote/reject —— 准入由确定性门决定。",
        "## 失败类型摘要",
        *[f"- {s}" for s in failure_summaries],
        "## 输入特征",
        *[f"- {f}" for f in input_features],
        "## 输出",
        "输出 candidate_v0 的 spec_content（模板文本，JSON）。",
    ]
    prompt = "\n".join(lines)
    scan_prompt_for_leakage(prompt)
    return prompt


__all__ = [
    "CREATED_BY_OFFLINE_INDUCTION",
    "CandidateStaticCheckError",
    "FORBIDDEN_TEMPLATE_TOKENS",
    "GPT6_INDUCE_PROMPT_VERSION",
    "INDUCE_PROMPT_VERSION",
    "InsufficientEvidenceError",
    "N_MIN_CROSS_SCENE",
    "apply_structured_diff",
    "build_induction_prompt",
    "cluster_traces",
    "induce_candidate_from_bundle",
    "induce_candidate_legacy_v9",
    "parse_candidate_metadata",
    "parse_full_spec_response",
    "sha256_of_spec",
    "structured_diff",
    "validate_candidate_against_parent",
]
