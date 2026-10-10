"""Candidate text checks: reject sample identities and answer leakage."""

from __future__ import annotations

import re

# 泄漏扫描模式：答案字样 + 官方 qa_id/sample 样式（真实实现：正则/字段扫描）
_ANSWER_PATTERNS = [
    re.compile(r"ground[_ ]?truth", re.IGNORECASE),
    re.compile(r"\banswer\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"correct\s+option", re.IGNORECASE),
]
# qa/sample id 样式：`qa123`、`qa_123`、`qa-1a2b3c`、`sample_0042`、`question-7`
_QA_ID_PATTERN = re.compile(
    r"\b(?:qa|sample|question|episode)[_-]?[0-9a-f]{1,}\b", re.IGNORECASE)
# 逐字 id 扫描的长度下限：VSI-Bench 官方 id 就是纯整数（"0"/"12"），
# 逐字匹配短数字必然大量误报，故纯数字且短于该值的 id 交给样式正则兜底（TODO_CALIBRATE）
MIN_FORBIDDEN_ID_LEN = 4


def _scan_forbidden_ids(text: str, ids) -> list[str]:
    """逐字扫描禁用 sample id（词边界匹配，避免子串误报）。"""
    hits: list[str] = []
    for sid in ids or ():
        sid = str(sid)
        if not sid:
            continue
        if sid.isdigit() and len(sid) < MIN_FORBIDDEN_ID_LEN:
            continue
        if re.search(rf"(?<![0-9A-Za-z]){re.escape(sid)}(?![0-9A-Za-z])", text):
            hits.append(f"sample_id:{sid}")
    return hits


def leakage_scan_text(text: str, forbidden_sample_ids: set[str] | None = None) -> list[str]:
    """扫描文本，返回命中的泄漏模式列表（空 = 通过）。

    `forbidden_sample_ids` 为本次 run 的 sample/episode id 集合（driver 传 induction
    侧 qa_id），命中即拒（硬约束 19）。
    """
    hits: list[str] = []
    for pat in _ANSWER_PATTERNS:
        if pat.search(text):
            hits.append(f"answer_pattern:{pat.pattern}")
    if _QA_ID_PATTERN.search(text):
        hits.append("qa_id_pattern")
    hits.extend(_scan_forbidden_ids(text, forbidden_sample_ids))
    return hits
