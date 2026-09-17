"""M17 split 物理隔离防火墙（硬约束 9/19）：

- induction / inner / outer / final-test 按 scene_id 分层切分，不共享 scene；
- final_test 数据目录不进沙箱挂载；
- GPT-6 不可见 final split 与任何 ground truth：构造 prompt 前扫描断言。
"""

from __future__ import annotations

import re

from skill3d.schemas import DataSplitConfig


class SplitContaminationError(AssertionError):
    """split 交叉污染：scene 集合交集非空。"""


class PromptLeakageError(AssertionError):
    """GPT-6 prompt 含 ground truth / sample ID / final-test 引用。"""


# 泄漏扫描模式：答案字样、qa_id、final_test 引用（真实实现：正则/字段扫描）
_PROMPT_FORBIDDEN_PATTERNS = [
    re.compile(r"ground[_ ]?truth", re.IGNORECASE),
    re.compile(r"\banswer\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"correct\s+(answer|option)", re.IGNORECASE),
    re.compile(r"\bqa_[0-9a-f]{8,}\b", re.IGNORECASE),
    re.compile(r"final[_ ]?test", re.IGNORECASE),
]


def assert_split_isolation(split: DataSplitConfig) -> None:
    """断言四个 split 的 scene 集合两两不相交（硬约束 9/19）。"""
    groups = {
        "induction": set(split.induction_scene_ids),
        "inner_validation": set(split.inner_validation_scene_ids),
        "outer_holdout": set(split.outer_holdout_scene_ids),
        "final_test": set(split.final_test_scene_ids),
    }
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = groups[a] & groups[b]
            if inter:
                raise SplitContaminationError(
                    f"split 交叉污染: {a} 与 {b} 共享 scene {sorted(inter)}（硬约束 19）"
                )


def scan_prompt_for_leakage(prompt: str,
                            extra_forbidden: list[str] | None = None) -> None:
    """GPT-6 prompt 构造后/发送前扫描：含答案/ID/final 引用即抛（硬约束 13/19）。"""
    hits = [p.pattern for p in _PROMPT_FORBIDDEN_PATTERNS if p.search(prompt)]
    for s in (extra_forbidden or []):
        if s and s in prompt:
            hits.append(f"extra:{s}")
    if hits:
        raise PromptLeakageError(f"GPT-6 prompt 泄漏扫描未通过: {hits}")


def assert_final_test_not_mounted(mounts: list[str],
                                  final_test_dir: str) -> None:
    """断言 final_test 数据目录未挂载进沙箱（硬约束 9：文件系统不挂载）。"""
    norm = final_test_dir.rstrip("/")
    for m in mounts:
        mnt = m.rstrip("/")
        if mnt == norm or mnt.startswith(norm + "/") or norm.startswith(mnt + "/"):
            raise SplitContaminationError(
                f"final_test 目录 {final_test_dir} 被沙箱挂载 {m} 覆盖（硬约束 9）"
            )
