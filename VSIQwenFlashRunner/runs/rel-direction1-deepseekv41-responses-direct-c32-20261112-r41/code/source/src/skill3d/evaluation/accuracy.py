"""MCA Accuracy（§4 M12）：选项字母抽取 + 精确匹配。"""

from __future__ import annotations

import re
from typing import Optional, Sequence

# 常见答案模式（按优先级）
_PATTERNS = [
    re.compile(r"(?:answer|答案)\s*(?:is|:|：)?\s*\(?([A-Z])\)?", re.IGNORECASE),
    re.compile(r"^\(?([A-Z])\)?[\.\s]"),  # 行首字母
    re.compile(r"\b([A-Z])\b"),  # 兜底：首个独立大写字母
]


def extract_option_letter(text: str) -> Optional[str]:
    """从模型输出抽取选项字母；失败返回 None（记 wrong，进 FailureTaxonomy）。"""
    if text is None:
        return None
    text = text.strip()
    for p in _PATTERNS:
        m = p.search(text)
        if m:
            return m.group(1).upper()
    return None


def mca_correct(predicted: str, ground_truth: str) -> bool:
    """MCA 精确匹配：先抽字母，与 ground_truth 字母比较。"""
    pred = extract_option_letter(predicted)
    gt = extract_option_letter(ground_truth) or ground_truth.strip().upper()
    return pred is not None and pred == gt


def mca_accuracy(predictions: Sequence[str], ground_truths: Sequence[str]) -> float:
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions 与 ground_truths 长度不一致")
    if not predictions:
        raise ValueError("空预测列表")
    correct = sum(mca_correct(p, g) for p, g in zip(predictions, ground_truths))
    return correct / len(predictions)
