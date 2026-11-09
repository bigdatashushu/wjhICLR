"""NA 任务 MRA（§4 M12 / §8）：**严格对齐官方口径**（B-4 决策）。

官方口径（`lmms_eval/tasks/vsibench/utils.py::mean_relative_accuracy`）：

```python
THETAS = np.linspace(0.5, 0.95, 10)          # 0.50, 0.55, …, 0.95
rel    = abs(pred - gt) / gt                  # 分母是 gt，**不对分母做 max 钳制**
score  = mean([rel <= (1 - theta) for theta in THETAS])   # θ 越大越严
```

口径要点（旧实现作废，NA 结果需重跑）：

- 判定方向是 `rel ≤ 1 − θ`：**θ 是"允许的相对误差上界"的补**，不是容差本身。
  因此 `rel=0.5` 时只有 θ=0.5 通过 → MRA=0.10；`rel=0.2` → 0.70（不是 0.60）；
- 分母为 gt，且 NA 任务 gt 无 0 值（官方排除 count=1 与 0 距离样本）；
- 非法值（无法解析为数字 / NaN / inf）记 **0 分**（对齐官方 `to_float` 的
  `except → WORST_CASE=0.0`）；
- `object_counting` 不要求整数预测，直接走 MRA 浮点打分；
- golden（§8.2）：rel ∈ {0, 0.05, 0.1, 0.2, 0.5, 1.0} → MRA ∈ {1.0, 1.0, 0.9, 0.7, 0.1, 0.0}。

附录诊断（**不进主表**）：`MRA@10% = mean(rel <= 0.1)`，标注为
"our diagnostic sub-metric"（§8.4）。
"""

from __future__ import annotations

import math
import re
from typing import Optional, Sequence

import numpy as np

# 官方 10 阈值（§8.1；与官方 `np.linspace(0.5, 0.95, 10)` 逐值一致）
THETAS: np.ndarray = np.linspace(0.5, 0.95, 10)
MRA_N_THETAS: int = int(THETAS.size)
# 官方接口默认参数（用于逐值对拍）
DEFAULT_START, DEFAULT_END, DEFAULT_INTERVAL = 0.5, 0.95, 0.05
# 非法预测的分数（官方 WORST_CASE）
WORST_CASE: float = 0.0
# 附录诊断阈值（§8.4，不进主表）
DIAGNOSTIC_REL: float = 0.1

# 数值抽取：首个带可选正负号/小数/科学计数法的数字
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _thetas(start: float = DEFAULT_START, end: float = DEFAULT_END,
            interval: float = DEFAULT_INTERVAL) -> list[float]:
    """阈值生成：`np.linspace(start, end, (end-start)/interval + 1)`。

    默认参数下与官方 `np.linspace(0.5, 0.95, 10)` **逐值一致**（10 个阈值）。
    注意不能用 `start + i*interval` 累加：`(0.95-0.5)/0.05 = 8.999999999999998`
    会因浮点误差少生成一个阈值（9 个），使 §8.2 的 golden 值全部偏移。
    """
    n = int(round((float(end) - float(start)) / float(interval))) + 1
    return [float(v) for v in np.linspace(float(start), float(end), n)]


def parse_numeric_answer(text: Optional[str]) -> Optional[float]:
    """从模型输出抽取数值（§4 M12 字段 9：不可解析 → 记 wrong / MRA 0）。

    NA 题答案可能带单位（"3.5 m"、"12 meters"、"48 m2"），取首个数值。
    """
    if text is None:
        return None
    m = _NUM_RE.search(str(text))
    if m is None:
        return None
    try:
        val = float(m.group(0))
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def is_legal(pred: Optional[float], gt: Optional[float]) -> bool:
    """预测与真值是否可参与 MRA（非数字 / NaN / inf / gt<=0 → 非法）。"""
    for v in (pred, gt):
        if v is None or not isinstance(v, (int, float)):
            return False
        if not math.isfinite(float(v)):
            return False
    return float(gt) > 0.0  # NA 任务 gt 无 0 值（分母为 gt）


def relative_error(pred: float, gt: float) -> float:
    """相对误差：`abs(pred - gt) / gt`（分母为 gt，不做 max 钳制，§8.1）。"""
    return abs(float(pred) - float(gt)) / float(gt)


def mra_one(pred: float, gt: float, *, start: float = DEFAULT_START,
            end: float = DEFAULT_END, interval: float = DEFAULT_INTERVAL) -> float:
    """单样本 MRA：`mean([rel <= 1 - theta for theta in thetas])`。

    非法值（不可解析/NaN/非正 gt）→ `WORST_CASE`（0.0），不抛异常不跳过。
    """
    if not is_legal(pred, gt):
        return WORST_CASE
    rel = relative_error(pred, gt)
    thetas = _thetas(start, end, interval)
    return float(np.mean([rel <= (1.0 - th) for th in thetas]))


def mean_relative_accuracy(preds: Sequence, gts: Sequence, *,
                           start: float = DEFAULT_START, end: float = DEFAULT_END,
                           interval: float = DEFAULT_INTERVAL) -> float:
    """官方同签名接口：逐样本 MRA 的算术平均（§8.3 第 8 任务聚合口径）。

    与官方 `lmms_eval/tasks/vsibench/utils.py::mean_relative_accuracy`
    在相同输入下逐值一致（合法输入）；非法预测按官方 `WORST_CASE=0.0` 计。
    """
    if len(preds) != len(gts):
        raise ValueError("preds 与 gts 长度不一致")
    if not preds:
        raise ValueError("空输入")
    scores = [mra_one(p, g, start=start, end=end, interval=interval)
              for p, g in zip(preds, gts)]
    return float(np.mean(scores))


def mra(preds, gts) -> float:
    """数组版 MRA（等价 `mean_relative_accuracy`；保留历史入口）。"""
    return mean_relative_accuracy(list(np.asarray(preds, dtype=np.float64).tolist()),
                                  list(np.asarray(gts, dtype=np.float64).tolist()))


def mra_single(pred: float, gt: float) -> float:
    """单样本 MRA（runner 逐题打分入口）。"""
    return mra_one(pred, gt)


def mra_from_text(pred_text: Optional[str], gt_text: Optional[str]) -> float:
    """文本 → MRA（不可解析记 0 分，§8.1）。"""
    return mra_one(parse_numeric_answer(pred_text), parse_numeric_answer(gt_text))


def mra_at_10pct(preds: Sequence, gts: Sequence) -> Optional[float]:
    """附录诊断指标 `MRA@10% = mean(rel <= 0.1)`（§8.4，**不进主表**）。

    非法样本计 0；全部非法 → None（不臆造）。
    """
    if len(preds) != len(gts):
        raise ValueError("preds 与 gts 长度不一致")
    if not preds:
        return None
    vals = []
    for p, g in zip(preds, gts):
        vals.append(1.0 if (is_legal(p, g) and relative_error(p, g) <= DIAGNOSTIC_REL)
                    else 0.0)
    return float(np.mean(vals))


def mra_with_detail(preds: Sequence, gts: Sequence) -> dict:
    """聚合 + 明细（论文表格与审计用）：主指标 + 诊断子指标 + 非法样本数。"""
    n = len(preds)
    illegal = sum(0 if is_legal(p, g) else 1 for p, g in zip(preds, gts))
    return {
        "mra": mean_relative_accuracy(preds, gts) if n else None,
        "mra_at_10pct": mra_at_10pct(preds, gts),
        "n": n,
        "n_illegal": illegal,
        "rel": [relative_error(p, g) if is_legal(p, g) else None
                for p, g in zip(preds, gts)],
    }
