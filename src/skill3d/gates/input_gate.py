"""M2 Input Gate：**被动**帧质量观测（§4 M2 伪代码实现）。

在线模块，严禁任何 GPT-6 相关依赖（硬约束 1）。
重建/Skill 路由之前必须先过本门禁（硬约束 16）。

**硬约束 21（统一固定 FrameSet）**：M2 只做被动观测 —— 输出
`quality_score` / `degradation_flags` / `quality_weight`，用于置信度、route 判定、
归因分析与消融实验；**绝不删帧、不换帧、不补帧、不重排帧**。旧设计里的
"删除劣化帧 / 补帧"（`drop_and_refill`）动作已废弃：帧集在 M1 之后冻结。

只有**输入合法性**是 hard fail（无法解码 / 空帧 / 尺寸非法 / 损坏 / 帧数 < 32），
这类 episode 记 `unanswerable`（G4 帧数完整性），不进入任何 split。
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from skill3d.gates import iqa
from skill3d.schemas.episode import FrameSet, InputFrame, InputGateVerdict

# ---- 阈值常量（TODO_CALIBRATE；下述绝对值为**本机实测标定**的起点）----
# 实测（2026-09-18，32 帧均匀采样，Laplacian 方差）：
#   arkitscenes 中位≈39（min 1.5）| scannet 中位≈50（min 9.6）| scannetpp 中位≈315
# 结论：文档起始值 100 会把几乎全部真实手持帧判为"模糊"（跨数据集相差近 10 倍），
# 故改为"绝对下界 + 相对本 episode 中位数"的双判据（见 blur_floor_for）。
TH_BLUR_VAR_ABS: float = 10.0     # TODO_CALIBRATE: 绝对下界（低于实测正常帧最小值）
TH_BLUR_REL: float = 0.35         # TODO_CALIBRATE: < 本 episode 中位×该比例记模糊
TH_BLUR_VAR: float = TH_BLUR_VAR_ABS   # 兼容别名（旧代码引用）
TH_OVER_EXPOSED: float = 0.05     # TODO_CALIBRATE: p_over > 5% 记曝光异常
TH_UNDER_EXPOSED: float = 0.05    # TODO_CALIBRATE: p_under > 5% 记曝光异常
TH_DEGRADED_RATIO: float = 0.25   # TODO_CALIBRATE: 劣化帧占比超此值 → 仅降权（不再删帧）
MIN_FRAMES: int = 32              # G4 帧数完整性（官方 32 帧；不足 = 输入合法性 hard fail）
TH_MOTION: float = 20.0           # TODO_CALIBRATE: G3 帧间光流均值阈值（px）
# 被打上劣化标记的帧的观测权重（TODO_CALIBRATE；只影响置信度，不影响帧集）
DEGRADED_FRAME_WEIGHT: float = 0.5


def frame_quality(frame: Union[InputFrame, np.ndarray]) -> tuple[float, float, float]:
    """返回 (blur_var, overexposed_ratio, underexposed_ratio)。

    输入为 InputFrame 时直接读已算好的字段；输入为原始图像 ndarray 时实时计算。
    """
    if isinstance(frame, InputFrame):
        return frame.blur_var, frame.overexposed_ratio, frame.underexposed_ratio
    try:
        blur = iqa.laplacian_var(frame)
        p_over, p_under = iqa.exposure_ratios(frame)
    except Exception:  # noqa: BLE001 - 非法帧（空帧/损坏/尺寸非法）算不出 IQA
        return float("nan"), float("nan"), float("nan")
    return blur, p_over, p_under


def blur_floor_for(blurs: Sequence[float]) -> float:
    """G1 双判据的模糊下界：绝对下界与"本 episode 中位×TH_BLUR_REL"取较严者。"""
    arr = np.asarray(list(blurs), dtype=np.float64)
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size >= 4:
        return max(TH_BLUR_VAR_ABS, TH_BLUR_REL * float(np.median(finite)))
    return TH_BLUR_VAR_ABS


def degradation_flags(blur: float, p_over: float, p_under: float,
                      blur_floor: float = TH_BLUR_VAR_ABS) -> list[str]:
    """单帧劣化标记（被动观测；不触发任何帧集变更）。"""
    flags: list[str] = []
    if not np.isfinite(blur) or blur < blur_floor:
        flags.append("blur")
    if p_over > TH_OVER_EXPOSED:
        flags.append("overexposed")
    if p_under > TH_UNDER_EXPOSED:
        flags.append("underexposed")
    return flags


def quality_weight_for(flags: Sequence[str]) -> float:
    """劣化标记 → 观测权重（TODO_CALIBRATE；只降权，不删帧）。"""
    return DEGRADED_FRAME_WEIGHT if flags else 1.0


def _is_degraded(blur: float, p_over: float, p_under: float,
                 blur_floor: float = TH_BLUR_VAR_ABS) -> bool:
    """单帧是否劣化（兼容入口，语义 = degradation_flags 非空）。"""
    return bool(degradation_flags(blur, p_over, p_under, blur_floor))


def _illegal_frame_ids(frames: Sequence) -> list[int]:
    """输入合法性 hard fail 的帧（空帧 / 尺寸非法 / 损坏 / 无法解码）。

    只对拿到像素的调用方有意义；`InputFrame` 记录无像素，尺寸非法时同样标出。
    """
    bad: list[int] = []
    for i, f in enumerate(frames):
        if isinstance(f, InputFrame):
            if f.width <= 0 or f.height <= 0:
                bad.append(i)
            continue
        if f is None:
            bad.append(i)
            continue
        try:
            arr = np.asarray(f)
        except Exception:  # noqa: BLE001 - 损坏/不可解码
            bad.append(i)
            continue
        if arr.size == 0 or arr.ndim < 2 or min(arr.shape[:2]) < 1:
            bad.append(i)
    return bad


def input_gate(
    frames: Sequence[Union[InputFrame, np.ndarray]],
    *,
    frame_set: Optional[FrameSet] = None,
    hard_fail_frame_ids: Optional[Sequence[int]] = None,
) -> InputGateVerdict:
    """按 §4 M2 伪代码实现的整体门禁判定（**被动观测**，硬约束 21）。

    - `level=pass` / `locally_degraded`：一律 `action="proceed"`，附带逐帧 flag 与权重；
    - `level=overall_unusable`：仅输入合法性 hard fail（空帧/尺寸非法/损坏/帧数 <32）
      → `action="unanswerable"`，整 episode 记 `unavailable`，不进任何 split。
    """
    scores = [frame_quality(f) for f in frames]
    blur_floor = blur_floor_for([b for b, _po, _pu in scores])

    all_flags: list[str] = []
    per_frame_flags: list[list[str]] = []
    for b, po, pu in scores:
        fl = degradation_flags(b, po, pu, blur_floor)
        per_frame_flags.append(fl)
        all_flags.extend(fl)
    degraded = [i for i, fl in enumerate(per_frame_flags) if fl]

    # 输入合法性：显式传入的 hard fail 帧 + 可判定的非法帧
    hard = sorted(set(list(hard_fail_frame_ids or [])) | set(_illegal_frame_ids(frames)))

    # G3 运动模糊：仅当输入为原始图像时可算；只降权（不判死）
    if frames and not isinstance(frames[0], InputFrame) and len(frames) >= 2:
        try:
            mags = [
                iqa.motion_score(frames[i - 1], frames[i])  # type: ignore[arg-type]
                for i in range(1, len(frames))
            ]
            if float(np.mean(mags)) > TH_MOTION:
                all_flags.append("motion")
        except Exception:  # noqa: BLE001 - 非法帧已在 hard 里标出，运动量算不出就不算
            all_flags.append("motion_unknown")

    weights = [quality_weight_for(fl) for fl in per_frame_flags]
    overall_weight = float(np.mean(weights)) if weights else 0.0
    if "motion" in all_flags:
        overall_weight = min(overall_weight, DEGRADED_FRAME_WEIGHT)
    # §4 M2 字段 6：quality_score = 无劣化 flag 的帧占比（与 quality_weight 口径不同：
    # 前者数"多少帧干净"，后者是"加权后的整体置信"）
    clean_ratio = (1.0 - len(degraded) / len(frames)) if frames else 0.0
    if "motion" in all_flags:
        clean_ratio = min(clean_ratio, 1.0 - TH_DEGRADED_RATIO)
    common = dict(
        degraded_frame_ids=degraded,
        degradation_flags=sorted(set(all_flags)),
        quality_score=float(clean_ratio),
        quality_weight=overall_weight,
        hard_fail_frame_ids=hard,
        n_frames=len(frames),
        frame_set_hash=(frame_set.frame_set_hash if frame_set is not None else ""),
    )

    # G4 帧数完整性 + 输入合法性 → 整体不可用（唯一 hard fail 通道）
    # 先于任何质量统计判定：非法帧连 IQA 都算不出来（空帧/尺寸非法/损坏）
    if len(frames) < MIN_FRAMES or hard:
        return InputGateVerdict(level="overall_unusable", action="unanswerable", **common)

    # 劣化帧占比超阈 → 记 locally_degraded（**只是标签**：帧集不变、不删不补，
    # 权重已按 DEGRADED_FRAME_WEIGHT 下调，供 route 判定与消融使用，硬约束 21）
    if frames and degraded and len(degraded) / len(frames) > TH_DEGRADED_RATIO:
        return InputGateVerdict(level="locally_degraded", action="proceed", **common)
    if degraded:
        return InputGateVerdict(level="pass", action="proceed", **common)
    return InputGateVerdict(level="pass", action="proceed", **common)


def annotate_frames(frames: Sequence[InputFrame],
                    verdict: InputGateVerdict,
                    pixels: Optional[Sequence[np.ndarray]] = None) -> list[InputFrame]:
    """把 M2 观测写回 InputFrame 记录（M1 只给占位，§4 M1 字段 7 注）。

    有 `pixels` 时补算真实 IQA 统计；只写统计/flag/权重字段，
    **返回的帧序与帧数不变**（硬约束 21：不删/不换/不补/不重排）。
    """
    if pixels is not None and len(pixels) != len(frames):
        raise ValueError(
            f"帧集与像素数量不一致（{len(frames)} vs {len(pixels)}）："
            "硬约束 21 禁止帧集变更，M2 不得删/补帧"
        )
    blurs = ([frame_quality(p)[0] for p in pixels] if pixels is not None
             else [f.blur_var for f in frames])
    floor = blur_floor_for(blurs)
    out: list[InputFrame] = []
    for i, fr in enumerate(frames):
        if pixels is not None:
            blur, p_over, p_under = frame_quality(pixels[i])
        else:
            blur, p_over, p_under = fr.blur_var, fr.overexposed_ratio, fr.underexposed_ratio
        fl = degradation_flags(blur, p_over, p_under, floor)
        out.append(fr.model_copy(update={
            "blur_var": float(blur),
            "overexposed_ratio": float(p_over),
            "underexposed_ratio": float(p_under),
            "degradation_flags": fl,
            "quality_ok": not fl,
            "quality_weight": quality_weight_for(fl),
        }))
    return out


def frame_quality_scores(frames: Sequence[np.ndarray]) -> list[tuple[float, float, float]]:
    """批量算 (blur, p_over, p_under)（供 M4 G1/G2 与审计复用）。"""
    return [frame_quality(f) for f in frames]
