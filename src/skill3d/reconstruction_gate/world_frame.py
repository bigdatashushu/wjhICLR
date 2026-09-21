"""M3 世界系契约估计：由**相机位姿束**估计 `world_up` + `handedness`（v6 §3.1 第②层 /
§5.2 / §7.1 / §9.8 / D5）。

**问题（P3）**：第①层 VGGT 只给相对几何，世界系 = 首帧相机系（`c2w[0]=I`），
**没有重力/竖直方向**。于是"左/右"（`relative_direction_of`）与路线规划在缺"世界
向上"约定时是**病态问题**：同一组 3D 点，换一个"上"方向就换一个左右答案。

**v6 的解法（本模块）**：M3 从相机位姿束估计 `world_up`（单位 3 向量）+ `handedness`，
写进 `ReconstructionArtifact`（§5.2 字段 / D5）；M4 只校验存在性与有限性；
方向/路线类 Tool 在约定缺失或非法时 **fail-closed**（§9.8：抛错/拒答，
**不退回无符号启发式**），见 `direction_of`。

**为什么不能拿场景几何猜"上"**（v5 真实缺陷，2026-09-21 实测 `acd95847c5`）：
v5 曾在缺位姿时用"包围盒最小 extent 轴"当竖直轴，该启发式 ① **无符号**（分不清
上下）；② 把房间形状当成重力代理，在长条形房间里选到了**水平轴** z
（extent 1.818 < 1.865 < 2.282，而相机上方向的主分量是 y）→ `relative_direction`
在错误的平面里算 left/right，四个 `object_rel_direction` 题全错。本模块
**只吃位姿**：签名里没有点云/包围盒/题目入口，场景几何无法泄漏进估计。

**算法（确定性；无随机、无墙钟、无隐式全局状态）**：

1. 相机约定为 OpenCV（`+y_cam` 指向图像下方）→ 逐帧"图像上方"在世界系 =
   `-R_f[:, 1]`，其中 `R_f = c2w[f][:3,:3]`；
2. 位姿束清洗：丢非有限/奇异位姿，丢**完全相同**的位姿（同一视重复计数会虚高置信）；
3. **稳健融合**：**几何中位数**（Weiszfeld 迭代，对少数离群帧稳健）——不用会被
   单帧拉偏的朴素均值；
4. **符号锚定（本模块最关键的正确性风险）**：位姿束里只有"相机上方"，没有任何外部
   重力/IMU 信息，所以符号只能由"**多数帧的相机上方指向天空**"这一采集假设确立
   （majority-hemisphere anchor）。归一化合矢长度 `R = ‖Σ up_f‖/n` 正是该假设可信度
   的直接度量：
   - `R < TH_UP_CANCEL`（两半球势均力敌 → 各帧估计近于相消）→ **符号不可定**：
     `world_up=None`，绝不把近零向量归一化后当答案（宁可 fail-closed）；
   - 否则把融合解锚到多数半球，并复核 `n_align > n_anti`（平票同样判"符号不可定"）。
5. `handedness` 由位姿束**手性**独立判定：`det(R_f) > 0` 为右手系（正常旋转），
   `< 0` 为镜像（左手系），两派平票 → `None`（不猜）。它与 `world_up` 相互独立，
   一起构成 §5.2 的"世界系契约"（`world_frame_status="available"` 要求两者都在）。
6. `status` 三值判定（§7.1 的 world_frame 行）：
   - `available`：一致度 ≥ `TH_UP_CONSISTENCY` **且** roll 可用帧数 ≥
     `TH_MIN_FRAMES_ROLL`（"roll 可用" = 该帧图像上方与融合解夹角 ≤ 60°，即相机
     大致竖直、真的提供了重力信息）；
   - `degraded`：给得出估计但置信低（一致度低 / roll 可用帧太少 / 相机离竖直远 /
     符号不可定）；
   - `unavailable`：位姿缺失、形状非法、非有限或去重后不足 `TH_MIN_FRAMES_UP` 帧
     → `world_up=None` + `handedness=None`（半个约定不许流出）。

   `roll_std_deg` 只作**诊断量**上报，不参与门限：它度量"各帧图像上方与融合解"的
   夹角离散度，与一致度是同一信息的两种刻度（一致度 ≥ 0.9 已经把它的取值压得很小），
   再设一道门只会变成永远不触发的死代码。

**诚实边界**（§10.5 风格，必须显式承认）：

- **一致 ≠ 正确 / 偏置与偏航覆盖**：逐帧"图像上方"只受相机**俯仰 + roll** 影响，
  其横向分量是相机朝向转出来的。于是：
  ① 偏航覆盖足够宽时横向分量相消，俯仰偏置自动抵消（`pitch=40°` 若绕一整圈，融合解
  仍精确竖直）；② 若所有帧以同样姿态俯仰 θ 且**偏航覆盖很窄**，横向分量不相消，
  融合解会整体倾斜 θ 而位姿束**无法自查**——此情形与"世界竖直轴真是水平"在数学上
  不可区分。它只能靠外部参照（IMU/重力、人工约定，或另加模型假设）消除；
  本模块只保证"**一致地**给出约定"并如实报告置信；③ 偏航覆盖宽但俯仰接近 90°
  （相机一直朝地板）时，竖直方向的约束权重 ∝ cos θ 已薄到不可靠，归一化合矢长度会
  塌到 `TH_UP_CANCEL` 以下 → 本模块直接 fail-closed（`world_up=None`），不给薄信息的解。
- 位姿束本身错了（VGGT 姿态崩坏）时，这里给出的 `world_up` 仍然"自洽"但无意义；
  该风险由 M4 的几何主门（§10）负责，不在本模块的可见范围内。
- `roll_std_deg` 度量的是"各帧图像上方与融合解"的夹角离散度：yaw（绕竖直转）不贡献
  该量，故它反映的是"相机是否接近竖直"，而不是相机抖动。

阈值全部 `[TODO_CALIBRATE]`：给出的起始参考值只让链路能跑起来，**必须**在自有场景
上重标后才当门用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

# ---- 版本标识（单一事实源，进 artifact / trace，§5.2/D10）----
WORLD_FRAME_VERSION: str = "world-frame-v6"

# ---- 阈值常量（全部 TODO_CALIBRATE，起始参考值）----
TH_UP_CONSISTENCY: float = 0.9     # TODO_CALIBRATE: 一致度下限（各帧 up 与融合解的平均对齐余弦）
TH_MIN_FRAMES_ROLL: int = 4        # TODO_CALIBRATE: "roll 可用"帧数下限（相机接近竖直、能提供重力信息的帧）
TH_UP_CANCEL: float = 0.25         # TODO_CALIBRATE: 归一化合矢长度下限（低于即"符号不可定"）
TH_MIN_FRAMES_UP: int = 2          # TODO_CALIBRATE: 可融合帧数下限（更少 → unavailable，单帧无法多数投票）
TH_UP_ALIGN_FRAME: float = 0.5     # TODO_CALIBRATE: 单帧"roll 可用"判据（与融合 up 的余弦 ≥ 0.5，即夹角 ≤ 60°）
TH_POSE_DUP_ROT_DEG: float = 0.5   # TODO_CALIBRATE: 位姿去重：旋转差（度）
TH_POSE_DUP_TRANS: float = 1e-3    # TODO_CALIBRATE: 位姿去重：平移差（VGGT 输出按中位距离归一化 → 场景尺度 ≈ 1）
TH_UNIT_TOL: float = 1e-3          # 单位向量容差（与 `schemas/reconstruction.py::_world_up_unit` 口径一致）
TH_FRONT_HALF_ANGLE_DEG: float = 45.0  # TODO_CALIBRATE: front/behind 半角（与 v5 `relative_direction` 口径一致）
TH_HORIZ_EPS: float = 1e-9         # 水平面投影退化阈值（归一化场景尺度下的绝对阈值）

# ---- 方法标识（进 trace / artifact receipt）----
METHOD_POSE_BUNDLE: str = "pose_bundle/image_up+robust_geomedian+majority_anchor"
METHOD_NONE: str = "none"          # 位姿不可用、根本没估（unavailable）

# ---- 内部数值常数（不是可调门限，不参与标定）----
_NUM_EPS: float = 1e-12
_GEOMEDIAN_MAX_ITER: int = 128
_GEOMEDIAN_TOL: float = 1e-12


@dataclass
class WorldFrameEstimate:
    """世界系契约估计结果（对应 §5.2 的 `world_up` / `handedness` / `world_frame_status`）。

    - `world_up`：单位 3 向量（世界系，= "重力来向"，即图像上方向）；`None` = 没给出
      约定 → 上层方向/路线 Tool 必须 fail-closed；
    - `handedness`：`"right"` / `"left"`；`None` = 判不出（不猜）；
    - `status`：`"available"` / `"degraded"` / `"unavailable"`；`available` 时
      `world_up` 与 `handedness` 必然非空（§5.2 不变量）；
    - `up_consistency`：各帧 up 估计与融合解的一致度 ∈[0,1]（带符号的平均对齐余弦；
      符号不可定时退化为归一化合矢长度，此时必然很小）；
    - `n_frames_used`：去重、去非法帧之后**进入/本可进入**融合的帧数；
    - `roll_std_deg`：帧间 roll 离散度（度）；不可定义（< 2 帧或 unavailable）时 `None`；
    - `method` / `version`：估计路径与版本（`unavailable` 时 method = `"none"`）；
    - `note`：人类可读的判定依据（进 trace，便于事后归因）。
    """

    world_up: Optional[list[float]]
    handedness: Optional[str]
    status: str
    up_consistency: float
    n_frames_used: int
    roll_std_deg: Optional[float]
    method: str
    note: str = ""
    version: str = WORLD_FRAME_VERSION


# ------------------------------------------------------------------ 输入清洗 ----

def _as_pose_array(c2w_list: Any) -> Optional[np.ndarray]:
    """把 `c2w_list` 归一为 `(N,4,4)` float64；形状/类型非法 → `None`（**不抛错**）。

    接受：`(N,4,4)`、`(N,3,4)`、`(N,3,3)`（只有旋转），以及单个 `(4,4)` / `(3,4)` /
    `(3,3)`（视作 N=1）。世界系 = 首帧相机系，故 `c2w[0]` 通常是 `I`，但本模块不做
    这个断言（校验属 M4 职责），只要求位姿本身可用。
    """
    if c2w_list is None:
        return None
    try:
        arr = np.asarray(c2w_list, dtype=np.float64)
    except (TypeError, ValueError):   # 参差 list / 非数值元素
        return None
    if arr.size == 0 or not np.issubdtype(arr.dtype, np.floating):
        return None
    if arr.ndim == 3 and arr.shape[1:] in ((4, 4), (3, 4), (3, 3)):
        return arr
    if arr.ndim == 2 and arr.shape in ((4, 4), (3, 4), (3, 3)):
        return arr[None, ...]
    return None


def _rotations_translations_valid(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ `(R (N,3,3), t (N,3), valid (N,) bool)`；valid = 旋转块有限且**非奇异**。

    非奇异是必要条件：`up = -R[:,1]` 只有在 R 是可逆姿态（含镜像）时才有意义。
    只有旋转的输入（`(N,3,3)`）视作 t = 0（此时"位姿去重"由旋转差决定）。
    """
    n = int(c2w.shape[0])
    R = c2w[:, :3, :3]
    t = c2w[:, :3, 3].copy() if c2w.shape[-1] == 4 else np.zeros((n, 3), dtype=np.float64)
    valid = np.all(np.isfinite(R), axis=(1, 2)) & np.all(np.isfinite(t), axis=1)
    det = np.zeros(n, dtype=np.float64)
    if bool(valid.any()):
        det[valid] = np.linalg.det(R[valid])
    valid &= np.isfinite(det) & (np.abs(det) > _NUM_EPS)
    return R, t, valid


def _frame_up_vectors(R: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐帧"图像上方"（世界系）= `-R_f[:, 1]`（OpenCV：`+y_cam` 指向图像下方）。

    返回 `(up (N,3), ok (N,) bool)`；列向量退化（模长 ~ 0）的帧标记为不可用。
    """
    up = -R[:, :, 1]
    norms = np.linalg.norm(up, axis=1)
    ok = valid & np.isfinite(norms) & (norms > _NUM_EPS)
    return up, ok


def _intrinsics_valid_mask(intrinsics: Any, n: int) -> tuple[Optional[np.ndarray], str]:
    """由 `intrinsics` 派生**帧有效性**掩码；返回 `(mask 或 None, note)`。

    `intrinsics` 在本模块**只用于输入校验**：某帧焦距非有限/非正、主点非有限 → 该帧
    不进融合。**它不参与方向估计本身**：世界"上"是纯位姿导出量，给 K 不会更准
    （契约要求签名带它，是为了让"位姿帧与内参帧错位"这类输入错误能被显式发现）。

    形状对不上时**不猜**：忽略该提示并在 note 里说明（不按错位索引丢帧）。K 允许是
    共享的 `(3,3)`（视作所有帧同一内参）。
    """
    if intrinsics is None:
        return None, ""
    try:
        K = np.asarray(intrinsics, dtype=np.float64)
    except (TypeError, ValueError):
        return None, "intrinsics 非数值 → 忽略该校验提示"
    if K.ndim == 2 and K.shape == (3, 3):
        if n <= 0 or not np.all(np.isfinite(K)):
            return None, "intrinsics 含 NaN/Inf → 忽略该校验提示"
        K = np.repeat(K[None, :, :], n, axis=0)
    if K.ndim != 3 or K.shape[1:] != (3, 3) or K.shape[0] != n:
        return None, f"intrinsics 形状 {tuple(K.shape)} 与帧数 {n} 不匹配 → 忽略该校验提示"
    fx, fy = K[:, 0, 0], K[:, 1, 1]
    ok = (np.isfinite(fx) & np.isfinite(fy) & (fx > 0) & (fy > 0)
          & np.all(np.isfinite(K[:, :2, 2]), axis=1))
    return ok, ""


def _rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """两个 3x3 姿态之间的测地角（度）：`θ = arccos((tr(R_aᵀ R_b) − 1)/2)`。

    对镜像矩阵（det = −1）也成立：同手性两两相乘后 `R_aᵀ R_b` 仍是正常旋转。
    """
    m = R_a.T @ R_b
    c = float(np.clip((float(np.trace(m)) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _dedup_pose_mask(R: np.ndarray, t: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """去掉**完全相同**的位姿（保序，保留首个出现）：重复位姿不提供新信息。

    为什么必须去重：32 帧全同的位姿束看着"32 帧一致"，实际只有 1 个观测量，
    直接按 32 帧算会给出 `available` + 一致度 1.0 的**虚高置信**。O(n²)、n=32，
    顺序确定 → 结果确定。
    """
    idx = np.flatnonzero(ok)
    keep = np.zeros(int(len(ok)), dtype=bool)
    for i in idx:
        duplicated = False
        for j in np.flatnonzero(keep):
            if (_rotation_angle_deg(R[int(i)], R[int(j)]) <= TH_POSE_DUP_ROT_DEG
                    and float(np.linalg.norm(t[int(i)] - t[int(j)])) <= TH_POSE_DUP_TRANS):
                duplicated = True
                break
        if not duplicated:
            keep[int(i)] = True
    return keep


def _geometric_median(vectors: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """单位向量的**几何中位数**（chordal）：Weiszfeld 迭代，起点固定 → 确定性。

    相对朴素均值的意义：单帧离群（例如某一帧位姿抖了 40°）不会把解拉走；
    点集近于对称（各方向相消）时结果趋近零点，调用方据此判"符号不可定"。
    """
    x = np.array(x0, dtype=np.float64)
    for _ in range(_GEOMEDIAN_MAX_ITER):
        d = np.linalg.norm(vectors - x, axis=1)
        j = int(np.argmin(d))
        if float(d[j]) <= _NUM_EPS:          # 迭代点落在样本上 → 该点即中位数（避免除零）
            return vectors[j].copy()
        w = 1.0 / d
        x_new = (vectors * w[:, None]).sum(axis=0) / float(w.sum())
        if float(np.linalg.norm(x_new - x)) <= _GEOMEDIAN_TOL * max(1.0, float(np.linalg.norm(x))):
            return x_new
        x = x_new
    return x


# ------------------------------------------------------------------ 公开接口 ----

def handedness_of(c2w_list: Any) -> Optional[str]:
    """由位姿束判定世界系**手性**：`"right"` / `"left"` / `None`（判不出，不猜）。

    判据：`det(R_f) > 0` = 正常旋转（右手系），`det(R_f) < 0` = 镜像（左手系，
    例如位姿被镜像重建出来）。按帧**多数**决定：两派平票（或没有可用帧）→ `None`。

    为什么要显式报它（§5.2/D5）：`(f × d)·u` 的符号依赖坐标系手性。坐标系若是
    镜像的，"物理上的右手边"对应叉积的**反号**；不报手性就等于偷偷假定了一种，
    正是 v5 固定 `f_x·d_z − f_z·d_x` 所犯的错。`direction_of` 因此要求显式传入。
    """
    c2w = _as_pose_array(c2w_list)
    if c2w is None:
        return None
    R, _, valid = _rotations_translations_valid(c2w)
    if not bool(valid.any()):
        return None
    det = np.linalg.det(R[valid])
    n_pos = int(np.sum(det > _NUM_EPS))
    n_neg = int(np.sum(det < -_NUM_EPS))
    if n_pos > n_neg:
        return "right"
    if n_neg > n_pos:
        return "left"
    return None


def estimate_world_frame(c2w_list: Any, *, intrinsics: Any = None) -> WorldFrameEstimate:
    """由相机位姿束估计世界系契约（§3.1 第②层 / §5.2 / D5）。

    `c2w_list`：(N,4,4) camera→world（世界系 = 首帧相机系，`c2w[0]=I`）；也接受
    (N,3,4)/(N,3,3) 与单个 (4,4)。`intrinsics`：可选，**只用于帧有效性校验**
    （见 `_intrinsics_valid_mask`），不参与方向估计。

    返回 `WorldFrameEstimate`（见其 docstring）。要点：

    - `world_up` 只由位姿导出（`-R_f[:,1]` 的稳健融合 + 多数半球符号锚定），
      **不碰**点云/包围盒/题目（v5 `acd95847c5` 缺陷的根因就在这里）；
    - 一致的帧里若两半球势均力敌（合矢长度 < `TH_UP_CANCEL`）→ 符号不可定 →
      `world_up=None`（`degraded`），**绝不**把近零向量归一化后当答案；
    - 位姿不可用（None/形状非法/非有限/去重后 < `TH_MIN_FRAMES_UP` 帧）→
      `unavailable` 且 `world_up=None`、`handedness=None`（半个约定不许流出）。

    确定性：同输入（含帧序）两次调用逐位一致；无随机、无墙钟、无全局状态。
    """
    c2w = _as_pose_array(c2w_list)
    if c2w is None:
        return WorldFrameEstimate(
            world_up=None, handedness=None, status="unavailable", up_consistency=0.0,
            n_frames_used=0, roll_std_deg=None, method=METHOD_NONE,
            note="位姿束缺失或形状非法（期望 (N,4,4)）→ 世界系契约不可给（不猜）")

    n_in = int(c2w.shape[0])
    R, t, valid = _rotations_translations_valid(c2w)
    up_raw, up_ok = _frame_up_vectors(R, valid)

    k_mask, k_note = _intrinsics_valid_mask(intrinsics, n_in)
    if k_mask is not None:
        up_ok = up_ok & k_mask

    n_valid = int(up_ok.sum())
    if n_valid == 0:
        why = "非有限/奇异位姿" + ("或被内参校验剔除" if k_mask is not None else "")
        return WorldFrameEstimate(
            world_up=None, handedness=None, status="unavailable", up_consistency=0.0,
            n_frames_used=0, roll_std_deg=None, method=METHOD_NONE,
            note=(f"{n_in} 帧全部不可用（{why}）→ 世界系契约不可给（不猜）"
                  + (f"；{k_note}" if k_note else "")))

    keep = _dedup_pose_mask(R, t, up_ok)
    n_used = int(keep.sum())
    n_dedup = n_valid - n_used
    hand = handedness_of(c2w)

    if n_used < TH_MIN_FRAMES_UP:
        return WorldFrameEstimate(
            world_up=None, handedness=None, status="unavailable", up_consistency=0.0,
            n_frames_used=n_used, roll_std_deg=None, method=METHOD_NONE,
            note=(f"去重后可用位姿仅 {n_used} 帧（< {TH_MIN_FRAMES_UP}）→ 无多视冗余，"
                  "世界系契约不可给（不猜）"))

    ups = up_raw[keep]
    summed = ups.sum(axis=0)
    resultant = float(np.clip(np.linalg.norm(summed) / n_used, 0.0, 1.0))

    # ---- 符号不可定：两半球势均力敌 / 各帧估计近于相消 ----
    if not np.isfinite(resultant) or resultant < TH_UP_CANCEL:
        return WorldFrameEstimate(
            world_up=None, handedness=hand, status="degraded", up_consistency=resultant,
            n_frames_used=n_used, roll_std_deg=None, method=METHOD_POSE_BUNDLE,
            note=(f"各帧 up 估计近于相消（归一化合矢长度 {resultant:.3f} < {TH_UP_CANCEL}）"
                  "→ 世界“上”的符号不可定，按 §9.8 fail-closed 返回 world_up=None"
                  "（不把近零向量归一化后当答案）"))

    geom = _geometric_median(ups, summed / n_used)
    geom_norm = float(np.linalg.norm(geom))
    if not np.isfinite(geom_norm) or geom_norm <= _NUM_EPS:
        return WorldFrameEstimate(
            world_up=None, handedness=hand, status="degraded", up_consistency=resultant,
            n_frames_used=n_used, roll_std_deg=None, method=METHOD_POSE_BUNDLE,
            note=("稳健融合解近于零点（各帧 up 估计对称相消）→ 符号不可定，"
                  "按 §9.8 fail-closed 返回 world_up=None"))
    u = geom / geom_norm

    # ---- 符号锚定复核：融合解必须落在**多数半球**，平票同样判"符号不可定" ----
    dots = ups @ u
    n_align = int(np.sum(dots > _NUM_EPS))
    n_anti = int(np.sum(dots < -_NUM_EPS))
    if n_anti > n_align:
        u = -u
        dots = -dots
        n_align, n_anti = n_anti, n_align
    if n_align == n_anti:
        return WorldFrameEstimate(
            world_up=None, handedness=hand, status="degraded", up_consistency=resultant,
            n_frames_used=n_used, roll_std_deg=None, method=METHOD_POSE_BUNDLE,
            note=(f"半球投票平票（对齐 {n_align} : 反向 {n_anti}）→ 世界“上”的符号"
                  "不可定，按 §9.8 fail-closed 返回 world_up=None"))

    consistency = float(np.clip(dots.mean(), 0.0, 1.0))
    roll_deg = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    roll_std = float(np.std(roll_deg)) if n_used >= 2 else None
    n_roll_ok = int(np.sum(dots >= TH_UP_ALIGN_FRAME))

    reasons: list[str] = []
    if n_in - n_valid > 0:
        reasons.append(f"跳过不可用位姿 {n_in - n_valid} 帧")
    if n_dedup > 0:
        reasons.append(f"去重完全重复位姿 {n_dedup} 帧（不虚高置信）")
    if k_note:
        reasons.append(k_note)
    if consistency < TH_UP_CONSISTENCY:
        reasons.append(f"一致度 {consistency:.3f} < {TH_UP_CONSISTENCY}")
    if n_roll_ok < TH_MIN_FRAMES_ROLL:
        reasons.append(f"roll 可用帧 {n_roll_ok} < {TH_MIN_FRAMES_ROLL}"
                       f"（相机离竖直太远 → 该帧几乎不提供重力信息）")

    available = consistency >= TH_UP_CONSISTENCY and n_roll_ok >= TH_MIN_FRAMES_ROLL
    status = "available" if available else "degraded"
    summary = (f"{n_used} 帧稳健融合（一致度 {consistency:.3f}，roll 可用 {n_roll_ok}/{n_used} 帧，"
               f"roll 离散度 {'n/a' if roll_std is None else f'{roll_std:.1f}°'}，"
               f"手性 {hand}）")
    note = summary + ("；" + "；".join(reasons) if reasons else "")

    return WorldFrameEstimate(
        world_up=[float(x) for x in u], handedness=hand, status=status,
        up_consistency=consistency, n_frames_used=n_used, roll_std_deg=roll_std,
        method=METHOD_POSE_BUNDLE, note=note)


# ------------------------------------------------------------------ 方向判定 ----

def _as_vec3(value: Any, name: str) -> np.ndarray:
    """校验成有限 3 向量；否则抛 `ValueError`（fail-closed，不静默补值）。"""
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 不是数值 3 向量（收到 {value!r}）") from None
    if arr.shape != (3,):
        raise ValueError(f"{name} 必须是 3 向量（收到 {value!r}）")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} 含 NaN/Inf（收到 {value!r}）")
    return arr


def _validated_world_up(world_up: Any) -> np.ndarray:
    """校验 `world_up`：非 None、有限、**单位**；不合法一律抛 `ValueError`。

    与 `schemas/reconstruction.py::ReconstructionArtifact._world_up_unit` 同口径
    （容差 `TH_UNIT_TOL`）：**不做静默归一化**——非单位向量说明上游口径已错，
    猜一个单位向量等于把错误藏起来（§9.8 硬合同）。
    """
    if world_up is None:
        raise ValueError(
            "world_up 缺失（None）→ 方向判定 fail-closed（§9.8：不退回无符号启发式）；"
            "请先由 M3 落盘世界系契约（estimate_world_frame）")
    arr = _as_vec3(world_up, "world_up")
    n = float(np.linalg.norm(arr))
    if not np.isfinite(n) or abs(n - 1.0) > TH_UNIT_TOL:
        raise ValueError(
            f"world_up 必须是**单位**向量（模长={n}，收到 {world_up!r}）→ fail-closed，不做静默归一化")
    return arr / n


def _validated_handedness(handedness: Any) -> str:
    """校验 `handedness` ∈ {"right", "left"}；缺失/未知 → 抛 `ValueError`。"""
    if handedness not in ("right", "left"):
        raise ValueError(
            f"handedness 必须是 'right'/'left'（收到 {handedness!r}）→ 方向判定 fail-closed"
            "（叉积符号依赖坐标系手性，缺它就只能猜）")
    return str(handedness)


def direction_of(observer_xyz: Any, facing_xyz: Any, target_xyz: Any, world_up: Any, *,
                 handedness: Any = "right") -> str:
    """在**水平面**内判定 target 相对 observer/facing 的方位（§9.8 硬合同）。

    返回 `"front" | "behind" | "left" | "right"`。

    几何（与 §9.8 一致）：
    - `f = facing_xyz − observer_xyz`：**facing_xyz 是位置**（`facing_at` 对象的位置，
      不是方向向量），facing 向量由两点之差得到；
    - `d = target_xyz − observer_xyz`；`u = world_up`（单位向量）；
    - 先投影到**垂直于 u 的水平面**：`f_h = f − (f·u)u`、`d_h = d − (d·u)u`
      （对倾斜相机/倾斜世界系同样成立，竖直分量不许混进左右判定）。

    判据（显式、可复现）：
    - `front`：`f_h` 与 `d_h` 夹角 ≤ `TH_FRONT_HALF_ANGLE_DEG`（默认 45°，即
      `cos ≥ 0.7071`：方向大致同向且横向分量小）；
    - `behind`：夹角 ≥ 180° − 上述半角（要转身 ≥ 135°，与 v5 口径一致）；
    - 其余（横向带）→ 左右，**右手系**：`right ⟺ (f_h × d_h)·u < 0`
      （即 `right = forward × up`）。因 `f_h ⟂ u`、`d_h ⟂ u`，叉积必与 `u` 平行，
      故该判定无需额外角度容差。
    - **左手系（`handedness="left"`）**：镜像世界里的叉积与物理右手定则反号，
      同一个几何下"物理右侧"对应 `(f_h × d_h)·u > 0`，故判据取反（左右互换）。

    fail-closed（§9.8，本函数的**硬合同**）：以下情形一律抛 `ValueError`，绝不返回
    猜测结果、也绝不退回"无符号启发式"（例如拿包围盒最小 extent 轴当竖直轴）：
    ① `world_up` 为 None / 非有限 / **非单位**；② `handedness` 缺失或不是
    `"right"`/`"left"`；③ 任一点位非有限或不是 3 向量；④ 水平面内退化
    （facing 与其竖直方向平行、target 与 observer 重合或只在正上/正下方）。
    """
    u = _validated_world_up(world_up)
    hand = _validated_handedness(handedness)
    obs = _as_vec3(observer_xyz, "observer_xyz")
    fac = _as_vec3(facing_xyz, "facing_xyz")
    tgt = _as_vec3(target_xyz, "target_xyz")

    f_vec = fac - obs
    d_vec = tgt - obs
    f_h = f_vec - float(np.dot(f_vec, u)) * u
    d_h = d_vec - float(np.dot(d_vec, u)) * u
    f_norm = float(np.linalg.norm(f_h))
    d_norm = float(np.linalg.norm(d_h))
    if not np.isfinite(f_norm) or f_norm <= TH_HORIZ_EPS:
        raise ValueError(
            "facing 向量在水平面上退化为零（facing_at 与 observer 只差竖直方向）"
            "→ 方位未定义，fail-closed")
    if not np.isfinite(d_norm) or d_norm <= TH_HORIZ_EPS:
        raise ValueError(
            "target 与 observer 重合或只差竖直方向 → 水平方位未定义，fail-closed")
    f_hat = f_h / f_norm
    d_hat = d_h / d_norm

    cos_theta = float(np.clip(np.dot(f_hat, d_hat), -1.0, 1.0))
    cos_half = float(np.cos(np.radians(TH_FRONT_HALF_ANGLE_DEG)))
    if cos_theta >= cos_half:
        return "front"
    if cos_theta <= -cos_half:
        return "behind"

    triple = float(np.dot(np.cross(f_hat, d_hat), u))
    if hand == "right":
        return "right" if triple < 0.0 else "left"
    # 左手系：坐标系本身是镜像的 → 同一个几何下"物理右侧"对应 triple > 0（左右互换）
    return "right" if triple > 0.0 else "left"
