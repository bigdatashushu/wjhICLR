"""世界系契约估计单测（v6 §3.1 第②层 / §5.2 / §7.1 / §9.8 / D5）。

覆盖：
1. 合成位姿束（水平看向已知"上"）→ 恢复的 up ≈ 期望、status=available；
2. **Phase 1 DoD 负向测试**：位姿束绕**水平轴**翻转 180°（世界倒置）→ 固定三元组的
   方向答案必须**相应翻转**（left↔right）；
3. 退化输入：全同位姿、单帧、up 估计相消 → `unavailable`/`degraded`，**绝不**给出
   一个来路不明的单位向量；
4. `direction_of` fail-closed：world_up 为 None / 非单位 / 非有限，handedness 缺失或
   未知 → 抛 `ValueError`；水平面退化同样抛错；
5. 确定性：同输入两次调用逐位一致；
6. v5 真实缺陷回归（`acd95847c5`）：长条形房间里"最小 extent 轴"是水平轴 z，
   但估计只由位姿决定，喂进误导性的场景几何不会改变任何东西。
"""

import inspect

import numpy as np
import pytest

from skill3d.reconstruction_gate import world_frame as wf
from skill3d.reconstruction_gate.world_frame import (
    TH_MIN_FRAMES_UP,
    TH_UP_CONSISTENCY,
    WORLD_FRAME_VERSION,
    WorldFrameEstimate,
    direction_of,
    estimate_world_frame,
    handedness_of,
)

# ------------------------------------------------------------------ 合成工具 ----

# 世界系 = 首帧相机系（`c2w[0]=I`），相机约定 OpenCV：看 +z、图像上方 = −y。


def _rx(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _ry(deg: float) -> np.ndarray:
    """绕世界**竖直轴**（y）转 = 相机转头/转向：不改变图像上方（故不影响 up 估计）。"""
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _rz(deg: float) -> np.ndarray:
    """绕相机光轴（z）转 = roll：把图像上方从竖直方向掰开。"""
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _pose(turn_deg: float, pitch_deg: float = 0.0, roll_deg: float = 0.0,
          t: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """构造一帧 c2w：先按相机系内的 pitch/roll 摆姿态，再绕世界竖直轴转头、平移。"""
    R = _ry(turn_deg) @ _rx(pitch_deg) @ _rz(roll_deg)
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = np.asarray(t, dtype=float)
    return c2w


def _level_bundle(n: int = 8) -> list[np.ndarray]:
    """n 帧"相机竖直、水平看"的位姿束：up 应恢复为 −y，且每帧都 roll 可用。"""
    return [_pose(turn_deg=-70.0 + 20.0 * i, pitch_deg=0.0,
                  t=(0.5 * i, 0.1 * i, 0.25 * i)) for i in range(n)]


def _flip_about_horizontal_axis(c2w: np.ndarray) -> np.ndarray:
    """绕**相机系 x 轴**（相机水平时它是世界水平轴）转 180° = 把相机拿成大倒个。

    右乘 = 相机系内旋转：世界点不动，只有相机的"图像上方"反向。
    """
    out = c2w.copy()
    out[:3, :3] = c2w[:3, :3] @ _rx(180.0)
    return out


def _rot180_world_x(c2w: np.ndarray) -> np.ndarray:
    """绕**世界 x 轴**（水平轴）转 180°：整个位姿束翻过来（世界倒置，手性不变）。"""
    F = np.eye(4)
    F[:3, :3] = _rx(180.0)
    return F @ c2w


# ------------------------------------------------------- 1. 正常位姿束 → 可估计 ----

def test_level_bundle_recovers_world_up_and_is_available():
    """相机竖直、多方向转头 → up ≈ −y、status=available、字段自洽（§5.2）。"""
    est = estimate_world_frame(_level_bundle(8))
    assert isinstance(est, WorldFrameEstimate)
    assert est.status == "available"
    assert est.world_up == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)
    assert est.handedness == "right"
    assert est.up_consistency == pytest.approx(1.0, abs=1e-9)
    assert est.n_frames_used == 8
    assert est.roll_std_deg == pytest.approx(0.0, abs=1e-9)
    assert est.method == wf.METHOD_POSE_BUNDLE
    assert est.version == WORLD_FRAME_VERSION
    # world_up 必须是可落盘的 list[float] 且**单位**（schema 校验口径，§5.2）
    assert isinstance(est.world_up, list)
    assert all(isinstance(x, float) for x in est.world_up)
    assert np.linalg.norm(est.world_up) == pytest.approx(1.0, abs=1e-12)
    # §5.2 不变量：available 必须两个约定都在
    assert est.world_up is not None and est.handedness is not None


def test_tilted_bundle_up_stays_vertical_and_confidence_drops_honestly():
    """带俯仰/roll 的位姿束：融合解仍接近竖直，但一致度如实下降（不虚报 1.0）。"""
    bundle = [_pose(turn_deg=-60.0 + 30.0 * i, pitch_deg=(-12.0 if i % 2 else 12.0),
                    roll_deg=(5.0 if i % 2 else -5.0), t=(0.3 * i, 0.0, 0.2 * i))
              for i in range(9)]
    est = estimate_world_frame(bundle)
    assert est.world_up is not None
    assert abs(est.world_up[1]) > 0.95              # 主分量仍是竖直轴 y
    assert 0.9 <= est.up_consistency < 1.0          # 如实反映"有姿态偏斜"
    assert est.status == "available"
    assert est.roll_std_deg is not None and est.roll_std_deg > 0.0


def test_estimator_uses_only_poses_not_scene_geometry():
    """v5 真实缺陷回归（`acd95847c5`）：包围盒最小 extent 轴是**水平轴** z，不影响估计。

    v5 用"最小 extent 轴"当竖直轴 → 在长条形房间里选到 z（1.818 < 1.865 < 2.282），
    而相机上方向的主分量是 y → 四个 rel_direction 题全错。本模块只吃位姿。
    """
    est = estimate_world_frame(_level_bundle(8))
    assert est.world_up == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)
    # 复刻缺陷：长条形房间的 bbox extent（x/y/z），最小的是 z
    extents = np.array([2.282, 1.865, 1.818])
    assert int(np.argmin(extents)) == 2                 # 启发式会选 z（水平轴）→ 缺陷本身
    assert abs(est.world_up[1]) > 0.999                 # 估计给出的竖直轴是 y → 无关场景形状
    # 上游签名里根本没有点云/包围盒入口 → 场景几何**无法**泄漏进估计
    params = set(inspect.signature(estimate_world_frame).parameters)
    assert params == {"c2w_list", "intrinsics"}


# ------------------------------------- 2. Phase 1 DoD 负向测试：翻转 → 答案翻转 ----

def test_flip_pose_bundle_about_horizontal_axis_flips_direction_answer():
    """Phase 1 DoD：位姿绕水平轴翻转 180° → 世界倒置 → 方向答案必须翻转（left↔right）。

    固定三元组（observer / facing_at / target）不变，只把位姿束翻转：
    两种等价写法各测一遍 ——（a）绕世界 x（水平）轴翻转整束；（b）每帧绕相机自身
    x（水平）轴翻转（= 把相机拿成大倒个）。
    """
    obs, facing_at, target = (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (2.0, 0.0, 0.0)
    upright = _level_bundle(8)

    est_up = estimate_world_frame(upright)
    assert est_up.status == "available"
    assert est_up.world_up == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)
    ans_up = direction_of(obs, facing_at, target, est_up.world_up,
                          handedness=est_up.handedness)
    assert ans_up == "right"

    # (a) 绕世界 x（水平）轴翻转整束：世界"上"变 +y（倒置），手性不变（正常旋转）
    est_a = estimate_world_frame([_rot180_world_x(c) for c in upright])
    assert est_a.status == "available"
    assert est_a.world_up == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)
    assert est_a.handedness == "right"
    ans_a = direction_of(obs, facing_at, target, est_a.world_up, handedness=est_a.handedness)
    assert ans_a == "left"

    # (b) 每帧绕相机 x（水平）轴 180° = 倒置相机拍摄：估计的"上"同样反向 → 答案同样翻转
    est_b = estimate_world_frame([_flip_about_horizontal_axis(c) for c in upright])
    assert est_b.status == "available"
    assert est_b.world_up == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)
    ans_b = direction_of(obs, facing_at, target, est_b.world_up, handedness=est_b.handedness)
    assert ans_b == "left"

    # 硬断言：翻转前后必须是同一对答案的互换，而不是"变成别的类别"
    assert (ans_up, ans_a, ans_b) == ("right", "left", "left")
    assert ans_up != ans_a


def test_front_back_are_invariant_under_world_flip_while_lateral_swaps():
    """世界倒置只互换左右：前后分区不受影响（判据与手性/上方向无关）。

    v7 §9.3：`front`/`back` 只在 **hard** 四象限模板里作为前缀出现；
    medium 模板的选项集合是 left/right/back，故这里用 hard 断言前后分区。
    """
    upright, flipped = _level_bundle(8), [_rot180_world_x(c) for c in _level_bundle(8)]
    u1 = estimate_world_frame(upright).world_up
    u2 = estimate_world_frame(flipped).world_up
    args = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    for u in (u1, u2):
        assert direction_of(*args, (0.0, 0.0, 2.0), u, difficulty="hard").startswith("front")
        assert direction_of(*args, (0.0, 0.0, -2.0), u, difficulty="hard").startswith("back")
        # medium：正前（0°）判左右（左右未定义 → 只要求落在合法选项内且确定）、
        # 正后（180°）判 back —— 与模板选项集合一致
        assert direction_of(*args, (0.0, 0.0, 2.0), u, difficulty="medium") in ("left", "right")
        assert direction_of(*args, (0.0, 0.0, -2.0), u, difficulty="medium") == "back"


# ------------------------------------------------------------- 3. 退化输入处理 ----

@pytest.mark.parametrize("bad", [None, [], np.zeros((0, 4, 4)), "not-a-pose",
                                 [1.0, 2.0, 3.0], np.zeros((2, 2, 2)),
                                 [[[1.0, 0.0], [0.0, 1.0]]]])
def test_unusable_pose_input_is_unavailable(bad):
    """位姿缺失/形状非法 → unavailable，且**两个约定都是 None**（不猜）。"""
    est = estimate_world_frame(bad)
    assert est.status == "unavailable"
    assert est.world_up is None and est.handedness is None
    assert est.n_frames_used == 0
    assert est.up_consistency == 0.0
    assert est.roll_std_deg is None
    assert est.method == wf.METHOD_NONE
    assert est.version == WORLD_FRAME_VERSION


def test_single_frame_and_all_identity_are_not_available():
    """单帧 / 全同位姿束：没有多视冗余 → 不得给 available，更不能编一个单位向量。"""
    single = estimate_world_frame([np.eye(4)])
    assert single.status == "unavailable"
    assert single.world_up is None and single.handedness is None
    assert single.n_frames_used < TH_MIN_FRAMES_UP

    # 32 帧完全相同的位姿（零基线退化束）：去重后只剩 1 个位姿 → 不得给 available
    same = estimate_world_frame([np.eye(4)] * 32)
    assert same.status == "unavailable"
    assert same.world_up is None
    assert same.n_frames_used == 1
    assert "去重" in same.note or "冗余" in same.note

    # 单个 (4,4) 裸数组同样按 N=1 处理
    assert estimate_world_frame(np.eye(4)).status == "unavailable"


def test_cancelling_up_estimates_report_degraded_and_no_bogus_vector():
    """一半帧竖直、一半帧倒置 → 各帧 up 估计相消、符号不可定 → degraded + world_up=None。"""
    half = 16
    bundle = ([_pose(turn_deg=-75.0 + 10.0 * i, t=(0.4 * i, 0.0, 0.3 * i)) for i in range(half)]
              + [_flip_about_horizontal_axis(_pose(turn_deg=-75.0 + 10.0 * i,
                                                   t=(0.4 * i, 0.0, 0.3 * i)))
                 for i in range(half)])
    est = estimate_world_frame(bundle)
    assert est.status == "degraded"                    # 位姿可用，但符号不可定
    assert est.world_up is None                        # 绝不把近零向量归一化后当答案
    assert est.up_consistency < wf.TH_UP_CANCEL        # 一致度如实反映"相消"
    assert est.n_frames_used == 2 * half               # 32 个位姿互不相同（没被去重吃掉）
    assert est.handedness == "right"                   # 手性与"上"的符号无关，仍可报
    assert "符号不可定" in est.note


def test_strong_majority_upside_down_bundle_is_signed_by_majority_anchor():
    """多数帧倒置（24 倒 8 正）→ 符号锚到多数半球（这是唯一的可行锚），一致度如实偏低。"""
    n_up, n_down = 8, 24
    bundle = ([_pose(turn_deg=-70.0 + 20.0 * i, t=(0.4 * i, 0.0, 0.0)) for i in range(n_up)]
              + [_flip_about_horizontal_axis(_pose(turn_deg=-70.0 + 6.0 * i,
                                                   t=(0.4 * i, 0.0, 0.0)))
                 for i in range(n_down)])
    est = estimate_world_frame(bundle)
    assert est.world_up == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)   # 锚到多数半球
    assert est.status == "degraded"                                   # 一致度 (24−8)/32 = 0.5
    assert est.up_consistency == pytest.approx(0.5, abs=1e-9)


def test_non_finite_frames_are_skipped_not_fatal():
    """少数帧含 NaN/Inf → 跳过该帧，其余帧照常给出（并在 note 里说明）。"""
    bundle = _level_bundle(8)
    bad = bundle[0].copy()
    bad[0, 3] = np.nan                                  # 平移列非有限
    bad2 = bundle[1].copy()
    bad2[:3, :3] = np.nan                               # 旋转块非有限
    singular = bundle[2].copy()
    singular[:3, :3] = np.zeros((3, 3))                 # 奇异（det=0）姿态
    est = estimate_world_frame([bad, bad2, singular] + bundle[3:])
    assert est.status == "available"
    assert est.world_up == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)
    assert est.n_frames_used == len(bundle) - 3
    assert "跳过不可用位姿 3 帧" in est.note


def test_intrinsics_only_gate_frame_validity_and_do_not_change_the_direction():
    """K 只做帧有效性校验：焦距非正的帧被剔除；K 不给/给全都不改变估计方向。"""
    bundle = _level_bundle(8)
    base = estimate_world_frame(bundle)
    K = np.repeat(np.eye(3)[None, :, :], 8, axis=0)
    K[:, 0, 0] = K[:, 1, 1] = 500.0
    assert estimate_world_frame(bundle, intrinsics=K).world_up == base.world_up
    assert estimate_world_frame(bundle, intrinsics=K).n_frames_used == 8

    K_bad = K.copy()
    K_bad[0, 0, 0] = -500.0                             # 焦距非正 → 该帧不可用
    est = estimate_world_frame(bundle, intrinsics=K_bad)
    assert est.n_frames_used == 7
    assert est.world_up == base.world_up                # 方向不受 K 影响

    # 形状与帧数不匹配 → 不按错位索引丢帧，忽略提示并说明（不猜）
    est_mismatch = estimate_world_frame(bundle, intrinsics=np.repeat(np.eye(3)[None, :, :], 3, 0))
    assert est_mismatch.n_frames_used == 8
    assert "intrinsics" in est_mismatch.note

    # 内参把所有帧都判废 → unavailable（不许硬撑）
    est_allbad = estimate_world_frame(bundle, intrinsics=np.zeros((8, 3, 3)))
    assert est_allbad.status == "unavailable" and est_allbad.world_up is None


# ----------------------------------------------------------- 4. 手性与 fail-closed ----

def test_handedness_of_reports_chirality_explicitly():
    """正常旋转束 → right；镜像束（det<0）→ left；判不出（缺失/平票）→ None。"""
    proper = _level_bundle(6)
    assert handedness_of(proper) == "right"

    base = _pose(turn_deg=20.0)
    mirror = base.copy()
    mirror[:3, :3] = base[:3, :3] @ np.diag([-1.0, 1.0, 1.0])     # det → −1（镜像）
    assert handedness_of([mirror]) == "left"
    assert handedness_of(proper + [mirror]) == "right"             # 6 : 1 → 多数派

    assert handedness_of(None) is None
    assert handedness_of([]) is None
    assert handedness_of("nope") is None
    assert handedness_of([mirror, mirror.copy()]) == "left"
    # 平票 → None（不猜）
    assert handedness_of([proper[0], mirror]) is None


def test_left_handed_world_frame_swaps_left_and_right():
    """手性必须显式影响左右：同一几何、同一 up，左手系下左右互换（§9.8）。"""
    u = [0.0, -1.0, 0.0]
    args = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (2.0, 0.0, 0.0))
    assert direction_of(*args, u, handedness="right") == "right"
    assert direction_of(*args, u, handedness="left") == "left"
    # 前后分区与手性无关（hard 模板下看 front-/back- 前缀）
    assert direction_of((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 3.0), u,
                        handedness="left", difficulty="hard").startswith("front-")
    # medium 只有三个选项：正前（0°）落进左右，绝不能返回 "front"
    assert direction_of((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 3.0), u,
                        handedness="left", difficulty="medium") in ("left", "right")


@pytest.mark.parametrize("bad_up", [
    None, [0.0, 0.0, 2.0], [1.0, 1.0, 0.0], [0.0, 1.0], [0.0, 0.0, 0.0],
    [float("nan"), 0.0, 0.0], [0.0, float("inf"), 0.0], "up",
])
def test_direction_of_fails_closed_on_bad_world_up(bad_up):
    """world_up 缺失/非有限/非单位 → 抛 ValueError（§9.8：不退回无符号启发式）。"""
    with pytest.raises(ValueError):
        direction_of((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (2.0, 0.0, 0.0), bad_up)


def test_direction_of_accepts_case_insensitive_difficulty():
    """难度大小写不敏感（模型常写 "Medium"）：归一化后正常求解。"""
    assert direction_of((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (2.0, 0.0, 0.0),
                        [0.0, 1.0, 0.0], difficulty="Medium") == "left"
    assert direction_of((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (2.0, 0.0, 0.0),
                        [0.0, 1.0, 0.0], difficulty="  HARD ") == "back-left"


@pytest.mark.parametrize("bad_diff", [None, "", "impossible", 1, "easyy", "medium-ish"])
def test_direction_of_fails_closed_on_unknown_difficulty(bad_diff):
    """难度未知 → 抛 ValueError（§9.3：选项集合由难度决定，缺它就只能猜）。

    大小写不敏感是**有意**的（`"Medium"` 归一为 `medium`，模型常写成大写）；
    真正的未知取值仍然 fail-closed。
    """
    with pytest.raises(ValueError):
        direction_of((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (2.0, 0.0, 0.0),
                     [0.0, 1.0, 0.0], difficulty=bad_diff)


@pytest.mark.parametrize("bad_hand", [None, "Right", "RIGHT", "unknown", 1, "", "left_handed"])
def test_direction_of_fails_closed_on_unknown_handedness(bad_hand):
    """handedness 缺失/未知 → 抛 ValueError（叉积符号依赖手性，缺它只能猜）。"""
    with pytest.raises(ValueError):
        direction_of((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (2.0, 0.0, 0.0),
                     [0.0, -1.0, 0.0], handedness=bad_hand)


def test_direction_of_fails_closed_on_degenerate_geometry():
    """水平面内退化（facing 竖直 / target 重合或只在正上方）→ 抛错，不猜一个方向。"""
    u = [0.0, -1.0, 0.0]
    # facing 与 observer 只差竖直方向
    with pytest.raises(ValueError):
        direction_of((0.0, 0.0, 0.0), (0.0, -3.0, 0.0), (2.0, 0.0, 0.0), u)
    # target 与 observer 重合
    with pytest.raises(ValueError):
        direction_of((1.0, 0.0, 2.0), (0.0, 0.0, 1.0), (1.0, 0.0, 2.0), u)
    # target 只在 observer 正上方（水平方位未定义）
    with pytest.raises(ValueError):
        direction_of((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -3.0, 0.0), u)
    # 点位本身非法
    with pytest.raises(ValueError):
        direction_of((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (float("nan"), 0.0, 0.0), u)
    with pytest.raises(ValueError):
        direction_of((0.0, 0.0), (0.0, 0.0, 1.0), (2.0, 0.0, 0.0), u)
    # 非退化：target 明显偏上但仍有水平分量 → 正常判定
    assert direction_of((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (2.0, -5.0, 0.0), u) == "right"


def test_direction_of_is_equivariant_to_world_frame_choices():
    """健全性：只改世界约定的表达方式（不真倒置场景）时答案不变 —— 刚性一致。"""
    u = np.array([0.0, -1.0, 0.0])
    obs, facing_at, target = np.array([0.1, 0.2, -0.3]), np.array([0.4, 0.6, 0.9]), np.array([1.5, 0.1, 0.2])
    base = direction_of(obs, facing_at, target, u, handedness="right")
    # 整个场景（点位 + 上方向）绕世界竖直轴转 +90°：答案必须逐字相同
    R = _ry(90.0)
    assert direction_of(R @ obs, R @ facing_at, R @ target, R @ u,
                        handedness="right") == base


# --------------------------------------------------------------- 5. 确定性 ----

def test_estimate_is_deterministic():
    """同输入两次调用逐位一致（无随机、无墙钟、无全局状态）。"""
    bundle = [_pose(turn_deg=-55.0 + 17.0 * i, pitch_deg=(-9.0 if i % 3 else 6.0),
                    roll_deg=(4.0 if i % 4 else -3.0), t=(0.31 * i, 0.07 * i, 0.19 * i))
              for i in range(9)]
    e1 = estimate_world_frame(bundle)
    e2 = estimate_world_frame(bundle)
    assert e1 == e2                                   # dataclass 逐字段相等
    assert e1.world_up == e2.world_up                 # 逐位（不是 approx）
    assert e1.up_consistency == e2.up_consistency
    assert e1.roll_std_deg == e2.roll_std_deg
    assert isinstance(e1.world_up, list)
    # 帧序互换（同一集合）→ 解不变（融合是集合级运算，不依赖扫描顺序）
    e3 = estimate_world_frame(list(reversed(bundle)))
    assert e3.world_up == pytest.approx(e1.world_up, abs=1e-9)


def test_wide_roll_dispersion_degrades_status():
    """相机 roll 离散度大（多数正立、少数侧放 70°）→ 一致度被拉低 → degraded，不虚报可用。"""
    level, tilted = [], []
    for i in range(6):
        level.append(_pose(turn_deg=-60.0 + 25.0 * i, roll_deg=0.0, t=(0.3 * i, 0.0, 0.2 * i)))
    for i in range(4):
        tilted.append(_pose(turn_deg=-60.0 + 25.0 * i, roll_deg=70.0, t=(0.3 * i, 0.0, 0.2 * i)))
    est = estimate_world_frame(level + tilted)
    assert est.status == "degraded"
    assert est.world_up is not None                      # 多数派仍能把约定定下来
    assert est.up_consistency < TH_UP_CONSISTENCY        # 一致度如实反映离散
    # roll 离散度（诊断量）明显高于"全帧竖直"的情形
    assert est.roll_std_deg is not None
    assert est.roll_std_deg > estimate_world_frame(level).roll_std_deg + 5.0


def test_balanced_yaw_coverage_cancels_pitch_bias():
    """yaw 覆盖一整圈时，逐帧俯仰带来的横向分量互相抵消 → 融合解回到真正的竖直。

    这是"平均"这个设计本身的正确性体现（俯仰不是误差，是绕竖直轴转出来的横向分量）；
    一致度仍如实低于阈值（各帧确实离竖直 40°）→ 状态保守地判 degraded。
    """
    bundle = [_pose(turn_deg=30.0 * i, pitch_deg=40.0, t=(0.4 * i, 0.0, 0.0))
              for i in range(12)]
    est = estimate_world_frame(bundle)
    assert est.world_up == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)   # 偏置被抵消
    assert est.status == "degraded"                                   # 但置信保守
    assert est.up_consistency == pytest.approx(np.cos(np.radians(40.0)), abs=1e-9)


def test_all_pitched_bundle_is_the_documented_blind_spot():
    """已知边界（模块 docstring"诚实边界"）：位姿束里**没有重力参照**。

    若所有帧以同一姿态俯仰 ~90°（相机一直朝地板）且 yaw 覆盖很窄，逐帧"图像上方"
    几乎与地面平行、横向分量不相消 → 融合解落在**倾斜平面**里。此时"相机一直俯视"
    与"世界竖直轴真是水平"在数学上不可区分：任何只吃位姿的估计器都无法自查。本用例
    把这一限制**钉死为行为契约**：估计给出（不是 None）、状态 available（帧间一致），
    但 up 明显不竖直 —— 想消除该偏差必须有外部参照（IMU/重力或人工约定），
    **不允许**用场景几何（包围盒/点云）去"修"它（那正是 v5 `acd95847c5` 的错误路线）。
    """
    narrow = [_pose(turn_deg=10.0 * i, pitch_deg=88.0, t=(0.4 * i, 0.1 * i, 0.0))
              for i in range(8)]
    est = estimate_world_frame(narrow)
    assert est.status == "available"                    # 帧间一致 → 位姿束"看不出"异常
    assert est.up_consistency > TH_UP_CONSISTENCY
    assert est.world_up is not None
    assert abs(est.world_up[1]) < 0.2                   # 估计落在近水平面（偏差可见）


def test_pitched_bundle_with_width_yaw_coverage_fails_closed():
    """同一俯仰角、但 yaw 覆盖一整圈 → 横向分量相消使合矢长度塌到阈值以下 → None + degraded。

    这一对照说明上面那条"窄覆盖"用例的问题不在俯仰本身，而在**信息量**：
    偏航覆盖足够宽时，"竖直"真的能从位姿束里被约束出来（见 40° 俯仰那条用例）；
    覆盖又宽、俯仰又接近 90° 时，竖直方向的约束权重（∝ cos θ）已薄到不可靠，
    本模块据此 **fail-closed**（`world_up=None`），而不是给一个薄信息的解。
    """
    balanced = [_pose(turn_deg=30.0 * i, pitch_deg=88.0, t=(0.4 * i, 0.0, 0.0))
                for i in range(12)]
    est = estimate_world_frame(balanced)
    assert est.status == "degraded"
    assert est.world_up is None
    assert est.up_consistency < wf.TH_UP_CANCEL         # 合矢长度塌陷 = 符号/方向不可定
    assert "符号不可定" in est.note


def test_threshold_constants_are_module_level_and_documented():
    """阈值必须是模块级 TODO_CALIBRATE 常量（可标定），起始参考值按规格给出。"""
    assert wf.TH_UP_CONSISTENCY == TH_UP_CONSISTENCY == 0.9
    assert isinstance(wf.TH_MIN_FRAMES_ROLL, int) and wf.TH_MIN_FRAMES_ROLL >= 1
    assert 0.0 < wf.TH_UP_CANCEL < wf.TH_UP_CONSISTENCY
    assert wf.WORLD_FRAME_VERSION == "world-frame-v6"
    src = inspect.getsource(wf)
    for name in ("TH_UP_CONSISTENCY", "TH_MIN_FRAMES_ROLL", "TH_UP_CANCEL",
                 "TH_MIN_FRAMES_UP", "TH_UP_ALIGN_FRAME", "TH_FRONT_HALF_ANGLE_DEG"):
        assert f"{name}: " in src and "TODO_CALIBRATE" in src
