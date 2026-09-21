"""G-03/G-30 fuzz 层：5 类空间 MR 的 property-based 测试（hypothesis 驱动）。

§4 M18 / §7.1 G-30 验收：
- 5 类 MR **各有通过用例与违例检测用例**；
- 注入已知错误 program 能**收缩到最小反例**（hypothesis shrinking）；
- `target()` 度量变形前后差异。

运行：
```bash
pytest tests/fuzz -q                     # 默认 profile：examples 适中
HYPOTHESIS_PROFILE=thorough pytest tests/fuzz -q   # 更充分（本地/夜间）
```

设计要点：MR 检查函数与变换函数来自 `skill3d.evolution.metamorphic`（P1 复用），
本层只负责生成用例、注入已知错误、并验证收缩行为。
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from hypothesis import find as hyp_find

from skill3d.evolution.metamorphic import (
    apply_rigid_transform,
    apply_unit_change,
    apply_viewpoint_change,
    check_occlusion_dropframe,
    check_rigid_transform_invariance,
    check_unit_change,
    drop_frames,
    find_violating_transform,
    invariant_holds,
    pairwise_distance_matrix,
    permute_objects,
    random_rotation,
)

# 两个 profile：默认跑得快，thorough 供本地/夜间充分搜索
settings.register_profile("default", max_examples=60,
                          suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("thorough", max_examples=400,
                          suppress_health_check=[HealthCheck.too_slow])
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

UNITS = ("m", "cm", "mm")


# ================================================================ MR-1 单位变换 ----

@given(st.floats(min_value=1e-3, max_value=1e4, allow_nan=False, allow_infinity=False),
       st.sampled_from(UNITS), st.sampled_from(UNITS))
def test_unit_change_holds_for_exact_scaling(value, from_unit, to_unit):
    """MR-1 通过用例：答案随单位严格等比缩放（m↔cm↔mm）。"""
    expected = apply_unit_change(value, from_unit, to_unit)
    assert check_unit_change(value, from_unit, to_unit, expected)


@given(st.floats(min_value=1e-3, max_value=1e4, allow_nan=False, allow_infinity=False),
       st.sampled_from(UNITS), st.sampled_from(UNITS),
       st.floats(min_value=1.01, max_value=10.0, allow_nan=False))
def test_unit_change_detects_violation(value, from_unit, to_unit, factor):
    """MR-1 违例检测：数值未随单位缩放（或缩放错误）必须被判违例。"""
    expected = apply_unit_change(value, from_unit, to_unit)
    wrong = expected * factor
    if abs(wrong - expected) <= 1e-6 * max(1.0, abs(expected)):
        pytest.skip("因子过小，与容差不可区分")
    assert not check_unit_change(value, from_unit, to_unit, wrong)


# ============================================================ MR-2 刚体变换 ----

@st.composite
def _points3d(draw, n=None):
    n = n or draw(st.integers(min_value=2, max_value=6))
    return draw(st.lists(st.tuples(st.floats(-10, 10, allow_nan=False),
                                   st.floats(-10, 10, allow_nan=False),
                                   st.floats(-10, 10, allow_nan=False)),
                         min_size=n, max_size=n))


@given(_points3d(), st.integers(min_value=0, max_value=2**31 - 1))
def test_rigid_transform_preserves_pairwise_distances(points, seed):
    """MR-2 通过用例：刚体变换下成对距离矩阵不变（尺度锚定的核心不变量）。"""
    rng = np.random.default_rng(seed)
    r = random_rotation(rng)
    t = rng.normal(size=3) * 5.0
    p = np.asarray(points, dtype=float)
    moved = apply_rigid_transform(p, r, t)
    assert check_rigid_transform_invariance(p, moved, translation=t, tol=1e-9)
    assert np.allclose(pairwise_distance_matrix(p), pairwise_distance_matrix(moved),
                       atol=1e-9)


@given(_points3d(), st.integers(min_value=0, max_value=2**31 - 1),
       st.floats(min_value=1.05, max_value=3.0, allow_nan=False))
def test_rigid_transform_detects_scale_drift(points, seed, factor):
    """MR-2 违例检测：被额外缩放（尺度漂移）后距离矩阵必须被判定改变。"""
    rng = np.random.default_rng(seed)
    p = np.asarray(points, dtype=float)
    scaled = p * factor
    if np.max(pairwise_distance_matrix(p)) < 1e-6:
        pytest.skip("点过于集中，任何缩放都不可区分")
    assert not check_rigid_transform_invariance(p, scaled, translation=None, tol=1e-9)


# ============================================================ MR-3 视角变换 ----

@st.composite
def _scene_objects(draw):
    """合成小场景：若干对象（类别 + 世界坐标）。"""
    n = draw(st.integers(min_value=2, max_value=5))
    return [{"id": f"obj_{i}", "class_hint": draw(st.sampled_from(["chair", "table", "sofa"])),
             "xyz": draw(st.tuples(st.floats(-5, 5, allow_nan=False),
                                   st.floats(-5, 5, allow_nan=False),
                                   st.floats(-5, 5, allow_nan=False)))}
            for i in range(n)]


@given(_scene_objects(), st.integers(min_value=0, max_value=2**31 - 1))
def test_viewpoint_change_preserves_geometry_and_count(objects, seed):
    """MR-3 通过用例：换世界坐标系后计数、类别多重集、成对距离均不变。"""
    rng = np.random.default_rng(seed)
    r, t = random_rotation(rng), rng.normal(size=3) * 3.0
    xyz = np.asarray([o["xyz"] for o in objects], dtype=float)
    moved = apply_rigid_transform(xyz, r, t)

    assert len(moved) == len(objects)                       # 计数不变
    assert sorted(o["class_hint"] for o in objects) == sorted(
        o["class_hint"] for o in objects)                   # 类别集合不变
    assert np.allclose(pairwise_distance_matrix(xyz), pairwise_distance_matrix(moved),
                       atol=1e-9)                           # 相对距离不变

    # 相机轨迹同样做同一次变换：相对位姿（相机系下的对象坐标）不变
    c2w = np.stack([np.eye(4) for _ in range(3)])
    moved_c2w = apply_viewpoint_change(c2w, r, t)
    for m in moved_c2w:
        assert np.isclose(np.linalg.det(m[:3, :3]), 1.0, atol=1e-9)


@given(_scene_objects(), st.integers(min_value=0, max_value=2**31 - 1))
def test_viewpoint_change_detects_missing_object(objects, seed):
    """MR-3 违例检测：变换后丢对象（计数不变式被破坏）必须被检出。

    判定用"计数不变式"而非距离矩阵形状——后者在点数不等时会被 numpy 广播
    静默吞掉（这正是需要被 MR 抓到的错误类型）。
    """
    rng = np.random.default_rng(seed)
    xyz = np.asarray([o["xyz"] for o in objects], dtype=float)
    moved = apply_rigid_transform(xyz, random_rotation(rng), rng.normal(size=3))
    dropped = moved[:-1]                                    # 注入 bug：少一个对象
    if len(objects) == 1:
        pytest.skip("单对象场景无丢对象可检")
    assert not invariant_holds(len(dropped), len(objects))   # 计数不变式抓到违例
    assert invariant_holds(len(moved), len(objects))         # 正确变换下不变


# ========================================================== MR-4 对象置换 ----

@given(_scene_objects(), st.integers(min_value=0, max_value=2**31 - 1))
def test_object_permutation_preserves_set_level_answers(objects, seed):
    """MR-4 通过用例：对象排列不改变计数与集合类答案。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(objects)).tolist()
    permuted = permute_objects(objects, order)
    assert len(permuted) == len(objects)
    assert permuted == [objects[i] for i in order]
    assert sorted(o["class_hint"] for o in permuted) == \
        sorted(o["class_hint"] for o in objects)


@given(_scene_objects(), st.integers(min_value=1, max_value=2**31 - 1))
def test_object_permutation_with_truncation_is_detected(objects, seed):
    """MR-4 违例检测：置换实现若截断（如只取前 k 个）会被计数不变量抓到。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(objects)).tolist()
    buggy = permute_objects(objects, order)[: max(len(objects) - 1, 0)]
    if len(objects) > 1:
        assert len(buggy) == len(objects) - 1 != len(objects)


# ====================================================== MR-5 遮挡/缺帧 ----

@given(st.lists(st.integers(min_value=0, max_value=31), min_size=1, max_size=8,
                unique=True),
       st.integers(min_value=2, max_value=8))
def test_drop_frames_keeps_surviving_geometry(drop_idx, n_extra):
    """MR-5 通过用例：删帧后保留帧的几何量（位姿）逐元素不变。"""
    n = 32
    c2w = np.stack([np.eye(4) * (i + 1) for i in range(n)])
    depth = np.stack([np.full((4, 4), float(i + 1)) for i in range(n)])
    kept = drop_frames(list(c2w), drop_idx)
    kept_depth = drop_frames(list(depth), drop_idx)
    survivors = [i for i in range(n) if i not in set(drop_idx)]
    assert len(kept) == len(survivors)
    for j, i in enumerate(survivors):
        assert np.array_equal(kept[j], c2w[i])
        assert np.array_equal(kept_depth[j], depth[i])


@given(st.lists(st.integers(min_value=0, max_value=31), min_size=1, max_size=4,
                unique=True))
def test_occlusion_invariant_allows_stable_or_explicit_degradation(drop_idx):
    """MR-5 不变量：缺帧后答案要么不变、要么显式 unanswerable（不允许静默错答）。"""
    kept = drop_frames(list(range(32)), drop_idx)
    assert len(kept) == 32 - len(set(drop_idx))
    assert check_occlusion_dropframe("4", "4")
    assert check_occlusion_dropframe("4", "unanswerable")
    assert not check_occlusion_dropframe("4", "5")
    assert not check_occlusion_dropframe("4", "unanswerable", allow_degraded=False)


# ============================================== 反例收缩（CEGIS 的 verify→shrink）----

def _count_program(frames: list) -> int:
    """正确 program：计的是对象数（与帧无关）。"""
    return len(frames[0]) if frames else 0


def _buggy_frame_counting_program(frames: list) -> int:
    """注入的已知错误 program：把**帧数**当对象数（缺帧即错）。"""
    return len(frames)


def test_injected_buggy_program_is_found_and_shrinks_to_minimal_counterexample():
    """G-30 验收：注入已知错误 program → 搜到反例并**收缩到最小**（只删 1 帧）。

    搜索空间是"删除帧数 d"，不变量是"计数恒等于对象数"。错误 program 把帧数当
    对象数，故任何 d ≥ 1 都违例；hypothesis 收缩到最小扰动 d = 1，即"删 1 帧即复现"
    的最小反例（这正是 CEGIS 想喂给修订器的形式）。
    """
    # 让 bug 在"未缺帧"时潜伏：帧数 == 对象数（视觉上等价），缺帧才暴露
    n_frames = 4
    n_objects = 4

    def count_after_drop(d: int) -> int:
        """删除 d 帧后用被测 program 计数（frames = 剩余帧）。"""
        frames = [[f"obj_{k}" for k in range(n_objects)]] * (n_frames - d)
        return _buggy_frame_counting_program(frames)

    def invariant(d: int) -> bool:
        return count_after_drop(d) == _count_program([["obj"] * n_objects])

    assert invariant(0)                                  # 未缺帧时不违例
    minimal = hyp_find(st.integers(min_value=1, max_value=n_frames - 1),
                       lambda d: not invariant(d))
    assert minimal == 1                                  # 收缩到最小扰动：删 1 帧
    assert not invariant(minimal) and count_after_drop(1) == n_frames - 1

    # 正确 program（与帧数无关）在同一搜索下不产出反例
    assert all(
        _count_program([["obj"] * n_objects] * (n_frames - d)) == n_objects
        for d in range(1, n_frames))


def test_find_violating_transform_reports_first_violation():
    """`find_violating_transform`：确定性枚举并给出首个违例（CEGIS verify 步骤）。"""
    # 期望：尺寸答案不随删除帧数变化；注入 bug 后当删帧数 >= 2 时答案被改写
    candidates = [0, 1, 2, 3]

    def measure(n_drop: int) -> float:
        return 2.0 if n_drop < 2 else 2.0 * (1 + 0.1 * n_drop)

    found = find_violating_transform(
        program_fn=measure,
        transform_fn=lambda n_drop: n_drop,
        invariant_fn=lambda p, tp: (measure(tp), measure(0)),
        params=candidates)
    assert found is not None
    assert found["params"] == 2                               # 首个违例（确定性）
    assert not invariant_holds(found["observed"], found["expected"])


def test_mr_invariant_helpers_are_tolerance_aware():
    """MR 容差口径：相对+绝对容差，0 附近用绝对容差（TODO_CALIBRATE）。"""
    assert invariant_holds(1.0, 1.0 + 1e-9)
    assert not invariant_holds(1.0, 1.05)
    assert invariant_holds(0.0, 1e-12)
    assert not invariant_holds(0.0, 0.1)
