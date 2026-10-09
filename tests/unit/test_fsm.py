"""§6 状态机测试：在线 happy path / STATIC_CHECK 回退 / 离线 outer 失败→REJECT。"""

from skill3d.fsm.offline_fsm import OfflineFSM, OfflineState
from skill3d.fsm.online_fsm import OnlineFSM, OnlineState


def test_online_happy_path_to_log_trace():
    fsm = OnlineFSM()
    assert fsm.state is OnlineState.INGEST
    fsm.step("episode_ready", {"split_ok": True})
    assert fsm.state is OnlineState.INPUT_GATE
    fsm.step("gate_done", {"action": "proceed"})
    assert fsm.state is OnlineState.RECONSTRUCT
    fsm.step("done")
    # v6 D5/D1：重建后先观测**世界系契约**（world_up/handedness 存在性）与
    # **度量尺度融合**状态，再进质量门禁；两态都不改 scene_route
    assert fsm.state is OnlineState.WORLD_FRAME
    fsm.step("done", {"world_frame_unavailable": False})
    assert fsm.state is OnlineState.METRIC_FUSION
    fsm.step("done", {"metric_fusion_failed": False,
                      "scale_dispersion_high": False})
    assert fsm.state is OnlineState.QUALITY_GATE
    fsm.step("gate_done", {"action": "proceed"})
    assert fsm.state is OnlineState.CLASSIFY_TASK
    fsm.step("done")
    fsm.step("done")
    assert fsm.state is OnlineState.SYNTHESIZE_PROGRAM
    fsm.step("done")
    fsm.step("pass")  # STATIC_CHECK
    assert fsm.state is OnlineState.SANDBOX_EXECUTE
    fsm.step("ok")
    fsm.step("pass")  # GEOMETRY_VERIFY
    fsm.step("done")  # BENCHMARK_EVAL
    assert fsm.state is OnlineState.ANSWER
    fsm.step("done")
    assert fsm.state is OnlineState.LOG_TRACE
    assert fsm.terminated


def test_online_scale_states_do_not_block_on_failure():
    """v6 §6.2/D4：世界系契约缺失 / 度量融合失败**都不**中止 episode。

    二者只把对应**分项能力**降级（`world_frame` / `metric_scale`），
    使逐题 `question_tool_scope` 收窄（方向类/米制 Tool 被收回）；
    `scene_route` 与其余能力（几何/检测/时序/图像）完全不受影响 ——
    这就是 §7.2"单项失败只收回依赖该证据的工具"。
    """
    fsm = OnlineFSM()
    for ev, ctx in [("episode_ready", {}), ("gate_done", {"action": "proceed"}),
                    ("done", {})]:
        fsm.step(ev, ctx)
    assert fsm.state is OnlineState.WORLD_FRAME
    fsm.step("done", {"world_frame_unavailable": True})
    assert "world_frame_unavailable" in fsm.answer_flags
    fsm.step("done", {"metric_fusion_failed": True, "scale_dispersion_high": True})
    assert fsm.state is OnlineState.QUALITY_GATE      # 仍在正常链上
    assert "metric_fusion_failed" in fsm.answer_flags
    assert "scale_dispersion_high" in fsm.answer_flags
    assert "unanswerable" not in fsm.answer_flags


def test_online_reconstruct_skip_bypasses_scale_states():
    """复用 artifact / mock_light：世界系契约与度量融合已随 artifact 落盘
    → 跳过两态直达质量门禁。"""
    fsm = OnlineFSM()
    for ev, ctx in [("episode_ready", {}), ("gate_done", {"action": "proceed"})]:
        fsm.step(ev, ctx)
    fsm.step("skip")
    assert fsm.state is OnlineState.QUALITY_GATE


def _to_static_check(fsm: OnlineFSM) -> None:
    """驱动到 STATIC_CHECK（含 v4 新增的 SCALE_ESTIMATE/SCALE_CALIBRATE 两态）。"""
    for ev, ctx in [("episode_ready", {}), ("gate_done", {"action": "proceed"}),
                    ("done", {}), ("done", {}), ("done", {}),
                    ("gate_done", {"action": "proceed"}),
                    ("done", {}), ("done", {}), ("done", {})]:
        fsm.step(ev, ctx)


def test_online_static_check_fail_regenerate_limited():
    fsm = OnlineFSM(max_regen=3)
    _to_static_check(fsm)
    assert fsm.state is OnlineState.STATIC_CHECK
    # 第 1、2 次失败 → 回 SYNTHESIZE_PROGRAM
    for _ in range(2):
        fsm.step("fail")
        assert fsm.state is OnlineState.SYNTHESIZE_PROGRAM
        fsm.step("done")
        assert fsm.state is OnlineState.STATIC_CHECK
    # 第 3 次失败 → 超限 → ANSWER(unanswerable)
    fsm.step("fail")
    assert fsm.state is OnlineState.ANSWER
    assert "unanswerable" in fsm.answer_flags
    fsm.step("done")
    assert fsm.state is OnlineState.LOG_TRACE


def test_online_geometry_reject_returns_to_solver_then_still_logs():
    fsm = OnlineFSM()
    _to_static_check(fsm)
    for ev in ("pass", "ok"):
        fsm.step(ev)
    fsm.step("reject")  # GEOMETRY_VERIFY 拒绝
    assert "geometry_rejected" in fsm.answer_flags
    assert fsm.state is OnlineState.SYNTHESIZE_PROGRAM
    fsm.step("solver_failed")  # driver 预算耗尽/恢复失败后，不经评分直接收尾
    assert fsm.state is OnlineState.ANSWER
    fsm.step("done")
    assert fsm.state is OnlineState.LOG_TRACE


def _offline_to_l3(fsm: OfflineFSM) -> None:
    fsm.step("clustered", {"cross_scene_ok": True})
    fsm.step("candidate_ready")
    fsm.step("pass")  # LEAKAGE_CHECK
    fsm.step("done")  # LOOP_SYNTHESIZE
    fsm.step("pass")  # STATIC_CHECK
    fsm.step("pass")  # L1
    fsm.step("pass")  # L2
    assert fsm.state is OfflineState.TEST_L3


def test_offline_outer_failure_rejects():
    fsm = OfflineFSM()
    _offline_to_l3(fsm)
    fsm.step("fail")  # outer 失败 → REJECT 终态（不允许再修订，硬约束 10）
    assert fsm.state is OfflineState.REJECT
    assert fsm.terminated
    assert fsm.outer_attempted


def test_offline_happy_path_promotes():
    fsm = OfflineFSM()
    _offline_to_l3(fsm)
    fsm.step("pass")
    assert fsm.state is OfflineState.PROMOTE


def test_offline_leakage_check_rejects():
    fsm = OfflineFSM()
    fsm.step("clustered", {"cross_scene_ok": True})
    fsm.step("candidate_ready")
    fsm.step("fail")  # LEAKAGE_CHECK 不过
    assert fsm.state is OfflineState.REJECT


def test_offline_insufficient_samples_rejects():
    fsm = OfflineFSM()
    fsm.step("clustered", {"cross_scene_ok": False})
    assert fsm.state is OfflineState.REJECT


def _offline_to_loop_synthesize(fsm: OfflineFSM) -> None:
    fsm.step("clustered", {"cross_scene_ok": True})
    fsm.step("candidate_ready")
    fsm.step("pass")  # LEAKAGE_CHECK
    assert fsm.state is OfflineState.LOOP_SYNTHESIZE


def test_offline_loop_macro_edges_are_terminal_reachable():
    """§5.2：内环整体执行完（driver 把 L1→L2→L3 委托给 run_optimization_loop）时，
    LOOP_SYNTHESIZE 必须有 pass/fail 宏边——否则 PROMOTE 不可达，
    准入/promote（硬约束 12/13）在 driver 里永远不会被触发。"""
    ok = OfflineFSM()
    _offline_to_loop_synthesize(ok)
    ok.step("pass")                       # 内环整体通过（含 L3 outer 一次）
    assert ok.state is OfflineState.PROMOTE
    assert ok.terminated
    assert ok.outer_attempted             # outer 只跑一次的审计位（硬约束 10）

    bad = OfflineFSM()
    _offline_to_loop_synthesize(bad)
    bad.step("fail")                      # 内环整体失败 → REJECT（不得 Revise）
    assert bad.state is OfflineState.REJECT
    assert bad.terminated

    granular = OfflineFSM()
    _offline_to_loop_synthesize(granular)
    granular.step("done")                 # 逐状态驱动仍可用（→ STATIC_CHECK）
    assert granular.state is OfflineState.LOOP_STATIC_CHECK
