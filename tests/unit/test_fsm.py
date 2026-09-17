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


def test_online_static_check_fail_regenerate_limited():
    fsm = OnlineFSM(max_regen=3)
    for ev, ctx in [("episode_ready", {}), ("gate_done", {"action": "proceed"}),
                    ("done", {}), ("gate_done", {"action": "proceed"}),
                    ("done", {}), ("done", {}), ("done", {})]:
        fsm.step(ev, ctx)
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


def test_online_geometry_reject_still_logs():
    fsm = OnlineFSM()
    for ev, ctx in [("episode_ready", {}), ("gate_done", {"action": "proceed"}),
                    ("done", {}), ("gate_done", {"action": "proceed"}),
                    ("done", {}), ("done", {}), ("done", {}),
                    ("pass", {}), ("ok", {})]:
        fsm.step(ev, ctx)
    fsm.step("reject")  # GEOMETRY_VERIFY 拒绝
    assert "geometry_rejected" in fsm.answer_flags
    fsm.step("done")
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
