"""M6 Mock 三档切换单测（§4 M6 字段 11 / §9.2）。"""

import json

import numpy as np
import pytest

from skill3d.schemas import SceneState
from skill3d.tools import MockContaminationError, MockSwitch, SceneHandle, call_tool


def _handle() -> SceneHandle:
    return SceneHandle(
        SceneState(
            artifact_ref="artifact://demo",
            route="full_3d",
            frame="world",
            scale_known=True,
            objects=[],
            summary="",
        ),
        objects=[],
    )


ARGS = {"point_a": [0, 0, 0], "point_b": [1, 0, 0]}


def test_real_mode_returns_real():
    r = call_tool("euclidean_distance", ARGS, _handle(), mode="real")
    assert r.source == "real"
    assert json.loads(r.value) == pytest.approx(1.0)


def test_mock_interface_returns_placeholder():
    switch = MockSwitch()
    r = call_tool("euclidean_distance", ARGS, _handle(), mode="mock_interface", mock_switch=switch)
    assert r.source == "mock_interface"
    assert json.loads(r.value) == 0.0  # float 占位


def test_mock_replay_hit():
    h = _handle()
    switch = MockSwitch()
    switch.load_replay("euclidean_distance", ARGS, h.state_digest(), 42.0)
    r = call_tool("euclidean_distance", ARGS, h, mode="mock_replay", mock_switch=switch)
    assert r.source == "mock_replay"
    assert json.loads(r.value) == 42.0


def test_mock_replay_digest_mismatch_degrades_real():
    h = _handle()
    switch = MockSwitch()
    switch.load_replay("euclidean_distance", ARGS, "stale_digest", 42.0)
    r = call_tool("euclidean_distance", ARGS, h, mode="mock_replay", mock_switch=switch)
    # state_digest 不匹配 → 降级 real 并记录
    assert r.source == "real"
    assert json.loads(r.value) == pytest.approx(1.0)
    assert switch.degradation_log
    assert switch.degradation_log[0]["reason"] == "state_digest_mismatch"


def test_mock_replay_miss_degrades_real():
    switch = MockSwitch()
    r = call_tool("euclidean_distance", ARGS, _handle(), mode="mock_replay", mock_switch=switch)
    assert r.source == "real"
    assert switch.degradation_log[0]["reason"] == "replay_miss"


def test_real_mode_mock_source_raises():
    """准入阶段（mode=real）出现 mock_* source 必须抛错（T4 防污染）。"""
    with pytest.raises(MockContaminationError):
        MockSwitch.assert_admission_clean("mock_interface", "real")
    # 正常 real 不抛
    MockSwitch.assert_admission_clean("real", "real")
    MockSwitch.assert_admission_clean("mock_light", "mock_light")
