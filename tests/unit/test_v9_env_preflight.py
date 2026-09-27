"""v9 P0：环境预检（§17.1「环境依赖」+ §18.1 fail-closed）。

预检的判据必须是**行为**：直接调用运行时真正依赖的 scipy 能力。版本号比较会
漂移（scipy 的 numpy 兼容上界变过多次），而"这个函数现在能不能用"不会。
"""

from __future__ import annotations

import pytest

from skill3d import env_preflight
from skill3d.env_preflight import (
    RuntimeEnvironmentError,
    assert_runtime_dependencies,
    describe_environment,
    runtime_dependency_problems,
)


def test_current_environment_passes_behavioural_probes():
    """在正确环境下预检必须通过（否则真实运行根本起不来）。"""
    assert runtime_dependency_problems() == []
    env = assert_runtime_dependencies(context="test")
    assert env["numpy"] and env["scipy"]


def test_describe_environment_covers_the_dependencies_that_broke():
    """§17.1：numpy/scipy 属于环境依赖，必须进 RunManifest 留档。"""
    env = describe_environment()
    for key in ("python", "python_executable", "numpy", "scipy"):
        assert key in env and env[key]
    assert env["python_executable"]


def test_assert_raises_when_a_dependency_probe_fails(monkeypatch):
    """任一探针失败 → 拒绝启动（而不是降级后照跑）。"""
    def boom() -> None:
        raise ValueError("NoneType copy mode not allowed")

    monkeypatch.setattr(env_preflight, "_PROBES", (
        ("scipy.spatial.cKDTree", boom, "M4 点云重叠子项"),))
    with pytest.raises(RuntimeEnvironmentError) as exc:
        assert_runtime_dependencies(context="unit")
    message = str(exc.value)
    assert "cKDTree" in message and "M4 点云重叠子项" in message
    # 报错必须给出可执行的修复方向
    assert env_preflight.FROZEN_ENV_PYTHON in message
    assert "shim" in message


def test_problem_list_reports_every_failing_probe(monkeypatch):
    def boom_a() -> None:
        raise RuntimeError("a 坏了")

    def boom_b() -> None:
        raise RuntimeError("b 坏了")

    monkeypatch.setattr(env_preflight, "_PROBES", (
        ("probe.a", boom_a, "能力 A"), ("probe.b", boom_b, "能力 B")))
    problems = runtime_dependency_problems()
    assert len(problems) == 2
    assert "probe.a" in problems[0] and "probe.b" in problems[1]


def test_inference_env_versions_includes_numpy_and_scipy():
    """RunManifest 的 inference_env 必须含 numpy/scipy（曾只记模型框架而漏掉）。"""
    from skill3d.infra.version_lock import inference_env_versions

    env = inference_env_versions()
    assert env.get("numpy") and env["numpy"] != "absent"
    assert env.get("scipy") and env["scipy"] != "absent"
