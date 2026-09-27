"""测试套件的环境闸（v9 §18.1：报告必须区分收集/执行/通过/跳过/失败）。

在错误的解释器下跑测试，结果不是"部分失败"而是**不可解释**：base conda env
（numpy 1.26.4 + scipy 1.18.0）里 `import scipy.spatial` 就抛 `np.long`，有人会加一个
`np.long = int` 的 shim 让收集通过 —— 但 shim 补不了 ABI，`cKDTree` 仍然抛
`copy=None`，于是 M4 子项 fail-closed、`scene_route=fallback_2d_only`，得到一份
"6 通过 12 失败"的、既非真也非假的报告。

所以这里在**会话建立时**就用运行时自己的行为探针判定环境，不满足则拒绝开跑。
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    try:
        from skill3d.env_preflight import (
            FROZEN_ENV_PYTHON,
            describe_environment,
            runtime_dependency_problems,
        )
    except Exception as exc:  # noqa: BLE001 - 连预检都 import 不了 → 环境有问题
        raise pytest.UsageError(
            f"[环境] 无法导入 skill3d.env_preflight（{type(exc).__name__}: {exc}）。"
            f"请用冻结实验环境运行：{FROZEN_ENV_PYTHON}") from exc

    problems = runtime_dependency_problems()
    if problems:
        env = describe_environment()
        detail = "\n".join(f"  - {p}" for p in problems)
        raise pytest.UsageError(
            "[环境] 依赖行为探针未通过，拒绝运行测试套件（此环境下的结果不可解释）：\n"
            f"{detail}\n"
            f"  当前解释器：{env['python_executable']}"
            f"（python {env['python']} / numpy {env['numpy']} / scipy {env['scipy']}）\n"
            f"  本项目冻结实验环境：{FROZEN_ENV_PYTHON}\n"
            "  不要用 NumPy shim 绕过：shim 只补符号，补不了 ABI，会产出真假混杂的报告。")
