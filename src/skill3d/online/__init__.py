"""在线推理链编排层（§6.1）：把 M1–M13 接成一条可运行的链。

- `runner`：`run_episode` 驱动 `OnlineFSM`，依次调用 M1–M13 模块；
- `eval`：`python -m skill3d.online.eval`（§13.5 在线评测入口）；
- `synthetic`：mock_light 合成输入（仅管道验证/集成测试，绝不出实验结论）；
- `config`：读 `configs/*.yaml`（本机无 Hydra，用 yaml 直读，结构对齐 §13）。

硬约束 1：本包属于在线链，禁止 import governance / gpt6（由
`tests/unit/test_no_gpt6_online.py` 静态守卫）。
"""

from .runner import (
    EpisodeOutcome,
    OnlineRunConfig,
    run_episode,
    run_split,
)

__all__ = ["EpisodeOutcome", "OnlineRunConfig", "run_episode", "run_split"]
