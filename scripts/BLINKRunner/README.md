# BLINKRunner

BLINK 主实验入口和 2026-10-31 的 24 题测试代码集中在此目录。

| 路径 | 用途 |
| --- | --- |
| `run.py` | 主实验：direct、no_skill、initial、training、freeze_skill、final、score |
| `pilots/skill24/pilot.py` | 固定 trajectory-1 Skill＋工具的 24 题测试，22/24（91.7%） |
| `pilots/direct24/pilot.py` | 同批 24 题纯 qwen3.8-flash 对照，21/24（87.5%） |
| `pilots/*/continuation.py` | 核验历史请求前缀、输入身份和冻结源码 |
| `pilots/*/start.py` | 当时的 Titan 启动器，核验准备回执后通过 tmux 启动 `pilot.py` |
| `pilots/skill24/test_continuation.py` | 当时使用的离线请求前缀测试 |
| `pilots/source_manifest.json` | 收录文件的原始位置、SHA-256 和对应 Titan 实验目录 |

## 主入口

在仓库根目录预览实验配置：

```bash
.venv/bin/python -B scripts/BLINKRunner/run.py --dry-run
```

输入检查：

```bash
.venv/bin/python -B scripts/BLINKRunner/run.py --check
```

配置仍使用 [`configs/experiments/blink.json`](../../configs/experiments/blink.json)；
公共求解、评分及冻结逻辑使用 `src/skill3d/evaluation/blink_experiment.py`。
现有缓存和宿主答案路径由配置指定。完整输入、预算及续跑说明见
[`docs/blink_experiment.md`](../../docs/blink_experiment.md)。

原 `scripts/run_blink_experiment.py` 转发到本目录入口。新运行冻结时同时携带
转发入口和 `BLINKRunner/run.py`、`__init__.py`，续跑继续使用运行目录中的冻结源码。
历史冻结目录只有原入口时，仍按原目录结构核验。

## 24 题测试代码

`pilots/` 中的 Python 文件按原字节收录，保留当时的参数、导入和绝对路径。
这两轮测试固定 seed 137，从同一批 40 道训练题抽取 24 题，模型为
`qwen3.8-flash`，thinking 开启、temperature 0、最大输出 8192 token，
单次超时 600 秒、最多 4 并发。Skill 组累计 50 次请求、30 次成功工具调用，
纯 Qwen 组 24 次请求。

当时的服务器目录：

```text
/home/wangy/wangjh/harness3d/four-datasets-20261031/
├── blink-pilot24-revised-20261031/
│   ├── pilot.py
│   ├── continuation.py
│   └── start.py
├── blink-pilot24-direct-20261031/
│   ├── pilot.py
│   ├── continuation.py
│   └── start.py
└── blink-recovery-20261031-1ee33f28c07e/code/source/
```

这些历史驱动依赖原实验的 `manifest.json`、预算前缀、冻结源码和缓存，
用于查阅当时的测试实现。`start.py` 对已有启动记录拒绝再次启动，
`pilot.py` 核验驱动哈希并拒绝自动重发未完成请求。
未来实验通过上面的主入口创建新运行目录。

测试结论及逐题对照见
[`docs/blink_skill_pilot_20261031.md`](../../docs/blink_skill_pilot_20261031.md)。
该轮为训练集机制诊断。API 凭据、图片、几何缓存、标签和运行结果保留在原位置。

## 离线验证

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests \
  .venv/bin/python -B -m pytest tests/test_blink_experiment.py \
  tests/test_blink_recovery.py scripts/BLINKRunner/pilots/skill24/test_continuation.py \
  -q -p no:cacheprovider --basetemp=/private/tmp/blinkrunner-validation
```
