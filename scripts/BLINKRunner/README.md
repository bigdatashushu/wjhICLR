# BLINKRunner

BLINK 主实验入口和 2026-10-31 的 24 题测试代码集中在此目录。

| 路径 | 用途 |
| --- | --- |
| `run.py` | 主实验：direct、no_skill、initial、training、freeze_skill、final、score |
| `paired.py` | 原内部 test 集 93 题：固定 trajectory-1 Skill＋工具 vs 纯 Qwen |
| `paired_start.py` | 经 stdin 瞬态接收凭据，核验后通过 tmux 启动双组对照 |
| `paired_audit.py` | 完成后独立核验原图图序、实际请求参数、Skill 交付和评分 |
| `paired_report.py` | 将完整验收结果生成中文 Markdown 报告 |
| `sr.py` / `sr_data.py` | SR 单图 yes/no：官方 val/test 数据、预检、预计算、双组求解和独立评分 |
| `sr_trial.py` / `sr_adapter.py` | SR val 43/100 随机划分，固定测试 20 道，8 并发配对实验及完整 JSON 轨迹 |
| `pilots/skill24/pilot.py` | 固定 trajectory-1 Skill＋工具的 24 题测试，22/24（91.7%） |
| `pilots/direct24/pilot.py` | 同批 24 题纯 qwen3.8-flash 对照，21/24（87.5%） |
| `pilots/*/continuation.py` | 核验历史请求前缀、输入身份和冻结源码 |
| `pilots/*/start.py` | 当时的 Titan 启动器，核验准备回执后通过 tmux 启动 `pilot.py` |
| `pilots/skill24/test_continuation.py` | 当时使用的离线请求前缀测试 |
| `pilots/source_manifest.json` | 收录文件的原始位置、SHA-256 和对应 Titan 实验目录 |

SR 有 val 143 题（公开答案）和 test 143 题（隐藏答案），均为一张原图、
`A=yes / B=no`。适配说明与命令见 [`SR.md`](SR.md)；
默认离线检查：`.venv/bin/python -B scripts/BLINKRunner/sr.py --check`。

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

## 93 题完整对照

在 Titan 上从 24 题 Skill 测试的冻结 campaign 创建独立新目录：

```bash
PYTHONPATH=/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/cpu_compat \
  /home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python \
  -B scripts/BLINKRunner/paired.py \
  --root /path/to/new-blink-paired93-run \
  --prepare-from /path/to/blink-pilot24-revised-20261031/campaign
```

该入口核验并复制 24 题实验冻结的 194 份源码，沿用模型、求解和 Skill 参数；
两组各测 93 道原内部 test 题，不训练或演化。共享独立的 465 请求预算、最多 4 并发，
Skill 组单题最多 4 请求，纯 Qwen 单题 1 请求。纯 Qwen worker 不接收几何缓存位置。
未完成请求和服务错误会停止调度，不自动重发；已完成结果通过哈希核验后保留。

`paired_start.py --root <run>` 只从 stdin 读取 `{"api_key": ...}`，由本地现有私有凭据
加载器经 SSH stdin 注入；勿把密钥写入命令、源码、配置或报告。
预测完整冻结、求解 worker 退出后，宿主侧才读取 test 标签并评分。
主进程成功退出后运行 `paired_audit.py --root <run>`；下载完整证据后运行
`paired_report.py --root <local-results>` 生成报告。

2026-11-04 的独立运行目录为
`/home/wangy/wangjh/harness3d/four-datasets-20261031/blink-paired93-20261104-r01`。
本地离线边界与结果防篡改检查：`pytest scripts/BLINKRunner/test_paired.py`。

## 离线验证

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests \
  .venv/bin/python -B -m pytest tests/test_blink_experiment.py \
  tests/test_blink_recovery.py scripts/BLINKRunner/pilots/skill24/test_continuation.py \
  -q -p no:cacheprovider --basetemp=/private/tmp/blinkrunner-validation
```
