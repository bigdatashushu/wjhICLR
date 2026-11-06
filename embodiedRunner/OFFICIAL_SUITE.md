# 官方协议的三类任务接入

本轮按用户指定范围接入 EB-Navigation、EB-ALFRED 和 EB-Habitat，各选 `base` 子集的前三题，使用裸 Qwen3.8-Flash。官方源码固定为 `9be4e980e9cd6bcb38373cd4aab7c32724bdd401`。

## 2026-11-05 真实试跑结果

| 环境 | 完成题数 | 成功题数 | 模型请求 | 环境动作 | 最终审计 |
|---|---:|---:|---:|---:|---|
| Navigation | 3 | 3/3 | 11 | 30 | [14 项通过](runs/qwen-bare-official-suite-nav-3/audit.json) |
| ALFRED | 3 | 3/3 | 11 | 33 | [19 项通过](runs/qwen-bare-official-suite-alfred-3/audit.json) |
| Habitat | 3 | 2/3 | 13 | 28 | [19 项通过](runs/qwen-bare-official-suite-habitat-3/audit.json) |

本轮共执行 9 题、35 次模型请求、91 个环境动作，三个运行均正常结束。结果正文：[Navigation](runs/qwen-bare-official-suite-nav-3/summary.json)、[ALFRED](runs/qwen-bare-official-suite-alfred-3/summary.json)、[Habitat](runs/qwen-bare-official-suite-habitat-3/summary.json)；统一计数见 [official-suite-summary.json](runs/official-suite-summary.json)。

ALFRED 前两题分别使用 1 次模型请求、11 和 8 个环境动作成功；第三题使用 9 次模型请求、14 个环境动作成功，累计 7 次无效动作，其中 5 次为计划解析错误。Habitat 第一题执行 20 步、累计 10 次无效动作后失败，后两题均以 1 次模型请求、4 个环境动作成功。无服务故障或仿真运行故障，没有自动重发模型请求。

离线工程检查覆盖导航 26 项、家务 6 项及 4 项子检查。依据分别为 [官方协议检查](checks/official-suite-pytest-v2.log) 和 [最终家务检查](checks/household-final-offline-tests.log)。真实试跑与离线模拟检查分别保存，表中只统计真实执行结果。

## 协议

| 项目 | Navigation | ALFRED | Habitat |
|---|---|---|---|
| 官方环境 | AI2-THOR 5.0.0 | AI2-THOR 2.1.0 | Habitat-Sim / Habitat-Lab 0.3.0 |
| 模型图像 | 当前 RGB，500×500 | 当前 RGB，500×500 | 当前 head RGB，500×500 |
| 文本示例 | 官方 3-shot | 官方 10-shot | 官方 10-shot |
| 环境动作上限 | 20 | 30 | 30 |
| 动作空间 | 官方 8 个导航动作 | 当前场景的官方动态高层动作 | 官方高层动作 |
| 执行与反馈 | 按官方策略逐步执行 | 执行计划；动作失败时重规划 | 执行计划；动作失败时重规划 |

使用官方 system prompt、示例、消息组织、动作含义和成功判定。ALFRED/Habitat 的规划适配器从经哈希校验的官方 `VLMPlanner` 提取并执行提示、消息、反馈和解析方法；每题重置规划状态，ALFRED 每题更新动作表。

“裸模型”保留官方评测自带的提示与文本示例。没有注入本项目 Skill、几何感知工具、受限程序或 Skill 演化。模型收到图像、任务指令、可用动作与官方动作反馈；成功判定和任务进度仅用于记录。

Qwen 请求明确设置 `temperature=0`、`seed=137`、`enable_thinking=false`、`max_tokens=4096`。接口使用 JSON-object 模式替代官方 SDK 的结构化 schema。服务或仿真运行故障会停止，并保留失败目录；没有自动重发模型请求。ALFRED/Habitat 非整数动作视为解析错误，空计划停止、非法计划计一次无效动作，累计 10 次停止。每题最多 40 次模型请求，每个三题运行最多 120 次。

Habitat 指定 `shuffle=false`、`group_by_scene=false`、`seed=137`，并检查真实 episode ID，确保跑的是数据集的 0、1、2 题。

## 运行入口

Mac 运行模型客户端，Linux 运行无模型凭据的独立仿真工作进程。远端根目录为 `/home/cvailab/wangjh/embodiedRunner`；仿真依赖位于本任务的 `venv-nav` 和 `venv-household`。

以下命令从本地 `embodiedRunner` 目录执行，使用项目已有的 Python 环境：

```bash
../.venv/bin/python -B bare_run.py run --output-dir runs/new-navigation-3

../.venv/bin/python -B household_run.py check --environment alfred --output-dir runs/new-alfred-3
../.venv/bin/python -B household_run.py run --environment alfred --output-dir runs/new-alfred-3

../.venv/bin/python -B household_run.py check --environment habitat --output-dir runs/new-habitat-3
../.venv/bin/python -B household_run.py run --environment habitat --output-dir runs/new-habitat-3
```

每次 `run` 使用新的输出目录；已有结果不会覆盖。`check` 只检查依赖和身份，模型请求为零。上述命令需要本地已授权的 Qwen 凭据，以及经认证的任务 SSH 连接；凭据不部署到仿真服务器。

```bash
../.venv/bin/python -B audit_bare_run.py runs/new-navigation-3
../.venv/bin/python -B audit_household_run.py runs/new-alfred-3
../.venv/bin/python -B audit_household_run.py runs/new-habitat-3
```

审计核对源码身份、题目顺序、真实图像哈希、动作和请求数量、运行上限及成功率计数。

## 数据与安装

`fetch_household.py` 获取完整的官方 ALFRED/Habitat 源码和 Git LFS 内容；清单为 `household_sources.json`。ALFRED 当前准备了 `base` 前三题的真实轨迹，清单为 `alfred_pilot_data.json`。扩展到其它 ALFRED 题目时，需要按相应 split 下载真实轨迹。

Habitat 使用官方 0.3.0 下载脚本规定的资产标签：ReplicaCAD `v1.6`、hab_fetch `v2.0`、YCB `v1.2`。`fetch_habitat_assets.py` 解析标签的不可变 commit，并验证每个文件的 Git blob 或 LFS SHA256；通过 Mac 下载、中转到 Linux，避开服务器的 Hugging Face 连接问题。资产清单为 `habitat_assets.json`。

`setup_household.py` 在任务目录安装 Python 3.9、Habitat-Sim 0.3.0 的 `headless withbullet` 版本、Python 依赖、CPU PyTorch 和 Habitat-Lab。ALFRED 的 Xvfb、字体和 Unity 2.1 build 都位于任务目录，使用 Mesa 软件渲染；不替换系统字体或全局运行环境。

本轮实际验证的版本为 Python 3.9.23、AI2-THOR 2.1.0、Habitat-Sim / Habitat-Lab 0.3.0、Gym 0.23.0、NumPy 1.23.5、PyTorch 2.4.0+cpu，回执见 [household-runtime-ready.json](checks/household-runtime-ready.json)。Habitat 使用已授权的物理 GPU 7；ALFRED 的软件渲染限制为 2 个线程。

ALFRED 和 Habitat 均通过真实环境 reset／step 检查，模型请求为零；初始图像分别见 [ALFRED](checks/alfred-initial.png) 和 [Habitat](checks/habitat-initial.png)。Python 与 Habitat/Magnum 原生日志统一写入任务日志，JSON 观测和动作使用独立通道，避免原生渲染输出污染通信。

运行目录保存 `startup.json`、`config.json`、`summary.json`、逐题 `result.json`、经筛选的 `trace.jsonl` 和真实 PNG。日志只保留请求/响应哈希、数值 token 用量、动作、输入身份和结果字段，不保存模型请求正文或凭据。

## 结果使用

这些前三题运行用于验证模型调用、图像传输、规划与真实动作执行的完整闭环。三题成功率只能描述本轮小样本；完整基准成绩需要覆盖官方全部相关子集，并使用同一协议与本项目方法成对比较。
