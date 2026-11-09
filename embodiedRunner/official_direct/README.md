# 四环境官方直接测试

独立入口位于 `official_direct/`。执行闭环为官方 reset、官方公开观测、官方规划提示、模型返回完整动作计划、官方 step、官方终止与评分。使用三个现有快照的固定 revision `9be4e980e9cd6bcb38373cd4aab7c32724bdd401`。

已完成本地实现、离线验证和独立服务器部署。四环境各 5 题的补充基线配置见 `baseline-20261107*.json`，历史复用核验见 `verification/reuse-20261107.json`；真实结果按运行身份保存在 `runs/`。

2026-11-07 的 20 道基线已完成：Navigation 4/5、ALFRED 5/5、Habitat 4/5、Manipulation 1/5。9 道复用历史官方多动作结果，11 道为 r03 新测。API 并发上限 32，实际峰值 6。Navigation 的一条无效模型计划计失败；其余 19 道有官方终止回执。报告见 `baseline-five-report-20261107.md`，数据和输入重放审计见 `baseline-five-results-20261107.json`。三次部署源码和失败运行日志另存于 `deployment-snapshots/`、`remote-evidence/`。

| 环境 | 官方动作 | 动作上限 | 同一计划遇到失败动作 |
|---|---|---:|---|
| EB-Navigation | 8 个离散导航动作 | 20 | 中断计划，重新规划 |
| EB-ALFRED | reset 后的场景相关高层动作 ID | 30；累计无效动作 10 次 | 中断计划，重新规划 |
| EB-Habitat | 官方高层动作 ID | 30；累计无效动作 10 次 | 中断计划，重新规划 |
| EB-Manipulation | `[x,y,z,rx,ry,rz,gripper]` 七维动作 | 15 | 继续计划；官方 done 时停止 |

默认每环境选择 `base:000`，运行到任务终止或官方上限。默认模型为官方配置的 `gpt-4o-mini`，3/10/10/4 个官方示例，单张当前观测，环境反馈开启，聊天历史关闭。没有任务卡、子目标门禁或额外感知工具。请求预算分别为 20/40/40/15，ALFRED/Habitat 额外的额度用于官方无效计划计数分支。

## 本地检查

在 `embodiedRunner` 目录中：

```bash
../.venv/bin/python -B -m official_direct check
```

也可从任意目录使用绝对入口：

```bash
/Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B /Users/bytedance/wangjh/wjhICLR/embodiedRunner/official_direct/__main__.py check
```

`check` 校验 926 个官方文件、任务选择和四种提示构造；不会连接网络、启动子进程、仿真或模型。Manipulation 的任务选择依据已有资产清单，实际资产完整性由远端预检确认。

## 真实运行入口

以下是新入口部署完成、用户授权真实测试后的命令；本次没有执行。控制进程运行在 Mac，仿真通过 SSH 在 4090 节点的三个已有 Python 环境中运行。只需要部署新增的 `official_direct/*.py`；远端代码与本地代码哈希必须一致。官方源码、已有运行和安装环境不覆盖。

先用新配置确认模型、任务和物理 GPU。默认要求 GPU 空闲；显式设置 `runtime.gpu_min_free_mb` 后按可用显存检查，保留原有服务。允许的物理 GPU 为 0/1/3/7。补充基线使用 GPU 7、至少 4096 MiB 空余显存，并限制 Navigation/Habitat 合计同时 1 道；ALFRED/Manipulation 使用软件渲染。此检查不预留资源。

```bash
../.venv/bin/python -B -m official_direct probe
../.venv/bin/python -B -m official_direct run --run-id direct-four-r01
```

`probe` 连接服务器、只读核对版本、源码/新入口哈希、资产和 GPU，不启动仿真或模型。`run` 在每题启动前再次核对。SSH 使用已有配置或 `checks/ssh4090` 的有效连接，启用 `BatchMode` 和严格 host-key 检查；没有认证连接时直接失败。

只跑一个环境：

```bash
../.venv/bin/python -B -m official_direct run --environment manipulation --run-id direct-man-r01
```

运行身份必须唯一；同名目录存在时拒绝覆盖，不支持自动续跑。需要重做时使用新身份，并保留原失败轨迹。

## 配置模型与任务

复制 `default.json` 到本目录内的新配置，修改 `episodes`、模型字段、预算和运行路径，然后指定 `--config official_direct/<配置名>.json`。任务 index 为官方 split 内的零基索引。Manipulation 通过资产清单绑定官方 index 与已安装资产的 runtime index；当前清单只有 `base:000`。ALFRED 当前清单覆盖 base 的 0–4，另可读取原 pilot 资产。新任务资产未准备好时预检失败。

`model.api_key_env`、`model.base_url_env` 只存环境变量名。凭据通过控制进程的环境提供，不写入配置或传给 worker。默认读取 `OPENAI_API_KEY`；未设置 `OPENAI_BASE_URL` 时使用 OpenAI 的 `/v1` 地址。模型客户端采用 OpenAI-compatible HTTP，不需要安装 OpenAI SDK。

Qwen 或其他兼容服务需明确配置模型名和环境变量名。默认使用经哈希校验的官方 JSON Schema；服务不支持时可显式选择 `json_object`，变更会记入运行身份。`extra_body` 仅允许 `seed`、`enable_thinking` 和 `reasoning_effort`，默认不额外设置。GPT-5 系列按现有官方源码使用 `max_completion_tokens`，省略 temperature。

官方提示/示例直接从校验后的源码提取；保留官方模型名称判断下的模板追加行为。Navigation 的首轮 1–2 动作、后续 5–6 动作是提示建议，不是宿主硬上限。宿主执行整个计划的合法前缀，最多截到官方剩余动作数。

## 输入、评分和记录

模型输入只来自 `public` 部分：任务指令、当前图像、官方动作空间、官方环境反馈，以及 Manipulation 的官方匿名坐标。私有评分、成功、进度、距离和累计无效动作数放在 `official` 部分，不进入模型消息。未知或额外的公开字段会被拒绝。

Manipulation 保留官方 `form_object_coord_for_input` 的匿名 `object N` 坐标，并在模型决策时按官方流程绘制 XYZ 轴和检测框。坐标由官方仿真观测推导，来源单独标注；私有 waypoint、原始 mask/depth 和语义对象元数据留在 worker 内。其位置与旋转接受官方提示规定的 0–100、0–120 整数，gripper 为 0/1。

本地结果在 `official_direct/runs/<run-id>/`：

- `identity.json`：源码与新入口哈希、任务清单和明确的协议差异。
- `config.json`：使用的配置，只有凭据/端点环境变量名。
- `<环境>-<子集>-<index>/trajectory.jsonl`：逐条同步落盘的完整事件轨迹，包括官方公开观测与评分。
- `request-NNN.json` / `response-NNN.json`：实际模型 payload 和返回消息、用量、返回模型标识。
- `observation-NNN-0.png`：对应轨迹的图像。
- 每题 `result.json`、总 `results.json` 与 `summary.json`：官方指标、终止原因及固定选择分母。

远端工作文件在配置的 `work_root/<run-id>/workers/<题目>/`。THOR 缓存复制到新工作目录，避免写入旧缓存；这会占用额外磁盘空间。仿真日志、错误调用栈和新进程的输出也集中在该目录。系统 Xvfb 使用其标准临时 socket。

服务、仿真或协议失败停止整组测试，余下题目标为 `not_started`，仍保留在选择分母内。动作发送前先保存 intent；收到对应观测再保存 ack；连接中断时保留未知结果，不重发。摘要中的成功率是选定题目的统计，不能视为全量 benchmark 成绩。

可选 `execution` 配置独立限制总仿真、各环境、GPU 仿真和 HTTP 并发，API 上限支持 32。每题仍按官方顺序执行完整计划。首次操作失败关闭请求入口，保存已在途的响应与动作回执；其他活跃题在下一动作或请求边界停止。新启动身份才能进行明确的修复后重跑。

补充 Manipulation 资产通过 `--manipulation-catalog`（本地）与 `runtime.manipulation_catalog` / `runtime.manipulation_dataset`（服务器）指定，旧资产清单和安装环境保持只读。原生 PyRep 库路径在 Python 启动前注入；零请求预检同时验证原生库可导入。

## 明确的官方协议差异

保留官方动作、提示、公开观测、执行策略和评分。用户要求下列失败处理改变，都会记录在 `identity.json`：

- 不继承官方服务异常后的等待重试。
- 不继承 Navigation/Manipulation 的随机动作兜底；坏计划结束该题。
- ALFRED/Habitat 沿用官方空计划停止、坏计划累计无效动作、首非法 ID 前截断规则；非整数 ID 视为坏计划。
- 七维动作做格式和官方范围校验，不把 Manipulation 当作动作 ID。
- Habitat 关闭迭代器 shuffle/grouping，以 requested index 和真实 episode ID 绑定任务；不修改官方文件。
- Habitat 官方 elapsed time 的单元素 tuple 转为数值，保留其值。

## 离线验证

```bash
env -i PATH=/usr/bin:/bin \
  HOME="$PWD/official_direct/verification" \
  TMPDIR="$PWD/official_direct/verification" \
  PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  ../.venv/bin/python -B -m pytest official_direct/tests \
  --confcutdir=official_direct/tests \
  --basetemp=official_direct/verification/pytest-local \
  -p no:cacheprovider -q
```

测试阻断网络，使用模拟返回/动作反馈核对四环境闭环；两项进程测试只运行标准库的虚拟 NDJSON 端点。它们不启动实际环境或模型。验证内容包括完整官方 planner 的初始/反馈消息一致性、多动作策略、上限、无效动作计数、评分隔离、匿名坐标、操作失败不重试、轨迹与固定分母、Python 3.9 语法及 1601 个历史文件的校验值。
