# 逐轮快照动作适配 V2

2026-11-05。实现集中在本目录，复用共享 `src/skill3d` 的程序装配、数值安全、工具授权和 small-image 感知，采用任务专用 `action_runtime/`。已接入正式 live 入口、Linux 隔离和官方 V2 worker；阶段验收见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)，本次实施规划见 [LIVE_IMPLEMENTATION_PLAN.md](LIVE_IMPLEMENTATION_PLAN.md)。

最新 r06 部署已完成三环境 × 三臂的真实首动作联调：9/9 轮执行一个官方动作，25 次实际推理，服务器与本地独立审计均通过。[最终验收收据](checks/action-live-acceptance-20261105.json) 记录实际范围；本次没有开展完整任务成功率或方法效果比较。

2026-11-06 的 Manipulation r04 部署补齐第四环境：`base:000` 的 Direct/Full 各完成 1 个真实动作，**2/2 首动作联调通过**，完整轨迹和本地独立审计已保存。[第四环境验收](checks/manipulation-acceptance-20261106.json) 记录范围和感知覆盖；当前四环境均有真实首动作接入证据。

## 闭环与观测

四个环境均走同一流程：

`官方 open → 当前 RGB/动作表/反馈 → 冻结快照 → 当前局部感知 → coding Python → 新隔离进程 → SubmitActions → 宿主执行首动作 → 新观测`

正常 `step`，包括官方动作失败，都返回 RGB 和官方反馈。相机不一定移动，RGB 也可能不变；宿主仍递增 step、mutation 和 observation 版本，撤销旧 decision 的当前工具结果，下一轮重新感知和生成程序。ALFRED/Habitat 的格式 `invalid` 不调用真实 step，复用原 RGB、增加 invalid/mutation 和 observation 版本，仍产生新 decision。不能把这种反馈更新称为新拍摄的图像。

程序提交的是 `ActionPayload` 数据，不能持有或操作 simulator/bridge。宿主验证整个计划，执行第一个合法动作。`SubmitActions` 和 `YieldObservations` 均立即结束本段；空动作列表结束题目，成功与进度只取自官方评分。

EB-Manipulation 使用七维 `[x, y, z, roll, pitch, yaw, gripper]` 动作，动作表为空。官方匿名物体坐标和标注图由可信 worker 提供；原始 RGB 单独用于预测感知。位置索引 0–100，Euler 索引 0–120，每格 3°，夹爪 0/1。接入进度和实际验证范围见 [MANIPULATION_IMPLEMENTATION_PLAN.md](MANIPULATION_IMPLEMENTATION_PLAN.md)。

每段使用新的进程和用户变量空间。语法、运行、工具参数错误与 yield 可在同 decision 的剩余预算内继续；yield 给同一 coding agent 返回至多 3 个公开结果页，反馈总计至多 4000 字节，宿主读取也计入 reader 预算。当前 decision 的结果账本跨段保留，动作或 invalid 后撤销。允许携带来源明确的简短子目标记忆，历史记录不会获得当前几何权限。

## 运行臂与固定条件

| 臂 | 模型输出 | 当前感知/工具 | 方法 |
|---|---|---|---|
| Direct | 原官方 `executable_plan`，宿主执行首动作 | 不加载感知或几何；零工具/reader | 原官方 prompt、文本示例和家务 parser |
| B01 | 唯一 JSON 字段 `program_source` | 当前图像的局部工具 | 同执行合同，无 Skill |
| B11 | 唯一 JSON 字段 `program_source` | 同 B01 的感知配置与工具合同 | 整题冻结的环境 Skill，普通/修复请求均全文交付 |
| Full | 同 B11，用于 Direct/Full 配置 | 当前图像的局部工具 | 同 B11 的完整冻结 Skill |

上述各臂均固定 Qwen3.8-Flash、seed 137、temperature 0、thinking false、JSON 输出、4096 输出 tokens、120 秒模型超时、single_action。返回的模型身份若与冻结配置不符，立即停止并拒绝审计。

| 预算 | Navigation | ALFRED / Habitat（各自） | Manipulation |
|---|---:|---:|---:|
| 真实动作 / 格式 invalid | 20 / 0 | 30 / 10 | 15 / 0 |
| Qwen 请求 / 全部模型请求 | 20 / 40 | 40 / 70 | 30 / 45 |
| 类别核验、DINO、SAM、MoGe（分别） | 20 | 30 | 15 |
| 工具 / reader / 程序段 | 240 / 120 / 60 | 360 / 180 / 90 | 180 / 90 / 45 |
| 题目墙钟秒数 | 600 | 900 | 900 |
| 观察到的 Qwen token 软阈值 | 150000 | 250000 | 250000 |

每 decision 至多 3 次生成/修复、12 次工具、6 次 reader、3 段执行；每段至多 3 次 reader。每个真实服务调用先预留 permit，失败仍计费；token 仅记录数值，缺失 usage 明确标注。格式解析失败时仍保留已返回的 usage。token 阈值允许最后一次调用超过阈值。所选各臂分别持有 run 总预算，汇总按预先选择的题目固定分母。

服务故障不触发自动重试。动作 ACK 丢失、错配或不可验证时记录 `action_outcome_unknown`，停止且不重发/reset；已经预留的动作额度保留。V2 worker 对同 UUID、同内容在存活期间返回原 ACK，不能用这一缓存声称崩溃后的恰好一次执行。

## 离线命令

在任意目录运行；不需要读取凭据、安装依赖或启动仿真。以下输出文件必须不存在，父目录必须已存在。

零请求预检：

```bash
env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp \
  PYTHONPATH=/Users/bytedance/wangjh/wjhICLR/src:/Users/bytedance/wangjh/wjhICLR/embodiedRunner \
  /Users/bytedance/wangjh/wjhICLR/.venv/bin/python3.11 -B \
  /Users/bytedance/wangjh/wjhICLR/embodiedRunner/action_run.py --check
```

四个环境 × 三臂，真实 NDJSON fake worker 进程、真实 prompt/assembler 和隔离程序执行：

```bash
env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp \
  PYTHONPATH=/Users/bytedance/wangjh/wjhICLR/src:/Users/bytedance/wangjh/wjhICLR/embodiedRunner \
  /Users/bytedance/wangjh/wjhICLR/.venv/bin/python3.11 -B \
  /Users/bytedance/wangjh/wjhICLR/embodiedRunner/action_run.py \
  --fake --process-worker \
  --output /Users/bytedance/wangjh/wjhICLR/embodiedRunner/checks/my-action-v2-offline-run.json
```

独立、只读复核报告：

```bash
env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp \
  PYTHONPATH=/Users/bytedance/wangjh/wjhICLR/src:/Users/bytedance/wangjh/wjhICLR/embodiedRunner \
  /Users/bytedance/wangjh/wjhICLR/.venv/bin/python3.11 -B \
  /Users/bytedance/wangjh/wjhICLR/embodiedRunner/action_audit.py \
  --run /Users/bytedance/wangjh/wjhICLR/embodiedRunner/checks/my-action-v2-offline-run.json
```

`--check` / `--fake` 始终离线；去掉 `--process-worker` 使用进程内 fake 环境 transport，生成程序仍在独立隔离进程执行。不得把 fake 的 1.0 成功率用作方法性能结论。

## 真实运行

真实入口为 `action_run.py --live-check` / `--live`，都显式要求 `--live-config`，`--live` 另外要求新 `--output`。输出路径在任何 live 调用前检查。配置示例是 `config/action-live-4090.json`，预选三个环境的 base:000 和完整三臂；`integration_step_limit=1` 用于首动作联调，改为 `null` 才按完整题目预算运行。该参数随 manifest 冻结，审计不会把未执行首动作的程序失败记为联调通过。

Mac 端复用已认证 SSH socket，向全新服务器目录部署：

```bash
env PYTHONPATH=/Users/bytedance/wangjh/wjhICLR/src:/Users/bytedance/wangjh/wjhICLR/embodiedRunner \
  /Users/bytedance/wangjh/wjhICLR/.venv/bin/python3.11 -B \
  /Users/bytedance/wangjh/wjhICLR/embodiedRunner/action_deploy.py \
  --remote /home/cvailab/wangjh/embodied-action-v2-NEW
```

然后从 Mac 做零推理预检；把下面的部署目录改为上一步创建的同一目录：

```bash
env PYTHONPATH=/Users/bytedance/wangjh/wjhICLR/src:/Users/bytedance/wangjh/wjhICLR/embodiedRunner \
  /Users/bytedance/wangjh/wjhICLR/.venv/bin/python3.11 -B \
  /Users/bytedance/wangjh/wjhICLR/embodiedRunner/action_live_launch.py \
  --remote /home/cvailab/wangjh/embodied-action-v2-NEW \
  --check --output-name preflight.json
```

发起真实首动作联调：

```bash
env PYTHONPATH=/Users/bytedance/wangjh/wjhICLR/src:/Users/bytedance/wangjh/wjhICLR/embodiedRunner \
  /Users/bytedance/wangjh/wjhICLR/.venv/bin/python3.11 -B \
  /Users/bytedance/wangjh/wjhICLR/embodiedRunner/action_live_launch.py \
  --remote /home/cvailab/wangjh/embodied-action-v2-NEW \
  --output-name smoke.json
```

`--check` 核对官方依赖、源码、服务身份、权重哈希和实际 namespace/seccomp seal，0 推理、0 仿真启动。launcher 通过 SSH stdin 临时交付已授权 Qwen key；检测 token 从服务器原授权 loader 加载。生成程序和仿真 worker 不继承凭据。`PYTHONPATH` 必须同时包含 `src` 和 `embodiedRunner`。

也可在 Linux 上用正式 host Python，设置 `PYTHONPATH=<部署根>/src:<部署根>/embodiedRunner` 后直接调用：

```bash
python -B action_run.py --live-check --live-config config/action-live-4090.json --output checks/preflight.json
python -B action_run.py --live --live-config config/action-live-4090.json --output runs/live.json
```

直接调用需要宿主环境已有 `DASHSCOPE_API_KEY` 和 `HARNESS3D_PERCEPTION_TOKEN`；不把它们写到命令行、JSON 或日志。配置要求已部署的官方 Python、固定 SAM2/MoGe2 权重、受保护检测池和已固定身份的 Intern 审核服务，失败不回退 fake。当前资产部署是 pilot：ALFRED 为 base 的前三题；每个全新 Habitat worker 仅接受 index 0。超出范围在 live 请求前拒绝。

每轮回执及开跑 manifest 立即保存在 `<输出名>-episodes/`，最后输出配对汇总。服务/宿主/执行故障或未知 ACK 后，后续计划轮次写为 `not_started_after_operational_stop`，保留完整分母且不继续发起请求。模型格式/程序错误只使用已规定的有界修复预算。

官方 invalid 计数单独保存于宿主审计观测：它可能包含正常 step 的失败，不能等同于 host 的格式-invalid 调用数。初始图像完全一致由 `require_identical_initial_observation` 控制；默认首动作 smoke 为 `false`，审计仍核验相同任务/动作表/input，并显式报告 `initial_observation_match` 和 `paired_comparison_eligible`。首动作联调不具备方法效果比较资格；严格配对评测需要该字段为 `true`、完整题目预算和全部身份/观测条件通过。

## 身份与审计

`action_identity.py` 在开跑前冻结显式实现/共享源码清单、官方代码、方法库、预算、模型与感知配置、协议/执行合同和运行时版本。每题报告绑定同一 paired manifest、来源和配置身份；决策与模型请求绑定 RGB、动作表、观测版本和证据 hash，B11 回执绑定完整正文 hash 和字符数。轨迹可分叉；严格配对评测要求初始公开观测一致，首动作 smoke 按配置记录观测差异和比较资格。

`action_audit.py` 拒绝缺题、重复题、未选择题、配置/方法偏离、正文截断、漏记请求、预算超额、不合法迁移/链 hash、未知 mutation 后续执行、旧 decision 未撤销，以及 Direct 使用感知或工具。合法的端点/服务/动作未知失败仍保留在固定分母，单独汇报终止原因。审计复核当前磁盘源码；源码后来变化会拒绝原冻结报告，不能原地重新命名旧运行以绕过该检查。

报告仅保存去敏后的身份 hash、状态、数值 usage 和授权摘要；不保存 prompt、程序源码、RGB 字节、异常原文、服务地址、凭据或参考答案。报告 hash 用于完整性与一致性检查，不是外部签名或模型质量证明。

用户要求的完整轨迹单独保存在 `<输出名>-episodes/trajectory-*/`，含逐事件 JSON、实际公开 prompt、模型响应、生成程序、公开观测和按 hash 保存的 PNG/预测数组；不包含凭据或私有任务配置。Direct/Full 比较和 Manipulation 联调会自动保存，汇总入口为 `trajectories.json`。可用 `audit_action_trajectory.py --run <输出 JSON>` 只读核验全部资产、模型输入/输出 hash、观测与最终报告。

原 230 个源码、103 个旧基线文件和 active pointer 保护可用 `action_preservation.py` 复核，只读取文件字节计算 hash，不解析旧运行内容。

## 当前执行范围

macOS 的生成程序在可信依赖加载后、执行前进行一次 seatbelt seal，禁止文件读写、网络和 fork/exec，并关闭非通信 FD。Linux 使用 bwrap 的私有 mount/PID/network/user namespace，可信导入后安装 seccomp syscall allowlist；没有 open、socket、fork/exec、ptrace 等权限。原始/伪造 bwrap marker 不能绕过 seal。父进程监测 CPU、墙钟、输出和整个自有进程树的内存；worker 也用内核保留峰值拒绝超额提交。内存仍按采样和峰值验收，不声称逐次分配的硬上限。封锁失败直接拒绝执行。最新部署的 Linux 实测见 [linux-isolation-final-20261105.xml](checks/linux-isolation-final-20261105.xml)：14 passed、1 skipped（macOS 专用项）。

感知使用原始当前 RGB，固定公开类别词表、最多 4 类和 1 个候选；每当前 view 至多一次组合 DINO、SAM、类别核验和 MoGe。类别核验需要开题前已缓存的健康/模型身份；服务失败停止。单点深度为 5×5 局部中位数，深度和点距离只在同一视角的相对单位中有效。类别/box/mask 不会自动变成已验证物体表面，candidate 表面深度保持拒绝。当前交付不提供跨视角米制距离或完整世界地图。

Manipulation 的当前真实首动作指令含 `star` / `container`，固定词表未匹配到类别，故该题仅调用 MoGe，DINO/SAM/Intern 均未触发。完整轨迹中的候选表为空；本轮没有验证检测或分割质量。动态 Qwen 类别选择尚未实施，具体实际图像和覆盖见 [第四环境说明](MANIPULATION_IMPLEMENTATION_PLAN.md)。

本次真实 smoke 的生成程序未调用工具或 reader，对应执行分支由离线闭环和隔离测试覆盖。组件细节见 [P2_INTERFACE.md](P2_INTERFACE.md)、[P3_INTERFACE.md](P3_INTERFACE.md)；完整离线和真实联调验收记录见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)。
