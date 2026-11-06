# 正式入口、Linux 隔离与真实联调

2026-11-05。用户明确授权解决三项缺口，包含服务器部署和有界真实联调。更改集中在 embodiedRunner，不修改共享 src，不提交或推送。

| 阶段 | 上下文估计 | 验证结果 |
|---|---:|---|
| 当前服务和资产检查 | 5k–8k | 只读现场回执，零模型请求 |
| Linux namespace + seccomp | 8k–12k | 真实 Linux 文件、网络、进程、FD、资源边界检查 |
| live CLI、正式 worker、感知接入 | 12k–16k | 明确任务选择、冻结配置、脱敏审计 |
| 三环境真实 smoke 和修复 | 10k–16k | 固定分母、实际请求/动作/工具回执 |
| 回归与交付 | 5k–8k | 离线回归、旧源保护、运行说明 |

上述是上下文规划，不是在线预算。每阶段在结果返回后的干净边界整理状态。真实 smoke 使用独立目录、不自动重试、不向求解器交付 reference/GT；服务偏差与动作结果未知均停止。失败尝试保留。

## 已完成

- Linux 新增 `sandbox_linux.py`：bwrap 私有 mount/PID/network/user namespace，可信导入后 seccomp syscall allowlist；关闭额外 FD，无凭据继承。旧 `EMBODIED_BWRAP=1` 不能绕过 seal。
- Linux 原生访问与真实隔离执行：14 passed、1 skipped（macOS 专用项）；回执 `checks/linux-isolation-20261105.xml`。首次测试发现只采样 bwrap 的内存，现已改为 owned descendant tree，并拒绝内核保留峰值超预算的提交。
- macOS 回归：14 passed、1 skipped（Linux 专用项）。
- 新 `action_deploy.py` 将任务代码复制到全新 workspace，官方 runtime/assets 复用既有部署；每次修改使用新的部署目录，旧 V1 源码保持不变。
- V2 worker 增加零请求 `--probe`，统一依赖/native 输出隔离；官方 Navigation、ALFRED、Habitat probe 均 `runtime_ready=true`。
- `--live-check` / `--live`、显式冻结配置、真实 Qwen/检测/SAM2/Intern/MoGe2、逐轮落盘和全运行 operational stop 已实现。`action_live_launch.py` 仅通过 SSH stdin 和服务器原授权 loader 临时加载认证。
- 首批 r03：9 个官方任务实例、35 次实际推理调用、5 个真实动作。B01 与部分 B11 的程序被 AST 拒绝；旧审计还因官方 invalid 计数语义报错，保留完整失败历史 `checks/action-live-first-attempt-20261105.json`。导航三个独立初始渲染的哈希不同，不能用于严格配对效果比较。
- 补充预注入 API、扁平 runtime ctx、字符串动作表说明，增加有限 guard taxonomy。官方 invalid 计数与 host 格式-invalid permits 分开验证；默认 smoke 明确不要求独立渲染字节相同，并报告比较资格。要求初始观测完全一致的模式仍会拒绝差异。
- 最新独立部署为 `/home/cvailab/wangjh/embodied-action-v2-20261105-r06`，零推理预检通过，本地独立核对当前源码、配置和身份也通过：[预检回执](checks/action-live-final-preflight-20261105.json)。
- 三环境 × 三臂的真实首动作验收 **9/9 通过**，9 个官方动作全部取得 ACK。Qwen 9、DINO 6、SAM2 2、Intern 类别审核 2、MoGe2 6，共 25 次实际推理；6 个生成程序均完成 Linux 隔离执行，0 程序错误、0 abort。[运行回执](checks/action-live-final-run-20261105.json) 与 [本地独立审计](checks/action-live-final-audit-20261105.json) 均 passed。
- 最终离线回归覆盖本目录 tests、三个旧 baseline 测试及共享 AST/assembler/v11 边界：**245 passed、1 skipped、4 subtests passed**（79.18s），[测试回执](checks/action-live-final-regression-20261105.xml)。最新部署 Linux 隔离复验 **14 passed、1 skipped**（14.36s），[服务器测试回执](checks/linux-isolation-final-20261105.xml)。
- 最新原文件保护复核通过：230 个原源码、103 个旧 baseline 文件及 active pointer 均不变，[保护回执](checks/action-live-final-preservation-20261105.json)。

## 验收范围

三项实施工作均完成，[最终验收收据](checks/action-live-acceptance-20261105.json) 绑定本次代码身份和六份证据。当前配置每轮执行一个真实动作后停止，完整任务成功率实验尚未开展。生成程序在本次 smoke 中没有调用工具或 reader；这两个执行分支的验证来自离线闭环和隔离测试。

Navigation 的独立初始 RGB hash 不同，本次比较资格明确为 `false`；首动作结果不用于三臂方法效果比较。当前官方 pilot 资产支持 ALFRED base 前三题、全新 Habitat worker 的 index 0。使用完整题目预算可将 `integration_step_limit` 设为 `null`，更改后需新部署、冻结配置并重新预检。
