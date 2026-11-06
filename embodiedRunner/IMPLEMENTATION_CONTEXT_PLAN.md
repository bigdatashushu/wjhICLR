# 逐轮快照接入：实施与上下文预算

2026-11-05。用户已批准实施，并补充要求：“全部更改尽可能集中在 `/Users/bytedance/wangjh/wjhICLR/embodiedRunner`”。以下是实现 agent 的上下文估计，不是在线模型请求预算或硬性停止额度。

| 工作块 | 上下文 tokens 估计 | 结束条件 |
|---|---:|---|
| 来源隔离、合同定稿 | 6k–10k | 来源身份、public快照、动作、修订合同和假输入通过 |
| 执行适配和进程隔离 | 14k–18k | task-local profile/segment、完整方法交付、OS禁止策略实测 |
| 宿主FSM、预算、V2桥接 | 14k–18k | 单一动作写入口、逐step观察、去重/超时停止、三个adapter |
| 感知证据、方法全文和入口 | 12k–16k | 当前原图局部工具、独立完整Skill、prompt/CLI/脱敏audit |
| 综合验收与交付 | 12k–16k | 三环境fake闭环、旧源保护、运行说明和检查收据 |
| 实际故障余量 | 8k–12k | 只用于已出现的失败，不重复宽泛探索 |

每块结束后更新 `IMPLEMENTATION_STATUS.md`，在结果已经返回的干净边界切换上下文。到估计预算时压缩稳定结论，不能把未完成工作标为完成。

## 当前目录与固定选择

- 新增和修改集中在本目录；`action_runtime/` 保存任务专用的执行适配，依赖现有 `src/skill3d` 的数值安全、工具授权、程序装配和schema。不修改共享 `src`、VSI类型槽位、QA入口和active snapshot。
- 原先隔离worktree只保留前期来源参考，后续不作为实现或运行入口。
- 使用既有 `.venv/bin/python`，不安装依赖；临时目录和缓存位于 `/private/tmp`。
- 完整冻结Skill → coding Python → 隔离执行/工具 → SubmitActions → 宿主官方step。初版 `single_action`；Direct/B01/B11同调度，真实动作后重新生成程序。
- 正常step（含动作失败）均返回RGB和官方反馈；家务invalid不step、复用原RGB，但更新状态和观测修订。
- 不做跨帧世界重建；当前几何在任何mutation后撤销，保留审计账本。
- 用户已授权离线实现和验证。部署、真实模型/感知请求、仿真smoke仍需按原阶段另行处理；本次不复用旧裸模型实验授权。
- 不commit、push、清理或更改Git配置；保留其他任务dirty/未跟踪文件以及旧runs。
