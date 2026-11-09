# 四环境官方直接测评基线

模型：qwen3.8-flash。完成 20/20 道，每环境固定选择官方 base 索引 0–4。采用官方提示、示例、公开观测和完整动作计划；无 Skill、额外工具或参考答案。

| 环境 | 已完成 | 成功 | 成功率（固定分母 5） | 复用 | 新测 |
|---|---:|---:|---:|---:|---:|---:|
| navigation | 5/5 | 4 | 80% | 3 | 2 |
| alfred | 5/5 | 5 | 100% | 3 | 2 |
| habitat | 5/5 | 4 | 80% | 3 | 2 |
| manipulation | 5/5 | 1 | 20% | 0 | 5 |

复用 9 道历史官方多动作测评，重新核对源码哈希、逐步轨迹、图像、原始审计和新旧提示一致性。旧的 15 道首动作 Direct 与 Manipulation 首动作验收不计入本次成绩。

API 并发上限 32；仿真总上限 8，ALFRED 上限 2、Manipulation 上限 3，Navigation/Habitat 在 GPU 7 上合计同时 1 道，启动前至少保留 4096 MiB 可用显存。实际 HTTP 峰值：6。

首轮因 PyRep 原生库路径在 Python 启动后设置而停止，0 次模型请求；第二轮 Manipulation 请求返回 HTTP 400，其 JSON-object 输入缺少 JSON 格式指令。修复采用固定版本官方 template_manip，恢复原 schema 提供的输出格式说明。每次修复后使用新部署和新运行身份，保留旧失败轨迹。所有运行均无自动重试。

模型参数：temperature=0、seed=137、max_tokens=4096、enable_thinking=false、JSON-object。官方动作上限：Navigation 20，ALFRED/Habitat 30（累计无效动作 10），Manipulation 15。

新测每条模型消息均通过公开观测重放比对，动作发送与回执一一对应，实际返回模型与请求、步数核对见 JSON 的 episode_audits。历史轨迹保留消息/响应哈希和逐步动作回执，未保留原始模型请求/响应正文。

| 题目 | 来源 | 状态 | 成功 | 步数 | 模型请求 | 结果 |
|---|---|---|---:|---:|---:|---|
| navigation/base:000 | 复用 | official_done | 1 | 10 | 4 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/runs/qwen-bare-official-suite-nav-3/episodes/base-000.json) |
| navigation/base:001 | 复用 | official_done | 1 | 10 | 3 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/runs/qwen-bare-official-suite-nav-3/episodes/base-001.json) |
| navigation/base:002 | 复用 | official_done | 1 | 10 | 4 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/runs/qwen-bare-official-suite-nav-3/episodes/base-002.json) |
| navigation/base:003 | 新测 | planner_error | 0 | 12 | 10 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/official_direct/runs/baseline-five-20261107-r03/navigation-base-003/result.json) |
| navigation/base:004 | 新测 | official_done | 1 | 10 | 4 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/official_direct/runs/baseline-five-20261107-r03/navigation-base-004/result.json) |
| alfred/base:000 | 复用 | official_done | 1 | 11 | 1 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/runs/qwen-bare-official-suite-alfred-3/episode-000/result.json) |
| alfred/base:001 | 复用 | official_done | 1 | 8 | 1 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/runs/qwen-bare-official-suite-alfred-3/episode-001/result.json) |
| alfred/base:002 | 复用 | official_done | 1 | 14 | 9 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/runs/qwen-bare-official-suite-alfred-3/episode-002/result.json) |
| alfred/base:003 | 新测 | official_done | 1 | 4 | 3 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/official_direct/runs/baseline-five-20261107-r03/alfred-base-003/result.json) |
| alfred/base:004 | 新测 | official_done | 1 | 4 | 1 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/official_direct/runs/baseline-five-20261107-r03/alfred-base-004/result.json) |
| habitat/base:000 | 复用 | official_done | 0 | 20 | 11 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/runs/qwen-bare-official-suite-habitat-3/episode-000/result.json) |
| habitat/base:001 | 复用 | official_done | 1 | 4 | 1 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/runs/qwen-bare-official-suite-habitat-3/episode-001/result.json) |
| habitat/base:002 | 复用 | official_done | 1 | 4 | 1 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/runs/qwen-bare-official-suite-habitat-3/episode-002/result.json) |
| habitat/base:003 | 新测 | official_done | 1 | 4 | 1 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/official_direct/runs/baseline-five-20261107-r03/habitat-base-003/result.json) |
| habitat/base:004 | 新测 | official_done | 1 | 8 | 3 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/official_direct/runs/baseline-five-20261107-r03/habitat-base-004/result.json) |
| manipulation/base:000 | 新测 | official_done | 0 | 15 | 3 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/official_direct/runs/baseline-five-20261107-r03/manipulation-base-000/result.json) |
| manipulation/base:001 | 新测 | official_done | 1 | 5 | 1 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/official_direct/runs/baseline-five-20261107-r03/manipulation-base-001/result.json) |
| manipulation/base:002 | 新测 | official_done | 0 | 15 | 4 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/official_direct/runs/baseline-five-20261107-r03/manipulation-base-002/result.json) |
| manipulation/base:003 | 新测 | official_done | 0 | 15 | 3 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/official_direct/runs/baseline-five-20261107-r03/manipulation-base-003/result.json) |
| manipulation/base:004 | 新测 | official_done | 0 | 15 | 3 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/official_direct/runs/baseline-five-20261107-r03/manipulation-base-004/result.json) |

模型计划无效而提前终止的题目数：1，保留原始输出并计失败。官方终止回执覆盖 19/20 道。最终补跑操作性失败数：0。

这是 20 道固定抽样基线，不能作为全量 EmbodiedBench 成绩。
