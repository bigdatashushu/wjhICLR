# 原 20 题同步 RGB-D 复测

完成并审计 20/20 题。原 RGB 成功 14/20（70%），本次 RGB-D 成功 15/20（75%）。四环境均使用官方 base 索引 0–4；本轮全部重新运行。

| 环境 | 原 RGB | RGB-D | 成功率变化 | 模型请求 | 动作步数 |
|---|---:|---:|---:|---:|---:|
| navigation | 4/5 | 4/5 | +0 个百分点 | 31 | 59 |
| alfred | 5/5 | 4/5 | -20 个百分点 | 9 | 38 |
| habitat | 4/5 | 5/5 | +20 个百分点 | 22 | 42 |
| manipulation | 1/5 | 2/5 | +20 个百分点 | 13 | 60 |

模型：qwen3.8-flash；temperature=0，seed=137，max_tokens=4096，enable_thinking=false，JSON-object。沿用官方提示、示例和完整动作计划。仿真串行运行，GPU 渲染使用物理卡 7，ALFRED/Manipulation 使用 CPU 软件渲染。API 并发允许 32，本轮实际峰值为 1。

实际执行顺序为 ALFRED → Habitat → Manipulation → Navigation，全部串行。准备清单写了 Navigation 先执行，但 canonical JSON 的包名排序使三环境包先执行；保留原清单，实际顺序见 controller.json。

每次模型决策输入两张 500×500 图：当前 RGB 与同一观测的原生仿真深度。固定 0–10 米反向灰度：白色近，黑色表示 10 米及以上或无效像素。原始 float32 米制深度另存 NPY。模型消息只含 RGB 和深度可视化图，没有额外工具调用或参考答案。

审计覆盖 75 次模型请求、234 次同步观测和 199 个动作回执；逐请求重放一致、深度 PNG 与原始数值一致、动作发送与 ACK 一一对应。操作性失败 0，自动重试 0。3909 个原实验文件哈希保持不变，远程自有 worker 已全部退出。

相对原 RGB，2 题由失败变成功，1 题由成功变失败。成功以官方环境回执为准；模型自行声称任务已完成仍可能计为失败。

官方环境终止回执覆盖 19/20 题，模型空计划、无效计划或请求预算触发的提前终止共 1 题，均计失败。

初始 RGB 图像哈希与基线一致 14/20 题。原基线包含 9 题历史复用记录，其中没有原始模型请求/响应正文；当前结果用于观察同一批题的表现，尚不能把差异全部归因于深度输入。

| 题目 | 原 RGB → RGB-D | 当前状态 | 步数 | 请求数 | 初始 RGB 相同 | 结果 |
|---|---|---|---:|---:|---|---|
| alfred/base:000 | 1 → 1 | official_done | 11 | 1 | 是 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_three_envs/runs/rgbd-twenty-20261107-r01/alfred-base-000/result.json) |
| alfred/base:001 | 1 → 1 | official_done | 8 | 1 | 是 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_three_envs/runs/rgbd-twenty-20261107-r01/alfred-base-001/result.json) |
| alfred/base:002 | 1 → 0 | empty_plan | 11 | 5 | 是 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_three_envs/runs/rgbd-twenty-20261107-r01/alfred-base-002/result.json) |
| alfred/base:003 | 1 → 1 | official_done | 4 | 1 | 是 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_three_envs/runs/rgbd-twenty-20261107-r01/alfred-base-003/result.json) |
| alfred/base:004 | 1 → 1 | official_done | 4 | 1 | 是 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_three_envs/runs/rgbd-twenty-20261107-r01/alfred-base-004/result.json) |
| habitat/base:000 | 0 → 1 | official_done | 18 | 10 | 是 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_three_envs/runs/rgbd-twenty-20261107-r01/habitat-base-000/result.json) |
| habitat/base:001 | 1 → 1 | official_done | 4 | 1 | 是 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_three_envs/runs/rgbd-twenty-20261107-r01/habitat-base-001/result.json) |
| habitat/base:002 | 1 → 1 | official_done | 4 | 1 | 否 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_three_envs/runs/rgbd-twenty-20261107-r01/habitat-base-002/result.json) |
| habitat/base:003 | 1 → 1 | official_done | 4 | 1 | 是 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_three_envs/runs/rgbd-twenty-20261107-r01/habitat-base-003/result.json) |
| habitat/base:004 | 1 → 1 | official_done | 12 | 9 | 是 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_three_envs/runs/rgbd-twenty-20261107-r01/habitat-base-004/result.json) |
| manipulation/base:000 | 0 → 0 | official_done | 15 | 4 | 是 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_three_envs/runs/rgbd-twenty-20261107-r01/manipulation-base-000/result.json) |
| manipulation/base:001 | 1 → 1 | official_done | 5 | 1 | 是 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_three_envs/runs/rgbd-twenty-20261107-r01/manipulation-base-001/result.json) |
| manipulation/base:002 | 0 → 0 | official_done | 15 | 3 | 是 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_three_envs/runs/rgbd-twenty-20261107-r01/manipulation-base-002/result.json) |
| manipulation/base:003 | 0 → 0 | official_done | 15 | 3 | 是 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_three_envs/runs/rgbd-twenty-20261107-r01/manipulation-base-003/result.json) |
| manipulation/base:004 | 0 → 1 | official_done | 10 | 2 | 是 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_three_envs/runs/rgbd-twenty-20261107-r01/manipulation-base-004/result.json) |
| navigation/base:000 | 1 → 1 | official_done | 9 | 4 | 否 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_trial/runs/rgbd-twenty-20261107-r01/navigation-base-000/result.json) |
| navigation/base:001 | 1 → 1 | official_done | 11 | 4 | 否 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_trial/runs/rgbd-twenty-20261107-r01/navigation-base-001/result.json) |
| navigation/base:002 | 1 → 1 | official_done | 10 | 4 | 否 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_trial/runs/rgbd-twenty-20261107-r01/navigation-base-002/result.json) |
| navigation/base:003 | 0 → 0 | official_done | 20 | 15 | 否 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_trial/runs/rgbd-twenty-20261107-r01/navigation-base-003/result.json) |
| navigation/base:004 | 1 → 1 | official_done | 9 | 4 | 否 | [记录](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_trial/runs/rgbd-twenty-20261107-r01/navigation-base-004/result.json) |

[初始 RGB/深度图总览](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_twenty/rgbd20-initial-pairs.png) · [逐题结果和完整审计](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_twenty/rgbd20-results.json) · [远程最终检查](file:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_twenty/server-after.json)

深度来源：Navigation 使用 AI2-THOR 同一事件的米制深度；ALFRED 原生毫米除以 1000；Habitat 使用与 RGB 相同位置、朝向及视场角的深度传感器；Manipulation 使用同一 front 相机观测，并按 near + depth × (far − near) 转换为米。这是仿真原生深度试验；迁移至真实 RGB 输入时需要另行接入深度估计。
