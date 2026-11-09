# 目标检测与裁剪复核：实现交付

当前源码入口为 `target_detection_r05`。问题提目标、逐类别 GroundingDINO、完整链接重叠分组、图外 G 编号、原 RGB 裁剪复核和官方完整动作执行已接入。修正版已部署并完成四环境零模型请求预检。

九项离线检查通过，覆盖公开来源与目标冻结、重叠链、类别缓存、编号失效、裁剪像素、工具名、阶段推进、动作形式、工具阶段零仿真步和完整计划 ACK。真实 Navigation 的七次响应已离线重放，修正后的复核处理无需新模型请求。

## 真实运行结果

真实试跑使用冻结的 R04，四环境各一题，官方成功 1/4。合计 66 次 Qwen 请求、50 次工具调用、24 次逐类别检测，40 个动作与 40 个 ACK。

| 环境 | 结束状态 | 官方成功 | Qwen 请求 | 检测请求 | 裁剪 | 动作/ACK |
|---|---|---:|---:|---:|---:|---:|
| navigation | planner_error | 0 | 7 | 2 | 4 | 2/2 |
| alfred | official_done | 1 | 4 | 2 | 1 | 11/11 |
| habitat | request_budget | 0 | 40 | 12 | 15 | 14/14 |
| manipulation | request_budget | 0 | 15 | 8 | 8 | 13/13 |

Navigation 在第二次观测后的纯复核响应省略 tool_request 外层，R04 提前结束。R05 按明确复核处理，保留原视觉判断并执行相同校验。裁剪工具名被文件名覆盖的问题也已修正。R05 尚未进行新的在线试跑，上述真实结果均属于 R04。

Habitat 和 Manipulation 分别达到固定的 40 和 15 次模型请求预算后结束；检测、裁剪和复核轮次消耗同一预算。这是本次观察到的成本限制，不能把更多工具调用直接当作表现改进。

## 证据与入口

- [源码与运行说明](README.md)
- [实际截图与逐环境说明](../target_detection_r04/acceptance-report.md)
- [RGB、编号与裁剪对照 PNG](../target_detection_r04/detection-review-preview.png)
- [真实 R04 完整离线审计](../target_detection_r04/runs/target-detection-20261108-r04/audit.json)
- [真实失败响应的修正版回放](verification/navigation-review-replay.json)
- [修正版四环境运行预检](verification/preflight.json)
- [修正版源码与部署身份](verification/frozen.json)
- [最终服务器与原文件保留检查](verification/server-after.json)

当前原实验、前四版源码和运行证据保持字节一致，已无本轮仿真 worker。检测服务仍为固定 GPU1/SwinB 身份。模型 key 仅由原授权入口在内存加载，交付文件扫描未发现原始 key。

四题用于工程接入检查，不代表完整数据集上的成功率收益。检测框、分组和模型裁剪判断不等于已核实的实例或表面几何。
