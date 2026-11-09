# 目标检测与局部裁剪复核：真实验收

本次真实试跑入口为 `target_detection_r04`，冻结配置 `experiment-20261108-r04.json`，运行 `target-detection-20261108-r04`。查询与验收完成时间：2026-11-08T07:28:30.346738+00:00，节点 `cvailab@10.10.0.198`。

四题中官方判定成功 1/4。本次实际记录 66 次 Qwen 请求、50 次工具调用、24 次逐类别 GroundingDINO 推理，40 个动作与 40 个 ACK。

| 环境与题目 | 结束状态 | 官方成功 | Qwen 请求 | 检测推理 | 工具调用 | 裁剪 | 动作/ACK |
|---|---|---:|---:|---:|---:|---:|---:|
| navigation base0 | planner_error | 0 | 7 | 2 | 5 | 4 | 2/2 |
| alfred base0 | official_done | 1 | 4 | 2 | 3 | 1 | 11/11 |
| habitat base0 | request_budget | 0 | 40 | 12 | 30 | 15 | 14/14 |
| manipulation base1 | request_budget | 0 | 15 | 8 | 12 | 8 | 13/13 |

## 实现行为

同一个 Qwen 从公开指令提取类别，数量、属性和关系单独记录；原文片段必须匹配公开指令，根目标整题冻结。模型按需逐类别请求检测，低阈值 0.15 保留全部原始提议和空标签。IoU ≥ 0.8 完整链接分组保留每个成员，避免重叠链误并。编号放在图外侧栏，每页 12 组，后续页均可取。

模型请求原始 RGB 局部裁剪后，独立调用 `review_candidates` 给出 match/not_match/uncertain 及理由，再输出官方完整动作计划。复核记录绑定当前观测；换帧后旧 G 编号失效。全部模型轮次计入原预算，工具阶段没有仿真 step/observe。

实际复核中，Navigation 拒绝了非面包候选；ALFRED 将金属容器样裁剪判为 not_match，没有把它选作 table。Manipulation 分别选中星形物体与蓝色容器。这些是模型的视觉判断记录，不能作为独立语义验证。

Manipulation 的原官方匿名编号图和动作坐标保留，检测使用同帧未标注 RGB。匿名坐标投影只提供可能对应，G 编号与官方 object 编号不等价。

## 真实截图

[四环境原图、检测编号及局部裁剪对照](detection-review-preview.png)

- navigation：类别 bread；所示同一帧为 3 个提议 / 3 组。 [原图](runs/target-detection-20261108-r04/navigation-base-000/observation-000-raw.png) · [检测图](runs/target-detection-20261108-r04/navigation-base-000/perception-000-tool-001-page-1.png)
- alfred：类别 ladle、table；所示同一帧为 2 个提议 / 2 组。 [原图](runs/target-detection-20261108-r04/alfred-base-000/observation-000-raw.png) · [检测图](runs/target-detection-20261108-r04/alfred-base-000/perception-000-tool-001-page-1.png)
- habitat：类别 toy airplane、right counter；所示同一帧为 7 个提议 / 6 组。 [原图](runs/target-detection-20261108-r04/habitat-base-000/observation-000-raw.png) · [检测图](runs/target-detection-20261108-r04/habitat-base-000/perception-000-tool-001-page-1.png)
- manipulation：类别 star、container；所示同一帧为 11 个提议 / 6 组。 [原图](runs/target-detection-20261108-r04/manipulation-base-001/observation-000-raw.png) · [检测图](runs/target-detection-20261108-r04/manipulation-base-001/perception-000-tool-001-page-1.png)

## 审计

八项离线测试通过。真实验收中全部模型消息和工具结果已离线重放，目标冻结、候选编号/失效和当前裁剪判断均复核；检测请求绑定原始 RGB 哈希，裁剪与原 RGB 对应区域逐像素一致。原图、深度 PNG 和原始数值深度文件均校验。动作顺序与完整计划前缀一致，已发送动作与 ACK 数量一致。

旧本地 5219 个文件与前三版结果保持字节一致；服务器旧证据和最终源码身份一致。当前已无本轮 worker 或工作目录内原生仿真进程，检测服务仍为固定 GPU1/SwinB 身份。

[完整审计 JSON](runs/target-detection-20261108-r04/audit.json) · [运行摘要](runs/target-detection-20261108-r04/summary.json) · [源码与运行说明](README.md) · [最终服务器检查](verification/server-after.json)

## 接入版本记录

- `target_detection_trial`：7 次模型请求，stopped_on_operational_failure；原源码、请求、响应和终态均保留。
- `target_detection_v2`：18 次模型请求，finished；原源码、请求、响应和终态均保留。
- `target_detection`：4 次模型请求，stopped_on_operational_failure；原源码、请求、响应和终态均保留。

首版的 Navigation/ALFRED 未输出目标，记录为 planner_error；Habitat 已检测和裁剪，第二动作等待 ACK 时连接关闭，结果未知，没有重发，suite 当时停止。第二版将目标和工具阶段放入系统指令，ALFRED 完成了 11 个动作并成功；Navigation 动作响应遗漏裁剪判断字段。第三版采用独立复核工具阶段；Navigation 明确给出 [整数 ID, 描述] 的动作形式，原字典解析器拒绝了该格式。最终版兼容明确的整数 ID、[整数 ID, 描述] 和官方 action_id 字典，并在系统指令中写明动作格式。不会补动作、推测 ID 或修改动作顺序。

第三版的 ALFRED 在 OpenGL 初始化处持续超过六分钟，尚无模型请求；已按 PID/创建时间/参数哈希核验归属后停止这一套初始化资源，失败轨迹封存。[受控停止回执](verification/r03-controlled-initializer-stop.json)记录了该次处理。

## 修正版与真实试跑的区别

R04 Navigation 在第二次观测后的复核阶段直接返回 perception_review 和 selected_targets，没有 tool_request 外层，也没有动作计划；R04 将其送入动作解析后提前结束。R04 裁剪结果还有一个工具名被裁剪文件名覆盖的问题；本报告按原始 tool_request 识别裁剪事件，原始轨迹不改写。

独立的 `target_detection_r05` 修正了工具名，并兼容上述明确的纯复核响应，仍执行同样的当前裁剪与目标校验，不生成动作。九项离线检查通过，原 Navigation 七次响应已离线重放，确认第七次响应能完成复核并进入动作规划。R05 尚未进行新的在线试跑，不能把 R04 的结果写成 R05 的结果。

[修正版源码说明](../target_detection_r05/README.md) · [真实失败响应回放](../target_detection_r05/verification/navigation-review-replay.json)

Habitat 和 Manipulation 分别达到 40 和 15 次模型请求预算后结束，已执行 14 和 13 个动作。工具轮次与动作规划共用原预算，增加复核轮次会压缩可用于后续规划的请求数量；本轮没有修改预算或追加重跑。

这四题验证接入和动作链，不能据此判断完整数据集上的成功率收益。检测框、分组和 Qwen 判断也不能视为已验证的物体实例或几何表面。
