# 原 20 题：问题定向检测与裁剪复核实验

完成并审计 20/20 题。历史 RGB-D 成功 15/20，本轮 R05 成功 10/20；由失败变成功 1 题，由成功变失败 6 题。

| 环境 | 历史 RGB-D | 本轮 R05 | Qwen 请求 | 检测推理 | 工具调用 | 动作/ACK |
|---|---:|---:|---:|---:|---:|---:|
| alfred | 4/5 | 4/5 | 29 | 14 | 21 | 30/30 |
| habitat | 5/5 | 2/5 | 86 | 38 | 64 | 32/32 |
| manipulation | 2/5 | 1/5 | 44 | 22 | 33 | 47/46 |
| navigation | 4/5 | 3/5 | 61 | 20 | 41 | 43/43 |

四环境均为官方 base 索引 0–4。使用冻结的 target_detection_r05，Qwen3.8-Flash、temperature=0、seed=137、max_tokens=4096、enable_thinking=false，JSON-object。官方提示、示例和完整动作执行保持；RGB 与原生深度同步输入。

新增流程是同一模型提取目标、逐类别 GroundingDINO、IoU≥0.8 完整链接分组、图外 G 编号、原始 RGB 裁剪和明确的视觉复核。模型请求上限仍为 Navigation 20、ALFRED/Habitat 40、Manipulation 15，检测与复核轮次计入该预算。四环境与每题均串行，检测复用物理 GPU1，GPU 渲染用 GPU7，其余软件渲染。

合计 220 次 Qwen 请求、94 次检测、159 次工具调用、152 个动作和 151 个 ACK。操作性失败 1，动作结果未知 1。

与历史 RGB-D 的初始 RGB 像素一致 15/20 题；逐题记录同时比较公开指令、动作列表、官方 Manipulation 输入和深度图身份。这是同一批题的历史运行对照，感知材料、阶段提示和多轮模型交互共同变化。

旧 Full 工具版本可匹配 15/20 题，逐题附在结果 JSON 中。历史 RGB-D 自身没有额外工具调用；旧 Full 使用不同的子目标与执行协议，作为补充参考。

## 逐题结果

| 题目 | RGB-D→R05 | 结束状态 | 模型请求 | 工具调用 | 裁剪 | 动作/ACK | 初始 RGB 相同 |
|---|---|---|---:|---:|---:|---:|---|
| alfred/base:000 | 1→1 | official_done | 4 | 3 | 1 | 11/11 | 是 |
| alfred/base:001 | 1→0 | planner_error | 1 | 0 | 0 | 0/0 | 是 |
| alfred/base:002 | 0→1 | official_done | 12 | 9 | 6 | 10/10 | 是 |
| alfred/base:003 | 1→1 | official_done | 8 | 6 | 4 | 5/5 | 是 |
| alfred/base:004 | 1→1 | official_done | 4 | 3 | 2 | 4/4 | 是 |
| habitat/base:000 | 1→0 | planner_error | 37 | 28 | 18 | 13/13 | 是 |
| habitat/base:001 | 1→0 | planner_error | 1 | 0 | 0 | 0/0 | 是 |
| habitat/base:002 | 1→1 | official_done | 9 | 7 | 5 | 4/4 | 是 |
| habitat/base:003 | 1→1 | official_done | 20 | 15 | 8 | 6/6 | 是 |
| habitat/base:004 | 1→0 | planner_error | 19 | 14 | 6 | 9/9 | 是 |
| manipulation/base:000 | 0→0 | operational_error | 12 | 9 | 6 | 14/13 | 是 |
| manipulation/base:001 | 1→0 | planner_error | 1 | 0 | 0 | 0/0 | 是 |
| manipulation/base:002 | 0→0 | official_done | 12 | 9 | 6 | 15/15 | 是 |
| manipulation/base:003 | 0→0 | request_budget | 15 | 12 | 8 | 13/13 | 是 |
| manipulation/base:004 | 1→1 | official_done | 4 | 3 | 2 | 5/5 | 是 |
| navigation/base:000 | 1→1 | official_done | 12 | 9 | 5 | 9/9 | 否 |
| navigation/base:001 | 1→1 | official_done | 12 | 8 | 1 | 11/11 | 否 |
| navigation/base:002 | 1→1 | official_done | 16 | 12 | 7 | 12/12 | 否 |
| navigation/base:003 | 0→0 | request_budget | 20 | 12 | 0 | 11/11 | 否 |
| navigation/base:004 | 1→0 | planner_error | 1 | 0 | 0 | 0/0 | 否 |

## 实际截图与记录

- [alfred：五题原图、编号检测图和裁剪](results-alfred.png)
- [habitat：五题原图、编号检测图和裁剪](results-habitat.png)
- [manipulation：五题原图、编号检测图和裁剪](results-manipulation.png)
- [navigation：五题原图、编号检测图和裁剪](results-navigation.png)
- [逐题结果、输入身份与离线审计](results.json)
- [运行清单与续跑记录](verification/run-records.json)
- [源码与配置身份](verification/frozen.json)
- [最终服务器与原文件保留检查](verification/server-after.json)
- [运行说明](README.md)

正常结束题逐请求重放模型消息、工具结果和动作计划，验证裁剪像素、编号失效及 ACK。操作性失败单列，保留已有观测、请求身份与未确认动作，不将其写成模型的负向判断。原实验文件和冻结源码保持字节一致，最终没有自有 worker。

## 结果解释

相对历史 RGB-D，由失败变成功 1 题，由成功变失败 6 题。退步题中 6 题结束于模型/工具协议错误。这些错误包括原文绑定、跳过检测阶段和 JSON 字段位置；检测图与裁剪的任务收益同时受到交互协议和固定请求预算影响。

初始题目与动作列表全部一致，RGB 像素完全一致 15/20，深度图完全一致 18/20。导航五题存在 RGB 差异，下表用 0–255 像素值的平均绝对差和米制深度差量化；这些题不属于初始图像逐像素相同的对照。

| 导航题 | RGB 平均绝对差 | 深度 PNG 相同 | 深度平均绝对差/m | 深度最大绝对差/m |
|---|---:|---|---:|---:|
| navigation/base:000 | 0.205549 | 是 | 0 | 0 |
| navigation/base:001 | 0.128680 | 否 | 1.11774e-06 | 0.129147 |
| navigation/base:002 | 0.178912 | 是 | 0 | 0 |
| navigation/base:003 | 0.014048 | 否 | 5.06308e-06 | 0.708044 |
| navigation/base:004 | 0.140908 | 是 | 0 | 0 |

## 上下文与实际成本

| 项目 | 历史 RGB-D | 本轮 R05 |
|---|---:|---:|
| 模型请求 | 75 | 220 |
| 有 token 记录的响应 | 75 | 220 |
| 单请求最大输入 token | 7,333 | 25,728 |
| 累计输入 token | 317,074 | 2,292,984 |
| 累计输出 token | 23,827 | 22,447 |
| 累计总 token | 340,901 | 2,315,431 |

请求数为历史的 2.93 倍，累计总 token 为 6.79 倍。这些是服务返回的真实用量，包含多轮重发的文本与图片上下文。

| 本轮请求阶段 | 请求数 |
|---|---:|
| 官方动作生成 | 60 |
| 首次提取与检测 | 20 |
| 裁剪或翻页 | 47 |
| 视觉复核 | 50 |
| 新观测检测 | 43 |

同一观测的后续请求包含官方提示、当前 RGB-D、感知协议和已完成的工具对话。Manipulation 另带未标注原始 RGB，后续轮次再带候选图或裁剪。最终动作仍由 Qwen 根据这些上下文生成。

## 实际终止原因

| 结束状态 | 题数 |
|---|---:|
| official_done | 11 |
| planner_error | 6 |
| operational_error | 1 |
| request_budget | 2 |

| 错误代码 | 题数 |
|---|---:|
| target_not_grounded_in_public_instruction | 1 |
| tool_and_action_branches_must_be_separate | 3 |
| worker_closed_outcome_unknown | 1 |
| detection_required_before_action | 2 |

| 题目 | 报错 | 最后响应根字段 | 包含动作计划 |
|---|---|---|---|
| alfred/base:001 | target_not_grounded_in_public_instruction | targets, tool_request | 否 |
| habitat/base:000 | tool_and_action_branches_must_be_separate | perception_review, selected_targets, tool_request | 否 |
| habitat/base:001 | tool_and_action_branches_must_be_separate | current_observation_serial, targets, tool_request | 否 |
| habitat/base:004 | tool_and_action_branches_must_be_separate | perception_review, selected_targets, tool_request | 否 |
| manipulation/base:000 | worker_closed_outcome_unknown | executable_plan, language_plan, reasoning_and_reflection, visual_state_description | 是 |
| manipulation/base:001 | detection_required_before_action | targets | 否 |
| navigation/base:004 | detection_required_before_action | targets | 否 |

`tool_and_action_branches_must_be_separate` 也会在工具响应含未允许的根字段时触发；错误代码本身不证明模型同时输出了动作。上表按真实响应字段区分。

原文绑定失败发生在检测和动作执行之前。以下保留触发失败的真实字段：

| 题目 | 公开原题 | 未匹配的 source_phrase |
|---|---|---|
| alfred/base:001 | Set plate with a spoon in it on the kitchen table | Set plate with a spoon in it on the kitchen table.; Set plate with a spoon in it on the kitchen table.; on the kitchen table. |

旧 Full 可匹配的 15 题：旧 Full 成功 6/15，本轮成功 9/15。该补充对照的动作与子目标协议不同。

- [逐请求阶段、输入规模与真实用量](context-costs.json)
- [启动目录故障与零请求证明](verification/setup-failure.json)

Manipulation base 0 在第 14 个动作发送后连接关闭：13 个动作已收到 ACK，最后一个动作结果未知。承载 worker 的 SSH 进程返回码为 255，未保存 stderr，因此退出根因未定。检测服务与自有进程状态随后重新核验，只继续了未启动题；该题未重跑，动作未重发。

- [运行故障、回执边界与续跑记录](verification/service-stop.json)
