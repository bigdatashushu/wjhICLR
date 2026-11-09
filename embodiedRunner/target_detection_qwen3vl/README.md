# Qwen3-VL 检测框与深度图辅助规划

2026-11-09 开发版采用官方规划输入加两张图。主规划复用
`official_direct.OfficialPlanner` 的官方任务、RGB、示例、动作反馈和输出解析；
增加一个简短 system 消息，user 末尾只增加同观测的深度图与带框 RGB。

system 的完整新增内容：

```text
After the official RGB, the two additional images are synchronized depth (0-10 m: brighter is nearer; black is far or invalid) and RGB with detection boxes. Detection boxes may be incorrect.
```

user 的文字、官方动作列表、示例和反馈保持官方构造结果。新增的两个内容块都是
`image_url`，不添加图注、目标/候选 JSON、裁剪图或工具对话。Manipulation 的第一张
RGB 保留官方编号/坐标标注，最后的带框图使用同观测的原始未标注 RGB。
检测框旁只显示类别名称，不添加 G/T 编号或置信度。所有保留的候选组都画在一张图中，
超过 12 组也全部展示；类别标签放在原图之外。未检测到目标时，仍提供同观测的 RGB。

主规划沿用官方 `executable_plan` 和完整原生动作列表，无新增必填字段。
不再要求 `targets`、`tool_request`、`perception_review` 或 `selected_targets`，
没有首次感知纠正轮，也不额外添加 ALFRED 持物策略。
已有显式整数动作编号及 `[编号, 描述]` 的兼容解析继续保留，不推测或改写动作编号。

正常执行流程：

```text
首次观测 → 独立模型请求仅从公开任务指令提取目标（一次）
        → Qwen3-VL 对当前原始 RGB 按目标类别检测，自动准备带框图
        → 主模型接收官方输入 + 深度图 + 带框 RGB，输出完整计划
        → 执行原生动作并保留 ACK、官方反馈和指标
后续规划 → 按冻结目标自动检测当前观测 → 主模型直接规划
```

独立目标提取使用配置中的规划模型 Qwen3.8-Flash，只收到公开指令文字，不接收图像、
匿名坐标或评测私有元数据。目标校验沿用 R05 的字段与指令来源约束，提取后保持冻结。
Qwen3-VL-8B-Instruct-FP8 的检测模型、逐类别提示词及坐标换算保持原逻辑。
自动预处理不执行环境动作，也不产生额外观测。

目标提取和主规划共用原任务的 `max_requests`，分别统计请求数及服务返回的
prompt/completion/total tokens；检测请求和检测 tokens 另行统计。`planner_steps`
仅计主规划，`request_accounting.total_model_requests` 包含目标提取。
预算在目标提取后耗尽时不调用检测器或执行动作；服务失败不重试。

`development-minimal.json` 保留原 120 题清单、服务配置和预算，将示例数恢复为
ALFRED 10、Habitat 10、Manipulation 4、Navigation 3。旧配置保留，新源码加载时
不再把示例数截到 3。配置中的检测分页和裁剪参数供独立 `detect-image` 工具使用，
主规划始终接收全候选图，不接收裁剪。

在 `embodiedRunner` 目录执行离线检查：

```bash
/Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_qwen3vl check \
  --config target_detection_qwen3vl/development-minimal.json \
  --source-root target_detection_qwen3vl/extension120-20261108-r01/inputs/embodiedRunner \
  --manipulation-catalog target_detection_qwen3vl/extension120-20261108-r01/inputs/manipulation_assets.json
/Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m unittest \
  target_detection_qwen3vl.test_contracts target_detection_r05.test_contracts -v
```

本次 29 项离线测试通过，验证四环境的官方消息与反馈一致、恰好两个纯图片增量、
公开指令目标提取、全候选画框、预算计数、故障不重试及完整动作 ACK。
120 题配置通过离线检查。本次没有实测新的成功率。
此前 compact20 的结果及 `validation-protocol-20261109.json` 属于旧版实验和兼容修复，
不能作为当前最小提示词版本的效果证据。

本次修改说明与离线核验数据位于：

- `/Users/bytedance/wangjh/具身评测最小提示词修改说明-20261109.md`
- `/Users/bytedance/wangjh/具身评测最小提示词离线核验-20261109.json`

冻结源码、历史配置和实验产物保留。当前源码身份记录 Qwen 包、复用的 R05 包及
`official_direct` 的四个直接依赖文件。后续部署应使用新的运行源码目录及 run ID；
`development-minimal.json` 中的远端路径和服务进程绑定是已有配置，本次未部署或刷新。

Qwen 的 0–1000 框按原始 PNG 的实际 `W,H` 转为浮点像素边缘：

```python
bbox = [x1 * W / 1000, y1 * H / 1000, x2 * W / 1000, y2 * H / 1000]
```

显示矩形时才限制到 `W-1/H-1`；独立裁剪工具仍按左上 floor、右下 ceil 加边距取像素。
倒置、退化、越界或非有限框会报错。模型未提供校准置信度，不添加分数或 DINO 阈值。
深度图沿用仿真器同观测、同相机的固定 0–10 米逆灰度映射；
原始米制深度保留在日志产物中，不加入规划文字。

规划授权继续从环境变量读取，检测服务授权仅在服务器进程内存中使用，
配置及结果不保存密钥。独立图片检测入口仍可使用 `detect-image`，
该入口不会调用规划模型或启动仿真器。
