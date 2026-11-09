# 子目标任务卡接入

2026-11-06。新 live 运行的 Full/B11/B01 使用 `embodied-task-card/2`。v2 每轮发送完整动作表、最近 40 步实际动作与完整公开反馈，以及同轮进度判断；Direct 保留官方提示和原输出协议。所有修改位于 `embodiedRunner/`；历史阶段的回执按其冻结源码解释。

## 长度合同

**整次请求文本目标为 16,000 个 Unicode 字符。** 完整 Skill 与精简协议、完整动作表、原任务/状态/40步反馈、精选官方示例/按需证据共享额度；约 5500/5000/4000/1500 的分区目标允许动态借用。该目标包括 JSON 键、标点和转义，图片 token 另计。必要内容超出时完整发送，标记 `required_fields_over_budget=true` 和 `prompt_text_over_target=true`；原 96,000 字符执行上限仍保留。

`task_card.py:build_card` 完整保留总任务、当前子目标、已确认状态、最新反馈和错误分类、动作表、匿名官方坐标与最近 40 步动作反馈。逐步反馈保留完整公开字符串；ACK/decision hash 留在不可变运行状态及原始轨迹，提示中只发送修订、动作ID/原名、成功标记和反馈。没有字符串硬截断。`build_legacy_card` 和 v1 校验器保留历史格式兼容。

动作表按原始顺序，以每行 `ID:原动作名称` 紧凑发送，`catalog.omitted=0`。全部名称和 ID 同时在 `ctx.action_table` 中可用；匿名官方坐标完整装入 `objects.entries`。兼容 `catalog_request`，但正常请求已经包含全部动作，提示明确说明目录请求消耗一次调用。

Yield 的结果页和 ID 按完整页面/ID装入剩余额度，保留遗漏计数。精选官方示例来自哈希核验的公共示例文件，确定性选择至多两条完整行摘录，记录来源 hash 和索引；长示例不切句。可选结果页和示例不能挤掉完整动作表及 40步历史。

每次请求重新序列化并校验结构。回执同时记录正文长度、整请求 `text_chars`、目标/超长标记、完整目录、历史和示例数量与正文 hash。轨迹审计将发送内容与对应不可变快照逐项对照，核验原任务、子目标、事实、历史反馈、全部动作名称/ID、前轮进度和真实字符数。

## 每轮上下文与决策

任务卡包含总任务、当前图像对应的观测身份、宿主确认的状态和最近实际动作、当前子目标、完成条件、剩余预算、循环信号、当前证据摘要和完整的目录条目。完整图像通过一个 image_url 交付；原始 RGB 始终是预测感知的输入。

Qwen 一次输出：

```json
{
  "subgoal": {
    "id": "get_plate",
    "goal": "Acquire the plate",
    "object": "plate",
    "destination": "",
    "complete_when": "holding"
  },
  "transition": {
    "kind": "keep",
    "source": "none",
    "observation_revision": 0,
    "detail": ""
  },
  "progress": {
    "completed": "Located the plate from the current image",
    "remaining": "Pick up the plate and move it to the counter",
    "next": "Pick up the plate"
  },
  "perception_request": {
    "targets": [{"role": "object", "category": "plate"}],
    "depth": false
  }
}
```

`progress` 与动作程序在同一次调用中返回，包含 completed/remaining/next 三个短字符串，分别不超过 240 字符；缺失不增加动作门禁。宿主保存为带修订的 `model_assessment`，随下轮输入传递，不能覆写官方事实。`program_source`、`perception_request`、`catalog_request` 三者只能选择一个。感知、目录和修复均进入原预算；仍最多 3 次生成/修复，服务失败不自动重试。

## 自动状态与动作检查

`SubgoalLedger` 根据实际动作的 ACK 与公开反馈记录持物状态、控制目标、上次导航和近期动作，保留来源修订和动作身份。未执行计划后缀不写成已完成事实。模型的子目标与视觉完成评估单独记录，不能覆写宿主确认的状态或官方评分。

- 持物事实跨“导航成功”的通用反馈保留；成功释放后清除。已确认持物再拾取、空手释放会进入有界修复，拒绝的动作不执行、不增加真实 step。
- ALFRED 的控制目标按公开连接器语义失效：多数交互会在执行前清空目标，包含失败 pickup。需要指定容器的 put 必须保留匹配的控制目标。
- 当前子目标的 advance 必须有来源明确的支持；成功的官方 find 可以完成匹配对象的 find/locate 子目标，夹爪运动与导航成功不能作为抓取成功。其他视觉支持保持模型评估身份，任务是否成功仍由官方环境决定。
- 相同画面下反复导航或失败、重复的导航画面与动作序列会产生策略切换信号。宿主拦截继续该循环的预测动作，返回修正提示；拒绝的动作不执行。成功交互和静止相机本身不自动等同于无进展。
- 本次响应接受的子目标在执行程序前同步到不可变 `ctx.task_state`，并进入下一张任务卡；程序从 `ctx` 读取快照身份。响应中的意图不会写成已确认的持物或几何事实。

## 动态感知

类别来自 Qwen 当前子目标请求，支持 `star`、`silver container` 等固定词表之外的英文名词短语。每个 decision 至多 1 次感知请求，最多 4 个独立类别/角色；每个类别最多保留 1 个候选，SAM 批量处理，类别审核逐候选计费。深度只在 `depth=true` 时请求。原预算保持不变。

摘要区分未请求、无检测、类别未确认、类别确认但 mask 质量未审。没有启用独立分割质量审核，也没有授予物体表面或米制几何权限；旧几何和结果在动作后照常撤销。

## 入口与验收

新 live 入口默认 `task_cards=true`，需要保留旧当前快照协议时在独立配置显式设置 `task_cards=false`。直接构造 `ActionSolver` 默认保持旧协议兼容，应用入口负责选定模式。模型参数和动作预算未扩张。

离线四环境检查：

```bash
env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp \
  PYTHONPATH=/Users/bytedance/wangjh/wjhICLR/src:/Users/bytedance/wangjh/wjhICLR/embodiedRunner \
  /Users/bytedance/wangjh/wjhICLR/.venv/bin/python3.11 -B \
  /Users/bytedance/wangjh/wjhICLR/embodiedRunner/action_run.py \
  --fake --task-cards --process-worker \
  --output /Users/bytedance/wangjh/wjhICLR/embodiedRunner/checks/new-task-card-run.json
```

输出必须不存在；完整公开轨迹自动保存在同名 `-episodes` 目录。`audit_action_trajectory.py --run <JSON>` 复核实际模型消息。

前期验收包含真实隔离程序、四环境 NDJSON fake 闭环、原真实公开轨迹回放、中文/转义字符/长历史/长目录/长修复页测试及完整回归。回放只重建上下文和识别已有动作矛盾，不生成新模型决策，不声称原失败任务已被修复。最新 v2 真实复跑结果见文末。

取消拒发前的阶段结果：完整回归 **203 passed、1 skipped**；当时源码四环境三臂离线闭环 **12/12**，运行和完整轨迹独立审计均通过。实际发出的 32 张新协议卡片为 1,178–1,988 字符；16 份原公开轨迹回放重建 229 张卡片，最大 **3,912 字符**。原阶段收据保持原样；新的长必填字段用例验证完整发送、宿主继续执行和审计接受，不再沿用原拒发验收口径。

原阶段验收见 [task-card-acceptance-20261106.json](checks/task-card-acceptance-20261106.json)；原完整离线轨迹入口为 [trajectories.json](checks/task-card-process-r03-episodes/trajectories.json)，长度回放证据为 [task-card-archived-replay-r03.json](checks/task-card-archived-replay-r03.json)。

取消必要字段拒发后的新源码验收见 [task-card-overflow-acceptance-20261106.json](checks/task-card-overflow-acceptance-20261106.json)。完整回归 **207 passed、1 skipped**。四环境常规输入与超长必要字段的 fake 闭环均 12/12，通过完整轨迹独立审计；32 张超长卡片实际发送长度为 10,117–10,650 字符，完整保留总任务及 JSON。已有 229 张公开轨迹卡片按新规则重建后最大 3,948 字符。该结果验证长度不会中止模型调用与宿主执行，未开展真实 Qwen 复跑。

## 停止实验后的状态与导航修复

保留 `embodied-task-card/1` 的 JSON 字段兼容；状态同步、结果重绑、子目标完成与循环门禁语义写入 `POLICY`，由新运行的协议、配置与源码 hash 记录。原服务器源码、冻结清单和已停止的 15 题运行不改写。

`ActionHost._sync_task_state` 在接受子目标后立即同步，也在同 decision 的下一次请求前检查同步。Yield 留下工具结果时同样执行，消除了“有结果就跳过同步”造成的 `subgoal:null` 和旧子目标。`ResultStore.rebind_task_state` 必须证明新快照仅改变 `task_state`，重新验证原结果的授权与 hash，然后保留 result ID、内容和授权，更新 scope 与行 hash，撤销旧 store。读次数保留；观测、decision、预算、能力或证据改变均不能走此重绑。完整轨迹新增 `result_context`，独立审计检查前后上下文及完整结果集合，只允许任务状态与其快照身份变化。

`keep` 保持 id、object、destination、complete_when，允许 goal 措辞调整；真正的目标或完成条件变化仍需 revise。官方成功 find 仅支持匹配对象的 find/locate/identify/search 子目标，不支持放置、其他对象或持物完成。命名导航的 ACK 需匹配当前 waypoint；兼容旧响应在 goal 中写“Navigate to the sink in the kitchen to locate the plate”、而 destination 保留最终 right counter 的情况。通用 MoveAhead、Rotate ACK 不证明到达目标。拒绝时交付完成条件、当前 waypoint 和修正提示。

循环检测只读取公开像素 hash 与实际动作，不使用隐藏位姿或目标位置。内部保留最多 25 个导航观测，检查周期 2–12：至少三张连续重复的画面及其两个连接动作匹配，才产生 `cycle_period` 和 `cycle_next_action`。无需等第二轮路线完全结束，能覆盖预算耗尽前出现的部分重复路线。连续相同画面的同一导航动作或失败动作也保留原重复计数。宿主只拦截继续已识别循环的动作，允许模型提交其他合法动作；状态变化会重新评估循环信号。

本次离线验证脚本为 [task_card_repair_validation.py](checks/task_card_repair_validation.py)，覆盖真实隔离程序中的空候选 Yield、目标措辞调整后的 ReadObservation、四环境 fake 进程闭环，以及停止实验留下的 7 份完整公共轨迹。回放沿原始观测、动作和提案推进修复后的账本，检查门禁与卡片；它不产生新的环境轨迹或成功率。

最终固定源码验收：完整回归 **236 passed、1 skipped**；四环境进程闭环及专门的 Yield 结果重绑闭环共 **15/15**，运行和完整轨迹审计全部通过；7 题回放的 **63 张**任务卡全部完整，最大 **3,677 字符**。Navigation 002 在原第 16 步识别到周期 6，可拦截原第 17、18 步的继续循环动作。证据见 [task-card-repair-summary-r02.md](checks/task-card-repair-summary-r02.md) 与 [task-card-repair-acceptance-r02.json](checks/task-card-repair-acceptance-r02.json)。未恢复真实模型或官方仿真运行。

## v2 完整上下文真实复跑

`task-card-15-full-context-r04` 已完成 **15/15**，成功 **1/15（6.67%）**，与修复版 r02 总成功率持平；Navigation 004 从成功变失败，ALFRED 004 从失败变成功。Qwen 直答和旧 Full 存档分别为 12/15、8/15。各方案沿用相同模型、预算、参测 Skill 和任务输入；与 r02 相同初始 RGB 的任务为 10/15，因此这批已用开发任务用于诊断比较。

145 次请求全部发送完整 Skill、动作表和精选官方示例，最近 40 步实际动作及完整反馈与不可变快照逐项核对。请求平均 12,146.4 字符、最大 15,850 字符，超目标 0 次。135 次有效提案同轮返回进度，130 次后续请求携带前轮进度；执行时子目标同步不一致为 0。本轮请求前最多已有 17 步，未触发历史上限。

12 题耗尽调用预算，2 题以空计划停止，1 题由官方环境确认成功。修复请求仍出现 13 次子目标完成证据不足和 4 次重复感知；上下文交付已通过核验，总成功率没有改善。程序错误降至 0，格式错误为 10；调用量减少需结合提前结束解释。

试跑 r03 的 `observation_revision=current` 表达曾导致模型输出字符串 `"current"`。该试跑在确认提示歧义后停止并保留完整与中断轨迹；r04 明确整数类型及逐请求具体数值，在新冻结目录运行。相关提示检查 48 passed、Linux 检查 69 passed/1 skipped；此前完整回归 239 passed/1 skipped。真实运行和全轨迹审计通过。

详见 [完整结果与逐题对比](runs/task-card-15-full-context-r04-comparison.md)、[实际上下文覆盖](checks/task-card-15-context-live-coverage-r04.json) 和 [最终验证](checks/task-card-15-context-final-verification-r04.json)。
