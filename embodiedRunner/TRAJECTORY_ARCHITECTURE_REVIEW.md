# 五路轨迹审阅：逐轮快照架构的问题与整改顺序

2026-11-05。五个子 agent 各审阅三个任务的 Full/Direct 轨迹，覆盖两批合计 15 个任务、30 个最终实例。主线程核对原始事件、统计、历史记录及引用源码。原始证据根为 `/Users/bytedance/wangjh/embodied-comparison-20261105`，下文简称 `R`；源码根为本目录，简称 `P`。`eN` 指原轨迹的 `events[N].index=N`，不是动作步号。

**当前架构能完成真实单动作闭环，但状态延续、错误修复与感知证据消费没有形成稳定的决策机制。** Full 成功 8/15，Direct 成功 12/15；最值得先修的是这些接口和策略问题。现有证据不能把差距单独归于 Skill，也不支持“几何模块已经发挥收益”的结论。

## 1. 范围、分工与核验

| 子 agent | 实际审阅任务，每题含 Full/Direct | 分组报告 |
|---|---|---|
| 1 | Navigation base:000、001、002 | [报告](checks/trajectory-review-20261105/agent-1.md) / [JSON](checks/trajectory-review-20261105/agent-1.json) |
| 2 | Navigation base:003、004；Habitat base:003 | [报告](checks/trajectory-review-20261105/agent-2.md) / [JSON](checks/trajectory-review-20261105/agent-2.json) |
| 3 | ALFRED base:000、001、002 | [报告](checks/trajectory-review-20261105/agent-3.md) / [JSON](checks/trajectory-review-20261105/agent-3.json) |
| 4 | ALFRED base:003、004；Habitat base:000 | [报告](checks/trajectory-review-20261105/agent-4.md) / [JSON](checks/trajectory-review-20261105/agent-4.json) |
| 5 | Habitat base:001、002、004 | [报告](checks/trajectory-review-20261105/agent-5.md) / [JSON](checks/trajectory-review-20261105/agent-5.json) |

上下文预算事先写入 [REVIEW_SCOPE.md](checks/trajectory-review-20261105/REVIEW_SCOPE.md)：每组估计 8k–12k tokens，主线程估计 10k–14k。这是工作分配估计，未获得子 agent 实际 token 用量，不能当作实测预算。

本次实际执行的只读检查：

- 解析全部 46 份原始轨迹：30 份最终实例，1 份动作结果未知的中止实例，15 份未启动记录。46 份文件 SHA256 与原汇总逐项一致，事件索引连续。
- 从原事件独立重算 421 次最终 Qwen 请求、373 次动作提交、373 次动作后观测、12 次程序错误、36 次格式错误以及工具/感知计数；与原汇总核对无差异。含中止历史共 427 次 Qwen 请求。
- 逐份检查五个报告，255 条 finding/positive finding 的轨迹路径与事件索引均有效；这是引用位置核验，不能替代所有复杂表达式的值核验。子 agent 分别完成其字段核验；主线程另复查本报告的代表性事件和源码。
- 核对本报告引用的 10 个源码/方法文件，其当前字节 hash 均匹配三批冻结清单。三批整体源码身份不同，不能把整棵当前代码树宣称为同一冻结版本。
- 核对三批的模型、方法、感知配置及预算 hash 一致。读取已有 ALFRED 恢复诊断收据，未重新回放。

本次没有 SSH、模型/感知请求、仿真启动或复跑；没有修改源代码、官方数据、Skill、配置和原轨迹。新增内容限于本目录的报告及分组审阅产物。没有读取凭据。

## 2. 独立重算后的结果

| 环境 | Full 成功 | Direct 成功 | Full / Direct 动作 |
|---|---:|---:|---:|
| Navigation | 2/5 | 4/5 | 71 / 62 |
| ALFRED | 2/5 | 4/5 | 95 / 47 |
| Habitat | 4/5 | 4/5 | 50 / 48 |
| 合计 | **8/15，53.3%** | **12/15，80.0%** | **216 / 157** |

同任务结果为：双方成功 7 题，Full 独成功 1 题，Direct 独成功 5 题，双方失败 2 题。Full−Direct 为 −26.7 个百分点，仅描述这 15 个开发任务。

| 行为/成本口径，最终 30 实例 | Full | Direct |
|---|---:|---:|
| Qwen 请求 | 228 | 193 |
| Qwen 实报 token | 1,008,147 | 1,006,747 |
| 实例墙钟累计秒数，沿用原收据 | 1,744.162 | 1,311.196 |
| 预感知请求 | 678 | 0 |
| 工具调用，按唯一实际调用计 | 11 | 0 |
| 深度/点距离工具调用 | 0 | 0 |
| ReadObservation / Yield | 0 / 0 | 0 / 0 |
| 程序错误 | 12 | 0 |
| 模型格式错误 | 0 | 36 |
| 官方 invalid 计数 | 49 | 42 |
| 动作后反馈判为失败，含 Navigation 碰撞 | 61 | 16 |
| 观测数 | 231 | 208 |
| 成功提交的 memory_update 次数 | 5 | 0 |

官方 invalid 与动作失败不是相同口径：Navigation 的 invalid 字段一直是 0，但 Full/Direct 分别有 12/10 次失败动作反馈；家务环境 Full 的 49 次 invalid 均为动作失败，Direct 的 42 次由 36 次格式错误和 6 次动作失败构成。不能根据“49 对 42”判断双方动作策略接近。

Full 11/15 个实例全程记忆为空，231 个观测中的 193 个记忆为空；5 次成功记忆更新仅分布在四题。Direct 的 public_memory 同样为空，但其消息另有自动累计的实际动作反馈历史，不能拿 public_memory 空值推断它无历史。

678 次预感知分为：深度估计 218、检测 200、分割 130、类别审阅 130。11 次工具调用全部是 `view_candidates`，8 次返回空候选、3 次有候选。执行轨迹中继承保存的 results 会重复出现，去重后仍是 11，不能加总为 18 次调用。0 次 ReadObservation 也不单独证明工具未使用，因为同段程序可以直接读取工具返回的 data；本次更强的证据是没有数值几何工具调用，且多个程序误读候选结构。

## 3. 优先整改的问题

### R01 / P1：跨轮状态依赖主动记忆，成功动作后容易失去“刚做了什么”

**已确认实现问题。** `P/action_solver.py:92–119` 给 Direct 自动累计最近至多 40 条实际动作及反馈；`135–152` 给 Full 的是当前 snapshot 和 repair。`P/action_host.py:73–87` 只继承现有记忆，不自动添加上次已执行动作身份。家务成功反馈常只有“Last action executed successfully”，当前图像无法稳定恢复持物、已到达目的地和剩余子目标。

**已确认行为。** Habitat 盘子任务 `R/batch-2/comparison-episodes/trajectory-002-Full/trajectory.json`：

- e28 确认持有 plate，记忆为空。
- e44、e60、e76 的图像 hash 完全一致，当前反馈只有成功；对应程序重复导航右柜台，最后才放置。Full 6 个动作，Direct 4 个。

Habitat 橙子任务 `R/batch-3/comparison-episodes/trajectory-004-Full/trajectory.json`：e80 已确认持有 orange；e107 放到错误的 chair 1，e136 又尝试持物时拾取，e172 在不合适位置放置失败；e257 官方 invalid=10、progress=0.75 后结束。

**不能只补“非空记忆”。** ALFRED 毛巾任务 `R/batch-3/comparison-episodes/trajectory-003-Full/trajectory.json` 的 e227 已送达“持有 HandTowel，找浴缸再放置”的记忆与子目标，e231 仍提交拾取，e232 再次得到正在持物的错误。状态送达和状态被动作策略遵循是两件事。

建议：

1. 宿主维护有界的公共执行历史：实际 action_id/name、ACK、公开成功/失败反馈、来源 observation_revision。未执行计划后缀保持 pending。
2. 分开记录已确认持物、当前待完成子目标、尝试结果。程序的动作前 memory_update 不可提前宣布动作后完成；ACK 后按公开反馈更新/撤销。
3. 旧图像定位保持 historical；持物变更、放下等会影响逻辑状态，不可仅依靠文本 confidence 标签让模型自己消解冲突。
4. 原图相同且反复失败时记录“这个观测下已试过的动作”，帮助切换策略；不由失败推断永久无目标。

验收：用已有轨迹离线回放状态更新，确认“取得→导航成功→放置”期间持物及待放置子目标进入下一次请求；成功放下后旧持物记录失效；已持物再拾取的矛盾可被识别。然后再看真实新轨迹是否减少重复。因果收益仍待后续实测。

### R02 / P1：家务格式错误重开决策，解析失败诊断被丢掉

**已确认实现问题。** `P/action_solver.py:173–191` 要求 `executable_plan`；`P/action_host.py:185–191,306–316` 对家务格式失败执行 invalid mutation 并重开外层决策，`207` 把 feedback 清空。下一请求仍看到上次物理动作成功，缺少字段错误诊断。

主线程核对最终 36 次格式拒绝：其中 35 次还有后继请求；这 35 次后继请求的完整 messages hash 都与拒绝前相同。它们是宿主启动的新请求，不是网络自动重试。

具体结果：`R/batch-3/comparison-episodes/trajectory-000-Direct/trajectory.json` e155 的语言意图是放置纸卷，但输出键为 `plan`；e156 增加第 10 次 invalid，环境仍在 step 20，e157 官方结束、progress=0.5。可以确认该次放置没有被执行，不能据其文本认为物体已放好。

此外，Full 的普通程序修复只有 code/line/yielded IDs，见 `P/action_host.py:263–287`；模型收到新 system/user，通常没有前一程序、出错语句和实际响应 schema。已有 ResultStore 并不意味着新程序知道该读哪些结果。

建议：单独保存有界 `last_protocol_error`，跨 invalid revision 传递；包含要求字段、有限错误分类、前一程序的有界出错片段及当前合法结果 ID/schema。保持服务异常原文及凭据不进入反馈。不能静默接受替代字段后继续称原 baseline。

验收：`plan`/`action_plan` 等错误后，新请求明确携带解析诊断；没有 step；invalid 和模型请求各计一次；不发生自动重复动作。原 Direct 协议保留，新格式策略另冻结版本。

### R03 / P1：预感知已付成本，但数据几乎没有进入决策

**已确认实现与行为。** `P/action_host.py:140–159` 在生成程序前预计算；`P/action_solver.py:144–151` 的请求传 manifest 身份与能力，真实候选、深度和点仍需程序查询。Full 228 次真实请求均带完整环境 Skill，只有 11 次候选查询，0 次深度/距离查询、0 次 Yield。多个成功任务也全程零工具，不能用成功反推几何有效。

这里至少有两个不同缺口：

1. **任务类别解析遗漏。** `P/evidence.py:27–60` 的固定词表没有 toaster、ladle、lid；`countertop` 也不覆盖部分指令里的 counter。Toaster 任务 `R/batch-2/comparison-episodes/trajectory-003-Full/trajectory.json` e3 类别 hash 对应空列表，prepared candidates 为空，18 次感知全是深度估计，没有请求 toaster 检测。这不是检测器已经检测目标后漏检。盖子任务实际检测的是 cabinet/refrigerator/chair/table。检测后的裁剪还会按类别优先和置信度只保留一个总候选，见 `P/evidence.py:186–196`，不等于每个任务角色都有候选。
2. **产出与消费脱节。** Habitat 盘子任务 e44/e60/e76 同图，后续仍按新 revision 重做深度、检测、分割、审阅；程序未读结果。现有缓存 scope 包含 decision/revision，见 `P/evidence.py:140–164,244–246`。

建议：先做公开任务对象/容器角色的覆盖核验，再提供明确的查询触发与短证据摘要；按需感知或缓存纯像素计算产物是可选设计。即使复用同像素产物，也必须重新绑定当前观测身份和授权，不能复用旧 geometry/result 的当前使用权限。

验收：已知任务词静态解析正确；空候选区分“未请求 / 请求无检测 / 未确认”；至少出现可追溯的 tool→data 分支→动作链。以实际消费衡量效果，不把强制工具调用次数当收益，不将类别确认授权为物体表面或米制可达性。

### R04 / P1：模型面对的程序 API 缺少足够具体的结构与写入实例

**已确认行为及可改进的交付缺口。** 工具可调用、沙箱能执行；问题不是 Linux 路径尚未接通。12 次程序错误为 11 次 `violation_runtime`、1 次 `violation_syntax`。典型错误包括：

- Toaster e8：返回 data 是含 candidates/count/coverage 的字典，程序用 `r['data'][0]`，并读取不存在的 `box`。正确结构是 `r['data']['candidates']` 与候选 `box_xyxy`。
- Toaster e164/e174 继续误读结构，并用宽泛异常回退转向；e167 右转、e177 左转回到相同原图。receipt 无异常不代表正常候选决策。
- 生成程序尝试向冻结的 memory.records 原位 append，或给 memory_update 传字符串/缺字段记录，严格 validator 拒绝后动作也无法提交。
- 序列化 snapshot 的动作表是记录，运行时 ctx.action_table 是字符串 tuple。system 已有说明，但当前最小示例只演示无记忆更新提交，缺少完整候选与记忆操作实例。

源码：`P/action_solver.py:22–44`、`P/action_tools.py:38–56`、`P/action_contract.py` 的 `freeze`、`validate_memory`、`ActionPayload`、`validate_action_payload`。三份方法正文说明读 data，却没有候选条目的完整结构示例。

建议：交付与运行共用一份精确 schema 示例；补齐非空/空候选、正确 action_table 枚举、复制后写入记忆、六个溯源字段和不更新时的合法值。保持严格校验和数值边界。异常后记录有限原因，不让固定转向回退伪装成正常证据策略。

验收：不调用模型/服务的离线合同测试证明这些示例可执行；空候选不抛错，冻结容器不能被修改，合法写入可提交。实际生成程序是否更稳定，需后续新协议轨迹确认。

### R05 / P1：ALFRED 的官方动作前置语义没有转为明确任务状态

**已确认官方语义与不适配行为。** 这项不是官方评分器 bug。`P/upstream-household/embodiedbench/envs/eb_alfred/thor_connector.py:72–103`：

- 除 put/open 外的动作先清空 `cur_receptacle`。
- find 重新设置目标。
- pickup 会清空目标，即使 pickup 随后失败。
- 没有目标时 put down 走 drop 分支，放下成功不代表放到了要求的容器。

ALFRED base:001/002 的 Full 出现找容器后再尝试拾取、接着直接放下的链条；agent-3 的 G3-02 给出 e151→e167、e25→e35 等具体引用。Direct base:002 同样有未执行 find 后缀却直接 put 的行为，见 G3-07。

清洗也有任务前置语义：ALFRED base:000 Full e138 明确尝试“持有 ladle 时打开水龙头”，没有先放入水槽；反复开水不等于完成清洗。官方清洗依赖 SinkBasin 内对象，源码依赖见 agent-3 G3-03。

建议：环境适配层提供公开控制语义的有限说明，方法明确区分 acquire→find destination→put，以及处理任务的容器/开关前置条件。由已执行动作推导的目标失效可进入任务状态；不能读取隐藏 `cur_receptacle` 或 object metadata，也不能替换官方动作/评分。

验收：fake/offline 顺序核验覆盖失败 pickup 清空目标、put/drop 区别、清洗前提和多对象实例计数；状态只依据公开动作与反馈。正式执行仍由固定官方 worker 完成。

### R06 / P2：预算只有宿主可见，重复与修复消耗没有形成策略信号

**已确认接口缺口。** `P/action_contract.py:235–246` 只有 budget_sha256，`P/action_runtime/execution_contract.py:26–28` 的 runtime 字段没有剩余动作、模型请求、invalid、墙钟或修复额度。宿主严格执行预算，但模型不能据此调整探索。

Navigation Pot、Toaster 分别在 17 个动作后用完 20 次 Qwen 请求，终止为 `budget_exhausted:qwen`；各有 3 次程序错误，见两个轨迹的末尾 e223/e187。其他导航任务还有反复碰撞和来回旋转。不能将此说成超预算，现有停止规则正常工作。

建议：提供只读的剩余额度和本轮修复次数；对同观测下重复失败设置有界策略切换信号。总请求成本保留，不能把 repair 隐藏或改为预算外调用。

验收：快照预算随每次计费单调变化；不增加权限；接近用尽时模型可见额度；重复动作检测区分主动复查、环境失败、同图循环。更改总预算需要两臂新运行共同固定。

### R07 / P2：比较存在文本历史、初始观测和控制提示差异

**已确认实验条件差异。**

- Full 与 Direct 文本历史不对称，见 R01。`current_only` 表示当前原图，不意味着相同文本上下文。
- ALFRED/Habitat 10 个任务初图两臂一致；Navigation 5 个任务初图均不同。因此不能把导航配对差异只归于方法。初图一致的家务 10 题是 Full 6/10、Direct 8/10，同样只是小样本观察。
- Direct 保留官方批计划提示，宿主却验证整计划后只执行第一动作。实际后缀不能写成已完成历史；若改提示为明确单动作协议，需要新的 baseline 版本，见 agent-1 G1-F07、agent-3 G3-07。
- ALFRED 恢复批关闭了未使用的官方 depth/class 渲染。已有诊断支持命令/评分/初图一致；逐动作状态 hash 不同，末帧有像素差异，不能说完整轨迹完全等价。跨批墙钟差不能全部解释成架构计算成本。

建议：继续按用户选择只比较 Full 与官方 Direct。新协议明确公共历史范围、单动作调度与输出约束；保留原始 Direct 和原结果。Navigation 查明 reset 差异后再做严格配对；方法改善不得使用这 15 题继续作为独立泛化测试。

### R08 / P2：汇总成本与无效动作需要更清楚的口径

**已确认账本限制。** 130 次类别审阅 usage_missing=true。Full 的模型类请求是 228 次求解 Qwen 加 130 次类别审阅，总计 358 次；Direct 为 193 次。1,008,147 对 1,006,747 是求解 Qwen token，不是完整架构全部模型 token/金额。

原汇总保留 1 次 `action_outcome_unknown` 和 15 次未启动记录，处理正确。中止实例有 6 次 Qwen 请求、41,100 token、5 次动作提交但仅 4 次 ACK；重置恢复后的实例不能覆盖这段未知动作历史，也不能把未启动当物理失败。

建议：继续分别保存 attempted/ACK/failed-action、format-invalid、program-error、service-error、各角色请求/token、未知 usage、历史恢复成本。任务分母与实际执行成本采用独立字段。最终选用结果与首次尝试历史同时可见，不能从只统计完成实例的耗时得到稳定效率结论。

## 4. 已有效的边界应保留

此次轨迹支持以下正向事实：

1. 373 次最终实际动作提交各有一个动作后观测；包含动作失败后的 RGB/反馈。36 次格式 invalid 不执行物理步，另产生观测修订。无需再次用“环境是否返回图像”解释缺少状态连续性。
2. Full 全部 228 次求解请求实际交付完整冻结环境 Skill；不能解释为 Skill 未发送。普通/repair 都走真实模型请求和计费。
3. 官方成功由宿主/环境判定；当前成功例说明桥接、程序提交及官方结束可工作。Habitat base:000 Full 12 动作成功、Direct 30 动作失败是一个真实反向案例，但没有几何消费证据，不能宣称几何导致胜出。
4. 当前几何按观测修订撤销，相对深度没有冒充米制世界几何。改善状态延续应保留历史来源，不恢复跨轮旧工具授权。
5. 动作结果未知时停止、保留历史，无自动重发动作。此边界应继续保持。

## 5. 推荐实施顺序和通过条件

下表是后续整改建议，本次未实施或执行验收。所有主要修改可集中于 `embodiedRunner/`，不必先改共享 QA runner 或全局 Skill snapshot。

| 顺序 | 文件/接口 | 具体工作 | 最小通过条件 |
|---|---|---|---|
| 1 | action_contract.py、state.py、action_host.py、action_solver.py | 公共实际动作历史、ACK 后逻辑状态更新、pending 与完成分离、逻辑失效规则 | 轨迹离线回放保留持物/待放置，release 后不沿用旧持物；无隐藏 metadata |
| 2 | action_host.py、action_solver.py | 跨 invalid revision 的解析诊断；普通程序修复的有界代码/schema/合法结果提示 | 格式失败后请求正文不再原样；无多余物理动作；请求和 invalid 账本一致 |
| 3 | action_solver.py、action_runtime/execution_contract.py、methods/*.md | 统一工具/候选/记忆示例、环境动作前置语义 | 正确候选读取、冻结记忆复制写入、失败 pickup 和清洗顺序合同离线可验证 |
| 4 | evidence.py、action_tools.py、action_host.py | 任务词及角色覆盖、感知触发、产物与授权分离的同像素缓存、消费指标 | 不再遗漏已知目标；实际 tool/data/动作依赖可追溯；旧授权保持撤销 |
| 5 | budget.py、action_contract.py、action_host.py、action_audit.py | 只读剩余额度、循环信号、统一失败与角色成本字段 | 预算可见且不可扩张；碰撞/格式/程序/未知动作分开；审阅 usage 缺失显式标注 |
| 6 | action_live.py、独立新 config/run 目录 | 原 Direct 对新 Full 的小样本对比，保持模型和总预算固定 | 先离线合同及三环境 smoke，再执行新轨迹；保留旧结果和失败历史 |

修状态和修复反馈应先于扩大感知与重建规模。对本批开发轨迹的适配改善，可先离线确认接口正确，再用新任务看结果；无需加入 B01/B11。优先检查重复动作、动作前置条件、每次有效决策成本，再判断成功率变化。

## 6. 来源与恢复记录

三批整体冻结源码身份：

- 上一批：`514adda9df8f469c01ab2da55cbb472179cfcc70a1bbe03779839881a1715ec0`
- 新增原批：`8a52f291046c9cb569a7df40637a586b0df721b711d00f95bd2d32929ff2e783`
- 恢复批：`7f5ec984c11b6534250222d5c647c66d4b9f694d64a78f245b6154358314b656`

身份依据分别为 `R/comparison.json`、`R/batch-2/comparison.json`、`R/batch-3/comparison.json` 的 `manifest.implementation_source`。10 个具体引用文件哈希和重算指标见 [TRAJECTORY_ARCHITECTURE_REVIEW.json](TRAJECTORY_ARCHITECTURE_REVIEW.json)。

完整轨迹索引：`R/all-trajectories.json`。最终逐题选择及原历史映射：`R/combined-results.json` 的 `rows` 与 `retained_history`。恢复诊断：`R/batch-3/recovery-diagnostics/{baseline,candidate}/receipt.json`。
