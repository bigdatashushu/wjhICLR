# harness3D v7 实现与实测报告

> 日期：2026-09-22｜对应目标态：《系统架构v7.md》｜性质：实现对照 + 真实实测报告
> 纪律声明（§2.1）：本报告把 **implemented / connected / 真实验证 / 性能结果** 分开写。
> 所有"实测"数字都来自本次真实运行落盘的 trace；未运行的项明确写"未验证"。

---

## 0. 一句话结论

v6 的工具臂（C1）比直答基线（C0）**低 22.8 分**（18.36 vs 41.17，逐题配对 C0 赢 50 / C1 赢 8）。
逐题溯源定位到**六个各自独立、可复现**的根因（其中两个是**官方口径写错**，
一个是**检测器词表缺项**导致题面物体解析失败）。本报告记录这些根因的证据、
v7 的修复、以及修复后的实测结果。

---

## 1. P0 核验：现状报告的真伪（§17.3）

《系统状况报告0922.md》的核验结论（逐条 `confirmed / refuted / unresolved`）：

| # | 报告陈述 | 结论 | 证据 |
|---|---|---|---|
| 1 | 128 题 real 运行完成，拒答率约 32.03% | **confirmed** | `data/v6_inner128/c1_seed0/traces/evaluation_result.jsonl` 逐题统计：coverage=0.6797（abstain=41/128=32.0%） |
| 2 | 存在 19 个工具契约错误 episode | **confirmed** | 同 run：`tool_contract` 命中 19/128，`answer_source=abstained` 的 episode 与之重合 |
| 3 | C0 基线未完成 | **refuted（已过时）** | 报告时点之后 C0 三个 seed 已跑完：`data/v6_inner128/c0_seed{0,1,2}` 各 128 题，coverage=1.0 |
| 4 | 活跃 Skill 快照仍为 genesis | **confirmed** | 6 个 run 的 `active_snapshot_ref` 均为 `genesis`；`data/active_snapshot.json` 不存在 |
| 5 | 检测服务返回空列表的错误路径 | **confirmed，且已定位为性能主因** | 见 §2.2：逐题补漏**只**走 VLM 出框，而该 8B 模型出框不可靠 → 题面物体进不了清单 |
| 6 | Docker 沙箱未接线（`use_docker` 无人读取） | **confirmed** | 全库仅 `runner.py` 一处字段定义，无读取点；M10 仍走进程内 kernel |
| 7 | G5 固定 `not_available`、G8 不恢复 | **confirmed** | `schemas/reconstruction.py` 的 `Literal["not_available"]` 硬校验 |
| 8 | 全部能力 `paper_eligible=False` | **confirmed** | `readiness/manifest.py:CURRENT_BASELINE` 自述表 |
| 9 | 尺度融合 23/24 场景成功 | **confirmed** | 24 份 `*_scale_receipt.json` 中 23 份 `success` |
| 10 | seed1/2 正在运行、噪声底未出 | **refuted（已过时）** | 三个 seed 均已完成 |

**新增发现（报告未提及、但影响最大）**：工具臂的失败**不是**"工具不好用"，
而是六处**接口/口径级缺陷**在系统性扣分。详见 §2。

---

## 2. P0 根因：为什么 C1 比 C0 低 22.8 分

逐题型配对（同 128 题、同帧集、同模型，v6 代码）：

| task | C1 | C0 | Δ | 主因（证据见下） |
|---|---|---|---|---|
| object_counting | 0.300 | 0.787 | **−0.487** | R3 实例重复未合并 |
| object_abs_distance | 0.312 | 0.438 | −0.125 | R4 米制值静默 None + R2 相机口径 |
| object_size_estimation | 0.325 | 0.581 | −0.256 | R4 + R5 点云污染 |
| room_size_estimation | 0.281 | 0.613 | −0.331 | R4 |
| object_rel_distance | 0.062 | 0.188 | −0.125 | **R1 官方口径写错** |
| object_rel_direction | 0.125 | 0.375 | −0.250 | **R1b 难度模板选项集合错** + R6 物体解析失败 |
| route_planning | 0.062 | 0.125 | −0.062 | R6 |
| obj_appearance_order | 0.000 | 0.188 | −0.188 | R6（15/16 abstain） |
| **主表** | **18.36** | **41.17** | **−22.81** | C0 赢 50 题 / C1 赢 8 题 |

### R1 「相对距离」官方口径写错（**依据官方定义的纠错**）

v6 的 prompt 与工具文档都写着：

> **相对距离题（object_rel_distance）的官方口径就是「观察点→各候选对象」分别算再比较**

题面实际是：

> Measuring from the closest point of each object, which of these objects
> (telephone, computer mouse, keyboard, table) is **the closest to the trash can**?

官方口径是「**题面参照对象**（trash can）→ 各候选」，候选类别有多个实例时取最近实例
（v7 §2.2，依据 VSI-Bench 原论文附录 B.1）。v6 让模型算「相机→候选」——
参照物换了，排序几乎必然不同。实测该类题 C1=0.062（低于 4 选项随机水平 0.25）。

### R1b 「相对方向」把三个难度压成一个方向词表（**依据官方模板的纠错**）

inner 档 16 道方向题**全部是 medium**，题面为：

> ... is the cup to my left, right, or back? An object is to my back if I would
> have to turn at least 135 degrees in order to face it.

选项集合只有 `{left, right, back}` —— **没有 front**。而 v6 的 `direction_of` 在
夹角 < 45° 时返回 `"front"`，模型只能把一个**不在选项里**的方向词硬映射到三个候选之一。
v7 §9.3 要求按难度分档：easy 只给左右、medium 在 |θ|≥135° 才给 back、hard 才给四象限。

### R2 「绝对距离」被写成相机到对象

v6 工具文档写「绝对距离题（object_abs_distance）用它」指的是 `camera_object_distance`
（**相机→对象**）。官方口径是**题面点名的两个对象**之间的最近距离。
实测 16 道题里 6 道因键名/口径不符而作废。

### R3 计数：同一实例被重复检出却未合并

scene `7b6477cb95` 问 "How many monitor(s)"（GT=5），清单里有 **12** 条
monitor/tv-monitor 记录 → `count_objects` 返回 12（MRA=0）。
实测 12 条记录的质心两两距离 0.09–0.86（世界单位），点云**双向重合率**呈干净双峰：

- 同一实例：0.376 – 0.689（8 对）
- 不同实例：≤ 0.141（其余）

v6 的 3D 去重要求**时序重叠**才合并，而 SAM2 把同一显示器分裂成支撑帧互不相交的
多条 track → 时序判据永远不满足 → 重复记录全部留下。

### R4 米制 Tool 静默返回 `None` → 变成空答案

`object_distance_m`/`plane_fit_room_size`/`object_3d_extent` 在点云有效点不足时，
原语返回 `distance_metric=None`，工具**照常返回成功**，只是值是 `None`。
模型取键 → `ReturnAnswer(None)` → 提交值归一成空串 → 评分解析失败 = 0 分，
而且**不会**触发恢复或零工具兜底（管线上这次调用是"成功"的）。
实测 smoke：`object_abs_distance` 因此提交空答案。

### R5 检测器词表缺项 → 题面点名的物体根本不存在（**影响面最大**）

`OBJECT_VOCABULARY` 只有 40 条，而 VSI-Bench 题面实际点名 79 个不同类别。缺表后果：

- 16 道 rel_direction 里 **13 道**拿不到参照物（`power strip`/`ceiling light`/
  `coat rack`/`paper bag`/`exhaust fan` 都不在表内）；
- 计数题里 `ceiling light`(GT=4)、`heater`(GT=2)、`bucket`(GT=2) 全部返回 0；
- `obj_appearance_order` 15/16 直接 abstain。

更关键的是**逐题补漏也没兜住**：`_bind_question_supplement` 只走
`box_prompts_from_vlm`（让 VLM 输出 0–1000 JSON bbox），而
`open_vocab_detector.py` 的文件头**自己就记录过**该 8B 模型出框不可靠
（"纯文本列举能说出物体名，一旦要求输出 bbox 就返回 `[]`"）。
于是补漏的框始终为空 → 题面物体永远进不了清单。

### R6 模型只 `def solve` 不调用 → 整题零动作

prompt 推荐 `def solve(ctx): ... return ReturnAnswer(...)`，模型照做，
但执行器只是 `exec` 源码、**没有任何人调用 `solve`**：函数体从未运行，
零工具调用、无答案、episode 记 `unanswerable`。8 题 smoke 里 4 题（全部 MCA）栽在这里。

---

## 3. v7 实现清单

### 3.1 已实现并接线（implemented + connected）

| 项 | 位置 | 对应 v7 |
|---|---|---|
| 取消拒答：prompt/头部/恢复反馈全部改写；`abstain` 不再是合法答案 | `synthesis/prompt_builder.py`、`tools/registry.py:docs_header`、`online/runner.py` | D1 |
| `ReturnAnswer` 改为**终结操作**（抛 `BaseException` 子类，`except Exception` 吞不掉） | `sandbox/kernel.py` | §10.2 |
| 新增 `YieldObservations` 控制接口 | `sandbox/kernel.py` | §10.2 |
| 多轮执行：yield → 观察回灌 → 同一 agent 写下一段程序 | `online/runner.py`（M10 循环） | §11.1/D4 |
| 观察反馈携带**实际 payload**（工具名+参数+返回值+降级标记），并列出预算余量 | `online/runner.py:_yield_feedback` | §11.1 |
| 有界预算：`max_agent_rounds/max_tool_calls/reserve_final_rounds` + finalization | `online/runner.py`、`OnlineRunConfig` | §11.2 |
| host 负责调用 `def solve(ctx)` 入口 | `sandbox/kernel.py:_invoke_entry` | §10.2 |
| 顶层 `return` 自动包装（两种写法都支持） | `sandbox/ast_guard.py:normalize_program_source` | §10.2 |
| 工具返回值附带 `result_id`（否则模型无法点名要回灌哪条结果） | `sandbox/kernel.py` | §10.2 |
| 有图必答**单一收口点**：M10 之后任何无答案状态都走零工具作答 → 直答视觉兜底 | `online/runner.py`（M10b） | D1 |
| 有答案就不允许停在 `unanswerable`（格式问题单列 flag） | `online/runner.py:_finalize` | D1/§16.1 |
| 契约违规不再作废答案（工具层已隔离失败值），保留答案+留痕+来源降级 | `online/runner.py` | §10.1 |
| `relative_distance_rank`：参照对象→候选（取最近实例），官方口径 | `tools/geometry_tools.py` | §2.2/§9.2 |
| `object_distance_m`：两题面对象最近距离（米） | `tools/geometry_tools.py` | §9.2 |
| `direction_of(..., difficulty=)` 按 easy/medium/hard 输出**各自选项集合** | `reconstruction_gate/world_frame.py` | §9.3 |
| 左右平局显式确定性判据（不再由 `-0.0` 符号决定） | 同上 | §9.3 |
| 实例整合：双向点云重合并查集合并（阈值 0.30，实测分离度支撑） | `tools/distance_primitives.py:consolidate_instances` | §9.1 |
| 米制 Tool 缺失值**显式失败**而非静默 None | `tools/geometry_tools.py:_require_metric` | §10.1 |
| 检测器词表 40 → 83（覆盖题面 79 类） | `segmentation/sam2_tracker.py:OBJECT_VOCABULARY` | §9.1 |
| 同义词表补 30+ 条（power strip/computer tower/coat rack/exhaust fan…） | `tools/category_match.py:SYNONYMS` | §9.1 |
| 逐题补漏改用**确定性检测器**（GroundingDINO），VLM 出框降为次选 | `segmentation/sam2_tracker.py:ovd_boxes_for_nouns` | §9.1 |
| M5 成本控制：词表不收 `wall`/`floor`/`ceiling`（巨大 mask，传播慢一个数量级），基础清单按检测置信度截断到 40 条 | `segmentation/sam2_tracker.py`（`_cap_detections`） | §15.2 |
| 键名兼容别名（`distance_metric` ⇄ `surface_distance_metric`）+ 文档写明返回键 | `tools/geometry_tools.py` | §10.1 |

**关于 M5 成本（实测）**：扩表后单场景对象数 53–83（旧 38–78），而 M5 成本≈线性于
SAM2 传播条数（单条实测 2–25 s，`wall` 类大 mask 是 1.4 it/s vs 普通物体 13 it/s）：
未截断时一个场景 **25 min**，24 个场景不可接受。加"去 wall + 按置信度截断到 40"后
单场景降到 **3–10 min**。截断**不丢题面物体**：逐题补漏会按题面名词用确定性检测器
补检（那是按需付费），这也是把补漏从 VLM 出框改成检测器出框的另一个收益。

### 3.2 尚未实现 / 未接线（诚实列出）

| 项 | 状态 | 说明 |
|---|---|---|
| Docker 沙箱隔离 | **未接线** | 仍是进程内 AST + SIGALRM；`use_docker` 字段仍无人读取（§15.1 要求"正式真实实验必须有实际生效的隔离"—— **这是本报告范围内未完成的硬项**） |
| 3DGS 致密化 | 未实现 | `gsplat_trainer` 仍恒 `return None`；v7 §2.3 已把它移出必做范围 |
| G5 重投影残差 | 未实现 | 仍固定 `not_available`（v7 允许） |
| 方法型 Skill 学习闭环（晋升） | **未跑通** | 离线链在真实模式仍未产出候选；`active_snapshot_ref` 仍为 `genesis`（§14） |
| B0–B4 五臂对照 | 部分 | 本报告只有 C0（≈B1 的对照物）与 C1（近似 B4）；B2/B3 未跑 |
| inner 阈值标定（`quantile_q`、预算、检索 top-k） | 未做 | 仍用起始值 + `[TODO_CALIBRATE]` 标记（§8.3/§12.1） |

---

## 4. 真实实测

### 4.1 运行配置（两臂唯一差异）

同 32 帧 FrameSet、同 Qwen3-VL-8B-FP8（vLLM, port 8100）、同 VGGT + MoGe-2 产物目录、
同 seed。C0 不带工具与 `--moge2`；C1 带工具面与米制尺度。

**样本口径（重要）**：本次 v7 结果先跑 **inner_validation × 8 题/题型 = 64 题 × 1 seed**
（`data/v7_run/`）。这 64 题是 v6 那 128 题（16 题/题型）的**严格子集**
——抽样是"按 meta 行序先到先得"的确定性规则，与 seed 无关（见
`scripts/run_inner128_batch.sh` 头部说明），所以两边的题是**同一批**，
下面的对照因此成立，不是换样本换出来的。

64 题只是**开发档**（v7 §16.2 允许在 inner 反复开发）；论文主表仍需 ≥3 seed ×
每题型 ≥32 题（§16.2），本次未跑，报告不把它当主表。

**在飞的实验**：128 题档（`--sampling-per-task 16`，与 v6 那批 128 题完全同名同序）
已于 21:07 启动，脚本 `/tmp/launch_v7_full.sh`、输出 `data/v7_full/`。
它仍在为尚未缓存的场景重跑 M5（单场景 4–15 min，视物体大小），**本报告写作时未完成**。
落地后读法：

```bash
sed -n '/^per-task:/,$p' data/v7_full/logs/c1_seed0.log        # 分题型 + 主表
python3 scripts/report_v7_inner128.py --root data/v7_full --seeds 1 \
        --ref-root data/v6_inner128                            # 与 v6 对照
```

**数据卫生说明（如实记录）**：本次调试期间我删过两个自己创建的中间目录
（`data/v7_inner64`、`data/v7_inner128`）——后者含有 c0_seed0/1/2 与 c1_seed1/2
的**跨代码版本**运行（它们启动于我最后两处 M5 补丁之前，与 `data/v7_run/` 不是
同一版本，混进任何对照都违反"同版本同输入"前提）。v6 的六个 128 题基线
（`data/v6_inner128/*_seed{0,1,2}`，各 128 题）与全部重建产物**完好未动**。

### 4.2 结果（64 题，配对同题）

| task | **v7 C1** | v6 C1 | v6 C0 | v7 C1 − v6 C0 | v7 C1 − v6 C1 |
|---|---|---|---|---|---|
| object_counting | 0.6000 | 0.3875 | 0.7375 | −0.1375 | +0.2125 |
| object_abs_distance | 0.4250 | 0.2500 | 0.2958 | **+0.1292** | +0.1750 |
| object_size_estimation | 0.4125 | 0.4292 | 0.6333 | −0.2208 | −0.0167 |
| room_size_estimation | 0.5625 | 0.4625 | 0.6042 | −0.0417 | +0.1000 |
| object_rel_distance | 0.3750 | 0.0833 | 0.1250 | **+0.2500** | +0.2917 |
| object_rel_direction | 0.6250 | 0.3333 | 0.3750 | **+0.2500** | +0.2917 |
| route_planning | 0.2500 | 0.0833 | 0.1250 | **+0.1250** | +0.1667 |
| obj_appearance_order | 0.2500 | 0.0000 | 0.3750 | −0.1250 | +0.2500 |
| **主表（×100）** | **43.75** | **25.36** | **40.89** | **+2.86** | **+18.39** |
| coverage | **1.0000** | 0.7135 | 1.0000 | — | — |

逐题配对（64 题，v7_C1 vs v6_C0）：**v7 赢 18 / v6_C0 赢 18 / 平 28**。

### 4.3 怎么读这张表

**三件已确证的事：**

1. **v7 把工具臂从"比直答低 22 分"变成"略高于直答"**：v6 C1=25.36 → v7 C1=43.75，
   而直答基线是 40.89。**拒答率 32% → 0%**（coverage 0.7135 → 1.0000）。
2. **3D 工具在"2D 先验弱"的题型上明确胜出**（这正是应当用工具的地方）：
   相对方向 +0.25、相对距离 +0.25、路线规划 +0.125、绝对距离 +0.129。
   方向题 8 题里 5 题直接答对 —— 上一版是 1 题。
3. **在"2D 先验已很强"的题型上工具仍然略输**：计数 −0.1375、尺寸 −0.2208、
   房间面积 −0.0417、外观顺序 −0.125。这说明**不是**"工具不好用"，而是这些题
   VLM 自己看图就很准（计数 C0=0.7375），而当前测量链的噪声还没压到那个水平以下。

**不能过度宣称的事：**

- 主表 +2.86 在 64 题上**不足以声称工具系统整体优于直答**：逐题配对是 18 胜 18 负，
  同 scene 内题目不独立，n=64 的配对差标准差远大于 2.86 分。要做显著性检验必须
  按 scene 聚类 bootstrap 且扩到 ≥3 seed（v7 §16.4），本次**没有做**。
- 尺寸/计数/房间面积三项仍是负贡献，是下一步最该压的地方（见 §5）。
- `appearance_order` 从 0.00 提到 0.25，但仍然低于直答 —— 工具路径（按
  `visible_frames` 最早帧排序）在检出不全时反而不如模型自己看视频。

### 4.4 已完成的真实验证（与性能分开）

| 验证 | 结果 | 证据 |
|---|---|---|
| 单元/集成测试 | **744 passed, 1 skipped** | `python -m pytest tests/unit` |
| 有图必答（8 题 smoke） | coverage **1.0000**、refusal 0.0000 | `data/v7_smoke/`（修复前 0.5000） |
| 有图必答（64 题真跑） | coverage **1.0000** | `data/v7_run/c1_seed0/` |
| `def solve` 代调用 | 4 个 MCA episode 从 `unanswerable` 转为 `answer` | smoke run 的 episode_trace |
| 米制 Tool 缺失值显式失败 | `object_abs_distance` 从空答案 → 2.5（MRA 0.90） | smoke run |
| 检测器扩表生效 | 场景清单 54→65 对象、25→34 类别；新检出 `power strip`/`computer tower`/`door`(9/9 帧) | 新旧 `*_inventory_*.json` 对比 + 逐帧探测 |
| 方向模板分档生效 | 16 道 medium 题不再返回不在选项里的 `front` | `direction_of(difficulty=)` 单测 + 真跑 5/8 命中 |
| 实例整合 | monitor 12 记录 → 6 实例（GT=5）；16 道计数题净 MRA 0.325→0.356 | 真实 artifact 实测 |
| M5 成本 | 单场景 25 min → 3–4 min（去 wall + 面积过滤 + 置信度截断 + 补漏跳过） | 运行日志时间戳 |

---

## 5. 结论、剩余风险与下一步

### 5.1 结论

1. **根因是接口与口径，不是"3D 工具不 work"。** v6 工具臂低 22.8 分这件事，
   逐题溯源到六个各自独立的缺陷，其中两个是**把官方口径写错**
   （相对距离的参照系、相对方向三个难度被压成一个方向词表），
   一个是**检测器词表缺 39 类**导致题面点名的物体根本不存在。
   修完之后主表 25.36 → **43.75**，且**首次超过直答基线**。
2. **有图必答已达成**：coverage 0.7135 → **1.0000**（64 题真跑），
   拒答率 32% → 0。这是纯增量：v6 里那 32% 的题一律记 0 分。
3. **3D 工具的价值集中在"2D 先验弱"的题型上**，且幅度很大：
   相对方向 +0.25、相对距离 +0.25、路线规划 +0.125、绝对距离 +0.129。
   这四类正是"必须知道物体的真实三维关系才能答"的题，也是这篇论文最该讲的点。
4. **在 VLM 2D 先验已经很强的题型上，工具仍是负贡献**：计数 −0.14、尺寸 −0.22、
   房间面积 −0.04、外观顺序 −0.13。诚实的读法是"这四类题不需要我们的测量链"，
   而不是"测量链坏了"。

### 5.2 真实负结果（按 v7 §8.3，负结果照实记录，不改写成成功）

- **`object_3d_extent` 的尾部分位不该改**：实测 4 道尺寸题的点云 extent
  在不同分位下的 MRA（q=0.0/0.01/0.05/0.10/0.20 → 0.025/0.225/0.175/0.125/0.075），
  当前默认 `q=0.01` 已经是这批里最好的，**放大分位会更差**。
  同时发现该测量里模型的实际得分（0.65）明显高于"纯 extent"重建分（0.225），
  说明模型**已经在用视觉估计覆盖**坏测量 —— 因此尺寸题的 0.22 差距
  不能靠调分位解决，需要单独做实例选择（哪个 monitor 才是题面那个）的标定。
- **实例整合是净正但幅度小**：16 道计数题 MRA 0.325 → 0.356（+0.031），
  与"记录数→实例数"的期望一致，但远不足以补上计数题 0.14 的差距。
  阈值 0.30 是在同一批数据上定的，**存在过拟合风险**，扩到更多场景需重标。

### 5.3 未完成项（不能算作完成）

| 项 | 状态 |
|---|---|
| **Docker/容器隔离** | **未接线**。仍是进程内 AST + SIGALRM。v7 §15.1 把它定为"正式真实实验必须有实际生效的隔离"的硬项 —— **这一项没做**，因此当前结果按 v7 口径**不能算 `paper_eligible`** |
| 主表 ≥3 seed × 每题型 ≥32 题 | 未跑。本次只有 64 题 × 1 seed（开发档） |
| 显著性检验 | **未做**。18 胜 18 负 / n=64 不足以支撑任何显著性宣称；需按 scene 聚类 bootstrap（§16.4） |
| outer_holdout / final_test | 未跑 |
| 方法型 Skill 学习闭环 | 未跑通，`active_snapshot_ref` 仍是 `genesis`（§14） |
| B0–B4 五臂 | 只有近似 B4（C1）与近似 B1 的对照物（C0）；B2/B3 未跑 |
| inner 阈值标定（`quantile_q`、预算、检索 top-k、`MAX_BASE_DETECTIONS`） | 未做，仍是起始值 + `[TODO_CALIBRATE]` |

### 5.4 下一步（按预期收益排序）

1. **把"工具弱于直答"的三个题型纳入逐题型策略并冻结**：系统已有
   `direct_answer_tasks` 机制（当某题型程序路径弱于直答时改用直答，策略在 inner 定、
   outer 验证）。预期主表 +0.15 左右，且**必须在 inner 定、在 outer 验证**，
   不能用同一批 64 题既选策略又报成绩。
2. **尺寸题的实例选择标定**：题面对象常有多个实例（多个 monitor），
   选错实例是 0.22 差距的主因（实测一例：GT=13 cm 的对象被取到 271 cm 的点云）。
3. **计数题**：目前 0.60 vs 直答 0.7375。把"检测器实例数"与"模型看图数数"
   做一次**显式对照**（让 Agent 两条路都走一遍再选），而不是只信检测器。
4. **跑满主表**：128 题 × 3 seed + C0 对照 + 按 scene 聚类 bootstrap。
5. **接线 Docker 隔离**（做 paper_eligible 的前置硬项）。

---

## 6. 运行命令

```bash
cd /nas/wangjh/harness3d/skill3d_codebase
source /home/cvailab/anaconda3/etc/profile.d/conda.sh && conda activate skill3d-exp
export PYTHONPATH=third_party/vggt:src HF_HUB_OFFLINE=1
export SKILL3D_DETECTOR_ENDPOINT=http://127.0.0.1:20022   # GroundingDINO

# 单元测试
python -m pytest tests/unit -q

# 主实验（C1 ×3 seed → C0 ×3 seed）
bash scripts/run_v7_inner128.sh

# 配对比较
python scripts/paired_v5_v6.py data/v7_inner128/c1_seed0/traces \
                             data/v7_inner128/c0_seed0/traces --label-a v7_C1 --label-b v7_C0
```

## 7. 迁移与兼容

- v6 的 trace/产物**原样保留**，未改标签；v7 结果写入新目录 `data/v7_inner128/`。
- 检测器词表进入 M5 缓存键（`_inventory_key` 含 `vocab`），因此扩表后**自动失效**
  v6 的清单缓存，不会出现"新代码配旧清单"的静默混用。
- 语义变更必须注意：`ReturnAnswer` 从"只记录"改为"立即终结"；
  `unanswerable` 不再是正常终态；`abstain` 不再是合法答案。
  这些是 v7 的**有意替换**，不是回归。
