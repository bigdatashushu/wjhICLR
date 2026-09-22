# 决策记录：route_planning 的根因是属性引用接地与场景门，不是 connectivity_graph（2026-09-22）

- 日期：2026-09-22
- 决策人：Coding Agent（依据真实 trace 的逐题归因）
- 触发：交接文档把 route_planning 的失败写成
  "§9.10 `connectivity_graph` 在部分场景抛错（点图/竖直轴），根因未查"
- 影响条文：§9.10、§16.6、§13.3（abstain 作为合法终态）
- 不改动：任何 fail-closed 门槛、任何阈值

## 结论（先说结果）

4 道 route_planning 题**没有一道调用过 `connectivity_graph`**，
"`connectivity_graph` 抛错"这条根因假设**不成立**。真实根因是两个彼此独立的机制：

1. **属性引用接地缺失**（3/4 题）：题面用颜色/属性限定实例
   （"the **blue** chair"、"the **black desk** chair"、"the **red** desk chair"），
   而 `list_objects` 只返回 `category_name`（全是 `chair`）→ 模型无法确定"哪一个椅子"，
   主动 `ReturnAnswer("abstain")`；
2. **场景未过 M4 主门 → 工具面收窄**（1/4 题）：scene `6115eddb86` 的
   `overall_quality=0.000` → `question_tool_scope=fallback_2d_only` →
   米制/3D 工具全部收回，题面工具清单里**只剩 `euclidean_distance`** →
   模型弃答。这是 **fail-closed 的正确行为**，不是缺陷。

## 证据

### E1　`connectivity_graph` 从未被调用

统计**全部真实运行**的 `program_trace.jsonl`（`data/v6_scoped/*/traces/` +
`data/v6_inner128/*/traces/`，**177 个真实程序**、累计 373 次工具调用）：

| Tool | 调用次数 |
|---|---|
| `list_objects` | 226 |
| `robust_distance` | 46 |
| `object_centroid` | 36 |
| `count_objects` | 18 |
| `relative_direction_of` | 18 |
| `surface_distance_between_objects` | 9 |
| `object_3d_extent` | 8 |
| `plane_fit_room_size` | 6 |
| `euclidean_distance` | 4 |
| `object_visible_frames` | 2 |
| **`connectivity_graph`** | **0** |

4 道 route 题（4963/4978/4992/4995）的 `calls` 全是
`list_objects` + `object_centroid` + `relative_direction_of`（4995 只有 `ReturnAnswer`）——
**`connectivity_graph` 出现 0 次**。它既没被调用，也就不可能抛错。

### E2　3 题的 abstain 是"找不到被属性限定的实例"

- 题面：`You are a robot beginning at the blue chair and facing the column ...`
  实际 `category_name` 只有 `chair`（5 个实例），没有颜色字段 →
  程序的 `if 'blue' in obj['category_name'].lower()` 恒假 → `blue_chair_id=None`
  → 走 `ReturnAnswer("abstain")`。
- 同一 scene 的 4978（"black desk chair"/"red desk chair"）、4992（"black sofa"）同型。

### E3　abstain 之后继续调 Tool 被正确兜住（但不是分数问题）

3 题的 `ProgramExecutionTrace` 末尾有一条
`error_code="answer_already_given"`：模型写了
`if blue_chair_id is None: ReturnAnswer("abstain")` 之后**继续**调用
`object_centroid(blue_chair_id)`，运行层按 §15.1 抛出 `AnswerAlreadyGiven`，
episode 被归入 `tool_contract` 桶、`answer_untrusted=True`。

- 行为是**正确的**（没有崩成假服务故障，也没有据此给出答案）；
- 但它把"模型本来就想 abstain"记成了"契约违规"，**归因口径失真**——
  两者在主榜都按错计，分数不受影响，故**不改**，仅在此登记，
  后续若要做"失败类型分布"分析需先区分这两者。

### E4　4995 的 abstain 来自场景门，工具面只剩 1 个 Tool

该题的 M8 prompt 头写的是
`question_tool_scope=fallback_2d_only；可用重建产物=frames, intrinsics`，
可用 Tool 列表里**只有** `euclidean_distance`。模型据此判断"无法确定相对方向"
并 abstain —— 与 §13.3"拿不到就不要编"一致。

## 决定

1. **不修 `connectivity_graph`**：没有证据表明它被调用过或抛错过；
   按"不为未复现的假设动代码"处理。
2. **登记两条真实缺口**（进实施报告，标 `[待实验]`/未解决，不得写成已修）：
   - **属性接地缺口**：`list_objects` 不提供颜色/材质等限定属性，
     而 VSI-Bench 的 route_planning 大量用属性限定实例。
     要修就得让 M5 产出属性标签（例如用已有 VLM 客户端对每个实例的掩码裁剪做属性判读），
     这是一条**新能力**，需走 §17.4 的工具/能力准入，**不得**在题面字符串上做特判。
   - **场景门过严的可能性**：24 个 inner scene 里 3 个 `overall_quality=0.000`
     → 这些 scene 的**所有** 3D 题型（含非米制）都被收窄。
     τ_warp/τ_cloud 仍是 `[TODO_CALIBRATE]`，0.000 是"真坏"还是"阈值过严"
     **只能由 §10.6 的标定 PoC 回答**，不得靠调阈值。
3. `connectivity_graph` 的**存在本身**保留（§9.10 的规格要求），
   但在 route_planning 上它目前是**未被使用的**——这一点必须如实写进报告，
   不得表述成"route 由连通图保证"。

## 诚实边界

- 本记录基于 **4 道题**（n=4/题型）的逐题归因，**不构成显著性**，
  只用于"根因定位"；比例性陈述(3/4, 1/4)不得进论文主表。
- "属性接地缺失"是从这 4 题的题面结构与程序行为推出的机制，
  尚未在更大样本上量化（需要 128 题的 route_planning 子集 trace 才能给比例）。
