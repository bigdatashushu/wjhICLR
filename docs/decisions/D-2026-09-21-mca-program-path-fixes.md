# 决策记录：MCA 程序路径的三处真实缺陷修复（2026-09-21）

- 日期：2026-09-21
- 决策人：Coding Agent（依据《系统架构5.md》v5.0 已确认设计与实测证据）
- 触发：outer_holdout 32 题真实跑批（`data/v5_scoped/full/c1/`）的失败归因
- 影响条文：§3"尺度能力门控"、§4 M5/M8、HC21/HC23/HC33
- 不改动：任何 fail-closed 门槛、任何统计口径、任何阈值

## 背景：C1（程序路径）在 outer_holdout 上为何远低于 C0 直答

32 题逐题归因（`episode_trace.jsonl` / `program_trace.jsonl` / 复跑原始输出）后，
失分不是"模型不会推理"，而是三类**工程缺陷**：

| 现象 | 题数 | 根因 |
|---|---|---|
| episode 记 `unavailable`（`n_images_to_synthesizer=0`） | **9/32** | M8 输出退化/截断 → 代码块**未闭合** → 旧解析器整段放弃 |
| prompt 里只剩 1 个 Tool、模型据此 abstain | **12/32** | 未授权米制题型被改成 `route=fallback_2d_only` → 产物集塌成 `{frames,intrinsics}` |
| 对象清单残缺（同 scene 只剩 2 个对象） | 影响全部对象类题 | 检测器服务故障**静默**返回 `[]`；清单每个 episode 重算且与问题耦合 |

以下三项修复都已落码并带回归测试；**没有一处放宽门槛**。

## 决定 1：M8 program 提取增加"未闭合代码块"回退（§4 M8）

**决定**：`extract_program_source` 的回退顺序改为
① 闭合 ```python 块 → ② 闭合 ``` 块 → ③ **未闭合围栏之后的内容，从尾部逐行丢弃
直到 `ast.parse` 通过** → ④ 整段文本。

**理由**：实测（`/tmp/m8_raw/`，本次留档）9 个失败 episode 里 6 个的输出是"模型在
'必须给数值'与'不能猜'之间反复自辩"的重复段落，被 `max_tokens` 截断 → 没有闭合围栏。
旧实现对这种情况直接抛 `SynthesisError` → episode 记 `unavailable`，等于**白丢 28% 的题**。

**边界（不得被误读为放宽）**：
- 只回退"能拿到一段**模型自己写出的**可执行代码"；拿不到可解析代码**仍然抛错**；
- 不补全、不臆造任何语句（残行被丢弃，不会生成 `ReturnAnswer`）；
- 解析出的 program 照旧过 M9 AST 白名单 + M10 沙箱 + M11 几何校验；答案真实性不受影响。

## 决定 2：未授权米制题型**只收回米制 Tool，不得改变 route**（§3 已确认设计）

**决定**：`question_gate` 对 `object_abs_distance` / `object_size_estimation` /
`room_size_estimation` 未授权时，**保持** `scene.route` 不变，只把
`allowed_metric_tasks` 记为空集并打 `v5_metric_task_not_authorized` 标记。
米制 Tool 的收回由三条既有通道完成：`route_artifacts_for_question` 去掉 `scale`、
`REGISTRY.docs()` 裁剪、执行期 `check_metric_task_contract` 抛 `ConfidenceGateError`。

**依据（规格原文，非本次新口径）**：
- §3："`scale_confidence=low` 时**只移除 `scale` artifact 与米制工具，不得改变原本合格的
  `full_3d` route**"；
- HC33："`low` 只收回依赖米制尺度的 Tool/Skill，**不得**把可用的非米制
  `depth/poses/point_cloud/objects` 一并降级"；
- §11 保守倾向："存疑时判 `full_3d`（误删工具的代价是整题不可答）"。

**实测影响**：旧实现让 12/32 题落 `fallback_2d_only`，其产物集 `{frames,intrinsics}`
连带砍掉 `objects/poses/point_cloud/depth` → prompt 的 Tool 清单**只剩
`euclidean_distance` 一个**（且该 Tool 无 3D 点可算）→ 模型只能 abstain；
同时 prompt 头部 `route=fallback_2d_only` 与场景摘要 `route=full_3d` **自相矛盾**。
修复后这 12 题的 Tool 清单回到 7 个非米制 Tool，米制 Tool 仍然不可调用（三重 fail-closed 不变）。

**不变**：这三类题型在无冻结校准器时**仍然拿不到米制答案**（`allowed_metric_tasks=∅`），
分数不会因此提高——这是 §10.2 标定数据的 blocker，**不得**用本修复冒充"尺度可用"。

**测试同步**：`tests/unit/test_question_gate.py` 的断言从"route 降级"改为
"route 不变 + 米制 Tool 收回 + 非米制 3D 产物保留"（语义更强，非放宽）。

## 决定 3：M5 对象清单改为 scene 级产物（缓存 + 逐题补漏 + 检测器降级出声）（§4 M5）

**决定**：
1. **清单构造与问题解耦**：检测器改用**完整室内词表**（旧实现只补 `OBJECT_VOCABULARY[:12]`），
   并新增一次 VLM **通用清单**提示（"列出最主要物体"，与题目无关）；
2. **按 `(scene, frame_set_hash)` 缓存复用**（`<scene>_inventory_<key>.json` +
   G7 动态掩码 npy）：同一 scene 的后续 episode 不再重跑 SAM2/检测器，只对
   **题目点名却没进清单**的对象做一次补漏绑定（`_bind_question_supplement`）；
3. **检测器故障必须出声**：`ovd.detect` 记录 `last_error`，整轮零检出即重试一次，
   仍为零则按"服务故障降级"或"本轮确实没检出"分别记 note（旧实现静默返回 `[]`，
   调用方只在"新增了框"时记 note → 故障不可见）。

**理由**：对象是 scene 级产物（§4 M5）。旧实现①每 episode 重算一次（~1.5 min/题，
是扩大样本的瓶颈）、②同一 scene 不同 episode 看到**不同**的对象清单（id 不稳定）、
③检测器一挂清单就塌缩（实测同一 scene 一题只剩 2 个对象，`list_objects("whiteboard")`
为空 → 方向题不可能答对）。

**不改动**：3D 去重三判据（类别 + 世界质心距离 + 时序重叠，`[TODO_CALIBRATE]` 0.15 起点）
与 §4 M5 的"单帧没检测到不等于不存在"纪律。缓存键含 `frame_set_hash` → 换帧集必然重算。

**测试**：`tests/unit/test_m5_inventory_cache.py`（10 项：缓存往返 / 键敏感性 / 损坏即未命中 /
故障出声 / 健康无检出不得误判 / 重试恢复 / 端到端复用与换帧集不混用）。

## 未修复但已定位（写入实施报告 blocker）

- **米制 Tool 未乘 `metric_scale`**：`object_distance_meters` / `object_size_longest_dim` /
  `room_size_m2` 返回的是**世界单位**，而 HC29 定义 `metric_scale` 为"米/世界单位"
  （`scale_ci_abs_m = metric_scale × scale_ci_rel`）。当前 `allowed_metric_tasks` 恒空
  → 三个 Tool 不可调用 → **影响为零**；但标定数据到位后若不修，三个米制题型的答案会
  系统性偏差 `metric_scale`（本机实测 ~5.23，面积类 ~27 倍）。见实施报告 §blocker。
- `vggt_sparse_ba` L1 三角化 0 点、尺度 L2/L3 缺 GT、多 seed 主表：仍按原 blocker 挂着。

---

## 追加（2026-09-21 晚）：v_d 回归的如实记录与"不回退"的决定

### 事实

修复后在两个 split 上各跑了多版（同一协议：32 题、seed 0、单 endpoint）：

| split | 版本 | Avg | 计数 MRA | rel_dir Acc |
|---|---|---|---|---|
| outer | v_a / v_b（M8 解析 + 逐题 route + M5 场景清单） | 8.44 | 0.675 | 0 |
| outer | v_c（+ 世界竖直轴与手性） | **20.94** | 0.675 | **100** |
| outer | v_d（+ `ReturnAnswer` 纪律 + 定向补漏，**当前代码**） | 13.75 | 0.300 | 75 |
| inner | v_a | 22.81 | 0.775 | 0 |
| inner | v_d | 11.56 | 0.250 | 0 |

**v_d 在两个 split 上同时低于先前版本，计数题是两个 split 共同的落点。**

### 决定：保留 v_d 的两项改动，不回退

理由（逐项）：
1. **`ReturnAnswer` 不中止**是 `_AnswerSlot` 的**事实语义**（§4 M10 反作弊设计的副作用：先写答案再触发契约失败 → 答案不得采纳）。prompt 把这条事实写清楚，纠正的是"模型误以为它中止"这一错误预期；v_a/v_c 的更高分数来自"程序崩溃后落到 no-tool CoT 兜底"，**不是来自工具路径本身**。
2. **题面点名物体的定向补漏**提高的是清单召回（旧实现里 `whiteboard`/`bookshelf`/`trash can` 根本不进清单，相关题只能 abstain）。召回不足是缺陷，不是特性。
3. 若为分数回退这两项，等于**用"让程序更容易失败"来换分**——这与 §0.2 HC33/HC34 的诚实性纪律直接冲突，也会让论文的核心叙事（程序路径的贡献）永远无法成立。

### 因此，下一轮的第一优先级是修 **M5 清单的精度与召回**，而不是回退 prompt

- **精度**：外测 v_d 计数出现 `12`（内测）等**多算**，说明同一实例被绑成多份——3D 去重阈值（`0.15 × 深度中位数`）是 `[TODO_CALIBRATE]`，且在"检测器框 ∪ VLM 通用清单框 ∪ 逐题补漏框"三路并集下更易产生副本；
- **召回**：外测 v_d 计数出现 `0`，说明该类实例一个都没绑到；
- **纪律**：任何阈值改动都必须走 `inner_validation` 定、`outer_holdout` 验，且先测噪声底（同配置重复 ≥3 次）——"小于噪声底的增益不算增益"。
- 这两项都**不能**在没有标定数据的情况下靠放宽门槛解决（§10.2）。

### 需要 v6 回答（已写入 `问题报告v5.md`）

对象清单的召回义务与故障语义、去重阈值在无标定数据时的默认方向、以及主表激励是否应改为"允许多路候选 + 记录来源"以避免"越诚实越低分"。
