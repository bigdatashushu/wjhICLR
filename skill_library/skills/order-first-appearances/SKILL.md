---
name: order-first-appearances
description: 用于 obj_appearance_order：按冻结帧真实时间顺序核对目标类别首次可见证据，区分漏检与未出现并选择顺序选项。
metadata:
  harness3d-skill-id: S07
  harness3d-version: 1.0.0
  harness3d-question-type: obj_appearance_order
  harness3d-family: appearance
  harness3d-source-format: '1.0'
  harness3d-validation: document-checked-runtime-pending
---

# 目标类别首次出现顺序判断

## 目标

判断题面各目标类别首次出现的先后顺序，输出当前题目的合法选项，unit=option。

## 适用条件

时间信息可靠时可调用相应工具；若时间工具不可用，仍可依据保留顺序的原始帧进行视觉判断。

## 执行约束

仅在元数据指定的规范题型分区内检索本方法。读取框架提供的题面、原始冻结帧、当前证据摘要和实际可用工具文档；前置重建已由框架处理，失败也继续求解。

只调用当前注册并获准的工具，按实际参数与返回 Schema 编程。下文工具名来自 v8 目标合同，不代表当前实现已接通。检测不自动证明绑定成功，图像路径不代表已观察。需由模型判断新裁剪或结果歧义时，使用 YieldObservations 交回同一在线 agent；只从冻结帧集选择图片。

框架负责更新题级证据、撤销依赖结果和决定恢复权限；不要直接改状态、阈值或缓存。按剩余求解轮数决定是否继续；同一失败操作最多重试 3 次。finalization 只用已观察的有效证据和原图提交，不再调用探索工具或 yield。

最终通过统一程序入口和 ReturnAnswer 提交 AnswerPayload。tool_derived 仅用于答案可由有效工具结果及明确计算得到的情况；纯图片判断为 visual_estimate；融合工具证据与视觉推断为 mixed。填写实际使用的有效 result_id；有图时证据不足不构成拒答理由。

## 证据条件

```yaml
hard_requirements:
  image_2d:
  - available
  - degraded
evidence_preferences:
  preferred:
  - 冻结帧的物理身份、顺序和有效时间戳
  - 目标类别在各帧的可见证据
  - 已检查区间与漏检风险
  local_requirements:
    object_visible_frames:
    - temporal
    - 对象有效绑定
  not_required:
  - geometry_3d
  - metric_scale
  - world_frame
  visual_route: 沿原始帧顺序查找类别首次可辨认的出现，并核对相近时段。
```

## 求解步骤

1. 列出待排序类别和题目选项，按源帧时间/顺序组织原始图片；重复占位图和缺失帧不能当成新的真实时刻。

2. 先查看已提供原图和有效检测/可见帧记录。object_visible_frames 获准时获取实例的递增可见帧；题目问类别时，合并该类别所有已知实例的最早可见证据。

3. 对每个类别记录已知最早出现和此前已检查范围。工具首次检出只是当前观测上界；未检测、服务失败和检出为空不能证明此前不存在。

4. 优先检查会改变选项顺序的早期帧和相邻区间，使用已注册的检测或 inspect_frames；新裁剪需 yield 后才由模型判断。不得重新采样冻结集合之外的视频帧或恢复不存在的时间戳。

5. 用物体局部、遮挡变化及后续清晰视角核对早期疑似出现；不要把后续明确识别时间自动当成首次出现。视觉判断与检测冲突时检查具体帧，不用工具状态取代实际可见性。

6. 比较候选顺序与各类别的已知先后约束；同一采样间隔内无法确定先后时，不编造亚帧时间，根据现有可见线索和题目选项选最有依据的完整顺序，再提交合法标签。

## 提交前检查

- 使用源帧顺序，未把布局显示顺序或文件名排序当作时间。
- 类别的多个实例已按首次出现语义聚合。
- 工具首次检出没有被当成真实首次出现的无误差证明。
- 输出与当前 options 对应，不捏造未采样帧。

## 代码示例

```json
[]
```

本版使用方法说明和计算关系，不提供依赖未知实现的可执行示例。程序入口、工具调用和 AnswerPayload 构造以运行框架实际注入的合同为准。

## 失败教训

```json
[]
```

尚无本系统真实轨迹归纳；上面的检查属于初始方法设计，不冒充实测失败经验。

## 局限

- 冻结采样可能无法分辨在两个相邻帧之间发生的先后变化。
- 首次可辨认时间与标注可见性标准可能存在偏差。

## 答案依据

```json
["visual_estimate", "tool_derived", "mixed"]
```

## 来源

```yaml
origin: document_grounded_seed
source_refs:
- ref: LOCAL_V8
  sections:
  - '4'
  - '5'
  - '9.1'
  - '9.2'
  - '9.4'
- ref: VSI
  sections:
  - B.1
  use: 首次出现语义
- ref: SKILL3D_PROJECT
  sections:
  - Qualitative Cases / Appearance Order
  use: 按时间组织检测与对象证据
- ref: DECISIONS_20260925
  use: 八题型硬隔离、简化轮数预算和状态/观察合同
source_trace_refs: []
source_split: not_applicable_no_trajectory_induction
label_access: false
parent_skill_versions: []
candidate_record_ref: null
inducer_config_ref: generation_record.json
validation_decision: document_checked_runtime_pending
```
