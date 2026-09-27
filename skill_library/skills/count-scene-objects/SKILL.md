---
name: count-scene-objects
description: 用于 object_counting：在冻结视频帧中定位目标类别，关联重复观测，检查遗漏并提交非负整数计数。仅在本题型分区召回。
metadata:
  harness3d-skill-id: S01
  harness3d-version: 1.0.0
  harness3d-question-type: object_counting
  harness3d-family: counting
  harness3d-source-format: '1.0'
  harness3d-validation: document-checked-runtime-pending
---

# 跨帧对象去重计数

## 目标

估计题面指定空间中目标类别的独立实体数量，输出 unit=count 的非负整数。

## 适用条件

可使用纯视觉、二维检测/跟踪或获准的几何辅助。目标尚未绑定、检测失败或三维不可用时仍适用；这些状态只改变后续步骤。

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
  - 多视角目标候选及覆盖说明
  - 跨帧身份与重复嫌疑
  - 可用时的几何位置一致性
  local_requirements:
    count_objects:
    - object_detection
    - track_consensus
    geometry_assistance:
    - geometry_3d
    - 相关对象有效绑定
  visual_route: 利用原图中的外观、相邻物体、固定结构和观察顺序核对实体身份。
```

## 求解步骤

1. 从题面确定目标类别、空间范围及同义词；按当前题面语义处理部件与整体，不能把同一件家具的部件重复计入。

2. 复用有效候选与覆盖记录；可用时通过 list_objects 查看 category_name、obj_id、visible_frames、track_id 和重复嫌疑。将每条记录视为候选，不把记录条数当作答案。

3. 建立题内实例表，记录各候选出现的帧、相邻结构和相似对象。用多个线索合并同一实体的重复观测；两个同款物体在同一帧同时出现时保留为不同实体，不凭外观相似就合并。

4. 优先检查会改变计数的未覆盖区域、遮挡后重现和重复 track。若 detect_objects 已获准，仅对目标类别和有用帧补检；需要精细边界或身份关联时，仅使用实际注册的分割/跟踪能力。需看新裁剪时调用 inspect_frames 后 yield。

5. count_objects 获准时读取它的去重结果、覆盖说明和降级标记，与实例表交叉核对。三维位置只作为获准的辅助证据；二维方法可独立继续。服务失败、未运行、健康空检出、截断分别处理。

6. 工具与视觉冲突时定位冲突到具体实例，补看最能消除歧义的冻结帧；无法解决时根据覆盖及身份线索选最有依据的整数。计数为零须由观察支持，不能由服务错误推得；随后按共同合同提交。

## 提交前检查

- 同一实体跨帧只计一次；同款多实例没有被误合并。
- 截断、遮挡和未检查区域没有被当作完整覆盖。
- 没有用检测故障或空 result payload 推导零计数。
- 答案是非负整数；依据与实际使用结果一致。

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

- 有限帧集可能漏掉完全不可见对象。
- 实例表是当前题的推断，不能把模型推断写成已验证的几何绑定。

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
  - '9.1'
  - '9.2'
  - '10'
  - '13'
- ref: VSI
  sections:
  - '3'
  - B.1
  use: 计数任务定义
- ref: SKILL3D
  sections:
  - '3.1'
  - '3.2'
  - Appendix D
  use: 多视角去重与证据组织
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
