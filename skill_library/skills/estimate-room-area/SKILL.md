---
name: estimate-room-area
description: 用于 room_size_estimation：识别题面房间或合并空间范围，结合有效平面边界或视觉布局估计面积，输出平方米。
metadata:
  harness3d-skill-id: S05
  harness3d-version: 1.0.0
  harness3d-question-type: room_size_estimation
  harness3d-family: metric
  harness3d-source-format: '1.0'
  harness3d-validation: document-checked-runtime-pending
---

# 房间边界与面积估计

## 目标

估计题目指定空间的平面面积，规范输出 unit=m2 的有限正数。

## 适用条件

可使用授权的平面/边界工具，也可从原图布局估计；缺尺度或平面不可靠时不整体屏蔽本方法。

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
  - 房间/多房间范围与连通关系
  - 地面或水平边界的覆盖
  - 尺度与平面拟合质量
  local_requirements:
    plane_fit_room_size:
    - geometry_3d
    - metric_scale
    - 当前米制调用获准
  visual_route: 用墙面连接、门洞、房间转角和尺寸参照建立简化平面布局。
```

## 求解步骤

1. 从题面确定单房间还是多房间合并范围，依据帧中的门洞、墙面和相机行进关系区分不同空间；同一空间的重复拍摄只算一次。

2. 读取前置重建及覆盖摘要，检查是否有有效水平面和边界。家具遮挡的地面不等于房间缺失面积；地板像素数、墙面面积、房间对角线都不直接等于所问面积。

3. 若 plane_fit_room_size 获准，核对返回 area、单位、边界/平面质量及覆盖范围是否与题意一致。可计数对象清单没有 floor 或 wall，不妨碍利用已有结构证据；不能虚构未注册的地面分割调用。

4. 处理非矩形和多个区域时，依据可靠边界做分解与并集，避免重叠重复计入；不要将所有可见区域直接取外接矩形/凸包而填满凹口和空间间隔。采用的近似应有视觉依据。

5. 理解单位与尺度：归一化面积到平方米需乘 s²，已是平方米不再缩放。尺度或边界失效时撤销依赖测量的使用，由框架管理恢复。

6. 几何不足时，根据多个原图视角构建简化平面草图，利用墙段、门洞和家具等参照估计主尺寸；按有依据的矩形或少量区域分解求面积。参照物常识与不可见边界均属于估计，不能作为已验证几何。

7. 检查空间范围、重复区域和未观察边界是否主导误差，选取最有依据的单一面积并按平方米提交。没有可靠边界也保留视觉作答，不能把工具失败返回为零面积。

## 提交前检查

- 计算的是所问空间面积，单位为平方米。
- 长度尺度用于面积时平方，已经是米制面积时不重复处理。
- 重复拍摄、重叠区域和家具遮挡未引入重复计数或错误扣除。
- 没有把有限可见地面直接称为完整房间边界。

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

- 视野未覆盖边界时，面积只能依据部分证据估计。
- 官方面积构造与本系统拟合代理可能不同，需通过开发集检验偏差。

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
  - '9.3'
- ref: VSI
  sections:
  - B.1
  use: 面积及多房间范围定义
- ref: SKILL3D_PROJECT
  sections:
  - Qualitative Cases / Room Size
  use: 房间尺度的多证据组织
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
