---
name: rank-object-distances
description: 用于 object_rel_distance：比较题面参照对象到各候选类别的最近实例距离，选择当前题目的合法选项；不要求米制尺度。
metadata:
  harness3d-skill-id: S03
  harness3d-version: 1.0.0
  harness3d-question-type: object_rel_distance
  harness3d-family: relative_geometry
  harness3d-source-format: '1.0'
  harness3d-validation: document-checked-runtime-pending
---

# 对象间距离比较与候选排序

## 目标

识别与参照对象最近的候选类别/实例，输出当前题目给定的选项标签，unit=option。

## 适用条件

统一有效相对几何足以支持排序，metric_scale 缺失不屏蔽本 Skill；无法可靠定位或计算时使用视觉布局比较。

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
  - 参照对象和各候选实例绑定
  - 同一有效坐标系内的对象间距离
  - 候选类别覆盖与多实例信息
  local_requirements:
    relative_distance_rank:
    - geometry_3d
    - 参照及候选实例有效绑定
  not_required:
  - metric_scale
  visual_route: 从原图建立局部布局，比较参照物与候选的边界接近程度并考虑深度。
```

## 求解步骤

1. 分别整理题面参照对象、候选类别及选项映射。使用当前题目的选项标签，不固定为 A–D，不按对象离相机的远近作答。

2. 定位参照实例，并尽可能覆盖每个候选类别的实例。题意按类别比较时，对同类别多实例取与参照对象最近者；缺失类别标为未知，不能直接赋无限距离或从候选中删除。

3. 若 relative_distance_rank 对本次实例获准，调用并核对对象间距离定义、候选覆盖和类别聚合规则。若只获得多个归一化距离，只有坐标、尺度因子和距离定义一致时才能直接比较。

4. 对最接近的候选重点核查遮挡、点集污染、重复或错误绑定；检查表面距离代理与题面最近点语义是否存在明显偏差。不得因为 metric_scale 缺失而放弃有效相对几何。

5. 几何不足时，用同帧共现或跨帧共享结构形成局部布局，判断哪些候选真正邻近参照对象。图像上的二维间隔仅为线索；同时考虑深度、视角和遮挡。优先查看会改变前两名排序的现有帧或裁剪。

6. 比较所有合法候选，综合未覆盖候选和残余歧义选最有依据的一项；不要以选项顺序作为通用平局规则。转换成题目原始标签，按共同合同提交。

## 提交前检查

- 参照对象没有变成相机。
- 同类别多实例按最近者聚合，缺失候选没有被静默排除。
- 排序使用一致的几何定义和坐标，未要求无关的米制尺度。
- 答案标签属于本题 options，答案依据包含实际使用的工具结果。

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

- 候选遗漏可能改变排序；检测到一个实例不证明该类别已覆盖。
- 相对几何存在非均匀畸变时，尺度无关并不意味着排序可靠。

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
  - '6.2'
  use: 参照对象、最近实例聚合与布局比较
- ref: VADAR
  sections:
  - Approach
  use: 组合定位、属性查询和比较步骤
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
