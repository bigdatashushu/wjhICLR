---
name: estimate-object-distance
description: 用于 object_abs_distance：按题面两个对象的最近距离口径，选择有效米制测量或视觉估计，输出米。仅在本题型分区召回。
metadata:
  harness3d-skill-id: S02
  harness3d-version: 1.0.0
  harness3d-question-type: object_abs_distance
  harness3d-family: metric
  harness3d-source-format: '1.0'
  harness3d-validation: document-checked-runtime-pending
---

# 对象间绝对距离估计

## 目标

估计两个题面对象之间的最近直接距离，输出 unit=m 的有限非负数值。

## 适用条件

优先使用满足逐工具权限的几何测量；无尺度、绑定待确认或几何失败时保留视觉估计路径。

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
  - 两个题面对象的有效实例绑定
  - 一致坐标中的有效对象几何与距离定义
  - 当前有效尺度及米制授权
  local_requirements:
    object_distance_m:
    - geometry_3d
    - metric_scale
    - 两个实例有效绑定
    - 当前米制调用获准
  visual_route: 用同时可见的边界、跨视角布局及合理参照尺寸估计对象间隔。
```

## 求解步骤

1. 解析两个对象的角色及消歧描述，明确测量对象间的最近直接距离；不能把参考对象换成相机，也不能默认使用中心距离。

2. 核对候选和题级绑定。需要时对相关类别补检、检查局部图像，结合周边物体确认实例；具体工具不可用时依靠原图定位，不虚构 obj_id 或绑定成功状态。

3. 若 object_distance_m 对实际两个对象获准，调用后检查 status、有效数值、单位、依赖和距离定义。pointcloud_surface_proxy 与 bbox_nearest_proxy 均为声明过的代理，不能把可见表面代理当成官方标注真值。

4. 理解工具已完成的单位转换。只有文档明确返回归一化距离并允许使用同一有效尺度时才做 d_m=s*d_norm；已经是米的值不能再乘尺度。量化阈值、点集清洗和分位数由冻结工具合同决定。

5. 检查最近区域是否被遮挡、点集是否混入背景、两对象是否来自一致坐标。结果失效就停止使用并交给框架处理；不要把旧数值手动改成有效，或用视觉猜测恢复米制权限。

6. 测量不能支持结论时，比较多个原图视角中的相邻关系与接近边界；结合合理参照物的尺寸范围估计间隔，修正明显的透视和深度差异。参照尺寸属于估计；不要把不同深度上的像素比例直接当米制比例。

7. 结合当前可靠证据选取最有依据的数值；跨帧一致性支持估计，但同一误差来源的多次结果不等于独立验证。按米提交，并正确区分测量、视觉估计和混合依据。

## 提交前检查

- 测量的是两个题面对象，不是相机到对象。
- 距离口径与单位正确，没有中心距离替代最近距离或重复乘尺度。
- 米制测量的尺度、几何和绑定都仍有效。
- 视觉参照没有被记录为工具测量或真实尺度。

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

- 可见表面距离与完整对象包围框距离可能有系统偏差。
- 完全遮挡的最近边界只能估计，不能宣称精确恢复。

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
  - '8'
  - '9.2'
  - '9.3'
  - '10'
- ref: VSI
  sections:
  - B.1
  use: 对象间最近距离及米单位
- ref: SKILL3D
  sections:
  - Appendix D
  use: 边界相关的距离证据
- ref: VADAR
  sections:
  - Approach
  use: 将定位和计算分解为可组合步骤
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
