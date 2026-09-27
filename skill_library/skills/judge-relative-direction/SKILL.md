---
name: judge-relative-direction
description: 用于 object_rel_direction：绑定站位、面向和目标角色，按 easy/medium/hard 定义判断方向并匹配当前选项。
metadata:
  harness3d-skill-id: S06
  harness3d-version: 1.0.0
  harness3d-question-type: object_rel_direction
  harness3d-family: relative_geometry
  harness3d-source-format: '1.0'
  harness3d-validation: document-checked-runtime-pending
---

# 指定站位和朝向下的方向判断

## 目标

从题面假定观察者的站位和朝向判断目标方向，输出当前题目的合法选项，unit=option。

## 适用条件

保留方向难度与原始选项。几何工具依赖 world_frame，但世界朝向不可用时仍可从图片进行视角推断。

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
  - 三个角色的有效实例绑定
  - 一致坐标、world_up 和 handedness
  - 水平朝向与目标位置
  local_requirements:
    relative_direction_of:
    - geometry_3d
    - world_frame
    - 相关对象有效绑定
  not_required:
  - metric_scale
  visual_route: 从原图中的物体关系形成局部布局，再转到题面观察者视角。
```

## 求解步骤

1. 分别记录 observer、facing_at 和 target 的题面对象，保留 difficulty 与选项集合。这里的“面向某对象”是观察者看向它，不是该家具自身的正面朝向。

2. 利用已有绑定或原图确定三个角色；有多个同类实例时按题面描述和周边结构消歧。用实际可用的补检/裁剪获取有帮助的证据，不把检测候选自动升级为有效绑定。

3. 若 relative_direction_of 对实际三个实例和难度获准，读取方向、角度与退化信息。若授权环境明确允许对已有有效坐标作纯计算，在统一右手系中，将 observer→facing_at 与 observer→target 投影到以单位上向量 u 为法向的水平面；前者归一化为 f，后者为 d，r=f×u，θ=atan2(d·r,d·f)，正角为右。

4. 按题面难度解释角度：easy 判断左右；medium 中 |θ|≥135° 为 back，其余按左右；hard 依据前后和左右分量确定象限。零向量、无可靠上向量或边界歧义时，不能把 atan2(0,0) 或数值符号当成可靠方向；使用受控反馈或视觉判断。

5. 几何不可用时，从包含两个或三个角色的原图建立局部布局，利用跨帧共有结构连接关系；在心中把站位对象放到观察原点，朝向面向对象，再判断目标落在哪一侧。不能直接沿用摄像机画面的左右。

6. 针对最影响选项的歧义查看冻结帧；仍不确定时按现有证据选择最可能的合法方向。将方向映射为本题选项标签并提交，视觉推断参与则如实记录依据。

## 提交前检查

- 站位、面向和目标没有互换，家具朝向未替代观察者朝向。
- difficulty 与 options 来自当前题目，medium 的背后定义没有被 hard 四象限覆盖。
- 几何计算使用同一有效水平系并处理零向量与边界。
- 缺尺度不阻塞相对方向；缺 world_frame 不伪造几何授权。

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

- 接近方向边界时，小的定位误差会改变分类。
- 有限视角中的视觉布局可能存在镜像或深度歧义。

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
  - '6'
  - '9.3'
- ref: VSI
  sections:
  - B.1
  use: 三类方向难度的题意
- ref: THINK3D
  sections:
  - 3D Transformation
  - Spatial Reasoning Agent
  use: 全局与观察者视角转换思想；不引入其新视角渲染工具
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
