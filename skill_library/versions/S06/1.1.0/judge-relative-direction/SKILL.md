---
name: judge-relative-direction
description: 用于 object_rel_direction：绑定站位、面向和目标三个角色，按题目难度判断观察者视角中的方向并映射到当前选项。
---

# 指定站位和朝向下的方向判断

## 方法

1. 从题面分别提取 `observer`、`facing_at` 和 `target`，同时保留原始选项。这里的“面向某对象”表示观察者朝向该对象，不是家具自身的正面方向。
2. 对三个角色分别完成实例消歧；同类对象有多个时，利用题面修饰、共现结构和可见帧确定具体 `obj_id`，不能依赖类别名默认命中的第一个实例。
3. 根据选项集合确定 `difficulty`：只有 left/right 使用 `easy`；left/right/back 使用 `medium`；四个前后左右象限使用 `hard`。
4. 几何与世界系获准时，调用 `relative_direction_of(observer_id, facing_at_id, target_id, difficulty)`。读取并核对 `direction`、`difficulty`、`world_up_used` 和 `handedness_used`。
5. 将返回方向严格映射到当前题目选项。easy 只接受 left/right；medium 只接受 left/right/back；hard 只接受 front-left、front-right、back-left、back-right。
6. 工具不可用时，从原图建立三个角色的局部布局：把 observer 放在假想原点，朝向 facing_at，再判断 target 落在该观察者的哪一侧；不能直接沿用摄像机画面的左右。

## 分支与检查

- `observer`、`facing_at` 和 `target` 不能互换，也不能把 `facing_at` 当作家具朝向属性。
- 相对方向不需要米制尺度；缺少 metric scale 不能单独阻塞本方法。
- 缺少有效世界系、上方向或对象绑定时不得伪造工具方向，应使用视觉路径。
- 工具返回的 `difficulty` 必须与题目选项集合一致；返回词不在选项中表示调用参数或映射有误。
- 接近方向边界、水平向量退化或视觉布局存在镜像歧义时，应优先核查会改变选项的帧和实例绑定。
- 提交前检查三个角色、难度词表和选项映射，确保输出是当前题目的合法选项。
