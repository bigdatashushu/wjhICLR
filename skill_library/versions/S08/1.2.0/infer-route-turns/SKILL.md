---
name: infer-route-turns
description: 用于 route_planning：解析给定路线的起点、初始朝向和有序路标，利用有效世界上向量和手性逐段判断水平转向。
---

# 沿给定路线判断转向

## 方法

1. 将题面展开为起点、初始面向对象、有序路标、直行段和待填写转向。保留每个选项的完整动作序列，求解题目指定路线，不自行改成另一条更短路线。
2. 使用原图和对象记录确认各路标实例及顺序。相同类别路标有多个时，利用题面描述、门洞、相邻结构和观察顺序消歧，不能把重访同一位置当作新路标。
3. 对每个转弯点建立“前一位置、当前位置、下一位置”三元组。到达当前位置后的朝向继承前一段行进方向；下一动作相对于该方向判断。若题目问起点处的首次转向，前向量取“初始面向对象 − 起点”，目标向量取“首个路标 − 起点”。
4. 几何获准时，用 `object_centroid(obj_id)` 的 `centroid_normalized` 取得坐标；从同一返回中读取 `world_up_used` 与 `handedness_used`。仅在各点处于同一坐标系、方向元数据有效且一致时计算。令单位上向量为 u、来向 f = 当前 − 前一位置、去向 g = 下一位置 − 当前。先分别投影到水平面：f_h = f − dot(f,u)u，g_h = g − dot(g,u)u，再求有符号转角。任一水平向量近零，或方向元数据为 None 时，不猜坐标轴、手性或转角。
5. 以下函数返回角度（度）：正值为右转，负值为左转，零附近为直行，绝对值接近180为掉头。手性为 left 时翻转符号。近共线时结合点积区分同向和反向，不仅凭叉积正负判左右；词表、角度容差及动作序列必须结合题面和视觉布局，不能套用方向题 medium 的135度阈值。

```python
import numpy as np
import math

def route_turn_angle(previous, current, following, world_up, handedness):
    if world_up is None or handedness not in ("right", "left"):
        return None
    u = np.asarray(world_up, dtype=float)
    if u.shape != (3,) or not np.all(np.isfinite(u)) or np.linalg.norm(u) < 1e-9:
        return None
    u = u / np.linalg.norm(u)
    f = np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)
    g = np.asarray(following, dtype=float) - np.asarray(current, dtype=float)
    if f.shape != (3,) or g.shape != (3,):
        return None
    f = f - np.dot(f, u) * u
    g = g - np.dot(g, u) * u
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(g)):
        return None
    if np.linalg.norm(f) < 1e-9 or np.linalg.norm(g) < 1e-9:
        return None
    f = f / np.linalg.norm(f)
    g = g / np.linalg.norm(g)
    side = float(np.dot(g, np.cross(f, u)))
    if handedness == "left":
        side = -side
    return math.degrees(math.atan2(side, float(np.dot(g, f))))
```

6. `connectivity_graph()` 只用于核查有证据支持的可达关系和不可能路段；对象欧氏距离近、位于同一房间或图中相邻都不能单独证明可通行。
7. 几何或世界方向约定不可用时，沿题面路线在原图布局中逐段移动假想观察者，用墙面、门洞、通道和相邻物体核对转向。每完成一段都更新当前朝向。
8. 按题面顺序完整演练每个候选动作序列，检查各段是否到达指定路标和终点，再选择最有依据的合法选项。

## 分支与检查

- 相对转向不需要米制尺度；metric scale 缺失不能单独阻塞路线判断。
- `relative_direction_of(observer_id=前一位置, facing_at_id=当前位置, target_id=下一位置)` 的观察者仍在前一位置，只能辅助判断折线的左右侧，不能替代在当前位置算转角，尤其不能用于判掉头。
- `connectivity_graph` 依赖世界系和点云；缺失或失败时不得把未知通行性写成不可达。
- 没有明确转弯点或可通行边界时，对象中心折线只能作为线索，应结合原图布局判断。
- 提交前检查每个动作都相对于到达该路标时的朝向，并与当前题目的完整动作序列选项一致。
