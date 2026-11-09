# GroundingDINO 环境截图与标框实测

完成 14 次 GroundingDINO-SwinB 推理和 5 次 Qwen3.8-Flash 静态识别。四张真实 500×500 截图；其中 Manipulation 单独采集无标注 RGB/深度，采集期间动作数为 0。检测使用现有物理 GPU 1 副本；原 RGB-D 源码保持原哈希；没有自动重试。

| 环境 | 宽泛类别 0.15 | 问题类别 0.15 | 问题类别 0.35 |
|---|---:|---:|---:|
| alfred | 15 | 3 | 2 |
| habitat | 14 | 6 | 2 |
| manipulation | 13 | 9 | 3 |
| navigation | 18 | 1 | 1 |

前 12 次是四截图 × 三配置；后 2 次是在同一 Manipulation 原图上分别查询 `star.` 和 `container.`。

框数本身不足以预测模型判断。真实风险包括重复区域、错误类别、空标签以及编号和物体的视觉对应关系。

## 实际观察

Navigation：问题只查 bread 时，从宽泛扫描的 18 框降到 1 框，且保留了面包区域。

ALFRED：问题查纸卷和马桶时，从 15 框降到 3 框，提高阈值后为 2 框。这一帧可见一卷纸，检测框数量不代表任务要求的两卷都已处理。

Habitat：6 框的低阈值结果中，机械臂被标为 toy airplane；0.35 去掉此框，剩 2 框。但两次 Qwen 都认为黄色玩具外形是车而不是飞机，并拒认 toy airplane。减框并没有解决任务名称与视觉外形之间的类别歧义。

Manipulation：复合查询 `star shaped block. container.` 返回 9 框，有 6 对 IoU≥0.75 的高重叠框；0.35 后剩 3 框，却把两个容器也标成 shaped block。单独查询 container 能给两个容器正确标签；单独查询 star 时，真正星形块分数约 0.271，而机械臂误检为 star 的分数约 0.437。因此不能一律提高阈值，也不能把最高置信度当作正确目标。

## Qwen 静态对照

同一原始 RGB 和深度始终保留，第三张图提供候选框；模型参数固定 temperature=0、seed=137、max_tokens=4096、enable_thinking=false。密集和过滤条件使用同一问题、相同候选来源，过滤是分数≥0.35和相同标签 IoU≥0.8 的 NMS。

Manipulation：9 框时，Qwen 说是青绿色星形块，却选择了 IDs 3 和 8；8 实际覆盖绿色容器。3 框时，它选择 ID 4，而 4 实际是绿色容器；可见星形块数量还从 1 报成 2。两种条件下蓝色容器均选 ID 5。

图像检查发现，原图编号 4 画在绿色容器框的上方，恰好落到星形块附近。随后保留同样 3 个框、相同候选表和原始提示，只移开编号并增加一条引线，额外调用 Qwen 1 次。以下是这次真实结果：

```json
{
  "object": {
    "visible": true,
    "candidate_ids": [
      3
    ],
    "description": "A small teal star-shaped block located in the center of the table."
  },
  "destination": {
    "visible": true,
    "candidate_ids": [
      5
    ],
    "description": "A blue rectangular container located on the left side of the table."
  },
  "visible_distinct_task_objects": 2,
  "required_quantity": null,
  "uncertainties": []
}
```

这些是每条件单次静态识别结果，不能将变化全部归因于框数量或编号位置，也不代表环境任务成功率。

## 接入建议

按问题限制类别，但保留足够候选；按查询保留原始标签与置信度，明确未验证。高重叠区域可以分组展示，但不要把相邻物体或部件关系一律删除。避免编号落在其他目标上，必要时用引线和局部裁剪核对框与实例。采用类别/任务相关的阈值和语义复核，避免全局最高分、固定 Top-K 或一律 0.35。

[四环境原图与检测图](computer:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/groundingdino_probe/groundingdino-comparison.png) · [同样三框的编号布局对比](computer:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/groundingdino_probe/images/id-layout-comparison.png)

[12 次主检测完整记录](computer:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/groundingdino_probe/detections.json) · [2 次单类别补测](computer:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/groundingdino_probe/single-category-probes.json) · [4 次密度对照](computer:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/groundingdino_probe/qwen-density-results.json) · [1 次编号布局对照](computer:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/groundingdino_probe/layout-result.json)
