# RGB 与深度图输入：说明预算和编码约定

这是一种独立的 RGB+Depth 输入版本。官方 Direct 原图基线保持其既有提示和输入身份。
本次先生成受限说明与长度检查函数；尚未接入图像渲染、模型调用或服务器实验。

## 先确定预算

- 固定英文说明：目标约 **60 个文本 token**，硬上限 **80 个文本 token、40 个英文词**。
- 全部补充文字：最多 **160 个文本 token**，包含每张图的标签、真实尺寸和固定说明。
- 最多 6 张原始 RGB 图及其 6 张对应深度图。尺寸只在 RGB 标签中列一次。
- 不把题干、原始选项、官方字母输出指令和图像 token 算入“补充说明”预算。
- token 计数必须标明 tokenizer；`o200k_base` 可作本地计数基准，不能冒充
  `gpt-5.6-terra` 的已核验 tokenizer 或服务实际账单。
- 生成后重新计数；超过限制先选择等义短版，仍超限则拒绝输出。
  不删真实尺寸，不省略图像对应、无效像素和尺度含义。

这个预算是覆盖必要信息的工程目标，不宣称存在已证明的全局最短提示。
可执行约束位于 `depth_prompt.py` 的 `build_annotation(sizes, token_counter)`。

## 生成后的长度核验

本次使用 `tiktoken` 的 `o200k_base` 对补充文字计数，未调用模型。
编码数据 SHA-256 为
`446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`。

| 内容 | 文本 token | 英文词 |
| --- | ---: | ---: |
| 固定说明 | 59 | 34 |
| 受限短版 | 55 | 30 |
| 2 图示例：标签、1920x1080 尺寸、固定说明 | 83 | — |
| 6 图示例：标签、1920x1080 尺寸、固定说明 | 131 | — |
| 6 图示例：标签、99999x99999 尺寸、固定说明 | 131 | — |

初稿为 44 个英文词，已压缩为当前 34 词版本，并保留 30 词备用版。
所有示例均低于 80/160-token 上限；实际每题仍按其真实尺寸重新计数。
离线检查已覆盖：超过固定说明预算时切换短版、仍超总预算时拒绝，
以及非法尺寸、空输入和超过 6 图的输入拒绝。

## 发送顺序与尺寸

先按原始顺序发送全部 RGB，再按同一顺序发送全部深度图，最后发送简短说明、题干、原选项和官方答题指令。
图像标签紧贴其对应图片：

```text
RGB1 1920x1080
[原始 RGB 图 1]
RGB2 1920x1080
[原始 RGB 图 2]
Depth1
[RGB 图 1 的深度图]
Depth2
[RGB 图 2 的深度图]
[固定说明]
[原题、原选项、官方字母输出指令]
```

上面的尺寸仅示范格式；实际必须来自原图校验回执。`WxH` 表示宽×高。
题目中的图 N、“第二张图”或“最后一张图”只对应原始 RGB 顺序。
深度图必须与对应 RGB 的尺寸、像素位置一致；不拉伸、裁剪或换序来配对。

## 深度图必须实现的显示约定

- 读取已有数值深度及有效掩码，校验原图绑定；不读取标准答案。
- 有效值为掩码有效、有限且大于 0 的预测 camera-depth。
- 每张图单独按有效深度的 P2–P98 区间线性映射并裁剪到灰度：深色近，浅色远。
- 无效像素用洋红色 `(255, 0, 255)`；不与近处像素混用。
- 区间退化时，有效像素统一使用中灰；无有效深度时不生成假图，须另行报告缺失。
- 归一化仅改变显示。缓存中的数值深度不被修改，亮度不代表米或统一的跨图尺度。
- 相机朝向、物体朝向和罗盘方向仍从原 RGB 与题意确定。

这些是待渲染器实现的约定，不是对目前缓存已有 PNG 配色的断言。
只有渲染结果与约定一致，才可以使用以下说明。

## 固定英文说明

```text
Photo/image references mean RGBs only. Depthi matches RGBi pixels/size (WxH); x right, y down. Predicted depth: dark near, light far; magenta invalid. Per-view scaling: no meters or cross-view depth comparison. Directions come from RGBs/question.
```

受限短版保留同样的图像配对、尺寸坐标、预测属性、颜色、无效像素、尺度和朝向说明：

```text
Photo/image references mean RGBs only. Depthi matches RGBi (WxH; x right, y down). Predicted depth: dark near, light far; magenta invalid. Per-view scaling; no meters/cross-view depth comparison. Orient from RGBs/question.
```
