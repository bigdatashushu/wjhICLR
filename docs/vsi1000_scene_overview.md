# VSI1000 场景总览输入

VSI1000 可将已生成的 640×480 彩色点云总览随每题送给模型。原始 FrameSet 保持 32 帧，首轮请求为一张总览加 32 张原帧；后续轮次总览固定占一个独立位置，其余 32 个位置沿用原来的原帧/裁剪选择。总览不计入裁剪配额。

在 `configs/experiments/vsi_m4_1000.json` 的 `inputs` 中设置：

```json
"scene_overviews": "{data_root}/scene_overviews"
```

该目录包含 `manifest.json` 和 `<scene_id>.png`。不配置此项时，原有输入和账本协议保持原样。只允许在显式 train/test 数据配置中启用。

`experiment.sh --check` 检查 1,000 题对应的 164 个场景都有总览，校验 PNG/RGB 哈希与原始场景引用。运行器加载总览描述时再次验证文件；像素解码缓存有界且不可写，缓存不会跳过内容校验。正式 campaign 把整个总览目录纳入输入冻结，恢复时拒绝变更后的图片。

在线账本使用独立的 `scene_overview` 类型，记录渲染版本、图片内容、原始重建和冻结场景身份。总览没有视频帧编号、时间戳或原帧裁剪框。提示明确说明它是低清重建剖视图，可能缺面、有噪声，视图中的左右不自动等于题目要求的方向，也不能授权米制几何。

训练、父版/候选 Gate、初始/最终测试使用相同机制，包括程序修复、最终作答和系统视觉补答。图片只有实际进入请求并得到响应后才记为 observed。独立 `prepared-32` 直答 baseline 保持 32 帧协议，其输入不包含重建总览。

学习轨迹仍保留恰好 32 个原帧引用。启用时使用额外的 `scene_overview_ref` 和新的离线输入 schema；每场景只保存一份总览 PNG，分析与修订通过同一内容引用还原输入。原始 M4/M5 产物、FrameSet、300/700 划分、12 轨迹修订窗口和双 seed Gate 不变。

运行入口增加 `--scene-overview-manifest`，由统一 launcher 自动传递；通常无需手动设置图片数量或修改 `online_model.n_frames`。模型请求预算仍由原 request governor 管理，正式运行需单独给定预算。
