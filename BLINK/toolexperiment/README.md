# BLINK 四工具实测

直接调用原四数据集实验的冻结 `small_image` 工具，读取已发布的 133 题、
266 张图像对应的缓存。测量单图工具的正确性、可用性、稳定性和有限的几何合理性。

实验不调用 Qwen 或预计算模型，不读取 BLINK 答案，不修改原实验和缓存。

已完成实测，结论见 [实验结果](实验结果.md)，原始逐题记录位于 `results/raw.jsonl`。

## 检查内容

- 全部缓存通过原 `load_prepared` 的字节及内容身份核验。
- 通过原 `ToolRegistry.call_tool` 实际调用四个工具；全部 133 题同时经过原沙箱执行。
- `view_candidates`：检查实际候选与返回值，并区分空候选和物体不存在。
- `view_depth_summary`：独立重算有效像素数、10/50/90% 分位数。
- `point_depth`：每张图固定 5×5 网格采样，与原深度图独立计算结果比较；检查 ±2 像素扰动及无效坐标。
- `pair_point_distance`：同视图网格点对独立重算，检查交换对称、同点距离、端点扰动。
- 原图先行标注 12 组明显前后关系，再检验深度工具的排序。标注文件先冻结，保留全部结果。
- 在 RGB 上使用 SIFT 双向匹配和基础矩阵 RANSAC，比较匹配点构成的距离比在两视图是否稳定；各视图允许一个独立尺度。该检查不提供真实深度、真实距离或相机位姿。

数值一致性证明工具忠实使用缓存。RGB 前后关系与跨视图距离比检查提供有限的几何诊断。
BLINK 原始数据没有深度/距离真值，因此本实验不会把这些诊断称为米制准确率，也不产生
“工具提高 BLINK 答题准确率”的结论。

测量程序、协议、逐题输出和最终报告保存在本目录；大缓存保留在 Titan。

## 复现命令

```bash
# 在有兼容 OpenCV 的本地 Python 环境中准备 RGB 匹配：
python prepare_rgb_checks.py --dataset-root /path/to/dataset --output rgb_correspondences.json
# 使用已有 SSH 认证执行。输出目录须为新目录，程序不会覆盖既有结果：
python run_remote.py --output /path/to/new-results
# 现有私有 askpass 可以通过 --askpass /private/path/helper 传入；不复制凭据。
# 核验本次 results/ 下的记录并重新生成报告：
python summarize.py --dataset-root /path/to/dataset --font /path/to/Times-New-Roman.ttf
```

Titan 使用原运行环境的 `cpu_compat` OpenCV。初次检查系统 OpenCV 的 GLIBC 不匹配，
实际测量使用已有兼容包，不安装或替换服务环境。单样本准备检查不计入全量实测次数。
视觉标注只是测试工具的定性诊断，不会进入 Skill 修订或在线求解。
