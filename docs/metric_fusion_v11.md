# v11 MoGe-2 度量尺度链路

本链路是待验证的零样本尺度估计，不改变 VGGT 主线、M4 主门、冻结英文 S0 或评测划分。
`scale_fusion_status=success` 只表示本次融合满足当前数值规则，不等于 C1–C6
全部验收通过，也不证明真实尺寸准确或模型得分提高。

## 输入、输出与单位

MoGe-2 与 VGGT 使用同一组冻结原始帧、同一帧序和同一裁剪网格。网格固定为
VGGT 官方 `mode="crop"`：宽 518、高按 14 取整，必要时中心裁剪到 518，
混合高度批次再居中填充。`image_grid.py` 复现这套 PIL bicubic 预处理。
MoGe 接收该网格的 RGB 和 VGGT 内参，不再把原图预测深度直接缩放到 VGGT 深度图。
原始帧仍交给 Qwen 和 SAM2；SAM2 原图掩码以最近邻执行相同几何映射。

固定的 `moge.model.v2.MoGeModel.infer` 已经在内部应用尺度，返回的深度以米为单位。
包装器不再猜测额外的 `metric_scale`，不兼容性重试旧 API，不填造缺失深度或掩码。
模型有效掩码与非黑边、真实图像区域、有限正深度相交后，才参与逐像素比值。
每帧取 `median(metric_depth / VGGT_depth)`，跨帧按已有 MAD/median 规则融合。
VGGT 的深度、位姿平移和世界点图不在磁盘上缩放；工具以该单一比例换算长度，
以其平方换算面积，已有 cm/m 输出规则保持不变。

## 收据、失败与缓存

artifact 增加 `reconstruction_input_identity`、`metric_input_identity`、
`metric_request_sha256`、`metric_receipt_sha256`、`metric_depth_maps` 和
`metric_valid_masks`。收据记录模型源码和权重哈希、运行库版本、融合源码哈希、
预处理、阈值、冻结帧身份、源几何数组哈希、逐帧耗时、FOV 和异常详情。
深度和有效掩码保存为独立 NPY 文件；失败帧保留 NaN 深度和全 false 掩码。
收据写入失败会明确报错，不再静默丢失证据。

真实度量模型的米制门先验证收据及数组身份，再使用有效帧比例。
移植收据、篡改尺度、替换源数组或缺失文件均不授权米制工具。
主门仍为 warp ≥ 0.5 且 cloud ≥ 0.3，尺度离散度 ≤ 0.15、
有效帧比例 ≥ 0.75、有效帧数 ≥ 3；这些仍是尚未统计标定的规则。
尺度门失败只收回依赖尺度的工具，视觉作答和独立评分继续执行。

批量重建不再因“同名 JSON 存在”就跳过。复用需要匹配实际像素、融合开关和
模型/权重/参数身份；冲突时要求新目录或明确的 `--force`。旧缓存没有完整身份时
不能自动推断其正确性。复用历史 VGGT 时，先用历史冻结清单核对原图像素与几何文件，
在新目录 `copy_geometry`，写入已验证的 `reconstruction_input_identity`，
再用 `attach_metric_evidence` 添加新证据；不覆盖历史结果。

B01/B11 在共享准备结束后得到私有副本。两个新增 NPY 引用纳入配对输入哈希与复制；
`_clone_tree` 保留尺度收据的原始字节，避免重新格式化 JSON 破坏收据哈希。
求解阶段禁止再次传入度量模型，沿用既有配对执行约束。

## 启动

`./start_harness.sh` 是 `start_harness3d.sh` 的薄入口，两者使用同一套 bootstrap、
源码冻结、配置、日志和 GPU 检查流程。外部 YAML 中显式设置
`weights.moge2.enabled: true`，已有权重可通过 `MOGE2_EXISTING_PATH` 传入。
默认配置仍不自动启用待验证的尺度分支。

可用 `HARNESS3D_EXISTING_PYTHON` 指向已经安装模型依赖的解释器；启动时检查
Python、vLLM、项目依赖与模型导入，并记录实际包版本，不在该环境执行 pip。
若经过单独回归验证的现有依赖低于项目声明，可在外部 YAML 的
`environment.existing_dependency_versions` 显式固定该版本，差异会写入环境收据；
未声明的冲突仍会阻断启动。普通安装模式仍按项目声明安装。

先执行 `--dry-run`，再执行 `--smoke`，长任务保存 tmux 日志与 PID。
新实验采用 8192 输出 token；历史 c020345/4096 结果保持独立，不能据此直接
推断 MoGe 或 Skill 的因果收益。真实 GPU 验证结果应以各运行目录的报告为准。

### 固定题单验收

外部配置的 `experiment.metric_validation` 可指定 `panel`（预登记 JSON）、
`archive_root`（历史冻结输入与几何证据目录）和可选的 `prepared_cache`
（此前单场景的 `metric_validation` 目录）。此路由仅用于 `--dev` / `--smoke`，
不执行默认入口的 induction 或演化。面板固定 inner_validation、英文、S0 和 B01/B11，
配置预算必须与预登记一致。`--smoke` 只准备面板首个场景；`--dev` 准备全部场景并运行配对。

`run_metric_validation_v11.py` 先复核历史清单、输入文件、像素、几何数组与质量证据，
再复制至新目录添加度量数据。同场景去重发生在推理前，失败帧和失败尺度原样保留，
不换题。缓存复用同时校验面板、历史清单、权重、源码、参数和完整收据。
配对阶段只读准备结果，保存完整逻辑请求、输出、usage、耗时、输入身份和 S0 交付身份。
SDK 内部传输重试仍遵循原客户端策略，完整请求记录不声称逐次记录每个传输尝试。

求解输入的 ground_truth 必须为空。标准引擎输出的分数是空标签占位值，
由 `solve_receipt.json` 明确标注；须在求解全部完成后用独立评分进程生成真实分数，
不得将占位分数用于实验结论。
