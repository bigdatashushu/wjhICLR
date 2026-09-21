# 决策记录：标定池数据集口径（v5.1 修订 HC32 的"指定 ARKitScenes"）

- 日期：2026-09-20
- 决策人：用户（拍板）；实施：Coding Agent
- 影响条文：《系统架构5.md》HC32（标定严格排除评测重叠场景）与 §10.2
- 取代：本记录生效后，"标定池必须来自 ARKitScenes"不再成立

## 背景

评测口径已限定为 **scannet + scannetpp**（用户 2026-09-20 指示，先不用 ARKitScenes）。
原 HC32 要求标定池取"ARKitScenes 非重叠场景"，并在代码里固定成"排除 VSI-Bench 的 150 个
ARKitScenes scene"。这在新的评测口径下有两处不适配：

1. **分布不匹配**：评测跑的是 ScanNet / ScanNet++ 视频（RGB-D 扫描家族），标定若用
   ARKitScenes（iPhone ARKit VIO 管线），覆盖保证的转移论证反而更弱；
2. **数据获取耦合**：把标定卡在 ARKitScenes 上，会连带阻塞 38.1% 的米制题型
   （inner_validation: 489/1283），而这些题在当前系统里恒为 0 分。

三家数据集都提供米制真值：ScanNet 由 RGB-D 深度定尺度；ScanNet++ 由地面激光扫描给出
更精确的米制几何；ARKitScenes 由 ARKit VIO 轨迹给出米制位姿。因此"米制真值有无"不是差别。

## 决定

1. 标定池口径改为：**按被评测数据集选择同源非重叠标定池**，同源优先。
2. 硬门不变（不放宽）：标定集 / conformal 留出集 / 被评测场景集，按原始 scene ID
   **两两互斥**；交集非空即 hard fail，拒绝生成 `scale_calibration_id`。
3. 被评测场景集按数据集参数化：`scannet` 88 / `scannetpp` 50 / `arkitscenes` 150，
   规模与官方不符即 hard fail（防止排除清单不完整导致隔离形同虚设）。
4. 支持**多校准器并存**：目录约定 `<dir>/<dataset>.json`，在线按被评测数据集选用；
   目录内 `calibrator.json` 仅作跨数据集 fallback，加载时记 `dataset_match=false`，
   论文必须显式论证覆盖转移。
5. 记录口径：artifact / trace / RunManifest 均落 `scale_calibration_dataset`、
   `evaluation_datasets`、`dataset_match`、`calibration_split_hash`、
   `excluded_vsibench_scene_hash`。
6. 在线纪律不变：测试时只读冻结校准器，绝不读 GT 位姿 / GT 深度 / 逐场景尺度；
   不设 oracle 路径。GT 只在标定构建期、在非重叠场景上使用。

## 升级门槛不变（§10.2，全部 `[TODO_CALIBRATE]`）

- `medium`：标定集 median relative error < 25%；预测不确定性与真实绝对误差的
  Spearman ρ > 0.5；名义 90% 区间的经验覆盖在容差内接近 90%；该档 Measurement MRA
  不低于 2D-only 基线；
- `high`：更严且预先冻结的误差/区间宽度/覆盖/最小样本门槛；
- 否决：median relative error > 50%、ρ < 0.2、平面身份错误率 > 30%、锚点冲突超限、
  覆盖明显低于名义值——任一发生即不得升级。

## 后续输入（仍需用户提供）

- ScanNet（首选 ScanNet++）**非重叠**场景的米制 GT 位姿及其原始 scene ID 清单；
  需要与 VSI-Bench meta 中该数据集的 scene 集合做审计并持久化哈希。
- 若将来把评测扩展到全量 288 scene，则把三个数据集的排除集合起来，并为每个被评测
  数据集各冻结一份校准器。

## 实现落点

`src/skill3d/reconstruction/scale_calibration.py`（`vsibench_scene_ids_by_dataset` /
`build_scene_id_audit(dataset=…, evaluation_datasets=…, check_expected_count=…)` /
`load_calibrator_for` / `calibrator_path_for`）、`scale_assessment.py`（`calibration_dir` +
`evaluation_datasets`）、`online/{runner,eval}.py`、`schemas/reconstruction.py`
（`scale_calibration_dataset` / `scale_dataset_match`）、`configs/config.yaml`
（`scale.evaluation_datasets` / `scale.calibration_dir`）；
测试：`tests/unit/test_scale_calibration_v51.py`（12 项）。
