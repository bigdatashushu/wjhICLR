# Skill3D

在 **VSI-Bench**（Visual-Spatial Intelligence Benchmark）上，一个 3D 空间感知 VLM 自演进
智能体：**在线小模型 Qwen3-VL-8B 确定性执行 + 离线强模型 GPT-6 归纳治理**，自动从轨迹中
归纳并准入跨任务 Memory/Skill（`系统架构.md` §0.1）。

本仓库按 **`系统架构4.md`（v4 工程规格）** 实现。v4 采用「完整母版 + 保守增量修订」：
v3 的全部硬约束 1–28、模块规格、接口契约继续有效，v4 新增 **29–34** 条
（尺度 CI 统一口径 / 置信度不得靠常量或放宽门槛提升 / 多锚点鲁棒融合 / ARKitScenes 标定严格
排除评测场景 / 尺度按题型授权 / Experiment Readiness Gate）。逐条落地记录见
[`架构v4实现对照.md`](架构v4实现对照.md)（v3 记录见 [`架构v3实现对照.md`](架构v3实现对照.md)）。
历史规格与审计报告（`系统架构.md` / `系统架构2.md` / `系统架构3.md` / `实现缺口报告.md` /
`问题报告3.md`）保留在仓库根目录上一层供对照。原 Skill-3D 论文代码保留在 `legacy/`
仅作参考（见 [`legacy/MIGRATION.md`](legacy/MIGRATION.md)），新系统不 import 它。

> **本机资源已实测**：GPU/模型权重/数据集/上游 repo 的位置与可用性见
> [`本机环境实测.md`](本机环境实测.md)（把 §15.1 的 `TODO_USER_INPUT` 逐条换成了已核验事实）。
> 一句话：**首个真实 episode 端到端已跑通**（真实视频 + VGGT + Qwen3-VL-8B-FP8 + SAM2，13 状态全走通）。
> 冻结实验环境为 `skill3d-exp`（vLLM 0.19.1；含 vggt 源码/pycolmap/open3d/sam2）。

## 快速开始

```bash
pip install -e . --no-build-isolation        # 本机无网络时加 --no-deps
pytest                                       # 单测 + 集成测试（无需 GPU / 数据集）
```

> **运行环境（必读）**：本项目**必须**用冻结实验环境 `skill3d-exp` 运行：
>
> ```bash
> EXP=/home/cvailab/anaconda3/envs/skill3d-exp/bin/python
> PYTHONPATH=src $EXP -m pytest tests/ -q     # 全量套件
> ```
>
> base conda env（numpy 1.26.4 + scipy 1.18.0）会**装得上但跑不动**：scipy 自 1.14
> 起要求 numpy≥2.0，于是 `import scipy.spatial` 抛 `np.long`、`cKDTree` 抛
> `copy=None`。危险之处是它**不报错退出** —— M4 主门子项 fail-closed 之后
> `scene_route=fallback_2d_only`，整条链照常跑完，几何与米制能力静默失效，
> 看起来像"场景质量不够"。为此 `skill3d.online.eval` 启动时先做**行为级**环境
> 预检（`src/skill3d/env_preflight.py`）：依赖不可用即拒绝启动，不在降级状态下
> 出结果。不要用测试进程内 NumPy shim 掩盖该问题 —— shim 只能补一个符号，
> 补不了 ABI。

四条入口（§13.5 + G-35 离线 driver）：

```bash
# P1 批量 3D 重建（需 VGGT 权重 + 原始视频；缺则明确报错，不伪造产物）
#   --ba 启用 BA route（§10.1 [Conditional Go]，前置不满足自动回退 feed-forward）
#   M4 质量随重建一起算并写回 artifact（方案 X），P2 加载即得实算值
python -m skill3d.reconstruction.run --split induction,inner_validation,outer_holdout --method vggt

# P2 在线评测（-h 看全部参数）
python -m skill3d.online.eval --split inner_validation --source synthetic --mode mock_light
python -m skill3d.online.eval --split test --active-snapshot data/active_snapshot.json   # 需 --allow-final-test
#   --recon-method vggt_ba / --ba：BA route；--allow-tool-contract-replay：D-3 回灌恢复层
#   --scale-calibration <cal.json>：冻结尺度校准器（HC32；缺省读 configs/config.yaml 的
#       scale.calibration_path；不存在 → scale_confidence 恒 low、米制题型逐题收回）
python -m skill3d.online.eval --mode real --source vsi_bench --split inner_validation --limit 1 \
       --vllm-endpoint http://127.0.0.1:8100 --vllm-model qwen3vl-8b-r0

# P3 离线候选优化循环（L1→L2→L3 → 准入 → 原子 promote）
python -m skill3d.evolution.optimize_loop --root-candidate-id <id> --candidates data/candidates.jsonl

# G-35 离线演进 FSM driver（CLUSTER→归纳→泄漏门→优化循环→准入；每步落 checkpoint 可 resume）
python -m skill3d.evolution.offline_driver --mode real --run-id gen-0001
python -m skill3d.evolution.offline_driver --resume --checkpoint data/offline_runs/latest.json
```

v4 尺度三件套（§10.2 PoC 阶梯 / HC32 标定构建 / HC34 readiness）：

```bash
# §10.2 D-2 PoC 阶梯：L0 口径 + L1 合成多锚点（离线可跑）；L2/L3 需真实标定数据
#   缺数据时输出 UNVERIFIED 且退出码 1 —— 不得用合成数据冒充标定结果
python -m skill3d.reconstruction.scale_poc --out data/scale_poc_receipt.json
python -m skill3d.reconstruction.scale_poc \
    --l2-records data/scale_calibration/records.jsonl \
    --meta data/vsi_bench_meta/test.jsonl --out data/scale_poc_receipt.json

# HC34 Experiment Readiness Gate：写/查 readiness_manifest.json（--check 未达级退出码 1）
python -m skill3d.readiness.manifest --out data/readiness_manifest.json
python -m skill3d.readiness.manifest --check scale_recovery      # 未达 paper_eligible → 1
```

> **标定构建（HC32）**：冻结校准器由 `skill3d.reconstruction.scale_calibration` 的
> `build_scene_id_audit` + `fit_conformal_calibrator` + `save_calibrator` 在**离线标定期**
> 生成；构建时会强制断言「标定 scene ID ∩ VSI-Bench 150 个 ARKitScenes scene = ∅」，
> 交集非空即 hard fail。在线只加载冻结产物，绝不读 GT 位姿/深度/逐场景尺度。
> 当前缺 ARKitScenes 非重叠场景 GT 位姿（`TODO_USER_INPUT`）→ 无校准器 → 一律 low。

split 构建（G-09，离线一次性）：

```bash
python scripts/build_splits.py                      # 读 data/vsi_bench_meta/test.jsonl
python scripts/build_splits.py --final-ratio 0.2 --seed 0
```

trace 聚合（M13，JSONL → Parquet）：

```bash
python -m skill3d.trace.store --trace-dir data/traces --out-dir data/parquet
```

消融档（§16.1）：`python -m skill3d.evolution.offline_driver --ablation E3`（E0–E5）、
`--governance G1_no_review`（G0–G2）、`python -m skill3d.online.eval --inject-wrong-skill`（C5）、
`--skill-spec spec.json`（C2）；全表见 `skill3d.evolution.ablation.format_ablation_markdown()`。

## 当前可跑 / 待接入（诚实边界）

| 能力 | 状态 | 条件 |
|---|---|---|
| 在线链 M1→M13 端到端（含 AST→沙箱→几何校验→评测→trace） | ✅ 可跑 | `--mode mock_light`（合成输入 + 确定性 stub program） |
| 同 seed 重放字节级一致（§4 M8/M17 验收） | ✅ 可跑 | `--deterministic-replay` |
| **统一固定 FrameSet**（硬约束 21）：32 帧唯一均匀采样 + `frame_set_hash`；M2 只被动观测（不删/不换/不补/不重排帧） | ✅ 可跑 | `adapters/frame_set.py`；`tests/unit/test_v3_contracts.py` |
| **Tool 执行期 fail-closed**（硬约束 23）：缺产物抛 `ArtifactUnavailableError`（不静默返回 False/0）；`docs(route)` 按 route 静态裁剪；**程序自捕获契约异常后作答也不采纳** | ✅ 可跑 | `tools/contract.py` + `REGISTRY.docs(route=...)`；`tests/unit/test_v3_gap_fixes.py` |
| **D-3 契约恢复阶梯**：回灌一次 → 裁剪 prompt 一次 → 显式 abstain（答案不得采纳、主榜按错计）；rung1 失败会继续落到 rung2 | ✅ 可跑（机制） | `--allow-tool-contract-replay`；回灌修复率 ≥50% 才启用（[Conditional Go]） |
| **质量单一事实源**（硬约束 22）：`quality_status` 三态（含 `failed` 生产者）+ route fail-closed（NaN 不得 full_3d）；P1 写回（方案 X）/ P2 兜底写回（方案 Y）+ **M5 的 G7/G9 增量补写** | ✅ 可跑 | `reconstruction_gate/quality_metrics.py`；真实 artifact 已验证（overall=0.7924） |
| **MRA 严格官方口径**（硬约束 27）：`rel=|pred-gt|/gt`、`rel<=1-θ`、`θ=linspace(.5,.95,10)`；golden rel→{1,1,.9,.7,.1,0} | ✅ 可跑 | `evaluation/mra.py`；`tests/unit/test_evaluation.py` |
| **route 只由质量决定**（Appendix A）：尺度不参与全局 route，只走逐题通道（v4：`allowed_metric_tasks` 逐题型授权 + G9 附加条件）；**尺度为 low 只收回米制 Tool，不连累非米制 3D 能力**（HC33） | ✅ 可跑 | `reconstruction_gate/scene_state.py` + `tools/contract.py`；`tests/integration/test_metric_task_authorization_v4.py` |
| **v4 尺度 CI 统一口径**（HC29）：`scale_ci_rel` = 无量纲半宽分数，`scale_ci_abs_m = metric_scale × scale_ci_rel`；百分数/全宽/标准差/NaN/历史字段全部 fail-closed | ✅ 可跑 | `reconstruction/scale_units.py`；`tests/unit/test_scale_units_v4.py`（21 例） |
| **v4 多锚点鲁棒融合**（HC31）：地平面/相机高 + 门/桌/椅等标准物体锚点；log 空间共识窗口 + Huber 精修；每锚点留来源/估计/不确定性/残差/接受状态；显式冲突检测 | ✅ 可跑 | `reconstruction/metric_scale.py`（`fuse_scale_anchors_robust`）；`tests/unit/test_scale_assessment_v4.py` |
| **v4 置信度派生**（HC30）：未加载冻结校准器 → **一律 low**（不因锚点多/CI 小升级）；口径异常/锚点冲突/经验覆盖超容差/置信水平不一致 → low；数据层兜底禁止未标定的 medium/high | ✅ 可跑 | `reconstruction/scale_assessment.py` + `schemas/reconstruction.py`；`test_uncalibrated_never_leaves_low` |
| **v4 conformal 校准 + 标定隔离**（HC32）：ARKitScenes 标定集与 VSI-Bench 150 个 scene 按原始 ID 求差，**交集非空即 hard fail**；在线只读冻结校准器；校准只加宽不收窄 | ✅ 机制就位 / ⏳ 标定数据未到位 | `reconstruction/scale_calibration.py`；`tests/unit/test_scale_calibration_v4.py` |
| **v4 §10.2 PoC 阶梯**：L0 口径 / L1 合成多锚点 / L2 标定规模消融（N∈{10,30,100}，N=10 仅诊断）/ L3 真实 held-out 逐题型验收；缺数据一律 UNVERIFIED | ✅ L0/L1 PASS / ⏳ L2/L3 UNVERIFIED | `python -m skill3d.reconstruction.scale_poc` |
| **v4 Experiment Readiness Gate**（HC34）：implemented ⊂ connected ⊂ real_poc_verified ⊂ paper_eligible 四级单调；未达级 `assert_paper_eligible` 拒写主表；实况落 `readiness_manifest.json` | ✅ 可跑 | `python -m skill3d.readiness.manifest`；`tests/unit/test_readiness_v4.py` |
| **v4 尺度报告**（§7）：逐题型 MRA/refusal/coverage + 锚点触发率/接受率/冲突率 + 名义 vs 经验覆盖 + 来源分布（mock 来源强制标注），**单列小节不混主表** | ✅ 可跑 | `evaluation/scale_report.py`；`online/eval.py` 打印小节 |
| **§7 统计口径**：paired A/B 用 BCa bootstrap（`n_resamples=9999`）+ Wilcoxon + Bonferroni + Cliff's delta / Cohen's d；退化样本 p→None（判不显著） | ✅ 可跑 | `evolution/paired_score.py`；`tests/unit/test_paired_score_spec.py` |
| **BA route 两条 route**（§10.1 [Conditional Go]）：PoC 探测 + 正方形预处理前提 + `min_inlier=64` 策略 + 失败回退（G5 记 None、弱代理） | ✅ 机制就位 / ⏳ PoC 未跑 | `reconstruction/ba_route.py`；`--ba` |
| **§9 坐标适配层**：VLM-1000→像素、mask→深度网格最近邻、光流在深度网格、c2w 统一暴露 | ✅ 可跑 | `src/skill3d/coords.py` |
| **D-2 尺度档** | ✅ 机制就位（v4 重写） / ⏳ 标定未跑 | `reconstruction/scale_assessment.py`：多锚点融合 + 冻结 conformal 校准 + 逐题型授权；**未标定 → 一律 low**（HC30） |
| 输入/重建质量门禁 G1–G11、MRA/Accuracy、准入硬门、promote 原子切换 | ✅ 可跑 | 纯 CPU 确定性实现 |
| **VSI-Bench 四层 split**（G-09）：288 scene / 5130 QA，四列表互斥、10 题型均有代表 | ✅ 已生成 | `configs/vsi_bench_split.yaml` + `configs/contamination_check.log` |
| **尺度锚定**（G-11，自研替代 PaGeR）：标准物体先验 + 地平面 RANSAC + 鲁棒融合（log 空间共识窗口 + Huber）+ MAD CI | ✅ 可跑 | 合成真值上尺度误差 <15%（`tests/unit/test_metric_scale.py`） |
| **G5/G7/G9 数据源接线**（G-18）：BA 重投影 / 刚性残差动态掩码 / track IoU；G9 在 **3D 去重后**对象集上算 | ✅ 可跑 | 缺产物时为 NaN，不伪造；**G8 已按附录 A 删除**（`bbox_point_coverage` 度量本体一并移除）；G11 改分数口径，未授权题型逐题收回米制 Tool |
| **M5 3D 去重**（§3 M5 字段 5）：类别 + 世界质心距离 + 时序重叠三判据 | ✅ 可跑 | `segmentation/sam2_tracker.py`；`tests/unit/test_sam2_tracker.py` |
| **离线演进闭环**（G-28/G-35）：归纳→泄漏门→L1/L2/L3→准入→原子 promote（新增 PROMOTE 可达性护栏） | ✅ 可跑 | mock GPT-6 注入下跑通全状态链 + checkpoint/resume（`tests/unit/test_offline_driver.py`） |
| **在线 episodic 记忆**（G-26）+ RunManifest（G-67，含 v3 推理参数）+ 过程指标/统计（G-65/G-66） | ✅ 可跑 | final_test 不写 Memory；统计走 scipy/numpy 双实现 |
| **系统可靠性指标**：coverage / refusal_rate / tool_contract 率 / coverage-conditioned MRA（**不混主表**） | ✅ 可跑 | `evaluation/process_metrics.py`；`online/eval.py` 打印小节 |
| **语义检索**（G-20）：hashing 嵌入 + LanceDB 持久化 + reranker 回退 | ✅ 可跑 | 语义排序与关键词排序口径不同（`tests/unit/test_skill_retrieval_vector.py`） |
| **离线监控 → 回滚/再审**（G-37/G-39）+ 审计回溯（G-40） | ✅ 可跑 | 指标超阈 → review/rollback；`revision_id` 可回溯全链（缺环显式标 missing） |
| golden / fuzz 测试层级（G-02/G-03） | ✅ 已补 | `tests/golden/evolution/`（冻结 artifact + 字节级重放）、`tests/fuzz/metamorphic/`（hypothesis 5 类 MR） |
| **消融矩阵**（G-62/G-63/G-64）：C0–C5 / E0–E5 / G0–G2 | ✅ 配置化可跑 | `skill3d.evolution.ablation`；E 档逐格对应 §8.3 表格 |
| 真实重建（VGGT / DUSt3R / COLMAP）、SAM2 绑定 | ✅ 已实测跑通 | 权重与视频已就位（`本机环境实测.md`）；`--recon-method vggt` |
| 在线 program 生成（Qwen3-VL-8B，多模态 32 帧） | ✅ 已实测跑通 | `bash scripts/serve_qwen3vl_dp8.sh` 起本地 vLLM，再 `--mode real --vllm-endpoint ...` |
| GPT-6 端点 | ⏳ 待接入 | `TODO_USER_INPUT`：`SKILL3D_GPT6_ENDPOINT/MODEL_ID/API_KEY` |
| **VSI-Bench 原始视频** | ✅ 已就位 | `/nas/wangjh/Skill-3D/dataset/VSI-Bench`（288/288 scene，布局与代码约定一致） |
| **Qwen3-VL-8B 本地服务** | ✅ 实测跑通 | FP8 权重已在 HF 缓存；`bash scripts/serve_qwen3vl_dp8.sh` |
| **VGGT 重建** | ✅ 实测跑通 | 源码 `third_party/vggt` + 权重 4.7GB；32 帧 65.9s |
| **SAM2 对象绑定** | ✅ 实测跑通 | `sam2==1.1.0` + SAM2.1 Hiera-Large；4 探测帧提示 → 双向传播 → 世界点云 → 3D 去重 |
| **端到端真实 episode** | ✅ 已跑通 | M1→M13 全链（VGGT + Qwen3-VL + SAM2 均真实） |

> **mock_light 不是实验结果**：合成输入 + stub program 只用于管道验证与验收
> （§9.2 / §12 M0-M1）。准入门强制 `real`（§5.6b），mock 跑不会 promote 任何候选。

## 目录结构（§2）

```
configs/                 Hydra 风格 YAML（本机无 hydra 时用 yaml 直读，见 online/config.py）
  config.yaml / vsi_bench_split.yaml / qwen3vl_dp8.yaml / sandbox_docker.yaml
  admission_thresholds.yaml          准入门阈值（全部 TODO_CALIBRATE）
docker/                  Dockerfile.sandbox（只读根 FS / --network none / 不挂 GPU）
                         Dockerfile.vllm（Qwen3-VL-8B）
scripts/serve_qwen3vl_dp8.sh           Qwen3-VL-8B DP×8 启动（不 TP，ADR-2）
src/skill3d/
  coords.py    §9 坐标适配层（VLM-1000→像素 / mask→深度网格最近邻 / 光流同网格 / c2w 约定）
  adapters/    M1 VSI-Bench adapter + frame_set.py（32 帧帧集单一事实源 + frame_set_hash）
               + episode 数据源 + split_builder（G-09 切分与污染断言）
  gates/       M2 输入门禁（**被动观测**：flag/weight，不删帧）与 IQA 算子
  reconstruction/   M3 VGGT feed-forward 主线 + dust3r/COLMAP 备选
                   + v4 尺度支路：scale_units（HC29 口径）/ metric_scale（HC31 鲁棒融合）
                   / scale_calibration（HC32 冻结校准器 + 隔离审计）/ scale_assessment（HC30+HC33）/ scale_poc（§10.2 L0–L3）
                    ba_route.py：两条 route（feed-forward / BA [Conditional Go]）+ PoC 探测
                    ba.py：pycolmap 读回 sparse 模型 → 重投影残差/COLMAP artifact（G-13/G-15）
                    run.py = P1 CLI（M4 质量随重建写回，方案 X）
  reconstruction_gate/  M4 G1–G10（G8 已删）/ 置信度 / SceneState + 质量写回（quality_status 单一事实源）
  segmentation/ M5 SAM2 视频分割与对象绑定（4 探测帧提示 + 双向传播 + 3D 质心去重）
  tools/       M6 Tool Registry（真实/三档 Mock）+ contract.py（ROUTE_ARTIFACTS 与 fail-closed 异常族）
  routing/     M7 题型识别 + Skill 硬过滤（v4：米制 Skill 按 allowed_metric_tasks 逐题型授权）/检索
  synthesis/   M8 program 合成（多模态 messages / route 裁剪 prompt / vLLM 客户端）
  sandbox/     M9 AST 白名单 + M10 持久 kernel/容器 + receipt 哈希链（tool_contract 归因）
  verifier/    M11 确定性几何校验（不调 VLM）
  evaluation/  M12 Accuracy / MRA（严格官方口径）+ process_metrics（G-65 与可靠性指标）
               + scale_report.py（v4 §7 逐题型 MRA/refusal/coverage + 锚点/冲突率，不混主表）
               + multi_seed_aggregator（G-66）
  trace/       M13 Trace Store（JSONL→Parquet）
  memory/      M14 三层记忆 + 巩固/防污染 + online_memory（G-26 在线 episodic + 三因子检索）
               vector_index.py = G-20 hashing 嵌入 + LanceDB 持久化（可选）
  skills/      M15 Skill Registry 语义版本 + 原子 promote/回滚
  governance/  M16 GPT-6 离线归纳/修订/审查（**仅离线**）
  evolution/   M17–M19 演进沙箱/配对评分/反例/优化循环 + panel.py（共用面板装配）
               offline_driver.py = G-35 离线 FSM driver；ablation.py = G-62/63/64 消融档
               monitoring.py = G-37/G-39 离线监控→回滚/再审；optimize_loop.py = P3 候选级 CLI
  scheduling/  M20 8×4090 DP 调度（DP×8 不 TP）
  infra/       M21 Hydra/MLflow/版本锁定/断点恢复 + audit.py（G-40 证据链回溯）
  schemas/     §5 Pydantic v2 单一事实源（extra="forbid"；含 readiness.py 的 HC34 四级证据门）
  readiness/   HC34 Experiment Readiness Gate（readiness_manifest.json + 主表写入门）
  fsm/         §6 在线/离线状态机
  online/      在线链编排（runner= M1→M13 driver，eval.py = P2 CLI，synthetic= mock_light）
scripts/build_splits.py               G-09 四层 split 构建 CLI
tests/unit/ tests/integration/        §2 测试层级（含 v3 契约：FrameSet / fail-closed /
                                      D-3 恢复阶梯 / MRA golden / D-2 尺度档）
tests/golden/evolution/               G-02 冻结 A/B artifact + 幂等重放 + 统计基线
tests/fuzz/metamorphic/               G-03 hypothesis 驱动的 5 类空间 MR
data/                                 不入库（§2）；trace/重建产物落此处
legacy/                               原 Skill-3D 代码（参考，不参与新系统）
```

## 硬约束（红线摘要，全文见 §0.2）

1. **在线链绝对无 GPT-6**——`governance` 只能离线 import；静态守卫见
   `tests/unit/test_no_gpt6_online.py`（扫描 `src/skill3d` 全部在线目录）。
2. **GPT-6 仅离线**：归纳 candidate_v0 / 产 patch / 语义审查；不执行分支、不评分、不决定 promote。
3. 一切阈值标 `TODO_CALIBRATE`，GPT-6 参数标 `TODO_USER_INPUT`，不得当实测结论。
4. 重建先于 Skill 路由；Tool 只经 SceneState 句柄访问产物；Tool 预封装、不每题生成。
5. 候选不可变；promote 原子切换可回滚；paired A/B 复用同一 ReconstructionArtifact
   （**逐 episode 断言同 artifact ref 与同 `frame_set_hash`**：`skills/paired_ab.py`）。
6. **Final test 完全隔离**：默认拒绝进在线链与批量重建（需显式 `--allow-final-test`，仅盲评一次）。
7. 几何校验为确定性硬门（不引入第二 VLM 当裁判）。

v3 新增（21–28）：

21. **统一固定 FrameSet**：全链路 32 帧同一帧序与同一 `frame_set_hash`；M2 只被动观测
    （不删/不换/不补/不重排帧），禁止双帧集。
22. **`quality_status` 是质量唯一事实源**：非 `computed` / `quality is None` / `overall_quality`
    为 NaN → **不得**停在 `full_3d`（禁止 `q.overall_quality < TH` 这类 NaN 下恒假的判断）。
23. **Tool 执行期 fail-closed**：缺产物抛 `ArtifactUnavailableError`，绝不静默返回 `False/0`；
    `exists_in_scene(name)==False` 当且仅当 objects 产物可用且真无此实例。
24. 在线链无 GPT-6（重申）；正式主结果用本地冻结 Qwen3-VL-8B-Instruct-FP8。
25. vLLM 与 VGGT/SAM2/BA **禁止同卡**（分卡部署，实测同卡 OOM）。
26. **M8 必须多模态**：同一 FrameSet 的 32 张图像 + 文本；不得纯文本生成程序、不得静默丢帧。
27. **MRA 严格对齐官方**：`rel=|pred-gt|/gt`，`rel<=1-theta`，`theta∈linspace(0.5,0.95,10)`。
28. 已修项（A/C/E 类）即硬约束/接口契约（见 `架构v3实现对照.md` §2–§3）。

v4 新增（29–34）：

29. **尺度不确定性统一口径**：`scale_ci_rel` 是 `[0,+∞)` 的**无量纲半宽分数**，
    `scale_ci_abs_m = metric_scale × scale_ci_rel`。禁止百分数/全宽/标准差/不同置信水平
    混写；历史字段只读迁移且**不可用于准入**（`reconstruction/scale_units.py`）。
30. **置信度不得靠常量或放宽门槛提升**：`scale_confidence` 由「已触发锚点 × 锚点冲突 ×
    经验校准覆盖率 × 逐题型授权」共同派生；未标定/口径异常/锚点冲突/非有限值一律 `low`
    （`reconstruction/scale_assessment.py` + Schema 数据层兜底）。
31. **多锚点鲁棒融合**：相机高/地平面 + 门/桌/椅等标准物体锚点；每锚点保存来源/估计/
    不确定性/残差/接受状态；log 空间共识窗口 + Huber 融合 + **显式冲突检测**；物体尺寸先验
    一律 `TODO_CALIBRATE`。
32. **ARKitScenes 标定严格排除评测重叠场景**：标定集、conformal 校准集与 VSI-Bench 使用的
    150 个 ARKitScenes scene 按原始 scene/video ID 求差并持久化审计清单；**交集非空即 hard fail**；
    测试时只读冻结校准器，不读 GT 位姿/深度/逐场景尺度。
33. **尺度按题型授权**：`allowed_metric_tasks` 决定哪些米制 Tool/Skill 可用；
    `low` **只收回米制工具**，不得把可用的非米制 `depth/poses/point_cloud/objects` 一并降级。
34. **Experiment Readiness Gate**：`implemented ⊂ connected ⊂ real_poc_verified ⊂ paper_eligible`
    四级单调布尔；任一为否不得写进论文主表（`readiness/manifest.py`，`--check` 强制）。

## 复现与版本锁定（§7 / §13.6 / §16.4）

`RunManifest` 记录 `code_commit / docker_digest / checkpoint_sha256 / pip_freeze_hash /
config_hash`，以及 v3 要求的**全部影响结果的推理参数**：`vllm_model / n_frames / max_pixels /
max_model_len / vllm_endpoints / frame_set_hash / ba_enabled / scale_source`。
`EvaluationRun` 带 `code_commit` 与 `active_snapshot_ref`。
`--deterministic-replay` 下时间戳/耗时/id 取确定性占位，保证同 seed 字节级一致。

```bash
git rev-parse HEAD > code.commit
docker image inspect --format '{{.Id}}' skill3d-sandbox:latest > docker.digest
pip freeze > requirements.lock
```

## 待办（按文档标注）

- `TODO_USER_INPUT`：GPT-6 endpoint/model_id/auth；VSI-Bench 原始视频许可；VGGT-1B-Commercial
  checkpoint；SAM2 checkpoint；是否用 Qwen3-VL Thinking 变体；
  **ARKitScenes 非重叠场景的 GT 位姿与原始 scene/video ID 映射**（缺它则尺度永远 low：
  无冻结校准器 → HC30 一律 low → 三个米制题型逐题收回）。
- `TODO_CALIBRATE`：`configs/admission_thresholds.yaml` 全部阈值、G1–G11 门禁阈值、
  M2 双判据（≤10 / 中位×0.35）、M5 去重阈值（深度中位×0.15）、M8 `max_pixels=131072`、
  D-3 回灌修复率 ≥50%、检索 top-k、`N_min`、split 比例、沙箱超时；
  **v4 新增**：`scale.confidence_level`（默认 0.90）、锚点冲突阈值（1.5）、离群残差门（0.4）、
  冲突权重占比（0.4）、有效锚点最小数（medium 2 / high 3）、CI 上限（0.40）、
  经验覆盖容差（0.10）、物体尺寸先验分布（门/桌/椅）、逐题型授权门槛。
- **PoC 未跑（[Conditional Go]，不得写成已验证结论）**：BA route（§10.1）、
  v4 尺度阶梯 L2/L3（§10.2，需 ARKitScenes 标定数据）、D-3 回灌恢复率（§10.3）。
- **v4 改动后尚未在真实 GPU 上重跑端到端**（本机 CPU 已用真实 artifact 副本验证
  `SCALE_ESTIMATE→SCALE_CALIBRATE` 两态与逐题授权）：需分卡部署 vLLM 与 VGGT 复跑 1 个
  episode，重点核对 `scale_ci_rel/scale_ci_abs_m` 自洽、`allowed_metric_tasks` 落盘、
  米制 Tool 的逐题门控、`尺度能力` 小节与 RunManifest 的三个尺度哈希字段。
- 论文消融（C/E/G 三表）的**结果列**需在真实面板上填充；机制开关与表格骨架已就位。
  写主表前先过 `python -m skill3d.readiness.manifest --check <capability>`（HC34）。
- `configs/vsi_bench_split.yaml` 由 `scripts/build_splits.py` 生成，请勿手工编辑；
  重切分（改 seed/比例）会改变 `split_version`，需同步重跑实验并记录进 RunManifest。
