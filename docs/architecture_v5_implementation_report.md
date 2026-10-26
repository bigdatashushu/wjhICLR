# Skill3D v5 实施报告（architecture_v5_implementation_report.md）

> 生成日期：2026-09-20
> 目标规格：《系统架构5.md》v5.0（唯一目标态）
> 基线：`0d0660db7c3042d91878911740842b6cfb0ebe0e`（v4 代码，558 passed / 4 skipped）
> 本文档逐条映射 HC1–39 与 §14.1 实施顺序，并**如实标注未完成项与 blocker**。
> 纪律：本报告不含任何"Conditional Go / 待实码核验"内容的既成事实化表述；未跑通的一律写明。

---

## 1. 本次实施结论（TL;DR）

| 项 | 结果 |
|---|---|
| 全量 CPU 测试 | **622 passed / 4 skipped**（v4 基线 558/4 → 净增 64 项测试，0 失败、0 删除测试） |
| 静态检查 | `python -m pyflakes src`：**undefined-name = 0**；仅 2 条 f-string 提示（v4 遗留，非缺陷） |
| Schema 5.0 与 legacy 隔离 | **已实现并接线**（运行时唯一 artifact 加载入口走 legacy 门） |
| 活动质量指标 / G5 三态 / G8 退役 | **已实现**（G5 条件入分母、G8 字段拒绝、`coverage_ok` 常量删除） |
| 官方 VGGSfM BA | **已退出生产**（`rejected_on_24g_oom`，生产启用请求报 `UnsupportedConfigurationError`） |
| v5 golden | **已重生成**（`v5-golden-1`），旧 golden 只读归档并标 `incomparable_with_v5=true` |
| 真实 GPU 端到端（无 BA） | **已完成**：1 个真实 episode 全链 M1–M13，13 项 v5 核对全 PASS（`data/v5_smoke/SMOKE_RECEIPT.json`） |
| 分层子集真实结果 | **部分完成**：5/8 题型 × 4 episodes 产出真实 per-task 数字（管道级证据，非主表；见 §6） |
| `vggt_sparse_ba` | L0 合同/合成测试完成（21 项）；前端+后端已落码；**L1 单 episode 实测 `rejected_on_l1_gate`**（LightGlue 断言失败）→ 按 §10.1 止损，生产接线保持关闭 |
| 尺度 multi-anchor L2/L3 | **blocker**：缺非重叠场景米制 GT 位姿（`TODO_USER_INPUT`；v5.1 起改用评测同源数据集：优先 ScanNet++，其次 ScanNet）→ 无冻结校准器 → `scale_confidence` 恒 `low` |
| `paper_eligible` | **无任何能力达标**（HC34：需数据隔离 + ≥3 seed + 统计门 + 非 mock 证据） |

> **并发实施说明**：本次实施由两个工作流并行推进 —— 本报告作者负责 §14.1-1~5 与真实 smoke（schema/legacy/质量口径/主线清理/golden）、尺度 L0/L1 与全量验证；`reconstruction/sparse_ba/{features,matching,pycolmap_backend,runner}.py` 与分层子集跑批由并行工作流落码。两者接口一致（`SparseBAReceipt` / `BAOutcome` / `l1_gate` / `sparse_ba_enabled`），全量测试合并后仍为 **622 passed / 4 skipped**。

---

## 2. §14.1 强制实施顺序：逐步状态

| 步 | 要求 | 状态 | 证据 |
|---|---|---|---|
| 1 | 建立安全基线 | ✅ | `data/v5_baseline/{baseline.txt,git_status.txt,pip_freeze.txt,pyflakes_src.txt}`；基线 558/4 复现 |
| 2 | Schema 5.0 与 legacy 隔离 | ✅ | `schemas/reconstruction.py`、`schemas/legacy.py`、`schemas/sparse_ba.py`、`legacy/readers.py`；`tests/unit/test_v5_schema_and_legacy.py`（22 项） |
| 3 | 质量聚合与路由 | ✅ | `reconstruction_gate/quality_metrics.py`（活动集合常量 + `g5_is_computed`）、`scene_state.py`（`coverage_gate_status`/`plane_quality_ok`）、`runner.py`（删 `coverage_ok=True`） |
| 4 | 正式主线清理 | ✅ | `legacy_vggsfm_ba/`（只读历史码 + 生产 hard-disable）、`sparse_ba/`（pair graph/tracks/receipts）、`RunManifest` v5 字段、`data/readiness_manifest.json` 重写 |
| 5 | 重生成 v5 golden | ✅ | `tests/golden/v5/{make_golden.py,data/}`（`golden_version=v5-golden-1`）；旧 golden → `tests/golden/archive_v4/` + `ARCHIVE.json` |
| 6 | 真实 GPU smoke（无 BA） | ✅ | `data/v5_smoke/SMOKE_RECEIPT.json`（13 项全 PASS），详见 §5 |
| 7 | 稀疏 BA 限定 PoC | ⏸ **L0 通过；L1 实测被拒（止损）** | `reconstruction/sparse_ba/`、`tests/unit/test_sparse_ba_l0.py`（21 项）、`data/v5_sparse_ba/41069043_sparse_ba_receipt.json` |
| 8 | 尺度 L0–L3 | ⏸ **L0/L1 完成**；L2/L3 缺 GT 输入（blocker） | `tests/unit/test_scale_v5.py`（9 项）、`scale_poc.py` |
| 9 | 系统级实验 | ⏸ **入口接通 + 子集试跑**；多 seed 主表未跑（算力/时间） | `scripts/v5_subset_sweep.sh`、`evaluation/main_table.py`、`data/v5_subset/` |

---

## 3. HC1–39 逐条映射

> 状态含义：`✅ 落实` = 代码 + 测试；`✅ 护栏` = 负向测试保证不得越界；`⏸ 部分` = 见 §7 blocker。

| HC | 要求要点 | 落点 | 状态 |
|---|---|---|---|
| 1 | 在线链绝对无 GPT-6 | `online/*`、`synthesis/vllm_client.py`（本地 FP8） | ✅ 落实（真实 smoke 用本地 vLLM） |
| 2 | GPT-6 仅离线 | `governance/`、`evolution/`（在线不引用） | ✅ 落实（在线 import 图无 GPT-6） |
| 3 | 阈值标 `[TODO_CALIBRATE]` | 全量阈值常量均带标注 | ✅ 落实 |
| 4 | GPT-6 配置 `[TODO_USER_INPUT]` | `configs/`、报告 | ✅ 落实（未虚构 endpoint/model_id） |
| 5 | Skill3D 真实 repo `TODO_USER_INPUT_SKILL3D_REPO` | 目录树 | ✅ 落实 |
| 6 | 链接仅来自已核验报告 | 文档 | ✅ 落实 |
| 7 | 不编造 API/版本 | 新代码只用已核验 API（pycolmap 4.2 / lightglue / vllm 0.19.1） | ✅ 落实 |
| 8 | 显存数字必有公式 | 报告/常量注释 | ✅ 落实 |
| 9 | Final test 完全隔离 | `run.py`/`eval.py` 对 `final_test` 直接拒绝（`--allow-final-test` 才放行在线盲评） | ✅ 护栏 |
| 10 | Inner 可迭代 / Outer 一次 | `evolution/optimize_loop.py`、`fsm/offline_fsm.py` | ✅ 落实（v4 起） |
| 11 | 候选不可变 + parent_version | `schemas/evolution.py`、`skills/` | ✅ 落实（v4 起） |
| 12 | promote 原子切换可回滚 | `skills/registry.py`、active snapshot | ✅ 落实（v4 起） |
| 13 | 硬测试一票否决 | `evolution/` 门禁、几何校验 | ✅ 落实 |
| 14 | Tool 是预封装稳定函数 | `tools/registry.py`、`tools/contract.py` | ✅ 落实 |
| 15 | Skill 是题型级模板 | `schemas/skill.py`、`skills/` | ✅ 落实 |
| 16 | 重建先于 Skill 路由 | `fsm/online_fsm.py`（RECONSTRUCT/QUALITY_GATE 早于 CLASSIFY/RETRIEVE） | ✅ 落实 |
| 17 | Tool 只经 SceneState 句柄访问 | `tools/scene_handle.py` | ✅ 落实 |
| 18 | paired A/B 复用同一 artifact | `skills/paired_ab.py`（`assert_same_reconstruction_artifact`）；**新增** P2 复用 P1 落盘 artifact | ✅ 落实 + **本次修复接线缺口** |
| 19 | 防泄漏（无答案/ID、scene 分层） | `adapters/split_builder.py`、`memory/leakage.py` | ✅ 落实 |
| 20 | 文档纯文本 | 本文档 | ✅ 落实 |
| 21 | 统一固定 32 帧 FrameSet | `adapters/frame_set.py`；smoke 实测 32 帧 + hash 一致 | ✅ 落实（真实链核对） |
| 22 | `quality_status` 唯一事实源，fail-closed | `scene_state.route_from_quality`（NaN/未计算 → 不得 `full_3d`） | ✅ 落实 + 护栏 |
| 23 | Tool 执行期 fail-closed | `tools/contract.py`、`sandbox/kernel.py`（抛错不返回 False） | ✅ 落实 |
| 24 | 在线链无 GPT-6（重申）| 同 HC1 | ✅ 落实 |
| 25 | vLLM 与 VGGT/SAM2 分卡 | smoke：vLLM@GPU4、VGGT+SAM2@GPU5 | ✅ 落实（真实运行） |
| 26 | M8 必须多模态 32 帧 | `prompt_builder.build_image_messages` + trace `n_images_to_synthesizer=32` | ✅ 落实 + 实测核对 |
| 27 | MRA 严格官方口径 | `evaluation/mra.py`（`rel<=1-θ`，θ=linspace(0.5,0.95,10)，分母 gt） | ✅ 落实 |
| 28 | 已修项即硬约束/契约 | §11 陷阱全部有对应代码/注释 | ✅ 落实 |
| 29 | 尺度 CI 统一口径 | `scale_units.ci_abs_m/check_ci_rel/ci_consistency_status` | ✅ 落实 + L0 测试 |
| 30 | 置信度不得靠常量提升 | `scale_assessment.grade_confidence_v4` + Schema 兜底（无 `calibration_id` → low） | ✅ 护栏 |
| 31 | 多锚点鲁棒融合 + 冲突显式 | `metric_scale.fuse_scale_anchors_robust`；**本次修正**：`scale_conflict` 只表示锚点冲突（HC31 语义），降级改写进 `scale_method` | ✅ 落实 + 修正 |
| 32 | 标定严格排除评测重叠 scene | `scale_calibration.SceneIdAudit`（三向互斥 + 哈希留档）；**v5.1 修订**：标定池按被评测数据集选择同源非重叠场景（不再绑定 ARKitScenes），排除集参数化且规模校验，多校准器 `<dir>/<dataset>.json` 按数据集选用 | ✅ 落实 + v5.1 修订（见 `docs/decisions/D-2026-09-20-calibration-dataset-policy.md`） |
| 33 | 尺度按题型授权，不摧毁 3D | `metric_tasks_after_quality_gates`；smoke 实测非米制 3D Tool 仍可用 | ✅ 落实 + 实测 |
| 34 | Experiment Readiness Gate | `readiness/manifest.py` + `data/readiness_manifest.json`（四项单调布尔） | ✅ 落实 |
| 35 | 正式主线不依赖 BA | `vggt_runner` 固定 feed-forward；`legacy_vggsfm_ba` 生产 hard-disable | ✅ 落实 + 护栏 |
| 36 | 轻量稀疏 BA 仅一次止损 PoC | `reconstruction/sparse_ba/`（默认关闭、启用即报错）+ L0 测试；L1 失败即 `rejected` 并保持生产关闭 | ⏸ 部分（L0 通过 / L1 止损） |
| 37 | 无真 BA 时 G5 明确不可用 | Schema 校验器（非 `computed` 有值即 hard fail）+ `reprojection_status`；smoke 实测 `not_available`+`None` | ✅ 落实 + 护栏 |
| 38 | G8 永久退役、不设替代门 | `QualityMetrics` 拒绝 `g8_bbox_coverage_min`；`SceneState.coverage_gate_status` 只能是 `not_defined`；删除 `coverage_ok=True` | ✅ 落实 + 负向测试 |
| 39 | v5 与历史数据版本隔离 | `legacy/readers`（唯一旧数据入口 + hard fail）、`evaluation/golden_v5.py`（版本三元组）、旧 golden 归档 | ✅ 落实 + 护栏 |

---

## 4. 关键代码落点（本次新增/修改）

**新增**

| 文件 | 作用 |
|---|---|
| `src/skill3d/legacy/readers.py` | 旧产物唯一读取入口：`read_legacy_artifact` / `load_artifact_v5`（版本不符即 hard fail）/ `audit_tree` |
| `src/skill3d/schemas/legacy.py` | `LegacyArtifact`（`eligible_for_runtime/statistics` 恒 False）+ `LegacyArtifactError` |
| `src/skill3d/schemas/sparse_ba.py` | `SparseBAReceipt`（§4.5 全字段 + L2 报告字段 + 止损码） |
| `src/skill3d/reconstruction/sparse_ba/{__init__,pair_graph,tracks,receipts}.py` | 限定 PoC：pair graph / track merge / 回执与 L1-L2 门槛 / `BAOutcome` 统一载体 |
| `src/skill3d/reconstruction/legacy_vggsfm_ba/{__init__,route}.py` | 已否决的官方 BA：只读历史码 + `assert_official_ba_disabled` 生产护栏 |
| `src/skill3d/evaluation/golden_v5.py` | golden 版本注册、环境绑定校验、旧 golden 混用 hard fail |
| `tests/golden/v5/{make_golden.py,data/}` | v5 golden（`v5-golden-1`，无 G8、G5=None 不入分母） |
| `tests/golden/archive_v4/` | 旧 golden 只读归档 + `ARCHIVE.json`（`incomparable_with_v5=true`） |
| `tests/unit/test_v5_schema_and_legacy.py` | v5 Schema/legacy 负向测试（22 项） |
| `tests/unit/test_sparse_ba_l0.py` | 稀疏 BA L0 合同（21 项） |
| `tests/unit/test_scale_v5.py` | 尺度口径 L0/L1（9 项，含真实 smoke 回归） |
| `scripts/v5_subset_sweep.sh` | 8 题型分层子集真实链跑批 |

**修改（要点）**

- `schemas/reconstruction.py`：加 `schema_version/quality_metric_version/reprojection_status/sparse_ba_receipt_ref`；`recon_method` 词汇表改 `vggt|vggt_sparse_ba|dust3r_mast3r|colmap`；删 `scale_ci`/`g8_bbox_coverage_min`/`CoverageMap`；加 legacy 字段拒绝 + G5 可用性校验器；`SceneState` 加 `coverage_gate_status`/`scale_conflict` 并删 legacy `scale_ci`。
- `reconstruction_gate/quality_metrics.py`：`ALWAYS_ACTIVE_METRICS`/`CONDITIONAL_METRICS`/`RETIRED_METRICS` + `g5_is_computed`；`_norm_scores` 只在 G5 computed 时入分母；G5 缺失归一为 `None`；删 `bbox_coverage` 兼容参数。
- `reconstruction_gate/scene_state.py`：`scale_is_usable` 去掉旧绝对 CI 回退；`question_gate(plane_quality_ok=...)` 三态 fail-closed（`None`/`False` 均不授权）。
- `online/runner.py`：删 `coverage_ok=True`；新增 trace 层 v5 事实（`n_images_to_synthesizer`/`reprojection_status`/`coverage_gate_status`/尺度授权）；**P2 复用 P1 落盘 artifact**（方案 X，零重算）；artifact 加载统一走 `load_artifact_v5`。
- `reconstruction/scale_assessment.py` + `scale_units.py`：新增 `ci_consistency_status`，"有点估计但无区间"= 未标定 → 降级 low + 清空授权（**不再让 P1 崩掉**，真实 smoke 中修出的缺陷）。
- `schemas/trace.py`：`RunManifest` 增 v5 字段；`EpisodeTrace` 增逐 episode v5 事实。
- `readiness/manifest.py` + `data/readiness_manifest.json`：能力表改为 v5（新增 `official_vggsfm_ba`/`vggt_sparse_ba`/`v5_schema_and_legacy_isolation`/`v5_golden`/`g5_reprojection_semantics`）。

---

## 5. 真实 GPU smoke 回执（§14.1-6）

`data/v5_smoke/SMOKE_RECEIPT.json`（13 项核对全 PASS）：

| 核对项 | 实测 |
|---|---|
| schema / 质量口径版本 | `5.0` / `v5-no-g8-g5-optional`（artifact 与 trace 一致） |
| quality | `status=computed`，`overall_quality=0.9792`，等于版本化聚合器重算值 |
| G5 / 重投影 | `reprojection_status=not_available`，`g5_*=None`，不参与 `overall_quality` |
| G8 | artifact 无任何 `g8*` 字段 |
| coverage 门 | `coverage_gate_status=not_defined`（不写 passed） |
| FrameSet | 32 帧、`frame_set_hash=02599dcb…`，M1–M8 全链同源 |
| M8 多模态 | `n_images_to_synthesizer=32`（未丢帧） |
| 尺度 | `low` + `allowed_metric_tasks=[]` + `authorized=[]`（逐题收回），锚点 1 个、冲突 0 |
| 非米制 3D 能力 | 仍可用（`euclidean_distance`/`object_centroid`/`relative_direction`/`reproject` 等） |
| BA | `ba_enabled/sparse_ba_enabled/official_vggsfm_ba_enabled` 全 False |
| route | `full_3d`（质量 computed 且有限） |
| 端到端 | M1–M13 全走通，`object_counting` 预测 2 vs GT 2 → MRA=1.0，`coverage=1.0`、`refusal=0`、`tool_contract=0` |
| 部署 | vLLM FP8@GPU4（20.6 GiB）与 VGGT+SAM2@GPU5 分卡（HC25） |

> **不声称**：单 episode 只证明链路可跑，不是精度结论，不进主表（HC34）。

---

## 6. 分层子集真实结果（管道级证据，非论文主表）

`scripts/v5_subset_sweep.sh`：`split=inner_validation`、每题型 `limit=4`、单 seed，真实视频 + VGGT + Qwen3-VL-8B-FP8 + SAM2，**不启用任何 BA**。

**已产出的 per-task 数字（5/8 题型完成；其余 3 题型的跑批被中断，未产出 → 不臆造）**：

| 题型 | 类别 | n | 指标 | 值 |
|---|---|---|---|---|
| `object_counting` | NA(MRA) | 4 | MRA | **0.525** |
| `object_size_estimation` | NA(MRA) | 4 | MRA | **0.275** |
| `room_size_estimation` | NA(MRA) | 4 | MRA | **0.075** |
| `object_abs_distance` | NA(MRA) | 4 | MRA | **0.000** |
| `object_rel_distance` | MCA(Acc) | 4 | Accuracy | **0.000** |
| `object_rel_direction` / `route_planning` / `obj_appearance_order` | MCA(Acc) | — | — | 未产出 |

**过程指标（20 个 episode）**：`abstained=0`、`tool_contract_hits=1`、`n_images_to_synthesizer=32`（全部）、`scale_confidence=low`（全部）、`authorized_metric_tasks=∅`（全部）；route 分布含 `full_3d` 与 `fallback_2d_only`。

**如何解读（不得越读）**：

- 这是"指标口径与过程指标可产出"的**管道级证据**：主表 8 任务平均规则（§8.3）、MRA 严格口径（§8.2）、逐题米制授权与 route fail-closed 都在真实数据上按规格生效。
- **三个米制题型 MRA 极低（0.000–0.275）是当前规格的必然结果**：无冻结校准器 → `scale_confidence=low` → `object_abs_distance/size/room` 的米制 Tool 被逐题收回（HC30/33）。这不是缺陷，也不得通过放宽门槛"修好"；恢复它们需要 §10.2 的标定数据（blocker）。
- `object_rel_distance`（不需要米制尺度）Acc=0/4 说明 **MCA 程序合成质量才是当前主要短板**，与尺度 blocker 无关 —— 这是论文改进的第一优先级方向。
- 样本量 n=4/题型、单 seed → **`real_poc_verified` 仍为 false**，不得写入主表（HC34）。

完整产物：`data/v5_subset/recon/vggt/`、`data/v5_subset/traces/`（含 `evaluation_run.jsonl` / `episode_trace.jsonl` / `scale_report.json`）、`data/v5_subset/run_manifest_<task>.json`、`data/v5_subset/sweep.log`。

## 7. Readiness（`data/readiness_manifest.json`）

| 能力 | implemented | connected | real_poc_verified | paper_eligible | blocker |
|---|---|---|---|---|---|
| `vggt_feed_forward_mainline` | ✅ | ✅ | ❌ | ❌ | 需 ≥3 seed 真实端到端 + 与基线统计比较 |
| `official_vggsfm_ba` | ✅ | ❌ | ❌ | ❌ | `rejected_on_24g_oom`（HC35，production hard-disabled） |
| `vggt_sparse_ba` | ✅ | ❌ | ❌ | ❌ | L1 单 episode 实测 `rejected_on_l1_gate`（LightGlue 断言失败，`skip_reason=matching: …`）；按 §10.1 止损 → 生产接线关闭。**不是** `rejected_on_24g_oom`（显存不是本次失败原因） |
| `v5_schema_and_legacy_isolation` | ✅ | ✅ | ❌ | ❌ | 需真实产物按 v5 Schema 复核（smoke 已完成 1 例，待多场景） |
| `v5_golden` | ✅ | ✅ | ❌ | ❌ | 已生成；需与真实统计跑通对比后才谈验证 |
| `g5_reprojection_semantics` | ✅ | ✅ | ❌ | ❌ | 主线固定 `not_available`；待 `vggt_sparse_ba` PoC 才可能 computed |
| `scale_recovery` | ✅ | ✅ | ❌ | ❌ | `TODO_USER_INPUT`：ARKitScenes 非重叠 GT 位姿 → 无冻结校准器 |
| `metric_tasks_authorization` | ✅ | ✅ | ❌ | ❌ | 上层（scale_recovery）未验证 → 恒收回米制题型 |
| `tool_contract_replay` | ✅ | ❌ | ❌ | ❌ | 回灌修复率无真实数字（门槛 ≥50%），默认关闭 |
| `dust3r_mast3r_baseline` | ❌ | ❌ | ❌ | ❌ | 桩：未接真实权重 |
| `gsplat_densification` | ❌ | ❌ | ❌ | ❌ | 桩：恒 `None` |
| `docker_sandbox` | ❌ | ❌ | ❌ | ❌ | 模板未接线（M10 走 in-process kernel） |
| `multi_seed_main_table` | ✅ | ✅ | ❌ | ❌ | 待真实 ≥3 seed 结果 |
| `gpt6_offline_governance` | ✅ | ❌ | ❌ | ❌ | `TODO_USER_INPUT`：GPT-6 endpoint |

---

## 8. 生产调用图（无调用者即 `connected=false`）

| 能力 | 非测试生产调用者 |
|---|---|
| v5 artifact 加载门 `load_artifact_v5` | `online/runner.py`（M3 RECONSTRUCT 复用分支）✅ |
| 官方 BA 护栏 `assert_official_ba_disabled` | `reconstruction/vggt_runner.py`、`reconstruction/run.py` ✅ |
| `sparse_ba_enabled()` / `BAOutcome` | `reconstruction/vggt_runner.py` ✅（默认 False → 不接线） |
| `SparseBAReceipt` / `receipts.l1_gate/l2_gate` | 仅测试与 PoC 工具 → **`connected=false`**（符合 HC36 未过 PoC 前不得接线） |
| `legacy.readers.read_legacy_artifact/audit_tree` | 审计脚本用途，无生产调用者 → 只读工具（不影响运行时） |
| `golden_v5.golden_versions` | `online/eval.py`（写 RunManifest）✅ |
| `golden_v5.load_golden_stats` | `tests/golden/v5/test_golden_paired_ab_v5.py`（统计门）→ 测试侧 |
| `quality_metrics` 活动集合常量 | `quality_metrics` 自身聚合 + `runner.py` trace 落版本 ✅ |

**本次修复的两个真实接线缺陷**（都在真实 GPU 运行中暴露）：

1. **未标定尺度导致 P1 直接失败**：`metric_scale` 有点估计、`scale_ci_rel=None`（无冻结校准器）时 `assert_ci_consistent` 抛 `CiUnitError`，整个 scene 重建报废。v5 正确行为是 fail-closed 为 `low` + 清空授权、产物照常落盘（HC30）。
2. **P2 不复用 P1 的 artifact**：`mode=real` 下即使 P1 已落盘 artifact（含实算质量），P2 仍无条件重跑 VGGT——违反 §2.2/方案 X"P2 加载即得实算值、零重算"，也使 HC18（同源 artifact）在常规流程里失去保障。已改为优先复用 + 帧集哈希校验（不符则重算并留痕）。
3. **`--question-types` 在 vsi_bench 源被静默忽略**（`load_vsi_bench_items` 无该参数、`eval.py` 只对合成源生效）→ 分层子集实验会误跑成全量。已修复为在抽帧前按规范题型过滤。

---

## 9. 未完成项 / Blocker（含影响与下一输入）

| 项 | 影响 | 下一输入 |
|---|---|---|
| `vggt_sparse_ba` L1 复跑（§10.1） | 无法报告 G5、无法进可选消融行；主线不受影响（HC37 允许 `not_available`） | 先修前端断言失败：LightGlue 在 pair=(0,1) 触发断言（`kp=(1,354,2)/d=(1,354,256)`，形状一致 → 疑似 `requires_grad`/dtype/`n` 维度约定）；修好后重跑 L1，仍未过则按 §10.1 一次性止损关闭，不得反复扩大范围 |
| 尺度 L2/L3 | `scale_confidence` 恒 `low`，三个米制题型全部收回 → Measurement 4 题只能靠非米制替代路径 | `TODO_USER_INPUT`：ARKitScenes 非重叠场景 GT 位姿 + scene/video ID 映射（并须审计排除 VSI-Bench 150 scene） |
| 多 seed 主表（≥3 seed） | 无法给出 mean±std 主表；`multi_seed_main_table` 仍 `real_poc_verified=false` | 算力：全量 5130 QA 单遍成本 ≈ vLLM 14–28 h（单卡）/2–4 h（DP=8）+ 重建 288 scene ≈ 5.3 h（仅一次，跨 seed 复用；P2 已修好复用 P1 artifact） |
| 分层子集跑批完成 8/8 题型 | 当前只有 5 题型数字；3 题型（`object_rel_direction`/`route_planning`/`obj_appearance_order`）无数据 | 续跑 `N=4 bash scripts/v5_subset_sweep.sh`（断点恢复幂等，已重建 scene 会 skip） |
| D-3 回灌恢复层真实修复率 | 默认关闭（正确性不依赖它） | 真实 8B 上的回灌修复率实测（门槛 ≥50%） |
| GPT-6 离线治理 | 离线演进真实模式只能 QUARANTINE | `TODO_USER_INPUT`：GPT-6 endpoint/model_id/auth |
| Docker 沙箱 / LanceDB 分支 / 监控回滚 | 未接线；M10 走 in-process kernel | 按 v5 §11 描述逐项接线（不阻塞主线） |

---

## 10. 复现命令（本次实际执行）

```bash
# 环境
source /home/cvailab/anaconda3/etc/profile.d/conda.sh && conda activate skill3d-exp
export PYTHONPATH=third_party/vggt:src

# 1) 全量 CPU 测试 + 静态检查
python -m pytest -q -p no:cacheprovider            # 622 passed / 4 skipped
python -m pyflakes src | grep -c "undefined name"  # 0

# 2) v5 golden 重生成（一次性；产物入库）
python tests/golden/v5/make_golden.py

# 3) 真实服务（分卡：vLLM@GPU4，重建/SAM2@GPU5；HC25）
CUDA_VISIBLE_DEVICES=4 bash scripts/serve_qwen3vl_dp8.sh        # FP8, port 8100

# 4) P1 重建（含 M4 质量随 artifact 落盘，方案 X）
CUDA_VISIBLE_DEVICES=5 python -m skill3d.reconstruction.run \
  --split inner_validation --method vggt --limit 1 --gpus 0 \
  --recon-dir data/v5_smoke/recon

# 5) P2 在线链（M1–M13；复用 P1 artifact，零重算）
CUDA_VISIBLE_DEVICES=5 python -m skill3d.online.eval \
  --mode real --source vsi_bench --split inner_validation --limit 1 \
  --recon-dir data/v5_smoke/recon --recon-method vggt \
  --vllm-endpoint http://127.0.0.1:8100 --vllm-model qwen3vl-8b-r0 \
  --trace-dir data/v5_smoke/traces --run-manifest data/v5_smoke/run_manifest.json

# 6) 分层子集（8 题型 × N，真实链）
N=2 bash scripts/v5_subset_sweep.sh

# 7) Readiness 快照
python -m skill3d.readiness.manifest --out data/readiness_manifest.json
```

---

## 11. 完成定义（§14.3）自检

| DoD 条款 | 状态 |
|---|---|
| 全量 CPU 测试与静态检查通过，且没有通过删除测试/放宽断言获得假绿 | ✅ 622/4（含并行工作流新增模块合并后复跑），pyflakes 0；测试净增 64 |
| 默认配置在 8×4090 不触发官方 BA | ✅ 默认关 + CLI/驱动层双重 hard fail |
| 正式主线真实 GPU smoke 完成 | ✅ `data/v5_smoke/SMOKE_RECEIPT.json` |
| v5 artifact 不含 legacy G8/旧尺度字段 | ✅ 负向测试 + 真实产物核对 |
| G5 unavailable 为 None 且不参与 overall_quality | ✅ Schema 校验器 + 真实产物核对 |
| 所有版本字段落盘 | ✅ artifact / trace / RunManifest 三处 |
| `coverage_ok=True`、活动 G8、旧 golden 混用、代理 G5、官方 BA 生产启用均有负向测试并 fail-closed | ✅ 5 类负向测试齐备 |
| 只有证据完整的能力才更新 Readiness | ✅ 本报告未提升任何 `real_poc_verified`/`paper_eligible` |

---

## 12. 2026-09-21 追加：MCA 程序路径缺陷修复与真实复测

> 本节是 v5 实施报告的**追加章节**，记录本轮会话的全部改动、真实 GPU 数字与**负面结果**。
> 决策记录：`docs/decisions/D-2026-09-21-mca-program-path-fixes.md`；面向 v6 的问题清单：`../问题报告v5.md`。

### 12.1 逐题失败归因（修复前 `data/v5_scoped/full/c1/`，outer_holdout 32 题）

| 现象 | 题数 | 根因（代码级） | 证据 |
|---|---|---|---|
| episode 记 `unavailable`（`n_images_to_synthesizer=0`） | **9/32** | M8 输出退化/截断 → 代码块**未闭合** → 旧解析器整段放弃 | `data/v5_scoped/diag/m8_raw/`（含 20311 字节的重复段落样本） |
| prompt 里只剩 1 个 Tool → 模型 abstain | **12/32** | 未授权米制题型被降级为 `route=fallback_2d_only` → 产物集塌成 `{frames,intrinsics}` | `full/c1/episode_program.jsonl` 的 `m8_prompt` |
| 对象清单残缺（同 scene 只剩 2 个对象） | 全部对象类题 | 检测器服务故障**静默**返回 `[]`；清单逐题重算且与问题耦合 | `full/c1/program_trace.jsonl`（`list_objects()`→`["obj_0","obj_1"]`） |
| 方向题 4/4 全错 | 4 | 竖直轴用"包围盒最小 extent"猜（选中水平轴 z），左右判据硬编码手性 | 场景 `acd95847c5`：extent 2.282/1.865/**1.818**，相机导出的竖直轴是 y |

### 12.2 修复清单（全部带回归测试，CPU 测试 642 → 671 passed / 4 skipped，pyflakes undefined-name = 0）

1. `synthesis/program_assembler.py`：新增"未闭合代码块 → **可解析前缀**"回退（不补全、不臆造；拿不到可解析代码仍抛错）。
2. `synthesis/prompt_builder.py`（模板 `program_synth_v2`）：明确 `ReturnAnswer` **只记录不中止**；MCA 先映射选项文本再返回字母；不得假设存在 `objects` 等隐藏全局变量。
3. `reconstruction_gate/scene_state.py` + `online/runner.py`：未授权米制题型**只收回米制 Tool、不改变 route**（按 §4 M4"v5 尺度能力门控"）。
4. `segmentation/sam2_tracker.py` + `open_vocab_detector.py`：M5 清单改为 **scene 级 + `(scene, frame_set_hash)` 缓存**；检测器故障**出声**并重试；新增**题面点名物体定向补漏**；VLM 框提示加请求级 seed。
5. `tools/scene_handle.py` + `tools/geometry_tools.py`：`up_direction()` 由相机位姿导出（含符号）；`relative_direction` 改为"水平面投影 + `right ⟺ (f×d)·u < 0`"；新增 `relative_direction_of` / `object_visible_frames`；`ObjectInstance` 增 `visible_frames`。
6. `tools/geometry_tools.py`：三个米制 Tool 乘 `metric_scale`（面积按平方），缺系数即 `domain_value` fail-closed——**修掉一个潜伏缺陷**（旧实现返回世界单位；本机实测 `metric_scale=5.233`，面积类偏 ~27 倍）。

### 12.3 真实 GPU 复测（outer_holdout / inner_validation，32 题，seed 0，单 endpoint）

列序 = `Avg | 计数 | 绝对距离 | 尺寸 | 房间 | 相对距离 | 相对方向 | 路线 | 外观`

| split | 版本 | Avg | 计数 | abs | size | room | rel_dist | rel_dir | route | appear |
|---|---|---|---|---|---|---|---|---|---|---|
| outer | C0 直答（留档） | 46.88 | 67.5 | 42.5 | 72.5 | 67.5 | 25 | 25 | 50 | 25 |
| outer | C1 修复前（留档） | 11.56 | 42.5 | 0 | 0 | 0 | 0 | 0 | 25 | 25 |
| outer | v_a | 8.44 | 67.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| outer | v_b | 8.44 | 67.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| outer | v_c | 20.94 | 67.5 | 0 | 0 | 0 | 0 | **100** | 0 | 0 |
| outer | **v_d（当前代码）** | 13.75 | 30 | 0 | 0 | 5 | 0 | 75 | 0 | 0 |
| inner | v_a | 22.81 | 77.5 | 45 | 25 | 10 | 0 | 0 | 0 | 25 |
| inner | **v_d（当前代码）** | 11.56 | 25 | 45 | 0 | 22.5 | 0 | 0 | 0 | 0 |

产物：`data/v5_scoped/full/c1_fix{,2,3,4}/`、`data/v5_scoped/internal/c1_fix{,2}/`、`logs/full_c1_fix{1,2,3,4}.out`、`logs/internal_c1_fix{1,2}.out`。

### 12.4 可以说的与不能说的

**可以说的（真实、可复现）**：
- `unavailable` 从 9/32 降到 **0/32**（M8 解析回退 + prompt 纪律）；
- M5 清单从"2 个对象"恢复到 **62 个**（含题目点名的 `laptop`/`whiteboard`），**同一 scene 后续 episode 从 76 s 降到 5 s**（缓存复用）；
- `object_rel_direction` 从 0/4 到 **3~4/4**（v_c 4/4、v_d 3/4），C0 直答基线是 1/4——**这是本项目第一次出现"程序路径在某个题型上超过直答"**；
- 计数题 MRA 从 0.425 升到 0.675（v_c，外测）。

**不能说的（负面结果，必须如实写）**：
- **内测没有复现方向题的增益**（两版都是 0/4）：内测的失败是"题目点名的物体没进清单"（`list_objects('trash can')` 为空）与"模型误以为 `ReturnAnswer` 会中止"两类；
- **最新版（v_d）在两个 split 上同时回落**（外测 20.94→13.75、内测 22.81→11.56），落点集中在计数题，方向一致；
- **根因是 M5 清单的召回与精度都不足**：外测 v_d 计数预测 `0,5,3,0` vs GT `2,4,4,2`，`0` = 漏绑、`12`（内测）= 重复实例；
- **v_c 的 20.94 不宜当作目标**：它是"程序更容易崩 → 落到更强的 no-tool CoT 兜底"的产物，恰好说明规格激励（abstain/答错都按错计）与系统实际能力相反；
- **每题型 4 题、单 seed、无噪声底** → 除"两个 split 同向的计数回落"外，表内所有差异都不能当增益结论；
- 也没有 inner 的 C0 基线，无法在 inner 上做 C0/C1 对比。

### 12.5 Readiness 与 blocker 变化

- `real_poc_verified` / `paper_eligible` **仍然全部为 false**（本轮的 32 题、单 seed 不满足 HC34）；
- 新增 blocker（写给 v6，详见 `../问题报告v5.md` P1–P10）：米制未授权题型的合法行为未定义；`route` 同时承担质量分流与工具授权两个语义；**世界坐标系约定（竖直轴/手性）未进 Schema**；M5 清单契约（scene 级/故障语义/召回义务/去重阈值）缺失；M8 输出契约（`ReturnAnswer` 语义、截断回退）缺失；每题型的"最小原语矩阵"与 Tool 准入流程缺失；主表激励与系统能力相反；逐 episode 判读信息与模板版本未落盘。
