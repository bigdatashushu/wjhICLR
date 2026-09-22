# harness3D v6 实施报告（architecture_v6_implementation_report.md）

> 生成日期：2026-09-21
> 目标规格：`/nas/wangjh/harness3d/系统架构v6.md`（v6.0，唯一目标态）
> 基线：tag `系统架构v5`（commit `3b1aa2a`，671 passed / 4 skipped）
> 本文逐条对照 §23.1 的 Phase 0–6 与 DoD，并**如实标注未完成项与 `[待实验]` 项**。
> 纪律：本文不含任何"已验证/效果显著"的既成事实化表述；未跑的一律写明。

---

## 1. 结论（TL;DR）

| 项 | 结果 |
|---|---|
| 全量 CPU 测试 | **726 passed / 3 skipped / 0 failed**（v5 基线 671/4 → 净增 55 项，0 删除、0 放宽断言换假绿） |
| 静态检查 | `python -m pyflakes src` 的 **undefined-name = 0** |
| v6 契约一致性测试 | `tests/unit/test_v6_schema.py`（45 项）：Schema 6.0 / legacy 拒绝 / G5 固定 None / 世界系契约 / 证据门 / 逐 Tool 收窄，全绿 |
| 真实 GPU 链路（M3+M4） | **已跑通**：真实 VGGT → artifact（schema 6.0）→ M4 主门通过（warp 0.668 / overlap 0.551）→ 世界系契约落盘 → EvidenceProfile → 逐题 scope；回执 `data/v6_smoke/SMOKE_RECEIPT_V6.json` |
| 度量尺度融合（MoGe-2） | **未跑**（`[待实验]`）。`metric_scale=None` + `scale_fusion_status="not_run"`；C1–C6 六项验收全部未通过 |
| `paper_eligible` | **无任何能力达标**（需数据隔离 + ≥3 seed + 统计门 + 非 mock 证据） |
| v5 回归测试 | 31 个测试文件归档到 `tests/archive_v5/`（含逐项替代物说明），**未删除** |

---

## 2. Phase 0–6 逐步状态

### Phase 0　基础骨架 + EvidenceProfile ✅

| 要求 | 落点 | 状态 |
|---|---|---|
| FrameSet / SceneState 双路由 | `schemas/reconstruction.py`（`scene_route` × `question_tool_scope`） | ✅ |
| EvidenceProfile 8 项 × 三值 | `schemas/evidence.py`（有序三值 + `CAPABILITIES` 词汇表） | ✅ |
| Tool 注册表 `requires_evidence` 声明 | `schemas/tool.py`（**无默认值**，不声明构造即失败）+ `tools/registry.py` 注册期校验 | ✅ |
| 异常族（含 `ConfidenceGateError` / `AnswerAlreadyGiven`） | `tools/contract.py` §9.12 四类齐全 | ✅ |
| **DoD**：EvidenceProfile 落盘 | `EpisodeTrace.evidence_states` + `TraceRecord.evidence_profile` | ✅ |
| **DoD**：Tool 隐藏按 `requires_evidence` 自动裁剪 | `registry.names_for_scope(scope, evidence_profile=…)` | ✅ |
| **DoD**：属性测试每 Tool × 每能力状态与 `docs()` 一致 | `tests/unit/test_v6_schema.py::test_registry_docs_tracks_evidence_and_scope` 等 | ✅ |
| **DoD**：CPU 测试全绿 | 726 passed / 0 failed | ✅ |

**真实 GPU 实测修正（重要）**：首轮真实数据上 `geometry_3d=degraded`（M4 主门**通过**，但有"旋转跳变 43.3°"诊断告警）导致整个 3D Tool 面被隐藏 —— 与 §6.2"主门告警未崩 → 仍走 full_3d"、§2.3"4 个 MCA 题型不受度量路线影响"冲突。已按 §7.2 的正确口径修正：**除 `metric_scale` 外，其余能力的 `degraded` 一律容忍**（工具暴露 + 答案带 `evidence_degraded:<能力>` 标记）；`metric_scale` 保持 D3 硬契约（只在 `available` 时放行米制 Tool）。修正后真实数据上非米制题保留 **11 个 Tool**（v5 实测同类题只剩 1 个）。

### Phase 1　M4 质量门 + 世界系契约 ✅

| 要求 | 落点 | 状态 |
|---|---|---|
| warp 内点率 + 分组点云重叠率主门 | `reconstruction_gate/m4_main_gate.py` | ✅ |
| track 重投影 / 旋转平滑诊断 | `quality_metrics.track_reprojection_residual` / `rotation_smoothness`（只产告警） | ✅ |
| conf-warp 单调自检 | `m4_main_gate.conf_warp_monotonic`（退化返回 `None` 而非 `False`） | ✅ |
| `world_up` / `handedness` 由 M3 落盘、M4 校验 | `reconstruction_gate/world_frame.py` + `vggt_runner` + artifact Schema | ✅ |
| 方向 Tool 缺约定 fail-closed | `relative_direction_of` → `world_frame.direction_of` 抛 `ValueError` → `DomainValueError` | ✅ |
| G5 永久 `not_available`、G8 退役 | Schema 层**不声明** G5/G8 字段；出现即 hard fail | ✅ |
| **DoD**：主门交叉、多指标不得单挑 | 测试覆盖"warp 通过但 overlap 失败"与反向两例，均判不通过 | ✅ |
| **DoD**：负向测试 = 位姿绕水平轴翻转方向答案必须翻转 | `tests/unit/test_world_frame.py`（42 项）含双向翻转断言 | ✅ |
| **DoD**：注入退化可被主门识别 | `tests/unit/test_synthetic_geometry_gate.py`：打乱位姿 → warp 0.32；逐帧深度缩放 → warp 0.17；半数深度 ×1.5 → overlap 0.0005，均判失败 | ✅ |

**真实 GPU 证据**（`data/v6_smoke/SMOKE_RECEIPT_V6.json`）：scene `41069043`（arkitscenes），
`warp_inlier=0.668`（τ=0.5）、`cloud_overlap=0.551`（τ=0.3）→ **主门通过** → `scene_route=full_3d`；
`world_frame_status=degraded`、`world_up=[-0.169,-0.110,-0.979]`、`handedness=right` 落盘。

> 阈值 τ_warp / τ_cloud 仍是**起始参考值**，必须按 §10.6 的最小 PoC（1 个好 episode + 抽稀/跨场景帧/运动模糊三种注入退化）在自有数据上标定。

### Phase 2　MoGe-2 度量尺度融合 PoC ⏸ **未跑（`[待实验]`）**

| 要求 | 落点 | 状态 |
|---|---|---|
| 逐帧 `s_k = median(d_metric / d_VGGT_norm)` | `reconstruction/metric_fusion.py::fuse_metric_scale` | ✅ 代码就位 |
| 跨帧 `s_global = median_k(s_k)` + 离散度 | 同上（`scale_self_consistency = std/median`） | ✅ |
| 相机对齐（VGGT K → fov_x 传 MoGe-2） | `fov_x_deg_from_intrinsics` + `make_moge2_model` | ✅ |
| PoC receipt（32 个 s_k、median、MAD、离群帧） | `write_per_frame_receipt` | ✅ |
| C1–C6 六项验收 | `acceptance_report(...)`：`implemented/connected/real_poc_verified/paper_eligible` **全从 false 起步** | ⏸ **未运行** |
| 默认行为 | `metric_model="none"` → `metric_scale=None` + `scale_fusion_status="not_run"` | ✅ |
| Metric3D v2 同场景对照 | 接口预留；**LICENSE 待用户直接读官方仓库核实** | ⏸ `[TODO_LICENSE]` |

**为什么未跑**：MoGe-2 权重未在本机就位（`moge` 包与 `Ruicheng/moge-2-vitl` 未下载），且 §11 全部为 `[待实验]` —— 在自有 PoC 数据出来前，`metric_scale` 必须保持 `None`。**已按规格如此实现，未做任何冒充。**

失败纪律已落实：融合不达标 → `status="failed"` + `metric_scale=None`，**代码里没有任何回退到多锚点/校准池的路径**（那些模块已移入 `legacy/retired/`）。

### Phase 3　距离原语 + Tool 注册表 ✅

| 要求 | 落点 | 状态 |
|---|---|---|
| 稳健低分位距离（q 默认候选 1%） | `tools/distance_primitives.py::robust_distance_to_reference` | ✅ |
| 体素降采样 + conf-warp 软权重 + 最小点数 | `preprocess_points`（`C>2` 仅作**可选**掩码，绝不硬门） | ✅ |
| abs / rel / surface 三类分离 | `camera_object_distance` / `robust_distance` / `surface_distance_between_objects` | ✅ |
| 降级条件（点数不足 / 重复 / 污染） | `N_min` → 无精确值；`suspect_duplicate`；`point_contamination_suspect`（抑制数值） | ✅ |
| `min(q=0)` vs `q∈{0.5,1,2,5}%` 消融 | `ablation_quantiles` | ✅ |
| **DoD**：输出含全部 trace 字段 | `{distance_normalized, distance_metric, scale_version, quantile_q, voxel_size, conf_warp_version, n_valid_points, degradation_flags}` | ✅ |
| **DoD**：negative 测试受控错误不崩成服务故障 | 域值错误统一 `DomainValueError`；`connectivity_graph` 的 `KeyError` 已改为受控错误 | ✅ |

### Phase 4　选路器 + Skill 检索 ✅

| 要求 | 落点 | 状态 |
|---|---|---|
| 题型 + EvidenceProfile 签名匹配 | `routing/skill_retriever.py::retrieval_decision` | ✅ |
| 5 个 Skill 族 | `schemas/skill.py::FAMILY_QUESTION_TYPES` + 构造期族覆盖校验 | ✅ |
| paired A/B 同帧/输入/模型/模板/硬件 | `evaluation/experiment_protocol.py::assert_paired_ab` | ✅ |
| direct 来源单列 | `answer_source` 四值 + `evidence_state_breakdown` 的胜负率单列 | ✅ |
| **DoD**：不同证据签名下 Skill 分开积累互不污染 | `RetrievedSkill.matched_evidence_signature` + §17.2 分离测试 | ✅ |
| **DoD**：检索/执行双重 fail-closed | 检索校验 gate + 版本；执行层 `check_evidence_contract` 二次校验 | ✅ |
| **DoD**：选择率/覆盖率/胜负可按证据状态分组报告 | `evidence_state_breakdown(...)` | ✅ |

### Phase 5　partial recovery + ReturnAnswer ✅

| 要求 | 落点 | 状态 |
|---|---|---|
| 保留未受污染成功结果回灌 | `online/recovery.py::collect_validated` + `build_feedback` | ✅ |
| 级联撤销 + EvidenceProfile 能力降级 | `cascade_invalidate` + `downgrade_profile` → 重新派生 `question_tool_scope` | ✅ |
| 重置命名空间后注入 validated observations | kernel `reset_user_namespace` + 回灌文本 | ✅ |
| 有限次数（`[TODO_CALIBRATE]`） | `MAX_RECOVERY_ATTEMPTS` + `recovery_exhausted` | ✅ |
| AST/运行双层禁止答后调 Tool | `ast_guard`（静态，按行号）+ kernel（`AnswerAlreadyGiven`，§9.12） | ✅ |
| 解析回退单列 `m8_parse_recovered` | `assemble_program_ex` → `recovered` 标记 → `synthesis_source` | ✅ |
| 退化输出触发重生成 | `degenerate_reason` + `_regenerate_non_degenerate` | ✅ |
| **DoD**：`AnswerAlreadyGiven` 受控抛出不崩成服务故障 | 归 `tool_contract` 桶（子类码在 `contract_violations` 留痕），不再 IndexError | ✅ |
| **DoD**：`used_result_ids`/`recovery_count` 落盘 | `EpisodeTrace` + `TraceRecord` | ✅ |

> 本项直接针对 v5 [已实测] 缺陷：模型写出 `if not ids: ReturnAnswer("abstain")` 后继续
> `object_centroid(ids[0])` → IndexError → `violation_runtime`，内测方向题全栽在这里。
> 现在静态层直接拒绝该程序（要求重生成），运行层兜住 AST 拦不住的写法。

### Phase 6　实验协议 + trace ✅

| 要求 | 落点 | 状态 |
|---|---|---|
| 三级隔离（inner 可迭代 / outer 一次 / final 冻结） | `experiment_protocol.py`：`STAGE_RULES` + `RunLedger`（持久化，final 缺账本即拒绝） | ✅ |
| 样本量三档 | `SAMPLE_TIERS`（inner 16×3 / outer 32×3 / final 32+×5）+ `check_sample_size`；池 < 32 时用全池、靠 seed 补足并如实报池大小 | ✅ |
| 噪声底协议 | `noise_floor_report` + `assert_round_robin_not_for_comparison` | ✅ |
| paired A/B 统计 | `paired_score.py`：McNemar（小样本精确 / 否则 Yates χ²）+ 配对 bootstrap + Cliff's delta/Cohen's d + Bonferroni | ✅ |
| 退化样本 p=nan → None | `normalize_p` + "退化样本 → 判为不显著" | ✅ |
| 95% CI + 按证据状态分组 | `per_task_ci95` / `wilson_ci` / `evidence_state_breakdown` | ✅ |
| `paper_eligible` 四要件 | `check_paper_eligible`：数据隔离 / ≥3 seed（paper ≥5）/ 统计门 / 非 mock；并额外拒绝四个 `[待实验]` 项 | ✅ |
| `synthesis_source` 拆 6 类 | `schemas/trace.py` + 生产端接线（另加显式非论文值 `mock_stub`） | ✅ |
| 版本字段全落盘 | `TraceRecord`（模板/tool-face/EvidenceProfile/gate/距离原语/融合版本）+ `RunManifest` §19.2 | ✅ |
| split 访问审计 + 红线 9 | `SplitAccessLog`（读 outer 必留理由）+ `check_strategy_provenance`（outer 调出的策略不得进论文） | ✅ |

---

## 3. 与 v5 的机制差异（用户可据此核对是否走偏）

| 机制 | v5 | v6（本实现） |
|---|---|---|
| 米制尺度 | 多锚点 + log-scale 融合 + conformal 校准池（需 GT 位姿标定数据 → 数据 blocker） | 零样本度量深度跨帧融合（`[待实验]`）；无 GT / 无标定池 / 无 BA |
| 路由 | 单一 `route`（一词三义） | `scene_route`（只由 M4 质量定）× `question_tool_scope`（逐题派生，只收窄） |
| Tool 暴露 | `requires_artifacts` + 逐题型米制授权 | `requires_evidence` + `tolerates_degraded`（逐 Tool 证据匹配） |
| 尺度失败的影响 | 米制题 route 降 2D-only → 连带砍掉非米制 3D Tool（实测 12/32 题只剩 1 个 Tool） | 只收窄 `question_tool_scope`，**非米制 3D Tool 全保留**（真实数据实测 11 个） |
| 世界系 | 无契约；`relative_direction` 在 `up_direction()` 为 None 时退回包围盒启发式（实测选错轴） | `world_up`+`handedness` 进 Schema；缺失即 fail-closed，**不猜** |
| 质量门 | G1–G11 加权 scalar（含 G5 代理值风险） | 交叉双指标主门（warp ∧ 重叠），多指标不得单挑；G5 永久 `not_available` |
| 答后调 Tool | IndexError → `violation_runtime`（假"服务故障"） | AST 拒绝 + `AnswerAlreadyGiven` 受控异常 |
| 离线治理模型 | GPT-6（中转站，单次 ~243s） | DeepSeek-V4.1-Flash（`deepseek-flash` @ `https://api.deepseek.com`） |
| Skill 检索 | 产物 + 最低质量分 | 题型 + `required_evidence_signature`；米制 Skill 双重 fail-closed |

---

## 4. 未完成项 / blocker（如实列出）

| 项 | 状态 | 说明 |
|---|---|---|
| MoGe-2 权重与 C1–C6 PoC | **未跑** | 需下载 `moge-2-vitl` 并占用一张卡；`[待实验]`，未跑前 `metric_scale` 恒 `None` |
| Metric3D v2 同场景对照 | 未开始 | `[TODO_LICENSE]`：需用户直接读官方仓库 LICENSE 核实 |
| τ_warp / τ_cloud / τ_scale_disp / N_min / quantile_q / 体素大小 等阈值 | 全部 `[TODO_CALIBRATE]` | 必须是起始参考值；须按 §10.6 最小 PoC 在自有数据上标定 |
| conf-warp 单调性自检在生产路径的实际接线 | **未接线** | `compute_quality` 未调用 `conf_warp_monotonic`，故真实路径 `conf_warp_monotonic` 恒 `None` → conf 不作过滤（保守）。接线会改变真实路径的质量分类，需要独立标定后再开 |
| M8 多模态合成 + 完整 M1–M13 真实 smoke | 未跑 | 本次真实 smoke 覆盖 M3+M4 证据链；M8 需 vLLM 服务（`scripts/serve_qwen3vl_dp8.sh`） |
| 系统级实验（inner/outer 主表） | 未跑 | 依赖 M8 与更大样本；`paper_eligible` 无任何能力达标 |
| `RunLedger` 的跨进程原子性 | 已知限制 | `check`+追加不是跨进程事务（有极小 TOCTOU 窗口）；单进程内正确 |
| synset `EpisodeTrace.synthesis_source` 与 `TraceRecord.synthesis_source` 的 `mock_stub` | 已加显式值 | v6 §19.3 给的是 6 类；`mock_stub` 是**实现补充**（mock_light 专用，显式排除在 paper-eligible 外）——写成 6 类中任一个都是假话 |

---

## 5. 归档与隔离

- **源码归档** `src/skill3d/legacy/retired/`：`scale_calibration.py`、`scale_assessment.py`、
  `scale_units.py`、`metric_scale.py`、`scale_poc.py`、`scale_report.py`、`ba.py`、
  `sparse_ba/`、`legacy_vggsfm_ba/`、`sparse_ba.py`、`colmap_baseline.py`、
  `dust32_mast3r_fallback.py`。附 `__init__.py` 写明逐项废止依据与替代物；
  **任何运行时代码不得 import 本目录**。
- **测试归档** `tests/archive_v5/`：31 个 v5 回归测试文件 + `README.md`（逐项替代物表），
  已加入 `pyproject.toml` 的 `norecursedirs`（刻意不收集）。
- **Schema 隔离**：`ReconstructionArtifact` 只接受 `schema_version="6.0"`；22 个 v5
  尺度/BA/G8/pad 字段列入 `LEGACY_ONLY_FIELDS`，出现即 hard fail（不静默忽略）。
  旧产物只能经 `skill3d.legacy.readers` 只读审计。

---

## 6. 下一步（按对 benchmark 分数的价值排序）

1. **接通 M8 并跑 inner 主表**（vLLM 服务 + 现有 scoped 样本）：这是唯一能产出
   真实数字的路径，也是判断 v6 证据机制是否真的涨分的前提。
2. **MoGe-2 PoC（§11 C1–C6）**：决定 3 个米制题型能否进主表；未过则按 §11.4
   关闭尺度融合支路，系统退纯相对几何（MCA 四题不受影响）。
3. **阈值标定（§10.6）**：τ_warp / τ_cloud 的最小 PoC 标定，然后按 §18.2 扩样本。
4. **conf-warp 自检接线**：接上 `compute_quality` 后重新测主门通过率与 `degraded` 分布。

---

## 7. 真实 GPU 端到端实测（M1–M13，v6 全链）

命令（vLLM FP8 单卡 GPU4；VGGT/SAM2 在 GPU3，分卡纪律）：

```bash
python -m skill3d.online.eval --mode real --source vsi_bench --split inner_validation \
  --datasets arkitscenes --limit 2 --seed 0 \
  --vllm-endpoint http://127.0.0.1:8100 --vllm-model qwen3vl-8b-r0 \
  --recon-dir data/v6_smoke/recon --trace-dir data/v6_smoke/traces2
```

| 题 | scene | scene_route | M4 主门 | track_consensus | 程序 | 结果 |
|---|---|---|---|---|---|---|
| qa 2 | 41069043 | `full_3d` | warp **0.668** / overlap **0.551** → 通过 | `degraded`（碎片化 34%、重复嫌疑 33%） | `count_objects('table')` → `count['count']` | **答 2 = GT 2，MRA 1.00** |
| qa 12 | 41159572 | `fallback_2d_only` | warp **0.422** / overlap 0.458 → **不通过**（warp 未达 τ=0.5） | `degraded` | 模型直接 `ReturnAnswer("abstain")`（Tool 已被收回） | 按错计（fail-closed 正确） |

**这张表的两条关键证据**：

1. **track 共识计数在真实数据上跑通**：v5 [已实测] 的计数失败根因是"清单既有重复（12）
   又有漏绑（0）"，v6 改成 `count_objects` 按 `track_id` 共识计数 + `duplicate_suspect`
   降级标记后，模型用 `count_objects` 直接答对（MRA 1.00）；
2. **"多指标不得单挑"在真实数据上生效**：qa 12 的 `cloud_overlap=0.458` 单看是**过**的
   （τ=0.3），但 `warp_inlier=0.422` 未达 τ=0.5 → 主门整体判**不通过**。
   若按 v5 的单指标思路放行，这个几何不自洽的场景会被当成 `full_3d` 使用。

### 本轮实测修出的三个真实缺陷（均已修 + 加测试）

1. **`degraded` 容忍口径**（见 §2 Phase 0）：`geometry_3d=degraded` 曾隐藏整个 3D Tool 面。
2. **`track_consensus` 判据口径**：原判据用"物体可见帧率"，实测恒低（0.14–0.19），
   会把 counting 题的程序路径**结构性掐死**；而 v5 实测的失败根因是重复/碎片化。
   已改为 `碎片化率 ∧ 重复嫌疑占比`（旧口径保留为诊断 subvalue，改口径前后可直接对比）。
3. **eval 路径 artifact 未落盘**：现场重算的 scene 只把 artifact 留在内存 → 下次运行
   重新跑一遍 VGGT（~1 分钟/场景）且 trace 无可审计 artifact。已显式落盘（方案 X 在
   eval 路径同样成立）。

### 诚实边界

- 2 题 **不是精度结论**，不进主表（§18.6）；这里只主张"链路跑通 + 机制按设计生效"。
- τ_warp / τ_cloud 仍是**未标定**的起始参考值：qa 12 在 0.422 被判不通过，
  是过严还是该场景真坏，**只能靠 §10.6 的标定 PoC 回答**，不能靠调阈值。
- 米制三题在本轮全部走 `direct_vlm_routed`（`metric_scale=unavailable`）——
  这是 MoGe-2 PoC 未跑的**预期行为**，不是缺陷。

---

## 8. 2026-09-22 会话增量（inner 档扩样本 + 三处实测缺陷 + 两处根因纠正）

> 本节由接手会话追加；只写**今天真实跑过/真实复现**的东西，未跑的一律标 `[待实验]`/未完成。

### 8.1 已跑通的（有真实产物）

| 项 | 结果 |
|---|---|
| 全量 CPU 测试 | **768 passed / 3 skipped / 0 failed**（与交接基线一致） |
| inner 档重建（§18.2：16 题/题型） | `inner_validation × scannet+scannetpp` 需要 **24 个 scene**，6 卡分片重建 **全部完成、0 失败**；manifest 合并落盘 `data/v6_scoped/recon_moge2/vggt/manifest.json` |
| MoGe-2 度量尺度（§11） | 24 scene 中 **23 success / 1 failed**（融合失败即 `metric_scale=None`，按纪律不伪装） |
| 世界系契约 | `world_frame_status`：**available 14 / degraded 10** |
| M4 主门 | **3/24 scene 的 `overall_quality=0.000`** → 这些 scene 的全部 3D 题型被收窄为 `fallback_2d_only`（τ 未标定，见 §8.4） |
| **DeepSeek-V4.1-Flash 真实健康检查** | **通过**：`GET https://api.deepseek.com/models` 返回 `["deepseek-flash","deepseek-v4-pro"]`，规范要求的 `deepseek-flash` **确实存在**、key 有效（`endpoint_hash=a34e2a47…`）。这是该项目**第一次对真实离线模型发出请求**（此前 `run_manifest_offline.json` 的 `provider`/`model_id` 为空串） |
| inner 档主实验（16 题/题型 × 3 seed × 2 臂） | **已启动**（`scripts/run_inner128_batch.sh`，128 题 × 6 run）。**截至本节写作时仍在跑**，尚无最终数字 → 本报告不预填任何分数 |

### 8.2 本轮修出的真实缺陷（均带回归测试）

1. **`_stack_masks` 尺寸守卫导致掩码产物整片为 0**（`segmentation/sam2_tracker.py`）
   - 现象：真实 corpus 里**每个**对象的 `*_mask.npy` 数组 `sum()==0`（7 scene、54 对象全中），
     而清单里的 `visible_frames` 由**未缩放**的掩码算出、依然非空 → **落盘产物与清单自相矛盾**。
   - 根因：SAM2 掩码在**视频分辨率**（如 960×1280），本函数拿到的 `shape` 是
     **VGGT 深度网格**（518×392）；旧实现 `if np.asarray(m).shape == out.shape[1:]`
     尺寸不等就**静默跳过**，于是整片留 0。
   - 修复：按 `_resize_mask_to`（纯等比最近邻，§20 唯一正确映射）映射后再堆。
   - 影响面（已核对）：`mask_per_frame` 在 `src/` 内**没有任何其他消费方**
     （工具/评测/门都不读它）→ **本次缺陷不影响任何分数**，属**产物可审计性**缺陷。
   - 回归测试 `test_bind_masks_to_world_persists_mask_at_video_resolution`：
     已实测在**旧实现上失败、新实现上通过**（旧 sum=0 / 新 sum=9600）。
     旧测试因让 mask 与 depth **同尺寸**而恰好绕过该路径 —— 这是"测试与生产形状不一致"的教训。
2. **分片重建互相覆盖 manifest**（`reconstruction/run.py`）
   - 现象：6 个 `--scene-shard` 进程都写同一个 `manifest.json` → 最终只剩最后一个分片的 5 条 job，
     **整批 24 scene 的 provenance 丢失**。
   - 修复：分片各写 `manifest_shard<i>of<n>.json`，由 `scripts/recon_inner128.sh` 合并；
     本轮已完成批次的 `manifest.json` 由磁盘 artifact **重建**，并在文件里显式标注
     `manifest_origin="rebuilt_from_artifacts"` 与原因（不虚构 `gpu_rank` 等运行期字段）。
3. **`evaluation_result` 不落原始答案 → 失败无法归因**（`online/runner.py`）
   - 现象：`predicted=None` 有两种截然不同的原因 —— 模型 abstain（无答案）与
     模型给了自由文本但抽不出选项字母；旧 trace 两者不可分（实测 qa 2770 属后者，
     却只留下一只"普通错题"）。
   - 修复：`evaluation_result` 增落 `answer_text` / `answer_source` / `abstained` / `failure_code`
     （§19.1「不依赖重跑即可归因失败」）。

### 8.3 两处根因纠正（交接文档里的假设经实测**不成立**）

1. **外观顺序题的瓶颈不是"visible_frames 有空洞"**
   → 见 `docs/decisions/D-2026-09-22-appearance-order-bottleneck.md`。
   实测 40 题：`naive_min` 与"首次连续可见段"给出**完全相同的排序（0/40 题不同）**
   → 该修法是**可证明的空操作**；而 `naive_min` 命中 GT 仅 **5.0%**，
   4 类别随机命中率 1/24≈4.2% → **该机制在本数据上不携带信号**。
   真瓶颈是接地召回：**85% 的题在 4 个候选类别里至少缺 1 类、平均缺 1.40 类**。
2. **route_planning 的失败与 `connectivity_graph` 无关**
   → 见 `docs/decisions/D-2026-09-22-route-planning-rootcause.md`。
   在 **177 个真实程序、373 次工具调用**里 `connectivity_graph` 被调用 **0 次**（不可能抛错）。
   真根因：① 题面用**颜色属性**限定实例（"the blue chair"）而 `list_objects` 只给类别名
   → 模型无法定位实例 → 主动 abstain（3/4 题）；② 1 题所在 scene 未过 M4 主门
   → `fallback_2d_only` → 工具面只剩 `euclidean_distance` → 弃答（**fail-closed 正确行为**）。

### 8.4 仍未完成 / 新增 blocker（如实列出）

| 项 | 状态 | 说明 |
|---|---|---|
| inner 档 3-seed 主表数字 | **未出** | 跑批进行中；本轮不预填、不预估 |
| 噪声底（§18.3） | 工具就位、数字未出 | 新增 `scripts/noise_floor.py`（逐题一致率 + 跨 seed 极差 + 配对前提校验，qa_id 不一致即拒绝）；输出待跑批结束 |
| **outer_holdout 的消耗** | **未消耗（刻意）** | 离线演进闭环的 L3 面板口径就是 `outer_holdout`（`offline_driver.load_panels`），而 §18.1/硬约束 10 规定 outer **只跑一次**。是否现在就把这一次花掉，属**用户决策**，本会话不动。 |
| 离线归纳（DeepSeek 真实调用） | **未跑**（数据前置就位中） | 需要 induction split 的**真实轨迹**；induction 重建已在跑（`data/v6_induction/recon`）。health_check 已通过（§8.1）。 |
| 属性接地（颜色/材质） | 新发现的缺口 | 需 M5 产出实例属性标签；属新能力，须走 §17.4 准入，**不得**在题面字符串上特判 |
| 门过严的可能性（3/24 scene 判 0） | 未标定 | 只能由 §10.6 标定 PoC 回答，不得调阈值 |
| §10.3 conf-warp 自检接线（item I） | 仍未接线；**影响面已查明，比原记录更严重** | 见 §8.6 |
| `AnswerAlreadyGiven` 被归入 `tool_contract` 桶 | 已登记、刻意不改 | "模型本想 abstain"被记成"契约违规"；两者主榜都按错计 → 不影响分数，仅影响失败类型分布分析 |

### 8.6 §10.3 conf 软权重在真实数据上**完全失效**（对原记录的修正）

旧记录写的是"`conf_warp_monotonic` 恒 `None` → **conf 不作过滤**（保守）"，
并把接线风险表述为"会改变真实路径的**质量分类**"。本轮把整条消费链读完后修正如下：

1. **取值已核实**：24 个真实 artifact 的 `quality.conf_warp_monotonic` **全部为 `None`**。
2. **消费链**：`geometry_tools.py:400/443` 把 `handle.conf_warp_monotonic` 传给
   `distance_primitives._conf_weights`；而该函数对 `monotonic is None` 的分支返回
   **`conf_usage="ignored_unverified"` → 既不加权也不过滤，`point_conf` 被完全不用**。
3. **真实后果**：M5 逐对象落盘的 `pointconf_world`（每对象 20000 个逐点 VGGT 置信度）
   在**所有**真实距离计算里**从未被使用**。§10.3/§12.2-2 的"conf 只作软权重"
   这条纪律在真实数据上不是"保守地降权"，而是**整条机制不生效**。
4. **接线风险的正确表述**：把自检接上（在 `main_gate` 用 `warp_residual_map`
   现场算）对**主门判定是中性**的（`conf_weight` 在 `m4_main_gate` 之外**没有任何消费方**，
   也已核实它不落盘、不进 `overall_quality`）；但它会**改变工具输出**——
   `monotonic=True` 时 `point_conf` 会第一次真正参与距离加权（`conf_usage="weighted"`），
   从而改变 `robust_distance` / `extent` 等米制数值。
   **所以"接线后要重跑"是对的，但原因不是质量分类，而是距离原语数值。**
5. **本轮不做**：跑批进行中，不在主实验期间改动最安全关键的模块；接线 + 标定登记为下一步。

### 8.5 本轮代码状态的诚实说明

- 3-seed 跑批**启动于** `evaluation_result` 增字段之后、**掩码/分片修复之前**；
  掩码修复不影响任何分数（§8.2 第 1 条已核对无消费方），故跑批数字与修复后的代码**在分数口径上**一致。
  跑批的 `RunManifest.code_commit` 记为 `9c2df54`，而工作区含本轮未提交改动 —— 提交后需注意这一对应关系。
