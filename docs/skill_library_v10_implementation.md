# v10 实现记录（P0 → P3）

> 规范：《系统架构v10.md》（根目录，唯一目标态）
> 代码基线：`aa5810f2177dcc5b190a3fb66cf3b9bfca4932c5`（v9 落库点）
> 本轮范围：v10 §15 的 P0 → P3；§20 的三项待确认**仍待用户裁决**（见文末）

本文逐项记录"改了什么 / 依据的规范原文 / 明确未完成项"，与 v9 记录
（`docs/skill_library_v9_implementation.md`）并列，不覆盖后者。

---

## 0. 本轮新增与修改面

**新增**

| 文件 | 作用 |
|---|---|
| `src/skill3d/schemas/experience.py` | §6.1 `ExperienceEvent`、§6.3 `ExperienceBundle` + 词汇表（四态检索状态、排除原因码、受控摘要键） |
| `src/skill3d/evolution/experience.py` | 经验资格与经验包的**确定性构造**（§6.2 七条 → 可执行判定；`usage_supported` 规则；按 `skill_id@version` 聚合；落盘/读取） |
| `src/skill3d/evolution/campaign.py` | §11 `EvolutionCampaign` 状态机（两代唯一调度入口）+ §14.1 十类收据 + `seed_campaign_library` |
| `scripts/run_evolution_campaign.py` | 真实运行入口（冻结面板 / 建 campaign 库 / 接在线与离线模型 / 跑状态机） |
| `tests/unit/test_v10_campaign.py` | 两代闭环结构测试（注入合成 runtime） |
| `tests/unit/test_v10_candidate_contracts.py` | §5.4/§7.2/§7.4 候选合同测试 |
| `tests/integration/test_v10_fixed_injection.py` | §8.2/§8.3/§8.4/§5.3 真实 runner 路径测试 |

**修改**（逐条对应 §15 的 P0/P1/P2）

| 文件 | 改动 |
|---|---|
| `governance/induce.py` | 新主入口 `induce_candidate_from_bundle`（经验包 → 完整候选）；`validate_candidate_against_parent`（§5.4 静态检查唯一实现）；`structured_diff`/`apply_structured_diff`；旧路径改名 `induce_candidate_legacy_v9` |
| `governance/revise_patch.py` | 删除 `# PATCH` 文本追加；`apply_json_merge_patch`（结构化作用在解析后的对象上）+ `revise_with_full_spec`（完整候选路径）；`parse_patch` 拒绝非 JSON `patch_content` |
| `evolution/panel.py` | `run_fixed_skill_evaluation`（§8.2 A/B 固定注入）+ `decide_promotion`（§8.6/§13 逐 seed 条件）；`fixed_injection_binding` |
| `skills/promote_atomic.py` | `publish_candidate_snapshot`（§10.1 完整快照 + §5.3 谱系竞争与历史化）、`publish_lock`、`build_candidate_snapshot`、manifest 复算 |
| `schemas/evolution.py` | v10 合同：`SkillCandidate`（§7.2）、`SkillEvaluationBinding`（§8.2）、`EvolutionCampaign`（§11.2）、`StaticValidationReceipt`/`PairedPanelReceipt`/`CampaignDecision`/`PromotionReceipt`/`PostPublishUseReceipt`（§14.1） |
| `schemas/skill.py` | v5 形状的 `SkillCandidate` 改名 `SkillCandidateV5`（只读历史），包级 `SkillCandidate` 指向 v10 定义 |
| `schemas/retrieval.py` | 新增原因码 `lineage_version_not_selected` / `fixed_injection_*`；记录新增 `lineage_selections` / `evaluation_binding` / 选择阶段请求 hash；`record_lineage_selection` / `select_version_in_lineage` / `mark_fixed_injection` |
| `routing/skill_retriever.py` | §5.3 谱系分组：先题型过滤 → 按谱系分组 → 谱系内选一个版本 → `top_k` 针对谱系；分数/tie-break/原因落盘 |
| `skills/registry.py` | 在线 loader 跳过 `state == historical` 的条目（§5.3-5），并保留警告 |
| `online/runner.py` | 固定注入分支（§8.2/§8.3）；§8.4 两阶段版本选择（选择请求 + 请求/响应 hash + 确定性回落）；`FixedSkillInjectionError` |
| `evolution/offline_driver.py` | v9 路径改用 `induce_candidate_legacy_v9`（明确标注不满足 v10 合同） |

---

## 1. 七个阻断点的逐项修复证据（v10 §1.2）

| ID | v10 要求 | 实现落点 | 证据 |
|---|---|---|---|
| EV-01 归纳器聚合全部 episode | 建立 `ExperienceBundle`，按具体 Skill 版本分桶 | `schemas/experience.py::ExperienceBundle`（构造期校验"必须 `skill_id@version`、单题型"）、`evolution/experience.py::build_experience_events/build_experience_bundle`（只聚合 `skill_version == parent_skill_key` 的事件，跨谱系直接抛错） | `tests/unit/test_v10_campaign.py::test_experience_bundle_excludes_inner_split_and_unbound_traces`（混桶抛错）、`tests/unit/test_firewall.py`（prompt 由父 Skill + 经验包摘要构成） |
| EV-02 候选默认 `parent_version=None` | 修订必须声明父快照与父 Skill 版本 | `schemas/evolution.py::SkillCandidate`（`operation` 只有 `revise`；`parent_snapshot_id`/`parent_skill_version`/`candidate_skill_version` 必填；跨谱系/版本不一致直接 ValidationError）；`governance/induce.py::induce_candidate_from_bundle` 只产这种候选 | `tests/unit/test_v10_candidate_contracts.py::test_candidate_schema_rejects_cross_lineage_and_empty_diff` |
| EV-03 A 臂空 Skill | A 固定注入父 Skill、B 固定注入候选 Skill | `evolution/panel.py::run_fixed_skill_evaluation` + `fixed_injection_binding`；`OnlineRunConfig` 在 `evaluation_binding` 非空时**要求恰好一条** Skill（0 条或多条 → `ValueError`） | `tests/integration/test_v10_fixed_injection.py::test_fixed_injection_requires_exactly_one_skill`、`::test_fixed_injection_delivers_exact_arm_body_in_real_mode` |
| EV-04 `# PATCH` 文本追加 | 结构化应用，输出完整可校验 SkillSpec | `governance/revise_patch.py`（删除字符串追加；Merge Patch 作用在解析后的对象上；`parse_full_spec_response` 显式拒绝含 `# PATCH` 的文本） | `tests/unit/test_v10_candidate_contracts.py::test_json_merge_patch_semantics_and_full_spec_output`、`::test_free_text_patch_content_is_rejected`、`::test_free_text_patch_shape_is_rejected` |
| EV-05 promote 只追加 entry | 发布完整候选快照，显式登记谱系与竞争集合 | `skills/promote_atomic.py::publish_candidate_snapshot`（重算 entries/`skill_versions`/`active_skill_versions`/`historical_skill_versions`/`competing_lineages`/`generation`/manifest hash；`PromotionReceipt` 记父子关系与回滚点） | `tests/unit/test_v10_campaign.py::test_two_generations_promote_and_receipts_complete`、`tests/integration/test_v10_fixed_injection.py::test_third_active_version_historizes_the_oldest_at_publish` |
| EV-06 两轮/双 seed 是死配置 | 新增 `EvolutionCampaign` 消费配置 | `evolution/campaign.py`（状态机 + checkpoint + 收据）；`scripts/run_evolution_campaign.py` 从 `configs/config.yaml` 真读 `max_evolution_rounds` 与 `candidate_validation_seeds` | `tests/unit/test_v10_campaign.py`（全文件：两代、双 seed、恢复）；CLI 打印读到的配置值 |
| EV-07 候选面板混题型 | 候选继承父唯一规范题型，只从独立 inner 子集加载同题型题目 | `run_evolution_campaign.py::FrozenPanels`（按规范题型过滤 + inner 按 scene 整块切成互不重叠的 g0/g1 + 面板 hash 落盘）；`SkillCandidate` 校验 `canonical_question_type == 父题型` | `tests/unit/test_v10_campaign.py::test_two_generations_promote_and_receipts_complete`（bundle 题型一致）、面板冻结清单 `panels.json` |

---

## 2. 关键判定规则（写成可执行代码的口径）

### 2.1 `usage_supported`（§6.2-4 的"可观察的程序使用线索"）

`evolution/experience.py::usage_state_of` 逐 (episode, 版本) 给出四态：

- `not_retrieved`：没出现在任何检索记录里，或记录了但硬条件未过；
- `retrieved_not_delivered`：进了 top-k，但正文没进任何实际发出的请求；
- `delivered_not_used`：正文进了请求，但没有任何可观察使用线索；
- `usage_supported`：正文进了请求，且至少一条线索成立 —— `declared_in_program=true`
  （程序文本里字面出现该版本）**或** `template_tool_overlap` 非空（模板点名的 Tool
  与程序里出现的 Tool 有交集）。

线索来自 `online/runner._skill_usage_clues` 的**机械交叉检查**，只用于资格判定，
不用于给 Skill 记功（§13.6 口径不变）。

### 2.2 资格判定（§6.2 七条 → 排除原因码）

`build_experience_events` 对每条事件给出 `eligible_for_induction` 与
`exclusion_reasons`（词表见 `schemas/experience.py::EXCLUSION_REASON_CODES`）：
`not_parent_snapshot` / `snapshot_identity_unconfirmed` / `split_not_learning` /
`split_unknown` / `retrieval_record_missing` / `skill_not_retrieved` /
`skill_not_delivered` / `delivered_but_no_usage_clue` / `result_identity_missing` /
`schema_corrupted` / `skill_identity_unmapped`。
经验包另给 `exclusion_summary`（原因 → 条数），"这批经验为什么不能用于归纳"可回答。

### 2.3 准入条件（§8.6 + §13）

`panel.py::decide_promotion` 逐 seed 判定并在 `decision.json` 里逐项落盘：
`panel_score_strictly_improved`（严格提高）、`run_error_not_increased`、
`valid_answer_rate_not_decreased`、`candidate_delivered_at_least_once`、
`both_arms_body_entered_request`、`same_episode_and_artifacts`、
`no_schema_or_permission_violation`、`content_and_experience_eligible`。
`promote=True` 要求**全部**条件为真（Schema 层强制）；缺任一 seed 的结果直接
`same_episode_and_artifacts=False`。

### 2.4 版本递增（§7.3 的落地口径）

`governance/induce.py::validate_candidate_against_parent`：

- `skill_id` 必须等于父谱系；版本必须严格递增；
- 改动**检索前提**（题型 / 证据签名 / 米制 gate / 族 / 来源）→ 需 MAJOR → **首期拒绝**；
- 只改描述 / 步骤 / 检查等的 MINOR 变更**接受**（§7.3"修改描述…：MINOR bump"）；
  "只修错字不改变行为"这一类机制上无法与"改进描述"区分，因此由
  `no_effective_diff`（除版本号外零字段变化）与下游配对评测把关；
- 工具名必须来自当前 ToolSpec；§12.3 的禁止修改项（阈值 / top_k / 采样 …）命中即拒；
- 泄漏扫描（答案模式 / qa_id / 题号）命中即拒；
- `structured_diff` 必须非空、且**再应用**回父对象能还原候选（一致性复核，不靠人读）。

### 2.5 谱系竞争（§5.3）

- 检索：题型分区 → 谱系分组 → 谱系内选一个版本（分数降序 + 稳定键 tie-break，
  **版本号不参与打分**）→ `top_k` 截断的是**谱系**；同谱系其余版本记
  `lineage_version_not_selected`；
- 发布：谱系内最多两个在线竞争版本，第三个进入时最旧（版本号最小）转 `historical`，
  仍留在快照里但不再参加检索（在线 loader 跳过并留警告）；
- 记录：`lineage_selections` 逐谱系记候选、分数、`ranking_policy`、
  `reason`（`single_eligible_version` / `higher_ranking_score` /
  `score_tie_break_stable_key` / `model_selected_from_summaries` /
  `deterministic_fallback`）。

### 2.6 两阶段选择（§8.4）

正常检索在同谱系存在 >1 个可选版本时发起**选择请求**（只含短摘要，不含完整正文）；
模型返回合法版本 → 记录 `model_selected_from_summaries` 并只把该版本正文放进后续
请求；返回不可解析/不在候选里 → 保留确定性结果并记 `deterministic_fallback`
（**失败也留痕**：请求与响应 hash 都落盘）。固定注入臂不参与检索选择，因此这条路径
不影响 A/B。

---

## 3. 测试

```bash
EXP=/home/cvailab/anaconda3/envs/skill3d-exp/bin/python
PYTHONPATH=src $EXP -m pytest tests/ -q -p no:cacheprovider
```

- 全量：见文末"本轮回归数字"；
- 新增/改写的测试：`tests/unit/test_v10_campaign.py`、`tests/unit/test_v10_candidate_contracts.py`、
  `tests/integration/test_v10_fixed_injection.py`；
- **被 v10 取代的旧断言**（改写时已在 docstring 写明取代关系）：
  - `tests/unit/test_firewall.py::test_induction_prompt_contains_no_ground_truth`：
    v10 §7.1 取代 v9 的 `build_induction_prompt(failure_summaries, input_features)`；
  - `tests/unit/test_offline_driver.py` 的 REVISE 替身：v10 §5.4 取代 v9 的
    `# PATCH` 文本追加（替身现在输出**完整候选 SkillSpec**）。

---

## 4. 明确未完成 / 不得由本收据宣称的项

1. **`operation=create`**：v10 §5.2 明确延后到单题型两代闭环之后；当前
   `SkillCandidate.operation` 只接受 `revise`。
2. **outer_holdout / final_test**：首期不跑（§0/§8.5），campaign 只加载
   `induction`（= learning）与 `inner_validation`。
3. **§8.5 的"inner 逐题内容不进归纳器"**：本实现保证 inner 结果只做选择门、
   **不参与**经验包构造（经验包只从 learning 运行的 trace 构建）；但 inner 面板的
   `paired_seed_*.json` 仍写在 campaign 收据目录里 —— 归纳器的**输入**只有经验包。
4. **成本报告**：§13"成本只报告，不作为首期准入门"——目前收据里记录了
   `n_items` / 轮数等，但没有统一的 token / GPU 成本汇总表。
5. **§5.3-2 的"先按题型过滤再按谱系分组"中，谱系排序的语义排序权重**仍来自
   `configs/config.yaml` 的冻结策略；本版没有引入任何"版本越新加分"的规则。
6. **两代完成度**：见第 5 节（真实运行收据）。若第一代被拒，`completion_status`
   记 `completed_with_rejection`，**不**声称"两代 Skill 积累完成"。

---

## 5. 真实运行收据（两代 campaign）

已完成（2026-09-28）：

- **`camp-v10-002`**（冻结代码下的正式两代运行）：`data/v10_campaign/runs/camp-v10-002/`
  —— 两代都真实跑完、两个候选都被真实拒绝，`completion_status=completed_with_rejection`；
- `camp-v10-001`（开发过程运行，含上述第 7 节 8 条修正的轨迹）：
  `data/v10_campaign/runs/camp-v10-001/`。

数字、可声明 / 不可声明结论与未完成项见 **`系统架构v10_实现与验收报告.md`**。
口径修正（第 7 节）中影响数值的三条（MRA 正确率、`label_access`、面板标签）只影响
**后续运行**：`camp-v10-002` 的收据是修正后口径（`label_access` 例外，见第 10 条），
`camp-v10-001` 的旧口径收据原样保留以便对账。

---

## 6. 决策记录（本轮新增）

| 事项 | 结论 | 依据 / 说明 |
|---|---|---|
| 演化发生在哪个 Skill 库 | **另一个库**（`data/v10_campaign/library`），从仓库 `skill_library/` 复制快照与 manifest | 仓库 S0 库有 `--check` 不变量（S0 快照 digest + active 指针），一次 promote 就会让它失败；复制后 S0 库保持只读 |
| campaign 的 learning 层取名 | 仓库 split 名 `induction` → 规范层 `learning`（显式映射表 `SPLIT_ALIASES`） | §6.1 的 `split` 字面量是 `learning`/`inner_validation`；不把 `induction` 静默当 learning |
| 父版本的选取 | 上一代晋升的版本；无晋升记录时取谱系内最高版本 | §11.3"若 S01@1.1.0 晋升且被真实使用：以其经验修订为 S01@1.2.0"；晋升后新旧共同竞争，必须显式指认父版本 |
| `max_competition=2` | `publish_candidate_snapshot` 默认 2（可配置） | §5.3-4 |
| 离线模型 | §3.4 冻结的 DeepSeek 官方 endpoint + `deepseek-flash` | 密钥只从环境变量注入（本机凭据文件），不进收据、不落盘 |
| 面板抽样 | scene 级确定性抽样（`stratified_per_task=3N` + `max_per_scene=1` 后去重截取） | §5.3"以 scene 为单位控制覆盖"、"禁止按文件行序取每类前 N 题"；抽样收据与面板 hash 落盘 |

---

## 7. 真实运行中发现并修复的问题（P3 的第一手材料）

下列问题**全部由真实运行暴露**（合成/注入测试没有覆盖到），修复后都补了回归测试：

| # | 现象 | 根因 | 修复 | 回归测试 |
|---|---|---|---|---|
| 1 | 经验包 `answer_correct_rate=0.0`，但所有题的 `answer_correct` 都是 `None`（`object_counting` 是 MRA 题型，官方判分是相对精度） | 指标的分母写成了"合格经验数"，把"没测过对错"当成"错" | `answer_correct_rate` 只对有判分布尔量的题给值（分母 `n_answer_graded`）；MRA 另记 `n_mra_graded`/`mra_mean`，且只统计**合格**经验；`BEHAVIOR_SUMMARY_KEYS` 同步补齐 | `tests/unit/test_v10_campaign.py::test_mra_panel_reports_mra_mean_and_never_calls_it_zero_percent_correct`、`::test_behavior_summary_keys_are_the_declared_controlled_vocabulary` |
| 2 | 一次 `resume --max-generations 2` 拒绝执行第二代：`从 checkpoint 恢复 … COMPLETE` 后立即收口 | `_finish()` 把**完成说明**写进了 `_stop_reason`，而 `_stop_reason` 是恢复时的闸门 → "两代都未晋升"这类正常结论被当成阻断原因 | 完成说明另存 `completion_note`；`_stop_reason` 只承载 blocked/failed/发布不可见 | `test_resume_does_not_reconsume_panels_or_model_calls`（另：`_finish` 语义在 checkpoint 往返中保持） |
| 3 | `resume` 时 `promotion.json` 解析失败：`PromotionReceipt` 少了 `skill_version`/`snapshot_after`，多了 `promoted`/`reasons` | reject 时写的是"字段留空的 promote 收据"，形态含糊、也解析不了 | 新增 `RejectionReceipt`（`outcome="rejected"`），`promotion.json` 以 `outcome` 字段区分两态；历史无 `outcome` 的记录按"`promoted=false` → reject"解释 | `test_rejection_keeps_active_pointer_and_next_generation_retries` |
| 4 | 第三代候选生成时 DeepSeek 超时，**未捕获异常把 campaign 进程打崩**，checkpoint 停在半途 | `_generate_candidate` 只捕获"经验不足/静态检查失败"，没接 §3.4 的离线失败族 | 捕获 `OfflineAuthError`/`OfflineServiceUnavailable`/`OfflineRequestError`/`OfflineResponseError` → 写 `offline_failure.json` + `blocked:<结局码>`，不切在线链、不造伪候选 | `test_offline_failure_blocks_instead_of_crashing` |
| 5 | 合法候选被判"引用未知工具 `append`"（第二轮又出现 `else`） | 工具名扫描把**属性访问**（`tracks.append(...)`）与 Python 关键字当成工具调用 | 正则要求工具名**前面不是 `.`**；`_NON_TOOL_CALL_WORDS` 补齐常用容器方法与 Python 关键字 | `test_tool_reference_check_ignores_method_calls` |
| 6 | 候选第一次不合法就直接 blocked，没有 §7.4 的"结构化错误反馈 → 新 revision" | 缺重试路径 | `_generate_candidate` 按 `max_candidate_attempts`（默认 3）把静态检查问题清单回灌进归纳 prompt 重试；每次尝试收据 `static_validation_attemptN.json`；用尽才 blocked | `test_static_validation_failure_retries_with_structured_feedback` |
| 7 | 静态检查收据属于**另一个**候选（上一次失败留下的）却被当成本次结论复用 | 复用判断只看文件是否存在 | 收据的 `candidate_id` 必须与当前候选一致才复用 | 同上（真实运行日志里出现过 `忽略属于其它候选的静态检查收据`） |
| 8 | "锁定代码 commit"在**工作区脏**时不足以锁定身份（v10 改动都在工作区） | 只记 `git rev-parse HEAD` | 运行清单新增 `code_state`：commit + `dirty_worktree` + 13 个关键源码的 sha256 与聚合摘要 | 运行清单 `parent_run_manifest.json` |
| 9 | `label_access` 全为 `false`，但经验包里确实有 `mra_mean`（归纳器看过误差信号） | §6.4 的"误差"也是标签信号，旧实现只认布尔对错 | `label_access` = learning ∧ 合格 ∧（有对错布尔量 ∨ 有 MRA 分值）——被排除的经验不进经验包，故记 false | `test_label_access_true_only_for_eligible_learning_experiences` |
| 10 | A/B 收据的 `panel_id` 写成 `inner_g1`/`inner_g2`，而内容用的是 `inner_g0`/`inner_g1` | 面板命名从 1 起、面板本身从 0 起，标签没对齐 | 改为 `inner_g{generation-1}`；`camp-v10-001/002` 的旧收据保留旧标签（`per_item` 里的 qa_id 才是权威内容，核对无误） | `tests/unit/test_v10_campaign.py` |

另有一条**流程事实**（不是缺陷）：`data/v10_campaign/runs/camp-v10-001/gen3_beyond_scope/`
是超出首期范围（两代）的第三次运行残留（CLI 曾以 `--max-generations 3` 启动一次
resume）。它保留在盘上以便对账，但 `campaign.json` 的 `generation_receipt_refs` 只列
`gen1`/`gen2`，**不参与两代结论**。
