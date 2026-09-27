# v9 Skill Library 接入记录

日期：2026-09-26（P0 证据诚实性与运行环境更新：2026-09-27；P6 检索与交付记录：2026-09-27；D1/D2 用户裁定落地：2026-09-27；P4 主动图像与真实 vLLM 链：2026-09-27）
状态：S0 文档/静态导入完成；真实模型、工具闭环和演化尚未验收。

## P0：证据诚实性、Schema 身份与运行环境

起因是一次误判：把 base conda env 的 numpy/scipy 不兼容当成了 M4 缺陷，并在
"6 通过 12 失败"的 shim 结果上继续推理。实际结论是——M4 逻辑正确，问题全在环境
与记录口径。本节记录由此产生的修正。

- **运行环境预检**：`src/skill3d/env_preflight.py` 用**行为探针**（真实调用
  `cKDTree` / `rankdata` / `ndimage.label`）判定依赖是否可用，`skill3d.online.eval`
  启动时先检后跑，不可用即拒绝启动。理由：M4 子项 fail-closed 会把环境故障记成
  `scene_route=fallback_2d_only`，把工程故障伪装成场景质量。`tests/conftest.py`
  在会话建立时做同一检查，拒绝在不可解释的环境下产出报告。
- **依赖 pin**：`pyproject.toml` 的 `numpy`/`scipy` 加兼容区间（scipy ≥1.14 要求
  numpy≥2.0）。冻结环境为 numpy 2.1.2 + scipy 1.16.2。
- **环境依赖留档**：`inference_env_versions()` 纳入 python/numpy/scipy/opencv
  （§17.1"环境依赖"），此前只记模型框架。
- **`synthesis_source` 不再冒充 `vllm_ok`**：`_synthesis_source_enum` 的未知取值
  兜底由 `vllm_ok` 改为空串；`TraceRecord.synthesis_source` 由 Literal 改为 `str`
  并补 `round_trigger` / `finalization_used`（§12.2 要求三者分开记）。此前
  mock_light 的收口答案在 TraceRecord 里被记成 `vllm_ok`，且 `stub_program_ratio`
  漏计。
- **轮次原因与程序来源分离**：四处覆盖 `synthesis_source` 的写法（`finalization`、
  `forced_answer`、`forced_answer_final`）改为写入真实来源（mock_light 下即
  `mock_stub`），轮次原因改记 `round_trigger`
  （`initial`/`observation`/`error_recovery`/`finalize`）。
- **执行前拒绝按失败归因**：`_record_failed_call` 现在写
  `status="failed"`、可追踪 `result_id`、`request_digest`（与成功路径同一口径）与
  `evidence_version`。此前这些记录为 `status="ok"`、`result_id=""`，可能经
  `collect_validated` 作为"有效观测"回灌给模型。
- **EpisodeTrace Schema 身份升版到 9.0**：`EPISODE_TRACE_SCHEMA_VERSION = "9.0"`；
  旧语料（6.0）保留原身份，由 `legacy.readers.read_legacy_episode_trace` 只读解释，
  `read_episode_trace` 对非当前版本 **hard fail** —— 直接 `model_validate` 旧 JSON
  会让缺字段被默认值静默补齐，把"当时没记"读成"当时不存在"。RunManifest 的
  `schema_version` 同步到当前合同，避免 manifest 与 trace 自相矛盾。
- 新增测试：`tests/unit/test_v9_trace_honesty.py`（19）、
  `tests/unit/test_v9_env_preflight.py`（5），并在
  `tests/integration/test_online_chain.py` 的 blur_all 用例中锁定"mock 收口答案
  不得记成 vllm_ok"。

回归：`skill3d-exp` 下 `tests/` 全量 **825 passed, 3 skipped**（P0 之前为
801 passed）。`build_skill_library.py --check` 通过，S0 manifest digest 仍为
`61259a…`，active snapshot 未变。

## P1：多轮执行的账本可追溯性与轮次落盘

- **跨轮 tool 账本不再被抹掉**：`reset_user_namespace()` 只清用户命名空间与两个
  控制槽，**不再清 `tool_calls`/`tool_results`/`contract_violations`**。这三者是
  §17.1 的 Tool 层落盘事实，不是用户变量。新增 `mark_cell_start()` 与
  `cell_calls`/`cell_results` 提供逐轮视图（`ProgramExecutionTrace` 仍只报本轮），
  以及显式的 `reset_episode_ledger()`。
  此前 runner 在 yield 路径上"先存后恢复"账本的写法会被下一次 `_execute_program`
  的 reset 立刻抹掉，导致 `collect_validated` / `used_result_ids` / 级联撤销只看得到
  最后一轮；v9 §11.1 的点名回灌与 §14.1 的观测追溯因此都只对上最后一轮。
- **逐轮记录落盘**：`EpisodeOutcome`/`EpisodeTrace`/`TraceRecord` 新增 `rounds`
  （序号、触发原因、程序来源、**实际执行的程序文本与 sha256**、本轮 result_id、
  工具调用数、结束码）。记录点放在 `_execute_program` 之后这**唯一**位置，因此没有
  哪条 `continue`/`break` 分支能漏记；`round_trace_refs` 由它派生。收口轮（在 while
  之外）单独补记。
- **轮数/预算落盘**：`agent_rounds`、`yield_count`、`budget`（`max_solver_rounds` /
  `finalization_rounds` / `max_retries_per_operation` / `max_regen`）进入两个 trace。
- **恢复轮计入求解轮预算**：契约恢复分支每次 `agent_rounds += 1`，触及预算边界即
  停止恢复、直接零工具收口。此前恢复轮不消耗轮数，与 §11.2"不能靠三次重试在每轮
  之外无限添加恢复请求"冲突，且 trace 报的轮数小于实际消耗。
- 新增测试 `tests/unit/test_v9_round_ledger.py`（5）：连续两次 yield 的轮次事实与
  **跨轮** result_id 追溯、收口轮记录、恢复轮计预算、账本跨 reset 存活与逐轮视图。

回归：全量 **830 passed, 3 skipped**。

## P2：答案合同（§10.1/§4）与工具归因核验（§12）

- **规范示例此前通不过实现**：§10.2 的 `ReturnAnswer(AnswerPayload(...))` 里
  `AnswerPayload` 在 `src/` 中不存在，`ast_guard` 会以"REGISTRY 外未定义函数调用"
  拒掉它。新增 `src/skill3d/schemas/answer.py` 落地 §10.1 的
  `EpisodeStatus` / `AnswerBasis` / `AnswerPayload`，并把 `AnswerPayload` 注入
  sandbox 命名空间与控制接口白名单（`CONTROL_INTERFACE_NAMES`），同时保留"禁止
  重赋值"。
- **单位按 §4 官方合同登记，不猜**：`CANONICAL_UNIT_BY_QUESTION_TYPE` 就是 §4 那张
  "题型 → 单位"表（计数/米/厘米/平方米/选项）。历史短写
  `ReturnAnswer(value)` 走显式适配器 `parse_answer_payload`：单位按题型合同登记、
  **basis 保守记 `mixed`**（模型没声明工具贡献，无法证明完全由工具推导）、
  问题记录写明适配器版本。未知题型不猜单位，记问题。
- **`derivation` 限定为登记操作**：`DERIVATION_OPS` = 字段提取/换算/计数/排序/
  argmin/选项映射；校验 `op` 在册、引用的 result_id 必须已在 `used_result_ids`
  声明、拒绝未登记键（"不执行其中的任意代码"落到"只认名字 + 形状"）。
- **框架核验取代模型自称**：`verify_attribution` 用本 episode 台账核验声明，
  产出 §12 的六字段台账 —— `attempted/succeeded_tool_calls`、
  `observed/declared/verified/ignored_result_ids` 三分离（"调用不等于使用"）。
  无法充分证实的 `tool_derived` **降级为 `mixed` 并保留问题**，答案照常进分
  （§12：不凭归因问题否决格式合法的预测）；被级联撤销的声明证据不计入核实结果。
- **落盘**：`EpisodeTrace`/`TraceRecord` 新增 `episode_status`、`answer_basis`、
  `answer`（载荷）、`attribution`（台账 + 降级 + 问题）。`final_state` 保留历史
  碎片值供历史统计，§10.1 的三值终态由 `_episode_status_of` 显式映射。
- **顺带修掉一个 §10.2 缺口**：`_AnswerSlot` 此前没有二次提交保护，第二次
  `ReturnAnswer` 会静默覆盖首个答案；现在硬拦（`AnswerAlreadyGiven`），否则被
  记录的 `AnswerPayload` 也可能被覆盖，§12 的核验就没有意义。
- 新增测试 `tests/unit/test_v9_answer_contract.py`（22）：规范示例过 M9、
  §4 单位合同八题型齐备、短写适配不冒充 `tool_derived`、derivation 操作在册、
  六字段台账三分离、降级不否决、二次提交被拦、episode 端到端落盘。

已知边界：`used_result_ids` 仍是"全部已验证结果"的历史口径（供既有统计），
v9 §12 的"声明—核实"语义在 `attribution.verified_used_result_ids`；两者含义不同，
不应互替。`derivation` 只做静态在册校验，**实际重放**尚未实现。

## P3：授权收据（§6.4）、原因码（§6.1）与门三态（§8.2）

- **新增 `src/skill3d/schemas/authorization.py`**：`ToolAuthorizationReceipt` 带齐
  §6.4 的九个字段（`episode_id / tool_id / argument_digest / evidence_version /
  dependency_refs / allowed / reason_codes / metric_gate_result /
  decision_version`），并补 `result_id` 与参数、作用域、失效线索，使收据与
  `ToolResult` 可互相定位。
- **单一判定函数**：`tools.contract.authorize_tool_call` 按既有顺序执行与执行期相同的
  检查（作用域 → 证据 → 米制题级 → 产物），但**不抛异常**而是折成结构化决策。
  `call_tool` 先取决策、据此生成收据，再按原顺序与原错误文案执行原有四道检查 ——
  收据与"实际是否放行"必然同源，行为不变。判定失败的原因码覆盖既有四类契约错误码
  与 `tools_disabled`/`scope_denied`/`object_unbound`。
- **允许也留痕**：此前授权只以异常形式存在，成功调用没有任何"谁授权了它"的凭据，
  于是 §6.4 的"证据摘要与授权收据不能互相矛盾"无法校验。现在成功调用同样产出
  `allowed=True, reason_codes=["allowed"]` 的收据，并写进 `ToolResult.authorization`
  与 `EpisodeTrace/TraceRecord.authorization_receipts`。
- **执行前拒绝也有收据**：finalization 封锁、答后调用、scope/证据拒绝都经同一函数
  生成 `allowed=False` 的收据（`_denial_receipt`），且与"确实没有执行"一致。
- **§8.2 门三态**：`MetricEvidenceGateResult` 增加 `status`
  （pass／fail／not_applicable），`gate_passed` 放宽为 `Optional[bool]`；
  `not_applicable` ⟺ `None`，三者自洽性由 validator 强制（伪造通过仍被拒）。
  收据里非米制工具的 `metric_gate_result` 记 `not_applicable` + `gate_passed=None`
  —— 满足 §6.4"不能用默认 False 伪装失败"。
- **§6.1 原因码**：证据画像新增 `state_reasons`（`not_run / producer_failed /
  invalidated / unsupported` 四值词表，受 validator 校验，缺省为空即"原因未登记"，
  不猜）。
- 新增测试 `tests/unit/test_v9_authorization_receipt.py`（16）：收据字段齐备、
  允许/拒绝两侧都有收据、收据与决策同源、参数摘要与 `ToolResult.request_digest`
  一致、非米制门为 `not_applicable`、门三态自洽、原因码词表校验、episode 端到端落盘。

已知边界：`state_reasons` 的**词表与校验已落地，但生产者只填了少量场景** ——
哪些能力在什么条件下记哪个原因码，仍需逐个生产者接线（下一步）。

## P5：与规范直接冲突的两处修正（§17.3 核验清单）

- **§11.3「无围栏但完整程序不计解析恢复」**（规范 L429 原文）。`extract_program_source_ex`
  的最后一条分支此前 `return stripped, True`，把"只是没写围栏、内容完整可解析"的正常
  产出记成 `m8_parse_recovered`，虚增回退率并低估 `vllm_ok`。现改为返回 `False`
  （干净路径）；未闭合围栏的可解析前缀与截断恢复仍正确标记为回退。
- **§14.3 准入必须在完整 inner 面板上判定**。此前 `admit()` 消费的是
  `L3_outer_holdout` 的结果，参数叫 `outer_items` —— 用 holdout 选代会让该 holdout
  失效，且与 §16.2"outer 用于冻结方案验证"矛盾。现在准入输入改为 **L2（完整 inner）**，
  参数改名 `panel_items` 并写明 §14.3；`outer_holdout` 的结果降为**快照冻结后的独立
  验证证据**（仍落盘 `l3_outcome`，但不再参与晋升判定）。checkpoint 新增
  `admission_outcome`，中断恢复后仍能继续准入；恢复时 L3 证据损坏不再阻断准入
  （它不是门）。
- **G-31 简化阶段门与 §14.3 的冲突**：`simplified_phase_gate` 会跳过 L2，导致没有准入
  面板结果、晋升必然被 `no_admission_panel_outcome` 拒绝。新增
  `_admission_safe_levels()`：保留简化意图（L1 仍是最小切片预筛）但**始终保留 L2**，
  并记一条说明，而不是让一次真实演化跑到最后才发现没有准入证据。
- 新增测试 `tests/unit/test_v9_spec_conflicts.py`（8）。

**未完成**：§5.3 的"抽样不得按行序取每类前 N 题 + 每个预登记 qa_id 必须有结果行 +
不足 32 帧不得静默消失"（§17.3 的"少帧输入／完整评估分母"）尚未修改 —— 它要动
数据加载契约（视频存在性筛选、按 seed 抽样、排除行与分母留档），需要单独一轮。

## P5b：先冻结清单再加载输入（§5.3）

- **抽样不再按文件行序取前 N**（§5.3 明文禁止）。新增 `_sample_by_scene`：
  以 **scene** 为单位、按 `random.Random(f"{seed}:{task}")` 对 scene 列表做确定性
  洗牌后顺序取，组内行按 `qa_id` 稳定排序 —— 因此抽样结果**只由 (seed, 题型, scene)
  决定，与输入行序无关**，同 seed 完全可复现、换 seed 结果不同（有测试守住）。
- **不以视频存在为筛选条件**（§5.3 原文）。视频存在性检查从抽样段移到加载段：
  清单先按元数据冻结，再逐项加载。
- **每个预登记 qa_id 都有结果行**。加载段每次丢弃都产出排除行
  （`video_missing` / `video_undecodable` / `insufficient_frames` /
  `scene_cap_reached`），由 `exclusions` 出参带回；`eval` 侧打印并把
  `n_excluded`、`exclusions`、`denominator_preserved=True` 写进 RunManifest。
- **抽样收据落盘**（§5.3："抽样算法、scene／qa_id 清单、seed 和 hash 都落盘"）。
  `_sampling_receipt` 产出 strategy／seed／per_task_cap／qa_ids／scenes／
  `qa_id_sha256`（由清单内容可复算），`eval` 写进 RunManifest 的
  `sampling_*` 字段。
- 新增测试 `tests/unit/test_v9_preregistered_manifest.py`（11）。

回归：全量 **887 passed, 3 skipped**。

**§5 仍未完成的三项**（不与上述结论混同）：

1. **§5.2 的"用可读帧继续"**：源视频不足 32 帧时当前记 `insufficient_frames` 排除行
   （分母保住了），但规范要求的是"**保留帧身份与缺失掩码，使用可读帧继续作答**"，
   并标记 `input_degraded` 独立报告 —— 这需要可变长帧集 + 缺失掩码 + "依赖完整帧集
   的工具按条件禁用"，属独立改动。全黑/低对比不得判为"没有图片"这一条当前已满足。
2. **§5.1 的质量诊断默认关闭**：模糊／曝光／对比度／运动诊断目前**无条件计算**
   （`gates/input_gate.py`），规范要求"默认关闭；可在独立诊断实验中启用"。
3. **FrameSet 的 13 字段**：规范列出的 `schema_version / episode_id / dataset_id /
   video_id / scene_name / frame_refs / decode_status / readable_frame_ids /
   preprocessing_version` 共 9 项当前不存在（现有 7 个字段是另一套命名）。

## P5c：FrameSet 的 §5.2 字段与缓存身份

- **补齐规范点名的字段**：`schema_version / episode_id / dataset_id / video_id /
  scene_name / frame_refs / decode_status / readable_frame_ids /
  preprocessing_version` 九项全部落地，并加 validator 强制 §5.2 的自洽条件
  （数组长度匹配、帧索引唯一且有序、可读帧必须是规划帧的子集）。
- **`readable_frame_ids` 的语义**：`frame_ids` 是**规划**槽位，`readable_frame_ids`
  是真正可解码的子集。当前实现下解码失败即输入合法性硬失败，所以正常路径两者相同；
  缺省从 `frame_ids` 补齐，既保证既有构造点语义不变，也让"部分可读"（真子集 +
  `decode_status="partial"`）**可表达** —— 这是 §5.2 缺失掩码的载体。
- **源标识参与缓存身份**（§5.2："源标识及内容校验值参与缓存身份，**禁止仅凭相同的帧
  索引列表跨视频复用**"）。新增 `FrameSet.cache_identity()`，把 dataset／video／
  scene 与 `frame_set_hash` 合成缓存键；`build_frame_set` 接收并填充源标识。
  **不改写 `frame_set_hash` 的定义**（仍是帧索引内容哈希），因此既有重建缓存与
  `assert_same_frame_set` 比较不受影响。
- 新增测试 `tests/unit/test_v9_frame_set_identity.py`（10），含"同帧索引、不同视频
  必须有不同缓存身份"。

回归：全量 **897 passed, 3 skipped**。

**待接线**：`cache_identity()` 已成为可用的缓存键，但重建产物路径目前仍按
`scene_name` 分文件（`runner.py` 的 artifact 路径）—— 把 artifact 缓存切到
`cache_identity()` 属下一步。

## P5d：§5.2 异常输入（用户决定：按 v9，取代硬约束 21）

**决策记录**：`adapters/frame_set.py` 的模块 docstring 原本声明"不到 32 帧即输入
合法性硬失败（G4 帧数完整性）"，并标注为"经用户确认的设计决策，**不得擅自更改**"；
v9 §5.2 要求相反（"源视频不足 32 帧但仍有真实可读帧：保留帧身份和缺失掩码，使用
可读帧继续作答"）。两者直接冲突且会改变"什么样的 episode 算合法"（进而影响分母与
历史可比性），因此停下来请用户裁决。**用户 2026-09-27 决定：三种解法中按 v9。**

- **`uniform_frame_ids`**：不足 `n_frames` 时返回**全部可用帧**（`0..total-1`，仍唯一且
  有序），只有"一帧都没有"才抛 `FrameSetError`。既不重复填充凑数（硬约束 21 禁止的
  做法），也不改采样算法 —— 仍是时间均匀采样，只是可采样的时间轴更短（§5.2"不重复
  填充或偷偷改采样"）。
- **`build_frame_set`**：`n_frames` 如实记实际规划帧数（不再沿用名义 32），并在不足
  名义帧数时标 `decode_status="partial"` —— §5.2"不伪称完整 32 帧输入"。
- **M2 门（`gates/input_gate.py`）**：把"全部缺失或无法解码"与"部分帧无法解码"分开
  处理。前者 `input_error`（`unanswerable` + `input_degraded`，记录原因）；后者与
  "帧数不足"一律 **`proceed` + `input_degraded`**，且非法帧仍逐条留在
  `hard_fail_frame_ids`（不静默丢弃）。帧集本身不动（M2 仍是被动观测）。
- 更新 4 个断言旧政策的测试（`test_adapters` / `test_input_gate` ×2 /
  `test_v3_contracts`），并在注释里写明"v9 §5.2 取代硬约束 21"，避免后人误以为是
  测试被削弱；新增 5 条短帧契约测试。

回归：全量 **903 passed, 3 skipped**（含新增短帧测试）。

**仍未接线**：`readable_frame_ids` 已是可表达"部分可读"的载体，但**解码层尚未回填**
—— 真实路径里"部分帧读失败"仍会在 `_read_frames` 抛错（属 `producer_failed`
一侧），把可读子集与缺失索引回填到 `FrameSet.readable_frame_ids` 是下一步。
"依赖完整帧集的工具按条件禁用"也尚未实现（需要 ToolSpec 侧的声明）。

## P5e：§5.1 M2 收窄为最小输入有效性检查（质量诊断默认关闭）

规范原文（§5.1）："M2 改为**最小输入有效性检查**：文件/像素是否存在、可解码、尺寸
合法。模糊、曝光、对比度和运动质量诊断**默认关闭**；可在独立诊断实验中启用，不能删帧、
换帧、重排或终止作答。原有 `quality_weight` **默认不影响主流程**。"

- **开关**：`input_gate(..., diagnostics=False)` 与
  `annotate_frames(..., diagnostics=False)` 默认关闭诊断；关闭时不计算 blur／曝光／
  运动，逐帧 `degradation_flags` 为空、`quality_weight` 恒 1.0、`quality_score` 恒 1.0。
  合法性判定（空帧／尺寸非法／帧数）不受影响，仍按 §5.2 的分流执行。
- **配置**：`configs/config.yaml` 新增 `input_diagnostics: false`，经
  `OnlineRunConfig.input_diagnostics` 贯穿 runner 的两处 gate 调用与两处
  `annotate_frames` —— 这就是 §5.1 说的"可在独立诊断实验中启用"的入口。
- **修掉一处真实的"影响主流程"**：`runner` 把 M2 的 `quality_weight` 传给合成质量
  （`synthetic.effective_quality = overall_quality × input_quality_weight`），于是
  模糊输入会把 `scene_route` 拉成 `fallback_2d_only`。按 §5.1"`quality_weight` 默认
  不影响主流程"，这是不该存在的影响；诊断关闭后权重恒 1.0，该路径成为恒等变换。
  实测：blur_all 输入现在走 `full_3d`、正常作答、`input_degradation_flags` 为空
  —— 与 §5.2"模糊但结构有效不得被判为没有图片"一致。
- **测试更新**：5 个诊断类单测显式开启诊断（`input_gate(..., diagnostics=True)`）；
  原 `test_input_gate_blur_all_downgrades_then_forced_visual_answer` 拆成两条 ——
  `test_blur_default_does_not_change_the_main_flow`（默认口径：路由不被 M2 左右）
  与 `test_diagnostic_experiment_blur_all_still_answers`（独立诊断实验口径：仍必须
  作答，并继续守住 P0 的 trace 诚实性回归锁：mock 收口答案不得冒充 `vllm_ok`）。
  轮次账本与 blur_some 用例同属诊断实验口径，显式开启诊断。

回归：全量 **904 passed, 3 skipped**。

## P5f：§5.2 缺失掩码回填（部分帧解码失败 → 用可读帧继续）

- **`_read_frames` 不再因单帧失败让整题失败**：签名改为
  `→ (可读帧像素, 缺失帧号)`。读失败只进缺失清单，返回的像素是**规划序下的可读子集**
  —— 不插占位帧、不重复邻近帧（§5.2"不重复填充或偷偷改采样"）。只有"一帧都读不出来"
  才由调用方按 `input_error` 记排除。
- **加载段顺序调整**：先按规划帧号读出像素、得到可读子集，再据此构造 FrameSet
  （`readable_frame_ids`），使掩码来自真实解码结果而不是假设。
- **部分可读的样本照常产出 episode**：这是关键 —— §5.2 要求"使用可读帧继续作答…
  **不从评分分母静默删除**"，所以它**不**进排除清单；缺失掩码单独登记到抽样收据的
  `partially_readable`（含 `n_planned` / `n_readable` / `missing_frame_ids`）并写进
  RunManifest，即规范要求的"独立报告"。`input_degraded` 标记由 M2 在帧数不足时自动加上。
- **`build_frame_set` 接受 `readable_frame_ids`**；`decode_status` 在"帧数不足**或**
  部分可读"时为 `partial`。可读集必须是规划帧的子集（validator 强制）。
- **M8 的帧数断言改用实际可读帧数**（`len(readable_frame_ids)`，3 处）：否则部分可读的
  帧集会在提示词层被自己的对齐检查炸掉。
- **成对比较加入可读集检查**（§5.2"成对实验必须复用同一**实际可读**集合"）：
  `assert_same_frame_set` 在规划帧哈希之外，额外要求 `readable_frame_ids` 一致 ——
  规划帧相同但缺失掩码不同时，两条臂实际看到的东西并不相同。
- 新增测试 `tests/unit/test_v9_partial_frames.py`（10），含用真实 mp4 验证
  "缺失帧号如实记录、像素里不插占位帧、掩码完整"。

**未完成**：§5.2 同段的"**依赖完整帧集的工具按条件禁用**"按用户 2026-09-27 指示
**不做**（已从中删除，不再列为待办）。

## P5g：artifact 缓存切到缓存身份（§5.2）

- **`artifact_path(..., frame_set=None)`**：传入 frame_set 时文件名用
  `<scene>__<cache_identity[:16]>.json`（源标识 + 帧集内容哈希），不传时保持历史命名。
  实测：同名 scene、不同视频现在落到不同文件（`s1__22f88d07….json` vs
  `s1__36a28582….json`），纯 scene 名会互相顶用的隐患消除。
- **`resolve_artifact_path(...) → (路径, 是否旧命名)`**：优先缓存身份路径，缺失时回退
  历史命名并如实返回 `used_legacy=True`。这是 §17.2 要求的迁移形态 —— 切缓存键**不得**
  让已算好的旧产物凭空失效（那会白烧一遍重建并改变对比基线）。
- **三处调用点一起切**（写方与两个读方必须同源，否则读不到自己的产物）：
  `plan_scene_jobs`（判断跳过）、`run_jobs`（落盘）、`panel.py`（A/B 两臂复用）。
- **消掉一处重复定义**：runner 内联的 `f"{scene}.json"` 与 `reconstruction.run.artifact_path`
  本来各写了一份同样的路径约定；现在 runner 委托同一函数（`_artifact_json_path`），
  并新增 `_resolve_existing_artifact` 做身份/旧命名两级定位。切缓存键时两边失配的风险
  由此消除，并有测试守住"读方路径 == 写方路径"。
- 更新 1 个硬编码旧文件名的测试（改为用同一解析器定位，并断言身份可复算）；
  新增 5 条缓存身份测试。

回归：全量 **919 passed, 3 skipped**。

## P3b：§6.1/§6.4 原因码的生产者接线

接线时发现一处**真实的误报**：`M5EvidenceSummary` 没有"是否运行过"的字段，于是
"M5 从未运行"会被 `detection_capability` 报成 `zero_detection_after_retry` ——
那暗示"重试后仍零检出"，而实际根本没跑过。这正是 §6.1"`unavailable` 必须通过原因码
区分**未运行**／**生成失败**"要消除的混同。

- **`M5EvidenceSummary.m5_ran`**：区分"未运行"与"运行了"。两个生产者（真实在线链
  `runner` 的 M5 汇总、合成路径 `synthetic`）在构造摘要处置 `True` —— 该函数被调用
  即说明 M5 跑过。
- **三类原因彼此可区分**：`m5_not_run`（未运行）／`detector_fault`（生产者失败）／
  `zero_detection_after_retry`（**成功执行但无匹配目标**的空检出状态，§6.4 明确要求
  单独表达）。`track_consensus` 同样拆开 `m5_not_run` 与 `no_track_statistics`。
- **`build_evidence_profile` 填 `state_reasons`**，原则是**只填代码确知的**：
  - `geometry_3d`：质量未计算 → `not_run`；
  - `metric_scale`：融合 `not_run` → `not_run`；融合 `failed` → `producer_failed`；
  - `object_detection`：未运行 → `not_run`；故障 → `producer_failed`；运行了但零检出 →
    `unsupported`（"成功但无匹配目标"这一类）；
  - `track_consensus`：未运行 → `not_run`。
  说不清是"没跑"还是"跑失败"的场景（如 M4 主门未过、world frame 退化）**留空**
  （= 原因未登记），不猜一个码填上。
- **级联撤销记 `invalidated`**：`downgrade_profile` 在降级能力时同步写入该原因码，
  与"没跑""跑失败"区分开（§6.4 四类之一）。
- 更新 4 处测试替身（构造即代表"运行过 M5"，与真实生产者一致）；空构造保留原语义
  （"未运行"，断言仍为 `unavailable`，且现在理由正确）。新增
  `tests/unit/test_v9_evidence_reasons.py`（9）。

回归：全量 **928 passed, 3 skipped**。

**边界（已于 D2 解决）**：`state_reasons` 当时只覆盖可确知场景；"运行成功但质量门未过"
这类**质量判定失败**不属于 §6.4 的四类，代码里无从表达，故留空。是否扩展词表交由用户
裁决 —— 用户 2026-09-27 裁定**扩展**，见下面「D2」。

## P6：§13.5/§13.6 检索记录、交付身份与重检索

规范原文（§13.6）：

> 每次检索记录规范题型、evidence_version、候选及过滤原因、排序分数、选中版本、
> 实际交付版本与正文 hash、配置版本。区分"检索选中但未送达模型"与"已交付"。
> 记录 `retrieved_skill_versions`、`delivered_skill_versions`、
> `declared_selected_skill_versions`、可观察的程序使用线索，以及每轮短方法摘要。
> 模型自称选择要与程序和观察交叉检查，不能当作方法成功或因果贡献的充分证明。

规范原文（§13.5）：

> top-k、排序规则和方法上下文上限在演化开始前写入配置并冻结。
> 上下文放不下时先减少完整条目，不能截掉检查、局部条件或来源后仍称"完整 Skill 已交付"。
> 证据更新后可在同一快照中重检索，更新实际交付记录；不在 episode 中发布新库。
> 超过服务限制的候选在静态检查中拒绝或修订。

改动前的事实：检索只落 `retrieved_skills`（命中列表）与 `selected_skill_semvers`，
被拒候选的原因只进日志；"选中"被当成"模型见过"；top-k／排序权重是模块常量（不来自
配置）；方法正文在 Jinja 模板里逐字段拼接，交付 hash 无从谈起；重检索没有第二个调用点。

- **交付文本唯一事实源**（新增 `src/skill3d/skills/delivery.py`）：
  `render_skill_entry()` 产出单条方法的完整正文，`plan_delivery()` 按**冻结上限**
  贪心取完整条目、放不下的**整条丢弃**（记 `context_cap_exceeded`），
  `skill_content_sha256()` 给出交付身份。prompt 渲染与 trace 里的
  `delivered_content_sha256` 都来自这一处 —— 否则"正文 hash"只是"某个看起来像
  正文的字符串"的 hash，证明不了模型收到的是哪段文本。
  `check_service_limit()` 是 §13.5 末句（超服务限制→静态检查拒绝）的判据。
- **检索策略冻结**（新增 `src/skill3d/routing/retrieval_policy.py` +
  `configs/config.yaml` 的 `retrieval:` 段）：`top_k`／`rerank`／`candidates`／
  `rank_weights`／`method_context_max_chars` 全部来自配置，读成不可变
  `RetrievalPolicy`；`version()` 是人类标签，`sha256()` 是**内容摘要**（改一个字符
  摘要就变，标签无法抵赖内容）；`source` 区分 config/default —— 缺键回退默认值时
  如实记 `default`，不冒充"已冻结配置"。未登记权重键直接报错（写错的权重不许被
  静默忽略）。`RunManifest` 里同时落标签、摘要、top-k、权重、上限、排序规则名，
  并在启动时打印实际生效的策略。
- **检索记录 Schema**（新增 `src/skill3d/schemas/retrieval.py`）：`SkillRetrievalRecord`
  带齐 §13.6 点名字段（规范题型／evidence_version／候选及过滤原因／排序分数／
  选中版本／交付版本与正文 hash／配置版本／快照身份），`SkillCandidateRecord`
  逐候选记 `reason_code`（词表：题型分区、签名、米制门三种失败、gate 结果缺失、
  非 active、未进 top-k、题型不可知）、名次、分数、正文 hash。
  **每次检索的所有候选都留痕**（包括被题型分区拦下的）—— 此前这些原因只写进日志，
  trace 里无法回答"这条 Skill 为什么没被选中"。
- **四态分离**（§13.6）：`n_skills_offered`（produced）→
  `eligible`（过硬条件）→ `retrieved_skill_versions`（**检索选中**＝top-k）→
  `delivered_skill_versions`（**实际送达模型**）→
  `declared_selected_skill_versions`（模型自称，仅线索）。
  自洽性由 validator 强制：`retrieved ⊆ eligible`、`delivered ⊆ retrieved`、
  `delivered ⟺ delivery_reason="delivered"`、delivered 必须带正文 hash。
  交付渠道只有 `model_request`（请求已发出且模型有返回）才算已交付；
  `request_failed`／`not_sent` 一律记未交付。
- **交付点接线**（`runner.py`）：`_build_prompt_ex()` 返回 `(prompt, plan)`，
  `_synthesize` 在 `client.chat` 返回后 `mark_delivered()`，失败则
  `mark_request_failed`；`_record_skill_delivery` 把交付事实并进**最近一次**检索记录
  （含每轮短方法摘要与使用线索）。C0／mock_light／未配置客户端＝没有请求，
  一条都不算交付。`program.skill_semver_used` 由"检索选中集"收紧为"**实际交付集**"
  （此前被上限丢掉的方法也会被记成"用过"）。
- **使用线索**（§13.6"可观察的程序使用线索"）：`declared_in_program`（程序文本里
  字面出现的方法）与 `template_tool_overlap`（模板点名的 Tool ∩ 程序里的 Tool 名）。
  字段名与注释都写明**只是线索**，不是方法成功或因果贡献的证明；模型自称进
  `declared_selected_skill_versions`，与"已交付"分列，两者不互替。
- **证据更新后重检索**（§13.5）：M10 级联降级（能力 `available→degraded`）后，
  在**同一快照**内重检索（`cfg.skills` 不变，不加载任何新库），追加
  `trigger="evidence_update"` 的检索记录并更新实际交付记录；第一条记录的交付事实
  保留不被覆盖。`evidence_states` 记三值快照 —— `profile_version` 是**合同版本**，
  降级前后不变，靠版本号看不出"证据真的变了"。
- **超限候选在静态检查拒绝**（§13.5/§14.4）：
  `library.static_check_skill_spec`（正文长度 + 单题型）接进
  `manage_skill_library.py validate-candidate`（不合格记 `quarantined` 且非零退出）
  与 `promote_atomic` 的 `strict_skill_specs` 路径（**切换 active 指针之前**拒绝）。
  此前 `validate-candidate` 只回显 `"status": "validated"`，什么检查都没做。
- 模板版本 `program_synth_v7 → v8`（方法条目改由交付计划渲染，其余口径逐字保留）。
- 新增测试：`tests/unit/test_v9_retrieval_records.py`（19）、
  `tests/unit/test_v9_candidate_service_limit.py`（6）、
  `tests/integration/test_v9_skill_delivery.py`（5，走真实 runner：mock_light 全链、
  real 模式 `_synthesize`、M10 级联降级后的重检索）。

**未完成且不并入上述结论**（P6 的边界）：

1. **尚无真实 run 产出过带检索记录的 9.0 语料**：上面的记录只在合成数据 + 假客户端
   （`_FakeClient`）路径上验证过。真实 vLLM／真实 episode 的 `delivered_*` 仍待验收。
2. "已交付"只证明**请求发出且模型有返回**，不证明模型读懂了方法；`declared_*` 是
   程序文本里的字面自称。二者都不能当作方法因果贡献的证据（§13.6 明文）。
3. 配置里的 `top_k=3` 是**当前代码口径**的登记值。§13.5 说"开发起点可取 top-k=1"，
   我没有擅自改起始值（S0 每题型只有一条，两者行为等价）。**用户 2026-09-27 裁定：
   保持 3**，见下面的决策记录表。
4. 策略的"冻结"目前是"写入配置 + 单点读取 + 摘要落盘"：配置仍可被改动，只是
   manifest／检索记录里的摘要会随之变化而可被看出来。**没有**单独的冻结校验脚本或
   写保护。
5. §14.4 的 `quarantine` 目前只是 CLI 的判定与非零退出，**没有**把不合格候选落成
   `candidates/quarantine/` 记录文件（§14.4 的完整 quarantine 记录类型仍未实现）。
6. 交付状态词表里"硬条件未过"与"排序未进 top-k"共用 `not_selected`；要区分看
   `reason_code`（`question_type_mismatch` / `evidence_signature_unmet` / …）。
7. `method_summaries[].round` 是"M8 生成该程序时已消耗的求解轮数"的尽力口径：
   收口轮在 while 之外生成，其轮号与 `rounds[]` 里 `index=agent_rounds+1` 可能差 1。
8. 静态检查现在**拒绝多题型候选**（§14.4"单题型"）。`SkillSpec` 本身允许同族多题型，
   若将来确有跨题型方法，需要先决定"放宽静态检查"还是"拆成多条候选"。
9. 交付只覆盖**程序合成**的模型请求；`inspect_frames`／`detect_objects` 的主动图像
   回灌（P4）仍未实现，图像相关的方法上下文也没有交付记录。

回归：全量 **958 passed, 3 skipped**（P6 之前为 928 passed）。

## D2：§6.1/§6.4 原因码扩展为五值（用户 2026-09-27 裁定）
**决策记录**：§6.4 给的是四类 —— `not_run / producer_failed / invalidated /
unsupported`。"**运行成功但质量门未过**"（M4 主门未过、world frame 置信低、尺度自洽
低于阈值、检出稀疏、track 统计超阈值）**不属于**这四类：生产者也跑了、也没报错，只是
数值不达标。此前代码里无从表达，只能留空 —— 审计读到的是"没记原因"，而不是"跑了但
不达标"。请用户裁决后，**用户 2026-09-27 裁定：扩展词表**。

- **新增第五值 `quality_gate_not_passed`**（`schemas/evidence.py` 的
  `UNAVAILABLE_REASON_CODES`），与 `producer_failed`（生产者自己出错）严格区分。
  注释里写明这是对规范四类的**补足**，不是替换；两条原因码不得互替。
- **生产者接线**（`reconstruction_gate/evidence_profile.py`）：
  - `geometry_3d`：质量未计算 → `not_run`；质量算了、值有限、M4 主门未过 →
    `quality_gate_not_passed`；
  - `metric_scale`：融合 `not_run`/`failed` 照旧；**融合 `success` 但六项子条件未全过**
    → `quality_gate_not_passed`；
  - `world_frame`：`degraded`（估计器跑出来了但置信低）→ `quality_gate_not_passed`；
    `unavailable` **仍留空**（分不清"从未估计"与"估计失败"，不猜）；
  - `object_detection`：未运行 → `not_run`；故障 → `producer_failed`；零检出 →
    `unsupported`（§6.4 的"明确空检出"）；**有检出但低于稀疏阈值** →
    `quality_gate_not_passed`；
  - `track_consensus`：未运行 → `not_run`；跑了但没有可统计 track → `unsupported`；
    **碎片化/重复率超阈值** → `quality_gate_not_passed`；
  - `object_grounding`：明确未命中 → `unsupported`；点名物校验没做 → `not_run`。
- 更新 `tests/unit/test_v9_evidence_reasons.py`（14，新增 5 条）：第五值可表达、
  与 `not_run`/`producer_failed` 不塌成同一个、`unavailable` 的 world frame 仍留空。

**未完成**：`grounding_filled`（走的是补绑路径）与"quality 标记为 computed 但值为
NaN"两种情况**仍留空** —— 前者是"用另一条路径拿到结果"，后者可能是数据缺失的约定
取值而非生产失败，都不属于"质量门未过"，我不替它们猜原因码。

## D1：§5.2 可重试加载 = 换来源/换副本（用户 2026-09-27 裁定）

规范原文（§5.2）："全部指定图像缺失或无法解码：`input_error`，记录原因；
**可重试加载**，不生成伪答案。"

**决策记录**："可重试加载"没写清重试什么。对**同一个文件**反复解码只会得到同一结果，
等于没重试，因此 2026-09-27 请用户裁决，**用户裁定：换来源/换副本**。

- **候选顺序**（`vsibench_loader.video_source_candidates`）：主约定
  `<root>/<dataset>/<scene>.mp4` → 同目录同名副本（`.mkv/.avi/.mov/.webm`）→
  每个备用根（`paths.raw_video_fallbacks`，默认空）：目录式镜像
  `<fb>/<dataset>/<scene>.mp4` 与平铺式镜像 `<fb>/<dataset>_<scene>.mp4`。
  **只有主来源失败才会用到后面的候选**，主来源正常时行为与改动前逐字一致。
- **加载段改为逐个来源尝试**（`episode_source._load_from_sources`）：文件缺失、
  视频打不开/0 帧、规划帧全部读不出，都会记下该来源的结果并**换下一个副本**；
  部分帧可读仍按 §5.2 用可读帧继续（P5f 的口径不变）。
- **全部来源都失败**才记排除行 `input_error`，原因按"走得最远"的失败给出：
  `video_missing` / `video_undecodable` / `insufficient_frames`（沿用既有词表，
  主来源是唯一候选时与改动前完全相同）。
- **用了副本必须留痕**：抽样收据新增 `source_retries`（哪一题、主来源、实际用的来源、
  每个来源的尝试结果），`eval` 写进 RunManifest 的 `source_retried_samples` /
  `n_source_retries` 并在 stderr 报出来。
- **副本不得与主来源共用重建缓存**（§5.2"源标识参与缓存身份"）：
  `video_id_for` 在从副本加载时给 `video_id` 加来源标签（`<scene>@<来源目录名>`），
  主来源保持原样 —— 于是副本算出的 artifact 落在不同的缓存键上，历史缓存键不失效
  （§17.2）。
- 新增测试 `tests/unit/test_v9_retryable_loading.py`（9）：候选顺序与"不隐式启用镜像"、
  `video_id` 标签、主来源缺失→平铺镜像、主来源不可解码→同目录副本、全部失败的原因码、
  收据只在真正换来源时写入、端到端（含 `video_id` 进 FrameSet）。
- **本机实测**：288/288 scene 的主来源**都齐备且可解码**，因此该重试路径**当前不触发**；
  274 个 scene 在主来源之外另有副本（`VSI_videos` 平铺镜像），一旦主来源损坏就会被
  换用并留痕。配置里的 `raw_video_fallbacks` 只在这些失败场景下生效，不影响现行结果。

**未完成**：副本的"内容校验值"没有参与缓存身份（§5.2 说"源标识**及内容校验值**参与
缓存身份"）—— 现在参与的是源标识（`video_id` 标签）+ 帧集内容哈希，**没有**对视频
文件本体做哈希比对；两个不同副本若被同一个根目录下的不同文件名覆盖，无法从身份上
区分。是否要引入视频文件级校验值（代价：每次加载读一遍文件）留给用户定。

## 决策记录：三项悬置问题的裁定（用户 2026-09-27）

| 问题 | 裁定 | 落地 |
|---|---|---|
| §5.2"可重试加载"的语义 | **换来源/换副本**（不是重新解码同一文件） | D1（本节上一条） |
| §6.1/§6.4 原因码是否扩展 | **扩展**：新增第五值 `quality_gate_not_passed` | D2 |
| §13.5 检索 `top_k` 起始值 | **保持 3**（不改 1） | 无需改动；配置继续登记 `top_k: 3`，内容摘要 `4b497fdb…` |

回归：全量 **972 passed, 3 skipped**（D1/D2 之前为 958 passed；新增
`tests/unit/test_v9_retryable_loading.py`（9）、`test_v9_evidence_reasons.py` +5）。

**D1/D2 的两处边界**（不并入上面的完成结论）：

- 副本的**内容校验值**尚未参与缓存身份（§5.2 说"源标识**及内容校验值**"）—— 现在参与
  的是源标识（`video_id` 标签）+ 帧集内容哈希，没有对视频文件本体做哈希。是否引入
  文件级校验值（代价：每次加载读一遍文件）需用户定。
- 原因码仍有两格**故意留空**：`object_grounding` 的"走补绑路径拿到结果"、以及
  "quality 标记 computed 但值为 NaN"（可能是数据缺失的约定取值，不一定是生产失败）。
  两者都不属于"质量门未过"，不替它们猜码。

## P4：§9.4 主动图像与补检（真实 vLLM 链已验收）

规范原文（§9.2 最小工具面）：

    | `inspect_frames(frame_ids, boxes=None)` | image_2d；允许 degraded |
      从冻结帧集中查看/裁剪真实图片；返回图像引用和变换，**不能新采样视频帧** |
    | `detect_objects(frame_ids, categories)` | image_2d 及健康检测服务 |
      返回候选、置信度和执行状态；**服务健康属于执行前置条件**，不要求已有检测成功 |

规范原文（§9.4）：

    「`inspect_frames` 和 `detect_objects` 必须验证真实生产调用、真实结果与后续模型
    输入；注册表中存在其他几何工具不能代替这两项。新检出结果由框架更新当前 episode
    对象记录、证据版本和工具可用性；**agent 不能直接修改共享状态**。」
    「裁剪必须来自**同一冻结 FrameSet**，并记录源帧、像素范围和变换。模型服务的图像
    数量／像素上限必须涵盖原图与派生图；**不得静默丢图**。可使用带明确映射的图像布局
    或分次观察，实际传入图像、缩放和 token 成本完整记录。**仅输出服务器文件路径不算
    模型已查看图片**。」
    「观察链分开记录 `produced / delivered / observed`：产物已生成不代表进入请求；
    模型收到实际内容并完成本轮响应后，才可记为 observed，且这仍不证明模型正确理解。
    …超出模型图像／token 服务上限时，使用事先声明的布局或分批观察方案；**未交付的
    材料保持 unobserved**，不静默丢原图或派生图。」

**动手前先探了服务**：8100 上确有 vLLM 在听（`qwen3vl-8b-r0`，max_model_len 32768）。
实测该服务的图像上限是 **32 张/请求**（发 33 张 → `400 At most 32 image(s) may be
provided in one prompt`），所以"声明布局"不是纸面要求，而是必须遵守的硬约束。

- **两个工具真的注册并实现**（新增 `src/skill3d/tools/vision_tools.py`）：
  `inspect_frames` 只从**本 episode 冻结帧集**取像素（整帧或裁剪，越界/过小的框报
  `DomainValueError`），缩放后登记进图像账本，返回 `image_id` + 源帧/框/尺寸/scale +
  内容哈希；`detect_objects` 调本地开放词表检测服务，返回 `status ∈ {ok, fault, empty}`
  —— **服务故障记 fault，不伪装成空检出**（§9.1）；新检出由**框架**并入 episode 对象
  记录（`grounding_status="tool_detection"`，只记 2D，不写伪 3D 质心/包围盒）。
  `fallback_2d_only` scope 下这两个工具仍然暴露（重建失败不收回"看原图"的能力）。
- **句柄持有冻结帧集与图像账本**（`SceneHandle._set_frames` / `_set_image_ledger`；
  写入口全是下划线私有 → `ast_guard` 禁下划线属性，模型无法直接改共享状态）。
  账本 `ImageLedger`（新增 `src/skill3d/tools/image_ledger.py`）是 episode 内唯一实例，
  随逐题 scope 重派生一起搬走（不丢已产出的图）。
- **图像穿过 `YieldObservations` 进入下一次请求**：yield 后按"点名结果产出的图 → 该结果
  payload 里的 image_id → 本轮新产出且未交付的图"三级解析；`_synthesize` 把派生图装进
  **实际发出**的请求，并在 prompt 里附**明确映射**（`## 本轮图像清单`：第 k 张 ↔
  image_id/源帧/裁剪框/尺寸/scale）。
- **声明布局**（新增 `configs/config.yaml` 的 `active_vision:` 段）：布局名
  `derived_plus_originals_v1` —— 派生图优先占位（`max_derived_images=8`），其余名额按
  帧序补原帧，总数 ≤ 服务上限（32）。没有派生图时退化为 `originals_only`
  （**第 1 轮行为与改动前逐字一致**）。被省略的原帧已在更早请求里交付过、记进
  `omitted_originals`；被省略的派生图保持 unobserved 并写明原因 —— 两者都不静默消失。
- **三态如实记录**：`produced`（工具产出像素并登记内容哈希）→ `delivered`（**请求发出
  前**标记"进了请求"）→ `observed`（**请求返回响应后**才标记）。请求失败 → 本轮已交付
  的图保持 unobserved 并写明失败原因；episode 结束时 `note_undelivered()` 给"产出过但
  从未交付"的图盖章写明原因。收口/恢复轮也会带上本轮待交付的派生图。
- **token 成本**：`VLLMClient` 记录每次请求的真实 `usage`（含图像 token），逐轮写进
  `ImageRound.prompt_tokens`；另有按 Qwen-VL 28×28 patch 口径的**估算值**（口径显式
  标注）。实测第 2 轮 32 张图（含 1 张裁剪）= 13850 prompt tokens。
- **落盘**：`EpisodeTrace`/`TraceRecord` 新增 `image_ledger`（三态 + 每轮清单 + 统计），
  首轮合成的 `first_synthesis.image_round` 记当轮布局/张数/token；工具面版本升到
  `tool-face-v9`（工具集合变了，trace 里的 `tool_face_version` 必须对得上）。
- 新增测试：`tests/unit/test_v9_active_vision.py`（14）、
  `tests/integration/test_v9_active_vision_chain.py`（4）。

**真实 vLLM 验收**（`scripts/run_active_vision_acceptance.py`；receipt
`data/p4_acceptance.json`；场景 `7b6477cb95` / qa 2487，真实视频 + 复用真实 VGGT artifact）：

| 轮 | 布局 | 图数 | 已交付 | 已观察 | prompt_tokens |
|---|---|---|---|---|---|
| 1 | originals_only | 32 | false | false | null |
| 2 | derived_plus_originals_v1 | 32（1 裁剪 + 31 原帧） | **true** | **true** | **13850** |

裁剪图 `img-crop-001-20797db2`：源槽位 8、框 `[0,0,96,96]`、内容哈希、
`delivered_rounds=[2]`、`observed_rounds=[2]`。**第 1 轮是脚本程序**（为了确定性地触发
裁剪链），脚本模式会把该轮的交付/观察标记**按事实撤回**（`honesty_note` 写明
"第 1 轮由脚本程序触发（未发真实请求）→ 该轮交付/观察标记已撤回；第 2 轮起的请求、
图像与程序均为真实"）。第 2 轮是**真实模型请求**：模型真的收到了那张裁剪图，并写出了
下一段程序（截图见 receipt 的 `n_images_per_request` 与 `derived_images`）。

**同类真实验证（4 条，全部真实模型）**：qa 2487/2485/2484/2425 —— 真实模型**都**直接
作答、**没有**主动调用 `inspect_frames`/`detect_objects`（`n_derived_produced=0`）。
所以本轮的证据是"**当**发生裁剪时，crop 真的进了下一次请求、下一段程序真的是模型写的、
三态与 token 成本真的落盘"；**不**宣称"真实模型会主动去裁剪"。后者属于提示词／方法
层面的问题，作为 P7（真实演化）的输入记在这里。

**P4 期间发现并修掉的四个真实缺陷**：

1. **帧身份口径错**（真实链路暴露）：句柄按 `FrameSet.frame_ids` 的**物理源帧号**
   （真实视频里是 0/116/233…）登记帧，而模型与其它工具都按"第几张"（槽位 0..31）指帧 ——
   模型写 `inspect_frames([8])` 直接 `KeyError`。现在账本按**槽位**登记，物理源帧号另记
   `source_frame_index`（审计"这张图来自视频哪一帧"）。
2. **工具参数错误被归成运行时崩溃**：越界帧号/非法框原本抛裸 `KeyError`/`ValueError`，
   内核记成 `violation_run`，模型拿不到可读原因。改为 `DomainValueError`，恢复层能把
   "帧槽位 99 不存在"回灌给模型。
3. **收口/恢复轮丢掉刚产出的观察**：预算边界上的 finalization 请求不带派生图，模型
   "要求看某块区域"的让出白丢，且那张图连"未交付原因"都没记。现在收口/恢复轮一并带上
   本轮待交付的派生图；episode 结束时仍未交付的图盖章写明原因。
4. **工具描述里点了被隐藏工具的名字**（既有测试抓到）：`inspect_frames` 的描述写了
   "与 reproject … 口径一致"，在 2D-only scope 下会把一个**不可用**的工具名塞进 prompt，
   诱导模型调用它。描述改为不点其它工具名。

**未完成且不并入上述结论**（P4 的边界）：

1. §9.4 要求新检出结果更新"对象记录、**证据版本和工具可用性**"：现在只做了**对象记录**
   （并入句柄，后续工具可见）；**没有**在 episode 中途改写 `EvidenceProfile` 或据此放开
   工具（那是证据/授权策略变更，需要单独裁决），只在工具返回值里如实记状态与端点摘要。
2. **补检后的跟踪与绑定未实现**：新检出是 2D 记录，没有跑 SAM2 传播/跨帧关联，也不产出
   mask/点云；因此新检出对象无法参与距离/尺寸类工具（几何工具按空点集 fail-closed，
   不会给假值）。
3. `contact_sheet` 这个图像种类（多张派生图拼一张）在 schema 里预留但**没有任何布局用到**；
   当前派生图逐张交付（受 `max_derived_images` 与 32 张上限约束）。
4. 真实模型**不主动**使用主动图像工具（4/4 条真实 episode 直接作答），因此"主动观察回灌"
   目前只有"被触发时正确"的证据，没有"自发使用"的证据。
5. `inspect_frames` 的裁剪入参**只收像素坐标**；VLM-normalized-1000 ↔ 像素的换算只在
   `detect_objects` 的返回里做了（`bbox_norm1000`），模型传归一化坐标会被拒（描述写明了口径）。

回归：全量 **990 passed, 3 skipped**（P4 之前为 972 passed）。


## 已落地

- `skill_library/skills/<name>/SKILL.md` 保存八个 S0 方法的唯一编辑源。
- `scripts/build_skill_library.py` 验证 bundle 路径、UTF-8 内容、SHA-256、front matter、固定章节、证据条件、步骤、来源和八题型唯一性。
- `src/skill3d/skills/source_compiler.py` 是文档源格式到当前运行 `SkillSpec` 的显式适配器。附件导出仍保留 `schema_version=8.0` 身份，不机械改名为 v9。
- `skill_library/generated/` 保存严格的当前运行规格；`generated/index.json` 保存源/派生哈希和编译器版本。
- `skill_library/snapshots/` 保存 S0 不可变快照和 active pointer。在线 loader 只读取 pointer 指向快照里的 inline `spec_content`，不扫描源文件、候选或 future 候选。
- `skill_library/manifests/`、`validation/`、`candidates/` 和 `candidates/future/` 已建立，后续候选必须记录父版本、题型、来源划分、经验关系、patch 和静态收据。
- `src/skill3d/skills/library.py` 提供 `SkillLibraryCandidate` ↔ `CandidateRevision` 的显式适配器；转换会严格验证 runtime `SkillSpec` 并保留 source/generated/manifest/record provenance。说明性 record 不会被当作在线 payload。
- `scripts/manage_skill_library.py` 提供普通增量 source 编译、future candidate staging、candidate validation 和显式 atomic promotion；future 文件本身不会改变 active pointer。
- 在线与离线默认路径已统一到 `skill_library/snapshots`，不再出现 promotion 写入 `data/skill_registry`、在线读取 `data/active_snapshot.json` 的分裂默认值。
- v9 在线预算字段已接入 `OnlineRunConfig`/runner：`max_solver_rounds`、`finalization_rounds` 和 `max_retries_per_operation` 是归一化主字段；旧 `max_agent_rounds`、`reserve_final_rounds`、`max_recovery` 仅保留构造兼容，未再引入独立 `max_tool_calls` 预算。
- active snapshot 的 `manifest_hash` 会在标准 online eval 路径进入 `EpisodeTrace`/`TraceRecord`；loader 会在 library manifest 可用时重新计算 canonical manifest digest，登记值不匹配则 fail-closed 为空 digest。

## 适配边界

当前仓库的 `skill3d.schemas.SkillSpec` 仍是 v6 兼容运行模型，字段包含 `description`、`call_graph_template`、`required_evidence_signature` 和工具调用模板，但不直接承载文档源中的全部 `steps`、`limitations`、`provenance` 字段。编译器将完整方法保留在 `SKILL.md`，将目标、适用条件、执行约束、检查和局部证据条件编入运行描述/模板，并把来源、版本及哈希放入库 manifest。这是显式迁移，不是静默丢弃方法内容。

初始 Skill 只有 `image_2d >= degraded` 的整体检索条件。几何、尺度、world frame、对象绑定和 temporal 条件保留为局部方法条件，由运行时 ToolSpec、EvidenceProfile 和题级授权门决定；米制工具的 gate 不会被方法源文本放宽。这样重建或尺度失败时仍保留有图必答的视觉路径。

## 未完成且不能由本收据宣称

- 当前 active S0 尚未完成真实在线模型/工具 episode 验收。
- 尚未完成至少两轮由真实命中经验驱动的候选演化，也没有晋升或 E−S0 性能结果。
- 主动图像与补检（P4）**已实现并有真实链路 receipt**，但边界仍在：真实模型不会主动
  去裁剪（4/4 条真实 episode 直接作答）、补检后的跟踪/绑定未做、新检出不改写
  EvidenceProfile/工具可用性 —— 见「P4」小节末尾的 5 条未完成项。
- `EpisodeTrace.schema_version` 升到 9.0 后，**尚无任何真实 run 产出过 9.0 语料**；
  9.0 只是当前合同的身份，不代表真实运行已验证（P6 的记录同样只在合成/假客户端路径
  上验证过）。
- `promote_atomic` 的旧测试允许非 JSON 模板作为历史管道候选（兼容默认
  `strict_skill_specs=False`）。生产发布路径（`manage_skill_library.py promote`）已强制
  严格 `SkillSpec` + §13.5 服务限制静态检查，但旧测试夹具**不能**当作 v9 发布证据。
- 以下两条在后续轮次已补上，此处保留原记录以免"当时没记"被当成"当时不存在"：
  - `ToolAuthorizationReceipt`（§10.1/§6.4）已由 **P3** 实现（允许与拒绝两侧都留收据）；
  - 逐轮 `ProgramRound` 事实（序号/触发/程序文本与 sha256/本轮观测）与
    `agent_rounds`/`yield_count`/`budget` 已由 **P1** 落盘，不再是"只在 runner 内存里"。
    图像布局与 `selected_image_refs` 仍未落盘（随 P4 一起做）。
- P6 自身的边界（尚未真实运行、交付≠因果贡献、quarantine 未落文件等 9 条）见
  「P6」小节末尾的"未完成且不并入上述结论"。

## 接入命令

```bash
EXP=/home/cvailab/anaconda3/envs/skill3d-exp/bin/python   # 冻结实验环境，必须用它
PYTHONPATH=src $EXP -m pytest tests/ -q                    # 全量套件
PYTHONPATH=src $EXP scripts/build_skill_library.py --check
PYTHONPATH=src $EXP scripts/manage_skill_library.py --help
PYTHONPATH=src $EXP -m skill3d.online.eval --active-snapshot skill_library/snapshots/active_snapshot.json
```

最后一条命令需要真实数据、模型服务和运行依赖；它不应被静态导入收据替代。若 active
指针损坏，在线 loader 会返回可审计 warning 和空 Skill baseline；只有没有 active
pointer 时才返回 `genesis`。启动时会先跑 `env_preflight` 的行为探针，依赖不可用即
非零退出（不在降级状态下出结果）。
