# 交接提示词（Skill3D v9 接入，2026-09-27，下一项 = P7）

> 用法：把「下面整段」复制给新对话的 agent 作为起始指令。旧的 v5 交接提示词见
> `HANDOFF_PROMPT.md`（历史，勿据此判断现状）。

---

你是接手的 Coding Agent，工作目录 `/nas/wangjh/harness3d/skill3d_codebase`。
任务：按《系统架构v9.md》（项目根目录，唯一目标态）继续完善 Skill 编译／验证／candidate／
promotion／snapshot／在线加载／运行审计架构。

**先做两件事再动手**：

1. 读 `docs/skill_library_v9_implementation.md` —— P0–P5g、P3b、P6、D1/D2、P4 每一节都记了
   "改了什么 + 依据的规范原文 + 明确未完成项"，末尾还有"未完成且不能由本收据宣称"清单。
2. 读 `系统架构v9.md` 里你要动的那一节原文。

不要重新开始，不要重跑已经跑通的东西，**不要重复询问已经明确的信息**（见下面「已裁决，
不要再问」）。

## 一、硬边界（始终有效，违反=白干）

- `skill_library/skills/*/SKILL.md` 是唯一可编辑源；`generated/`、`snapshot/`、`manifest/`
  都是派生物，**不得手改**。
- 文档级 schema **8.0 保留原身份**（不机械改名为 v9）；`EpisodeTrace` 当前合同身份是
  **9.0**，旧 6.0 语料由 `legacy/readers.py` 只读解释，不得静默升格。
- 在线 loader 只读 active snapshot 里 inline 的 `spec_content`；**不**扫描 `skills/`、
  `generated/`、`candidates/`、`candidates/future/`。
- future candidate 不会自动进入 active；晋升必须走 `scripts/manage_skill_library.py promote`
  （原子切换 + 严格静态检查）。
- `runtime_verified` 保持 `false`；**不得伪造**真实模型、真实工具、真实 Skill 晋升、两轮
  演化或 E−S0 结果。没接线/没真跑的一律单独列"未完成"，不混进完成结论。
- 不要修改全局环境或升级依赖。
- git 工作区本来就有大量既有 tracked 修改与 untracked v9 文件。**禁止** `git reset --hard`
  / `git checkout --` / `git restore`；不要删除或覆盖不是本轮创建的文件（例：根目录的
  `HANDOFF_PROMPT.md` 是 v5 历史交接，别改它）。
- `git diff --check` 只有一处**既有** trailing whitespace：`src/skill3d/segmentation/sam2_tracker.py:958`。
  他人遗留，**不要"修"**。
- S0 快照不可动：manifest canonical digest 必须始终是
  `61259a20580170f8bf3d0490972cd448a6a67cc5d6316b07c4a77b7db4708046`，active pointer 是
  `S0-seed-20260925-v1`。每轮收尾跑：`PYTHONPATH=src $EXP scripts/build_skill_library.py --check`。

## 二、环境（曾因用错环境误判过整轮）

- **必须**用冻结实验环境：`EXP=/home/cvailab/anaconda3/envs/skill3d-exp/bin/python`
- 全量测试：`PYTHONPATH=src $EXP -m pytest tests/ -q -p no:cacheprovider`
- **当前基线：990 passed, 3 skipped, 0 failed**（改完必须复核这个数）
- base conda env（numpy 1.26 + scipy 1.18）**装得上但跑不动**：`import scipy.spatial` 抛
  `np.long`、`cKDTree` 抛 `copy=None`，而且**不报错退出** —— M4 主门子项 fail-closed 后
  `scene_route=fallback_2d_only`，整条链照常跑完、几何能力静默失效。
  `src/skill3d/env_preflight.py` + `tests/conftest.py` 已有行为级预检，依赖不可用即拒绝启动。
- **不要**用测试进程内的 NumPy shim（只补符号、补不了 ABI，曾产出过一份"6 通过 12 失败"
  的假结果并被误当成 M4 缺陷）。

## 三、当前环境状态（接手先确认还活着）

- **vLLM 在 8100 端口监听**：模型名 `qwen3vl-8b-r0`（Qwen3-VL-8B-Instruct-FP8，
  max_model_len=32768）。实测 **图像上限 32 张/请求**（33 张 → `400 At most 32 image(s) may be
  provided in one prompt`）。启动脚本 `scripts/serve_qwen3vl_dp8.sh`。
- 检测器（GroundingDINO）在 **20022** 监听（`configs/config.yaml` 的 `detection.endpoint`）。
- 已有真实重建产物可复用：`data/reconstructions/vggt/7b6477cb95.json`（scene `7b6477cb95`，
  质量实算、主门通过、world frame degraded）。

## 四、已完成（都有测试与全量回归）

- **P0** 证据诚实性 + Schema 身份 + 环境预检（`synthesis_source` 不再洗成 `vllm_ok`；
  轮次原因分离；执行前拒绝记 `failed`；trace 升 9.0 + legacy 分派；env preflight）。
- **P1** 跨轮 tool 账本 + 逐轮 `rounds`（含程序文本与 sha256）+ `agent_rounds`/`yield_count`/`budget`
  落盘 + 恢复轮计入轮预算。
- **P2** §10.1/§4/§12 答案合同（`schemas/answer.py`、单位按题型合同、六字段归因台账、二次提交保护）。
- **P3** §6.4/§8.2/§6.1 授权收据（`schemas/authorization.py` + `authorize_tool_call` 单一判定；
  允许与拒绝都留收据；门三态 pass/fail/not_applicable）。
- **P5/P5b–P5g** 与规范直接冲突的两处修正（§11.3 无围栏不算解析恢复；§14.3 准入用完整 inner）、
  先冻结清单再加载输入、FrameSet 规范字段 + 缓存身份、§5.2 用可读帧继续、M2 收窄为最小
  输入有效性检查、缺失掩码回填、artifact 缓存切缓存身份。
- **P3b** 原因码生产者接线（拆开 `m5_not_run`/`detector_fault`/`zero_detection_after_retry`）。
- **P6** §13.5/§13.6 检索记录：交付正文 hash、全候选过滤原因 + 排序分数落盘、
  `retrieved/delivered/declared` 四态分离、证据更新后同快照重检索、检索策略（top-k/权重/
  方法上下文上限）进配置并冻结、超服务限制候选在静态检查拒绝。新增
  `skills/delivery.py`、`schemas/retrieval.py`、`routing/retrieval_policy.py`。
- **D1** §5.2 可重试加载 = **换来源/换副本**（用户 2026-09-27 裁定）：候选顺序 主约定 →
  同目录同名副本 → 备用根（目录式/平铺式镜像，配置 `paths.raw_video_fallbacks`）；
  用了副本记抽样收据 + `source_retried_samples` + `video_id` 带来源标签（不与主来源共用
  重建缓存）。本机 288/288 主来源齐备 → 该路径当前不触发。
- **D2** §6.1/§6.4 原因码**扩展为五值**（用户裁定）：新增 `quality_gate_not_passed`
  （"运行成功但质量门未过"），在 M4 主门/world frame/metric 自洽/检出稀疏/track 阈值处接线。
- **P4** §9.4 主动图像与补检：注册 `inspect_frames`/`detect_objects`（`tools/vision_tools.py`）、
  句柄持有冻结帧集与图像账本（`tools/image_ledger.py`）、图像经 `YieldObservations` 进入
  **下一次**请求（带明确映射清单）、`produced/delivered/observed` 三态 + 布局/缩放/token
  成本落盘（`schemas/image_ledger.py`、trace 的 `image_ledger`）、工具面升 `tool-face-v9`。
  **真实 vLLM receipt**：`data/p4_acceptance.json`（qa 2487，第 2 轮真实请求带 1 张裁剪图，
  prompt_tokens=13850；第 1 轮脚本程序触发，脚本轮标记已按事实撤回）。

## 五、已裁决，不要再问

| 事项 | 裁定（用户，2026-09-27） |
|---|---|
| §5.2 不足 32 帧 vs 硬约束 21 | 按 v9：用可读帧继续，不重复填充、不改采样（P5d） |
| §5.2"依赖完整帧集的工具按条件禁用" | **用户明确指示忽略**，不要实现、也不要再列为待办 |
| §5.1/§5.2"可重试加载"的语义 | **换来源/换副本**（不是重新解码同一文件） |
| §6.1/§6.4 原因码词表 | **扩展**：新增第五值 `quality_gate_not_passed` |
| §13.5 检索 `top_k` 起始值 | **保持 3**（不改 1） |

## 六、下一项：P7（§14/§16 两轮真实演化 + E−S0）

按约定顺序，P6/P4 之后就是 P7。当前事实（每一条都要先核验再动手）：

1. **`max_evolution_rounds` / `candidate_validation_seeds` 是死配置**（配置里登记了、零读取方）——
   §14.2 要求"至少两轮真实演化"、§14.3 要求"两个固定 seed 配对"，现在没有任何代码消费它们。
2. **E−S0 不存在**：§16.2 要求首轮开发用两个固定 seed 成对跑八题型面板（V1 臂 vs E 臂），
   现在没有 E 臂、没有配对运行。
3. **首次处理成本未测量/未报告**（§16.4"只报告首次处理成本"）：图像 token 已逐轮落盘
   （P4），但"每题首次处理成本"的汇总报告还没有。
4. **§17.1 的数据访问记录行完全没有对应记录类型**；`label_access`（§14.1：学习题标签允许
   离线分析并记 `label_access`）也没有落点。
5. **归纳器没有 split 过滤**：现在读 `--trace-dir` 下所有 trace，不看 split 字段 ——
   §14.1 明文"inner 的逐题材料只留在验证器，**不交给归纳器**"，这是硬隔离要求。
6. 另外：P4 发现**真实模型不会主动裁剪**（4/4 条真实 episode 直接作答，
   `n_derived_produced=0`），这属于方法/提示词层面的问题，是演化要处理的对象。

P7 的验收（§18 阶段 2）：至少两轮真实经验—候选—决定链，第二轮来源是**随后实际运行**产生的
经验（不能把同一批旧轨迹提交两次）；若晋升，提供后续加载/检索证据。

## 七、P4 的未完成边界（别当成已完成）

1. §9.4 要求新检出更新"对象记录、**证据版本和工具可用性**" —— 只做了对象记录（框架侧并入
   句柄）；mid-episode 改写 `EvidenceProfile` 或据此放开工具**没有做**（属证据/授权策略变更）。
2. 补检后的 SAM2 跟踪/绑定未实现（新检出是 2D 记录，不产 mask/点云；几何工具按其空点集
   fail-closed，不给假值）。
3. `contact_sheet` 图像种类在 schema 里预留但**没有布局用到**。
4. 真实模型不主动使用主动图像工具（见上）。
5. `inspect_frames` 裁剪入参只收像素坐标（归一化 ↔ 像素换算只在 `detect_objects` 返回里做了）。

## 八、需要用户裁决的悬置问题（不要自行决定）

1. **副本的内容校验值是否参与缓存身份**（§5.2 说"源标识**及内容校验值**"）：现在参与的是
   源标识 + 帧集内容哈希，没有对视频文件本体做哈希；代价是每次加载多读一遍文件。
2. **两处原因码仍留空**是否要细化：`object_grounding` 的"走补绑路径拿到结果"、
   "quality 标记 computed 但值为 NaN"（可能是数据缺失的约定取值，不一定是生产失败）。
3. **§9.4 的"新检出更新证据版本与工具可用性"** 是否要在 episode 中途升级证据/放开工具？
4. P7 的演化验收是否要求"真实模型主动裁剪"（否则 P4 的主动图像只有"被触发时正确"的证据）。

## 九、协作约定（前几轮一直这么做）

- **一轮做一个小项**，改完跑全量回归（`$EXP` + `-p no:cacheprovider`），报 **pass/skip/fail 实际数字**。
- 每项完成把**改动内容 + 依据的规范原文 + 明确未完成项**写进
  `docs/skill_library_v9_implementation.md`，不要只写"已完成"；决策记录要落档
  （冲突原文、用户裁决、日期），避免后人看到代码变了却不知是谁改的、为什么改。
- 改测试时若断言的是**被取代的旧政策**，改写后要在 docstring 里写明"X 取代 Y" + 规范原文，
  不要让人误以为测试被削弱。
- **不谎报**：未接线、未真实运行、只是词表落地没有数据的，单独列为未完成。
- 收尾核对：S0 的 `--check` 与 manifest digest、`git diff --check`、新增文件无尾随空白。
- 上下文/预算不够时**宁可少做也不要开一个改不完的头**（例如默认值翻转这类牵动多文件的改动
  要一次做干净，不留红的测试树）。

## 十、关键文件索引

- 规范：`系统架构v9.md`（根目录；v9 目标是 `<root>/系统架构v9.md`）
- 实现记录：`docs/skill_library_v9_implementation.md`（P0–P5g、P3b、P6、D1/D2、P4 + 决策记录）
- 本轮新增：`env_preflight.py`、`schemas/{answer,authorization,retrieval,image_ledger}.py`、
  `routing/retrieval_policy.py`、`skills/delivery.py`、`tools/{vision_tools,image_ledger}.py`、
  `tests/conftest.py`、`scripts/run_active_vision_acceptance.py`
- 主要修改面：`online/{runner,eval,config}.py`、`sandbox/kernel.py`、`sandbox/ast_guard.py`、
  `schemas/{trace,evidence,episode,tool,legacy,reconstruction,__init__}.py`、
  `tools/{contract,registry,scene_handle,__init__}.py`、`adapters/{frame_set,episode_source,vsibench_loader}.py`、
  `reconstruction/run.py`、`reconstruction_gate/evidence_profile.py`、`synthesis/{prompt_builder,program_assembler}.py`、
  `skills/{library,promote_atomic}.py`、`configs/config.yaml`、`scripts/manage_skill_library.py`
- 测试：`tests/unit/test_v9_*.py`、`tests/integration/test_v9_*.py`
- 验收 receipt：`data/p4_acceptance.json`（真实主动图像链）
