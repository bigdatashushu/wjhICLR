# Harness3D v11 案例学习与槽位流式实验

2026-10-23 起，CLI 默认使用 `full_cases_v1 + scene_disjoint + batch_v1`。
这是新的运行身份，不能续跑统计输入版 frozen run。旧同题探索仅通过显式
`--learning-input-mode aggregate_v1 --gate-mode same_question` 保留。

实验以 `(question_type, evidence_state)` 作为并发、经验缓冲和版本所有权单位。共有 19 个可达槽位：8 个 g0m0、8 个 g1m0、3 个米制任务 g1m1。active seed 物理保存 24 条；5 个非米制 g1m1 不参与运行、修订或 A/B，冻结时保持 seed 内容和版本。

`ready_579` 覆盖 16 槽位；`recommended_core_897` 增加绝对距离、物体尺寸和房间面积 g1m1。旧 `maximum_1166` 含不可达状态，不能进入新实验。运行前用 `upgrade-panel` 生成新 v2 文件，旧 panel、Direct Qwen 1070 响应、重建和绑定保留。新 panel 可以从 Direct Qwen 完成集按 QA 取子集，须继续验证原始问题、选项、32 帧、JPEG 和请求身份。

## 流与版本

每个槽位只属于一个 spawn worker。每 worker 有独立运行库，从全量 24 条 seed 写独立文件，不共享可写 pointer。一个 worker 可拥有多个槽位，内部按 scene 排序并串行执行，以使用进程本地 LRU；跨 worker 允许重复解码。manifest 冻结完整 panel 的槽位所有权，resume 不按剩余任务重新分配。

每题运行时从 worker 的 active snapshot 获取该槽位 parent，并记录真实交付的 version/hash。
完整案例模式在真实执行点冻结公开问题、选项和 32 帧 PNG，`offline_input.json` 的 hash
进入完成结果。离线适配器连接各轮程序、工具结果、观察／使用／撤销关系、初末证据历史、
图片 provenance、预测和题级 induction 参考答案，形成 canonical `V11ExperienceCase`。
`FitSetExperienceCase` 只作为流式缓冲兼容封装，完整案例保存在 `full_case`，
批次写出 `experience_bundle.json`。聚合统计继续作为概览，原始证据也实际交付修订器。
非 induction、父版本或交付 hash 错配、服务故障、缺失 delivery、SIGALRM 环境故障
不进入完整案例 buffer；不放宽生产标签隔离。

复用 `StreamingEvolutionRunner` 的 `new_since_attempt`：达到 12 条新合格轨迹立即暂停该槽位下一题，尝试离线修订。拒绝后保留同 parent 的已有经验，再收 12 条新轨迹触发下一次；晋升后换 parent、清空旧 parent buffer。下一题必须加载新版本。未满 12 条不做尾批修订。发布后正常 loader 和 hash 验证必须通过，才能继续。

## 严格准入

`scene_disjoint` 要求显式 `--gate-panel-name` 或 `--gate-variant-ids`，
每槽位至少 `gate_size=8` 个不同场景，且与全部 learning 场景隔离。gate 只允许
独立留出的 induction 或 inner-validation 输入；即使使用 induction 留出场景，其
逐题标签和反馈也只归 gate，不回流学习。构造 runner 时即冻结 gate rows、面板和
输入／绑定并核验全集覆盖；运行前对全部 gate 做无模型预检，不能被 `--max-variants`
缩小范围绕过。gate rows 不进入在线积累分母或学习 buffer。

历史 `same_question` 从积累 panel 选择不同 scene，属于同题拟合准入。该模式的结果
不能视为独立场景验证，且新的学习效果实验应使用默认隔离模式。

每候选在 seed 137 和 138 分别执行 parent/candidate，两臂独立 namespace、trace、临时目录、结果文件，共享只读问题、32 帧、artifact、binding 和配置。M5 identity 继续使用封存 seed，模型请求 seed 由 gate client 单独覆盖并记录；不重新计算或修改 binding。

两个 seed 都要满足分数严格提升、run error 不增加、program error 不增加、legal rate 不下降、无不可信几何、无服务错误、实际 Skill delivery/hash 正确、模型 seed 记录正确、两臂输入计划 hash 相同。静态检查保持同 lineage、题型、状态、name、完整 SKILL.md、大小上限、已知工具与泄漏扫描。Skill 不能改变框架工具授权。准入决定带 hash，候选正文与 parent/run 身份一一对应。

A/B 结果只能用于准入，不进入下一批 revision。离线模型接收当前 parent 正文、
本槽位工具合同、聚合概览和已授权完整 induction 案例；不能执行实验、评分或决定晋升。

## r10 后的反馈与修订协议

当前流使用 `harness3d-fitset-stream/5.0`、`skill-evolution-stream-v11/2.0`，
修订使用 `v11-fit-set-slot-revision/4.0` 与 `fit-set-slot-revision-v4-case-repair`，
canonical bundle 为 `skill-experience-bundle-v11/6.0`。旧 case 可离线读取；
旧 checkpoint 不能以新行为继续运行，须新建运行身份。历史结果和分母不修改。

`fitset-process-feedback/3.0` 分开记录图像交付、工具解析 ID、候选清单、动作和错误链。
图像收到响应不证明理解正确；零工具次数不表示没有视觉输入；候选数不是对象真值，
`resolved_tool_object_count` 不是语义选择或答案。没有遥测的旧记录保留 unknown/null。
聚合概览保持安全白名单；完整案例另行通过 induction 权限校验交付。仅
`aggregate_v1` 保持原题、选项、答案、对象 ID 和原始响应不进入 reviser 的旧行为。

在线 prompt 与修订器同源读取 `ExecutionContract`，合同携带版本/hash、方法完整字符上限、提交/观察动作及工具返回约束。工具目录从实际签名生成，明确能力上界和运行时对象/集合前置条件。静态检查增加带行号和合同依据的确定性冲突检查、示例编译、参数与返回类型检查；它不能证明任意自然语言算法正确，也不能代替性能 gate。

修订响应必须声明 `action: revise | no_change`。revise 包含完整正文、单一
change_hypothesis、实际变更的 changed_sections、expected_process_effect 和
evidence_signature_refs；改动声明与实际段落差异校验。no_change 不生成候选、
不运行 gate、不切换 parent、不阻塞后续批次，保存可恢复身份收据。静态重试返回
上一轮失败候选、原始 hash、安全错误码、行号与合同依据；样本复制按行脱敏，
跨行复制不会因另一行已经脱敏而漏过。候选超出修复预算则停止，禁止静默截断。

## 可选诊断与预算

`--learning-input-mode` 有四种冻结选择：

- `aggregate_v1`：历史统计直接修订；
- `full_cases_v1`：默认完整文本案例直接修订；
- `diagnosis_v1`：一次文本案例诊断，再以完整案例、诊断和原始来源片段修订；
- `diagnosis_multimodal_v1`：诊断额外接收经过 hash 校验的真实冻结图片。

诊断默认最多 4 个案例、64000 个输入字符、8 张图片、4096 个输出 token；
对应 `--diagnosis-max-cases/--diagnosis-max-input-chars/--diagnosis-max-images/--diagnosis-max-output-tokens`。
案例按成功、失败、部分得分稳定交错选择，超限整例不选并记录原因。图片优先
已记录观察的视图，然后覆盖冻结帧范围并跨案例分配预算。路径或 ledger 不表示看过
图片；未供应的图片仍留来源引用，不能成为视觉正确性的证明。

观察和原因假设分开保存，假设要求支持证据、父方法标题位置、置信程度和替代解释，
引用不存在则拒绝。诊断不执行工具、不补造检测或重建、不决定候选晋升。
原始完整 bundle 始终进入修订器。文本和多模态模式的真实模型效果尚待单独验证。

诊断、修订和静态修复都计入每槽位修订请求上限；启用 `--max-requests` 时，
所有请求与 solver/gate 共用累计 governor、并发 permit 和限速。`episode_pool`
要求显式全局额度，历史 `slot_shards` API 仍兼容未配置 governor 的调用。
每 campaign 的诊断只付费请求一次，修复复用
完成收据。intent 存在但 completion 不明时停止并保留消耗，不能自动重放。
完整案例路径固定采用 batch 调度，避免仅根据聚合签名相同跳过不同具体经验；
历史统计路径仍可选择 `novel_failure_v1`。

## 冻结与评分

`freeze` 只合并每个唯一 owner 的最后成功发布版本，不做额外终版修订。检查 worker 所有权、结果分母、未解决触发、active loader 和其他槽位仍等于 seed。全部候选被拒绝时也能冻结 seed；不声称已产生学习增益。

bundle 保存 24 份独立完整正文，记录可达数 19、实际更新数以及保留父版本数。
final 系统按精确槽位加载，不生成 runtime adapter 身份；5 个不可达槽位不能被执行。
复用 learning panel 的系统对比仍属于同题结果；scene-disjoint 晋升只是开发准入，
不能代替从未参与修订和晋升的 outer/final-test 泛化评测。

## 运行与验收

```bash
python scripts/run_fitset_v11.py upgrade-panel --source OLD_PANEL --output NEW_PANEL
python scripts/run_fitset_v11.py run-system --phase accumulation --panel NEW_PANEL --panel-name LEARNING_PANEL --gate-panel-name GATE_PANEL --workers 4 --gate-size 8 --max-requests REQUEST_BUDGET --metadata METADATA --bindings BINDINGS --direct-run-root DIRECT --output-dir NEW_RUN --preflight-only
python scripts/run_fitset_v11.py run-system --phase accumulation --panel NEW_PANEL --panel-name LEARNING_PANEL --gate-panel-name GATE_PANEL --workers 4 --gate-size 8 --max-requests REQUEST_BUDGET --metadata METADATA --bindings BINDINGS --direct-run-root DIRECT --output-dir NEW_RUN --resume --max-variants 16
python scripts/run_fitset_v11.py audit-pilot --system-root NEW_RUN --expected-variants 16
python scripts/run_fitset_v11.py freeze --accumulation-root NEW_RUN --output-dir NEW_BUNDLE
```

扩全量前 pilot 必须有真实执行的 Harness3D 程序、无基础设施错误、无服务错误、无 evidence mismatch、summary/result/outcome 一致。视觉补答不能遮蔽所有程序失败。preflight 只生成输入收据，不冻结正式题目的动态 Skill 版本。

SIGINT 只停止新增调度，等待在途 worker 完成后退出；父进程从结果文件重建 summary。恢复重新核验已完成题，去重回灌 buffer，未完成 trace 拒绝静默重跑。每题与候选/A/B 前检查 80 GiB 保护。API 凭据只读环境变量，不写入 manifest、日志或库。

r06-r02 的线程并发因 SIGALRM 主线程要求导致 520 次程序执行失败；106 个结果不能作为方法经验，已清理。本实现的模型无关回归不能替代真实云端 pilot；完成代码后不自动恢复该实验。
