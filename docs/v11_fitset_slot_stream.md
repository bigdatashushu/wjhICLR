# Harness3D v11 同题槽位流式实验

实验以 `(question_type, evidence_state)` 作为并发、经验缓冲和版本所有权单位。共有 19 个可达槽位：8 个 g0m0、8 个 g1m0、3 个米制任务 g1m1。active seed 物理保存 24 条；5 个非米制 g1m1 不参与运行、修订或 A/B，冻结时保持 seed 内容和版本。

`ready_579` 覆盖 16 槽位；`recommended_core_897` 增加绝对距离、物体尺寸和房间面积 g1m1。旧 `maximum_1166` 含不可达状态，不能进入新实验。运行前用 `upgrade-panel` 生成新 v2 文件，旧 panel、Direct Qwen 1070 响应、重建和绑定保留。新 panel 可以从 Direct Qwen 完成集按 QA 取子集，须继续验证原始问题、选项、32 帧、JPEG 和请求身份。

## 流与版本

每个槽位只属于一个 spawn worker。每 worker 有独立运行库，从全量 24 条 seed 写独立文件，不共享可写 pointer。一个 worker 可拥有多个槽位，内部按 scene 排序并串行执行，以使用进程本地 LRU；跨 worker 允许重复解码。manifest 冻结完整 panel 的槽位所有权，resume 不按剩余任务重新分配。

每题运行时从 worker 的 active snapshot 获取该槽位 parent，并记录真实交付的 version/hash。完成题形成独立 `FitSetExperienceCase`，保留真实 source split、过程签名、trace 引用和 hash。原始题目、答案和选项不复制到反馈中。读取标签有 fit-set 专用访问记录；生产 induction 的 Schema 和隔离规则不放宽。服务故障、缺失 delivery、SIGALRM 环境故障不进入经验 buffer。

复用 `StreamingEvolutionRunner` 的 `new_since_attempt`：达到 12 条新合格轨迹立即暂停该槽位下一题，尝试离线修订。拒绝后保留同 parent 的已有经验，再收 12 条新轨迹触发下一次；晋升后换 parent、清空旧 parent buffer。下一题必须加载新版本。未满 12 条不做尾批修订。发布后正常 loader 和 hash 验证必须通过，才能继续。

## 严格准入

每个槽位的 gate 在运行前从同题 panel 确定性选择最多 8 个不同 scene 的题目。不足 8 题的槽位不能晋升。gate 是同题拟合准入，不是独立验证集。

每候选在 seed 137 和 138 分别执行 parent/candidate，两臂独立 namespace、trace、临时目录、结果文件，共享只读问题、32 帧、artifact、binding 和配置。M5 identity 继续使用封存 seed，模型请求 seed 由 gate client 单独覆盖并记录；不重新计算或修改 binding。

两个 seed 都要满足分数严格提升、run error 不增加、program error 不增加、legal rate 不下降、无不可信几何、无服务错误、实际 Skill delivery/hash 正确、模型 seed 记录正确、两臂输入计划 hash 相同。静态检查保持同 lineage、题型、状态、name、完整 SKILL.md、大小上限、已知工具与泄漏扫描。Skill 不能改变框架工具授权。准入决定带 hash，候选正文与 parent/run 身份一一对应。

A/B 结果只能用于准入，不进入下一批 revision。离线模型只接收当前 parent 正文、本槽位工具文档和去标识的聚合过程反馈；不能执行实验、评分或决定晋升。

## 冻结与评分

`freeze` 只合并每个唯一 owner 的最后成功发布版本，不做额外终版修订。检查 worker 所有权、结果分母、未解决触发、active loader 和其他槽位仍等于 seed。全部候选被拒绝时也能冻结 seed；不声称已产生学习增益。

bundle 保存 24 份独立完整正文，记录可达数 19、实际更新数以及保留父版本数。final 系统按精确槽位加载，不生成 runtime adapter 身份；5 个不可达槽位不能被执行。最终全部 panel 题目用冻结后的 Skill 重新作答，与已完成的 Direct Qwen 逐题配对计 Accuracy/MRA、合法率、可用率、错误、调用次数和耗时。结果只能解释为同题拟合的完整系统对比。

## 运行与验收

```bash
python scripts/run_fitset_v11.py upgrade-panel --source OLD_PANEL --output NEW_PANEL
python scripts/run_fitset_v11.py run-system --phase accumulation --panel NEW_PANEL --panel-name ready_579 --workers 4 --gate-size 8 --metadata METADATA --bindings BINDINGS --direct-run-root DIRECT --output-dir NEW_RUN --preflight-only
python scripts/run_fitset_v11.py run-system --phase accumulation --panel NEW_PANEL --panel-name ready_579 --workers 4 --gate-size 8 --metadata METADATA --bindings BINDINGS --direct-run-root DIRECT --output-dir NEW_RUN --resume --max-variants 16
python scripts/run_fitset_v11.py audit-pilot --system-root NEW_RUN --expected-variants 16
python scripts/run_fitset_v11.py freeze --accumulation-root NEW_RUN --output-dir NEW_BUNDLE
```

扩全量前 pilot 必须有真实执行的 Harness3D 程序、无基础设施错误、无服务错误、无 evidence mismatch、summary/result/outcome 一致。视觉补答不能遮蔽所有程序失败。preflight 只生成输入收据，不冻结正式题目的动态 Skill 版本。

SIGINT 只停止新增调度，等待在途 worker 完成后退出；父进程从结果文件重建 summary。恢复重新核验已完成题，去重回灌 buffer，未完成 trace 拒绝静默重跑。每题与候选/A/B 前检查 80 GiB 保护。API 凭据只读环境变量，不写入 manifest、日志或库。

r06-r02 的线程并发因 SIGALRM 主线程要求导致 520 次程序执行失败；106 个结果不能作为方法经验，已清理。本实现的模型无关回归不能替代真实云端 pilot；完成代码后不自动恢复该实验。
