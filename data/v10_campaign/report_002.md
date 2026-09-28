# v10 两代 campaign 报告（camp-v10-002）

- 代码版本：c2da26ffed153be0358366b70686790bd3eb5d70（工作区脏=True；关键源码聚合 sha256=8165e4b9fbdee74d）
- 初始快照：S0-seed-20260925-v1
- 目标题型 / Skill：object_counting / S01
- 终态：completed_with_rejection （state=COMPLETE）
- 停止原因：（无）
- 冻结面板：learning=8 题 / 8 scene；inner_g0=6；inner_g1=6；互不重叠=True

## gen1
- learning 运行：run_id=camp-v10-002-g1-learning 题目 8 snapshot=S0-seed-20260925-v1 panel_hash=ab7179e24212
- 冻结配置：retrieval=ret-v9-1 sha=4b497fdbcc8c manifest_hash=61259a205801
- 经验包：bundle-2f849253ed981d5ea87a 父=S01@1.0.0 合格=4 scene=4 成功=0 失败=0
  - 行为摘要：{"answer_correct_rate": null, "mra_mean": 0.30000000000000004, "n_abstain": 0, "n_answer_correct": 0, "n_answer_graded": 0, "n_answer_wrong": 0, "n_declared_in_program": 0, "n_delivered_not_used": 0, "n_eligible": 4, "n_mra_graded": 4, "n_partial_tool_recovery": 0, "n_retrieved_not_delivered": 0, "n_run_error": 0, "n_template_tool_overlap": 4, "n_usage_supported": 4, "run_error_rate": 0.0}
  - 排除摘要：{"delivered_but_no_usage_clue": 4}
- 候选：cand-983abb64c7ce S01@1.0.0 → S01@1.1.0（diff 字段=['call_graph_template', 'description', 'validation_assertions', 'version']）
  - 假设：父版本中工具可用且被支持，但候选记录、去重结果与视觉实例表未充分对齐，导致计数系统性偏离；通过强制同帧共现保留、跨帧轨迹合并、覆盖审计和冲突定位，可解决重复与遗漏问题。
- 静态检查：passed=True problems=[]
- seed 0：父 0.2833（S01@1.0.0） vs 候选 0.2833（S01@1.1.0） Δ=+0.0000；run_error 0→0；合法答案率 1.000→1.000；正文入请求 6/6→6/6
- seed 1：父 0.2833（S01@1.0.0） vs 候选 0.2833（S01@1.1.0） Δ=+0.0000；run_error 0→0；合法答案率 1.000→1.000；正文入请求 6/6→6/6
- 决定：reject
  - both_arms_body_entered_request: True
  - candidate_delivered_at_least_once: True
  - content_and_experience_eligible: True
  - no_schema_or_permission_violation: True
  - panel_score_strictly_improved: False
  - run_error_not_increased: True
  - same_episode_and_artifacts: True
  - valid_answer_rate_not_decreased: True
  - 原因：['seed 0: panel_score_strictly_improved=False', 'seed 1: panel_score_strictly_improved=False']
- 发布：未发布（reject；['seed 0: panel_score_strictly_improved=False', 'seed 1: panel_score_strictly_improved=False']）
- 发布后使用：status=promoted_not_observed retrieved=False delivered=False hash_match=False 经验事件=0

## gen2
- learning 运行：run_id=camp-v10-002-g2-learning 题目 8 snapshot=S0-seed-20260925-v1 panel_hash=ab7179e24212
- 冻结配置：retrieval=ret-v9-1 sha=4b497fdbcc8c manifest_hash=61259a205801
- 经验包：bundle-f2f401f90e7716be8939 父=S01@1.0.0 合格=4 scene=4 成功=0 失败=0
  - 行为摘要：{"answer_correct_rate": null, "mra_mean": 0.30000000000000004, "n_abstain": 0, "n_answer_correct": 0, "n_answer_graded": 0, "n_answer_wrong": 0, "n_declared_in_program": 0, "n_delivered_not_used": 0, "n_eligible": 4, "n_mra_graded": 4, "n_partial_tool_recovery": 0, "n_retrieved_not_delivered": 0, "n_run_error": 0, "n_template_tool_overlap": 4, "n_usage_supported": 4, "run_error_rate": 0.0}
  - 排除摘要：{"delivered_but_no_usage_clue": 4}
- 候选：cand-72770e01830f S01@1.0.0 → S01@1.1.0（diff 字段=['call_graph_template', 'description', 'validation_assertions', 'version']）
  - 假设：上一版 4 次可评测执行全部命中 tool_overlap 且 usage_supported，但 mra_mean 仅 0.30、没有精确正确，说明问题不在工具是否可用，而在候选记录没有被显式转成“每条实例—帧/result_id”的可核对证据链：计数凭外观直觉给出，合并/拆分与覆盖缺口未经逐实例核对。因此按 §12.2 允许的范围，把目标类别口径、跨帧覆盖、重复实例与遮挡重现、故障/截断/空检出分支、工具—视觉冲突与提交前检查写成强制留痕步骤，可让同一批工具产出真正被计入答案。
- 静态检查：passed=True problems=[]
- seed 0：父 0.7000（S01@1.0.0） vs 候选 0.5333（S01@1.1.0） Δ=-0.1667；run_error 0→0；合法答案率 1.000→1.000；正文入请求 6/6→6/6
- seed 1：父 0.7000（S01@1.0.0） vs 候选 0.5333（S01@1.1.0） Δ=-0.1667；run_error 0→0；合法答案率 1.000→1.000；正文入请求 6/6→6/6
- 决定：reject
  - both_arms_body_entered_request: True
  - candidate_delivered_at_least_once: True
  - content_and_experience_eligible: True
  - no_schema_or_permission_violation: True
  - panel_score_strictly_improved: False
  - run_error_not_increased: True
  - same_episode_and_artifacts: True
  - valid_answer_rate_not_decreased: True
  - 原因：['seed 0: panel_score_strictly_improved=False', 'seed 1: panel_score_strictly_improved=False']
- 发布：未发布（reject；['seed 0: panel_score_strictly_improved=False', 'seed 1: panel_score_strictly_improved=False']）
- 发布后使用：status=promoted_not_observed retrieved=False delivered=False hash_match=False 经验事件=0

## 可声明结论

见报告正文（本骨架只汇总收据事实；口径声明必须人工书写，避免机器替项目下结论）。

## 不可声明结论

- 未跑 outer_holdout / final_test：不得声明「独立泛化收益」（§2.3/§8.5）。
- inner 小面板双 seed 改善只表示「开发门通过」，不表示统计显著（§8.6）。

## 未完成项

- 见 `docs/skill_library_v10_implementation.md` 第 4 节。
