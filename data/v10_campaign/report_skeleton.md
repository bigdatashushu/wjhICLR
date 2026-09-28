# v10 两代 campaign 报告（camp-v10-001）

- 代码版本：aa5810f2177dcc5b190a3fb66cf3b9bfca4932c5
- 初始快照：S0-seed-20260925-v1
- 目标题型 / Skill：object_counting / S01
- 终态：completed_with_rejection （state=COMPLETE）
- 停止原因：（无）
- 冻结面板：learning=8 题 / 8 scene；inner_g0=6；inner_g1=6；互不重叠=True

## gen1
- learning 运行：run_id=camp-v10-001-g1-learning 题目 8 snapshot=S0-seed-20260925-v1 panel_hash=ab7179e24212
- 冻结配置：retrieval=ret-v9-1 sha=4b497fdbcc8c manifest_hash=61259a205801
- 经验包：bundle-62440ed6d7b5ce21a75c 父=S01@1.0.0 合格=4 scene=4 成功=0 失败=0
  - 行为摘要：{"answer_correct_rate": null, "mra_mean": 0.3375, "n_abstain": 0, "n_answer_correct": 0, "n_answer_graded": 0, "n_answer_wrong": 0, "n_declared_in_program": 0, "n_delivered_not_used": 0, "n_eligible": 4, "n_mra_graded": 8, "n_partial_tool_recovery": 0, "n_retrieved_not_delivered": 0, "n_run_error": 0, "n_template_tool_overlap": 4, "n_usage_supported": 4, "run_error_rate": 0.0}
  - 排除摘要：{"delivered_but_no_usage_clue": 4}
- 候选：cand-45bf111fa315 S01@1.0.0 → S01@1.1.0（diff 字段=['call_graph_template', 'description', 'validation_assertions', 'version']）
  - 假设：父版本在 object_counting 上工具已交付但未被实际调用或未被证据引用，导致答案主要依赖纯视觉且 MRA 偏低；强制闭环可让实例表与去重结果交叉核对。
- 静态检查：passed=True problems=[]
- seed 0：父 0.2833（S01@1.0.0） vs 候选 0.2167（S01@1.1.0） Δ=-0.0667；run_error 0→0；合法答案率 1.000→1.000；正文入请求 6/6→6/6
- seed 1：父 0.2833（S01@1.0.0） vs 候选 0.2167（S01@1.1.0） Δ=-0.0667；run_error 0→0；合法答案率 1.000→1.000；正文入请求 6/6→6/6
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
- 发布：未发布（['seed 0: panel_score_strictly_improved=False', 'seed 1: panel_score_strictly_improved=False']）
- 发布后使用：status=promoted_not_observed retrieved=False delivered=False hash_match=False 经验事件=0

## gen2
- learning 运行：run_id=camp-v10-001-g2-learning 题目 8 snapshot=S0-seed-20260925-v1 panel_hash=ab7179e24212
- 冻结配置：retrieval=ret-v9-1 sha=4b497fdbcc8c manifest_hash=61259a205801
- 经验包：bundle-4d2a3b420ee48706acc5 父=S01@1.0.0 合格=4 scene=4 成功=0 失败=0
  - 行为摘要：{"answer_correct_rate": null, "mra_mean": 0.3375, "n_abstain": 0, "n_answer_correct": 0, "n_answer_graded": 0, "n_answer_wrong": 0, "n_declared_in_program": 0, "n_delivered_not_used": 0, "n_eligible": 4, "n_mra_graded": 8, "n_partial_tool_recovery": 0, "n_retrieved_not_delivered": 0, "n_run_error": 0, "n_template_tool_overlap": 4, "n_usage_supported": 4, "run_error_rate": 0.0}
  - 排除摘要：{"delivered_but_no_usage_clue": 4}
- 候选：cand-808cf4f1a925 S01@1.0.0 → S01@1.1.0（diff 字段=['call_graph_template', 'description', 'version']）
  - 假设：父版在 object_counting 上的中位准确率偏低且工具线索支持不足，主要来自跨帧重复、遮挡重现和检测空/截断分支处理不清；通过限定允许工具和实例表核对可减少重复与漏计。
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
- 发布：S0-seed-20260925-v1 → None；竞争集合=None；历史=None；manifest_hash_before= → 
- 发布后使用：status=promoted_not_observed retrieved=False delivered=False hash_match=False 经验事件=0

## gen3_beyond_scope
- learning 运行：run_id=camp-v10-001-g3-learning 题目 8 snapshot=S0-seed-20260925-v1 panel_hash=ab7179e24212
- 冻结配置：retrieval=ret-v9-1 sha=4b497fdbcc8c manifest_hash=61259a205801
- 经验包：bundle-96ec380524afd6ca78ae 父=S01@1.0.0 合格=4 scene=4 成功=0 失败=0
  - 行为摘要：{"answer_correct_rate": null, "mra_mean": 0.3375, "n_abstain": 0, "n_answer_correct": 0, "n_answer_graded": 0, "n_answer_wrong": 0, "n_declared_in_program": 0, "n_delivered_not_used": 0, "n_eligible": 4, "n_mra_graded": 8, "n_partial_tool_recovery": 0, "n_retrieved_not_delivered": 0, "n_run_error": 0, "n_template_tool_overlap": 4, "n_usage_supported": 4, "run_error_rate": 0.0}
  - 排除摘要：{"delivered_but_no_usage_clue": 4}

## 可声明结论

见报告正文（本骨架只汇总收据事实；口径声明必须人工书写，避免机器替项目下结论）。

## 不可声明结论

- 未跑 outer_holdout / final_test：不得声明「独立泛化收益」（§2.3/§8.5）。
- inner 小面板双 seed 改善只表示「开发门通过」，不表示统计显著（§8.6）。

## 未完成项

- 见 `docs/skill_library_v10_implementation.md` 第 4 节。
