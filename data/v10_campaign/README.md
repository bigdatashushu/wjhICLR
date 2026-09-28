# v10 campaign 收据索引

| 目录 | 是什么 |
|---|---|
| `library/` | campaign 自己的 Skill 库（从仓库 `skill_library/` 复制 S0 快照与 manifest）。**仓库 S0 库保持只读**：`scripts/build_skill_library.py --check` 的 digest 与 active 指针不因演化而变 |
| `runs/camp-v10-002/` | **正式两代运行**（冻结代码：`code_state` 记 commit + 13 个源码 sha256）。两代都真实跑完、两个候选都被拒绝 |
| `runs/camp-v10-001/` | 开发过程运行（含 8 条修正的轨迹）；`gen3_beyond_scope/` 是超出两代范围的第三次运行残留 |
| `report_002.md` | 由收据自动汇总的报告骨架（§18 模板） |
| `report_skeleton.md` | `camp-v10-001` 的同款骨架（旧口径，保留对账） |

每代目录（§14.1）：`parent_run_manifest.json`（含 `code_state`）/ `experience_events.jsonl` /
`experience_bundle.json` / `inducer_prompt.txt` / `inducer_receipt.json` / `candidate.json` /
`static_validation.json` / `candidate_snapshot.json` / `paired_seed_0.json` /
`paired_seed_1.json` / `decision.json` / `promotion.json` / `post_publish_use.json`。

结论、可声明 / 不可声明范围见仓库根目录 `系统架构v10_实现与验收报告.md`。
