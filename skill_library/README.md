# harness3D Skill Library

当前 active snapshot 为 `S0-v11-contract-repair`。每个规范题型只有一条 active 方法，
运行对象仅接受 `SkillSpecV11`，快照 Schema 仅接受 `runtime-skill-snapshot/2.0`。

## 方法源与运行合同

完整方法位于 `versions/<skill_id>/<version>/<name>/SKILL.md`，每个已发布版本不可覆盖。
框架在索引中维护 `skill_id`、版本、题型、`source_ref` 和 `content_sha256`。
加载器核对 manifest、源文件路径、UTF-8/LF 和全文 hash 后，原样交付完整 `skill_md`。
超出方法上下文上限时拒绝运行，不截断正文。方法内容不经 JSON SkillSpec 重编译。

在线合同固定为 `program_synth_v11_2`、`solver-v11.2-m11-acceptance` 与 `tool-docs-v11.1`。
B01 不交付 Skill；B11 交付当前题型的冻结 S0 方法。工具证据和权限由框架统一校验。

## 快照与发布

- `snapshots/active_snapshot.json`：唯一在线指针。
- `snapshots/snapshot_<id>.json`：不可变快照。
- `manifests/`：每个快照的内容身份。
- `versions/`：不可变完整方法源。
- `candidates/`：v11 修订候选与 provenance，不能直接参与正常在线查找。
- `imports/`、`validation/`：历史导入证据，不是运行入口。

验证当前基线：

```bash
PYTHONPATH=src .venv/bin/python scripts/build_skill_library_v11.py --profile contract-repair --check
```

修订入口为 `scripts/run_evolution_campaign_v11.py`，通过
`publish_v11_candidate` 写入完整候选与快照并原子切换指针。拒绝保持父版；
发布后正常 learning 未能确认新正文交付时回滚。旧快照只供审计，旧编译器的 `skills/` 与 `generated/` 已删除，旧格式不能在当前
运行时加载；复现实验需 checkout 对应 Git 提交。

`S0-v11-contract-repair` 是格式与接口修复基线，不是经真实 learning 晋升的性能版本。
S01/S08 为 1.2.0，其余六条为 1.1.0；既有源文件和快照身份保持不变。

## 配对与标签隔离

`scripts/run_skill_ablation_v11.py` 在相同输入、模型、预算及固定质量门上独立执行
B01/B11，输出逐题结果、交付 hash 和汇总。详见[配对协议](../docs/skill_ablation_v11.md)。

离线修订只能读取 induction 的题目级标准答案和反馈，并记录 `label_access=true`。
inner/outer/final 的逐题标签不得进入修订输入；在线求解始终不读取答案或 GT 三维标注。
候选正文不得包含具体样本身份或答案。
