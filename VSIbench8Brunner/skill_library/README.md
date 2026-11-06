# harness3D Skill Library

当前 active snapshot 为 `mixed-r04-20261026-v1`。八个规范题型各保留三个
初始证据状态，共 24 个活跃槽位、28 个索引条目（其中四个为历史版本）。
运行对象为 `SkillSpecV11`，快照 Schema 为 `runtime-skill-snapshot/3.0`。

本次混合选择仅替换两个槽位：

| 槽位 | 活跃版本 | 来源 |
|---|---|---|
| `object_size_estimation/g1m0` | `S04-g1m0@1.1.0` | r04 双 seed 诊断准入与实际交付通过 |
| `room_size_estimation/g0m0` | `S05-g0m0@1.1.0` | r04 双 seed 诊断准入与实际交付通过 |

其余 22 个槽位使用 `S0-v11-task-tools` Parent。下降与持平候选不进入当前活跃选择，
历史来源和完整处置保存在 [选择收据](validation/mixed_r04_20261026.json)。
新快照是按用户选择冻结的混合方法，尚无整库配对成绩；原 r04 混合统计不作为新库成绩。

fit-set 的可达范围仍为 19 槽：八个 `g0m0`、八个 `g1m0`，以及距离、尺寸、面积的
三个 `g1m1`。其余五个非米制 `g1m1` 保留原 Parent。

## 方法源与运行合同

完整方法位于 `versions/<skill_id>/<version>/<name>/SKILL.md`，每个已发布版本不可覆盖。
框架在索引中维护 `skill_id`、版本、题型、初始证据状态、`source_ref` 和 `content_sha256`。
加载器核对 manifest、源文件路径、UTF-8/LF 和全文 hash 后，原样交付完整 `skill_md`。
超出方法上下文上限时拒绝运行，不截断正文。方法内容不经 JSON SkillSpec 重编译。

普通轮完整交付所选方法，终结轮按框架合同省略方法正文；工具证据和权限由框架统一校验。
明确冻结 S0 的 `run_skill_ablation_v11.py` 与 outer baseline 保留原快照。
运行混合库的实验应单独记录实际 snapshot ID、manifest 与正文 hash。

## 快照与发布

- `snapshots/active_snapshot.json`：唯一在线指针。
- `snapshots/snapshot_<id>.json`：不可变快照。
- `manifests/`：每个快照的内容身份。
- `versions/`：不可变完整方法源。
- `candidates/`：v11 修订候选与 provenance，不能直接参与正常在线查找。
- `imports/`、`validation/`：历史导入证据，不是运行入口。

验证原 S0 基线文件（不切换指针）：

```bash
PYTHONPATH=src .venv/bin/python scripts/build_skill_library_v11.py --check
```

修订入口为 `scripts/run_evolution_campaign_v11.py`，通过
`publish_v11_candidate` 写入完整候选与快照并原子切换指针。拒绝保持父版；
发布后正常 learning 未能确认新正文交付时回滚。`S0-v11-task-tools` 保留为完整 Parent；
它以及两个采用版本的旧源文件均不覆盖。历史旧 Schema 的复现使用对应 Git 提交。

## 配对与标签隔离

`scripts/run_skill_ablation_v11.py` 在相同输入、模型、预算及固定质量门上独立执行
B01/B11，输出逐题结果、交付 hash 和汇总。详见[配对协议](../docs/skill_ablation_v11.md)。

当前 fit-set 修订默认使用去标识聚合反馈；具体诊断输入按冻结合同确定。
inner/outer/final 的逐题标签不进入修订输入，在线求解不读取参考答案或 GT 三维标注。
候选正文不包含具体样本身份或答案。
