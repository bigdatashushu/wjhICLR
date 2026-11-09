# harness3D Skill Library

当前 active snapshot 为 `nonmetric-g1m1-aligned-g1m0-20261106-v1`。
八个规范题型各保留三个初始证据状态，共 24 个活跃槽位；历史版本继续保留。
运行对象为 `SkillSpecV11`，快照 Schema 为 `runtime-skill-snapshot/3.0`。

本次按用户要求，将五类非米制题的 g1m1 方法正文替换为各自当前 g1m0 正文：

| 槽位 | 活跃版本 | 方法正文来源 |
|---|---|---|
| `object_counting/g1m1` | `S01-g1m1@1.0.2` | `S01-g1m0@1.0.0` |
| `object_rel_distance/g1m1` | `S03-g1m1@1.0.2` | `S03-g1m0@1.0.0` |
| `object_rel_direction/g1m1` | `S06-g1m1@1.0.2` | `S06-g1m0@1.0.0` |
| `obj_appearance_order/g1m1` | `S07-g1m1@1.0.2` | `S07-g1m0@1.0.0` |
| `route_planning/g1m1` | `S08-g1m1@1.0.2` | `S08-g1m0@1.0.0` |

完整方法正文逐字一致；只保留 g1m1 的名称、状态描述和真实初始能力声明。
三类米制题继续使用原版本：绝对距离 `S02-g1m1@1.0.2`、
物体尺寸 `S04-g1m1@1.0.1`、房间面积 `S05-g1m1@1.0.1`。
其余 19 个活跃槽位、旧源文件和旧快照均保留。

父快照为 `g1m1-manual-trace-feedback-20261105-v1`。
本次通过原生加载器的全文哈希、24 槽覆盖与五份完整方法交付检查，
记录见 [对齐核验](validation/nonmetric_g1m1_alignment_20261106.json)。
该检查没有发出模型请求，不代表新版本的实验得分。
原 `mixed-r04` 选择历史继续保存在
[旧选择收据](validation/mixed_r04_20261026.json)。

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
