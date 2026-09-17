# Skill3D

在 **VSI-Bench**（Visual-Spatial Intelligence Benchmark）上，一个 3D 空间感知 VLM 自演进
智能体：**在线小模型 Qwen3-VL-8B 确定性执行 + 离线强模型 GPT-6 归纳治理**，自动从轨迹中
归纳并准入跨任务 Memory/Skill（`系统架构.md` §0.1）。

本仓库按 `系统架构.md`（v1.0，1805 行实现规格）实现；原 Skill-3D 论文代码保留在
`legacy/` 仅作参考（见 [`legacy/MIGRATION.md`](legacy/MIGRATION.md)），新系统不 import 它。

## 快速开始

```bash
pip install -e . --no-build-isolation        # 本机无网络时加 --no-deps
pytest                                       # 单测 + 集成测试（无需 GPU / 数据集）
```

三条 §13.5 入口：

```bash
# P1 批量 3D 重建（需 VGGT 权重 + 原始视频；缺则明确报错，不伪造产物）
python -m skill3d.reconstruction.run --split induction,inner_validation,outer_holdout --method vggt

# P2 在线评测（-h 看全部参数）
python -m skill3d.online.eval --split inner_validation --source synthetic --mode mock_light
python -m skill3d.online.eval --split test --active-snapshot data/active_snapshot.json   # 需 --allow-final-test

# P3 离线候选优化循环（L1→L2→L3 → 准入 → 原子 promote）
python -m skill3d.evolution.optimize_loop --root-candidate-id <id> --candidates data/candidates.jsonl
```

## 当前可跑 / 待接入（诚实边界）

| 能力 | 状态 | 条件 |
|---|---|---|
| 在线链 M1→M13 端到端（含 AST→沙箱→几何校验→评测→trace） | ✅ 可跑 | `--mode mock_light`（合成输入 + 确定性 stub program） |
| 同 seed 重放字节级一致（§4 M8/M17 验收） | ✅ 可跑 | `--deterministic-replay` |
| 输入/重建质量门禁 G1–G11、MRA/Accuracy、准入硬门、promote 原子切换 | ✅ 可跑 | 纯 CPU 确定性实现 |
| 真实重建（VGGT / DUSt3R / COLMAP）、SAM2 绑定 | ⏳ 待接入 | `TODO_USER_INPUT`：权重 + 原始视频 |
| 在线 program 生成（Qwen3-VL-8B） | ⏳ 待接入 | `bash scripts/serve_qwen3vl_dp8.sh` 起本地 vLLM，再 `--mode real --vllm-endpoint ...` |
| GPT-6 归纳/修订（M16） | ⏳ 待接入 | `TODO_USER_INPUT`：`SKILL3D_GPT6_ENDPOINT/MODEL_ID/API_KEY` |
| VSI-Bench 数据与四层 split | ⏳ 待接入 | HF meta 仅给 QA；视频需自备（见 `adapters/episode_source.py` 的 `jsonl` 格式） |

> **mock_light 不是实验结果**：合成输入 + stub program 只用于管道验证与验收
> （§9.2 / §12 M0-M1）。准入门强制 `real`（§5.6b），mock 跑不会 promote 任何候选。

## 目录结构（§2）

```
configs/                 Hydra 风格 YAML（本机无 hydra 时用 yaml 直读，见 online/config.py）
  config.yaml / vsi_bench_split.yaml / qwen3vl_dp8.yaml / sandbox_docker.yaml
  admission_thresholds.yaml          准入门阈值（全部 TODO_CALIBRATE）
docker/                  Dockerfile.sandbox（只读根 FS / --network none / 不挂 GPU）
                         Dockerfile.vllm（Qwen3-VL-8B）
scripts/serve_qwen3vl_dp8.sh           Qwen3-VL-8B DP×8 启动（不 TP，ADR-2）
src/skill3d/
  adapters/    M1 VSI-Bench adapter + episode 数据源（vsi_bench / jsonl / synthetic）
  gates/       M2 输入门禁与 IQA 算子
  reconstruction/   M3 VGGT 主线 + DUSt3R/COLMAP 备选 + 尺度锚定（run.py = P1 CLI）
  reconstruction_gate/  M4 G1–G11 / 置信度 / SceneState
  segmentation/ M5 SAM2 视频分割与对象绑定
  tools/       M6 Tool Registry（真实/三档 Mock，强制 source 元数据）
  routing/     M7 题型识别 + Skill 硬过滤/检索
  synthesis/   M8 program 合成（prompt/程序组装/vLLM 客户端）
  sandbox/     M9 AST 白名单 + M10 持久 kernel/容器 + receipt 哈希链
  verifier/    M11 确定性几何校验（不调 VLM）
  evaluation/  M12 Accuracy / MRA
  trace/       M13 Trace Store（JSONL→Parquet）
  memory/      M14 三层记忆 + 巩固/防污染
  skills/      M15 Skill Registry 语义版本 + 原子 promote/回滚
  governance/  M16 GPT-6 离线归纳/修订/审查（**仅离线**）
  evolution/   M17–M19 演进沙箱/配对评分/反例/优化循环（optimize_loop.py = P3 CLI）
  scheduling/  M20 8×4090 DP 调度（DP×8 不 TP）
  infra/       M21 Hydra/MLflow/版本锁定/断点恢复
  schemas/     §5 Pydantic v2 单一事实源（54 个模型，extra="forbid"）
  fsm/         §6 在线/离线状态机
  online/      在线链编排（runner= M1→M13 driver，eval.py = P2 CLI，synthetic= mock_light）
tests/unit/ tests/integration/        §2 测试层级（golden/fuzz 待补）
data/                                 不入库（§2）；trace/重建产物落此处
legacy/                               原 Skill-3D 代码（参考，不参与新系统）
```

## 硬约束（红线摘要，全文见 §0.2）

1. **在线链绝对无 GPT-6**——`governance` 只能离线 import；静态守卫见
   `tests/unit/test_no_gpt6_online.py`（扫描 `src/skill3d` 全部在线目录）。
2. **GPT-6 仅离线**：归纳 candidate_v0 / 产 patch / 语义审查；不执行分支、不评分、不决定 promote。
3. 一切阈值标 `TODO_CALIBRATE`，GPT-6 参数标 `TODO_USER_INPUT`，不得当实测结论。
4. 重建先于 Skill 路由；Tool 只经 SceneState 句柄访问产物；Tool 预封装、不每题生成。
5. 候选不可变；promote 原子切换可回滚；paired A/B 复用同一 ReconstructionArtifact。
6. **Final test 完全隔离**：默认拒绝进在线链与批量重建（需显式 `--allow-final-test`，仅盲评一次）。
7. 几何校验为确定性硬门（不引入第二 VLM 当裁判）。

## 复现与版本锁定（§13.6 / §16.4）

`RunManifest` 记录 `code_commit / docker_digest / checkpoint_sha256 / pip_freeze_hash /
config_hash`；`EvaluationRun` 带 `code_commit` 与 `active_snapshot_ref`。
`--deterministic-replay` 下时间戳/耗时/id 取确定性占位，保证同 seed 字节级一致。

```bash
git rev-parse HEAD > code.commit
docker image inspect --format '{{.Id}}' skill3d-sandbox:latest > docker.digest
pip freeze > requirements.lock
```

## 待办（按文档标注）

- `TODO_USER_INPUT`：GPT-6 endpoint/model_id/auth；VSI-Bench 原始视频；VGGT-1B-Commercial
  checkpoint；SAM2 checkpoint；是否用 Qwen3-VL Thinking 变体。
- `TODO_CALIBRATE`：`configs/admission_thresholds.yaml` 全部阈值、G1–G11 门禁阈值、
  检索 top-k、`N_min`、split 比例、沙箱超时。
- 待补测试层级（§2）：`tests/golden/evolution/`（A/B 同 artifact、冻结差分、幂等重放）、
  `tests/fuzz/metamorphic/`（hypothesis + 5 类空间变换）。
