# 交接提示词（Skill3D v5 / v5.1，2026-09-20）

把下面整段复制给下一个 agent 作为起始指令。

---

你是接手的 Coding Agent，负责继续推进 Skill3D 项目（目标：在 VSI-Bench 上做出**可信、可复现、能进论文主表**的结果）。项目已有相当多实码与真实 GPU 证据，**先读完手上的材料再动手**，不要从零重做，也不要重跑已经跑通的东西。

## 0. 身份与纪律（先立规矩）

- 工作目录：`/nas/wangjh/harness3d/skill3d_codebase`（git repo，工作区 dirty，**不要**随便 `git checkout`/`reset`）。
- 目标规格：`/nas/wangjh/harness3d/系统架构5.md`（v5.0，唯一目标态）；对 HC32 的正式修订见 `docs/decisions/D-2026-09-20-calibration-dataset-policy.md`（v5.1，用户已拍板）。实现对照与实况：`architecture_v5_implementation_report.md`。
- 硬纪律：**不得**把 Conditional Go / 待核验 / mock 结果写成已验证事实；**不得**为了分数放宽 fail-closed 门槛；`paper_eligible` 必须有数据隔离 + ≥3 seed + 统计门 + 非 mock 证据；未跑通的一律如实写 blocker。在线链绝对无 GPT-6（GPT-6 仅离线）。
- 任何架构/口径层面的新决定，都要在 `docs/decisions/` 下写一条决策记录，并同步更新 `architecture_v5_implementation_report.md`。
- 当前测试基线：`642 passed, 4 skipped`；`python -m pyflakes src` 的 undefined-name 必须保持 0。**回归即修，不允许删测试/放宽断言换假绿。**

## 1. 环境（必须用冻结环境，别用 base）

```bash
source /home/cvailab/anaconda3/etc/profile.d/conda.sh && conda activate skill3d-exp   # py3.11
export PYTHONPATH=third_party/vggt:src
export HF_HUB_OFFLINE=1
python -m pytest -q -p no:cacheprovider          # 642 passed / 4 skipped
python -m pyflakes src | grep -c "undefined name" # 0
```
注意：base env 是 py3.12，跑批**必须**激活 `skill3d-exp`，否则 RunManifest 里记录的环境与实际执行不符（上一轮踩过）。

## 2. 当前正在运行的服务与卡占用（接手时先确认还活着）

- vLLM（Qwen3-VL-8B-Instruct-FP8，vllm 0.19.1）DP=3 常驻：端口 8100/8101/8102 分别对应 **模型名 `qwen3vl-8b-r0` / `r1` / `r2`**（名字按 rank 绑定，传错模型名会 404——上一轮踩过）。启动脚本 `scripts/serve_qwen3vl_dp8.sh`。
- M5 检测器兜底：组内 GroundingDINO 服务 `http://127.0.0.1:20022`，已在 `configs/config.yaml` 的 `detection.endpoint` 配置，代码会自动写入 `SKILL3D_DETECTOR_ENDPOINT`。
- 卡：8×RTX4090，用户授权占 6 张、**留 2 张**。当前占用 GPU5/6/7 = vLLM，GPU0 用于重建（VGGT），GPU1 用于评测（SAM2），GPU4 空闲备用；GPU2/3 有其他租户，**不要动**。开始任何 GPU 跑批前先 `nvidia-smi` 确认。

## 3. GPT-6（仅离线）：已配置，但很慢

- 凭据：`configs/gpt6.local.env`（0600，已在 `.gitignore` 的 `*.local.env`；**严禁入库**）。变量 `SKILL3D_GPT6_ENDPOINT/MODEL_ID/API_KEY`，模型 `gpt-5.6-sol`，中转站。
- 实测：单次极短调用约 **243 秒**。客户端默认超时已改为 600 秒，并提供 `chat_many(prompts, concurrency=...)` 并发。**离线演进必须批量 prompt + 并发**，不要逐条串行。
- 用途仅限：从 induction 轨迹归纳 Memory/Skill candidate、基于反例产 patch、语义审查写 `SkillGovernanceDecision`。不得进在线链。
- 客户端代码：`src/skill3d/governance/gpt6_client.py`；测试 `tests/unit/test_gpt6_client_latency.py`。

## 4. 评测口径（用户已定，不要擅自改）

- **数据集只跑 scannet + scannetpp**（ARKitScenes 暂不用；用户可能在拿数据）。
- 采样：`--datasets scannet,scannetpp --sampling-per-task N`（每题型均匀采样，meta 顺序先到先得，确定可复现）。recon 与 eval **必须用同一组采样参数**（loader 已做两段式：抽样只由 split/datasets/question_types/sampling/limit 决定，`max_per_scene` 只影响抽帧）。
- 分工：`inner_validation` 用于定策略/调参；`outer_holdout` 只用于验证，**不得**看 outer 结果回头改策略。`final_test` 完全隔离（仅论文盲评一次）。
- **确定性**：对比实验必须 `--seed 0` + **单 endpoint**。多副本轮询 + 连续批处理会带来数值差异（实测同一 32 题样本两次跑出 20.94 vs 14.38 的摆动）；单 endpoint + 请求级 seed 时两次运行逐题 8/8 一致。多副本只用于吞吐。
- 样本清单：`data/v5_scoped/sample.jsonl`（32 题 = 8 题型 × 4，7 个 scene）。

## 5. 已跑出的真实数字（别重复劳动，也别误报）

`outer_holdout`，32 题（8 题型 × 4），确定性配置，冻结环境：

- **C0 直答 VLM：Avg 46.88**（counting 67.5 / abs_dist 42.5 / size 72.5 / room 67.5 / rel_dist 25 / rel_dir 25 / route 50 / appearance 25）
- **C1 我们的系统：Avg 11.56**（counting 42.5 / 其余 0~25）

结论要诚实记住：**目前"用工具/程序"这条路整体是负收益，只有计数题在修复后接近直答水平（4 题抽查 5.00 → 77.50）**。任务级答案路由（`--direct-answer-tasks`，策略冻结在 `data/v5_scoped/policy_v1.json`）能把分数追平到 46.88，但那是"退回直答"，不是系统贡献。论文要立住，必须让程序路径在某些题型上**真正超过**直答。

## 6. 上一轮修出来的真实缺陷（别再踩，也别回退）

1. C0 直答在 real 模式**只发文本不发图** → 全 0。已修（`_direct_vlm_answer` 共用统一 FrameSet）。
2. `_best_effort_answer` 同样是纯文本，且会**覆盖程序里已经算对的工具结果**。已修（带图）。
3. M5 对象绑定：只探第 0 帧 + VLM 定位不可靠 → 对象清单为空 → 对象类 Tool 被整场景收回。已修：4 探测帧 + GroundingDINO 与 VLM 框**取并集** + 传播补 bf16 autocast。
4. 工具集缺"列举/计数"能力（只有布尔 `exists_in_scene`），模型拿它当计数器只能得 0/1。已补 `list_objects` / `count_objects`（`src/skill3d/tools/geometry_tools.py`）。
5. P2 不复用 P1 落盘 artifact → 重复重建；已改成复用 + 帧集哈希校验。
6. `--question-types` 在 vsi_bench 源被静默忽略；已修。
7. P1 会把整个 split 的所有 episode 全量抽帧（induction 1993 条 → ~58 GB 内存）→ 已改为每 scene 只抽一份（`--episodes-per-scene`，默认 1）。
8. 官方 VGGSfM BA 已在 24 GiB 上被 OOM 否决，生产入口 hard-disable（`reconstruction/legacy_vggsfm_ba/`），默认 `recon_method=vggt`。`vggt_sparse_ba` 前端已通（2979 匹配/2284 内点/172 track）但 L1 被三角化 0 点卡住、已按 §10.1 记 `rejected_on_l1_gate`，**不是** OOM。

## 7. 你接手后的优先级（按这个顺序做）

1. **提升 MCA 程序路径**（唯一不依赖外部数据、能自己涨分的部分）。现在 4 个 MCA 题型：rel_dir / rel_dist 掉到 0、route / appearance 各 25，而 C0 是 25~50。做法：从 `data/v5_scoped/full/c1/episode_program.jsonl`（已含 `m8_prompt`、`program_source`、`scene_summary`）与 `program_trace.jsonl`（含逐次 tool 调用与返回值）做失败归因，按类改 prompt / 工具语义 / Skill 模板，每改一版在 `inner_validation` 上量，再在 `outer_holdout` 上验证。
2. **M5 按 scene 缓存**：现在每个 episode 都重跑检测 + SAM2（约 1.5 分钟/题），是扩大样本的瓶颈。对象是 scene 级产物，应缓存复用（注意去重门槛 `[TODO_CALIBRATE]`）。
3. **扩大样本并测噪声底**：`--sampling-per-task` 从 4 提到 10/20；先用同配置重复运行测出噪声底，**小于噪声底的增益不算增益**。
4. **尺度标定（等数据）**：用户将提供 ScanNet++（优先，激光真值更准）或 ScanNet 的非重叠场景米制 GT 位姿与原始 scene ID。收到后：先跑 L0 口径单测与 N=10 诊断档，再上 N∈{30,100} 规模消融（报 median relative error、Spearman ρ、经验覆盖、区间宽度、冲突率、锚点触发率），产出 `<dir>/<dataset>.json` 冻结校准器（`load_calibrator_for` 按数据集选用），然后重跑三个米制题型看 `scale_confidence` 能否按门槛升到 medium。门槛与否决条件见规格 §10.2，**不得**放宽。
5. **离线演进闭环**：跑 induction split 产生 trace → GPT-6 归纳 candidate → 泄漏门 → L1/L2/L3 → 准入 → 在 `inner_validation` 上做 paired A/B（硬约束 18：同 artifact、同 `frame_set_hash`、同卡、同 snapshot；统计用 BCa bootstrap + Wilcoxon + Bonferroni + 效应量；退化样本判不显著）。这是论文的三大创新点之一，目前只有机制与 mock，没有真实数字。
6. 若有余力：`vggt_sparse_ba` L1 的三角化 0 点（修一次，修不好按 §10.1 止损关闭）。

## 8. 常用命令

```bash
# P1 重建（GPU0；注意每 scene 只抽一份帧）
CUDA_VISIBLE_DEVICES=0 python -m skill3d.reconstruction.run \
  --split inner_validation --method vggt --datasets scannet,scannetpp \
  --sampling-per-task 4 --gpus 0 --recon-dir data/v5_scoped/recon

# P2 评测（GPU1；单 endpoint + seed 保证确定性）
CUDA_VISIBLE_DEVICES=1 python -m skill3d.online.eval \
  --mode real --source vsi_bench --split outer_holdout \
  --datasets scannet,scannetpp --sampling-per-task 4 --seed 0 \
  --baseline C1_tools_program \
  --recon-dir data/v5_scoped/recon_outer --recon-method vggt \
  --vllm-endpoint http://127.0.0.1:8100 --vllm-model qwen3vl-8b-r0 \
  --trace-dir data/v5_scoped/full/c1 --memory-dir "" \
  --run-manifest data/v5_scoped/full/c1_manifest.json
# 对照基线：把 --baseline 换成 C0_direct_vlm
# 任务级路由：加 --direct-answer-tasks object_counting,object_abs_distance,...

# 读结果
python -c "import json;from pathlib import Path;print(Path('data/v5_scoped/full/c1/evaluation_run.jsonl').read_text())"

# Readiness 快照
python -m skill3d.readiness.manifest --out data/readiness_manifest.json
```

## 9. 交付标准（Definition of Done）

- 每项改动后：全量 CPU 测试 + pyflakes 全绿；受影响的真实 GPU 链路重跑一次并留 receipt（artifact/trace/manifest）。
- 结论只写在 `architecture_v5_implementation_report.md` 与决策记录里，标明证据路径；负面结果同样要写。
- 不要把 `data/**`、`configs/*.local.env`、任何密钥提交进 git。

---
