# harness3D

harness3D 是以 Skill3D 为基础 codebase、面向 VSI-Bench 的 3D 空间感知 VLM 实验系统。Skill3D 是代码基础，不是当前系统名称。在线链使用本地 Qwen3-VL 生成并执行受限程序，离线链按采集、修订、评测、发布四阶段更新完整 Skill。当前实现以[系统架构 v11](系统架构v11.md)为准；本文保留的服务器核查路径和资源状态是历史记录。

当前 checkout 已删除原 Skill-3D 副本、退役 BA/标定实现、v9/v10 演化与发布链、向量检索和 Memory 平台。历史实现通过对应 Git 提交复现。VSI-Bench 数据读取、split、帧集、VGGT/MoGe-2 重建、SAM2 分割、质量门及已有数据产物保持原实现。

在线入口固定 `C1_tools_program`；旧 C0 直答、C2/C5 手工注入、Memory 参数和检索排序配置已删除。正常运行只读取 v11 active snapshot，B01/B11 与父/候选固定注入由对应实验驱动器执行。

> **当前结论（2026-10-09）**：仓库代码与 v11 启动器已核查，v11 dry-run 已完成，正式 run `v11-s03-seed137` 已按真实入口执行，但在模型许可证门处退出。没有启动 vLLM，没有执行重建、induction、B01/B11 或演化，没有可报告的样本数、分数或 Skill 晋升结果。

## 1. 当前基线与证据口径

| 项目 | 当前值 |
|---|---|
| 系统名称 | `harness3D` |
| 基础 codebase | `Skill3D` |
| 仓库 | `/nas/wangjh/harness3d/skill3d_codebase` |
| 分支 | `main` |
| 服务器核查时 Git commit | `a5f9e65be841c876beb8a586f77869b0188086e3`（当前代码请执行 `git rev-parse HEAD` 核对） |
| 服务器核查时工作树 | 当时已核查为干净 |
| 报告日期 | 2026-10-09 |
| v11 默认配置 | `configs/gpu_experiment_v11.yaml` |
| v11 运行器 | `scripts/run_gpu_experiment_v11.py` |
| v11 配对协议 | `docs/skill_ablation_v11.md` |
| 当前 active snapshot | `S0-v11-contract-repair` |
| active snapshot 文件 | `skill_library/snapshots/active_snapshot.json` |
| active snapshot SHA-256 | `b26e79126b47d9587a30cbcafeb4146929b4dbd84681dd675ede0f04d29960e3` |

本 README 中的“已核查”只表示有命令输出、代码收据或文件哈希支撑；“已缓存”只表示本机有文件；“机制可跑”只表示合同测试或确定性夹具覆盖。只有包含完整输入身份、权重身份、质量确认、运行清单和结果文件的正式 run，才可以被称为正式实验结果。

本次外部实验工作目录为：

```text
/home/cvailab/experiments/wjhICLR-v11-20261009
```

该目录不属于 Git checkout。仓库外的配置副本为：

```text
/home/cvailab/experiments/wjhICLR-v11-config.yaml
```

所有外部绝对路径和本机资源状态都是本次服务器审计记录；在其他机器复现时必须重新核查并重新生成收据。

## 2. 系统边界

### 在线链

在线链从固定 FrameSet 和真实/冻结重建产物开始，经过输入与质量门、对象绑定、题型路由、Skill 检索、Qwen 多模态程序合成、AST/沙箱执行、确定性几何校验和官方评分，最后写入 trace 与运行记录。在线链不得调用 GPT-6，也不得把离线候选、未通过准入的 Skill 或 final split 输入混入正式在线评测。

主要模块边界如下：

- **M1 输入与 FrameSet**：VSI-Bench episode、32 帧均匀采样、帧顺序和 `frame_set_hash`。
- **M2 输入诊断**：被动记录质量，不擅自删帧、换帧或重排帧。
- **M3/P1 重建**：VGGT 主线，保留重建、深度、相机和对象产物身份。
- **M4 质量门**：质量结果为单一事实源；无效、缺失、非有限质量值必须 fail-closed。
- **M5 对象绑定**：SAM2 分割、跨帧传播、世界坐标对象合并和对象清单。
- **M6 工具**：只通过 SceneState/产物合同访问重建结果；缺产物不能静默返回 `False` 或 `0`。
- **M7 方法查找**：按规范题型查找唯一 active Skill，校验源文件与 hash 后完整交付。
- **M8 程序合成**：同一 FrameSet 的图像与文本共同发送给 Qwen，不得静默退化成纯文本。
- **M9/M10 执行**：AST 白名单、受限沙箱/持久 kernel、执行 receipt 和哈希链。
- **M11 校验**：确定性程序/几何校验，不引入第二个 VLM 充当裁判。
- **M12 评测**：Accuracy、官方 MRA 和过程可靠性指标；失败不从分母删除。
- **M13 Trace**：JSONL 轨迹、运行参数、输入身份和结果收据。
- **离线更新**：完整 induction 经验、完整 SKILL.md 修订、父/候选配对评测、原子发布与失败回滚。

### 离线演化链

离线演化只能在 parent trace、质量确认、API 密钥和配对结果等前置条件均满足时运行。候选必须经过固定面板、泄漏检查、质量门、准入门和原子 promote；失败时留下结构化收据，不切换成 mock 候选。当前服务器没有 `DEEPSEEK_API_KEY`，本次 v11 配置保持 `run_evolution=false`。

## 3. 冻结环境与硬件

本机实验环境使用已有 Conda 环境：

```bash
source /home/cvailab/anaconda3/bin/activate skill3d-exp
```

已核查版本：

```text
Python       3.11.14
PyTorch      2.10.0+cu128
CUDA runtime 12.8
torch.cuda   available=True, device_count=8
vLLM         0.19.1
```

v11 默认环境要求见 `configs/gpu_experiment_v11.yaml`：

- Linux。
- 至少 2 张 GPU。
- 选用的 Qwen 卡和 VGGT/SAM2 卡分别至少有 20 GiB 总显存。
- 工作盘至少有 80 GiB 可用空间。
- Qwen 与 VGGT/SAM2 必须分卡；配置中的 `qwen_gpu` 和 `geometry_gpu` 不得相同。
- vLLM 使用单卡 tensor parallel（`tensor_parallel_size=1`），不是把几何重建和 Qwen 放在同一张卡上。

服务器有 8 张 NVIDIA GeForce RTX 4090，每张总显存 24564 MiB。核查时 GPU 均有其他任务占用：GPU 0 约使用 22115 MiB，GPU 1 约使用 23592 MiB，GPU 2-7 约使用 23800 MiB，部分卡利用率很高。因此“机器有足够总 GPU”不等于“本次有空闲 GPU”；本次没有终止或干扰其他进程，也没有在显存不足时强行启动服务。

本次工作目录所在文件系统的 v11 preflight 记录可用空间约 164.08 GiB，高于配置下限。`/nas` 可用空间较少，正式实验产物不应默认写入 `/nas`。

## 4. 数据、分层和许可证

### VSI-Bench 数据

实际视频根目录：

```text
/nas/wangjh/Skill-3D/dataset/VSI-Bench
```

已核查：

| 项目 | 值 |
|---|---:|
| MP4 数量 | 512 |
| MP4 总大小 | 5,734,843,762 bytes（约 5.34 GiB） |
| `scannet` | 312 个视频 |
| `scannetpp` | 50 个视频 |
| `arkitscenes` | 150 个视频 |
| 仓库元数据 | `data/vsi_bench_meta/test.jsonl` |
| 元数据 SHA-256 | `aa3ff8cc2010e555f4f865ff96483812ec0b275005b68562f53bb114ad4db9db` |

仓库的 v11 真实入口只允许使用 `induction` 和 `inner_validation`。本次目标题型为 `object_rel_distance`，配置数据集为 `scannet,scannetpp`，不读取或运行 final split。

“视频文件存在”“视频可解码”“数据使用许可已确认”是三个不同状态。文件数量和路径核查不能替代 VSI-Bench 许可确认。

### 许可证门

正式模式要求外部配置中同时满足：

```yaml
experiment:
  accept_model_licenses: true
  accept_vsi_bench_license: true
```

这些字段只能在真实许可核对完成后设置为 `true`。不能用仓库 LICENSE、模型 README 或目录存在性替代当前使用场景的许可确认，也不能伪造确认文件。本次配置保持：

```yaml
accept_model_licenses: false
accept_vsi_bench_license: false
```

因此正式入口在启动任何 GPU 阶段之前拒绝运行。

## 5. 模型与权重

权重优先复用服务器已有缓存。正式运行器在现有路径可验证时不会重新下载；缺失时才会按配置中的 pinned revision 下载到独立 work-dir。不得覆盖或移动既有权重。

本次只读审计清单：

```text
/home/cvailab/experiments/wjhICLR-v11-20261009/weights/weight_manifest.json
```

清单标记为 `audited_existing_only`，`download_performed=false`，清单 SHA-256 为：

```text
68e9c652b4ef97360ee9e46745743a4d862ff64fd3745239bb9d482060c8f35fe
```

### Qwen3-VL-8B-Instruct-FP8

- Repo：`Qwen/Qwen3-VL-8B-Instruct-FP8`
- Revision：`9cdc6310a8cb770ce18efaf4e9935334512aee45`
- Snapshot：
  `/home/cvailab/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct-FP8/snapshots/9cdc6310a8cb770ce18efaf4e9935334512aee45`
- `refs/main` 与配置 revision 一致。
- 两个主要 safetensors 分片：
  - `model-00001-of-00002.safetensors`：`e2dea2e85e643ef7045c31a485a426631a1e0a84e463f8b2c7e9c6682a06eafd`
  - `model-00002-of-00002.safetensors`：`3dc64ec934af27a7007d265014e907aff9e16658d409e5cd4cf79869bf2bd8c1`

### VGGT-1B

- Repo：`facebook/VGGT-1B`
- 配置 revision：`860abec7937da0a4c03c41d3c269c366e82abdf9`
- 缓存：`/home/cvailab/.cache/huggingface/geothinker/VGGT-1B`
- 缓存存在，`model.safetensors` 大小为 `5026367224` bytes。
- `model.safetensors` SHA-256：
  `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`
- 该目录没有 `refs/main` 或 snapshot revision 元数据；审计清单中的 `actual_cache_revision` 为空，`revision_match=false`。

因此只能说 VGGT 权重文件已存在，不能说它已被证明与配置 pinned revision 完全匹配。不能在正式结果中把它当作 revision 已闭合的权重。

### SAM2.1 Hiera-Large

- Repo：`facebook/sam2.1-hiera-large`
- Revision：`227a114a2f535cd147f82442e7d2038cdd2e5d68`
- Snapshot：
  `/home/cvailab/.cache/huggingface/hub/models--facebook--sam2.1-hiera-large/snapshots/227a114a2f535cd147f82442e7d2038cdd2e5d68`
- `refs/main` 与配置 revision 一致。
- Checkpoint：`sam2.1_hiera_large.pt`
- Checkpoint SHA-256：
  `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`
- 配置文件使用该 snapshot 内的 `sam2.1_hiera_l.yaml`，不依赖仓库中不存在的相对路径。

### MoGe-2 ViT-L

MoGe-2 已下载，之前的“未找到”结论已更正。当前缓存为：

- Repo：`Ruicheng/moge-2-vitl`
- Revision：`39c4d5e957afe587e04eec59dc2bcc3be5ecd968`
- Snapshot：
  `/home/cvailab/.cache/huggingface/hub/models--Ruicheng--moge-2-vitl/snapshots/39c4d5e957afe587e04eec59dc2bcc3be5ecd968`
- `refs/main` 与配置 revision 一致。
- 权重：`model.pt`
- 大小：`1305030700` bytes
- 权重 SHA-256：
  `3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5`
- 本环境包版本：`moge 2.0.0`
- 模型 README 标记许可证为 MIT。

但 v11 默认配置仍是：

```yaml
moge2:
  enabled: false
  existing_path: ""
```

所以 MoGe-2 没有参与本次 dry-run 或正式失败 run，也没有被写入本次 v11 pipeline 的权重 manifest。若未来启用，只能在仓库外配置副本中填写上述 snapshot 路径并重新生成完整收据；不得把缓存存在写成“已参与本次实验”。

## 6. v11 配置层次

v11 有三个不同角色的配置，不能混用：

1. **启动器默认配置**：`configs/gpu_experiment_v11.yaml`。它固定代码要求、模型 repo/revision、资源下限、运行预算和安全默认值。
2. **系统运行时配置**：`configs/config.yaml`。重建、在线评测和 B01/B11 子程序通过 `--config` 读取它，包括 FrameSet、VSI 元数据、检索和沙箱合同。
3. **正式实验配置副本**：必须放在仓库外，例如本次的 `/home/cvailab/experiments/wjhICLR-v11-config.yaml`。它可以填写本机 `existing_path`、`video_root`、质量确认、端口和 GPU，但不得改写仓库程序的权重默认路径，也不得修改模型、Skill、质量门、求解预算或协议版本。

本次外部配置的关键值：

```yaml
runtime:
  qwen_gpu: 0
  geometry_gpu: 1
  port: 18110
  served_model_name: qwen3vl-v11
  max_model_len: 32768
  max_pixels: 131072
  n_images: 32

experiment:
  formal: true
  question_type: object_rel_distance
  datasets: scannet,scannetpp
  seed: 137
  induction_limit: 12
  inner_limit: 8
  reconstruction_sampling_per_task: 24
  run_evolution: false
```

端口必须先检查；冲突时只改外部副本并记录新端口。Qwen 服务和 VGGT/SAM2 必须分卡。实验目录必须是独立 work-dir，正式 run ID 不能复用已有目录。

## 7. v11 运行流程

### 7.1 环境检查

```bash
source /home/cvailab/anaconda3/bin/activate skill3d-exp
pwd
git rev-parse HEAD
git status --short
nvidia-smi
df -h
which python
which vllm
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

在正式模式下还要确认：真实 VSI-Bench 视频、至少两张空闲且显存足够的卡、空闲端口、有效质量确认 JSON、模型与数据许可证已经真实核对。

### 7.2 Dry-run

```bash
cd /nas/wangjh/harness3d/skill3d_codebase
source /home/cvailab/anaconda3/bin/activate skill3d-exp
PYTHONPATH=src python scripts/run_gpu_experiment_v11.py \
  --config /home/cvailab/experiments/wjhICLR-v11-config.yaml \
  --work-dir /home/cvailab/experiments/wjhICLR-v11-20261009 \
  --run-id v11-preflight \
  --dry-run
```

dry-run 应只打印将执行的命令并生成 `pipeline_manifest.json`，不下载权重、不启动 vLLM、不执行重建或评测。命令链应是：

```text
reconstruction
serve_qwen
parent_learning        # induction
b01_b11                # inner_validation
[evolution]            # 仅当 run_evolution=true 且前置条件满足
```

命令链不得包含 final split 或任何最终评测阶段。dry-run manifest 中应确认 `final_test_touched=false`。

### 7.3 正式 run

```bash
PYTHONPATH=src python scripts/run_gpu_experiment_v11.py \
  --config /home/cvailab/experiments/wjhICLR-v11-config.yaml \
  --work-dir /home/cvailab/experiments/wjhICLR-v11-20261009 \
  --run-id v11-s03-seed137
```

运行器会按顺序执行重建、Qwen 服务、induction 的 parent learning 和 inner validation 的 B01/B11。任何前置或阶段错误都写入该 run 的 `failure.json` 并停止，不自动切换 mock、不替换权重、不缩小分母、不重跑到得到更高分。

演化只有在以下条件同时满足时才可以打开：

- 质量确认有效并包含当前合同 SHA-256 与 `evidence_ref`；
- 模型和 VSI-Bench 许可已经真实确认；
- `DEEPSEEK_API_KEY` 已配置；
- B01/B11 正式阶段完成且结果完整；
- `run_evolution=true` 写在外部配置副本中。

否则必须保持 `run_evolution=false`，先完成或报告正式 B01/B11。

## 8. v11 B01/B11 协议

详细合同见 [`docs/skill_ablation_v11.md`](docs/skill_ablation_v11.md)。当前冻结协议的要点：

- 默认题型：`object_rel_distance`，S03。
- 两臂都通过真实 runner、同一题单、同一输入和同一质量门。
- B01 的交付 Skill 集合为空。
- B11 默认加载 `S0-v11-contract-repair` 的 snapshot、manifest 和正文。
- 当前入口不读取或修改 active pointer，不使用 v10 的 `evaluation_binding`。
- 两臂按顺序运行，当前不是延迟公平性实验。
- `seed` 进入准备阶段和模型请求，但不同 GPU 调度不承诺逐位一致。
- M11 通过后才评分；程序/格式失败、无答案或质量拒绝按协议计分，不能从分母删除。
- 服务失败保留 `score=null` 并把实验标为未完成，不能使用部分分数冒充完整结果。

典型输出包括：

```text
manifest.json
skills/
frozen/
preparation/<qa-hash>/model_requests.jsonl
arms/B01/<qa-hash>/result.json
arms/B11/<qa-hash>/result.json
paired_results.jsonl
summary.json
```

`summary.json` 应记录总体和题型级 Accuracy、MRA、合法答案率、程序错误率、最终运行错误率、输入错误率以及 B11-B01。`paired_results.jsonl` 必须按 `qa_id` 严格配对。只有两臂都完成的同题才用于差值；平分和负差值都是有效结果。

## 9. 质量合同与正式结果资格

查看当前质量合同：

```bash
PYTHONPATH=src python scripts/run_skill_ablation_v11.py --print-quality-contract
```

当前代码打印的质量合同 SHA-256：

```text
d877a72cc1d162d94e0453db5c08886f7c527dcfaf29d751e843e00cbcd5e678
```

正式质量确认 JSON 必须是已有真实证据对应的文件，例如：

```json
{
  "confirmed": true,
  "sha256": "d877a72cc1d162d94e0453db5c08886f7c527dcfaf29d751e843e00cbcd5e678",
  "evidence_ref": "已有门限验证记录的位置"
}
```

该文件只确认合同身份和已有验证证据，不改变代码门限。缺文件、SHA 不匹配、`confirmed` 不是 `true` 或缺少 `evidence_ref` 时，不能声称 `formal_result_eligible=true`。本次没有这样的确认文件，因此本次没有正式结果资格。

机制已实现或合同测试通过，不代表真实 GPU、真实视频、真实模型和完整质量合同已经闭合，也不代表可以写入论文主表。

## 10. 本次 v11 实验审计记录

### Dry-run

执行了：

```bash
PYTHONPATH=src python scripts/run_gpu_experiment_v11.py \
  --config /home/cvailab/experiments/wjhICLR-v11-config.yaml \
  --work-dir /home/cvailab/experiments/wjhICLR-v11-20261009 \
  --run-id v11-preflight --dry-run
```

收据：

```text
/home/cvailab/experiments/wjhICLR-v11-20261009/runs/v11-preflight/pipeline_manifest.json
```

dry-run 状态为 `dry_run`，记录的代码 commit 与当前 HEAD 一致，预检记录 `final_test_touched=false`。它只打印了 reconstruction、Qwen 服务、parent learning 和 B01/B11 命令，没有实际执行这些阶段。

### 正式 run

执行了：

```bash
PYTHONPATH=src python scripts/run_gpu_experiment_v11.py \
  --config /home/cvailab/experiments/wjhICLR-v11-config.yaml \
  --work-dir /home/cvailab/experiments/wjhICLR-v11-20261009 \
  --run-id v11-s03-seed137
```

收据：

```text
/home/cvailab/experiments/wjhICLR-v11-20261009/runs/v11-s03-seed137/failure.json
```

退出码为 `2`，实际错误为：

```text
GPUExperimentError: formal run requires experiment.accept_model_licenses=true
```

失败发生在许可证检查阶段，早于权重下载、GPU inventory 选择、重建和 vLLM 启动。最终状态：

| 阶段 | 状态 |
|---|---|
| reconstruction | 未执行 |
| Qwen/vLLM | 未启动 |
| parent learning / induction | 未执行 |
| B01/B11 / inner_validation | 未执行 |
| evolution | 未执行 |
| 正式评分 | 未生成 |
| active snapshot | 未变化 |

因此本次没有：

- B01/B11 样本数；
- Accuracy、MRA 或 B11-B01 得分；
- `summary.json` 或 `paired_results.jsonl`；
- parent learning trace；
- candidate、decision、publication、post-publish 或 rollback 收据；
- candidate promote 或 active snapshot 变化。

本次不能把旧目录、历史质量数字或 dry-run 命令输出当作 v11 正式结果。

## 11. 关键硬约束

1. 在线链绝不调用 GPT-6；强模型只属于离线归纳治理链。
2. 正式 run 必须从干净、固定 commit 的 checkout 启动。
3. Qwen 与 VGGT/SAM2 必须分卡；不能因为显存不足而同卡运行。
4. 全链路使用固定 32 帧和同一帧身份；M2 不能偷偷替换或删除帧。
5. 质量结果必须有明确状态；缺失、NaN、合同不匹配时 fail-closed。
6. Tool 缺产物必须抛出结构化错误，不能静默返回零值。
7. M8 必须把同一 FrameSet 的图像和文本一起交给模型，不能静默退化成纯文本。
8. M11 是确定性硬门，不用第二个 VLM 作为裁判。
9. Skill 候选不可变；promote 必须原子切换并可回滚。
10. 配对实验复用同一题单、同一输入和同一基础产物，失败题仍留在分母。
11. 本 v11 入口只允许 induction 和 inner_validation；最终评测是独立动作，不在本入口中运行。
12. 不得用 mock、替代权重、伪造许可、伪造质量确认或缩小分母绕过失败。

## 12. 当前已知限制与待办

以下事项在当前服务器核查中仍未闭合：

- **模型许可证确认**：`accept_model_licenses` 仍为 `false`。
- **VSI-Bench 许可证确认**：`accept_vsi_bench_license` 仍为 `false`。
- **质量确认**：没有与当前合同 SHA-256 匹配且带 `evidence_ref` 的确认 JSON。
- **GPU 调度**：所有 GPU 在核查时均被其他任务占用，没有适合正式启动的空闲卡。
- **VGGT 身份**：文件存在，但缓存没有可验证的 pinned revision metadata。
- **MoGe-2 使用状态**：缓存已存在且 revision 匹配，但 v11 默认 disabled，尚未参与本次实验。
- **正式 B01/B11**：尚未执行，因此没有正式配对分数。
- **演化**：未执行；`DEEPSEEK_API_KEY` 未配置，且正式 B01/B11 和质量/许可门尚未通过。
- **尺度标定**：若没有合规的非重叠 ARKitScenes 标定证据，未标定题型应保持低置信度，不得用默认常量升级。
- **论文资格**：没有本次正式结果，不能写入论文主表或宣称 Skill gain。

## 13. 测试与复现

合同测试用于检查协议、隔离、工具、评分和收据逻辑，不等于真实模型效果。v11 相关测试可按文档执行：

```bash
source /home/cvailab/anaconda3/bin/activate skill3d-exp
cd /nas/wangjh/harness3d/skill3d_codebase
PYTHONPATH=src python -m pytest -q \
  tests/integration/test_skill_ablation_v11.py \
  tests/integration/test_v11_skill_delivery.py \
  tests/unit/test_m11_submission.py \
  tests/unit/test_v11_prompt_contract.py \
  tests/unit/test_v11_tool_contract_repair.py
```

建议每次正式 run 都保存：

```text
git rev-parse HEAD
git status --short
python/vllm/torch/CUDA 版本
nvidia-smi
配置文件 SHA-256
代码 commit
权重逐文件 SHA-256
输入元数据和视频根目录身份
quality contract 与 confirmation
pipeline_manifest.json 或 failure.json
```

不要使用 `git reset --hard`、`git checkout --` 或删除用户数据来“清理”实验；不要覆盖已有权重。实验产物应写入独立 work-dir，Git checkout 只读使用。

## 14. 目录索引

```text
configs/
  config.yaml                    运行时主配置
  gpu_experiment_v11.yaml        v11 启动器默认配置
  vsi_bench_split.yaml           VSI-Bench 分层配置
scripts/
  run_gpu_experiment_v11.py     v11 dry-run/正式入口
  bootstrap_and_run_v11.sh      固定 commit 的隔离 bootstrap 入口
  run_skill_ablation_v11.py     B01/B11 配对入口
  run_evolution_campaign_v11.py 演化 campaign 入口
src/skill3d/
  adapters/                      VSI-Bench、FrameSet、split 和 episode 输入
  reconstruction/                VGGT、几何、尺度和重建产物
  reconstruction_gate/           M4 质量合同与 SceneState
  segmentation/                  SAM2 对象绑定
  tools/                         Tool Registry 和 fail-closed 合同
  routing/                       题型、Skill 路由和检索
  synthesis/                     多模态程序合成
  sandbox/                       AST、执行环境和 receipt
  verifier/                      M11 确定性校验
  evaluation/                    Accuracy、MRA、v11 配对评分
  trace/                         Trace Store
  skills/                        完整方法加载、交付与原子发布
  governance/                    离线模型客户端
  evolution/                     v11 四阶段更新与真实执行适配
docs/skill_ablation_v11.md       v11 B01/B11 详细合同
docs/reports/                    v11 运行诊断报告（正确率与证据链分析）
```

`src/skill3d/legacy/readers.py` 仍是当前重建产物的版本校验入口，保留其读取与拒绝旧产物的行为；它不是旧求解器或重建实现。文件名含 v3/v6/v9 的测试只要仍约束当前数据和工具合同，就继续执行。

## 15. 证据索引

本次审计使用的主要外部收据：

- 环境核查：`/home/cvailab/experiments/wjhICLR-v11-20261009/environment_check.txt`
- 外部配置：`/home/cvailab/experiments/wjhICLR-v11-config.yaml`
- dry-run pipeline：`/home/cvailab/experiments/wjhICLR-v11-20261009/runs/v11-preflight/pipeline_manifest.json`
- 正式失败：`/home/cvailab/experiments/wjhICLR-v11-20261009/runs/v11-s03-seed137/failure.json`
- 权重清单：`/home/cvailab/experiments/wjhICLR-v11-20261009/weights/weight_manifest.json`
- VSI 视频：`/nas/wangjh/Skill-3D/dataset/VSI-Bench`
- VSI 元数据：`data/vsi_bench_meta/test.jsonl`
- 当前质量合同入口：`scripts/run_skill_ablation_v11.py --print-quality-contract`

这些收据证明本次做过哪些核查和哪些阶段没有执行；它们不产生不存在的分数，也不替代正式许可、质量确认或未来可用 GPU。
