# harness3D

harness3D 是以 Skill3D 为基础 codebase、面向 VSI-Bench 的 3D 空间感知 VLM 实验系统。Skill3D 是代码基础，不是当前系统名称。在线链统一使用百炼 Qwen3.8-Flash 生成并执行受限程序、解析题目并审核对象语义；离线链按采集、修订、评测、发布四阶段更新完整 Skill。当前实现以[系统架构 v11](系统架构v11.md)为准；本文保留的服务器核查路径和资源状态是历史记录。

专题说明、旧版架构与历史实验报告统一从 [文档索引](docs/README.md) 查阅。

当前 checkout 已删除原 Skill-3D 副本、退役 BA/标定实现、v9/v10 演化与发布链、向量检索和 Memory 平台。历史实现通过对应 Git 提交复现。VSI-Bench 数据读取、split、帧集、VGGT/MoGe-2 重建、SAM2 分割、质量门及已有数据产物保持原实现。

在线入口固定 `C1_tools_program`；旧 C0 直答、C2/C5 手工注入、Memory 参数和检索排序配置已删除。正常运行只读取 v11 active snapshot，B01/B11 与父/候选固定注入由对应实验驱动器执行。

Harness3D 同题 fit-set 入口为 `scripts/run_fitset_v11.py`。每槽位由唯一 spawn
worker 串行消费，12 条新合格轨迹触发修订，经双 seed 严格 A/B 后才发布到独立
run-local library。ready_579 覆盖 16 槽位，米制扩展覆盖完整 19 槽位；最终冻结
每槽位最后发布版本，5 个不可达 g1m1 保留 seed。旧线程并发实验无效且不能恢复，
新实验需先运行 model-free preflight 和 pilot 审计。详见[同题槽位流合同](docs/v11_fitset_slot_stream.md)。

当前在线合同使用 `program_synth_v11_8_en`、
`solver-v11.8-task-tools` 和 `tool-docs-v11.8-task-tools`。类别匹配、对象覆盖、
完整排名验收及检测器重试由两臂共用；评测时原 solver 失败后可预算外追加一次
`eval-visual-v1` 请求，只携带冻结的 32 张原帧、问题、选项和答案格式。该路径不在
`induction` 使用，正常提交但答错也不会触发。当前 active snapshot 为
`S0-v11-task-tools`；质量确认哈希包含相关源码和合同版本，旧确认会被拒绝。详见
[配对运行合同](docs/skill_ablation_v11.md)。

正常模型工具目录为 13 个，按八题型与当前证据／产物／局部条件，每题最多暴露
4–8 个。两个表面距离入口统一为 `object_surface_distance(obj_a, obj_b, unit)`，
`normalized` 只表示重建单位，`m` 额外需要题级尺度授权。旧距离、相机距离、
存在性与简单欧氏距离入口保留内部兼容，正常模型调用会被拒绝。
新快照只迁移两条绝对距离正文至 1.0.1，其余 22 条及历史源 hash 保持；
该接口迁移不构成模型效果或 Skill 演进实验。

对象清单中的 `det_conf` 是检测器原始置信度，`mask_support_ratio` 是非空传播 mask
覆盖率。paired 运行会在全部题完成 M5 后审计跨题异类 mask 冲突；冲突对象仅禁用
对象几何，原图视觉路径仍可用。授权房间题在 B01/B11 前共享一次带身份收据的预计算，
不在两臂内重复拟合，也不计作 Skill 收益。

当前 M5 由百炼 Qwen3.8-Flash 解析题目对象，GroundingDINO 对每类扫描全部可读
冻结帧，逐帧 SAM2 后执行目标盲审与纯文本类别关系判断；只有语义和 mask 质量同时
通过的候选才生成局部点云。保留解析、检测、分割与语义收据，失败不记为对象不存在。
高 IoU 仅在明确不相容关系时触发绑定硬门；局部对象失败只撤销相关结果，不撤全局尺度。
答案解析、提交与评分分别记录，空 GT 保留预测与 null 得分；EpisodeTrace 版本为 9.4。
完整扩容入口 `scripts/run_full_experiment_v11.py` 固定八类各24题，192道唯一题、
两个seed同题单、B01/B11共768次。全部公共准备按场景审计封存后才逐对求解，
首批32臂核查后自动继续，答案全冻结后独立评分。详见[扩大面板合同](docs/skill_ablation_v11.md)。

2026-10-11 已在代码基线 `c020345dca6f3e34acd0883accc4d16956327773`
完成英文面板 256 次求解和 128 组 B01/B11 配对。该实验固定使用
`max_tokens=4096`；当前代码自 `e64fff5ee90713a5e9be5f189044685d0af051e7`
起将后续 Qwen 请求的最大输出改为 8192，两种预算的结果不得静默合并。

GPU 启动入口为 `./start_harness.sh`，与 `start_harness3d.sh` 共用同一流程。
检测器地址按 `HARNESS3D_DETECTOR_ENDPOINT`、原 YAML
`experiment.detector_endpoint`、`http://127.0.0.1:20022` 的顺序解析；
派生 YAML 保存最终地址及 env/yaml/default 来源。显式空串关闭检测器，
需同时设置 `HARNESS3D_REQUIRE_DETECTOR=0`；关闭配置会覆盖子进程继承的旧地址。
M5 由独立的 Qwen3.8-Flash 客户端解析本题对象，GroundingDINO 对每类扫描全部冻结
帧，再由逐帧 SAM2、目标盲审、类别关系判断、反投影和整对象验证生成本题绑定。
解析和本题绑定分层缓存，身份覆盖问题/选项/题型、RGB、重建数组、模型、seed、参数
及源码，引用校验完整 SHA256；可变云模型别名默认禁用跨运行磁盘复用。
旧基础清单不复用；正常重复观测不计作碎片。失败关闭相关对象几何，不代表不存在。
MoGe-2 显式启用后使用与 VGGT 相同的裁剪网格；米制工具需要通过输入、模型、
源数组及收据哈希校验。尺度融合仍属待验证分支，默认不开启。配置、缓存迁移和
证据字段见[度量尺度链路说明](docs/metric_fusion_v11.md)。

公共证据画像仅保留 `geometry_3d`、`metric_scale`，均为 `available/unavailable`。
几何取现有 M4 主门；尺度取现有尺度门前五项，题型授权另算。几何失效连带撤销尺度，
尺度失效保留非米制几何；软告警仅作诊断。固定图片与采样帧不再作为公共证据维度。
世界坐标约定在相关工具局部检查，展示与执行共用授权判定。
M5 提供候选和技术校验，不证明候选是题目所指实例；模型通过帧／裁剪消歧并传入
明确 `obj_id`，唯一候选也不代表指代正确。单对象工具仅校验所选对象，计数／完整
排名另查所需集合覆盖；三维不可用时仍可查看二维候选。M11 不认证指代真伪。
每个题型按初始两项状态建立 `g0m0/g1m0/g1m1` 三个槽位，完整基线共 24 条
方法，快照使用 `runtime-skill-snapshot/3.0` 与 `active_by_slot`。M7.5 后精确选择
一条 Skill，整题保持版本和正文 hash；运行期证据撤销只收回工具和结果，不重检索。
经验包 v11/5.0 记录初始槽位、能力／候选变化、实际对象 ID 和图像／工具观察；
整题归属于初始交付版本，答错不自动归因为绑定错误。修订、评测和发布均限定同槽位。
新源从英文历史源整理形成结构迁移基线，未运行演进实验。CPU 验证不代表模型收益。

> **历史核查结论（2026-10-09）**：当时的正式 run `v11-s03-seed137` 在模型许可证门处退出，未启动 GPU 阶段。下文第 3–10 节保留该次服务器审计记录，不代表当前运行状态。

## 1. 当前基线与证据口径

| 项目 | 当前值 |
|---|---|
| 系统名称 | `harness3D` |
| 基础 codebase | `Skill3D` |
| 仓库 | `/nas/wangjh/harness3d/skill3d_codebase` |
| 分支 | `main` |
| 当前代码提交 | 执行 `git rev-parse HEAD` 核对 |
| 清理前回滚点 | `pre-cleanup-v11-20261012`（`e64fff5ee90713a5e9be5f189044685d0af051e7`） |
| 文档状态日期 | 2026-10-13 |
| v11 默认配置 | `configs/gpu_experiment_v11.yaml` |
| v11 运行器 | `scripts/run_gpu_experiment_v11.py` |
| v11 配对协议 | `docs/skill_ablation_v11.md` |
| 当前 active snapshot | `S0-v11-task-tools` |
| active snapshot 文件 | `skill_library/snapshots/active_snapshot.json` |
| active snapshot SHA-256 | 由 `skill_library/snapshots/active_snapshot.json` 当前内容复算 |

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
- **M4 质量门**：整场 warp、分组重叠、两头相对差 P95、邻域长尾四项全部通过才保留三维路由。质量版本为 `v7-scene-warp-overlap-head-tail`；旧数组可复用，旧质量需重算并另存，缺失或非有限值使整场降级。
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

离线演化只能在 parent trace、质量确认、API 密钥和配对结果等前置条件均满足时运行。
经验按“题型＋初始证据状态＋实际父版本”积累：每新增 12 条合格 induction 轨迹触发
一次候选；拒绝保留累计经验并等待下一批 12 条，晋升则原子换版并由发布后 learning
开启新 buffer。未满 12 条且从未尝试的槽位在轮末执行一次尾批次。候选至少需要 8 个
独立完整配对才可晋升；服务／输入／trace 故障单独记录，不计入 buffer。失败时留下
结构化收据，不切换成 mock 候选。生产入口
`scripts/run_streaming_evolution_v11.py` 逐题冻结父快照和独立 trace，并在晋升后为
下一题重新加载 active。当前服务器没有 `DEEPSEEK_API_KEY`，真实模型配置保持
`run_evolution=false`。

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
xFormers     0.0.34（匹配 PyTorch 2.10.0+cu128）
SAM2         1.1.0（已补建并验证 sam2._C CUDA 后处理扩展）
```

2026-10-12 的环境验收以服务器 `skill3d-exp` 为准。SAM2 扩展安装会启用掩码
补洞，M5 清单缓存已升级为 `m5-inventory-9-sam2-runtime`，缓存键包含包版本和扩展
SHA256；当前缓存为 `m5-inventory-10-frozen-inputs`，同时核验完整输入与模型配置。
MoGe 米制缓存身份也记录 xFormers 版本。历史实验产物保持只读，新环境
验证使用独立工作目录，不能把环境修复解释为 Skill 收益。

v11 默认环境要求见 `configs/gpu_experiment_v11.yaml`：

- Linux。
- 至少 1 张 GPU。
- 选用的 VGGT/SAM2 卡至少有 20 GiB 总显存。
- 工作盘至少有 80 GiB 可用空间。
- 在线模型需要通过环境变量提供 `DASHSCOPE_API_KEY`，密钥不得写入配置或收据。

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

一键实验入口只编排 `induction` 和 `inner_validation`；普通在线入口还支持
`outer_holdout`，并仅在显式授权时允许 `final_test`。已完成的英文正式实验没有访问
`final_test`。

“视频文件存在”“视频可解码”“数据使用许可已确认”是三个不同状态。文件数量和路径核查不能替代 VSI-Bench 许可确认。

### 许可证门

正式模式要求外部配置中同时满足：

```yaml
experiment:
  accept_model_licenses: true
  accept_vsi_bench_license: true
```

这些字段只能在真实许可核对完成后设置为 `true`。不能用仓库 LICENSE、模型 README 或目录存在性替代当前使用场景的许可确认，也不能伪造确认文件。当前示例配置记录为：

```yaml
accept_model_licenses: true
accept_vsi_bench_license: true
```

换机器或改变使用范围时必须重新核对许可，不能沿用该布尔值替代真实授权。

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

### 在线 Qwen

- Provider：阿里百炼 OpenAI 兼容接口。
- Model：`qwen3.8-flash`。
- M5 题目解析、M8 程序生成及评测视觉补答共用同一模型配置。
- API Key 只从 `DASHSCOPE_API_KEY` 读取。

### M5 本地类别审核

- 默认使用原生 `internlm/Intern-Decision-4B`，P1 三图 yes/no，`p(yes) >= 0.77`。
- 只确认 mask 下物体的类别，不输出或推定掩码纯净度、可见性、完整性。
- 无需历史质量收据；类别通过后执行现有点云及绑定验证。
- 服务监听 `127.0.0.1:18445`，权重和运行时独立于主项目 Python 环境。
- `online_model.mask_review: null` 显式切回 Qwen 的盲审、关系和质量准入。
- 部署、调用和真实 M5 检查见 [Intern 脚本说明](scripts/intern_decision/README.md)。

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
  geometry_gpu: 1
  max_model_len: 32768
  max_tokens: 8192
  max_pixels: 131072
  n_images: 32

experiment:
  formal: true
  question_type: object_rel_distance
  evidence_state: g1m0
  datasets: scannet,scannetpp
  seed: 137
  induction_limit: 12
  inner_limit: 8
  min_evaluation_pairs: 8
  evolution_batch_size: 12
  reconstruction_sampling_per_task: 24
  run_evolution: false
```

检测器端口必须先检查；冲突时只改外部副本并记录新端口。实验目录必须是独立 work-dir，正式 run ID 不能复用已有目录。

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
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

在正式模式下还要确认：真实 VSI-Bench 视频、至少一张空闲且显存足够的卡、检测器端口、`DASHSCOPE_API_KEY`、有效质量确认 JSON、模型与数据许可证已经真实核对。

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

dry-run 应只打印将执行的命令并生成 `pipeline_manifest.json`，不下载权重、不执行重建或评测。命令链应是：

```text
reconstruction
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

不依赖 GPU 或模型密钥的流程合同实验使用真实 campaign、静态检查、配对准入、
不可变快照发布和 active pointer 切换，只注入确定性的模型结果：

```bash
PYTHONPATH=src python scripts/run_streaming_evolution_contract_v11.py \
  --output /tmp/streaming-evolution-contract-v11
```

预期尝试序列为 `rejected → promoted → rejected`：前 12 条拒绝后不丢经验，
再新增 12 条以累计 24 条重试并晋升；发布后新版本 case 进入新 buffer；另一槽位
以 5 条尾批次完成拒绝验证。该实验只证明流程合同，不证明 Skill 效果。

## 8. v11 B01/B11 协议

详细合同见 [`docs/skill_ablation_v11.md`](docs/skill_ablation_v11.md)。当前冻结协议的要点：

- 默认题型：`object_rel_distance`，S03。
- 两臂都通过真实 runner、同一题单、同一输入和同一质量门。
- B01 的交付 Skill 集合为空。
- B11 默认加载 `S0-v11-task-tools`，按题型与初始证据状态选择完整正文。
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

质量合同包含相关源码身份，任何代码提交都可能改变其 SHA-256；不得把 README
中的旧摘要当作当前值，必须在运行前执行上述命令取得。

正式质量确认 JSON 必须是已有真实证据对应的文件，例如：

```json
{
  "confirmed": true,
  "sha256": "<上述命令输出的当前sha256>",
  "evidence_ref": "已有门限验证记录的位置"
}
```

该文件只确认合同身份和已有验证证据，不改变代码门限。缺文件、SHA 不匹配、`confirmed`
不是 `true` 或缺少 `evidence_ref` 时，不能声称 `formal_result_eligible=true`。
2026-10-11 正式实验的确认只适用于其 `c020345` 代码与产物身份；当前代码合同变化后
必须重新生成匹配确认，不能改写旧收据。

机制已实现或合同测试通过，不代表真实 GPU、真实视频、真实模型和完整质量合同已经闭合，也不代表可以写入论文主表。

## 10. 2026-10-09 历史 v11 实验审计记录

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

失败发生在许可证检查阶段，早于权重下载、GPU inventory 选择、重建和在线模型请求。最终状态：

| 阶段 | 状态 |
|---|---|
| reconstruction | 未执行 |
| Qwen3.8-Flash | 未请求 |
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
11. 一键实验入口只编排 induction 和 inner_validation；普通在线入口的
    outer_holdout/final_test 仍受显式 split 授权与标签隔离约束。
12. 不得用 mock、替代权重、伪造许可、伪造质量确认或缩小分母绕过失败。

## 12. 当前已知限制与待办

当前仍需明确的边界：

- **预算基线**：已完成的256次正式实验属于`c020345`与4096输出上限；当前8192代码需要新运行和新确认。
- **质量确认**：当前合同摘要必须由入口实时生成，不能套用旧代码或不同产物身份的确认。
- **MoGe-2**：默认 disabled，不属于已完成正式实验的方法变量。
- **演化**：已完成实验只比较冻结S0有无，不构成Skill演化收益。
- **尺度标定**：未标定题型保持低置信度，不得用默认常量升级。
- **final_test**：已完成实验未访问；任何后续最终评测必须保持盲评和标签隔离。

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
