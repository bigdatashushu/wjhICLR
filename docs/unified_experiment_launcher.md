# 一行启动实验

入口是仓库根目录的 `experiment.sh`。选择数据集、已经准备好的数据目录和整次实验的请求上限后，一条命令依次执行：冻结初始 24 槽 Skill、初始验证、学习、冻结最后成功发布版本、最终验证、初始/最终成对评分、32 帧基线、基线评分、文件封存。

```bash
./experiment.sh --dataset vsi-bench --data-root /path/to/vsi-prepared --max-requests 10000
./experiment.sh --dataset revsi --data-root /path/to/revsi-prepared --max-requests 10000
```

`10000` 是命令示例中的总请求上限，需要按实际实验预算选择；覆盖 solver、reviser、诊断、A/B gate 与基线，耗尽即停止。默认配置不设有效预算，必须通过参数或配置给出。脚本使用已有 Python 环境，不安装包或启动服务。正式执行需要 Python 3.11+ 及原项目依赖，可以显式指定：

```bash
./experiment.sh --python /path/to/venv/bin/python --dataset vsi-bench --data-root /path/to/vsi-prepared --max-requests 10000
```

入口也会依次查找 `H3D_PYTHON`、仓库 `.venv`、仓库 `venv`、`python3.11`、`python3`。保留虚拟环境解释器路径，不把符号链接解析成系统 Python。

## 先预览和检查

```bash
./experiment.sh --dataset revsi --data-root /path/to/revsi-prepared --dry-run
./experiment.sh --dataset vsi-bench --data-root /path/to/vsi-prepared --max-requests 10000 --check
./experiment.sh --list-datasets
```

`--dry-run` 只解析配置并输出阶段命令，不写目录、不加载科研依赖、不调用模型、不连接服务；数据尚未就绪时也可预览。`--check` 对本地输入计算文件哈希，不创建实验目录、不访问模型，也不代替原 runner 的逐题 preflight。实际阶段在请求前完成原生零请求 preflight。

## 数据目录约定

默认 `--data-root` 包含：

| 文件 | 内容 |
| --- | --- |
| `panel.json` | 原生 v11 fit-set panel |
| `metadata.jsonl` | 数据集元数据，真实标签只由学习框架或离线评分使用 |
| `published_inputs.json` | 原生 `v11-published-fitset-inputs/1` 或 `/2`，包含已发布资产及完整身份链 |
| `qa_split.json` | 原生 `harness3d-qa-overlapping-scene-split/1`；其引用的 learning、validation 和 public scope 文件也需保留 |
| `online.yaml` | 已验证的原项目在线配置，使用环境变量名引用凭据 |

`published_inputs.json` 指向的所有资产必须在执行机器上可读且符合原协议。这些输入由原有预计算/转换流程准备。ReVSI 当前没有新增原始视频适配器，需要先转换为相同协议；切换参数不会自动转换原始数据。

模板是 `configs/experiments/prepared.json`，注册表是 `configs/experiments/datasets.json`。复制模板到仓库外后，用 `--config /path/to/profile.json` 指定真实文件名、panel 范围、模型配置和服务地址。配置不得放入密钥值。默认 `ready_579` 是原生 panel 的选择名称，题数由实际 panel 和 QA 分配决定，不硬编码成 749/524/225。

## 并发、输出与续跑

学习阶段默认使用 `analysis_synthesis_v1`：每槽位收齐 12 条新完整轨迹，先按实际请求容量分批独立分析，再以全部分析集合生成一次锚点修订。这是 `run_analysis_successor_v11.py` 的主方向在原生入口中的集成，复用其分析和编辑组件，不依赖历史实验操作层。分析、格式重放、最终编辑和静态修复都使用本 campaign 的共享请求账本，并计入原有每槽位修订额度；预留不足时零请求拒绝。候选仍由原生双 seed Gate 决定晋升。详见[主修订合同](reviser_analysis_v11.md)。显式 `full_cases_v1` 保留原逐批改写作对照；切换模式须新建冻结运行。

默认 16 个执行 worker、4 个固定 Skill owner、4 路 reviser、最多 4 个同时模型请求；初始/最终验证使用单 reviser 参数且不开演化。可设置 `--workers 8 --revisers 2`；episode 与请求容量等完整参数放在 profile 中冻结。RPC 服务、GPU 服务和集群配置沿用你显式给定的配置，入口本身不占用或重配置 GPU。

每次新实验创建 `runs/<dataset>/<UTC时间>/`，或使用明确的 `--output-dir`。它与数据输入目录隔离；已有目录只能核验续跑。代码冻结到该目录 `code/source`，源码 Git 必须干净且已提交。学习 owner 使用原 runner 的独立库，不改共享初始 Skill 库。

```bash
./experiment.sh --output-dir /path/to/completed-or-interrupted-run --resume
./experiment.sh --output-dir /path/to/completed-or-interrupted-run --resume --check
```

续跑核验源码、解释器、配置、数据集、输入文件、已完成阶段回执和全部阶段文件的哈希。已完成阶段不会再执行。未完成原生在线阶段通过原 runner 的 `--resume` 继续；未决请求或基线 intent 没有结果时拒绝再次请求，需要先核实原响应。预算/并发不能在原目录内更改，变更时新建实验。

`campaign_state.json` 保存当前阶段；各阶段 `receipt.json` 是完成证据，只有退出码为零且回执满足完整状态才进入下一步。失败记录异常类型；不保存环境、密钥、原始异常正文或额外原始模型提示。终端输出保留阶段进度。按 Ctrl-C 时控制器停止子进程组并等待收尾。

入口仅接受自己创建的 `campaign_manifest.json`，不能直接接管历史 live 实验目录或重启当前实验。

## 基线与审计范围

默认基线是 **prepared-32**：使用同一 published 输入的 32 帧 RGB，经原有 JPEG 消息编码，每个唯一 QA 一次请求；未额外缩放。评分前核对验证臂与基线的帧集合、帧索引和像素哈希。默认基线范围是选择 panel 的全部唯一 QA，最终系统比较范围仍为冻结 validation QAs。

这条基线未重新运行官方原始视频采样器，因此结果明确标记 `prepared-32`。`seal` 核验本地源码、输入、阶段回执及文件完整性；不宣称已经通过额外的集群/Titan科学审计。初始/最终评分验证实验身份和完整分母，报告观察差值，不自动晋升方法。

如需官方 sampler、对象存储搬运或独立最终审计，可用 `commands` profile 把已有脚本作为 argv 数组编排。每阶段必须声明完成回执和续跑策略；不会执行 shell 文本：

```json
{
  "schema": "harness3d-experiment-profile/1",
  "backend": "commands",
  "inputs": {"manifest": "{data_root}/manifest.json"},
  "stages": [{
    "name": "official_baseline",
    "argv": ["{python}", "{source_root}/scripts/your_existing_baseline.py",
             "--output-dir", "{run_root}/stages/official_baseline"],
    "resume": "receipt",
    "receipt": {
      "path": "{run_root}/stages/official_baseline/summary.json",
      "equals": {"status": "complete"}
    },
    "artifacts": ["{run_root}/stages/official_baseline"]
  }]
}
```

允许占位符：`data_root`、`run_root`、`source_root`、`python`、`dataset`。`source_root` 执行时为冻结源码。`receipt`/`never` 策略在中断后没有完整回执时均拒绝重启；`native-v11` 只用于兼容原生 `--resume` 的脚本。不往 argv 放凭据，使用进程环境加载。

## 已完成的验证

验证使用合成输入和本地子进程，不调用实际模型、GPU或远端服务：九阶段执行、完成阶段跳过、目录冲突、重复启动锁、代码/输入/结果篡改、阶段无回执和非零退出、崩溃后采用完整回执、未决请求拒绝、无 shell 求值、原生参数/QA角色与冻结 24 槽 Skill。真实数据的完整在线执行需要由上述启动命令发起。
