# GPT-6.1-sol 四套数据集全量执行

用户已于 2026-11-05 要求停止全部实验。R01、R02 及全部 worker 已停止，共 42 个进程，核验无剩余运行进程。R01 保留 306 个完整题臂，R02 保留 10 个完整题臂；原结果哈希未变。停止时未返回的 11 个已发送请求保留原账本，结果状态未知，不自动恢复。停止回执见 `deployment/all-experiments-stopped-20261105.json`。以下启动与阶段评分记录为历史。

2026-11-05。暂不测试 SR，使用主 GPT Key 和凭据文件中 9 个注释 Key，共 10 个独立凭据。用户追加“全量尝试重跑，已有结果尝试统计分数”后，已启动独立 R02，PID 930；旧 R01（PID 27421）保留。以下原 R01 运行说明继续留作历史，当前运行见本节。

R02 运行目录为 `/home/wangy/wangjh/harness3d/GPTexp/runs/gpt61-high-full1998-20261105-r02`，源码包为独立的 `GPTexp/retry-runtime-20261105-r02/GPTexp`。部署前后旧轮全部源码哈希相同。四套题单、题序、两臂 spec 和 13,070 次请求预算均与 R01 相同，复用四份原预检及 10 个成功 API 验证回执，新探测请求为 0。

R02 将 HTTP 524 等暂时性服务故障改为 60 秒冷却，冷却后继续处理后续题目；失败题臂保留，不自动重发同一请求。模型身份、high、输入身份等契约错误仍暂停 API。HTTP/排队/worker 的等待分别保持 10 小时/7 天/10 天。已经观测到全部 10 个 API 各有 1 个正式请求获准发送。

旧轮已冻结 304 个完整题臂并独立宿主评分。148 道共同完成题中 Direct 为 70/148（47.30%），Harness 为 67/148（45.27%）；只看两臂都合法的 115 道题，Direct 为 62/115（53.91%），Harness 为 61/115（53.04%）。详见旧轮的 `legal_answer_comparison.md` 与服务器 `partial-scoring-20261105-r02`。部分评分使用实际已观测分母，全量准确率暂不计算。

只读状态入口为 `python3 -B -m GPTexp.rerun_ops status`。当前两轮均已按用户要求停止，不再次运行 `stage` 或 `launch`；停止状态以各轮 `stopped_by_user.json` 为准。

## 范围与预算

| 数据集 | 固定题数 | Direct/Harness 题臂数 | 原两臂请求上限 |
| --- | ---: | ---: | ---: |
| BLINK Multi-view | 93 | 186 | 465 |
| CV3D Depth / Distance | 840 | 1,680 | 5,880 |
| MMSI | 365 | 730 | 1,825 |
| MindCube tinybench | 700 | 1,400 | 4,900 |
| 合计 | 1,998 | 3,996 | 13,070 |

SR、VSI-Bench 和 Omni3D-Bench 不进入本轮。原题单、输入、两臂题对身份、题序、Skill、工具、几何缓存、程序轮数、token 上限、答案解析与评分保持原样；本轮不进行 Skill 演化。

四份全量预检从此前零请求准备结果逐字节复用。服务器已验证新清单恰好等于原 2,098 题清单删除 SR 后的题对序列，模型请求及参考答案读取均为 0；Mac 与服务器代码哈希一致。

## 启动与并发

任务先为每个 Key 发一次相同的两图工程验证，核对模型 `gpt-6.1-sol`、`high` 和红/蓝图片顺序。10 次工程请求独立计账，不计入 benchmark 结果，也不占用 13,070 次正式请求预算。

每个 Key 在途上限为 1，采用相同限速和超时设置。验证通过的 Key 才进入正式清单；全部通过时总在途上限为 10，题臂进程上限为 20。同题 Direct/Harness 以及 Harness 的各轮固定使用同一 Key。

验证完成后，任务冻结 `run.config.json`、manifest 和源码快照，然后自动开始正式评测。失败 Key 不进入正式池；正式执行中的服务失败会暂停对应 Key，其余 Key 继续处理尚未派发的题。已发出但结果未知的请求保留原始记录，不自动重发或切换 Key。

HTTP 等待为 36,000 秒，排队为 604,800 秒，题臂进程为 864,000 秒；原程序执行上限保持 30 秒。较长的本地等待不能阻止上游提前断开。

## 已知 API 行为

此前成功回包报告 `temperature=1`、没有 seed、输出上限为 null，并带有客户端未发送的系统 instructions。这些字段不能证明服务内部实际执行的参数。

在披露上述差异后，用户要求使用现有和注释 API 开始执行。本轮据此允许接收这些已报告差异并保留原生回包，仍按原方法发送 `temperature=0`、`seed=137` 和各阶段原 token 上限。模型身份及 `high` 仍严格核对。此次处理和依据写入冻结配置。

## 结果与核验

服务器任务目录：

`/home/wangy/wangjh/harness3d/GPTexp/runs/gpt61-high-full1998-20261105-r01`

Mac 回收目录：

`/Users/bytedance/wangjh/wjhICLR/GPTexp/runs/gpt61-high-full1998-20261105-r01`

`bootstrap.json` 记录启动阶段及各 Key 验证结果；`api_validation.json` 保存独立工程验证统计。`manifest.json`、`preflight_reuse.json`、`readiness_audit.json` 和 `public_source_snapshot.tar.gz` 绑定本轮范围、原输入身份与代码。

正式轨迹逐题逐轮保留原请求、原生响应、API 身份、公开 reasoning、程序、工具输入输出、授权回执、观察读取和最终答案。预测全部冻结后，独立宿主才读取参考答案评分。统一输出为 `results_and_full_trajectories.json`、`experiment_trace.jsonl`、`comparison.json`、`scored_pairs.json` 和 `report.md`。

凭据文件保持原状；凭据仅在内存中加载并通过 SSH stdin 注入服务器进程环境。API Key 和 SSH 登录凭据不进入配置、源码快照、轨迹或报告。

## 运行入口

本轮已经启动，不要重复执行启动命令。新任务的入口为：

```bash
python3 -B -m GPTexp.deploy bootstrap --run <new-run-id>
```

回收本轮公开产物：

```bash
python3 -B -m GPTexp.deploy collect --run gpt61-high-full1998-20261105-r01
```

实际当前状态以同目录 `live_status.json` 的查询时间为准；启动阶段不代表全量评测已完成。

## 2026-11-05 当前任务：每题型至少 30%，全部复用

用户最新要求为“直接全部复用，每个题型够题单的百分之 30 即可”。Terra 全量 R02 已停止。其 213 个已完成题臂全部纳入新题单，包括失败和无效回答，已完成题臂不重新调用模型。

每个题型按原题量 × 30% 向上取整，先保留所有已有结果涉及的题目，再以种子 137 的哈希顺序补齐。题单固定为 606 题、1,212 个 Direct/Harness 题臂：BLINK Multi-view 28 题、CV3D 252 题、MMSI 112 题、MindCube 214 题。18 个题型的逐型配额见新目录 `selection_report.md`。只补跑缺失的 999 个题臂；新的请求总上限为 3,330。

新任务为 `gpt56-terra-high-type30-20261105-r01`，服务器目录在原 `GPTexp/runs` 下，Mac 同名目录在本项目 `GPTexp/runs` 下。新代码在独立 `type30-runtime-20261105-r01/GPTexp` 中执行，原全量运行时保持不变。模型仍为 `gpt-5.6-terra/high`，10 个 Key，每 Key 最多 1 个在途请求；HTTP、排队、进程等待、原 Prompt、Harness、每题请求上限及 API 参数保持原配置。

`reuse_source` 保存旧 manifest、原请求账本和逐文件校验的 213 份完整 JSON 轨迹。旧 episode 保留原运行身份及原字节，新 manifest 通过快照身份引用它们。只有一侧完成的 5 题在补跑另一侧时固定使用原 API。旧目录中的未完成请求和部分轨迹继续保留。

准备阶段发起 0 次模型请求、0 次新 API 探测，并核验了所有复用轨迹、配置、输入和 API 回执。新任务已启动，不重复执行 `stage`、`prepare` 或 `launch`。服务器与 Mac 均每 5 秒更新 `connection_status.json` 和 `connection_status.md`；监控不发额外模型请求。完成后独立宿主核验合并预测，再读取参考答案评分，生成原统一 JSON/JSONL 轨迹、`scored_pairs.json` 和报告。

查询当前任务：

```bash
python3 -B -m GPTexp.subset_ops status
```

回收当前任务：

```bash
python3 -B -m GPTexp.deploy collect --run gpt56-terra-high-type30-20261105-r01
```
