# GPT-6.1-sol 统一配对评测

2026-11-05 最新执行要求：暂不测试 SR，使用主 GPT Key 及凭据文件中 9 个注释 Key，运行其余四套共 1,998 题、3,996 个 Direct/Harness 题臂，正式请求上限 13,070。使用 high，保留原发送参数，并记录此前已披露的上游回包差异。后台任务已在 Titan 启动，运行名为 `gpt61-high-full1998-20261105-r01`。具体预算、流程和当前状态入口见 `execution_plan.md`。

新任务先以每 Key 一次两图请求验证模型、high 和图片顺序，再自动冻结通过验证的 API 池并启动正式评测；每 Key 在途上限为 1，全部通过时总在途为 10。四份全量输入预检直接复用，服务器已核对新题对序列与原清单去掉 SR 后完全一致。此前包含 SR 的 2,098 题零请求准备结果保留在 `runs/gpt61-high-full2098-20261105-r01`，不与本轮结果混用。以下原始调研记录反映启动前的范围与诊断；当前运行以 `execution_plan.md` 和新任务目录为准。

## 当前 API 探测结果

现有 `xcode-1` 使用 Responses。文本单请求耗时 81.50 秒，正确返回 `OK`。四个同时发送的两图请求中，一个耗时 81.64 秒成功，正确区分第一张红色、第二张蓝色；其余三个分别约 132.27、150.90、179.77 秒后传输断开，结果未知。四并发窗口共 179.78 秒，成功吞吐约 0.334 次/分钟。这个小样本不能确定稳定容量或服务 QPS。

两次成功回包报告模型 `gpt-6.1-sol`、reasoning effort `high`；同时报告 `temperature=1`、未报告 seed、`max_output_tokens=null`，并包含客户端未发送的系统 instructions。发送的实际参数是 temperature 0、seed 137、max_output_tokens 8192。参数被接口接受与参数被严格执行是两件事，目前这些回包不能支持“其他一律不改”的启动条件。

首轮单题、并发图片探测的原生回包和请求分别位于 `diagnostics/20261105-text-01`、`diagnostics/20261105-images-c4-01`。失败请求没有重试。正式配置当前 `validated=false`；客户端会拒绝明确报告的采样参数、输出上限和系统提示差异，保留原生回包并暂停该 API。

已采用：HTTP 36,000 秒（10 小时）、队列 604,800 秒（7 天）、题臂进程 864,000 秒（10 天）；程序执行仍为原 30 秒。较长的本地超时不能阻止中转站或网络提前断开。现有 API 的初始在途上限设为 1，全局设为 1，两臂进程数为 2；追加 API 可逐项配置独立 Key、并发和限速。

## 运行入口

所有本轮代码、输入部署包和结果位于本目录。服务器镜像目录为 `/home/wangy/wangjh/harness3d/GPTexp`；既有公共图片和 prepared 缓存只读复用。

MMSI 使用 GPT 的后续开发入口为 [`MMSIGPT/`](MMSIGPT/README.md)。MMSI 的输入准备、Direct/Harness、提示和工具权限在该包中维护；`scripts/MMSIRunner` 与 `frozen/mmsi` 保留原始归档。通用 worker 已路由到新包，部署和源码身份也包含新包。该本地迁移尚未部署或改变现有服务器冻结运行。

```bash
# 在 wjhICLR 目录运行；所有操作均使用 Python 标准模块入口。
python3 -B -m GPTexp.selfcheck
python3 -B -m GPTexp.deploy deploy
python3 -B -m GPTexp.deploy prepare

# API 参数行为符合既定协议，或得到明确的差异处理决定后：
python3 -B -m GPTexp.deploy execute

# 回收服务器上的公开结果、预检回执与完整轨迹到本目录的 runs/<run-id>。
python3 -B -m GPTexp.deploy collect
```

服务器上也可以直接使用：

```bash
export PYTHONPATH=/home/wangy/wangjh/harness3d
python -B -m GPTexp.run prepare --run /absolute/GPTexp/runs/run-id
python -B -m GPTexp.run execute --run /absolute/GPTexp/runs/run-id
```

这里的 `python` 应使用部署回执指定的虚拟环境解释器；Mac 的 `GPTexp.deploy` 会自动使用正确解释器和工作目录。

正式执行读取已经冻结的 manifest。`config.example.json` 提供两个 API 的示例，两项使用相同的每 API 在途上限、限速与超时；当前配置只有一个现有 API。`config.json` 中增加 API 项即可在准备新 manifest 时纳入池，字段包括 `id`、`base_url`、`protocol=responses`、`model`、`api_key_env`、`credentials_file`、`max_inflight`、`requests_per_second`、`timeout_seconds`、`enabled`、`validated`。新增 API 不共享环境变量切换；同题两臂和各轮固定在同一 API。配置或源文件变化需要新的准备结果，不能悄悄改变已有运行。

`GPTexp.probe --config ... --output ... --phase images --concurrency 1 --api-id xcode-2` 可以单独验证新增 API；每个探测目录只使用一次，工程探测不计作 benchmark 结果。只有核实服务行为后才能标记 `validated=true`。

跨进程总预算和在途锁使用文件锁及原子 JSON 账本；不依赖 Titan 上存在 GLIBC 兼容问题的 SQLite。已完成题臂可核验后跳过；已派发但未完成的题对列为 unresolved，不自动重答或切换 API。服务失败会暂停故障 API，其余可用 API 可以继续接收未派发题。

运行结束后自动由独立宿主进程验证预测、读取原参考答案、评分，生成 `comparison.json`、`report.md`、`results_and_full_trajectories.json` 和 `experiment_trace.jsonl`。JSON 导出逐题流式写入，避免把全部图片轨迹同时加载进内存。`GPTexp.export --run ...` 同时收录完整结果和未完成题臂已经保存的部分请求轨迹；部分轨迹不计入完成数。`GPTexp.score --run ...` 仅接受全部 4,196 个题臂冻结后的评分。

请求统计记录获准发送的尝试、收到的响应和结果未知的传输断开；尚未获准发送的排队意向单独计数。输入核验完成与 API 能否正式启动分别在 `preparation.json` 的 `inputs_ready`、`api_launch_ready` 标明。

评分后另生成 `scored_pairs.json`，逐题保留参考答案、两臂预测、状态、正确性和 API 身份；`comparison.json` 同时报告 Harness 独有正确、Direct 独有正确及准确率差。参考答案只在所有预测冻结后由宿主读取。结果回收包含请求预算账本和暂停回执，文件锁不参与回收。

## 题量与范围

排除 VSI-Bench 和 Omni3D-Bench，已找到以下专用适配：

| 任务 | 既有测试题量 | 划分及完成状态 | GPT 两臂请求上限 |
| --- | ---: | --- | ---: |
| BLINK Multi-view Reasoning | 93 | 官方 val 的内部 40 train / 93 test；93 题双组已完成 | 465 |
| CV3D Depth / Distance | 840 | 作者 test：Depth 420、Distance 420；840 题双组已完成 | 5,880 |
| MMSI 位置关系六题型 | 365 | 既有作者 test 题单；365 题双组已完成，后续另有 Skill 服务失败重试记录 | 1,825 |
| MindCube tinybench | 700 | tinybench 共 1,050 题；排除历史试跑及关联场景/图片后，固定选 700 题；补齐剩余 34 题后双组已完成 | 4,900 |
| BLINK Spatial Relation（SR） | 固定内部 test 100；已完成诊断 20 | 官方 val 143：43 train / 100 internal test；20 题已完成；其余 80 题的两个预计算分片已各完成 40 题，本地尚无该批模型完成回执 | 100 题为 400；20 题为 80 |

前四项共 **1,998 题、3,996 个题臂、13,070 次请求上限**。加入 SR 的固定 100 题后，共 **2,098 题、4,196 个题臂、13,470 次请求上限**。请求上限包含 Direct 每题一次及 Harness 每题原有最大请求数，不是预计实际用量。

MindCube 700 题包含 Among 399、Around 204、Rotation 97，涉及 354 张原图。剩余 tinybench 题目未指定训练用途，不能写成 350 train / 700 test。SR 官方隐藏答案 test 143 题不属于现有内部 100 题评分题单。

ReVSI、SPAR-Bench、VSTI-Bench 已有下载或输入核验材料，但本轮没有找到和上述任务同等完整的专用 Harness 配对适配。ReVSI 在通用注册表中出现，现有说明仍要求转换为 prepared v11 输入；不因此将这些数据计入本次既有测试范围。

## 复用哪些方法

以真实已完成实验的冻结源码、配置、Skill 和题单为来源。当前开发工作区存在未提交改动，直接使用最新目录不能保证“只改 API”。

| 任务 | 冻结方法来源 | Harness 最大请求数 / 题 |
| --- | --- | ---: |
| BLINK Multi-view | paired93 的 trajectory-1 Skill 和原 24 题 solver | 4；最多 2 个程序轮次，沿用其余原请求路径 |
| CV3D | box-review840 的 `cv3d-box-review-program/3` 和冻结 Depth / Distance Skill | 6；5 个程序槽 + 1 个最终回答槽 |
| MMSI | 365 题正式测试清单、六个冻结 Skill 槽位、归档 runtime | 4；最多 2 个程序轮次，沿用原请求路径 |
| MindCube | 700 题运行的 `source_snapshot.tar` 与 r06 方法 | 6；5 个程序槽 + 1 个最终回答槽 |
| SR | paired20 r03 修正后的 SR Skill、配置、工具契约及既有 100 题划分 | 3；2 个程序槽 + 1 个最终回答槽 |

题文、选项、图片原始字节及顺序、深度输入、缓存、候选过滤、工具数值、授权、程序解析、沙箱、反馈、终止规则、答案解析和评分均保持各任务原样。CV3D 保留既有 `test_development` 身份。

两臂全部重新调用 GPT。既有 Qwen 预测用于历史比较，不能替代 GPT Direct。此次不增加 Skill 训练、修订或演化。

## API 适配

`xcodeapi/profiles.json` 的 GPT 配置为：

- 模型：`gpt-6.1-sol`。
- 协议：Responses。
- 地址：`https://xcode.best/v1/responses`。
- 同步、非流式、单次调用；客户端不自动重试，不自动保存请求或回答。

现有 runner 使用 Bailian / Chat Completions 或现有认证网关。只修改地址无法让它们使用 Responses。应在 GPTexp 中实现兼容客户端，保留 solver 使用的 `chat()`、`build_payload()`、`audit_identity()` 及 `last_usage`、`last_response_identity` 等接口，并在传输边界转换：

1. 原消息文本和图片顺序转换为 Responses `input`；原图 data URL 内容保持一致。
2. `max_tokens` 转换为 `max_output_tokens`，保留每阶段原 token 上限；一般求解和 Direct 为 8,192。
3. 将返回文本、usage、model、状态和公开 reasoning 字段映射回现有客户端契约；原生响应同时留存。
4. 各任务的程序工具仍由原沙箱执行，不额外改为 GPT 原生 function calling。
5. 原来未发送强制 JSON 格式的路径继续依靠原提示词和解析器；原来发送格式参数的路径按原语义转换。

本地说明使用 `reasoning: {"effort": "high"}`，但没有证明它和 Qwen `enable_thinking=true` 等价，也没有证明本中转站对 GPT 接受 `temperature=0`、`seed=137`。不能在遭拒后自动删除参数重发。需要先确定思考档位和不兼容参数的处理规则，再通过小规模实际调用核实。两臂采用相同 GPT 参数，真实发送参数完整记录。

原请求超时为 600 秒；按用户要求，传输改为 36,000 秒，队列和外层进程采用上述更长等待。xcodeapi 档案默认的 180 秒不覆盖本轮配置，程序执行时限和科学轮数不变。

模型返回身份须检查。新增中转站即使使用相同模型名，也先核实其实际返回身份；按 API 记录分组结果。

## 多 API 与并发

同时支持同中转站不同 Key，以及多个中转站。公开配置每个 API 的 `id`、`base_url`、`protocol`、`model`、`max_inflight`、限速和超时；密钥通过独立环境引用或仓库外权限受限文件加载。

`xcodeapi.Profile` 当前仅允许固定的四个 Key 环境名，单实例请求计数也不是跨进程总预算。GPTexp 需增加独立凭据解析和共享预算，不能通过并发修改全局 `XCODE_GPT_API_KEY` 来切换密钥。

- 同题 Direct / Harness 分配到同一个 API，整题各轮保持 API 身份固定；配对任务按固定题单派发。
- API 池分别控制每个 API 的在途请求和限速，再施加总请求上限及总并发上限。
- HTTP 等待与沙箱执行分别控制资源；程序在子进程主线程执行，保留原程序超时语义。
- 接口失败保留请求意向和失败轨迹；不自动切换 API 重答，也不重复派发已经发送但结果未知的请求。
- 暂停或降速后的 API 不接新题；其他 API 可处理未派发题。是否继续其他题按用户确定的服务失败策略执行。
- 多个进程共享同一个 Key 不一定提高上游容量；并发上限需按中转站能力配置，初始配置可以保守设置后核实。

## 完整 JSON 轨迹

保留逐题逐轮完整 JSON，最终另导出可流式读取的 JSONL 与统一统计：

- 原始公开问题、图片字节及有序哈希、模型请求、实际参数、API 身份和时间。
- 原生响应、返回模型、usage、finish/status、公开 reasoning / summary 字段及字段缺失状态。
- 程序全文、AST / 程序解析结果、执行错误、累计上下文和下一轮反馈。
- 工具名称、参数、输出、授权回执、观察读取、证据版本和程序终止。
- 最终答案、解析方式、合法性、配对身份；预测冻结后由宿主独立评分。

API 未返回的内部推理不可恢复，记录缺失状态。认证头、Key、服务器登录凭据不进入轨迹；答案不进入模型或沙箱输入。失败题和未完成题保留在固定分母，并分别报告状态。

原始 JSON 与摘要统计应分开写入。写入使用原子替换，失败和中断留下已有证据；恢复只处理经过身份核验的未派发题。

## 建议实现目录

所有新增代码及最终结果位于用户指定的 `wjhICLR/GPTexp`：

```text
GPTexp/
  README.md
  dataset_inventory.json
  run.py
  config.example.json
  api_pool.py
  transport.py
  traces.py
  adapters/
  frozen/                 # 各任务实际冻结来源，保留来源哈希
  runs/<run-id>/
    manifest.json
    requests/
    episodes/
    experiment_trace.jsonl
    comparison.json
    report.md
```

上述实现代码已创建，实际适配文件为 `adapters.py`。既有数据和 prepared 缓存按原路径读取；SR 剩余 80 题的已完成预计算产物从本地复制到本次独立目录。SR 缓存的输入哈希包含原始绝对路径，本轮保留缓存原路径绑定，并核对官方问题、选项、来源以及原图字节；缓存与几何数值没有重算或修改。

MindCube 原运行器拒绝将输出放在其仓库内，并限制并发值；MMSI runtime 带有现场路径、历史预算和网关绑定，不能直接作为独立 CLI 启动。GPTexp 的运行副本需适配路径、输出准入、模型身份和调度，保留科学方法源码内容，并列出允许变化字段及其差异。

## 服务器核对

依据用户提供的 `server.md`，于本轮通过已有 SSH 认证连接 Titan `wangy@10.10.0.34`。只读检查返回主机 `gpu34`，未发起模型或 GPU 请求，也未访问答案：

- `/home/wangy/wangjh/harness3d/four-datasets-20261031` 存在。
- `smallimage/manifest.json`、`smallimage/prepared`、MindCube `prepared_catalog.json` 存在。
- CV3D840 的 `source_frozen` 存在。
- MindCube700 的实际目录使用 `source/`、`results/`、`campaign-accounting/`，不叫 `source_frozen/`；本地完整 `source_snapshot.tar` 可用于冻结来源。
- SR20 r03 有 `source/`、`dataset/`、`prepared/`、`episodes/`；剩余80题缓存已有本地预计算回执及产物包，执行前仍须核对部署位置和逐题身份。
- MMSI 后续重试目录存在；原正式测试源码通过本地冻结清单和 MMSIRunner 归档绑定，实施时按其真实来源定位。

建议 Titan 负责在线 HTTP 调度和 CPU 沙箱，复用其现有缓存，使用本次独立 GPTexp 运行目录；Mac 的 GPTexp 保存代码及完整收集后的结果。共享主仓库是 `/nas/wangjh/harness3d/skill3d_codebase`，运行时使用本次冻结副本。执行前完整哈希预检仍须逐项运行，目录存在不代替内容验证。

## 实施顺序与待确认事项

1. 确定 SR 本轮题量、GPT 思考设置及参数不兼容规则；确定 API 容量和服务失败策略。
2. 从既有冻结来源整理各任务适配，固定题单和方法，生成 GPTexp 配置及零请求预检。
3. 实现 Responses 兼容层、API 池和完整轨迹；用离线检查验证消息无损转换、预算和恢复身份。
4. 核实每个 API 的真实模型、图片输入和参数支持；再启动固定题单的两臂评测。
5. 预测冻结后独立评分，收集完整 JSON/JSONL、配对结果、成本和失败统计到 Mac GPTexp。

用户已确定 SR100、high 和超长等待。目前还需要处理实际探测发现的上游差异：

- 是否接受中转站报告的 temperature、输出上限和额外系统 instructions 差异并明确记录，或先取得能严格透传原参数的 API。
- 新增 API 在配置中逐项添加并验证后可用于新 manifest。当前仅测了已有 API；没有新增凭据可供实际测量。
