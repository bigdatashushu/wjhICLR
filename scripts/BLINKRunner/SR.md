# BLINK Spatial_Relation（SR）

SR 用一张自然照片判断两个物体的空间关系是否成立，固定 `A=yes / B=no`。
例如 “Is the bowl in front of the apple?” 和 “Does the suitcase contain the cat?”。
它覆盖位置、前后、支撑、接触、包含、远近、物体朝向和部分整体关系；
照片本身没有统一的额外点框标记。题目中的主语与参照物顺序必须保留。

官方来源：[BLINK 数据集](https://huggingface.co/datasets/BLINK-Benchmark/BLINK/tree/main/Spatial_Relation)。
2026-11-04 核验 revision `a3666eb249237ba3d5eca8db21176cc47967e040`：

| split | 题数 | 原图/题 | 公开答案 |
| --- | ---: | ---: | --- |
| val | 143 | 1 | 143 |
| test | 143 | 1 | 0；官方隐藏 |

保留官方 `val/test`，不改名为 Skill-3D 的内部 `train/test`。
数据位于 `BLINK/spatial_relation/data/`，公共 `inputs.jsonl` 与宿主
`references.jsonl` 分开；Parquet 和原图字节保留并核验 SHA-256。
现有 MVR campaign 使用自己的配置和冻结代码。

## 已接入的实现

| 文件 | 用途 |
| --- | --- |
| `sr.py` | 下载、离线检查、原生预计算、direct/skill_tools 求解、独立宿主评分 |
| `sr_data.py` | SR 公共输入适配、完整分母、图像哈希、官方 prompt 核验 |
| `sr_skill.txt` | SR 专用版本化 Skill；只依据题型组织证据 |
| `sr_config.json` | 独立 Qwen3.8-Flash 配置 |
| `sr_protocol.py` / `sr_adapter.py` | SR 关系解析、prompt、局部适配及完整执行轨迹 |
| `sr_trial.py` / `sr_start.py` / `sr_bundle.py` | 固定 30/70 划分、20/80 题配对批次、8 并发、公开输入部署及独立核验 |
| `test_sr.py` | 原图、标签隔离、真实 sandbox 工具执行、隐藏 test 评分验证 |

两组都发送一张原始照片和完整官方问句、选项。direct 单题 1 请求，不接收几何缓存
或 Skill；skill_tools 使用 SR Skill、原图和已核验的相对深度缓存，单题最多
2 轮程序求解加 1 轮最终回答。共享参数为 seed=137、temperature=0、thinking 开启、
最大输出 8192 token，顺序执行。两组跑 val 143 题时，请求上限为 572。
每次运行使用独立预算、输入/源码/Skill 哈希与请求回执；服务失败或未完成请求停止，
不自动重发。已完成预测可核验后续跑；评分在求解全部结束后单独启动。

SR 工具范围为 `view_candidates`、`view_depth_summary`、`point_depth`。
相对深度可以辅助前后判断；框重叠不能代替接触/包含证据，
候选类别确认也不意味着物体表面几何已验证。朝向等题型由原图及合适证据判断。

## SRv2：完整轨迹诊断后的接口与提示词

2026-11-05 的旧版 `small-image-blink-sr/1` 已完成 100 道内部测试：
direct 89/100，SR 80/100；SR 有 96 个合法答案，66 次工具尝试中 64 次成功。
新版 `small-image-blink-sr/2` 根据这批完整轨迹修改，尚无新版在线准确率。

照片和现有观察足够时，SR 直接提交严格 JSON：
`{"answer":"A","reasoning":"简短依据"}`，其中 reasoning 可省略。
仅接受 A/B；重复键、额外字段和从散文提取的答案不接受。模型原始响应保留，
宿主只把解析出的选项枚举组装成安全答案程序，并记录 `host_constructed`
与 `submission_format`。需要工具时仍生成 `def solve(ctx)`，调用后
`YieldObservations`；工具回合内的 `ReturnAnswer` 改为交回真实观察，
其预写答案不会提交。模型在下一请求结合观察和原图决定答案。

提示词给出原图尺寸、合法坐标和真实返回字段：
`point_depth` 的值在 `r['data']['depth']`，候选框在
`r['data']['candidates'][...]['box_xyxy']`。候选不完整且物体表面未认证。
工具调用必须使用列出的名字；动态 `getattr` 仍被守卫拒绝。点坐标不夹紧，
越界、未授权、服务失败和其他不可信结果继续停止。

可信普通 Python 运行错误可在原有最多 2 轮程序、3 次请求内请求修复，
原始错误、程序、工具结果和授权仍留在 `execution_trace.execution`；
`bounded_repair_requested` 说明原因，`effective_execution` 说明交回宿主循环的
观察状态。修复提示仅含错误类型、行号和字段说明，不把异常正文作为模型上下文。
这些逻辑通过 SR worker 内局部 hooks 接入，退出时恢复共享 runner、解析器和 kernel。

关系提示按问句分族：前后需要深度与透视，over/under 需要检查覆盖和垂直对齐，
普通 on 包括薄覆盖物介入的支撑；beside 可以与前后遮挡并存。朝向使用可见前轴，
反射、多个实例和嵌套对象先固定身份。Skill 不包含题号、参考答案或测试题示例。

这次更新已经查看全部 100 题内部测试轨迹，之后在同批题上的新版结果属于
测试反馈驱动的开发复测。43 道训练题仍未用于演化；泛化评估需要另留未参与
修改的数据。新版运行使用新的冻结身份与目录，旧 20/80 批次封存及成绩保持原样，
不能以新版源码续接旧版冻结运行。

## 使用

所有命令从仓库根目录运行。默认命令只检查，不读取答案或发送模型请求：

```bash
.venv/bin/python -B scripts/BLINKRunner/sr.py --download
.venv/bin/python -B scripts/BLINKRunner/sr.py --check
.venv/bin/python -B scripts/BLINKRunner/sr.py --check --split val \
  --prepared-root /path/to/sr-prepared
```

尚未生成 SR 的模型深度/候选缓存时，默认检查回报 `skill_tools_ready=false`；
这不影响公共输入、关系和两组 prompt 的离线核验。真实工具组运行要求已核验深度缓存。

在具备已核验服务端点和本地权重的机器上，可显式调用已有
GroundingDINO + SAM2 + Intern 类别确认 + MoGe2 预计算：

```bash
.venv/bin/python -B scripts/BLINKRunner/sr.py --precompute --execute --split val \
  --backend-config /path/to/verified-small-image.yaml \
  --prepared-root /path/to/sr-prepared
```

backend 配置使用 `scripts/run_small_image.py` 已有的 `precompute` schema。
只从公共问句抽取两个对象名，不使用参考答案。保留原像素网格，
检测服务仅允许一次尝试，预计算失败立即停止，已核验缓存不会被覆盖。

后续启动已授权模型实验时，API 凭据须通过配置所列环境变量瞬态注入：

```bash
.venv/bin/python -B scripts/BLINKRunner/sr.py --run --execute --split val \
  --prepared-root /path/to/sr-prepared --run-root /path/to/new-sr-paired-val
.venv/bin/python -B scripts/BLINKRunner/sr.py --score \
  --run-root /path/to/new-sr-paired-val
```

仅 direct 可通过 `--arm direct` 独立运行，无需几何缓存。
test 143 题只产生预测和合法回答统计；评分明确输出 `accuracy=null`，
不会把隐藏标签样本当成错误答案或计算虚假准确率。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests \
  .venv/bin/python -B -m pytest scripts/BLINKRunner/test_sr.py -q -p no:cacheprovider
```

## 20 道内部测试诊断

`sr_trial.py` 按 seed=137 将官方 val 143 道随机划为训练 43 道、内部测试 100 道，
再从内部测试均匀抽取 20 道，两路使用同一批题。官方 `source_split=val` 保持真实，
内部角色单独记录为 `internal_test_diagnostic`。抽样不看答案，训练集暂保留。

模型调用并发 8、全局最多 80 请求。预计算在 3090 两张独立物理卡上分片，
使用原始像素网格、DINO、SAM2、Intern 类别确认及 MoGe2。两台机器使用一致的绝对
运行目录，以保持输入和缓存的哈希契约。在线部署包不包含参考答案或 Parquet。
SR 在原生检测后按置信度为每个对象类别保留最多 2 个候选，总共最多 4 个，
完整原始检测列表、保留和丢弃索引写入缓存 provenance，避免全局截断漏掉参照物。

完整 JSON 记录每轮实际请求（含原图字节）、供应商响应及 reasoning_content、
生成程序、执行错误、工具输入与结果、授权和观察回执。求解全部结束后，
Mac 主机独立核验每条轨迹并打开本地参考答案评分。

冻结时对全部 20 题截获真实 solver 的首轮请求，核对其与预检 payload 的哈希一致，
并在调用供应商之前中止。`zero_request_delivery.json` 保存这一核验，模型请求为 0。
SR 注册表在实际 `names_for_scope` 路径上限制工具范围。

如首轮冻结核验发现实现问题，使用新的运行目录保留原始试验；显式续跑需要
已完成且无服务失败的父运行和已封存的直跑身份。它逐字节复制 20 条 direct episode
及全部请求回执，承接原预算，只提交新的工具组 operation，禁止重发已付费请求：

```bash
.venv/bin/python -B scripts/BLINKRunner/sr_trial.py freeze \
  --root /path/to/new-corrected-run \
  --continue-from /path/to/completed-parent \
  --direct-seal /path/to/direct_verified.json
```

`sr_continuation.py` 核对输入、模型配置、Skill 和共享实现未变，保留旧预算前缀。
`collect_sr.py` 在完成后核对冻结源码和几何缓存，对轨迹归档逐成员完整复读并校验，
随后生成独立持久备份。主机报告分别列出当前比较请求和累计请求；
旧工具组输出保留在原始归档，修正发生在读取答案和评分之前。

## 完成剩余 80 题与汇总 100 题

`cohort.json` 将新批次固定为 `remaining80`：只取原 `internal_test` 中不属于
`diagnostic20` 的题目，不重新抽样。训练集 43 题仍预留。`parent_manifest.json`
保留已核验 20 题批次的公开清单，冻结检查要求数据划分、模型、Skill 和科学实现
逐字节一致；只有 `sr_trial.py`、`sr_start.py` 两个批次调度文件允许变化。
新批次独立预算最多 320 请求，8 并发，160 条 episode。

`native_start.py` 检查 3090 物理卡 3/4、SAM2/MoGe2 权重哈希、检测和审核服务，
再启动两个预计算分片。`package_prepared.py` 核对全部 80 个缓存与分片回执，
并对几何归档完整复读。预计算和模型运行均禁止自动重试；已有付费 episode
不会再次发送。`transient_start.py` 只在启动时从既有私有配置加载 API 凭据，
经 SSH stdin 交付服务端，不把凭据保存到运行文件。

两批各自完成轨迹、预算、原图和工具授权核验后，再运行：

```bash
.venv/bin/python -B scripts/BLINKRunner/aggregate_sr.py \
  --first /path/to/verified-sr20 \
  --remaining /path/to/verified-remaining80 \
  --output /path/to/new-internal-test100-report
```

`aggregate_sr.py` 核对两批模型/Skill/科学源码、数据划分、几何归档和独立验收身份，
要求恰好覆盖 100 道内部测试题且无重复或训练题。它输出包含全部 200 条轨迹的
`results_and_full_trajectories.json`、配对结果、工具调用与错误诊断、中文报告。
有效比较请求和包含首批历史修正请求的实际付费总量分开记录。
