# VSIbench8Brunner

使用已有 `vsibench-m4-success-1000` 子集，保留 300 题训练、700 题测试、
164 个场景及原始 32 帧输入、场景总览、主模型、Skill 演化、工具、评分和直接基线。
代码、配置和验证用例均在本目录，运行时无需导入外面的 `wjhICLR/src`。

8B 描述步骤位于 M5 语义审核前：

```text
原检测与 SAM2 掩码
  → 原候选三图 + 原物体类别
  → Qwen3-VL-8B 描述实际覆盖的物体/可见部分
  → 原二分类审核器接收三图、类别和描述
  → 原阈值与对象绑定、几何验证、解题及评分
```

三图均来自同一帧：橙色检测框整帧、无掩码裁剪、青绿色掩码裁剪。
例如 `refrigerator` 可描述为 `the lower half of a white refrigerator`。
Qwen 描述是审核输入，物体类别、原检测查询和几何归属仍保留原值。
原 VSI 配置的审核器实际为 `Intern-Decision-4B` 的 Jev 风格二分类接口，
不是外部 Makers Jev；仍使用 P1、`yes_threshold=0.77`、`category-only`。
主解题模型仍为 `qwen3.8-flash`，按最新要求开启 `enable_thinking=true`，
8B 负责新增的描述步骤。主程序生成、最终作答及评测视觉补答均接入该开关。

## 选择题的独立视觉回退

2026-11-06 的比例 100 题轨迹中，六道选择题从对变错、三道从错变对，
选择题净损失 3 分；六道降分题都被核验为 `visual_estimate`，
且都没有触发独立 `eval_visual_fallback`。旧补答只处理未提交合法答案的
`run_error`。已经看过不完整排序、局部对象信息的模型，即使最后声明视觉估计，
仍会保留这些上下文。例如 QA2130 将“最近的可测对象”扩大成了“最近对象”；
方向和路线题还出现相机坐标替代题目观察者坐标、视频帧序替代导航动作。

新配置 `isolated_choice_visual_fallback: true` 只在验证或盲评角色生效：

- 选择题提交经核验的 `visual_estimate`，或求解以无答案 `run_error` 结束时，
  统一追加一次独立请求；路由只看依据与状态，不看真值或原答案是否正确。
- 请求只有原问题、原选项和同一冻结帧集的 32 张原始 RGB，
  不含 Skill、程序、恢复提示、工具结果、裁剪图或场景总览。
  提示要求使用题目指定的观察位置与朝向，并区分视频帧序与导航动作。
- 原结果、载荷、归因、M11 检查和各轮工具账本保留在审计中。
  新请求前撤销原答案并清空用户变量；服务或解析失败时不恢复旧答案、不自动重试。
- 有效 `tool_derived`、`mixed` 答案保留。非选择题继续原策略。
  `relative_distance_rank` 调用成功但内部不可判定时，恢复提示保留局部数据，
  并明确它不支持完整排序或最终赢家。

新策略与旧 `eval_visual_fallback` 共用最多一次增量请求：
当前求解预算仍为最多 6 次加 1 次补答。库默认关闭新策略，旧直接基线提示不变。
启动检查也明确记录既有试跑的 900 秒请求超时与原输入包 180 秒超时的差异，
只接受这一固定覆盖，不改变原始记录或允许任意超时改动。
`pilot40.py` 显式设置验证角色并保留真实 `source_split`，启动身份记录策略及提示版本，
拒绝用新策略续跑旧目录；正式新实验应使用独立目录与源码快照。
这些改动已经提供离线回归用例；准确率变化需要新的冻结配对实验验证。
本次包内用例与补答、场景总览、提交合同、推导回放回归合计 267 项通过，
其中新增选择题回退用例 39 项。公开 1000 题启动检查通过，真实模型请求为 0。

## 40 题试跑与缓存复用核验

当前试跑题单在 `runs/vsi8b-pilot40-20261105-r02/sampling_manifest.json`：
已有 700 题测试划分内的内部源分区，seed=137，8 题型各 5 题。

在 Titan 对这 40 题核验后，原重建和冻结帧目录均存在，原绑定的全部发布 payload
哈希核验通过。但 1,942 个原审核候选的三图请求文件均已随工作目录清理；
其中 1,252 个原未通过候选也没有保留为发布对象掩码。
当前状态为 `blocked_missing_candidate_assets`，真实模型请求为 0。
逐题、逐候选证据与最小补生成范围见该目录的 `report.md`、
`reuse_preflight.json` 和 `regeneration_plan.json`。

需要复用旧检测框补生成 SAM2 掩码和原三图，才能完整重审原候选范围。
该补生成等待用户确认；下文 `prepare` 的完整生成路径不能替代本轮要求的缓存复用试跑。

公开抽样和只读核验入口为 `scripts/pilot_cache_preflight.py`：

```bash
python scripts/pilot_cache_preflight.py sample \
  --data-root "/Users/bytedance/wangjh/vsibench-m4-success-1000" \
  --output-dir "/path/to/new-pilot-run"

# 在能读取原发布产物的 Titan 运行；只读，零模型请求。
python scripts/pilot_cache_preflight.py audit \
  --published-inputs "/data/wangy/wangjh/harness3d/experiment-inputs/vsibench-m4-success-1000-seed137/published_inputs.json" \
  --sampling "/path/to/sampling_manifest.json"
```

## 入口

统一使用 `run.py`。`scripts/`、`src/` 和原实验配置是收拢的运行依赖。

```bash
# 本机检查公开题单、划分、哈希，以及声明的方法配置差异；零模型请求。
"/Users/bytedance/wangjh/wjhICLR/.venv/bin/python" \
  "/Users/bytedance/wangjh/wjhICLR/VSIbench8Brunner/run.py" check

# 打印原 v11 实验阶段和新增缓存位置；零模型请求。
"/Users/bytedance/wangjh/wjhICLR/.venv/bin/python" \
  "/Users/bytedance/wangjh/wjhICLR/VSIbench8Brunner/run.py" dry-run
```

Mac 默认读取 `/Users/bytedance/wangjh/vsibench-m4-success-1000`。
服务器默认读取
`/data/wangy/wangjh/harness3d/experiment-inputs/vsibench-m4-success-1000-seed137`；
也可使用 `--data-root` 显式指定同一输入包。RGB、几何和原发布产物在 Titan。
`check` 的 `remote_scene_inputs_available=false` 表示当前机器没有实际场景目录；
公开题单检查完成不代表已运行模型。

## 描述服务

`runner.yaml` 沿用原 VSI 运行配置，增加 `mask_review.target_description`。
新增服务使用 OpenAI 兼容 `/v1/chat/completions`，默认
`http://127.0.0.1:8100/v1`、`Qwen3-VL-8B-Instruct-FP8`。

可在运行进程中设置：

```bash
export VSI_QWEN8B_BASE_URL="http://127.0.0.1:8100/v1"
export VSI_QWEN8B_MODEL="Qwen3-VL-8B-Instruct-FP8"
```

若描述服务要求认证，从 `QWEN3VL8B_API_KEY` 加载；原主模型继续从
`DASHSCOPE_API_KEY` 加载。密钥不写入配置、请求 JSON 或源码。
审核器保持 loopback 地址，可通过已授权的 SSH 隧道接入；覆盖地址与身份分别使用
`VSI_REVIEW_ENDPOINT`、`VSI_REVIEW_IDENTITY`，阈值保持不变。

若要启动已有权重，`serve_qwen8b.py` 接收明确分配的物理 GPU、现有模型目录和
已安装兼容 vLLM 的 Python。它检查显存占用，只绑定 loopback，
不会自行下载权重、安装环境或停止占用进程。

## 准备新 M5 缓存

原数据缓存直接复用历史类别审核；使用它会跳过新增描述。
`prepare` 复用原冻结帧和 M4 几何，按原 M5 检测/SAM2/实例合并/场景冲突审核流程
重新准备绑定，仅在语义审核前增加描述。
原发布目录保持原身份，新绑定写入本目录的 `prepared_cache/`。

在能读取 Titan 场景输入且具备原 M5 依赖与服务的运行环境执行：

```bash
python "/path/to/VSIbench8Brunner/run.py" prepare \
  --data-root "/data/wangy/wangjh/harness3d/experiment-inputs/vsibench-m4-success-1000-seed137" \
  --cache-root "/path/to/VSIbench8Brunner/prepared_cache" \
  --max-requests 20000
```

`20000` 是可显式修改的准备阶段上限，不代表预计请求量或足够覆盖全量。
该账本共同限制原物体解析、Qwen 描述和二分类请求。服务失败后停止，
不会自动重试或退回旧审核结果；再次执行同一准备命令是显式续作，
已封存结果必须通过身份与完整性检查。

解题和评分仍只有所选 1000 题。原绑定的场景冲突审核范围包含这 164 个场景的
3142 道源题，准备阶段保留这个范围；额外源题不进入训练或测试评分。
原 M5 seed 为 0，求解 seed 仍为 137。

缓存输出：

- `refresh_manifest.json`：完整缓存身份、原输入引用、新绑定引用和迁移回执。
- `request_budget.json`：请求计数与身份哈希，不保存凭据。
- `stores/<scene_id>/`：新内容寻址 M5 绑定及原数组身份。
- `trajectories/<qa_id>.json`：物体解析、全部候选审核、描述请求与原生响应。
- `trajectories/raw/`：准备阶段原始 JSON 文件，成功与失败轨迹都保留。

正式消费端只回放已完成的新缓存，preflight 不调用 Qwen、SAM2 或审核器。
描述、图像、原生响应、seed、模型配置和代码身份改变时拒绝混用。

## 原实验流程

准备完成后：

```bash
python "/path/to/VSIbench8Brunner/run.py" run \
  --data-root "/data/wangy/wangjh/harness3d/experiment-inputs/vsibench-m4-success-1000-seed137" \
  --cache-root "/path/to/VSIbench8Brunner/prepared_cache" \
  --output-dir "/path/to/VSIbench8Brunner/runs/vsi8b-r01" \
  --max-requests 20000
```

该预算属于原解题/演化/直接基线阶段，与准备预算分开。
阶段仍为 seed、训练前测试、训练演化、冻结、训练后测试、配对评分、
32 帧直接基线、基线评分和封存。程序、工具 I/O、授权回执和原轨迹格式继续由
原运行器保存，产物在 `runs/`。

新目录使用按文件内容冻结的独立源码快照，保留原 commit 作为来源，
实际运行身份以完整源码哈希为准；不会自动 commit。

```bash
python "/path/to/VSIbench8Brunner/run.py" resume \
  --output-dir "/path/to/VSIbench8Brunner/runs/vsi8b-r01"
```

## 构建验证

已通过 113 项离线验证，包括真实 loopback HTTP 的两阶段调用、同图传递、
Qwen 失败阻止后续请求、原阈值、完整 JSON、篡改拒绝、迁移后的缓存准备与回放，
以及原审核器与存储回归。

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests
```

本次完成代码构建与离线验证。未部署新的 8B 服务，未请求真实模型，
未启动 1000 题实验。当前服务端口和 GPU 占用不作为以后启动的资源保证。
