# BLINK Multi-view_Reasoning：Qwen 推理与评测

下载源为国内镜像 `https://hf-mirror.com`，数据仓库为
`BLINK-Benchmark/BLINK`，仅下载 `Multi-view_Reasoning` 子目录。
下载程序解析仓库提交号，校验 Parquet 文件大小及 SHA-256，并提取其中的原始图片。

数据共 266 题、532 张图片，每题两张按时间先后排列的视频帧。
`val` 有 133 题，GT 已公开；`test` 有 133 题，答案字段均为 `hidden`。
因此可以本地计算 val 准确率；test 保存预测，其准确率为 `null`。

## BLINK 主实验

新版完整实验入口为 `../scripts/run_blink_experiment.py`，配置为
`../configs/experiments/blink.json`。它使用原四数据集实验的 **40/93 划分与现有缓存**，
执行原图直答、无 Skill 代码求解、初始 Skill 测试、训练/Gate、冻结与最终测试。
求解和 Gate 开启 thinking，输出上限 8192，图片按原始字节发送，
保存原始回答、服务推理内容、请求身份和完整 token 明细。
预览：`python ../scripts/run_blink_experiment.py --dry-run`。
服务器检查与执行说明见 [BLINK 主实验文档](../docs/blink_experiment.md)。

本目录的 `evaluate.py` 继续作为独立直答入口；`toolexperiment/` 保留四工具实测。

## 本次运行状态

2026-10-31 按用户要求停止全量评测。开启 thinking 的正式运行已保存 26 条 val
响应，均调用成功并完成答案解析，其中答对 25 题；已完成部分的正确率为
`25 / 26 = 96.15%`。这只是部分样本的结果，完整 val 准确率仍为 `null`；
test 尚未开始推理。

停止时共开始 28 次请求，其中 2 次没有保存到响应；原始请求记录已保留。
`summary.json` 的 `run_control` 记录停止状态和未取得响应的样本，
`partial_val_metrics` 记录上述部分正确率。`results.jsonl` 已生成 GT 对照，
未完成题目标记为 `not_attempted`，没有计入部分正确率的分母。

随后关闭 thinking 从头完成全部 266 题，共 266 次 API 调用，
实际模型均为 `qwen3.8-flash`，服务返回的 thinking token 总数为 0。
按原始回答中的明确最终答案计分，val 答对 `40 / 133`，准确率为 **30.08%**；
test 的 133 题均已保存预测，GT 隐藏，准确率为 `null`。
在先前开启 thinking 已完成的同一组 26 题上，关闭 thinking 答对 8 题，
开启 thinking 答对 25 题。

该轮 76 条回答需要从正文末尾 JSON、明确选项行或明确答案声明中提取答案，
全部提取成功。使用原始解析规则的 val 准确率为 22.56%，原始统计仍保留在
`results/qwen3.8-flash-no-thinking/summary.json`；
上述 30.08% 及完整 GT 对照位于其 `parsed/` 目录。
`audit.json` 记录完整请求、532 张原图哈希、模型身份、thinking 状态和评分核验。

## 文件

| 路径 | 内容 |
| --- | --- |
| `download_data.py` | 镜像下载、校验、图片提取、数据索引生成 |
| `evaluate.py` | 多图 API 推理、断点续跑、本地 GT 对照与准确率计算 |
| `rescore_answers.py` | 从正文末尾的明确 JSON 或选项行提取答案，独立保存后处理结果 |
| `data/Multi-view_Reasoning/*.parquet` | 两个原始数据文件 |
| `data/images/{val,test}/` | 532 张原图，字节未修改 |
| `data/inputs.jsonl` | 题目、选项、官方提示词、图片路径和哈希；不包含 GT |
| `data/references.jsonl` | 独立 GT 索引；test GT 为隐藏状态 |
| `data/sample_preview.json` | val/test 各一条样本，可查看字段格式 |
| `data/manifest.json` | 镜像、仓库提交、文件哈希与数据统计 |
| `results/qwen3.8-flash/manifest.json` | 固定的提示词、API 参数、图片请求哈希和代码哈希 |
| `results/qwen3.8-flash/predictions.jsonl` | 原始模型回答、解释、解析选项、模型身份、耗时与 token 用量 |
| `results/qwen3.8-flash/results.jsonl` | 全部逐题结果，合并题目、图片、预测与 GT |
| `results/qwen3.8-flash/summary.json` | val 准确率、test 状态、失败统计与 token 汇总 |
| `results/qwen3.8-flash/test_predictions.jsonl` | test 的原始 idx 和预测选项 |
| `results/qwen3.8-flash/attempts.jsonl` | 请求开始前写入的记录，用于防止中断后重复请求 |
| `results/qwen3.8-flash-no-thinking/` | 本次关闭 thinking 的全量重跑；包含同样的预测、GT 对照及统计文件 |
| `results/qwen3.8-flash-no-thinking/parsed/` | 独立答案提取后的预测、GT 对照、准确率与提取过程记录 |

`results/credential_parse_failure/` 保留首次密钥读取截断造成的两条鉴权失败记录，
与正式评测结果分别存放。该问题已修复，密钥读取支持句点。
`results/qwen3.8-flash-no-thinking-interrupted/` 保留打开 thinking 之前的中断运行；
正式结果目录使用 `enable_thinking=true`，两种配置分别保存。
开启 thinking 时使用的代码已保存在
`results/qwen3.8-flash/evaluate.snapshot.py`，其哈希与该轮 manifest 中的代码哈希一致。

## 运行

在本目录的终端中使用 Python 3.10 或以上版本。Mac 上已有 `uv` 时：

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python download_data.py
.venv/bin/python evaluate.py preflight
.venv/bin/python evaluate.py run
```

`preflight` 验证全部请求但不调用模型。再次下载会复用通过哈希校验的原始文件。
也可以用 `download_data.py --revision <提交号>` 固定数据版本。

API 调用复用项目已有的
`../src/skill3d/synthesis/bailian_client.py`，请求地址为百炼的 OpenAI 兼容接口：
`https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`。
密钥优先从 `DASHSCOPE_API_KEY` 环境变量读取；环境变量未设置时，
读取已有的私有文件 `../qwenkey.md`。代码不会复制或输出密钥。
可通过 `--key-file /私有路径/密钥文件` 指定其他密钥文件。

默认对 val 和 test 全量推理。仅评测 val 可使用：

```bash
`.venv/bin/python evaluate.py run --splits val --output results/qwen3.8-flash-val
```

关闭 thinking 并从头推理 val 和 test：

```bash
.venv/bin/python evaluate.py preflight --no-thinking
.venv/bin/python evaluate.py run --no-thinking
```

`--thinking` 为默认配置；`--no-thinking` 将 API 的 `enable_thinking` 设为 `false`。
关闭后的默认输出目录为 `results/qwen3.8-flash-no-thinking`，与开启 thinking 的结果独立。
重新评分这一轮可使用 `evaluate.py score --no-thinking`。

模型有时在正文后附上最终 JSON，或以 `Answer: (A) left` 等明确选项行结束回答。
全量推理完成后，可读取这些最终答案并重新评分：

```bash
.venv/bin/python rescore_answers.py
```

此步骤不调用 API，答案提取只读取模型回答和选项，不读取 GT。
原始 `predictions.jsonl` 和原始解析规则的 `summary.json` 均保留；
后处理文件写入独立的 `parsed/` 目录，记录原始状态、原始预测、提取方式、
原始回答及程序哈希。GT 仅在随后的本地评分阶段读取。

正常中断后，确认没有未确定的请求，可以续跑已冻结的配置：

```bash
.venv/bin/python evaluate.py run --resume
```

已保存的预测会跳过。存在“已开始但没有结果”的请求或服务失败时，程序停止并要求诊断，
不会自动再次向模型提交同一题。修改提示词、参数、数据或代码后，应使用新的 `--output`
目录保存新的实验。对已完成结果重新评分无需调用 API：

```bash
.venv/bin/python evaluate.py score
```

## 推理协议

模型为 `qwen3.8-flash`，每题一次调用；默认并发 2，最多可设为 4。
参数为 `temperature=0`、`seed=137`、`max_tokens=8192`、
默认 `enable_thinking=true`，可通过 `--no-thinking` 关闭。由提示词要求最终回答采用 JSON 格式。
API 返回的 `reasoning_content`、简短解释和最终答案都原样保留。

使用每条数据自带的 `prompt` 字段，其中说明相机绕静态场景移动，
并定义 clockwise (left)、counter-clockwise (right)。
在官方提示词后追加输出格式要求：`{"reasoning":"简短解释","answer":"A或B"}`。
两张原图分别通过 `image_url` 的 Base64 data URL 按原顺序发送，不拼图、不缩放、不重新编码。
GT 和数据中的 `explanation` 字段不会进入模型请求。

`model_reasoning` 是模型按输出要求生成的简短视觉解释；
`reasoning_content` 保存开启 thinking 后服务实际返回的推理内容。
本目录的 `ThinkingBailianClient` 在现有百炼客户端外增加响应读取，
同时保留原客户端的请求构造、鉴权、超时、模型身份和 token 用量处理。
`model_response` 保留完整可见回答，`prediction` 保留从明确答案字段解析出的选项。
无法解析、API 失败及模型身份不符均记录状态，不根据 GT 修正模型答案。

## 准确率

```text
val_accuracy = val 中回答正确的题数 / 133
```

分母包含全部有 GT 的题目。错误答案、无法解析的答案及失败请求计为错误。
评测尚未完成时，准确率为 `null`，不会把尚未执行的题目当成完成的实验。
test 的 GT 隐藏，`gt_answer`、`is_correct` 和准确率均为 `null`。

逐题结果包含 `sample_id`、`idx`、`split`、`question`、`choices`、`options`、
`benchmark_prompt`、两张图片的路径和 SHA-256、原始回答 `model_response`、
解释 `model_reasoning`、`reasoning_content`、`thinking_enabled`、
`prediction`、`prediction_text`、
`gt_raw`、`gt_answer`、`gt_answer_text`、`gt_explanation`、`is_correct`、
`requested_model`、`actual_model`、`response_identity`、`usage`、
`latency_seconds`、请求时间、`payload_sha256` 及状态字段。

数据、图片、结果、密钥和虚拟环境均不会通过本目录的 `.gitignore` 自动纳入 Git。
本次没有创建 Git 提交。
