# CV3D 简化实验

## 六轮主 Skill 与直答

`program_paired.py --config <run_config.json> --execute` 调用实际的
`SmallImageSolver`，与官方 prompt 直答同时运行。两组均使用原 RGB 文件字节、
Qwen3.8-Flash thinking、temperature 0、seed 137 和 8192 output tokens。
Skill 组最多五轮 Python 程序加一次 JSON 最终回答；两组各保留四路并发。
工具实现、授权和 YieldObservations 重建保持主流程语义，逐轮保存真实参数、
公开工具结果、模型身份和实际请求哈希。

入口先零请求校验四道 train 与 840 道 test，再验证八个训练 episode 的合法回答
和成功工具交互，通过后进入 test。独立请求账本为直答保留 844 次请求。
服务、宿主或预算错误停止新请求，不自动重试。冻结 Skill 使用过 test 轨迹，
因此结果标记为 `test_development`。`status.py <root>` 与
`live_results.py <root>` 只读该运行目录，不调用模型。

## 原生工具简化组

模型输入只包含 Hugging Face `test_3d.jsonl` 中原样的 `prompt`（已经含 A/B 选项）、
工具使用说明、标明身份的原始 RGB 图和同尺寸深度图。没有 Skill、
Input JSON、求解规约或 Python 程序生成要求，也没有额外的 system 消息。

默认沿用 `qwen3.8-flash`、`thinking=true`、`temperature=0`、`seed=137`、
`max_tokens=8192`，评测题单沿用 Skill-3D 的 360 train / 840 test 划分。
原图直接以原始文件字节发送，不缩放、裁剪或重新编码。

## 一轮回答的具体含义

每题只有一个作答会话和一次最终回答。模型使用 OpenAI-compatible native
`tool_calls`；宿主执行已注册工具，将返回值作为 `role=tool` 消息传回同一会话。
模型可连续调用工具，也可以在一次响应中调用任意多个工具。首次不包含工具调用的
响应立即结束该题，不重新询问、不修订、不强制追加 JSON 答案阶段。

**这通常需要多次 HTTP 模型请求**：工具执行后必须发送返回值，模型才知道测量结果。
它不是“单个 HTTP 请求完成交互”。每题没有工具调用次数或工具交互轮数上限；
实验总模型请求预算由 `--max-requests` 控制，达到预算即停止。`max_tokens`
限制每次模型响应；服务仍可能有自己的上下文和单次工具调用限制。

保留本会话中的 assistant 工具消息、工具结果及服务返回的 `reasoning_content`。
此路径不生成或执行模型程序。

## 深度和工具

Image 1 是原始 RGB 图，唯一几何视角为 `view_index=0`。Image 2 是 MoGe2
预测深度的灰度 PNG，像素网格与 Image 1 完全一致：深色近、浅色远，无效值为
洋红色。显示按有效值 2%–98% 分位线性截断；原始深度和点图保持不变，
`point_depth` / `pair_point_distance` 始终读取原始数值。

Depth 提供 `view_candidates`、`view_depth_summary` 和 `point_depth`；
Distance 另提供 `pair_point_distance`。复用 `src/skill3d/small_image/tools.py`
的实现和运行时授权。成功返回的 `status` 为 `ok`，数值在 `data` 中；
失败返回 `status=error`、`data=null`，模型可据此自行处理，没有自动工具重试。
不提供 `ReadObservation` 或在线类别提取。

## 文件

| 文件 | 用途 |
| --- | --- |
| `data.py` | 官方 prompt 匹配、原图和缓存校验、深度图及消息构造 |
| `tools.py` | 原生工具 JSON Schema、参数校验、宿主工具分派 |
| `client.py` | Qwen HTTP 调用；每次请求只尝试一次 |
| `inference.py` | 单题会话、工具结果回传、首次最终答案解析、轨迹 |
| `evaluate.py` | CPU 预检、独立题单与预算、并发评测、断点检查和评分 |
| `common.py` | 配置、身份哈希和原子 JSON 写入 |
| `test_simplified.py` | 纯本地合成输入与假服务测试 |

## 运行

从仓库根目录执行，使用项目的冻结 Python 环境。官方标注必须是 Hugging Face
下载的 `test_3d.jsonl`，不能把 Skill-3D 的训练或测试 jsonl 当作官方标注。
默认命令不调用模型，也不做 GPU 预计算。

只核对本地官方 prompt 与作者题单：

```bash
.venv/bin/python tests/CV3D/evaluate.py \
  --dataset-root /Users/bytedance/wangjh/skill3d-paper-benchmarks-20261030/dataset \
  --official-annotations /Users/bytedance/wangjh/skill3d-paper-benchmarks-20261030/CV-3D/test_3d.jsonl \
  --split test --check-prompts
```

在已有预计算缓存的机器上先预检，设置独立输出目录和总请求预算：

```bash
python tests/CV3D/evaluate.py \
  --dataset-root /path/to/dataset \
  --official-annotations /path/to/CV-3D/test_3d.jsonl \
  --prepared-root /path/to/smallimage/prepared \
  --output-dir /path/to/cv3d-simple-experiment \
  --split test --max-requests 5000 --concurrency 4
```

`5000` 仅是命令示例中的预算，不是程序默认值或已批准的新实验预算。
要执行模型请求，在同一条预检命令后追加 `--resume --execute`。
API Key 从 `DASHSCOPE_API_KEY` 读取，也可通过 `--api-key-env` 指定变量名；
不把 Key 写入代码、命令参数、轨迹或 Git。自定义服务用 `--base-url` 显式指定。
可用 `--split train --limit 10` 或重复 `--sample-id` 选择题单；省略选择参数时
使用指定 split 的全部题目，输入顺序保持作者题单顺序。

缓存中的原图路径必须与当前输入身份一致；程序不改写已有缓存以适应搬迁路径。
预检会逐题核对实际原图、深度和点图，全部通过后才允许请求。

## 评分、轨迹和恢复

GT 只在宿主评分器中读取，不传入推理器、模型或工具。
接受明确的 A/B、精确选项文本、`{"answer":"A"}` 和明确的答案行。
不从模型思考或含糊解释中猜测选项，不为非法回答额外发请求。
非法答案、服务失败和宿主错误均计入完整题单分母。运行途中同时报告完整题单
准确率、已记录题目准确率以及 `not_run` 数量；只有 `complete=true` 才是完成状态。

输出包含 `manifest.json`、`preflight.json`、`request_preview.json`、
`depth_preview.png`、`scores.json`、`budget/state.json`，
以及 `episodes/<sample_id>/` 中的逐请求身份、原始模型响应、工具轨迹、
深度图和 `result.json`。每个请求（含失败）都记账。

服务错误或身份偏离立即停止新增请求；已在途请求收尾，不自动重试。
`--resume` 校验题单、源码、模型配置、图片、缓存和预算身份，跳过已有最终结果
（含非法答案）。遇到服务/宿主错误、预算耗尽或未解决的请求记录时拒绝自动恢复，
需要先诊断。重新汇总已有结果可使用同题单参数和 `--score-only`。
该实现不提交或推送 Git，也不修改旧冻结运行目录。

本地测试：

```bash
.venv/bin/python -m pytest tests/CV3D -q
```

## 双组共享调度

`paired.py` 从已有 label-free manifest 读取作者划分，先验证固定的 4 道训练题
（每类取源顺序前 2 道），再同时评测 840 道 test。`tools` 组使用上述双图和
工具；`direct` 组仅使用官方 prompt 与同一原图，单次请求，省略所有工具参数。
两组共用相同答案解析器和评分分母，不使用既有测试预测替代新 baseline。

默认每组 2 个任务，总并发 4；一组派完后将空闲额度交给另一组。一个独立总账本
覆盖训练验证和测试，预留 844 个直接回答请求。可通过 `shared_slots_root` 复用
其他正在运行实验的 4 个文件锁槽，限制跨实验总并发，不向对方账本追加或改写记录。

`transport.py` 和 `gateway.py` 支持原生工具 payload 的单次认证转发。网关只在
已有授权服务节点读取 API 环境变量；Titan 仅使用既有传输认证，不持有上游 API Key。
`paired.py --config <run_config.json> --execute` 会在零模型请求的输入预检通过后，
自动完成训练协议验证，再进入测试；任一服务错误停止新增请求，保留完整历史。
配置记录数据路径、源码与网关身份、6000 次总预算及共享锁目录，不记录凭据。
