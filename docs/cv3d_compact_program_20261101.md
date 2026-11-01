# CV3D 主流程：精简 Skill 与 6 次请求

本协议版本为 `cv3d-compact-program/1`。它继续使用主流程的 Python
程序求解和工具观察回传，采用 Hugging Face 原样题目与选项、
统一工具说明和按题型冻结的精简 Skill。

该 Skill 使用了用户于 2026-11-01 明确授权的 test 轨迹修订。
今后在相同 840 题上的结果属于 `test_development`，不能作为独立测试结果。
原简化实验和此前主流程的冻结源码、结果、请求账本均保留。

## 在线输入与历史

每次模型请求只有一条 user 消息，文本顺序为：

1. 官方 `prompt` 原文，包括 A/B 选项。
2. 原图身份、宽高、原始像素坐标和预测几何说明。
3. 对应 Depth 或 Distance 的冻结 Skill。
4. `Valid observations`：累计有效工具结果，首轮为 `[]`。
5. 剩余模型请求数以及 Python 变量清空的说明。
6. 现有程序工具接口和 `ReturnAnswer` / `YieldObservations` 执行规则。
7. 同一原始 RGB 图像的文件字节。

正式求解使用原始 RGB，预测深度和点图由现有工具读取。
没有整张深度图输入，也没有原生 `tool_calls`。
类别来自既有预计算，该入口不执行在线类别发现或 GPU 预处理。

下一轮重建题目、说明、Skill 和累计观察。不会加入前一轮程序、
assistant 消息、思考或 Python 局部变量。模型需要再次检查的数值，
必须在当前程序中通过 `YieldObservations` 交回。
程序直接 `ReturnAnswer` 时，数值比较仍由 Python 完成。

Depth 暴露 `point_depth`、`view_candidates`、`view_depth_summary`；
Distance 额外暴露 `pair_point_distance`。成功返回的外层 `status` 为 `ok`，
数值在 `data` 中，`result_id` 用于提交观察。
工具数值定义、授权校验、沙箱执行和错误终止路径沿用主流程。
`ReadObservation` 不属于这个入口。

## 请求预算

总计最多 6 次模型请求：最多 5 次程序求解，必要时 1 次最终 JSON 回答。
提前作答即结束，不强制测量或用满轮次。程序解析失败也消耗当前请求额度。
最后一次禁用工具，只使用原图和已有观察选择 A/B。

每个 HTTP 请求的上限为 8192 输出 token，thinking=true、temperature=0、
seed=137。单个程序的沙箱执行超时仍为 30 秒。
服务错误不自动重试；非法答案和执行错误保留在完整题单分母。

`max_solve_rounds: 5` 只计程序轮次；
`max_requests: 6` 包含最后一次最终化请求。
这与 VSI 的 `max_solver_rounds: 6` 包含最终化的参数命名不同，
实际预算均可表达为 5 次程序加 1 次最终化。

## 主入口

`scripts/run_small_image.py --benchmark cv3d` 默认采用
`configs/cv3d_program.yaml`。显式传入该配置也可以使用同一入口。
其他小图数据集继续采用原来的 `configs/small_image.yaml`。

冻结 Skill 清单：

`skill_library/candidates/cv3d-compact-testdev-1/frozen_skills.json`

清单绑定两份正文的 SHA256、提示词协议和 `test_development` 身份。
不替换原来的 trajectory-1 Skill。正文变化后必须创建新版本和清单，
现有正文或哈希不匹配会在发请求前被拒绝。

离线预检：

```bash
.venv/bin/python scripts/run_small_image.py \
  --benchmark cv3d \
  --config configs/cv3d_program.yaml \
  --manifest "<public_manifest.json>" \
  --dataset-root "<dataset_root>" \
  --official-annotations "<official_or_public_3d.jsonl>" \
  --prepared-root "<existing_prepared_root>" \
  --split test --limit 840
```

该命令验证原始输入、官方题目对应、冻结 Skill、已有缓存与请求构建，
不发送模型请求，不构建 GPU backend，不读取宿主评分答案。

实际发请求需要另行显式提供 `--execute`、新建 `--output-dir` 和
`--max-requests` 总预算。输出目录保留 manifest、请求账本和逐题
完整模型响应／程序轨迹。已有输出目录不能覆盖。

本次修改与 Skill 精简阶段不启动付费评测。
