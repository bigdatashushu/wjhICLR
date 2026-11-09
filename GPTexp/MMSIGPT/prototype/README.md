# MMSI 官方提示 Direct 最小原型

这是独立的单臂入口：

```bash
python -B -m GPTexp.MMSIGPT.prototype --help
```

链路为：原图与题目选项 → 一次 `gpt-5.6-terra` 请求 → 字母解析 → 预测冻结 → 宿主独立评分 → 简洁轨迹。
不依赖 Harness、生成程序、prepared 几何、候选、Skill 或共享求解循环。已有 Direct/Harness 代码与冻结实验保持原样。

## 官方依据与协议

- [MMSI-Bench 论文 Appendix E.2 / Table 7](https://arxiv.org/html/2505.23764v3)：Direct 提示原文为
  `Answer with the option's letter from the given choices directly. Enclose the option's letter within ``.`
- [VLMEvalKit MMSIBench 实现](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/dataset/mmsibench.py)：
  按原始顺序先放图片，再放题干、`Options: A: …, B: …, ` 与 Direct 指令。
  保留选项次序、重复选项文本和末尾逗号；不做 circular 选项重排。
- [作者数据集](https://huggingface.co/datasets/sihany/MMSI-Bench)，核查版本
  `6c988a330d5261bb4196da8c17df12912d75894a`。项目的 365 道位置关系题是 test 子集，不能称完整 MMSI 成绩。
- 官方提取选项字母后做 exact match，可使用 VLMEvalKit 的模型裁判回退。
  本原型只接受单字母、反引号/括号包裹字母或简短 `Answer: A` 表达；
  自由解释、多个字母、JSON 和越界选项记为无合法答案，不调用裁判。这个差异写入评分记录。

提示版本为 `mmsi-official-direct-table7-v1`，解析版本为 `single-option-letter-v1`。
旧 Direct 使用 JSON 输出要求；本原型使用新清单、源码哈希与运行目录，两者须分别报告。
现有冻结公共题单没有 Hint 字段，本原型不补充 Hint。

## 文件

| 文件 | 职责 |
| --- | --- |
| inputs.py | 抽取公开题目、双臂去重、原图字节和旧输入哈希核验、无标签抽样 |
| protocol.py | 官方提示、原图顺序、Responses 请求体、字母解析 |
| client.py | 单次 HTTP 请求、内存凭据、返回元数据与固定错误码 |
| direct.py | 零请求 prepare、独立运行身份、串行执行和请求预算 |
| evaluation.py | 预测冻结、固定分母评分、review.json 导出 |
| selfcheck.py | 合成图片、假响应、禁网离线检查 |

运行依赖 Python 3.10+、Pillow 和项目内已有的 `GPTexp/vendor/xcodeapi`；无额外服务。
源码身份只覆盖本原型及实际使用的小型只读公共工具，不加载旧冻结源码。
原图绝对路径参与旧 `SmallImageProblem` 输入哈希；路径或字节变化会拒绝执行，不自动替换路径。

## 离线检查

从项目根目录执行：

```bash
python -B -m GPTexp.MMSIGPT.prototype selfcheck
```

Mac 上若没有 `python` 命令，使用 `python3 -B -m GPTexp.MMSIGPT.prototype selfcheck`。

只用临时目录内的合成图、合成题和 fake client，清空继承环境并禁用网络。
不读取真实标准答案或凭据。检查包括原图/选项绑定、官方提示、参数、错误停止、禁止重复派发、
冻结前不评分、冻结后篡改拒绝、六类固定分母和请求/返回参数差异。

## 服务器运行示例（待后续授权执行）

这是服务器实验；本地离线检查不代表服务器线上验证。
下面示例不会由 selfcheck 执行。需在服务器对应的项目根目录使用**新运行目录**。
示例中的配置路径和 API ID 要换成实际已核验的外部 endpoint 配置。
配置沿用 GPTexp `apis` 格式；选定 API 必须 `validated=true`、`enabled=true`、`protocol=responses`。
认证仅在 execute 时通过该 API 的 `api_key_env` / 私有 `credentials_file` 解析，不写入清单。

```bash
python -B -m GPTexp.MMSIGPT.prototype prepare \
  --run GPTexp/runs/mmsi-official-direct-first6-r01 \
  --source-manifest GPTexp/frozen/mmsi/manifest.json \
  --config /absolute/path/to/endpoint-config.json \
  --api-id YOUR_VALIDATED_API_ID \
  --per-type 1

python -B -m GPTexp.MMSIGPT.prototype execute \
  --run GPTexp/runs/mmsi-official-direct-first6-r01 \
  --config /absolute/path/to/endpoint-config.json

python -B -m GPTexp.MMSIGPT.prototype freeze \
  --run GPTexp/runs/mmsi-official-direct-first6-r01
```

默认按六类各一题、seed=137 无标签确定性抽样，共 6 题，最多 6 次请求，串行执行。
也支持 `--ids MMSI_… MMSI_…`、`--all`、`--per-type 5`。
`--exclude-file` 接收已审阅题 ID 的 JSON 数组；诊断开发样本不再作为独立盲测。
prepare 可读取原始公共 `jobs`、GPTexp `pairs` 或 `mmsi-direct-public/1` 的 `items`，
只提取严格白名单 problem 字段；不读取 prepared 目录。

项目参数为：`gpt-5.6-terra`、`xcode-responses`、`reasoning_effort=high`、
`seed=137`、`temperature=0`、默认 `max_output_tokens=8192`。
超时取选定 API 配置（既有配置为 36000 秒）；这些是项目设定，不是官方 MMSI 必需参数。
token 上限可在 prepare 显式指定 `--max-output-tokens`，随后冻结。
外部配置文件、请求内容、原图、源码、Python/Pillow 版本均绑定到清单。

保留请求参数与服务报告参数的区别。`request_context` 模式允许报告采样元数据不一致，
逐题记录 `reported` 与 `reported_matches_requested`；不宣称 seed=137 / temperature=0 使服务确定性。
`strict` 模式按外部配置的 upstream policy 检查报告字段，发生偏离即停止。

## 冻结、独立评分和轨迹

freeze 后才能访问标准答案；在线入口不提供标准答案参数。宿主 reference 文件沿用项目格式：

```json
[{"benchmark": "mmsi", "sample_id": "MMSI_…", "reference_answer": "A"}]
```

也接受 `{"references": [...]}` 和 `dataset="mmsi"`。评分必须传入**已核验的原始 reference 文件哈希**：

```bash
python -B -m GPTexp.MMSIGPT.prototype score \
  --run GPTexp/runs/mmsi-official-direct-first6-r01 \
  --references /absolute/path/to/host_evaluation/answers.json \
  --references-sha256 VERIFIED_ORIGINAL_SHA256

python -B -m GPTexp.MMSIGPT.prototype export \
  --run GPTexp/runs/mmsi-official-direct-first6-r01
```

产物：

- `manifest.json` / `preparation.json`：公开输入、源码和请求身份，零请求准备回执。
- `traces/*.dispatch.json`：发送前持久化的请求哈希、原图绑定、公开参数。
- `traces/*.result.json`：正文、服务实际返回的 reasoning summary、usage、解析、返回元数据或固定服务错误码。
- `execution.json`：完成/停止/中断及已派发数。
- `frozen.json`：全部所选题预测与记录哈希；含无效回答、服务错误、未派发和未知结果。
- `score.json`：整体与分题型 exact match、合法率、合法回答条件准确率。
- `review.json`：题目、原选项、原图路径、输出与回执；评分后包含宿主答案和正误。

主准确率分母始终为全部所选题。无效回答、服务错误、未派发和未知结果均不计正确，
另行报告状态计数，不能以合法回答条件准确率替代主准确率。
服务错误停止；不重试、不恢复。发送前崩溃留下的派发记录按 `outcome_unknown` 保留。
再次 execute 会拒绝已启动目录；之后需要新实验身份和运行目录。
freeze / score / execute 共用非阻塞操作锁，防止评分与执行并行。
不保存请求 base64、原生服务响应体、认证头、URL、凭据值或异常正文。
