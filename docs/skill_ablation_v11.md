# v11 冻结 S0 的 B01/B11 配对运行

入口：`scripts/run_skill_ablation_v11.py`；实现：
`src/skill3d/evaluation/skill_ablation_v11.py`。默认题型为
`object_rel_distance`（S03），`--question-types` 接受逗号分隔的规范题型。

两臂均通过真实 `run_episode` 生成、执行程序，默认共同使用
`program_synth_v11_2`、`solver-v11.2-m11-acceptance` 和 `tool-docs-v11.1`，
质量门始终开启。`ReturnAnswer` 先暂存提交；M11 通过后才进入评分，拒绝时在同一
求解预算内撤销受影响结果并把结构化失败反馈给同一模型。声明了 `derivation` 时，
M11 只使用登记操作确定性重放并核对答案；计算声明错误不会撤销本身有效的工具结果。
B01 的交付集合为空；
B11 默认校验并加载 `S0-v11-contract-repair` 的 snapshot、manifest 和完整正文，
经正常题型查找与交付路径选择 Skill。入口不读取或修改 active pointer，
也不使用 v10 的 `evaluation_binding`。

## GPU 环境运行

先准备与输入清单一致的 P1 重建产物、模型服务及 M5 所需的检测/SAM2。
入口只复用既有重建，不在缺失或帧集不匹配时偷偷重建。示例在仓库根目录运行：

```bash
PYTHONPATH=src python scripts/run_skill_ablation_v11.py \
  --config configs/config.yaml \
  --source vsi_bench \
  --split inner_validation \
  --question-types object_rel_distance \
  --datasets scannet,scannetpp \
  --sampling-per-task 8 \
  --video-root data/raw_videos \
  --recon-dir data/reconstructions \
  --vllm-endpoint http://127.0.0.1:8100 \
  --vllm-model Qwen/Qwen3-VL-8B-Instruct-FP8 \
  --seed 137 \
  --output-dir data/evaluations/v11-s03-seed137
```

### 空机器一键入口

`scripts/bootstrap_and_run_v11.sh` 可在 Linux GPU 机器的空工作目录中完成固定提交拉取、
隔离 venv、权重复用或固定 revision 下载、逐文件 SHA-256 清单、重建、父版本
induction 运行和 B01/B11。配置中的 `existing_path` 优先复用当前已验证的 Qwen、
VGGT、SAM2 缓存，不改名、不移动、不重新下载；仅当该目录不存在时才按仓库 revision
下载到工作目录。`configs/gpu_experiment_v11.yaml` 固定 Qwen、VGGT、SAM2 及可选
MoGe-2 的仓库 revision；代码 revision 必须由调用者用 40 位 `CODE_REF` 提供：

```bash
CODE_REF=<包含本入口的40位Git提交> \
bash scripts/bootstrap_and_run_v11.sh \
  --work-dir /data/skill3d-v11 \
  --config /data/gpu_experiment_v11.yaml
```

正式运行前必须在外部配置副本中填写真实 `video_root`、与当前
`--print-quality-contract` 输出匹配的 `quality_confirmation`，并在实际完成许可核对后
把 `accept_model_licenses` 和 `accept_vsi_bench_license` 设为 `true`。脚本不会下载
VSI-Bench 视频、伪造许可确认或自动生成质量确认。默认要求两张显存不低于配置下限的
GPU，Qwen 与 VGGT/SAM2 分卡；任何阶段失败都会留下 `failure.json`，不会继续生成正式
结论。该入口没有 `final_test` 参数，只运行 `induction` 与 `inner_validation`。

先检查将执行的命令而不下载或启动服务：

```bash
PYTHONPATH=src python scripts/run_gpu_experiment_v11.py \
  --config configs/gpu_experiment_v11.yaml \
  --work-dir /tmp/skill3d-v11-check \
  --run-id dry-run --dry-run
```

配置 `experiment.run_evolution=true` 后，一键流程会在 B01/B11 之后调用
`scripts/run_evolution_campaign_v11.py`。此时还必须通过环境变量提供
`DEEPSEEK_API_KEY`；缺失或健康检查失败会阻断演化，不会以 mock 候选替代。

模型名需与服务实际的 served model name 一致。配对宜使用单个 endpoint；
请求 seed 会进入准备阶段和两臂的模型请求，但不承诺不同 GPU 批处理调度下逐位复现。
`--max-solver-rounds`、`--max-retries-per-operation`、`--finalization-rounds`
可覆盖 YAML；其余图像、token、沙箱和检索预算从 YAML 加载。两臂配置由同一份配置复制，
公共模板、执行协议、工具文档和 Skill 快照固定为当前合同，不提供 YAML 或 CLI
版本选择。

修订基线仅更新 S01/S08 为 1.2.0，其他六条仍为 1.1.0，修订记录不构成演化收益。
历史快照和历史提示词不再由当前入口加载；复现实验时 checkout 对应 Git 提交。
模板版本、执行协议版本、工具文档版本与工具实现版本仍分别进入运行记录。

预抽帧数据可使用：

```bash
PYTHONPATH=src python scripts/run_skill_ablation_v11.py \
  --source jsonl --episodes-jsonl data/panels/s03.jsonl \
  --artifacts-json data/panels/s03-artifacts.json \
  --expected-qa-ids data/panels/s03-qa-ids.json \
  --vllm-endpoint http://127.0.0.1:8100 \
  --seed 137 --output-dir data/evaluations/v11-s03-jsonl-seed137
```

JSONL 沿用 `load_jsonl_items` 格式：每行含 `qa_id`、`scene_name`、`dataset`、
`question_type`、`question`、`options`、`ground_truth`、`split` 和 `frame_paths`。
`artifacts-json` 是 `{ "qa_id": "/absolute/path/to/artifact.json" }`；
`expected-qa-ids` 是预登记 ID 的 JSON 数组，可检测缺题。建议使用绝对路径；
重建内引用可按当前工作目录或 artifact 目录解析，若两者都存在且不同则拒绝猜测。

输出目录必须不存在。输入身份缺失、重复 ID、帧集/源/数组维度不一致会明确失败。
VSI 加载遗漏预登记题时保留 `source_failure.json` 中的完整抽样和排除清单，
整次运行标为未完成，不把成功加载的子集冒充完整面板。

## 质量门配置确认

运行记录冻结实际门限、质量指标版本和质量模块源码 hash；已有 artifact 的质量结果
直接复用，未计算时在私有准备目录实算一次。各题实际 `gate_thresholds` 另行保存。
源码的 TODO 默认门限不会被自动标记为“已确认”。

查看当前身份：

```bash
PYTHONPATH=src python scripts/run_skill_ablation_v11.py --print-quality-contract
```

正式报告需传入已有验证依据对应的 JSON：

```json
{
  "confirmed": true,
  "sha256": "上条命令输出的 sha256，须与已验证配置一致",
  "evidence_ref": "已有门限验证记录的位置"
}
```

使用 `--quality-confirmation path/to/confirmed-quality.json`。它不改变门限；
配置 hash 或 artifact 实际门限不一致即拒绝。未提供时仍可运行和检查结果，
但 `formal_result_eligible=false`。不得仅为改变这个字段而伪造验证记录。

## 冻结与隔离

1. 校验所有题目、实际 RGB 像素（形状、dtype、内容 hash）、帧顺序、源帧号、
   时间戳、场景和基础产物所有非空引用。相同 artifact 身份对应不同内容时拒绝。
2. 每题只复制声明的基础数组到私有目录，忽略源目录旧 M5 缓存。
   在无 Skill 的准备阶段执行一次 M5（含题目定向补漏），冻结对象记录、点云、
   mask、统计和质量结果。所有题准备完成后才开始求解。
3. 每臂每题物理复制整个准备目录，重新定位 JSON 中的内部路径；
   不使用软链或硬链。`PreparedObjectBinding` 在 runner 内再次深复制，
   每次运行新建 SceneHandle、题级 SceneState、图像账本、kernel、答案/让出槽和结果登记
   （当前源码没有独立命名为 `QuestionState` 的类）。
   程序轮次之间的命名空间重置沿用原 runner 合同。
4. 求解只使用各自的像素副本和私有产物；episodic memory 不参与。
   验证两臂初始树 hash、公共配置 hash，以及剥离完整 Skill 块后的首轮实际请求 hash。
   每臂结束后核对冻结树和原输入未被改写。

M5 的耗时和请求只计在共享准备阶段，求解成本从两臂 trace 分别计算。
当前为顺序运行 B01、B11；输出会记录顺序，不是延迟公平性实验。
复制会占用约一份冻结输入加两份运行产物的空间，适合先做小规模切片。
这不是容器安全平台。

## 输出与评分

- `manifest.json`：实验 ID、预登记题单、实际 seed、公共配置、输入身份、
  冻结 snapshot/manifest、代码 hash、质量配置及每题准备产物 hash。
- `skills/`、`frozen/`：实际使用的正文、图像、基础产物、对象绑定和质量结果。
- `preparation/<qa哈希>/model_requests.jsonl`：共享准备阶段的请求收据。
- `arms/B01|B11/<qa哈希>/result.json` 和 `trace/`：独立 run/qa/arm 身份、
  答案、官方得分、运行错误、合法答案状态、实际交付版本/hash、
  请求参数与图像观察记录。
- `paired_results.jsonl`：按 `qa_id` 严格连接的逐题结果，两臂答案与逐题差值。
- `summary.json`：总体和各题型的 Accuracy、MRA、合法答案率、程序错误率、
  最终运行错误率、输入错误率及 B11−B01；指标范围为 0–1。

Accuracy 和 MRA 使用现有 runner 的官方评分器。程序/格式失败或无答案计零，
不从评分分母删除。程序错误率表示出现过执行错误、AST 拒绝或生成解析失败的题占比，
即使之后恢复成功仍计入。服务失败保留 `score=null` 并将实验标为 `incomplete`；
预登记题目的全部图像缺失或不可读时，两臂均生成 `input_error` 零分结果，不调用模型，
并继续保留在配对分母。不能把服务失败运行的部分分数作为完整实验结论。
差值只取两臂均完成的同一题集合。
平分和负差值均是有效实验结果。

复算汇总：

```python
import json
from pathlib import Path
from skill3d.evaluation.skill_ablation_v11 import summarize_pairs

path = Path("data/evaluations/v11-s03-seed137/paired_results.jsonl")
pairs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
print(json.dumps(summarize_pairs(pairs), ensure_ascii=False, indent=2))
```

CLI 退出码：0 = 完成（与分数正负无关），1 = 有未完成题，
2 = 输入合同、环境或其他执行错误。异常原因保存在 `failure.json` 或
`source_failure.json`；不会切换 mock，也不自动重跑到得到更高分。

## 本地合同测试

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/integration/test_skill_ablation_v11.py \
  tests/integration/test_v11_skill_delivery.py \
  tests/unit/test_m11_submission.py \
  tests/unit/test_v11_prompt_contract.py \
  tests/unit/test_v11_tool_contract_repair.py
```

测试使用合成几何、可注入假模型客户端及 M5 夹具，真实执行 runner、工具、
程序、质量路由与评分。它验证合同和隔离，不代表真实模型的 Skill 效果。
