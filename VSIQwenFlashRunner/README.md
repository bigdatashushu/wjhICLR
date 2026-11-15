# VSIQwenFlashRunner

本目录运行主数据集 VSI-Bench 抽取的 1000 题在 `qwen3.8-flash` 上的实验。
已迁移运行入口、九阶段编排、fit-set 执行、主求解器和 Skill 演化主干。
后续实验优化所需的源码、脚本、配置和验证均在本目录维护。

仓库外层代码已适配其他数据集和实验配置。本目录的开发约束见
[AGENTS.md](AGENTS.md)。本地模块优先加载，公共数据读取、几何、工具及其他
稳定依赖从外层 `src/skill3d` 只读加载。源码来源见 [migration.json](migration.json)。

## 入口

从本目录执行：

```bash
# 预览九阶段和当前数据包路径，零模型请求、零文件写入。
./experiment.sh --dry-run

# 核对本地模块加载路径及本地/共享源码身份，零模型请求。
./experiment.sh --show-source

# 本地题单、base catalog、场景概览和划分检查，零模型请求。
./experiment.sh --check
```

也可直接执行 `python -B /path/to/VSIQwenFlashRunner/run.py`，接受相同参数。
启动脚本优先使用 `H3D_PYTHON`、本目录或外层的已有虚拟环境，
也可通过 `--python /path/to/python` 指定解释器。

在能读取原场景资产的服务器运行时显式传入 `--data-root`：

```bash
./experiment.sh \
  --data-root /path/to/deployed/vsibench-m4-success-g1m1-1000-seed137-r01 \
  --output-dir /path/to/VSIQwenFlashRunner/runs/vsi-r01 \
  --max-requests 10000

./experiment.sh --output-dir /path/to/VSIQwenFlashRunner/runs/vsi-r01 --resume
./experiment.sh --output-dir /path/to/VSIQwenFlashRunner/runs/vsi-r01 --resume --check
```

`10000` 是预算示例，实际实验需指定自己的总请求上限；默认配置保留
`max_requests=0`，正式执行前要求正预算。默认输出为本目录
`runs/vsi-bench/<UTC时间>/`，显式输出也必须位于本目录 `runs/` 内。

九阶段沿用原流程：初始 Skill 冻结、训练前测试、训练演化、最终 Skill 冻结、
训练后测试、配对评分、32 帧直接基线、基线评分及文件封存。

## 目录

| 路径 | 用途 |
| --- | --- |
| `run.py` / `experiment.sh` | 本目录统一运行入口 |
| `src/skill3d/infra/` | 本地阶段编排、源码冻结和来源核验 |
| `src/skill3d/evaluation/` | 本地 fit-set 执行、评分与直接基线支持 |
| `src/skill3d/evolution/` | 本地槽位演化、修订和轨迹分析主干 |
| `src/skill3d/online/runner.py` | 本地主求解器 |
| `src/skill3d/online/vsi_execution_contract.py` | Python/最终 JSON 示例与 VSI 接口合同 |
| `src/skill3d/online/vsi_kernel.py` | 结果读取、跨轮分页及精确测量答案构造 |
| `src/skill3d/sandbox/ast_guard.py` | 本地静态守卫，新增接口按保留名检查 |
| `scripts/` | 本地 campaign、阶段和 fit-set CLI |
| `configs/vsi_m4_1000.json` | 原 VSI1000 实验配置的独立副本 |
| `tests/` | 本目录实现的必要验证 |
| `runs/` | 本目录实验的配置快照、轨迹、评分和运行记录 |
| `prepared_cache/` | 后续优化生成的新缓存 |

`runs/` 和 `prepared_cache/` 的运行产物由本目录 `.gitignore` 排除。

## 原始数据与配置来源

- 本机数据：`/Users/bytedance/wangjh/vsibench-m4-success-1000`。
- 服务器数据：`/data/wangy/wangjh/harness3d/experiment-inputs/vsibench-m4-success-1000-seed137`。
- 原配置：`../configs/experiments/vsi_m4_1000.json`。
- 原统一入口：`../experiment.sh`。
- 主模型：`qwen3.8-flash`；模型、并发和请求预算沿用原配置，预期场景数为 160。
- 当前数据：`prepared_cache/vsibench-m4-success-g1m1-1000-seed137-r01`，默认 base/g1m0。
- 题目划分：300 题训练、700 题测试，共 160 个场景；两个角色均覆盖 160 场景，
  seed 137。按核验清单替换 20 题（训练 7、测试 13），保留 980 题原始字节。
- 原始题型及方向难度、source split、训练/测试角色联合配额完全保持不变。
  新题单全部具有已核验的 metric/g1m1 资产，同时保留原 base 与 metric 引用。
- 原始帧输入为 32 帧，系统臂另有场景总览输入；直接基线沿用原 32 帧协议。

独立配置中的 `{data_root}` 默认指向当前独立数据包，`{seed_root}` 优先指向本目录
的 `skill_library/`；本目录没有独立库时才使用外层原始库。正式运行把实际选中的库
复制进该轮 `code/source/skill_library/`，所有阶段从这一冻结副本读取初始 Skill。
`--show-source` 会输出实际的 `seed_library_root`。已发布的 32 帧输入继续使用原冻结引用。
2026-11-10 按用户要求撤回 11 月 5 日的默认 metric/g1m1 切换。当前 panel 和发布输入
provider 使用旧 r01 包的 base catalog，1000 题均进入各自题型的 g1m0 Skill 槽位。
包名中的 g1m1 表示抽样时的 metric 资格；实际运行状态以 panel 和原资产为准。
g1m0 使用几何但不启用米制尺度。现有 24 槽库包含八类题的 g1m0 Skill，
旧的 19 槽产物仍可按原合同读取。

绑定从已发布 v3 资产派生为当前 v4，保留原来源和审核参数，不请求新模型。
派生缓存写入 `{run_root}/prepared_cache/derived_bindings_v4`，由三次系统阶段
复用；正式运行的逐题 preflight 会验证原场景资产、实际证据状态和绑定。
`--check` 验证本地输入合同，不代替服务器的逐题运行准备。

## 数据替换与比例核验

`scripts/update_metric_dataset.py` 按哈希固定的清单生成新包，并核验官方题目、
选项与 host-only 标签、原始联合配额、两个角色的场景覆盖和 1000 题 metric 资格。
`eligibility/` 保存原审计和 20 组替换清单；原数据包与原发布资产保持只读。

```bash
python -B scripts/update_metric_dataset.py validate \
  --package prepared_cache/vsibench-m4-success-g1m1-1000-seed137-r01 \
  --native
```

服务器上追加 `--replay` 可离线重放 20 道替补题的正常 provider 输入，核验 32 帧
与空参考答案。该操作不请求模型或 GPU，也不重新计算几何。

历史上的 `scripts/configure_g1m1_dataset.py` 从 r01 生成独立 metric 运行包。该包逐字节保留
题单、划分、模型配置、原 catalog、标签及 160 张概览 PNG；metric 绑定和概览源身份
通过原 catalog 及服务器探针连接。八类题的真实准备验证入口为
`scripts/validate_g1m1_runtime.py`，验证记录保存在 `runs/g1m1-config-20261105-r01/`。
metric-r02 包保留供显式 `--data-root` 复现；默认入口使用 r01。小样本对比仍显式传入
`--package`，其 controller 和 scorer 按冻结 panel 校验证据状态，兼容 base/g1m0 与
metric/g1m1。此次回退保留题单、模型参数、提示词精简和状态修复；相对方向原十题及
新增二十题复测结果见文末。

## 源码冻结与续跑

正式运行将本地实现和共享依赖合并到该轮 `code/source/`，共享同名文件由本地
版本覆盖。复制范围为源码、必要公共配置、初始 Skill 库和小型 VGGT 源码，
外层其他 runner、数据、权重、运行产物和凭据文件不进入快照。
冻结后的 Python 包只在该快照内查找模块。

`code/code_identity.json` 保存逐文件 SHA256 和本地/共享来源，并保留真实的
Git base commit 与 `git_dirty`。本目录以完整源码内容冻结运行身份，不要求
为迁移自动创建 Git commit。冻结时检查源文件变化，续跑检查快照、输入、
解释器、完整阶段回执和阶段产物，已完成阶段直接跳过。
原目录中断请求的处理规则沿用原编排器。

## 后续开发

新增实验入口及优化逻辑写在本目录；复用外层模块时保持只读。
需要修改共享模块时，按原 `src/skill3d/...` 路径复制到本目录。新建本地子包时
参考已有子包的 `__init__.py` 保留明确的共享查找路径，再运行 `--show-source`
核对实际加载来源。
现有 `VSIbench8Brunner/` 属于另一实验目录，本目录不自动继承其 8B 描述逻辑。

原数据和已发布缓存保持原身份，新增缓存和实验输出使用本目录独立路径。
训练题用于学习和候选选择，测试题用于冻结后的评估。每轮实验记录实际使用的
源码、配置和输入身份，在线求解保持参考答案不可见，凭据从环境变量读取。

## 离线验证

本地求解器新增三个普通 Python 阶段接口，入口直接加载本地 kernel，
正式源码冻结也包含这些实现：

- `ToolData(result)` 提取原始工具的列表、字典或标量数据；`ToolData(page)`
  返回分页的列表。工具结果取自当前真实账本，分页取自已记录的 host 副本，
  用户变量中的修改不会改写证据。原工具 envelope 必须携带实际数据且与账本一致，
  不能伪造仅含 ID 的 envelope 来绕过历史结果的分页读取。
- `ReadNextObservation(result_or_id, fields=None, limit=8)` 从未读区间继续读取，
  保留原 reader 的三次/轮、32 项/页、4000 字符/页上限。
  跨轮使用反馈中的真实 result ID；不重跑检测。省略 fields 时沿用上一选择，
  改变字段则独立累计覆盖。`page["vsi_read_progress"]["coverage_complete"]`
  表示这些字段的累计覆盖，原 `coverage_complete` 仍表示单页覆盖。
- `return ReturnAnswer(MeasurementAnswer(result))` 按当前题型构造完整载荷：
  距离为米、最长尺寸从米精确换算为厘米、房间面积为平方米、计数为整数。
  数值保留完整精度，推导仍由原始 verifier 对真实 ToolResult 重放；
  原有几何检查、证据授权及数值容差保持不变。

例如，在获得已授权的测量结果后，可直接提交：

```python
def solve(ctx):
    # Replace the literal with the actual valid measurement result ID in feedback.
    return ReturnAnswer(MeasurementAnswer("ACTUAL_RESULT_ID"))
```

普通轮输出单个可执行 Python 代码块，每条作答或观察分支显式终结。
最终轮输出单个 JSON 对象，使用 JSON 数字和 `null`；此时不调用工具或新增接口。
完整语法示例按当前题型注入请求，示例数值只展示语法。
请求及轮记录新增独立的 `vsi_interface_version` 和合同 SHA256；
基础执行合同哈希仍对应原合同。接口活动记入 `vsi_interface_calls`，
不计为工具调用；真实分页仍有原 reader 授权回执。

迁移验证使用合成输入和真实本地子进程；原 fit-set、槽位演化、场景总览、
修订失败处理、静态修复及直接基线验证在本地模块上回归。

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests
```

本次迁移与离线验证没有请求真实模型或启动 1000 题正式实验。

## 八类题型提示词精简

本地 `synthesis/prompt_builder.py` 统一精简八类题型的普通轮、恢复轮和最终 JSON
轮提示词。执行规则合并为一处，推导参数说明按题型显示；工具描述经过压缩，
签名、依赖、动态单位和真实授权仍由原 REGISTRY 提供。M5 重复失败按原因累计，
保留各自的阻断次数、角色、匹配/可用对象 ID、扫描覆盖和错误；完整回执继续留存。
原帧用有序槽位列表表示，派生图、裁剪来源和未交付原帧仍明确列出。

完整选中 Skill 逐字保留，交付哈希继续对应正文。执行合同继续使用原始完整合同
哈希；精简展示有独立的 `vsi-prompt-compaction/1.0` 版本。请求的工具目录哈希对应
实际发送的精简文档；最终轮明确记录未交付工具文档。旧实验继续使用其冻结源码。

在相同的合成问题、完整 Skill、G1M1 工具能力上界和 32 帧清单下，八类初始提示词
从 30,592–35,769 字符减至 15,813–20,406 字符，减少约 43%–48%。
QA373 的原始公开上下文离线投影从 **45,334 减至 21,611 字符（52.33%）**，
保留全部 33 张图的顺序映射和 5,796 字符的完整 Skill。
这些是文本长度检查，未运行模型准确率实验。

详细统计及 QA373 完整精简文本见
[精简报告](prompt-compaction-20261110/README.md)。
可用 `scripts/measure_prompt_compaction.py --baseline-ref <原版本commit>
--output-dir <新目录> --public-prompt <原始公开提示词.txt>` 离线复核全部
24 个题型/证据状态组合。

## 精简提示词相对距离与计数 20 题配对实验

2026-11-10，从固定测试集按 seed 137 各抽 10 题，比较精简工具求解器与
`qwen3.8-flash` 原始 RGB 直答。两组均为 thinking=true、temperature=0、
max_tokens=8192，使用相同的 32 张原始 JPEG；工具组另有场景总览和完整冻结 Skill。
参考答案仅供离线评分，未进行 Skill 演化。

| 题型 | 题数 | 指标 | 精简工具组 | 直答组 | 差值 |
|---|---:|---|---:|---:|---:|
| 相对距离 | 10 | Accuracy | 80.0% | 50.0% | +30.0 pp |
| 物体计数 | 10 | MRA | 32.0% | 66.0% | -34.0 pp |
| 两类均分 | 20 | 等权均值 | 56.0% | 58.0% | -2.0 pp |

工具组位于 `runs/compact-relcount20-c32-20261110-r06/`，直答组复用
`runs/compact-relcount20-c32-20261110-r05/` 的全部 20 个已完成结果。
工具组 32 次模型请求、744,385 Token；直答组 20 次请求、298,852 Token。
并发上限 32，实际峰值 19。QA4559 的检测服务未配置，保留其 incomplete
状态并按 0 分计入；其原始答案“1”及参考答案“2”均保存在离线逐题结果中。
两组实际请求参数、原始 JPEG、冻结源码、输入及收集回执均经离线评分脚本核对。

逐题结果见 [comparison.json](runs/compact-relcount20-c32-20261110-r06/comparison.json)。
可用 `scripts/score_small_comparison.py --run-root <工具组目录>
--direct-run-root <直答组目录> --report <输出.md>` 重新进行离线核验和评分。
此前 r04 的网关回执校验失败记录单独保留，其 40 次请求及 797,650 Token
不计入本次结果。

## 物体计数上下文特化

2026-11-10，按用户要求汇总五组轨迹审查，并仅对 `object_counting` 接入
`vsi-counting-context/1.0`。先独立枚举真实整体实例，再复核共现、跨帧身份、
遮挡／遗漏及整体／部件边界。图像充分时允许零工具；候选分页、裁剪和补检用于
具体疑点，固定字段沿 cursor 继续，避免重列和不变失败重试。

实际计数提示移除固定数字和默认全库存查询示例。`MeasurementAnswer` 继续可用，
说明其仅取值／回放、不认证实例完整。内层检测故障单列为失败诊断并保留完整错误。
终结轮分页用有界预览，保留原始条目路径、来源 ID、cursor 和覆盖／省略区别。

三个计数 Skill 槽位已发布为 `1.1.0`，其他 21 个活跃正文和旧版本保持。
111 项离线检查通过；与 r06 冻结源码对照，其他七题型 × 三状态的普通／终结提示、
工具文档和完整 Skill 全文哈希一致。九份普通计数提示平均
**20,602.89 → 17,221.89 字符（减少 16.41%）**；QA11 的终结轮重建与归档请求
哈希匹配，**12,787 → 10,554 字符（减少 17.46%）**。

本轮未调用真实模型，未测新准确率。完整审查和修改说明见
[报告](../../VSI物体计数上下文特化-20261110.md)，逐题新上下文和哈希见
[离线产物](runs/counting-context-offline-20261110/metrics.json)。
可用 `scripts/measure_counting_context.py` 复核冻结来源和长度；
Skill 发布身份见 [验证凭据](skill_library/validation/counting_context_20261110.json)。

## 物体计数状态与接口修复、原十题复测

2026-11-10，仅对计数路径接入 `vsi-counting-context/1.1`：
成功、AST 拒绝、工具错误及终结反馈复用同一 episode 状态，保留累计选字段读取覆盖和
最多 16,000 字符的原始页；AST 展示规则由实际守卫常量生成；
每张交付图前紧邻稳定 `image_id`；公开 `count_objects` 当前绑定门禁及恢复边界。
`CountingEvidence` 记录实例、图像、区域、纳入／排除／合并及理由，Host 只检查
当前请求已交付图像、合法区域与合并引用。计数校验布尔字段仅表示实际检查并通过，
提交许可、引用、推导重放、像素语义及实例覆盖另记 `verification_scope`。
三个 Skill 沿用已冻结 `1.1.0`。144 项相关离线回归检查通过。

原十题在 API 并发上限 10、thinking、seed 137、temperature 0、max_tokens 8192 下
完成复测，复用 r07 全部直答结果，本轮新增直答请求为 0。原题单和 32 帧 JPEG 身份
保持，系统组仍提供场景 overview。源码、输入、请求及收集身份已离线核验。

| 指标 | 上轮工具组 r07 | 修复组 r08 | 已有直答 |
|---|---:|---:|---:|
| 官方 MRA | 55.0% | 46.0% | 75.0% |
| 模型请求 | 17 | 11 | 10（复用） |
| 主动工具调用 | 6 | 0 | 0 |
| 总 Token | 415,291 | 300,695 | 175,707 |

9 题答案保持，仅 QA4559 从 2 改为 1，导致均分下降 9 个百分点。
9/10 题提交结构化实例记录，但 0/10 题主动调用工具。QA4462 的首轮 AnswerPayload
字段格式错误，下一轮恢复提交，未补交实例记录。全部 API 请求成功，无检测服务故障。
本轮没有分页或裁剪，因此未实际触发“读取完整页面后报错”的恢复场景；
像素语义、实例身份及覆盖仍标为未验证。接口修复本轮没有提高准确率。

实际请求池和上游峰值均为 8，总墙钟 10.1 分钟。初始提示平均
17,236 → 22,274 字符；轮次减少使总 Token 下降 27.6%。
完整逐题报告见 [状态修复十题对比](../../VSI物体计数状态修复10题对比-20261110.md)，
实际图像邻接标签 SHA256、分项校验、Token 明细及恢复记录见
[运行审计](runs/counting-statefix10-c10-20261110-r08/counting_state_audit.json)。
冻结配置及通用配对评分见
[comparison.json](runs/counting-statefix10-c10-20261110-r08/comparison.json)。

## 计数全景图多轮与直答对照

`scripts/run_counting_panorama_pair.py` 支持冻结清单中的 `runtime.max_rounds=5`。
直答组每题一次请求、32 帧；全景组每题最多五轮，每轮发送同样的 33 张图。
每轮保留原始问题，并在最后一张全景图前保留说明。提示词明确告诉模型历史不累计；
后续请求仅携带上一轮的 `next_prompt` 摘要，不携带此前的 messages 或 thinking。
确定时可提前输出整数；不确定时输出 `{"next_prompt":"供下一轮复核的摘要"}`。
第五轮明确要求必须给出答案。全部调用开启 thinking，沿用 seed 137、
temperature 0、max_tokens 8192，API 并发上限 32。

本轮目录为 `runs/count-panorama-multiround10-c32-20261110-r10/`。
每轮完整消息及图像、原始回复、摘要交接、Token 和传输身份分别保存于
`messages/`、`panorama/rounds/`、`requests/`。冻结产物不允许静默重复运行。
部署与收集入口为 `scripts/control_counting_panorama_multiround.py`；
运行完成后用 `.venv/bin/python VSIQwenFlashRunner/scripts/score_counting_panorama_multiround.py`
在仓库根目录执行本地评分，并逐轮核对原问题、最后一张图说明、33 张图、
最新摘要和实际 thinking 参数。

本轮两组全部完成：直答 MRA **73.0%**，全景多轮 **63.0%**，
差值 **-10.0 pp**；完全正确分别为 7/10、5/10。
3 题主动续轮且均在第二轮作答，其余 7 题首轮作答，共 13 次多轮请求。
总 Token 分别为 159,313、199,358，多轮增加 25.1%。
23 次请求均成功，无自动重试；实际并发峰值 20，总墙钟 340.8 秒。
逐轮核验通过，独立网关已停止。完整逐题结果见
[全景多轮对比报告](../../VSI计数全景多轮与直答10题对比-20261110.md)、
[comparison.json](runs/count-panorama-multiround10-c32-20261110-r10/comparison.json)。

## 相对方向十题 base/g1m0 回退复测

2026-11-11，使用上轮相对方向的同十题、同原始 RGB 和同模型参数，
恢复 base 资产及 `S06-g1m0@1.0.0`；十题逐题 preflight 均确认 g1m0。
thinking=true、seed 137、temperature 0、max_tokens 8192，API 上限 32，
本轮实际请求峰值 10。只运行工具组，复用 r11 的十个直答结果。

工具组仍为 **6/10（60%）**，上轮 metric/g1m1 同为 6/10，直答为 9/10。
QA3711 由错转对，QA1536 由对转错；QA1860、QA3663、QA4259 仍错。
新增模型请求 19 次全部成功，无超时或自动重试；Token **537,066**，
较上轮已知 898,443 减少 **40.2%**。32 次工具调用中有 5 次方向查询，
2 次被 confidence_gate 拒绝。此结论仅来自这十题。

运行目录为 `runs/rel-direction10-g1m0-c32-20261111-r13/`，186 个结果文件已收集
并核对哈希；独立网关已停止。报告见
[回退复测](../../VSI相对方向10题g1m0回退复测-20261111.md)，
逐题结果及实际请求审计见
[comparison.json](runs/rel-direction10-g1m0-c32-20261111-r13/comparison.json)。

## 相对方向新增二十题 base/g1m0 对照

2026-11-11，按 seed 137 从测试集排除原十题后新增抽取 20 题，覆盖 16 场景，
简单／中等／困难分别为 3／8／9 题。工具组实际使用 base 资产和
`S06-g1m0@1.0.0`；本轮两组均新跑，thinking、temperature 0、8192 tokens、
相同原始 32 帧及工具组的场景总览沿用原协议。API 上限 32，实际峰值 20。

| 范围 | 工具组 | 直答 | 差值 |
|---|---:|---:|---:|
| 新增 20 题 | 17/20（85%） | 16/20（80%） | +5.0 pp |
| 两批共 30 道不同题 | 23/30（76.7%） | 25/30（83.3%） | −6.7 pp |

新增工具组 47 次模型请求、已知 Token **1,201,524**；直答 20 次、
**399,141** Token，比例 **3.01 倍**。11/20 题主动调用工具；64 次调用中有
30 次方向查询，12 次被 confidence_gate 拒绝。QA4325 原请求等待约 661 秒失败，
仅该未完成题在独立批次把等待上限改为 1200 秒后补跑，3 次请求全部成功。
上述 47 次包含原失败请求，其未知 Token 未计入；其它题及直答组未重跑。

原批次位于 `runs/rel-direction20-g1m0-c32-20261111-r14/`，补跑位于
`runs/rel-direction20-g1m0-completion-c32-20261111-r15/`。
396＋37 个文件及归档哈希、参考答案隔离、实际参数和 RGB 配对均已核验，
两个网关的进程退出和端口关闭均已确认。原十题与新增二十题无重叠，
134 个求解器、提示词和 Skill 源文件一致。
完整结果见 [新增二十题报告](../../VSI相对方向新增20题g1m0与直答对比-20261111.md)、
[comparison.json](runs/rel-direction20-g1m0-c32-20261111-r14/comparison.json) 和
[三十题汇总](runs/rel-direction20-g1m0-c32-20261111-r14/comparison30.json)。

## 相对方向最小方法修订

2026-11-11，用户确认五 Agent 轨迹诊断后的改稿，发布
`S06-g1m0@1.1.0`。新增三角色 `obj_id/image_id/区域` 的短注释依据，
检查候选框或掩码本身；实例未确认时不以最近距离或首个成功结果代替消歧，
缺绑定时保留该物体的视觉位置。视觉回退在同一水平布局中比较
`observer→facing_at` 与 `observer→target`，保持 medium 的 135° 定义。
共享提示词同步替换一条候选身份规则，保留 M11 技术检查与实例核对的区别。

S06 正文增加 132 字符，共享单句减少 7 字符。通过原生发布器保留旧版本并切换
本地 active snapshot；其他 23 个槽位和 88 份旧不可变文件哈希一致。
默认仍使用 base/g1m0，配置、门禁、方向映射及旧 runs 未改。
完整新正文与共享规则已通过本地普通轮实际渲染检查。
来源和发布回执见
[修订核验](skill_library/validation/relative_direction_context_20261111.json)。
提示词交付、默认路由与源码冻结相关的 76 项离线检查全部通过；
同二十题的准确率复测记录如下。

## 相对方向 S06 新版同二十题复测

2026-11-11，重跑原新增二十题，使用 `S06-g1m0@1.1.0` 和替换后的共享规则；
base/g1m0、原始 32 帧、场景总览、thinking、seed 137、temperature 0 和求解预算沿用
原协议。20/20 条完整轨迹均核对了新版 Skill 的正文哈希和实际提示词交付。
已有直答结果逐题核验 RGB 和模型参数后复用，没有重复调用。

新版为 **16/20（80%）**，修改前为 **17/20（85%）**，下降 **5 个百分点**；
Qwen 直答为 **16/20（80%）**。QA1556、QA3575 错转对，
QA1558、QA2425、QA2426 对转错；新版错题为 QA1558、QA2425、QA2426、QA3588。
这批题曾用于轨迹诊断，本轮结论只描述同题复测的变化。

最终二十条完整轨迹的已知 Token 为 **1,042,041**，较修改前减少 **13.3%**。
原批次 QA2425 超时，QA1558、QA3588 因 ENOSPC 未保存完整 episode；
只补跑这三题，并把请求等待上限由 660 秒延长至 1200 秒，5 次补跑调用全部成功。
两道存储故障题在原批次中的 4 次成功调用耗费 **134,639** Token，全部保留。
实际已知总消耗为 **1,176,680** Token，较修改前减少 **2.1%**，为直答的 **2.95 倍**；
原超时请求的 Token 未知，未计入已知统计。

全程 41 次模型请求，最终二十题有 5 题调用工具，共 23 次调用。
API 上限 32，实际峰值 14。原批次请求种子、原始账本与网关数据库只读备份逐条核验，
通过旁路账本补齐两条未落盘的请求终态，原始状态、日志和未更新 summary 均保留。
原批次无正常调度结束回执，按存储中断证据收集；补跑正常退出并收集完整结束回执。
两批 289＋68 个结果文件、归档、源码和输入哈希均已核验，
两网关在 active=0 时核验进程身份后关闭，并确认进程退出与端口关闭。

完整结果见 [新版同二十题复测报告](../../VSI相对方向20题S06新版复测-20261111.md)、
[comparison.json](runs/rel-direction20-s06v110-c32-20261111-r16/comparison.json)；
原批次为 `runs/rel-direction20-s06v110-c32-20261111-r16/`，
补跑为 `runs/rel-direction20-s06v110-storage-completion-c32-20261111-r18/`。

## 相对方向分支与状态呈现修复

2026-11-11，依据上述复测和错题对照发布 `S06-g1m0@1.1.1`。
角色 ID 不确定时先读取候选；当前方向工具获授权、所选实例绑定和世界约定成立时
核对几何结果，实际前置条件阻塞后再视觉回退。三角色图像区域、O→F / O→T 和
135° 提醒保留；正文增加 51 字符。

方向题的提示词把候选覆盖与具体绑定分开：`candidate_coverage_complete`
描述候选集合覆盖，`usable_object_ids` 由当前句柄逐实例复核，
`relative_direction_of_authorized` 与实际工具目录使用同一注册表判定。
覆盖失败明细继续呈现；原始 M5 收据及运行门禁保留原合同，其他题型保留原呈现。
世界总览退化提示仍保留，但不能替代具体调用的权限和世界约定检查。

原生发布器保留 91 份旧不可变源／快照／manifest，其他 23 个槽位未变。
提示词、默认路由和源码冻结的 82 项离线检查通过，完整新正文交付已核对。
发布时尚未重跑准确率，新增三十题结果见下一节。修订记录见
[分支修复核验](skill_library/validation/relative_direction_branch_20261111.json)。

## S06 补丁版新增三十题对比

2026-11-11，从固定测试集按 seed 137 另抽 30 道相对方向题，排除此前 30 题，
覆盖 26 个场景，easy / medium / hard 分别为 7 / 11 / 12 题。
工具组使用 `S06-g1m0@1.1.1`、base/g1m0；Qwen 直答在相同题单上新跑。
原始 32 帧、thinking、temperature 0、8192 Token 和求解轮次预算沿用原协议；
本轮请求超时设为 1200 秒，API 上限 32，实际峰值 30。

| 组别 | 正确 | 准确率 | 模型请求 | 总 Token |
|---|---:|---:|---:|---:|
| S06 补丁版工具组 | 25/30 | 83.3% | 66 | 1,755,781 |
| Qwen 直答 | 25/30 | 83.3% | 30 | 525,580 |

两组准确率持平，工具组 Token 为直答的 3.34 倍。96 次请求全部成功，
无故障补跑或未知 Token 成本。工具组独对 1551、1571、2469、3620；
直答独对 1813、3584、3658、4240；共同错题为 3647。
工具组全部错题为 1813、3584、3647、3658、4240。
本轮题目与此前二十题不同，得分差不能单独归因于 Skill 修订。

分难度，工具组／直答为 easy 7/7／6/7、medium 10/11／9/11、
hard 8/12／10/12；前两类各多对 1 题，被 hard 少对 2 题抵消。
602 个回传文件的哈希及 30 对原始 32 帧身份核验通过；
30/30 题完整新版 Skill、共享规则及方向授权呈现均通过交付核验。
本轮自有网关进程已退出，27063 端口关闭。

运行目录为 `runs/rel-direction30-s06v111-c32-20261111-r19/`。
完整结果见 [新增三十题报告](../../VSI相对方向新增30题S06补丁与直答对比-20261111.md)
和 [comparison.json](runs/rel-direction30-s06v111-c32-20261111-r19/comparison.json)。

## 相对方向三十三题分层 Thinking 对比

2026-11-11，从原固定测试集按 easy / medium / hard 各 11 题随机抽取，
seed 137，排除此前 60 题，共 26 个场景。工具组继续使用
`S06-g1m0@1.1.1`、base/g1m0，并开启 thinking；相同题单的 Qwen 直答
关闭 thinking。两组原始 32 帧一致，工具组额外使用场景总览。
模型、temperature 0、max_tokens 8192 和工具组求解预算沿用原协议。

| 难度 | 工具组（thinking） | 直答（无 thinking） |
|---|---:|---:|
| easy | 9/11（81.82%） | 4/11（36.36%） |
| medium | 11/11（100.00%） | 8/11（72.73%） |
| hard | 6/11（54.55%） | 7/11（63.64%） |
| 总计 | 26/33（78.79%） | 19/33（57.58%） |

工具组多对 7 题，提升 21.21 个百分点；已知总 Token 为 1,998,821，
直答为 323,176，比例 6.18。工具组共有 74 次模型请求，直答 33 次。
QA1861 在 1200 秒等待上限中断，独立延长至 1800 秒补跑完成；
原来的 32 道工具组答案和全部 33 道直答答案未重跑。超时请求的未知
Token 未计入，全部成功请求与补跑的已知 Token 均计入。
工具组成功请求的输出 Token 为 798,735，其中 reasoning 为 780,149；
直答组输出 Token 为 66。

682 个回传文件的哈希、33 对原始帧身份、完整新版 Skill 和方向状态呈现
核验通过。106 个成功请求的实际 thinking 参数与失败请求的发出参数均核对；
API 上限 32，实际峰值 17。两个自有网关已退出，27064 / 27065 端口关闭。
本轮比较两套完整方案；此前直答也开 thinking 且题单不同，跨轮差值不能
单独归因于 Skill 或 thinking。

原批次为 `runs/rel-direction33-balanced-s06v111-c32-20261111-r20/`，
单题补跑为 `runs/rel-direction33-balanced-completion-c32-20261111-r21/`。
结果见 [三十三题分层报告](../../VSI相对方向33题分层Thinking对比-20261111.md)
和 [comparison.json](runs/rel-direction33-balanced-s06v111-c32-20261111-r20/comparison.json)。

## GPT Terra 分层对比接口

`prepare_small_comparison.py` 支持 `--model-provider xcodeapi`，使用独立复制到
本目录的 Xcode SDK，通过原生 `/v1/responses` 发送请求。显式指定
`--model gpt-5.6-terra --base-url https://xcode.best/v1`，
`--system-effort high --direct-effort low`；每个请求实际携带
`reasoning.effort`、`max_output_tokens`、`temperature` 和 `store=false`。
high 组使用原生 SSE 流式请求，low 组使用非流式请求；流式仅接受供应商的
`response.completed` / `response.incomplete` / `response.failed` 终态，
截断的增量文本不构成完整结果。抽样及框架 seed 保持 137；原生 Responses
不发送 seed。

网关在实际请求开始时轮换私密环境文件中的全部密钥，公开回执仅包含
密钥指纹、模型、推理强度和消耗。身份检查、配置构造及健康检查不会调用模型。
已知 Token 在响应解析前记录；失败请求的已知消耗参与成本统计。
几何输入通过 `--derived-replay` 和 `--derivation-root` 复用上一轮封存缓存，
核验完整原始场景 QA 来源和所有已发布派生绑定，每轮结果和源码另行冻结。

## GPT Terra 同题三十三题结果

2026-11-11，复用上一轮分层对比的同一组 33 题，easy / medium / hard
各 11 题。实际请求模型为 `gpt-5.6-terra`，工具组原生 reasoning effort
为 high，直答组为 low；base/g1m0、S06-g1m0@1.1.1、原始 32 帧及
工具组求解预算保持一致，封存预计算仍复用上一轮产物。

| 难度 | 工具组 high | 直答组 low |
|---|---:|---:|
| easy | 10/11（90.91%） | 4/11（36.36%） |
| medium | 7/11（63.64%） | 4/11（36.36%） |
| hard | 4/11（36.36%） | 5/11（45.45%） |
| 总计 | 21/33（63.64%） | 13/33（39.39%） |

工具组多对 8 题，差值 +24.24 个百分点；同题 Qwen 此前为
26/33（78.79%）和 19/33（57.58%）。本轮比较两套完整方案，
推理强度和协议不同，不能把差值单独归因于工具。

工具组 78 次题目请求，已知 Token 1,531,331；直答组 33 次，
Token 690,470，已知比例为 2.22 倍。工具组另有 24 次失败请求
消耗未知，未按零计入；11 次接口检查的 63,815 Token 单独记录。
工具组仅 5/33 题实际调用工具，共 11 次。

工具组初次启动因总览清单范围不一致而停止，未发出模型请求。
独立范围适配后保留 10 道非流式完整结果；23 道未完成题改用原生流式
补齐，36 次流式请求全部成功，其中 6 次耗时超过原来的约 127 秒
中转截断点。独立故障复现返回 HTTP 524。补跑未按正确性选择结果。

68 项本地检查、33 对题目及图片身份、完整 Skill 交付、111 次发送身份
以及 87 次成功响应的模型和 effort 核验通过。API 并发上限 32，
实际峰值 28；包括注释中的密钥在内，11 个密钥全部轮换使用。
四个自有网关的进程退出和 27066–27069 端口关闭均已验证，
本轮服务器及本地临时密钥副本已清理。

结果见 [GPT Terra 报告](../../VSI相对方向33题GPTTerraHighLow对比-20261111.md)
及 [comparison.json](runs/rel-direction22-gptterra-stream-completion-c32-20261111-r28/comparison.json)。

## DeepSeek V4.1 Flash 接口诊断

`deepseek-v4.1-flash` 使用本地 `DeepSeekResponsesClient`，通过
`/v1/responses` 发送流式请求；两组强度分别为 `reasoning.effort=max`、
`low`。成功响应严格核对 `deepseek-v4.1-flash-1017`，解析
`reasoning.content` 或中转服务返回的 `reasoning.summary`。模型版本、发送
参数、已知用量和失败传输记录分别保留；服务未回显 effort 时，不把发送审计
当作后台实际强度的独立证明。

2026-11-12，原始 32 图在 Chat 路由稳定复现约 63 秒 HTTP 504，
Responses 能返回完整结果，但两题完整流程验证仍有 504 和流中断。
等待上限已为 1800 秒；这些失败发生在上游响应阶段。只接受供应商的
Responses 终态，EOF 或提前 `[DONE]` 均不能用增量文本补成答案。
流诊断只保留事件类型、数量、字节和字符数，不记录远端文本。
17 项原生协议测试通过，完整 33 题复测尚未恢复。

详情见 [接口诊断与修复报告](../../VSI相对方向DeepSeekV41Flash接口诊断修复-20261112.md)。
