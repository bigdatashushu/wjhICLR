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

# 本地题单、metric catalog、场景概览和划分检查，零模型请求。
./experiment.sh --check
```

也可直接执行 `python -B /path/to/VSIQwenFlashRunner/run.py`，接受相同参数。
启动脚本优先使用 `H3D_PYTHON`、本目录或外层的已有虚拟环境，
也可通过 `--python /path/to/python` 指定解释器。

在能读取原场景资产的服务器运行时显式传入 `--data-root`：

```bash
./experiment.sh \
  --data-root /path/to/deployed/vsibench-m4-success-g1m1-1000-seed137-metric-r02 \
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
- 当前数据：`prepared_cache/vsibench-m4-success-g1m1-1000-seed137-metric-r02`。
- 题目划分：300 题训练、700 题测试，共 160 个场景；两个角色均覆盖 160 场景，
  seed 137。按核验清单替换 20 题（训练 7、测试 13），保留 980 题原始字节。
- 原始题型及方向难度、source split、训练/测试角色联合配额完全保持不变。
  新题单全部具有已核验的 metric/g1m1 资产，同时保留原 base 与 metric 引用。
- 原始帧输入为 32 帧，系统臂另有场景总览输入；直接基线沿用原 32 帧协议。

独立配置中的 `{data_root}` 默认指向当前独立数据包，`{seed_root}` 优先指向本目录
的 `skill_library/`；本目录没有独立库时才使用外层原始库。正式运行把实际选中的库
复制进该轮 `code/source/skill_library/`，所有阶段从这一冻结副本读取初始 Skill。
`--show-source` 会输出实际的 `seed_library_root`。已发布的 32 帧输入继续使用原冻结引用。
当前 panel 和发布输入 provider 使用原 metric catalog，1000 题均进入各自题型的
g1m1 Skill 槽位。g1m1 表示场景有可用几何和米制尺度；米制工具仍只对绝对距离、
物体尺寸、房间尺寸三类题授权。新运行使用 24 槽合同，旧的 19 槽产物可按原合同读取。

绑定从已发布 v3 资产派生为当前 v4，保留原来源和审核参数，不请求新模型。
派生缓存写入 `{run_root}/prepared_cache/derived_bindings_v4_metric`，由三次系统阶段
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

`scripts/configure_g1m1_dataset.py` 从 r01 生成独立 metric 运行包。新包逐字节保留
题单、划分、模型配置、原 catalog、标签及 160 张概览 PNG；metric 绑定和概览源身份
通过原 catalog 及服务器探针连接。八类题的真实准备验证入口为
`scripts/validate_g1m1_runtime.py`，验证记录保存在 `runs/g1m1-config-20261105-r01/`。

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
