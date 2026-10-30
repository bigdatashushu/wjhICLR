# 完整学习轨迹分析与锚点修订

入口为 `scripts/run_analysis_successor_v11.py`。它在独立目录内读取冻结学习轨迹，先逐批分析，再以完整分析集合生成一次锚点修订。配对评估由 `scripts/run_successor_gate_v11.py` 单独执行。运行目录、原 Skill 库与原请求账本分别绑定，不向原库发布候选。

## 可复用接口

`trajectory_analysis_v11.plan_analysis_evidence(native, analyst=..., parent=..., feedback=..., experience=..., on_fragment_refusal=...)` 接收已配置的原生证据规划器，返回 `(batches, plan)`。

接口复制规划器实例，并为复制实例建立独立函数命名空间。原生图编码、分批、重建和输入预算检查继续生效。扩展仅包含分析提示、stdout 对象字段投影和多字段分片回退，不修改原生实例或模块全局函数。规划拒绝回调接收字段大小、类型与固定错误码，便于保存内容无关的诊断。

stdout 投影识别带标签的对象、列表和 tuple，保留前后文字、测量值、审核正文及未知字段。指定的身份审计和存储字段留在原始 bundle 中。不能明确解析的输出原样保留。

多字段分片保留完整程序、原字段路径、类型、区间与源单元哈希。占位符明确说明值在其他片中；它不表示空结果。主机独立重建所有片并比较源哈希后，才允许规划通过。

`TrajectoryAnalyst.analyze(...)` 返回内存中的来源绑定分析。每个实际输入 case 必须有一条记录，轮次由主机绑定；分析文本仍需通过长度、标识符与样本复制检查。主机先扣除所有案例与轮次的 JSON 结构开销，再给出各文本字段的长度额度。首次投递全部批次后，对格式拒绝的批次逐个重放一次；未解决的批次仍阻止候选进入 Gate。投递和解决分别记账，服务或身份异常立即停止，不重试。`synthesis_evidence(...)` 将完整分析集合交给 `AnchoredSlotReviser`，后者由主机实施精确原文替换并执行静态检查。只有全部批次解决且产生静态有效候选，才允许进入 Gate。

`allocate_analysis_chars(...)` 保持每个 campaign 的 42000 字符总分析上限及单批 12000 字符上限，先预留每批真实案例和轮次的结构，再按实际案例记录数分配剩余正文额度。含多个案例的批次得到相应更多额度。分配结果在预检中逐批封存，并按投递证据哈希绑定；它不改变原轨迹、字段投递或 Gate 判据。

最终修订提示要求只修改有失败证据的具体子句；同一锚点中的其他步骤和前提须保留。重复对象冲突处理不能自动替代漏检补查、身份匹配或原图回退。`duplicate_suspect` 需区分重复检测、部件关系、mask 污染与真实邻接对象，实际工具拒绝仍限制几何使用。这是生成约束，候选是否遵守及是否改善仍由实际正文和完整 Gate 核对。

锚点协议 v2 提供由主机生成的原文句子目录。模型返回目录中的 `section_id`、`anchor_id` 和 `after`；主机根据当前原稿 SHA 与确切位置替换，模型无需重抄 `before`。目录排除 YAML、标题和 fenced code，重复句子各有独立位置；原有 SHA、重叠、实际改动、静态与内容复制检查继续生效。旧协议 v1 的精确子串编辑保留，新协议也可使用精确子串；不存在与重复的锚点分别返回 `edit_anchor_not_found` 和 `edit_anchor_not_unique`。拒绝稿不会替换工作稿，修复提示重新提供当前目录。本协议只解决编辑定位，不改变分析、原轨迹或准入判据。

分析与候选校验通过 `revision_guard_v11` 修正普通词 `sampled` 和 `questioned` 的词法误报。只有通用 ID 正则的全部匹配都是这两个普通词时，才去掉该通用匹配；逐字禁用 ID、其他 ID 样式及答案赋值仍拒绝。如果真实禁用 ID 本身是 `sampled`，也不会获准。静态校验使用独立函数命名空间绑定该守卫，原生 leakage 模块、原生校验函数全局状态和 admission v2 均不修改。守卫源码进入预检身份，已有封存运行继续使用原版本。

`run_reviser_successor_v11.isolated_learning_bytes(...)` 为学习 bundle 使用独立 NVMe 临时目录。原生 cold-reference 租约、索引验证、受信任成员传输及长度/SHA 检查继续生效；临时目的路径不占用原 HotCache 的预留池。每个 successor 的目录由输出路径哈希确定，进程锁限制同时只有一个成员，单成员最多 512 MiB，保留 25 GB 空间下限及内存余量检查。读取结束或传输失败即删除临时成员；崩溃遗留成员阻止新读取并要求检查，不自动删除或重试。新入口在预检和执行阶段均使用此读取方式，实际输入身份仍按原始字节封存。

## 独立运行

以下参数均需指向现场已核验的目录。`SOURCE_ROOT` 是本版本源码，`FROZEN_ROOT` 与 `STATE_ROOT` 是冻结运行和原生状态目录，`CAMPAIGN_INDEX` 是保留 campaign 索引。`TRIAL_ROOT` 与 `GATE_ROOT` 必须在冻结资产目录之外。

冻结运行时的 `src`、`scripts` 和 `ops` 需位于 `PYTHONPATH` 前部；新 checkout 仅通过显式模块绑定提供改动。项目的受控进程启动器负责建立这一顺序并清除继承的服务变量。真实运行由原生 service facade 临时加载授权服务配置。

```bash
python "$SOURCE_ROOT/scripts/run_analysis_successor_v11.py" preflight \
  --frozen-root "$FROZEN_ROOT" --state-root "$STATE_ROOT" \
  --source-root "$SOURCE_ROOT" --output "$TRIAL_ROOT" \
  --campaign-index "$CAMPAIGN_INDEX" --selection cohort \
  --reviser-request-cap "$REVISION_ALLOWANCE" --gate-request-cap "$GATE_ALLOWANCE"
```

只有 `preflight.json` 的 `status` 为 `ready`，且源码、输入、模型与剩余额度一致，才将 action 改为 `run`。预检根据实际批数 B 预留 `2B+3` 次请求；多 campaign 的总额为实际预留之和。Reviser 固定 seed 137、temperature 0、8192 输出 tokens、关闭 thinking、128000 输入字节和 72000 字符上限，transport retries 为 0。

针对已观察到的失败槽位，可用 `--slots object_abs_distance/g1m0` 从原先固定代表组中选择子集；该参数不重新挑选 campaign，也不减少该 campaign 的完整 12 条学习轨迹。子集结果按实际分母报告。

修订完成且 `revision_final.json` 含静态有效候选后，先通过原生工作区 preparer 建立独立 Gate 工作区，再执行：

```bash
python "$SOURCE_ROOT/scripts/run_successor_gate_v11.py" preflight \
  --frozen-root "$FROZEN_ROOT" --state-root "$STATE_ROOT" \
  --revision-root "$TRIAL_ROOT" --output "$GATE_ROOT" \
  --request-cap "$GATE_ALLOWANCE"
```

Gate 预检 ready 后将 action 改为 `run`，其余参数保持不变。Gate 沿用完整原生面板、seed 137/138 和 admission v2。每个 seed 均须严格涨分，终态错误不增加、合法率不下降，并通过配对、信任、交付、模型与 seed 身份检查。执行时隐藏验证答案；验证题与结果不进入 Reviser 输入。

源码、输入或准备身份变化，以及服务或请求预算异常，都会停止相应运行。已有请求 intent 和 startup 不会自动重试或续跑。后续方法版本使用新目录并计入同一总预算。

## 回执与解释

- `preflight.json`：固定选择、完整输入身份、批数、精简测量和请求额度；零模型请求。
- `analysis/analysis_final.json`：实际批次解决情况、拒绝类别和请求数；分析正文不持久化。
- `revision/final.json`：真实替换范围的哈希、静态检查和候选身份。
- `revision_final.json`：完整候选分母、实际请求与 token 用量、原资产哈希核对。
- Gate `preflight.json` 与 `final.json`：完整 episode 计划、独立配对决策、实际用量及操作异常。

静态有效不代表涨分；完整 Gate 未通过也不应删除失败候选。反复使用同一开发面板的结果属于开发验证，最终泛化需由独立冻结评估确认。
