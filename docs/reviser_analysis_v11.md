# 完整学习轨迹分析与锚点修订

入口为 `scripts/run_analysis_successor_v11.py`。它在独立目录内读取冻结学习轨迹，先逐批分析，再以完整分析集合生成一次锚点修订。配对评估由 `scripts/run_successor_gate_v11.py` 单独执行。运行目录、原 Skill 库与原请求账本分别绑定，不向原库发布候选。

## 可复用接口

`trajectory_analysis_v11.plan_analysis_evidence(native, analyst=..., parent=..., feedback=..., experience=..., on_fragment_refusal=...)` 接收已配置的原生证据规划器，返回 `(batches, plan)`。

接口复制规划器实例，并为复制实例建立独立函数命名空间。原生图编码、分批、重建和输入预算检查继续生效。扩展仅包含分析提示、stdout 对象字段投影和多字段分片回退，不修改原生实例或模块全局函数。规划拒绝回调接收字段大小、类型与固定错误码，便于保存内容无关的诊断。

stdout 投影识别带标签的对象、列表和 tuple，保留前后文字、测量值、审核正文及未知字段。指定的身份审计和存储字段留在原始 bundle 中。不能明确解析的输出原样保留。

多字段分片保留完整程序、原字段路径、类型、区间与源单元哈希。占位符明确说明值在其他片中；它不表示空结果。主机独立重建所有片并比较源哈希后，才允许规划通过。

`TrajectoryAnalyst.analyze(...)` 返回内存中的来源绑定分析。每个实际输入 case 必须有一条记录，轮次由主机绑定；分析文本仍需通过长度、标识符与样本复制检查。`synthesis_evidence(...)` 将完整分析集合交给 `AnchoredSlotReviser`，后者由主机实施精确原文替换并执行静态检查。只有全部批次解决且产生静态有效候选，才允许进入 Gate。

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
