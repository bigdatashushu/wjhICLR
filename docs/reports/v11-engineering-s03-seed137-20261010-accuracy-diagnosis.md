# v11-engineering-s03-seed137-20261010 正确率诊断

## 1. 结论摘要

本轮正确率偏低，不宜归结为单一的“模型能力不足”。现有轨迹显示，低正确率是场景证据质量、对象识别与 grounding、工具/程序执行失败，以及模型在证据不足时退回视觉猜测等因素共同造成的。

- Induction：12 题中只有 QA 1943 正确，逐题正确率为 1/12。QA 1944、1947 是运行错误；其余 10 题有预测，其中 9 题错误。
- B01/B11：计划 8 题，6 题完整可比较，2 题因 detector 准备失败而不完整。两个 arm 在 6 个完整配对上的 Accuracy 都是 0.5，配对差值为 0。
- 轨迹显示，模型多次在缺少可靠 3D 坐标、候选类别缺失、工具超时或程序失败后，仍然基于图像直觉输出选项。
- 因本轮最终状态为 `incomplete`，且 `formal_result_eligible=false`，这里的准确率只适合定位工程和推理链问题，不能作为正式论文结果。

最重要的系统性问题是：证据不足或质量门失败的状态没有稳定地阻止正式答案进入评分。于是“无法可靠判断”被转换成了一个看似正常的选项答案。

## 2. 逐题结果

### 2.1 Induction

| QA | 预测 | 正确答案 | 结果 | 轨迹中的状态或线索 |
|---|---|---|---|---|
| 1942 | A | C | 错 | 有程序响应；程序先通过图像描述推断 heater 对象身份 |
| 1943 | A | A | 对 | 使用对象列表和 `relative_distance_rank`；但部分候选类别未检测到 |
| 1944 | 无 | D | 运行错误 | `answer_source=abstain`，`failure_code=unknown` |
| 1945 | B | A | 错 | 程序称 table 最近，同时说明其他候选没有检测到 |
| 1946 | A | B | 错 | 候选距离返回相同数值，未触发平局或不确定处理 |
| 1947 | 无 | C | 运行错误 | `answer_source=abstain`，`failure_code=unknown` |
| 1948 | B | D | 错 | 直接以 `visual_estimate` 作答 |
| 1949 | B | A | 错 | 根据图像估计 heater 与 bookshelf 的相对位置 |
| 1957 | D | B | 错 | 程序明确称没有 3D 坐标/工具，只能视觉推断；`main_gate_passed=false` |
| 1958 | D | A | 错 | 依据“电话通常在桌上”等常识推断；`main_gate_passed=false` |
| 1959 | B | C | 错 | 把地图误认成 bookshelf，并声明依赖视觉估计；`main_gate_passed=false` |
| 1960 | A | D | 错 | 程序称工具调用超时，改用视觉判断；`main_gate_passed=false` |

QA 1944 和 1947 的预测为空，因此它们与普通的错误选项不同，应单独计为执行失败。其余有预测的 10 题中只有 1943 正确。

### 2.2 B01/B11 完整配对

| QA | B01 | B11 | 配对结果 |
|---|---:|---:|---|
| 1905 | 0 | 1 | B11 正确 |
| 1907 | 0 | 0 | 都错 |
| 1909 | 1 | 0 | B01 正确 |
| 1910 | 0 | 1 | B11 正确 |
| 1911 | 1 | 1 | 都对 |
| 1912 | 1 | 0 | B01 正确 |

B01 和 B11 各答对 3/6，Accuracy 均为 0.5；配对 Accuracy 差值为 0.0。QA 1906 和 1908 两个 arm 都因 `detector returned success=false` 未能完成对象准备，不能当作模型答错，也不能作为 0 分并入准确率。

## 3. 主要原因分析

### 3.1 场景几何质量不足，且质量门没有稳定阻断答题

重建报告记录场景 `5eb31827b7` 的 `warp_inlier_ratio=0.4074`，低于要求的 0.5；虽然 `cloud_overlap_ratio=0.4457` 高于 0.3，主质量门仍失败。

Induction 输入日志将 QA 1957–1960 关联到场景 `5eb31827b7`。对应的 `episode_trace.jsonl` 也记录这四题 `main_gate_passed=false`，但四题仍生成了答案，而且四题全错。这说明质量门失败没有阻止答案进入评测流程。

这些题问的是基于物体最近点的精确距离关系。几何重建不可靠时，视觉上的远近或常识推断不能代替 3D 距离测量。因此这四题的错误不能简单解释成模型在可靠几何证据下做错了比较；它们反映了系统在不合格场景中仍继续答题的问题。

QA 1942–1949 位于另一个场景 `fb5a96b1a2`；轨迹中的 `main_gate_passed` 均为 true。不能把这八题的错误归因到上述失败场景的质量门问题。

### 3.2 对象类别和 grounding 不稳定

工具轨迹中能看到多个类别边界和对象实例问题：

- `bookshelf`、`shelf`、`book` 等类别混杂。QA 1943 的 bookshelf 过滤结果同时包含 `book`、`shelf` 和 `bookshelf` 对象。
- 多个对象带有 `duplicate_suspect=true`，同一类别可能对应重复或歧义实例。
- 某些题目的候选类别未检测到，只能使用 `question_targeted_fill` 补充对象，或完全缺失。
- 缺少候选对象时，系统仍可能继续比较并给出选项，而没有要求重试或拒答。

QA 1943 是唯一答对的 Induction 题。其工具轨迹调用 `list_objects` 和 `relative_distance_rank`，但工具返回的候选中缺少 `sofa`、`whiteboard`、`computer mouse`。因此它的正确结果不能证明所有候选都得到了可靠比较，只能说明该次流程最后选对了答案。

QA 1946 的 `relative_distance_rank` 返回四个候选相同的距离值 `0.049392...`，模型仍输出 A，而正确答案是 B。轨迹中没有证据表明系统处理了距离并列或数值不可区分的情况。

### 3.3 模型在证据不足时退回视觉估计和常识猜测

多份 `episode_program.jsonl` 记录了模型直接声明依赖视觉估计的响应：

- QA 1948 以 `basis="visual_estimate"` 返回答案。
- QA 1949 根据 heater 看起来靠近 bookshelf 的图像描述作答。
- QA 1957 表示无法调用工具或获取 3D 坐标，只能视觉推理。
- QA 1958 根据“电话可能位于办公桌附近”推断 laptop 最近。
- QA 1959 明确说没有 3D 坐标或精确距离工具，转为视觉估计。
- QA 1960 称工具调用超时后，只能基于视觉观察判断。

这类回答使用的是“电话通常在桌上”“鼠标和笔记本通常相邻”等物体共现常识，而题目实际要求比较最近点 3D 距离。描述可以连贯，但不代表比较依据与题目定义相符。

### 3.4 模型先验对象识别可能造成 grounding 错误

有些程序在执行工具查询前，已经在注释中假设对象身份。例如 QA 1942 先将 heater 描述为“靠近 whiteboard 和 telephone stand 的白色圆柱物”。如果这个视觉对象识别错误，后续距离比较就可能基于错误的参考对象。

QA 1959 更明显：生成程序把 bookshelf 解释成墙上挂着的地图，并据此描述与候选物体的位置关系。这是对象语义漂移，不是单纯的距离计算误差。

### 3.5 Induction 中存在执行和解析失败

QA 1944、1947 没有正常预测，轨迹记录为 `run_error`、`answer_source=abstain`、`failure_code=unknown`。它们属于程序生成或运行路径失败，不应与“模型给出错误选项”混为一谈。

B01/B11 轨迹中也存在 `vllm_parse_error`、`violation_runtime` 和 `geometry_rejected`。例如：

- B01 QA 1905 为 `run_error`，没有有效答案；B11 QA 1905 答对。
- B01 QA 1907 的轨迹包含 `geometry_rejected`，B01 和 B11 最终都答错。
- B01 QA 1910 为 `run_error`，B11 答对。
- 多题的轨迹保留了模型生成的程序、错误反馈和后续轮次，表明部分答案是在多轮失败或修正后产生。

因此，B01/B11 的 0.5 也混合了模型判断和执行路径差异，不应直接理解为纯粹的模型能力差异。

### 3.6 detector 准备失败使阶段不完整

QA 1906、1908 的 B01 和 B11 结果均记录：

```text
service_errors: detector returned success=false
model_request_count: 0
tool_call_count: 0
```

模型没有收到正常的对象准备结果。这两题不是模型答错，而是服务/准备阶段失败；它们也不能简单从分母中删除，因此 B01/B11 阶段状态应保持 `incomplete`。

## 4. 质量与完整性边界

上一轮实验报告给出的重建汇总是 7/7 场景生成 artifact、没有重建进程失败，但只有 6 个场景通过主质量门，`5eb31827b7` 未通过。这里需要区分“重建产物已生成”和“该场景几何质量达到可用于正式测评的门槛”。前者不能替代后者。

此外，两个 B01/B11 题的 detector 准备失败，导致配对结果不完整。整个 run 的最终状态为 `incomplete`，`formal_result_eligible=false`。因此：

- Induction 的 1/12 可以用于描述本次链路实际产出的逐题结果，但不是正式论文性能估计。
- B01/B11 的 0.5 仅基于 6 个完整配对，不能代表计划的 8 题完整阶段结果。
- 服务失败不能记为模型答错，也不能因为缺失而无说明地从预注册分母删除。
- 质量门失败场景的答案不应与合格场景答案混为同一质量口径。

## 5. 改进优先级

1. **让质量门成为答题硬门禁。** `main_gate_passed=false` 时停止正式评分，或将结果明确标记为 exploratory/ineligible；不要继续生成可被误读为正式答案的分数。
2. **候选对象不完整时重试或拒答。** 对候选类别缺失、detector 失败、grounding 有歧义的情况，不能照常输出最近对象选项。
3. **限制视觉估计进入正式几何答案。** 对这类精确距离任务，正式答案至少应有可靠参考对象 ID、候选对象 ID 和可验证距离证据；`visual_estimate` 应单独标记为低可信或不合格。
4. **修复类别归一化和实例消歧。** 明确 `book`、`shelf`、`bookshelf` 的映射边界，并处理重复对象与 `duplicate_suspect`。
5. **增加距离并列和误差界处理。** 候选距离相同或差异低于几何不确定范围时，返回 `uncertain_geometry`，而不是强行选择一个选项。
6. **区分服务失败、程序失败、证据失败和模型答错。** 分别统计 `detector returned success=false`、`vllm_parse_error`、`violation_runtime`、`geometry_rejected` 和普通错误答案，保持预注册分母与阶段完整性。
7. **给模型提供规范化候选表。** 在生成决策程序前，向模型提供对象 ID、规范类别、可见帧、置信度、重复嫌疑、几何可用性及距离结果，减少模型从自然语言图像描述自行猜测对象。

## 6. 总结

这轮结果说明，在当前重建质量、对象 grounding、工具恢复和回答资格控制下，Induction 只产生了 1 个可验证正确答案；B01/B11 在完整配对题上各有 3/6 正确。但不能据此得出“模型真实空间推理能力只有 1/12”或“正式 Accuracy 为 0.5”的结论。

更准确的解释是：当前系统在不可靠几何、候选对象不全、工具失败或缺少 3D 距离证据时，仍会退回视觉猜测并输出正式选项；同时执行/准备失败与模型答错交织在同一阶段结果中。该实验因此主要暴露出证据链、质量门和完整性协议的问题，需要先修复这些环节，再运行可用于正式比较的实验。

## 7. 主要日志位置

实验收据根目录：

```text
/home/cvailab/experiments/wjhICLR-v11-engineering-20261010/runs/v11-engineering-s03-seed137-20261010
```

分析使用的主要记录：

- `parent_learning_trace/episode_input.jsonl`：Induction 输入、题目、选项、正确答案、场景 ID。
- `parent_learning_trace/episode_program.jsonl`：Induction 模型响应文本和生成程序。
- `parent_learning_trace/episode_trace.jsonl`：逐题状态、答案来源、轮次和质量门信息。
- `parent_learning_trace/program_trace.jsonl`：Induction 程序及工具调用执行记录。
- `parent_learning_trace/trace_record.jsonl`：工具/evidence 调用记录。
- `parent_learning_trace/evaluation_result.jsonl`：Induction 逐题预测和正确性。
- `b01_b11/paired_results.jsonl`：B01/B11 配对状态、逐题分数和可比较性。
- `b01_b11/arms/B01/*/result.json`、`b01_b11/arms/B11/*/result.json`：各 arm 的答案、执行错误和服务错误。
- `parent_learning_manifest.json`、`b01_b11/summary.json`：阶段配置、汇总和分母信息。
- `failure.json`：最终启动器/阶段失败记录。
