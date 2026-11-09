# MMSIGPT

MMSI 使用 GPT 的代码入口。后续 MMSI 的方法调整、提示修改和接口修复统一在这里开发；原 `scripts/MMSIRunner` 和 `GPTexp/frozen/mmsi` 保留为来源归档。

本包已实现 MMSI 专用上下文、提示和接口兼容，修复此前 JSON 审阅定位的接口错误。通用 small-image solver、沙箱、数据 schema 和工具实现作为只读依赖使用；请求预算、服务错误处理及评分流程沿用原实现。Direct 保持原提示、原图顺序和请求参数。

| 文件 | 职责 |
| --- | --- |
| `__main__.py` | MMSI 独立准备、执行命令 |
| `inputs.py` | 原图与 prepared 身份校验、label-free Direct/Harness spec |
| `runner.py` | GPT MMSI Direct/Harness 求解入口及接口审计结果 |
| `solver.py` | 实例级绑定 MMSI 提示、程序组装器及 kernel，保留共享求解循环 |
| `runtime.py` | 只读 MMSI 上下文、实际工具暴露和执行 trace |
| `public_input.py` | 提示和 `ctx.input` 共用的公开题目合同 |
| `program.py` | 受限兼容和原程序/规范程序哈希 |
| `prompts.py` | MMSI Harness 接口提示、答案最终化及原 Direct 提示 |
| `tools.py` | MMSI allowlist、相对深度采样和点图授权 |
| `selfcheck.py` | 合成图像、合成 prepared 和 fake client 的离线回归 |
| `packaging.py` | 本包源码身份和部署文件清单 |
| `evidence.py` | 有界公开观察摘要校验；缺失或无效仅标记，不补请求 |
| `diagnostic.py` | 每题型最多 5 题的分层抽样、原图/prepared 校验和诊断清单 |
| `AGENTS.md` | 本目录的后续开发约定 |

调用链：

```text
GPTexp.worker / GPTexp.terra_worker
  → GPTexp.adapters
  → GPTexp.MMSIGPT.inputs.make_specs / runner.solve
  → MMSISolver / MMSIKernel + read-only shared runtime + GPTexp request client
```

GPTexp 的通用调用链已将 `dataset=mmsi` 转发到本包，MMSI 不再导入旧 pilot。其他数据集继续走既有路径。新包的全部 Python 文件纳入 `run.code_identity()` 和 MMSI 预检回执的源码身份，常规 `GPTexp.deploy` 的完整部署与代码部署都会携带本包。

## 使用

在 `wjhICLR` 根目录运行：

```bash
python3 -B -m GPTexp.MMSIGPT --help
python3 -B -m GPTexp.MMSIGPT prepare \
  --run GPTexp/runs/mmsi-gpt-new-run \
  --config /absolute/path/to/mmsi-gpt-config.json
python3 -B -m GPTexp.MMSIGPT execute \
  --run GPTexp/runs/mmsi-gpt-new-run
```

`prepare` 只做零请求预检，配置采用 GPTexp 的既有 schema，但题单和预算必须对应 MMSI。当前完整归档题单为 365 题，两臂请求上限为 `365 × 5 = 1825`。模型、API、密钥来源和 worker 绑定仍由显式配置管理，源码中不存凭据。`execute` 只接收 MMSI 单数据集清单，并沿用清单、源码哈希、API 准入及请求预算校验。

后续实验需要新的运行清单及预检回执，本包的源码变动会改变运行身份。现有运行清单、冻结实验和历史专用 ops 的哈希不作修改。

## 已修复的接口

- `ctx.input` 提供 `benchmark/task_type/question/options/views`；`ctx.question` 提供题目字符串；`ctx.options` 是保持原标签顺序的 `({label, text}, ...)`。映射、嵌套内容、上下文和图像均只读，不包含路径、参考答案或 prepared 几何。
- 无别名且成员仅为 `ReturnAnswer/AnswerPayload` 的 `from harness3d import ...` 转为使用已有注入符号；`getattr(ctx, '公开固定字段')` 和默认值为 `None` 的三参数形式转为固定属性读取。其余导入、反射及非空默认值仍经过原 AST 拒绝检查。规范化保留源码行数并重新计算程序身份。
- `basis='visual_estimate:解释'` 仅规范为 `visual_estimate` 并记录适配事件；其他非法 basis 继续拒绝。新提示明确解释放在注释中，并明确注入控制符号、数值库白名单及禁用接口。
- `tools` 命名空间及顶层工具名只暴露当题授权目录；`ReadObservation` 不暴露。禁用工具后目录为空，跨轮重置仍保留完整工具账本。
- `point_depth.n_valid` 统计有效深度样本，独立于可选点图。错形或不足的点图不阻断有效深度；`pair_point_distance` 在无有效点图时不列入目录，在缺失点图的 view 上被授权门拒绝。局部点样本仍须通过原质量门。
- Harness 最终化提示只要求 `{"answer":"原选项字母"}`。工具失败仍使本题失败，服务错误仍立即停止；没有添加自动重试或视觉补答请求。

接口、提示和工具版本分别为 `mmsigpt-interface/3`、`mmsigpt-prompt/4`、`mmsigpt-tools/2`。结果增加 `interface_version`、`interface_adaptations` 和 `interface_adaptation_count`；适配记录保留原/规范程序 SHA256 及适配类型，basis 适配不额外保存解释文本。工具请求摘要、证据版本及授权决策版本包含新的工具身份。核心结果回执哈希按更新后的版本重新计算。

诊断版要求模型在程序顶层声明简短 `evidence_audit`，记录原图观察、题目参照系、空间结论、选项和工具用途说明。执行轨迹将其标记为 `provided/missing/invalid`；语义事实仍需人工核对，摘要不替代工具授权或测量结果。缺失、不一致或超出长度上限不会改变提交答案或发起额外请求。Direct 保留原协议。

提示 v4 根据分层诊断的公开轨迹补充空间绑定要求：区分相机移动与转向，绑定共享地标实例及物体正反面，明确题目原点和朝向来源，将位移转换到题目要求的坐标轴后再给出方位。六类题型各有相应指导；镜中区域、上楼方向和罗盘的地面轴需要明确绑定。程序接口和摘要 schema 沿用 v3，Direct 提示沿用原协议。v3 的服务器运行和回执保持原始身份；v4 的行为与准确率尚需独立线上验证。

```bash
python3 -B -m GPTexp.MMSIGPT prepare-diagnostic \
  --run /absolute/path/to/diagnostic-first6 \
  --source-run /absolute/path/to/stopped-mmsi-source \
  --per-type 5 --selection-seed 137 --stage first
```

`--stage remaining` 准备相同分层抽样中的其余 24 题。两批选择身份一致且不重叠，总请求上限为 150。抽样排除来源实验所有已派发题（含结果未知请求），保留真实 `source_split`，不读取标签。诊断执行结束只导出公开轨迹，不自动评分。

`MMSISolver` 使用共享 `solve` 函数的代码对象和独立 globals 字典绑定三个 MMSI 依赖，不改写共享模块全局变量。共享依赖升级后需复核调用接口并重新运行回归检查。

## 离线验证

```bash
python3 -B -m GPTexp.MMSIGPT.selfcheck
```

检查进程清空继承环境、禁用网络连接，只使用临时合成图像、哈希验证的合成 prepared 和 fake client。回归覆盖对应此前 11 次接口错误的合成程序、未授权导入/反射拒绝、只读上下文、工具暴露与账本、深度计数、点图缺失、最终化、服务错误零自动重试、Direct 隔离、prepared 内容篡改拒绝、公开摘要和分层诊断合同。

2026-11-06，v3 在本地及服务器通过 17 项离线回归；v3 分层诊断的公开轨迹用于提示 v4 的开发。离线检查不读取参考答案，不执行真实模型请求；线上准确率变化需用新运行清单另行评估。

## 联合相机位姿诊断

`python -B -m GPTexp.MMSIGPT.vggt_pose_trial` 是独立的原图／原图＋VGGT 相机位姿入口。每题两张原图联合运行 VGGT 聚合主干和相机头；不运行深度、点图或跟踪头。模型收到原始 RGB 和相机坐标系下的双向相对位置、旋转及朝向文字，位置使用共同归一化尺度。六题十二次请求在同一轮成对比较，零工具调用，无自动重试；先冻结全部预测，再由独立评分进程读取参考答案。输入、模型权重、源码、位姿文件和实际请求均记录哈希，已执行目录拒绝自动重启。

阶段为 `prepare-gpu`、`prepare-online`、`controlled-execute`、`freeze`、`score`；GPU 任务使用独立目录和已验证的本地权重，离线估计完成后回收位姿才能进行在线预检。`vggt_pose_selfcheck` 通过 fake client 检查请求预算、失败停止和冻结篡改；`vggt_pose_report` 汇总真实配对结果、服务实际回报参数和凭据扫描。

## 按题型绑定的相机位姿提示

`vggt_task_prompt.py` 实现六类题型的原点、前向和目标绑定，版本为 `mmsi-vggt-task-frame/1`。路由仅使用公开 `task_type/question`，不依赖样本编号、选项或参考答案。相机位置题可直接使用目标相机中心；其他题型先从 RGB 定位目标，再用位姿对齐视图并转换到问题参照系。相机中心、拍摄朝向、目标方向分别命名。明确指定“最后一帧”时优先提供该相机下的变换；身体朝向、区域前向和东向锚点取自题目。未明确解析的参照系留给原题和 RGB 绑定，不假设图 1 或房间前向。

`vggt_task_trial.py` 是新版本独立入口，准备阶段复用已冻结父实验的六题及 VGGT 位姿，验证原图、位姿、父请求和冻结身份，再创建新清单与源码身份。Direct payload 与父实验相同，位姿臂增加题型适配文字；两臂保留原始题目、选项和单字母输出协议。请求调度及冻结函数在独立 globals 中绑定，不修改父入口。准备阶段不读取凭据或参考答案，不重新运行 VGGT，也不发模型请求。

```bash
python3 -B -m GPTexp.MMSIGPT.vggt_task_selfcheck
python3 -B -m GPTexp.MMSIGPT.vggt_task_trial prepare \
  --parent /absolute/path/to/frozen-joint-pose-run \
  --directory /absolute/path/to/new-task-bound-run
python3 -B -m GPTexp.MMSIGPT.vggt_task_trial validate \
  --directory /absolute/path/to/new-task-bound-run
```

准备目录包含 `提示预览.md`、`manifest.json` 和零请求 `preflight.json`。实际执行另用 `controlled-execute`，完成后 `freeze`、`score`；仍为六题十二次请求、最多六个并发、服务失败停止、零自动重试。`vggt_task_selfcheck` 使用临时合成图像和题目、禁用网络、清空继承凭据，验证六类绑定、Direct 隔离、fake-client 完整闭环、内容篡改拒绝和已有入口的 globals 隔离。提示适配的准确率效果需要独立线上结果。

### 精简提示 v2

新入口为 `vggt_task_trial_v2`，提示版本为 `mmsi-vggt-task-frame/2`。保留坐标、尺度、字段含义和简短题目绑定，删除长段地标、镜面和实体匹配指导。相机位置题只展示相关相机中心；其他题展示一组相机中心和旋转，在题目明确问行动时附拍摄前向及简短行动说明。参考相机明确时使用该相机坐标系，语义参照系题使用图 1 相机作为位姿证据的共同坐标系，最终方向仍按题目指定的身体或场景轴判断。

v2 明确旋转只映射相机坐标向量，目标位置需要从 RGB 定位，相机坐标轴不自动充当身体或房间坐标轴。导航说明由公开问题中的行动要求触发，不由 Cam–Reg 类别单独触发；普通区域方位问句也可提取目标。v1 提示、入口和清单继续保留原身份。v2 复用原有准备和调度函数，使用独立 globals 绑定新提示与源码身份。

```bash
python3 -B -m GPTexp.MMSIGPT.vggt_task_selfcheck_v2
python3 -B -m GPTexp.MMSIGPT.vggt_task_trial_v2 prepare \
  --parent /absolute/path/to/frozen-joint-pose-run \
  --directory /absolute/path/to/new-compact-pose-run
python3 -B -m GPTexp.MMSIGPT.vggt_task_trial_v2 validate \
  --directory /absolute/path/to/new-compact-pose-run
```

两臂继续使用同一官方题目包装和字母输出协议。官方 Direct 与 v2 位姿臂比较的是“位姿＋简短绑定”的整体效果；单独评估位姿贡献需要另设只有同样绑定、没有位姿的对照。本次提示修改不增加线上请求或实验臂。

## 六类各四十题原图直答

`direct_batch_host` 为已有拍摄位置关系 240 题补齐原图直答。固定六类各40题，保持原始图片字节、顺序、问题与选项，使用官方字母回答指令。十个不同 API 槽各执行24题，最多十个请求同时在途。每个题目只派发一次；服务失败保存，随后继续该槽中尚未派发的题目。已有派发目录拒绝重新执行。

```bash
python3 -B -m GPTexp.MMSIGPT.direct_batch_host prepare \
  --directory /absolute/path/to/new-direct240-run
python3 -B -m GPTexp.MMSIGPT.direct_batch_host run \
  --directory /absolute/path/to/new-direct240-run
```

准备阶段执行 `GET /models`、逐题原图哈希核验及源码冻结，模型请求数为零。运行阶段通过标准输入向清空继承凭据环境的子进程传递十个密钥，在线子进程从 `source_frozen` 执行，只读取公开题目和原图。240个终态预测与480份派发／结果文件冻结后，独立宿主进程核验既有240题评分身份，再读取参考答案并输出同题比较。

请求设置沿用原实验：gpt-5.6-terra、high、temperature=0、seed=137、max_output_tokens=8192、timeout=180秒。`实验结果.md` 同时报告固定240题准确率、合法回答准确率、两臂均合法子集、六类明细和实际服务回报参数。两臂为不同轮次，结果保留轮次和服务输出波动的影响。正式完成以 `completed.json` 绑定的冻结、评分身份和完整请求计数为准。
