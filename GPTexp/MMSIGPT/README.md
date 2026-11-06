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

接口、提示和工具版本分别为 `mmsigpt-interface/2`、`mmsigpt-prompt/2`、`mmsigpt-tools/2`。结果增加 `interface_version`、`interface_adaptations` 和 `interface_adaptation_count`；适配记录保留原/规范程序 SHA256 及适配类型，basis 适配不额外保存解释文本。工具请求摘要、证据版本及授权决策版本包含新的工具身份。核心结果回执哈希按更新后的版本重新计算。

`MMSISolver` 使用共享 `solve` 函数的代码对象和独立 globals 字典绑定三个 MMSI 依赖，不改写共享模块全局变量。共享依赖升级后需复核调用接口并重新运行回归检查。

## 离线验证

```bash
python3 -B -m GPTexp.MMSIGPT.selfcheck
```

检查进程清空继承环境、禁用网络连接，只使用临时合成图像、哈希验证的合成 prepared 和 fake client。13 项回归覆盖对应此前 11 次接口错误的合成程序、未授权导入/反射拒绝、只读上下文、工具暴露与账本、深度计数、点图缺失、最终化、服务错误零自动重试、Direct 隔离和 prepared 内容篡改拒绝。

2026-11-05 本地离线回归通过。未部署、读取真实题单或执行真实模型请求；线上准确率变化需用新运行清单另行评估。
