# MMSIRunner

这里集中保存 MMSI 365 题双组测试实际使用过的代码。目录保留原始源码字节和恢复链，不包含预测结果、请求账本、参考答案、凭据或私有环境变量。

## 目录

- `runtime/`：Titan 正式运行目录中的 Python 源码原件。根目录负责输入、推理、评分和状态核验；`runtime/ops/` 保存恢复 2 至 8 的控制、超时处理和离线回归测试。
- `control/`：最终 `resume08` 启动、独立验收监控和实时打印入口。
- `services/`：4090 审计推理网关与只读超时回执服务源码。
- `retry/`：用户授权后，仅对 Skill 组 12 个服务失败样本各执行一次新 solver attempt，并重新统计的入口。
- `source_manifest.json`：每个收录文件的来源、字节数和 SHA-256。
- `verify_sources.py`：只读检查当前文件是否仍与清单一致。

通用的小图数据加载、预计算和单题 solver 仍由上级 [`../run_small_image.py`](../run_small_image.py) 及 `src/skill3d/small_image/` 提供，这里不复制其实现。

## 最终运行链

1. `runtime/preflight.py` 和 `runtime/rerun_preflight.py` 冻结 365 道作者 test 题、两组共 730 个 episode、原图顺序、六个 Skill 槽位、prepared cache 和预算身份。
2. `runtime/entry.py` 执行初始双组测试；遇到服务错误时停止派发并保留已完成结果。
3. `runtime/ops/resume08_control.py` 对前七次恢复留下的结果、请求、超时证据和账本前缀做零请求预检。
4. `runtime/ops/resume08_entry.py` 使用 4 个 worker 继续从未派发题目，并将经上游回执确认的单题超时记为零分后继续其他题。
5. `runtime/ops/resume08_monitor.py` 在 runner 退出后独立核验 730 个结果、两组各 365 题、冻结预测、配对身份和完整分母。
6. `retry/skill_service_retry.py` 是后续独立实验，只替换 Skill 组原有服务失败行；Direct 组和未选中行保持原字节。

## 使用边界

这些文件是可审计的现场源码归档，不是脱离任务目录即可执行的通用 CLI。运行时仍需要对应的 `manifest.json`、冻结 Skill、prepared cache、账本前缀、服务身份和任务内认证入口。源码中的绝对路径、运行身份和固定哈希属于实验协议。

不要直接在已有正式目录重跑入口，也不要修改 `runtime/` 后继续声称复现原实验。新实验应复制到新的任务根，重新生成 manifest、预算和源码清单，并先执行零模型请求预检。服务错误不得自动重试；只有明确授权的独立 retry 运行可以新增请求。

## 校验

```bash
python3 scripts/MMSIRunner/verify_sources.py
python3 -m compileall -q scripts/MMSIRunner
```

`verify_sources.py` 只读取本目录文件，不连接服务器，也不发送模型请求。
