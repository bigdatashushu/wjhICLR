# Qwen3.8-Flash 官方导航基线试跑

2026-11-05 已完成 EB-Navigation `base` 子集前 3 个官方任务。模型只使用官方输入协议，全部成功；这是 3 题试跑结果。

| 任务 | 结果 | 环境动作数 | 模型请求数 | 无效环境动作 |
|---|---|---:|---:|---:|
| base:000 | 成功 | 11 | 5 | 3 |
| base:001 | 成功 | 18 | 7 | 0 |
| base:002 | 成功 | 10 | 4 | 1 |
| 合计 | **3/3** | **39** | **16** | **4** |

三题均由官方环境判定成功，判定条件为与目标的水平距离不超过 1 米，每题最多 20 个环境动作。无效环境动作会按官方循环重新观察、规划；本次没有 JSON/规划错误、模型服务错误或自动重试。三题执行耗时合计约 144 秒，不含环境安装和构建下载。实际用量为输入 54,313 tokens、输出 4,679 tokens，合计 58,992。

## 输入与参数

- 官方 revision：`9be4e980e9cd6bcb38373cd4aab7c32724bdd401`。
- 模型：`qwen3.8-flash`；temperature=0、max_tokens=4096、seed=137、enable_thinking=false。
- 保留官方 Strategy 和 3 个文本示例；首轮建议 1–2 个动作，后续轮次建议 5–6 个动作。
- 每轮一个 user 消息，先放当前 500×500 第一视角 RGB，再放官方提示文字；后续文字包含已执行动作的官方反馈。
- 没有自有 Skill、几何工具、深度输入、检测框、额外视角、旧图片或 Skill 演化。模型不接收目标坐标、评分距离或成功标记。
- 每题最多 20 次模型请求，全 run 上限 60；本次使用 16 次。

4096 输出 tokens 来自固定 revision 的官方代码常量。模型请求在 Mac 运行，AI2-THOR 5.0.0 在 4090 服务器的独立 embodiedRunner 环境运行，使用物理 GPU 3。

## 与官方完整入口的差异

官方提示构造和消息构造方法直接复用，并与完整官方类逐字节比对通过。模型 API 使用 JSON-object 模式替代 SDK 的 parsed JSON schema，明确指定 seed 和 thinking；格式错误不使用官方随机动作兜底，服务故障不无限重试。这些差异已写入启动和结果收据。本次没有触发这些错误路径。

## 结果文件与复现

- [真实结果汇总](runs/qwen-bare-official-3shot-pilot/summary.json)
- [14 项真实运行审计](runs/qwen-bare-official-3shot-pilot/audit.json)
- [逐次请求与实际动作记录](runs/qwen-bare-official-3shot-pilot/trace.jsonl)
- [实际配置](runs/qwen-bare-official-3shot-pilot/config.json) · [启动及源码身份](runs/qwen-bare-official-3shot-pilot/startup.json)
- [真实渲染预检](checks/render-smoke-v3/receipt.json) · [完整构建 hash 校验](checks/unity-prefetch.json)

入口为 [bare_run.py](bare_run.py)，配置为 [bare_navigation.json](bare_navigation.json)，部署细节见 [USAGE.md](USAGE.md)。

```bash
../.venv/bin/python -B ssh_session.py
../.venv/bin/python -B bare_run.py run --output-dir runs/qwen-bare-repeat-001
../.venv/bin/python -B audit_bare_run.py runs/qwen-bare-repeat-001
```

工程协议检查 26 项通过；真实运行审计 14 项通过，覆盖源码身份、完整分母、成功率、请求和动作数、预算、单图输入、Skill/工具关闭及不自动重试。试跑结束后仿真 worker 已退出。
