# Manipulation 独立最小改动试验

从 `official_direct/deployment-snapshots/r03` 的已核验源码复制，五题固定为
`manipulation/base:000–004`。旧基线、官方源码、资产和服务器共享环境保留。

只测试一个组合改动：每轮执行一个 7D 动作后重新观察；增加短抓取核验指引，
并让模型看到前一次决策的公开图像、匿名坐标、实际动作和当前公开观测。
抓取判断仍由模型根据图像作出，宿主没有新增抓取传感器或隐藏状态。
匿名编号每帧独立，不能作为持续物体 ID。

保持 qwen3.8-flash、seed=137、temperature=0、max_tokens=4096、
enable_thinking=false、官方四条文本示例和每题 15 步上限。
API 并发上限 32；仿真并发 3；每题最多 15 次模型请求。
服务失败不自动重试。官方分数仅在执行侧和评测记录中使用。

本地入口（从 embodiedRunner 目录）：

```bash
../.venv/bin/python -B -m manipulation_minimal check \
  --manipulation-catalog official_direct/assets/manipulation_assets.json
../.venv/bin/python -B -m manipulation_minimal probe \
  --manipulation-catalog official_direct/assets/manipulation_assets.json
../.venv/bin/python -B -m manipulation_minimal.launch_baseline \
  --config manipulation_minimal/experiment-20261107-r01.json \
  --run-id single-step-visual-five-20261107-r01
```

启动器在内存中加载原先授权的凭据；凭据不进入配置、请求文件或仿真进程。
运行目录、模型响应、每步图像、动作回执及失败保留在本包 `runs/`。
该五题用于诊断，不能代表完整 benchmark 成绩；组合试验不能拆分证明
“单步执行”与“视觉核验”各自的效应。

实际五题已完成并核验：成功从 1/5 变为 2/5，新增成功为 base:004；
模型请求从 14 次变为 56 次。五题初始图像、匿名坐标、题目身份逐题完全一致。
剩余失败仍有目标形状、容器语义和坐标绑定不稳定。
见 [完整对照与轨迹判断](comparison-20261107.md) 和
[终态图像对照](verification/final-comparison.png)。
