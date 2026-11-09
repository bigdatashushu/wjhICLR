# Navigation 同步 RGB-D 试验

模型每次读取当前 RGB 时，同时读取同一仿真事件的 500×500 灰度深度图。
读取 AI2-THOR 原生 `last_event.depth_frame`，无需深度估计模型，也不增加仿真动作。
RGB 必须与该事件的 `frame` 逐像素一致；深度必须同尺寸、以米为单位。
不满足条件则停止，不使用旧深度或缩放补齐。

深度显示使用固定 0–10 米反向线性灰度：越亮越近，10 米以外显示黑色，
无效像素也显示黑色。黑色不能单独证明可通行。
模型看到 RGB、灰度深度图及这段解释；原始 float32 米制深度只保留作核验。
原始数值保存为每步 `observation-NNN-depth-meters.npy`，不截断到 10 米。

仅测试 Navigation 官方 base 索引 0–4。使用 qwen3.8-flash，
seed=137、temperature=0、max_tokens=4096、enable_thinking=false、JSON-object、
官方三条示例、原有完整动作计划与 20 步上限。
该版本没有加入 Manipulation 单步执行策略。

代码从核验后的 `official_direct/deployment-snapshots/r03` 复制，
使用新部署、新运行目录；旧基线、Manipulation 试验、官方源码及共享环境保留。

从 embodiedRunner 目录执行：

```bash
../.venv/bin/python -B -m rgbd_trial check \
  --manipulation-catalog official_direct/assets/manipulation_assets.json
../.venv/bin/python -B -m rgbd_trial probe \
  --manipulation-catalog official_direct/assets/manipulation_assets.json
../.venv/bin/python -B -m rgbd_trial capture \
  --manipulation-catalog official_direct/assets/manipulation_assets.json \
  --run-id nav-rgbd-capture-20261107-r01
../.venv/bin/python -B -m rgbd_trial.launch_baseline \
  --config rgbd_trial/experiment-20261107-r01.json \
  --run-id nav-rgbd-five-20261107-r01
```

`check/probe` 不启动仿真或模型；`capture` 只启动一题并抓取首帧，0 次模型请求。
正式运行不自动重试服务失败或未知结果的动作。凭据在原授权启动器中瞬态加载，
不进入模型输入、文件和 worker。参考答案和目标的隐藏几何状态不进入模型。

首次真实测试已完成：Navigation 5 题成功 4/5，与已有 RGB 基线相同。
26 次请求均输入两张同尺寸图，72 份同步观测的图像与数值深度通过核验。
原始响应中有 11 次显式提到深度图。五题首帧与旧基线的字节哈希不同，
因此成绩仅作本次诊断记录。
见 [测试报告](rgbd-test-report-20261107.md) 与
[RGB/深度首帧预览](rgb-depth-preview.png)。
