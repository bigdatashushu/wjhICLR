# 三环境同步 RGB-D 独立试验

此包接入 ALFRED、Habitat 和 Manipulation 的原生仿真深度。每个模型请求依次输入当前 RGB 和同像素坐标的 500×500 深度可视化图，附带一段灰度含义说明。显示范围固定为 0–10 米，亮近暗远，无效像素为黑色；原始 float32 米制深度另存为 `.npy`。

ALFRED 读取同一 `last_event` 的 RGB 和 `depth_frame`，毫米除以 1000。仅在拥有的工作进程中开启深度渲染，包括辅助动作的渲染选项。

Habitat 添加 `head_depth_sensor`，复制 RGB 传感器的位置、方向、视场角和投影类型，保留 500×500 网格，关闭深度归一化。传感器字段的插入使用工作进程内的 `open_dict`。

Manipulation 读取同一观测里的 `front_rgb` 和 `front_depth`，使用观测中相机裁剪平面换算：`near + depth × (far − near)`。保留官方 RGB 上的坐标轴和物体框，以及匿名物体坐标输入。官方动作失败时可能返回上次观测，双图仍绑定该观测。

当前版本配置为 [experiment-20261107-r02.json](./experiment-20261107-r02.json)。r01 的代码压缩包、冻结身份和首次抓图回执均保留；Habitat 在 r01 的配置插入阶段失败，尚无模型请求，r02 修正后通过抓图验收。各次运行目录不可覆盖。

```bash
# 从 embodiedRunner 目录执行；不发模型或仿真请求
python -B -m unittest discover -s rgbd_three_envs/tests -v
python -B -m rgbd_three_envs check \
  --manipulation-catalog official_direct/assets/manipulation_assets.json

# 本次运行：runs/three-rgbd-smoke-20261107-r02
# ALFRED 独立启动：runs/three-rgbd-alfred-recovery-20261107-r02
# 读取并核验保存的请求、观测、数值、动作与终态；不调用模型
python -B rgbd_three_envs/verification/audit_three.py
```

模型参数为 qwen3.8-flash、temperature=0、seed=137、max_tokens=4096、enable_thinking=false、JSON-object。沿用各环境的官方提示、示例、反馈、完整计划执行和步数上限。Manipulation 未改用单步版本。

只有本地模型协调器临时加载先前授权的密钥文件；工作进程和 SSH 的环境使用白名单，不携带 API 密钥。服务失败停止，不自动重试或重发动作。官方快照、资产、旧实验源码和结果按哈希核验。

本次只选择 ALFRED base:000、Habitat base:000、Manipulation base:001，各一题，用于验证 RGB-D 接口和运行链路。实际结果见 `rgbd-three-report-20261107.md`，不能将这三题结果用作完整 benchmark 成绩。

三环境同时启动时，ALFRED 在 Unity 图形初始化阶段超时，尚无观测、模型请求或动作。原始失败记录保留，残留的本轮 ALFRED 进程按运行目录精确清理。后续使用相同冻结代码和模型配置在新目录单独启动 ALFRED。该处理改变启动并发，未对失败 HTTP 请求或仿真动作做自动重发；并行初始化与该异常的因果关系尚未验证。
