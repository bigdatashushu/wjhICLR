# 三个环境同步 RGB-D 验收

ALFRED、Habitat、Manipulation 均已接入并实际运行。完成三题功能验收，全部 10 次模型请求包含当前 RGB 和同步深度两张图；42 份观测完成数值、显示图、同步绑定及消息重放核验。

| 环境 | 读取方式 | 换算为米 |
|---|---|---|
| ALFRED | 同一 last_event 的 RGB / depth_frame；开启辅助动作的深度渲染 | 原生毫米除以 1000 |
| Habitat | 增加 head_depth_sensor，复制 RGB 相机位置、方向、视场角和投影类型 | normalize_depth=False，直接读取米 |
| Manipulation | 同一观测的 front_rgb / front_depth | near + depth × (far − near) |

三种深度图均保留 500×500 像素网格，固定 0–10 米反向灰度：亮表示近，暗表示远。无效像素及显示范围外的远处像素为黑色。显示图是可视化；原始 float32 米制数组保存在 .npy，不进入模型消息。Manipulation 保留官方 RGB 坐标轴和物体框，深度对应相同相机网格。

模型附带的简短说明：第一张是 RGB，第二张是同像素坐标的同步深度；白色接近 0 米，黑色为 10 米以上或无效值；用 RGB 识别物体，用深度判断可见表面。黑色本身不能证明可通行，深度也不能直接当作世界坐标动作。

[三环境首帧双图](computer:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_three_envs/rgbd-three-initial-pairs.png)

| 环境 / 官方题目 | 成功 | 动作步数 | 模型请求 | 结束原因 |
|---|---:|---:|---:|---|
| alfred/base:000 | 1 | 11 | 2 | official_done |
| habitat/base:000 | 1 | 10 | 5 | official_done |
| manipulation/base:001 | 0 | 15 | 3 | official_done |

模型为 qwen3.8-flash，temperature=0、seed=137、max_tokens=4096、enable_thinking=false、JSON-object。保留各环境官方示例、动作反馈、完整计划执行与步数上限；Manipulation 未加入单步重规划版本。无自动服务重试、动作重发或额外模型工具调用。

三个环境均先完成零模型首帧验收，再各运行一题；成功 2/3。这是接口与运行链路验收，不能据此判断整体成功率或深度带来的增益。

动作失败时，Manipulation 官方环境可能返回上一次观测；此时 RGB 与深度仍来自同一份观测，不能将其描述为新物理帧。

新代码位于独立 rgbd_three_envs 包与新的服务器目录。旧基线、Navigation RGB-D 和 Manipulation 试验的 3686 个文件哈希保持不变。
服务器上的 r01/r02 源码、官方快照及选定任务资产核验通过，已无本轮工作进程运行。

[完整数值与逐帧审计](computer:///Users/bytedance/wangjh/wjhICLR/embodiedRunner/rgbd_three_envs/rgbd-three-results-20261107.json)

首次三环境并行启动时，ALFRED 停在 Unity 图形初始化，600 秒超时退出，尚无首帧、动作或模型请求。该失败记录完整保留。清理本轮残留进程后，在其他两环境已结束的条件下另建 ALFRED 单独运行目录完成验证；冻结代码、模型参数、任务和提示均保持一致。这里采用独立启动恢复，没有自动重发 HTTP 请求或原仿真动作；是否由并行图形初始化触发仍未做因果验证。
