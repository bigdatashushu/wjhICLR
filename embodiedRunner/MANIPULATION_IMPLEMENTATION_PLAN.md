# EB-Manipulation 接入

2026-11-06。用户要求先完成第四个官方环境，再修改动态检测类别。全部开发限于 `embodiedRunner/`；现有官方快照、三环境冻结部署和轨迹保持原样。

## 已核实的接口

固定官方 revision `9be4e980e9cd6bcb38373cd4aab7c32724bdd401`：

- 环境为 `EBManEnv`，底层是 amsolver / PyRep / CoppeliaSim。官方每题最多 15 次 `step`。
- 每个动作是 `[x, y, z, rx, ry, rz, gripper]`。位置在 100 格世界坐标体素中，旋转为 3° 离散 Euler 索引，夹爪为 0 或 1。不得映射成家务环境的 action ID。
- 官方评测每轮向模型提供当前 front RGB、匿名物体坐标、指令、动作反馈和任务类型对应的示例。匿名坐标由官方深度、相机参数和实例掩码计算；它们是官方协议公开的派生输入，须明确标记仿真来源。
- 私有 simulator 对象、原始深度/掩码/位姿、物体真名、waypoint 和配置内容只留在可信 worker。只发布匿名坐标，不给予预测几何工具官方坐标的测量权限。
- Full 和 Direct 使用相同官方派生输入及动作合同。Direct 不加载我们的感知、候选、工具或 Skill。Full 的局部预测深度仍为相对单位。
- 保留官方评分和官方失败动作反馈，失败也消耗 step。模型格式错误使用有界修复预算，不能用官方 parser 的随机动作兜底。

## 工作与验证

1. 核实官方依赖、任务资产和服务器现状；选择独立 runtime 和新部署目录。
2. 扩展共享观测、七维动作验证、mutation、沙盒 context 和审计；添加专用 worker、Direct prompt adapter、Skill 和配置。
3. 用无凭据、无网络的进程测试验证整个动作计划、同 UUID ACK、旧快照拒绝、输入白名单、官方评分以及四环境回归。
4. 实际 Linux reset/step 与图像检查；再做开发题 Full/Direct 的有界真实联调，保存完整 JSON 轨迹及只读独立审计。
5. 按实际结果更新接入说明。联调不能当作全量性能结论。

服务故障或未知动作 ACK 后停止，不自动重试。资源只用当前授权的 4090 物理卡 0/1/3/7，使用前查询占用。

上下文分段：官方接口与 runtime 核实；实现和离线验证；Linux 部署与真实联调。先前给出的 token 数量只是工作量估计，不是实验或工具预算。

## 当前已完成的验证

- 四环境 × 三臂的真实 NDJSON fake worker 与隔离程序闭环：12/12，通过独立审计，0 模型请求、0 仿真启动。早期回执在 `checks/action-manipulation-offline-r01.json`，后续源码变化不再与它的冻结身份相同。
- 当前源码完整 macOS 回归 **182 passed、1 skipped**（Linux 专用项），见 `checks/manipulation-regression-r04.xml`；四环境分母为 12。
- 官方源码独立快照：335 文件，203,907,493 字节；未改原 `upstream/` 与 `upstream-household/`。
- 只下载 `base:000` 的 3 份官方配置资产，8,012,261 字节；Host 未反序列化 pickle。资产版本 `4b6c210a3c7539330c2f0d228e778a9400061025`，catalog hash `29080ac75de636aa8621c9013d7c8ea8315721430b2e7ede58408bc52cde64a1`。
- 新配置 `config/action-manipulation-4090.json` 预选 Direct/Full 首动作联调并自动保留完整轨迹；专用 Python 与 runtime 已安装并通过预检。
- 独立 runtime 已安装：CoppeliaSim 4.1.0、PyRep 4.1.0.3、NumPy 1.26.4、SciPy 1.15.3；8,248 份运行文件固定。回执见 `checks/manipulation-runtime-20261106.json`。下载站请求头、IPv4 和宿主 cwd 的 `tools.py` 冲突已经处理；失败安装脚本及日志保留。
- 真实 0 模型 reset/step 已通过，动作执行成功，实际 raw RGB 与官方标注图已人工检查，见 `checks/manipulation-reset-step-r02/`。Python 启动前设置 CoppeliaSim 动态库路径，避免仅在进程内设置 `LD_LIBRARY_PATH` 导致导入失败。
- 最终 r04 Linux 七维动作和真实隔离检查 **30 passed、1 skipped**（macOS 专用项），见 `checks/manipulation-linux-r04.xml`。
- 最终部署 `/home/cvailab/wangjh/embodied-action-v2-manipulation-20261106-r04` 通过 0 推理、0 仿真预检，见 `runs/manipulation-preflight-r03.json`。文件名沿用尝试编号，实际运行源码身份以 manifest 为准。
- 新增 `manipulation_smoke.py` 用于 0 模型的真实 reset/step 诊断；`audit_action_trajectory.py` 独立校验逐事件 JSON、消息与响应 hash、全部资产和 Direct 输入边界。

## 最终真实验收

第四环境已完成安装、动作与公开输入适配、真实 reset/step 和 Direct/Full 首动作联调。当前资产与真实验证范围为 **`base:000`**，不是全量任务评测。

- [真实运行](runs/manipulation-smoke-r02.json)：两臂各执行 1 个官方动作，均取得有效 ACK、新 RGB 和官方动作成功反馈。0 abort、0 程序错误；任务尚未完成，官方进度均为 0，终止原因为 `integration_step_limit`。
- 共 3 次真实推理：Qwen 2 次、MoGe 1 次。Direct 返回合法七维计划；Full 返回合法 Python，在新的 Linux 隔离进程中执行。两臂初始公开输入一致；首动作上限仍使 `paired_comparison_eligible=false`。
- [本地独立运行审计](checks/manipulation-live-audit-r04.json) 和 [完整轨迹审计](checks/manipulation-trajectory-audit-r04.json) 均通过，复核当前全部源码、方法交付、预算、mutation/ACK、输入输出 hash 和全部 13 份轨迹资产。
- [完整轨迹入口](runs/manipulation-smoke-r02-episodes/trajectories.json) 保留实际公开消息、模型响应、程序、前后观测、官方标注图及预测深度/点图。没有下载 worker 的私有配置目录。
- 当前源码的四环境 × Direct/B01/B11 真实 NDJSON fake 闭环 **12/12**，0 真实请求、0 仿真启动；[独立审计](checks/action-manipulation-offline-audit-r04.json) 通过。
- 原 230 份源码、103 份旧基线和 active pointer 保持原样，见 `checks/manipulation-preservation-20261106.json`。没有 commit/push。
- [最终验收收据](checks/manipulation-acceptance-20261106.json) 绑定以上文件、运行和当前源码身份。

## 实际图像与感知范围

[图像复核](checks/manipulation-visual-review-r04.png) 使用保存的真实 RGB、官方标注图、MoGe 相对深度和 Full 动作后 RGB。两臂夹爪均移动到星形物体上方，未执行后续抓取和放置。官方容器框较大，左侧容器框还覆盖邻近物体的图像区域；这些公开标注不作为预测工具的已验证物体表面。

这题指令中的 `star` 和 `container` 不在现有固定公开词表中，七维环境的动作表也为空，因此检测类别为空：**DINO、SAM、Intern 类别审核和工具/reader 均未调用，候选数为 0**。不能把本轮验收称为检测或分割质量通过。Full 仅准备了 500×500 的 MoGe 相对深度和同视角点图；全部深度像素有效、有效点有限，但这只说明数值可读取，不证明测量精度。深度图可见机器人、桌面和背景的粗结构，小物体细节仍未验证。

按用户顺序，当前保留固定检测逻辑；后续动态 Qwen 类别选择应覆盖这些漏检类别，并单独验证检测、分割与审核路径。此前 r03 的 Direct 服务失败及 Full 未启动记录完整保留在 `runs/manipulation-smoke-r01.json` 和其轨迹目录；最终版本补入已核验的官方 JSON 输出模板及去敏服务错误字段。
