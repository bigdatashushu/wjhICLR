# harness3D 接入 EmbodiedBench 的适配分析

**逐轮快照 V2：Direct/B01/B11 的单动作宿主、不可变当前观测、局部感知、完整 Skill、隔离 Python 与配对审计已接通。离线入口和执行范围见 [ACTION_USAGE.md](ACTION_USAGE.md)，验收状态见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)。以下真实裸模型结果属于此前基线，V2 尚未运行真实模型/感知/仿真。**

检查日期：2026-11-05。项目代码基线：`1c4fb72479967266107ae3015694f5bcb506c8fe`。官方评测源码固定为 `9be4e980e9cd6bcb38373cd4aab7c32724bdd401`（2026-05-30）。

**接入进展：Navigation、ALFRED、Habitat 的官方协议裸 Qwen3.8-Flash 入口均已实现，各完成 3 道真实任务，成功题数依次为 3/3、3/3、2/3。本轮共 35 次模型请求、91 个环境动作；三个运行的最终审计均通过。保留官方提示、文本示例、动作和评分，没有注入自有 Skill 或几何工具。协议、结果与运行入口见 [OFFICIAL_SUITE.md](OFFICIAL_SUITE.md)。历史导航试跑保留在 [BARE_BASELINE.md](BARE_BASELINE.md)。三维工具、受限程序和 Skill 演化属于后续方法接入。**

## 离线验证与本地配置

`tests/` 可在干净 Git 检出中离线运行。比较配置检查需要本机的
`config/action-compare*.json`；缺少这些文件时，该项测试会明确跳过。
源文件核验使用随代码发布的 `source_inventory.py` 清单，并重新计算文件哈希。
运行配置由对应运行清单的 `live_config_sha256` 单独固定。
历史实验的保护收据仍由 `action_preservation.py` 核验。

## 适配判断（实现前检查）

**可以适配。现有静态空间问答入口与官方动作接口存在差异，需要独立的具身 runner 来建立“观察—规划—动作—环境反馈”交互循环。本目录现已完成 Navigation、ALFRED、Habitat 的模型基线接入；本轮范围由用户指定。**

此初始判断有实际接口检查支撑：官方四个环境名称均被现有 VSI 题型路由和 small-image 数据集入口拒绝；现有沙箱提交的是答案，官方环境接收的是动作。初次离线预检时，Mac 未安装仿真依赖，也没有执行具身任务。[初次预检收据](preflight.json)。后续已在 Linux 独立环境运行真实裸模型试跑，结果见上方进展。

博客介绍的是 EmbodiedBench，正式项目为 ICML 2025 Oral。原始论文覆盖 1,128 个测试任务、四个环境和六类能力；其中 Navigation 没有 spatial_relationship 子集，Manipulation 没有 long_horizon 子集。它评估执行任务的成功率，不能用静态空间问答的准确率替代。<a href="#source-1">[cite:1]</a><a href="#source-2">[cite:2]</a><a href="#source-3">[cite:3]</a>

## 四个环境的接入范围

下表中的工程量是根据接口差异作出的判断，不是已测量的工时。

| 环境 | 原始测试规模 / 步数上限 | 官方动作与输入 | 现有项目的复用范围 | 接入判断 |
|---|---|---|---|---|
| EB-Navigation | 300 / 20 步 | 当前第一视角 RGB、指令、动作反馈；8 个离散动作 | Qwen 客户端、检测与分割、单视角深度、受限程序、请求预算 | **优先接入**；动作接口简单，主要新增闭环和导航策略 |
| EB-Manipulation | 228 / 15 步 | 默认 front RGB、标记和物体坐标；7 维离散动作 | 视觉指代、坐标计算、工具推理、程序执行与校验 | **可以接入**；官方控制器已提供，需新增操作策略、动作合同和坐标适配 |
| EB-ALFRED | 300 / 30 步 | RGB、反馈和动态高层动作清单；find/pick/open/put 等 | 模型调用、计划生成、工具框架、轨迹记录 | **可以接入**；增加物体状态与长程计划管理，现有距离/面积 Skill 的直接作用较小 |
| EB-Habitat | 300 / 30 步 | RGB、反馈和高层导航/拾取/放置/开关动作 | 模型调用、计划生成、轨迹记录 | **可以接入**；还需 Habitat 场景资产和任务语义适配 |

规模来自论文；步数、动作及输入来自固定版本官方代码。Navigation 的五份任务清单已下载并计数，五个子集各 60 个，合计 300 个。Habitat 所需的 ReplicaCAD、YCB、机器人和官方任务资产均已安装；ALFRED 已准备 `base` 前三题的真实轨迹。表中其它环境的总体规模采用论文定义。<a href="#source-2">[cite:2]</a><a href="#source-5">[cite:5]</a><a href="#source-6">[cite:6]</a><a href="#source-7">[cite:7]</a><a href="#source-8">[cite:8]</a>

### Navigation

动作 ID 的实际映射是：0/1 前进/后退 0.25 米，2/3 右移/左移 0.25 米，4/5 右转/左转 90°，6/7 相机上仰/下俯 30°。成功判定是机器人到目标的水平距离不超过 1 米，由环境计算。模型正常获得的是图像、指令和动作反馈；评测侧的目标位置、距离及完整 metadata 不应直接交给规划器。<a href="#source-5">[cite:5]</a>

第一阶段可用当前 RGB、动作反馈和一个小型历史状态运行闭环。检测、分割和单视角相对深度有现成接口；若后续采用多帧重建，只能使用动作执行后已经获得的历史观测。旋转造成的低视差、相对尺度和跨轮坐标一致性仍需处理，不能直接将相对深度换算为米制移动距离。

### Manipulation

官方环境使用 `ABS_EE_POSE_PLAN_WORLD_FRAME`，已经包含末端位姿运动规划接口，接入时不需要从零实现机械臂控制器。模型输出的是位置索引 0–99、欧拉角索引 0–119 和夹爪状态 0/1；官方转换函数将它们转换为**世界坐标下的绝对末端位置、四元数和夹爪状态**。不能按博客问题定义中的通用增量位移示例解释实际接口。<a href="#source-6">[cite:6]</a><a href="#source-9">[cite:9]</a>

官方输入中的 `avg_obj_coord` 来自模拟器深度、实例 mask、相机内外参和对象信息，再进行点云平均及体素离散化；YOLO 用于图像框标注。默认只展示 front RGB 时，坐标生成函数仍处理四个相机的深度和 mask。**这些三维坐标是评测提供的辅助信息，不能归因为 harness3D 从 RGB 自主重建的结果。**按标准输入接入能够减少定位工作，仍需模型完成视觉指代和操作规划。<a href="#source-9">[cite:9]</a><a href="#source-10">[cite:10]</a>

若额外开展自主感知实验，需要单独定义输入条件。现有单视角深度工具输出的是每视角相对单位，VGGT 重建坐标也不能未经变换就写入机械臂工作空间；该空间的边界和体素转换由官方代码明确指定。

## 现有代码的复用与改动

| 项目模块 | 已有能力 | 接入时需要处理的差异 |
|---|---|---|
| [BailianClient](../src/skill3d/synthesis/bailian_client.py) | 多模态消息、seed、温度、输出预算、身份收据 | 可复用；具身动作 JSON 的解析和模型兼容性需要另行验证 |
| [small-image 感知后端](../src/skill3d/small_image/backends.py) | GroundingDINO、SAM2、类别审核、MoGe2 的适配接口 | 可复用；每次新观测需绑定自己的图像和产物身份 |
| [small-image 工具](../src/skill3d/small_image/tools.py) | 候选、点深度、同视角点距离 | 可复用局部视觉证据；没有跨视角米制或机器人控制坐标保证 |
| [工具注册表](../src/skill3d/tools/registry.py)、[沙箱](../src/skill3d/sandbox/kernel.py) | 参数校验、受限程序、工具结果与收据 | 可复用基础机制；新增动作输出合同，环境执行放在 host 层 |
| [VSI Episode](../src/skill3d/schemas/episode.py)、[题型路由](../src/skill3d/routing/task_classifier.py) | 固定帧集、八类问答任务 | 新增具身 Episode/Observation；不能通过改数据集名称完成接入 |
| [现有在线 runner](../src/skill3d/online/runner.py) | 在冻结场景上多轮查看工具证据并提交答案 | 新增环境 reset/step、状态更新、动作历史、重规划和终止处理 |
| Skill 库与演化机制 | 方法全文交付、来源 hash、轨迹修订和配对门 | 新增导航/操作/高层计划 Skill；配对指标改为任务成功率，保持训练与测试分离 |
| 评分与运行记录 | 问答 Accuracy/MRA、过程可靠性 | 主指标读取环境 `task_success`；附加动作合法率、无效动作数、环境步数、模型请求数 |

`YieldObservations` 在现有系统中表示读取同一冻结场景的工具证据，不会移动机器人或改变仿真世界。动作执行后，持有的物体、遮挡、相机位置和几何证据可能变化，因此旧证据需要更新或失效。原有 32 帧视频合同应保留在 VSI 路径，具身路径另设观测合同。

现有 SkillSpec 能承载方法正文和身份，但当前检索及活跃方法内容围绕 VSI 问答任务组织。框架机制的复用不能代替新增动作策略，更不能证明在具身任务上已有收益。

## 接入位置与运行环境

建议在本目录新增一个规划器，实现官方所需的 `reset`、`act` 和 `update_info`，并保留官方环境的任务加载、动作语义和成功判定。Navigation 的 `act(observation, user_instruction)` 返回动作及 JSON 计划；Manipulation 另外接收 `avg_obj_coord` 和 `task_variation`。官方 evaluator 会依次执行计划中的动作并更新图像与反馈。<a href="#source-10">[cite:10]</a><a href="#source-11">[cite:11]</a><a href="#source-12">[cite:12]</a>

接入可分为两个层次：

1. **模型基线**：使用现有 Qwen 客户端接收官方输入，输出官方动作计划，验证环境闭环和评分。该结果衡量模型在官方代理框架中的表现。
2. **harness3D 方法**：接入视觉工具、受限程序和具身 Skill，由程序输出经过校验的动作计划。工具计算和环境动作分别计数，所有动作仍由官方环境执行。

当前 Mac 为 Darwin arm64、项目 Python 为 3.11.16。预检确认 `ai2thor`、`habitat`、`habitat_sim`、`pyrep`、`rlbench`、`amsolver` 均未安装。官方提供三个 Python 3.9 Conda 环境；ALFRED 使用 AI2-THOR 2.1.0，Navigation 使用 5.0.0，Habitat 使用 Habitat-Sim 0.3.0，Manipulation 安装方案包含 Ubuntu 20.04 对应的 CoppeliaSim 4.1.0。官方 Docker 方案也采用 Linux x86_64。<a href="#source-4">[cite:4]</a>

**仿真器运行在 Linux，并与现有 Python 3.11 推理/感知环境分进程通信。**这有明确的依赖依据：项目要求 NumPy ≥2、Pydantic ≥2.13，官方导航环境冻结 NumPy 1.26.4、Pydantic 2.10.5。直接把官方依赖装进当前 `.venv` 会产生版本冲突。后续已在 4090 服务器的独立 `embodiedRunner/venv-nav` 和 `venv-household` 安装并验证三类仿真器，由 Mac 发起 Qwen 请求。

## 正式运行前需要固定的协议

- **版本和预算**：固定源码、场景数据、模型参数、分辨率、示例、观测历史、工具与模型请求上限。论文设置为 2048 输出 tokens，但本次固定的官方 `remote_model.py` 已设为 4096；使用哪套预算必须写入实验身份。ALFRED 导航与指令、Habitat 长程数据也已有官方修复说明，不能将不同版本结果静默合并。<a href="#source-2">[cite:2]</a><a href="#source-4">[cite:4]</a><a href="#source-12">[cite:12]</a>
- **动作和故障**：官方 Navigation 的解析器及环境对非法动作存在随机替换，evaluator 的异常路径存在持续重试。独立适配层应在调用环境前严格校验动作，保留错误记录并限制请求，避免将随机替代动作或无限重试当作模型策略；底层环境成功条件保持原定义。<a href="#source-5">[cite:5]</a><a href="#source-11">[cite:11]</a>
- **输入和评分**：规划器只接收约定的观测、反馈和官方辅助信息；成功指标留在 host。配对运行固定同一初始任务，分别记录两种策略随后形成的轨迹，不能要求动作后的观测保持相同。所有登记任务保留在分母中。
- **学习与测试**：已有公开轨迹可用于方法开发，正式测试前需要按任务/场景隔离学习与测试范围并冻结 Skill，避免从测试结果修订后继续报告同一测试表现。
- **输出路径**：官方环境默认写入相对路径 `running/`，官方 `main.py` 在导入时还尝试创建 `data` 链接。后续运行应从本目录启动，并显式将运行结果、模型/数据缓存、临时目录和 Hydra 输出定位到本目录；不得从项目根直接启动官方入口。<a href="#source-5">[cite:5]</a><a href="#source-12">[cite:12]</a>

## 本次完成的检查与文件

初次分析下载并逐文件验证了 35 份官方代码/配置/任务清单，约 519 KB；后续补充 Navigation `utils.py`、`planner_utils.py` 和导航文本示例，共 38 份。验证包含 Git blob 身份、SHA256 和 Python 源码语法。实际 Navigation 任务计数为五个子集各 60；提取并检查了四种环境的 step 接口和步数上限；运行了现有路由对四个官方环境名称的拒绝检查。

初次离线预检期间网络连接被禁用，网络连接尝试、模型请求、仿真器启动和具身任务执行均为 0。后续导航 26 项、家务 6 项及 4 项子检查通过；三类真实环境 reset/step 和本轮 9 题试跑均完成。Navigation 的 14 项运行审计、ALFRED 和 Habitat 各 19 项运行审计通过。完整 harness3D 方法仍需接入并与裸模型使用同一评测协议比较。

| 文件 | 用途 |
|---|---|
| `README.md` | 当前适配分析 |
| `preflight.py` / `preflight.json` | 可重复的离线接口检查与本次结果 |
| `fetch_sources.py` | 获取固定版本的小范围官方证据 |
| `upstream_tree.json` / `official_sources.json` | 官方源码索引和逐文件身份 |
| `upstream/` | 本次查阅的官方源码、配置和五份导航任务清单；为部分证据快照，不是完整可运行安装 |
| `run.py` / `navigation.json` / `USAGE.md` | 初版导航接入入口、冻结 pilot 配置及运行说明 |
| `bare_run.py` / `official_bare.py` / `bare_navigation.json` | 官方提示协议的 Qwen 裸模型入口 |
| `BARE_BASELINE.md` / `runs/qwen-bare-official-3shot-pilot/` | 3 题真实试跑报告、结果和审计 |
| `OFFICIAL_SUITE.md` / `runs/official-suite-summary.json` | 本轮三类任务共 9 题的协议、结果与统一计数 |
| `household_run.py` / `household_planner.py` / `household_worker.py` | ALFRED/Habitat 官方提示适配、真实执行和独立仿真进程 |
| `household_protocol.py` / `audit_household_run.py` | 家务观测合同与真实结果审计 |
| `fetch_household.py` / `fetch_household_data.py` / `fetch_habitat_assets.py` | 官方家务源码、真实轨迹与不可变资产下载 |
| `household_sources.json` / `alfred_pilot_data.json` / `habitat_assets.json` | 源码、轨迹及资产身份 |
| `planner.py` / `navigation_skill.md` | Qwen 规划器和初始导航 Skill |
| `protocol.py` / `bridge.py` / `simulator_worker.py` / `runner.py` | 动作合同、进程通信、官方导航环境和评测循环 |
| `test_navigation.py` / `test-output/` | 工程测试及其产物 |

重新执行离线检查：

```bash
../.venv/bin/python -B preflight.py
```

## 来源

<a id="source-1"></a>[cite:1] 用户提供的[博客介绍](https://www.cnblogs.com/emergence/p/18818926)。

<a id="source-2"></a>[cite:2] [EmbodiedBench 论文 v2](https://arxiv.org/html/2502.09560v2)，重点核对第 3、4、5.1 节。

<a id="source-3"></a>[cite:3] [官方项目主页](https://embodiedbench.github.io/)，核对项目身份与四环境定义。

<a id="source-4"></a>[cite:4] [固定版本官方 README](https://github.com/EmbodiedBench/EmbodiedBench/blob/9be4e980e9cd6bcb38373cd4aab7c32724bdd401/README.md)。安装依赖同时核对本地 `upstream/conda_envs/` 与 `upstream/Docker/Dockerfile`。

<a id="source-5"></a>[cite:5] [EBNavEnv.py](upstream/embodiedbench/envs/eb_navigation/EBNavEnv.py)，动作、成功条件、默认反馈与随机动作替换。

<a id="source-6"></a>[cite:6] [EBManEnv.py](upstream/embodiedbench/envs/eb_manipulation/EBManEnv.py)，绝对位姿控制与任务成功条件。

<a id="source-7"></a>[cite:7] [EBAlfEnv.py](upstream/embodiedbench/envs/eb_alfred/EBAlfEnv.py)，动态动作清单和步数。

<a id="source-8"></a>[cite:8] [EBHabEnv.py](upstream/embodiedbench/envs/eb_habitat/EBHabEnv.py)，高层动作、反馈和步数。

<a id="source-9"></a>[cite:9] [eb_man_utils.py](upstream/embodiedbench/envs/eb_manipulation/eb_man_utils.py)，动作转换和第 269–347 行的坐标来源。

<a id="source-10"></a>[cite:10] [Manipulation evaluator](upstream/embodiedbench/evaluator/eb_manipulation_evaluator.py)，第 99–178 行的实际输入和闭环。

<a id="source-11"></a>[cite:11] [Navigation evaluator](upstream/embodiedbench/evaluator/eb_navigation_evaluator.py) 与 [Navigation planner](upstream/embodiedbench/planner/nav_planner.py)，动作执行、解析和重试。

<a id="source-12"></a>[cite:12] [官方入口](upstream/embodiedbench/main.py)、[模型调用](upstream/embodiedbench/planner/remote_model.py)、[动作 JSON schema](upstream/embodiedbench/planner/planner_config/generation_guide.py) 和 [Manipulation planner](upstream/embodiedbench/planner/manip_planner.py)。

所有本地官方证据的完整固定版本 URL、Git blob 身份和 SHA256 均见 [official_sources.json](official_sources.json)。项目侧判断依据表格中链接的实际源码及 [preflight.json](preflight.json) 中的源码 hash。
