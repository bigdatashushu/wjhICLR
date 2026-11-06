# 交接提示词：harness3D 接入 EmbodiedBench 三个环境的实现前准备

你要接手 `/Users/bytedance/wangjh/wjhICLR` 项目，为我们的 harness3D 架构接入 EmbodiedBench 做完整的实现前准备。

本次“三个子集”指 **EB-Navigation、EB-ALFRED、EB-Habitat 三个环境**；已完成的裸模型试跑来自各环境 `base` 子集的前三题。Manipulation 不在本次接入范围。不要将三个环境误解为三个能力子集，也不要据此默认启动所有能力子集的完整实验。

**当前阶段只完成调研、接口核对、可行性比较、详细设计和实施计划。完成所有可以自主完成的前置工作后，提交具体方案，停在修改代码之前，等待我批准实施。不要只列一个待办清单就结束；也不要抢先实现接口、修改配置或试跑新方法。**

## 1. 目标及当前状态

我们已有可运行的官方协议裸 Qwen3.8-Flash 基线。下一步要把 **完整 Skill 交付 → coding agent 生成程序 → 隔离执行 → 感知/几何工具及校验 → 动作提交 → 官方环境执行 → 新观测与反馈** 的方法接入三个环境。

这次不是重新安装评测环境，也不是在现有 JSON 动作规划器的提示里加一段 Skill 文本。必须明确我们现有架构哪些部分能直接复用，哪些部分需要为交互式任务增加接口。

`embodiedRunner/run.py`、`planner.py`、`navigation_skill.md` 是早期导航 Skill 文本注入入口，没有完成完整的程序执行和几何工具接入。`bare_run.py`、`official_bare.py`、`household_run.py`、`household_planner.py` 是已验证的裸模型基线入口。不能把上述初版 Skill 注入当作完整架构接入的完成证据。

## 2. 工作目录、版本与工作树

- 主项目：`/Users/bytedance/wangjh/wjhICLR`
- 本次任务根：`/Users/bytedance/wangjh/wjhICLR/embodiedRunner`
- Mac 已有项目 Python：`/Users/bytedance/wangjh/wjhICLR/.venv/bin/python`，Python 3.11。
- 交接时项目 HEAD：`1c4fb72479967266107ae3015694f5bcb506c8fe`，当前分支为 `main`。接手时重新核对，不把交接记录当作永久状态。
- 工作树有其他任务的未提交改动，包括 `GPTexp/`、`scripts/BLINKRunner/`、`server.md`，以及未跟踪的 `VSIbench8Brunner/`、`embodiedRunner/`。保留这些工作；不要 reset、clean、覆盖、自动提交或同步到远端。
- 先查适用的 `AGENTS.md`、本地项目规则和用户长期约束。工程接入不需要擅自切换成论文写作或自主科研流程。
- 本次文档、证据、检查产物均放在 `embodiedRunner/`。后续若确实需要改 `src/skill3d/`，在方案中列出具体文件和原因；当前不修改。
- 主项目、`VSIbench8Brunner/` 和其他实验目录可能含不同版本的代码。追踪真实 import 和启动入口，明确将复用哪份源码，不能混用副本。

## 3. 必须先读的资料

先阅读以下文件，再沿真实调用链做针对性检查：

1. 主项目 `系统架构v11.md`、`README.md`。
2. `embodiedRunner/README.md`、`OFFICIAL_SUITE.md`、`USAGE.md`。
3. `embodiedRunner/runs/official-suite-summary.json`。
4. 三个 `runs/qwen-bare-official-suite-{nav,alfred,habitat}-3/` 内的 `startup.json`、`config.json`、`summary.json`、`audit.json`，以及需要核对的逐题输入和 trace。
5. `embodiedRunner/official_sources.json`、`household_sources.json`、`alfred_pilot_data.json`、`habitat_assets.json`、`habitat_lab_source.json`。
6. 现有基线、桥接和工作进程代码，见下面的接口清单。

架构文档和 README 有历史基线与版本说明。**当前源码及实际运行收据优先于文档里的历史默认值。** 对文档与源码的差异建立表格，不能静默采用其中一份。

## 4. 官方版本、环境及数据位置

官方项目为 EmbodiedBench：

- 官方仓库：`https://github.com/EmbodiedBench/EmbodiedBench`
- 本轮固定 revision：`9be4e980e9cd6bcb38373cd4aab7c32724bdd401`
- 博客只用于背景：`https://www.cnblogs.com/emergence/p/18818926`
- Navigation 证据快照：本地 `embodiedRunner/upstream/`，身份由 `official_sources.json` 验证。
- ALFRED/Habitat 源码、配置及相关 Git LFS 内容：本地 `embodiedRunner/upstream-household/`，身份由 `household_sources.json` 验证。
- 不覆盖这两份官方快照，不改原始评分器、任务指令、动作语义或原始数据。若需要比较新版官方实现，应单独记录版本差异。

仿真器已安装在 **4090 服务器 `cvailab@10.10.0.198`**：

| 内容 | 远端绝对路径 |
|---|---|
| 任务根 | `/home/cvailab/wangjh/embodiedRunner` |
| Navigation Python | `/home/cvailab/wangjh/embodiedRunner/venv-nav/bin/python` |
| ALFRED/Habitat 共用 Python | `/home/cvailab/wangjh/embodiedRunner/venv-household/bin/python` |
| 家务官方代码与数据 | `/home/cvailab/wangjh/embodiedRunner/upstream-household/` |
| Habitat-Lab 0.3.0 源码 | `/home/cvailab/wangjh/embodiedRunner/runtime/habitat-lab-v0.3.0/` |

安装和运行情况：

- Navigation：AI2-THOR 5.0.0，独立 `venv-nav`；本轮配置为物理 GPU 3。
- ALFRED：AI2-THOR 2.1.0；Unity、Xvfb 和字体均在任务目录。使用 Mesa 软件渲染，`LP_NUM_THREADS=2`，不依赖系统字体替换。
- Habitat：Habitat-Sim / Habitat-Lab 0.3.0，`headless withbullet`；本轮物理 GPU 7，工作进程将其映射为逻辑设备 0。
- 家务环境已验证版本：Python 3.9.23、Gym 0.23.0、NumPy 1.23.5、PyTorch 2.4.0+cpu。回执为 `checks/household-runtime-ready.json`。
- 三个环境均通过真实 reset/step 检查；家务初始图像在 `checks/alfred-initial.png`、`checks/habitat-initial.png`。
- Mac 发起模型请求，Linux 工作进程仅负责仿真。保留进程隔离，不把旧仿真依赖装进主项目 `.venv`。

数据情况：

- Navigation：五个能力子集的 JSON 任务清单已准备，各 60 题。
- ALFRED：目前只下载并验证了 `base` 前三题的真实轨迹，见 `alfred_pilot_data.json`。不能据此声称全量 ALFRED 数据已就绪。
- Habitat：三个环境资产仓库共 994 文件，约 690 MB，逐文件身份已验证并中转到远端；官方补充任务数据包也已安装。
- Habitat 资产标签：ReplicaCAD `v1.6`、hab_fetch `v2.0`、YCB `v1.2`，不可变 commit 和文件 hash 在 `habitat_assets.json`。
- 远端 Hugging Face 下载此前遇到连接问题，已有 Mac 下载与中转脚本。优先复用已验证资产，当前不要重新下载或复制大数据。

如果需要只读 SSH 核对，先读 `harness3d-cluster-ops` skill 及其服务器清单。该 skill 管服务器操作，不负责本任务的科学结论。

优先复用本地 `embodiedRunner/checks/ssh4090` socket：

```bash
ssh -T -o BatchMode=yes -o ControlMaster=no \
  -S /Users/bytedance/wangjh/wjhICLR/embodiedRunner/checks/ssh4090 \
  cvailab@10.10.0.198 '<经过核对的只读命令>'
```

socket 可能过期。需要恢复时先读 `ssh_session.py` 与 `remote_probe.py`，确认边界后使用既有任务入口；认证在程序内处理，不打印、复制或展示 `server.md` 的内容，不绕过 host-key 检查。

4090 本项目可使用的物理卡是 0、1、3、7；其他卡属于共享资源。只读查询当前占用，不能用历史空闲量直接承诺可启动重建。本阶段不启动模型、感知、重建或新仿真任务。

## 5. 已完成的裸模型基线

三个运行都来自各自 `base` 的索引 `[0,1,2]`：

| 环境 | 成功题数 | 模型请求 | 环境动作 | 最终审计 |
|---|---:|---:|---:|---|
| Navigation | 3/3 | 11 | 30 | 14 项通过 |
| ALFRED | 3/3 | 11 | 33 | 19 项通过 |
| Habitat | 2/3 | 13 | 28 | 19 项通过 |

共 9 题、35 次模型请求、91 个环境动作。这里的审计通过说明运行身份和计数检查通过，不说明所有任务成功，也不证明主架构收益。

重要观察：

- ALFRED 第三题最终成功，但累计 7 次无效动作，其中 5 次为计划解析错误。
- Habitat 第一题执行 20 步，累计 10 次无效动作后失败；后两题均以 1 次模型请求、4 步成功。
- 日志不保存原始模型请求/响应正文。不要仅凭错误计数断言解析错误或失败的根因；证据不足时列出后续需要补采的最小信息，当前不复跑。
- 这些题已用于接入调试，是开发样本。未来正式泛化或最终测试要另设独立任务，不能用这 9 题证明方法泛化。
- 历史导航目录 `runs/qwen-bare-official-3shot-pilot/` 是另一次试跑，不能和本轮目录混合计数。

基线协议：

- `qwen3.8-flash`，`temperature=0`，`seed=137`，`enable_thinking=false`，`max_tokens=4096`。
- 当前模型输入为 1 张原始当前 RGB，500×500；不额外给标准答案、GT 几何或完整模拟器 metadata。
- 保留官方 system prompt、文本示例、反馈组织和成功判定。Navigation 是官方 3-shot，ALFRED/Habitat 是官方 10-shot。
- JSON-object 模式替代官方 SDK 的结构化 schema，这一协议差异已记录。
- Navigation：20 个环境动作上限，每题最多 20 次模型请求，三题最多 60 次。
- ALFRED/Habitat：30 个环境动作上限、10 次无效动作上限，每题最多 40 次模型请求，三题最多 120 次。
- 家务任务逐个执行计划，动作失败后重新规划；空计划停止、非法计划累计无效动作。模型输出非法与服务运行故障是两类事件。
- Habitat：`shuffle=false`、`group_by_scene=false`、`seed=137`，校验实际 episode ID。
- 服务或仿真运行故障停止，没有自动重发模型请求。
- 裸模型不加载自有 Skill、prepared geometry、候选、额外图像或工具。

保留现有基线目录、配置和收据。本阶段不修订基线、不增加模型请求。后续若要改 JSON 格式策略、预算或历史图像策略，需建立新协议和新运行目录，对比臂一起固定条件。

## 6. 需要追踪的真实代码接口

### 官方环境桥接

在 `embodiedRunner/` 核对：

- Navigation：`bare_run.py`、`official_bare.py`、`bare_navigation.json`、`protocol.py`、`runner.py`、`simulator_worker.py`、`remote_runtime.py`。
- ALFRED/Habitat：`household_run.py`、`household_planner.py`、`household_protocol.py`、`household_worker.py`。
- 共用通信：`bridge.py` 中的 `SimulatorBridge`；家务扩展为 `HouseholdBridge`。
- 共用 Qwen 客户端入口：`run.py::model_client`，真正读取主项目 `src/skill3d/synthesis/bailian_client.py`。
- 审计：`audit_bare_run.py`、`audit_household_run.py`。

通信是有界 NDJSON，包含请求身份、版本、经过筛选的观测和动作。核对 `open/reset`、`step`、非法计划、关闭和超时路径，建立端到端时序。

现有协议身份为 `embodiedrunner-navigation/1` 和 `embodiedrunner-household/1`。家务桥接端有评分字段，但传给模型的 `Observation` 只有图像、step、指令、动作表和输入身份；新架构不能绕开此筛选。

注意已经解决的运行问题：

- ALFRED 动作表按场景变化，每题必须更新；首题真实动作表为 208 项，不能硬编码成所有场景都如此。
- Habitat 是 70 个官方高层动作，首轮 reset 后需要核对实际 episode ID。
- Habitat 图像必须使用 `observations_to_image(obs, {}, key="head_rgb")`，不能拼入第三视角或深度。
- `household_worker.py::native_log` 同时捕获 Python 与 Habitat/Magnum 原生输出，JSON 通信使用独立 fd，不能用普通 stdout 重定向替换它。
- 权威动作和评分定义在固定版本的官方环境中，不重新实现一套近似评分器。

Navigation 8 个动作依次为前/后/右/左移动 0.25 米，右/左旋转 90°，相机上/下俯仰 30°。ALFRED/Habitat 的 `find`、导航、拾取、放置、开关等是官方高层动作；不要拆换为不同控制粒度后继续报告同协议成绩。

### harness3D 架构

从主项目 `src/skill3d/` 追踪：

- `online/runner.py`：`OnlineRunConfig`、`run_episode`、`_solve_program`、`_synthesize`、`_execute_program`、Skill/证据交付记录。
- `synthesis/prompt_builder.py`、`execution_contract.py`、`request_governor.py`、`bailian_client.py`。
- `sandbox/kernel.py`：`RestrictedNamespaceKernel`、程序终止、`ReturnAnswer`、`YieldObservations`、`ReadObservation`。
- `online/submission.py`、`observation_reader.py`、`progress.py`：提交、证据引用、观察分页与终结边界。
- `schemas/episode.py`、`schemas/skill.py`、`schemas/tool_response.py`、`schemas/authorization.py`、`schemas/trace.py`。
- `routing/task_classifier.py`、`tools/registry.py`、`tools/task_policy.py`、`tools/contract.py`、`tools/docs_v11.py`。
- `skills/v11_library.py`、`skills/delivery.py`；`skill_library/snapshots/active_snapshot.json`、对应 snapshot、manifest 和完整 `SKILL.md`。
- `segmentation/` 的本题对象解析/绑定，`infra/m5_identity.py`，重建、质量门和 `verifier/` 的实际依赖。
- `small_image/runner.py`、`backends.py`、`tools.py`、prepared evidence/storage：评估少量图像的感知和相对几何能否复用。
- `evaluation/skill_ablation_v11.py`、`evaluation/direct_qwen_v11.py`；仅调研实验隔离与预算，不执行学习/演化。

当前核对到的主 v11 身份：

| 合同 | 身份 |
|---|---|
| 普通提示词 | `program_synth_v11_14_en` |
| 预算内终结 | `finalization_v11_4_en` |
| 执行协议 | `solver-v11.13-evidence-boundaries` |
| 公共执行合同 | `execution-contract/5.1` |
| 工具文档 | `tool-docs-v11.13-evidence-dependencies` |
| 工具面 | `tool-face-v11.12-runtime-dependencies` |
| Skill/快照 | `SkillSpecV11` / `runtime-skill-snapshot/3.0` |

交接时 active snapshot 指向 `mixed-r04-20261026-v1`。它面向现有空间问答，不能默认适用于三个具身环境，也不能直接修改全局 active pointer。

主 v11 统一求解与 small-image 路径的合同不同。例如主 v11 有有界 `ReadObservation`，当前 small-image prompt 明确不支持该接口。核对实际源码，不把它们拼成一套不存在的 API。

主线 v11 普通轮完整交付冻结 Skill，终结轮省略完整正文并记录省略原因。普通轮生成 Python，通过受限执行和工具验收；现有终结提交的是答案。具身动作需要新的、明确定义的提交边界，不能直接把动作数组塞进原有答案字段。

## 7. 必须完成的实现前工作

### A. 核对工程和运行事实

重新确认工作树、源码副本、真实 import 来源、环境版本、官方 source/data hash、已完成基线和数据覆盖。必要时只读核对远端路径、现有进程和资源；不启动任务。

记录“已验证事实 / 当前代码支持 / 尚未实现或未验证”三者的区别。对旧报告、旧 prompt、旧接口和新源码的差异给出路径与函数依据。

### B. 画清楚两条完整调用链

分别给出：

1. 当前裸模型：官方环境观测 → 消息/动作计划 → 桥接 → 环境动作 → 官方反馈/评分。
2. 拟接入方法：合法观测 → 感知/证据与冻结 Skill → 程序生成 → 隔离执行与校验 → 动作提交 → 宿主执行 → 新观测。

明确哪些数据留在环境/评分侧，哪些字段可以进入模型、工具和沙箱，哪些过程需要真实模型请求。模型调用、几何/感知调用和环境动作分别计数。

### C. 比较至少两种可行方案，提出推荐

至少比较：

- 新增具身 Episode/Observation 与动作提交适配层，复用 v11 的方法交付、程序执行、工具校验和预算机制。
- 将每个交互决策点封装为有身份的局部观测快照，复用现有求解组件，再由宿主推进环境。

如认为复用 small-image 路径更合适，要说明它与完整 v11 的差异及需要补齐的机制。评估复制一整套 runner 带来的协议分叉与维护成本。

比较接口改动、三个环境适配性、状态更新、复用程度、实验可比性、成本和回归范围。不要在核对源码前硬定方案，也不要用“直接调用已有 runner”跳过当前静态问答合同的限制。

### D. 设计动作提交与交互状态合同

提出字段、状态机和错误语义，至少覆盖：

- episode / environment / subset / task / step 身份、当前观测和动作表版本。
- 程序只能提交经过校验的动作数据；真实 `reset/step/SSH/HTTP` 留在宿主，不能向沙箱暴露凭据或裸环境对象。
- 多动作计划的提交/执行边界，以及何时必须基于新图像重新生成程序。
- 过期观测或过期动作表下的计划如何拒绝；重复请求、断连和超时如何防止动作重复执行。
- 环境动作后程序变量、result ID、对象绑定、持有状态、可用工具和历史证据如何保留、撤销或重新验证。
- 正常任务结束、模型空计划、格式错误、非法动作、预算耗尽、服务故障的区别。
- 模型不得通过自行声称成功、`ReturnAnswer(True)` 或沙箱读取评分数据结束任务；成功由官方环境决定。

`ActionPayload`、`SubmitActions` 等若作为建议名称，应明确它们尚未实现，给出拟定接口而不是宣称已有可调用 API。

### E. 处理动态感知、几何与 Skill 的适用性

现有系统主要面对冻结 FrameSet，具身环境会移动相机、移动对象、改变持有和开关状态。必须设计：

- 每个观测版本的原图、候选、mask、深度/点、来源 hash 和能力声明。
- 是否使用历史帧；只使用动作后实际获得的帧，不读取未来帧、测试答案、模拟器 GT 位姿/深度/对象位置。
- 旋转低视差、单视角相对深度、对象移动和跨轮对应关系的处理；旧几何结果不能自动当作当前世界事实。
- ALFRED 的 `find`、Habitat 高层导航等不能假设为连续、已知位姿的小步移动。
- 官方动作标称移动量是允许的控制知识，但不能把失败动作或高层动作等同于已执行的确定相机位移。
- 无可信尺度时不把相对几何变成米制值；同视角相对深度/点距离不等于跨视角距离。
- 类别确认、检测框与对象表面/mask 验证分别处理；不能将 category-confirmed detection 当成可靠物体表面。
- 初始 Skill 的选择、适用性检查、整题冻结和普通轮完整交付；动态工具权限与方法版本冻结是两个问题。
- 是否需要独立的具身任务 Skill/路由，怎样避免强行映射到 VSI 八类问答题型，以及怎样保留现有问答和全局方法库。
- Skill 演化只列未来接口和阶段。本阶段不修改方法正文、发布候选或用这 9 题做演化后再当独立测试。

不要自动继承 VSI 的固定 32 帧输入、八类题型、现成 prepared geometry 或预算外视觉补答。逐项说明可迁移机制和需要重新定义的合同。

### F. 设计与裸模型的公平比较及后续验收

先规划，当前不跑：

- 必须有保留官方提示/示例的 Direct Qwen 对比臂。
- 如检验 Skill 作用，再规划 B01/B11：二者都是同一 coding agent 生成并执行程序，工具、质量门、公共合同、预算相同，唯一方法变量是完整冻结 Skill 的交付。
- 系统对裸模型的差异不能直接解释为 Skill 单项收益；新增多帧、检测图或额外反馈也必须登记输入条件。
- 固定任务、初始状态、官方版本、模型参数、分辨率、文本示例、历史图像策略、终止和格式解析规则。交互后两臂轨迹可以不同，不要求动作后的图像逐帧相同。
- 所有角色的真实模型请求与 token、感知/几何服务成本、程序错误、环境动作、无效动作和墙钟时间分别记录；提出统一的总预算及各阶段子预算。
- 主要指标为官方 `task_success`；任务进度、错误率、成本作辅助。分母包含失败和运行故障，保留失败历史。
- 解释这 9 个开发样本、后续独立验证和最终测试的隔离；小差异需要噪声或 A/A 诊断，不能从三题成功率承诺收益。
- 列出有意义的离线合同测试、状态失效测试、沙箱动作边界测试、三环境 smoke 和配对小样本验收顺序。将来从只读检查、离线假输入，到用户授权的真实执行逐级推进。

### G. 写出可以直接实施的变更计划

列出精确文件/函数，说明新增或修改的接口、依赖关系、实施顺序、验证命令、通过标准及恢复方式。明确：

- 哪些变更只在 `embodiedRunner/`，哪些必须进入共享架构。
- 哪些实现需要先抽出稳定复用边界，避免复制大型私有函数或修改既有问答语义。
- 如何隔离 dirty 工作树与实验产物；若建议 worktree，说明未跟踪的 `embodiedRunner/` 如何保留和绑定来源。
- 三个环境共同基础层与各自特有逻辑的划分。
- 必需事项与可后续扩展的事项；给出真实阻塞点，不用虚构工时或未经测试的资源容量。

## 8. 当前阶段的执行边界

你可以自主进行文件/代码读取、静态分析、Git 状态检查、经过核对的只读 SSH 查询，以及写前置文档和脱敏检查回执。经源码确认不会读取真实密钥/GT 或触发外部模型、感知、重建、仿真调用的既有离线合同测试也可以执行；清除继承的服务凭据和端点，使用新的任务内输出目录，不覆盖已有基线或初次预检回执。

当前不要修改源码、测试、配置、官方数据、Skill 或快照；不要新增可执行接入代码；不要安装依赖、部署、发出新的模型/感知/重建请求或启动仿真。已有裸模型运行授权不是本阶段追加实验的授权。

已有某些 `check` 入口会调用 `load_authorized_key()`。本阶段不要盲目执行这些入口；先检查调用链，优先使用不读真实密钥、不调用服务的静态检查或工作进程 `probe`。不要打开或展示 `qwenkey.md`；未来获授权的 live run 才通过已有加载器在内存使用凭据。

保护正在进行的 BLINK、GPTexp、FullQA 和其它冻结实验；不复用或改动其 run/config/cache/budget，不重启服务。不得读取真实凭据文件到报告，日志只保留脱敏身份、hash、数值用量和错误类别。

如果单独获准使用工程助手，它们可以帮助梳理代码和检查方案，但不能给在线被评测模型补答案、额外方法或更改冻结变量。当前任务不要求启动新的 agent。

遇到可以靠只读核对解决的问题，自主推进。真正无法确定的选择集中写入文档，解释现有证据、推荐项和需要我决定的原因，不要每读几个文件就停下来问。

## 9. 必须交付和停止点

在 `embodiedRunner/` 交付：

1. `ARCHITECTURE_PREPARATION.md`：实际源码/环境核对、两条调用链、接口复用表、三环境差异、方案比较与推荐、必要设计决定和证据。
2. `ARCHITECTURE_IMPLEMENTATION_PLAN.md`：文件与函数级实施计划、数据/动作/证据合同、状态机、预算与实验矩阵、回归与验收标准、分阶段执行顺序。
3. 如需要，`checks/architecture-preparation/` 下仅保存脱敏的只读检查结果；注明哪些检查本次真实执行、哪些只是后续计划。

两份文档应足够让下一阶段按步骤实现，不能只有“查看、适配、测试”这种泛化任务名。每个重大结论给出具体源码路径、函数或已存在收据；没有验证的内容明确标为待验证。

最后用中文简要汇报推荐方案、可复用部分、必须新增的边界、后续批准后第一步，以及文档链接。**此时结束前置阶段，等待我批准实施，不开始修改代码。**
