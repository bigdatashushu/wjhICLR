# EmbodiedBench 接入与运行

Navigation、ALFRED、Habitat 的官方裸 Qwen3.8-Flash 入口、协议和三题试跑说明见 [OFFICIAL_SUITE.md](OFFICIAL_SUITE.md)。以下保留 Navigation 的原始使用说明。

## 官方协议的 Qwen 裸模型入口

`bare_run.py` 使用固定官方 revision `9be4e980e9cd6bcb38373cd4aab7c32724bdd401` 的导航提示构造方法，并调用 `qwen3.8-flash`。每轮只发一个 user 消息，内容顺序为当前 500×500 第一视角 PNG、官方提示文字。后续轮次包含全部已执行动作的官方反馈，不包含上一轮图片或模型输出。官方策略和 3 个文本示例均保留；没有自有 Skill、深度、检测框、额外视角、几何工具或演化。

默认配置为 [bare_navigation.json](bare_navigation.json)：base 前 3 个任务，每题最多 20 个环境动作、20 次模型调用，总请求上限 60；temperature=0、max_tokens=4096、seed=137、enable_thinking=false。4096 来自固定 revision 的 `remote_model.py` 常量。首轮建议 1–2 个动作、之后建议 5–6 个动作，遵循官方原提示；执行时按剩余环境步数截取计划。

模型请求在 Mac 发起，仿真在 4090 服务器的独立目录 `/home/cvailab/wangjh/embodiedRunner` 和 `venv-nav` 内运行，使用物理 GPU 3。模型密钥仅在 Mac 进程内从已有私有凭据文件加载，不传给服务器。

```bash
../.venv/bin/python -B ssh_session.py
../.venv/bin/python -B bare_run.py check
../.venv/bin/python -B bare_run.py run
```

`check` 不请求模型，也不启动 Unity。实际渲染必须另外通过 `remote_runtime.py smoke` 的真实 reset/step 检查。默认 run 目录必须不存在，重复测量使用 `--output-dir runs/<新目录>`。

远端最小部署包含 `upstream`、`official_sources.json`、`protocol.py`、`bridge.py`、`simulator_worker.py`、`remote_runtime.py`、`setup_remote.py`、`ipv4_pip.py`、`setup_vulkan.py` 和 `requirements-nav.txt`。使用远端已有 Python 执行 `setup_remote.py` 安装独立环境；安装入口仅在本进程跳过服务器的慢 IPv6 路径。若系统缺少 `vulkaninfo`，`setup_vulkan.py` 下载并解压 Ubuntu Focal 的固定版本工具包到任务内，不修改系统包。依赖、Unity 构建、临时文件、XDG 配置和 Unity 日志均位于 embodiedRunner 下。Unity 构建缓存集中在 `cache/ai2thor`，同目录内的预检和 pilot 可以复用。

与官方完整入口的差异已写入每次启动和结果：API 使用 JSON-object 模式替代 OpenAI SDK 的 parsed JSON schema；明确设置 seed/thinking；非法 JSON 或动作结束当前题，不随机补动作；服务及仿真故障停止，不无限重试。动作执行只使用 `action_id`，与官方一致。`uses_official_prompt=true`、`uses_official_messages=true` 表示输入协议复现；`uses_official_default_planner=false` 表示没有直接运行包含 SDK 和随机兜底的完整官方规划器。

首次下载 Unity 构建时，可先在远端执行 `prefetch_unity.py`。它并行下载固定官方构建的 HTTP ranges，并完整读回核对官方 SHA-256；归档放在 `cache/unity-download`，环境只在内容 hash 与官方值一致时复用。构建获取改用 HTTPS，场景与执行逻辑保持官方版本。

修改副本配置中的 `n_shots` 为 0 可以运行官方提示的零示例变体；官方 Strategy 仍然保留。零示例与默认 3 示例结果应分别记录。

## 原始接入入口

已实现可连接官方 EB-Navigation 环境的独立入口。现有 Qwen 客户端负责模型请求，独立 Python 3.9 子进程负责仿真，两个进程通过标准输入输出交换经过字段筛选的观测和动作。

首版提供 `direct` 和 `skill` 两种规划方式。`skill` 将本目录的完整导航方法正文注入模型；这份方法是手工初始化的策略。三维工具、受限程序求解和 Skill 自动演化尚未接入，因此当前实现应记为“导航闭环与 Skill 注入初版”，不能记作完整 harness3D 方法的具身实验。

## 本机检查

在 `embodiedRunner` 目录运行：

```bash
../.venv/bin/python -B run.py check
```

检查会读取固定官方数据和代码、核对仿真 Python 的依赖、检查模型凭据是否配置；不会请求模型或启动仿真器。结果写入 `checks/startup.json`。`prerequisites_missing` 表示检查完成但运行条件未满足。

当前默认清单为 `base` 子集前 3 个官方任务：最多 60 次模型请求，每个任务最多 20 次，每个计划最多 2 个动作，图像分辨率 500，输出 2048 tokens，seed=137，temperature=0，thinking 关闭。配置在 [navigation.json](navigation.json) 中；该清单是接入 pilot，不能代替 300 任务全量结果。

## Linux 仿真环境

推荐将本目录与现有 `src/skill3d` 放在同一 Linux 工作区。模型侧继续使用项目 Python 3.11 环境，仿真侧单独创建 Python 3.9 环境。

仿真 worker 的最小 Python 依赖如下，安装在独立环境内：

```bash
/path/to/embench_nav/bin/python -m pip install \
  ai2thor==5.0.0 gym==0.26.2 numpy==1.26.4 \
  Pillow==10.4.0 opencv-python==4.10.0.84
```

AI2-THOR 的 Unity CloudRendering 还需要可用的 Linux Vulkan/显示驱动；Python 依赖检查不会验证实际渲染。首次启动可能下载 Unity 构建，缓存被指定到本目录的 `cache/ai2thor`。需要实际 `reset` 和 `step` 成功后才能确认仿真可用。

模型凭据由模型侧进程从 `DASHSCOPE_API_KEY` 临时读取。凭据不写入配置或结果，仿真子进程会移除继承的模型密钥、token 和服务 endpoint。

配置好后先检查：

```bash
../.venv/bin/python -B run.py check \
  --simulator-python /path/to/embench_nav/bin/python
```

`ready` 表示固定输入、依赖和模型凭据已通过启动检查，仍需实际运行验证渲染和模型服务。

## 运行 pilot

```bash
../.venv/bin/python -B run.py run \
  --simulator-python /path/to/embench_nav/bin/python \
  --output-dir runs/navigation-pilot
```

`run` 会执行真实模型请求和真实仿真动作。输出目录必须位于 `embodiedRunner` 内，而且必须是新目录；已有结果不会被覆盖。首版不支持从动作中途恢复，服务或仿真故障后需要使用新目录重新运行。

模型侧可通过 `code_root` 使用项目 `../src`，也可指向 `../VSIbench8Brunner/src` 来复用便携副本中的客户端。`upstream_root` 默认是本目录已固定的官方导航源码与任务清单，启动时检查源码和数据 hash。官方环境的动作及成功条件保持原定义。

`mode=direct` 不读取或注入 Skill；`mode=skill` 使用 `skill_path` 中的正文并记录内容 hash。两种方式使用本项目的初版规划提示，没有复现官方默认规划器的 few-shot 示例。结果记录 `uses_official_default_planner=false`，比较时需固定模型参数和其他输入条件。

## 输出与故障处理

| 输出 | 内容 |
|---|---|
| `startup.json` / `config.json` | 固定输入、代码和 Skill 身份及配置 |
| `episodes/*.json` | 每个登记任务的状态、成功标记、环境步数、请求数与错误码 |
| `trace.jsonl` | 请求/图像/响应/Skill hash、合法动作 ID、数值 token 用量 |
| `summary.json` | 总体及各子集任务成功率、完整分母、失败/未执行计数、停止原因 |
| `cache/` / `tmp/` | 仿真及第三方组件的缓存和临时文件 |

输入指令、模型推理正文、原始请求、参考目标位置和密钥不写入模型 trace。图像通过内存 PNG 传输，规划器仅收到当前第一视角图像、指令和动作反馈；目标位置、评分距离、成功标记和完整 simulator metadata 不进入规划提示。

模型动作必须满足 ID 0–7、名称与 ID 对应、计划非空且不超过剩余预算。格式错误结束当前任务，不使用官方解析器中的随机动作替换。碰撞等环境无效动作会放弃计划余下动作并重新观察。模型服务故障、观测身份错误、仿真超时和运行总请求预算耗尽会停止整个 run，不自动重试；余下任务记录为 `not_run` 并保留在分母。

`summary.status=incomplete` 的成功率包含未执行任务的零值，属于运行诊断，不应当作完整评测结果。任务终止与成功仍由官方环境决定，20 步上限与 1 米水平距离阈值保持不变。

## 离线验证

```bash
../.venv/bin/python -B -m pytest test_navigation.py -q \
  -p no:cacheprovider --basetemp=test-output/pytest-tmp
```

测试使用明确的工程夹具，覆盖动作校验、预算、数据隔离、碰撞重规划、故障退出、进程通信，以及官方动作映射和成功判定代码；测试结果不构成 benchmark 成功率。测试目录和日志集中在 `test-output`。
