# LIBERORunner

`gpt6.1terra` 接入 LIBERO 的基础评测代码。配置中的模型 ID 为 `gpt-6.1-terra`，使用现有 `wjhICLR/xcodeapi` 的 Responses 接口。真实官方仿真已在 Titan Linux 通过预检，包括双路 RGB 和 5 步全零动作。2026-11-09 首次 GPT 请求返回 HTTP 503，未获得可执行的 GPT 动作，动作闭环尚未完成。详见 [试跑报告](checks/smoke-20261109/试跑报告.md)。

本机已建立 `LIBERORunner/.venv`（Python 3.11），可运行离线检查。Titan 上已建立独立 Python 3.10 仿真环境，安装 CPU PyTorch、robosuite、MuJoCo、bddl，并加载已核验的原始官方源码归档。

## 评测流程

实现对应原始 LIBERO 的 `libero/lifelong/metric.py::evaluate_one_task_success`，用 GPT 动作适配器替换原来的训练策略网络：

1. 从原始官方仓库加载任务及 `init_files` 中的固定初始状态，按索引顺序评测。
2. 使用官方 `OffScreenRenderEnv`、Panda、`OSC_POSE` 控制器和 20 Hz 控制频率。
3. 每次 `reset → set_init_state` 后先执行 **5 步全零动作**，不计入策略步数。准备阶段会从源码校验这段逻辑。
4. 评测回合数、最大策略步数和图像尺寸默认读取官方 `eval/default.yaml` 与 `data/default.yaml`，配置中的 `null` 表示不覆盖。
5. 模型每次接收两路 RGB 和机器人自身状态，返回一个 7 维动作；执行一步后再读取新观测。动作不会被裁剪、偷偷重试或自动补正。
6. 以官方环境 `done` 判定成功，并与 `check_success()` 交叉检查。服务故障使运行停止并保持不完整状态。

默认支持 Spatial、Object、Goal、LIBERO-10 四个评测套件。LIBERO-90 可显式选择，保留其预训练套件身份，不与 LIBERO-10 合并。

LIBERO 自身提供的是策略网络评测流程，没有 GPT 官方提示词。本目录的提示词、JSON 动作格式和机器人状态序列化属于新增的 GPT 适配层。它没有训练过 LIBERO 动作策略，其效果需要实际评测。

这里采用 **原始 LIBERO** 流程；OpenVLA/openpi 常用的 10 步等待、50 回合及各套件不同的步数上限属于另外的评测配置。不要混用两种口径。

## 安装

推荐在 Linux 服务器上安装仿真环境，使用独立 Python 3.10 环境。本机 Mac 可以做离线代码检查。安装脚本仅使用原始官方仓库，不会把 PyPI 上 Hugging Face 维护的 `libero` fork 当成原始版本。

在 `wjhICLR` 根目录运行：

```bash
bash LIBERORunner/scripts/install.sh
```

脚本将创建：

- `LIBERORunner/.venv`：独立 Python 环境。
- `LIBERORunner/third_party/LIBERO`：原始官方源码与仿真资产。

可通过 `LIBERO_REF=<官方commit>` 固定新克隆的源码版本；通过 `LIBERO_SOURCE_ROOT` / `LIBERO_VENV_ROOT` 指定独立路径。脚本不会更新已有 checkout。源码、依赖版本、任务 BDDL、初始状态文件及 Runner 代码的身份会写入运行 manifest。

`requirements-sim.txt` 保留官方 NumPy 1.22.4、OpenCV 4.6.0.66、robosuite 1.4.0 等仿真依赖，不安装训练策略所需的 wandb、transformers、robomimic、Hydra；另补齐 robosuite 未声明的 termcolor 依赖。`constraints-sim.txt` 固定 MuJoCo 2.3.7、CPU PyTorch 2.2.2 和兼容的 SciPy/Numba。官方历史安装说明使用 Python 3.8 / PyTorch 1.11；这里采用 Python 3.10 以兼容现有 API 客户端，**该组合已在 Titan 的软件渲染环境通过真实仿真预检**。其他机器仍需先预检。

已有独立 Python 3.10 时，可设置 `LIBERO_PYTHON_BIN=/绝对路径/bin/python3`，不要求安装 uv。网络无法 Git clone 时，可设置 `LIBERO_SOURCE_ARCHIVE=/绝对路径/官方tar.gz`；目前仅接受已核验的官方 commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` 归档（SHA256 `4fd4859991523b2fb0469c346382f5b7b02b69c2f57c05761acf87f747918508`）。归档与逐成员哈希会保留并在运行前校验，不会伪造 Git commit。

无需下载人类演示数据集即可进行评测；需要官方源码中的 BDDL、初始状态和仿真资产。

Titan 当前缺少 NVIDIA EGL 图形组件，本次使用任务目录内提取的 Ubuntu 18.04 OSMesa 20.0.8 软件库，系统驱动保持原样。Mesa 20 的浮点颜色缓冲区不能正常激活上下文，`rendering.py` 在设置 `MUJOCO_GL=osmesa`、`LIBERO_OSMESA_BUFFER=uint8` 时使用 8 位颜色缓冲区。此适配仅作用于当前进程；实际 RGB 仍为 128×128、uint8。渲染后端、缓冲区格式和本地渲染库哈希会写入 manifest，并在运行时核对。

已有远程配置保存在 Git 忽略的 `LIBERORunner/remote.local.json`。工作区的 Run 按钮执行已经验证的远程预检脚本，每次新建一个任务、一个初始状态、最多 3 步的准备目录，模型请求数为 0。显式模型运行使用 `scripts/remote_live.py`，API 密钥仅通过 SSH 标准输入传递到进程内存。首次 503 的目录保留原始状态，不自动重试。

服务器 IPv6 连接不可用时，可对独立安装设置 `LIBERO_IPV4_ONLY=1`；`scripts/network.py` 仅让本次 Python 进程使用 IPv4，不修改全局网络配置。

## 命令

以下命令均在 `wjhICLR` 根目录运行。也可在其他目录设置 `PYTHONPATH=/绝对路径/wjhICLR`。

先做零请求检查：

```bash
LIBERORunner/.venv/bin/python -m LIBERORunner selfcheck
LIBERORunner/.venv/bin/python -m LIBERORunner check
```

`selfcheck` 使用本地测试夹具验证接口，不需要 LIBERO 或密钥。`check` 检查静态依赖与请求序列化；依赖缺失时退出码为 2，且不发送请求、不读取密钥。

准备一个任务、一个初始状态的真实仿真预检：

```bash
MUJOCO_GL=egl LIBERORunner/.venv/bin/python -m LIBERORunner prepare \
  --suites libero_spatial --task-ids 0 --episodes 1 \
  --max-requests 600 \
  --output LIBERORunner/runs/gpt61terra-spatial-t0-r01
```

`prepare` 对每个选中任务实际创建环境、渲染两路 RGB 并执行 5 步全零动作，然后冻结 manifest。**这一阶段模型请求数为 0**。

运行已经准备好的任务：

```bash
MUJOCO_GL=egl LIBERORunner/.venv/bin/python -m LIBERORunner run \
  --run-dir LIBERORunner/runs/gpt61terra-spatial-t0-r01
```

密钥沿用 `XCODE_GPT_API_KEY` 环境变量或现有 `xcodeapi/credentials.local.env`，也可通过 `--credentials-file` 指定符合 xcodeapi 格式的私有文件。不要把密钥写进配置、命令参数、代码或运行报告。

默认请求超时为 3600 秒，不设置可能不兼容的 temperature / API seed 参数；`seed=42` 用于本地环境。服务返回的模型名称必须与请求模型完全一致，否则动作不会执行。

完整四套件准备：

```bash
MUJOCO_GL=egl LIBERORunner/.venv/bin/python -m LIBERORunner prepare \
  --max-requests 480000 \
  --output LIBERORunner/runs/gpt61terra-four-suites-r01
```

这是基础版的一步一请求策略；预算上限很高，实际消耗以轨迹为准。应先验证上面的单任务流程，再安排全量运行。配置默认预算为 600；`prepare` 会冻结选定预算，运行中不能增改。`--episodes`、`--max-steps` 或图像方向覆盖值会记录在 manifest，子集结果不能描述为完整官方评测。

生成或核验统计：

```bash
LIBERORunner/.venv/bin/python -m LIBERORunner report \
  --run-dir LIBERORunner/runs/gpt61terra-spatial-t0-r01
```

`--resume` 只跳过已经完成的回合，允许继续尚未开始的回合。存在服务故障、请求结果不明或未确认的动作时，会拒绝原目录续跑；恢复应建立独立运行并保留原始记录。

## 模型输入输出

模型输入白名单为：

- `task.language`。
- `agentview_image` 和 `robot0_eye_in_hand_image`。
- 机器人末端位置、四元数、夹爪和关节位置。
- 当前控制步编号及上一条动作。

不会把场景物体真值坐标、BDDL 成功谓词、初始状态数组、奖励、仿真内部状态或标签送给模型。默认 `image_orientation=raw` 保持原始数组；GPT 输入可显式选择 `flip_vertical` 或 `rotate_180`，并记录为输入适配变更。

输出格式：

```json
{"action": [0.1, 0.0, -0.1, 0.0, 0.0, 0.0, -1.0]}
```

前三维是归一化平移增量，接下来的三维是增量轴角旋转指令，最后一维 `+1` 关夹爪、`-1` 开夹爪。实际范围与缩放从官方控制器读取。拒绝越界、NaN、布尔值、错误维度和额外控制字段。

## 结果文件

每次实验单独保存于 `runs/<run-id>/`：

| 文件 | 内容 |
|---|---|
| `manifest.json` / `manifest.sha256.json` | 冻结配置、源码和任务身份、初始状态索引、零请求预检 |
| `episodes/<id>/trace.jsonl` | 请求开始、响应元数据、动作解析、动作开始和环境 ACK |
| `episodes/<id>/observations/` | 模型实际使用的 RGB PNG 和机器人状态 |
| `episodes/<id>/result.json` | 该回合结果及控制步数 |
| `episode_results.json` | 按冻结清单汇总的回合状态 |
| `summary.json` | 每套件与总体成功率、完成数、剩余数、协议错误数 |
| `final_collection.json` | 所有回合完成后生成的结果与轨迹哈希清单 |

完整回合分母包含模型非法动作导致的失败；服务故障和预算耗尽会使整批保持未完成，最终成功率为 `null`。临时已完成回合统计为 `success_rate_completed`，不能当作全量成功率。汇总时校验 manifest 身份和动作 ACK 数，避免把已发送但未执行的动作记为执行成功。

## 代码位置与依据

`environment.py` 对接官方任务、环境和控制器；`policy.py` / `observation.py` 实现 GPT 输入输出；`rollout.py` 实现闭环；`runner.py` 管理冻结、日志和结果；`python -m LIBERORunner` 提供命令入口。

核验资料（2026-11-09）：

- 原始官方源码：<https://github.com/Lifelong-Robot-Learning/LIBERO>，重点是 `metric.py`、`env_wrapper.py`、`bddl_base_domain.py` 和 `benchmark/__init__.py`。
- 官方结构与安装文档：<https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/code_structure.html>；<https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/installation.html>。
- 本次实际运行使用原始官方 commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` 的 GitHub 归档，完整压缩包和 1,116 个成员文件均有哈希记录。原始官方源码保持不变。
