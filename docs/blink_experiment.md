# BLINK 主实验入口

`scripts/run_blink_experiment.py` 将成功直答评测使用的原图和 thinking 设置接入
BLINK 的代码求解、Skill 学习和父候选 Gate。实验在 Titan 上读取已发布缓存，
不重新调用检测、分割或 MoGe2。

## 实验和输入

入口使用原四数据集实验的 **40 题训练、93 题测试**。这 133 题来自官方
`Multi-view_Reasoning/val`，保持原题单、图片顺序和完整任务定义。
官方另一个 133 题的 `test` 答案隐藏，不进入这次内部 40/93 实验。

按顺序执行：

1. `direct`：Qwen 原图直答，不提供 Skill 或工具。
2. `no_skill`：同一代码求解器和四工具，不提供 Skill。
3. `initial`：同一求解器与初始 Skill。
4. `training`：在 40 题上积累轨迹，每 12 条触发一次修订。
5. `freeze_skill`：保存最后晋升的 Skill；末尾 4 条不追加修订。
6. `final`：冻结训练后 Skill，执行相同的 93 题测试。
7. `score`：全部测试预测及 Skill 冻结后，宿主读取测试答案统一评分。

修订沿用原少图实验的聚合反馈方式：模型看当前 Skill、四工具清单、
正确数/合法数及失败状态统计，不能看到逐题 GT 或题目。
每个候选从训练题中以固定种子抽 8 题，父候选在 seed 137/138 下逐一执行。
**两个 seed 都必须**正确数严格增加，合法答案数不降，程序错误、终态运行错误和
拒绝授权次数不增才晋升。`no_change` 和无效修订保留记录并等待下一批轨迹。
Gate 核对实际请求 seed、交付 Skill 哈希、输入、缓存、模型及配置；
返回的后端指纹存在时核对一致，提供商缺失指纹时记录
`backend_fingerprint_observed=false`，不宣称已验证不可观测的后端实例。

## 请求设置

配置：`configs/experiments/blink.json`。

| 请求阶段 | thinking | 最大输出 token | 响应格式 |
|---|---|---:|---|
| 直答、代码求解、最终提交、父候选 Gate | 开启 | 8192 | 提示词约定；不发送强制 `json_object` |
| 类别提取（缓存无现成类别时的独立求解入口） | 关闭 | 256 | `json_object` |
| Skill 修订 | 关闭 | 2048 | `json_object` |

模型为 `qwen3.8-flash`，temperature=0，默认 seed=137。
客户端和求解器输出上限同时为 8192，首次请求前拒绝客户端截断配置。
直答、普通求解和终结轮都按原顺序发送原图字节及真实 MIME，不重新 JPEG 编码。
所有方法都使用单图独立相对几何，四工具不提供跨图位姿或米制授权。

预算对所有阶段共享，`max_requests=0` 表示尚未指定可执行预算。
当前调度每次启动一个 spawn 执行进程，在其主线程执行沙箱；实际请求串行。
`request_concurrency=4` 是共享请求额度的上限，不表示本入口当前有四路活跃请求。
直答每题一次，系统每题最多两个代码轮次加一次终结请求；成本报告分别统计请求和 token。

终结和直答只从 JSON、明确最终选项或答案声明提取答案，不执行模型返回的 JSON 内容。
普通求解仍经原 Python 组装、AST 检查和沙箱；`reasoning_content` 单独保存，
不拼接进 Python 程序。

## 零请求检查

在 Mac 仓库根目录预览，不读取服务器输入：

```bash
.venv/bin/python scripts/run_blink_experiment.py --dry-run
```

在 Titan 的代码 checkout 中使用绝对解释器和已有 CPU OpenCV 兼容路径：

```bash
PYTHONPATH=/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/cpu_compat \
/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python \
scripts/run_blink_experiment.py --check
```

`--check` 是默认模式。逐题核验全部原图、输入哈希、缓存文件与内容身份、初始 Skill，
不读取 GT 文件内容、不访问模型服务、不启动 GPU 推理。
GT 文件在此阶段只记录文件身份；实际执行的宿主另冻结文件内容哈希。

默认服务器输入位置：

```text
/home/wangy/wangjh/harness3d/four-datasets-20261031/smallimage/
├── manifest.json
├── prepared/<input_sha256>/{metadata.json,arrays.npz}
├── seed_skills.json
└── host_evaluation/answers.json
```

原图路径保持 `/tmp/h3d-four-smallimage-public-20261031/dataset/BLINK_images/`。
更换路径须新配置、输入身份和运行，不能改写原缓存身份。

## 执行与续跑

正式运行前源码需已提交且干净；脚本只复制 Python 包、实验入口和配置文件，
不复制数据、密钥或已有结果。代码保存在新运行目录的 `code/source/`，
由冻结入口实际执行。当前修改阶段不会自动提交、同步服务器或运行模型。

准备好实际预算和新输出目录后：

```bash
python scripts/run_blink_experiment.py --execute \
  --max-requests REQUEST_BUDGET --output-dir /path/to/new-blink-run
```

`REQUEST_BUDGET` 是需要替换的请求数，不是默认授权。
凭据由现有私有环境注入 `DASHSCOPE_API_KEY`，入口不读取或复制密钥文件。
本入口通过配置的百炼地址调用；未来接入其他网关须独立冻结并核验真实服务身份。

```bash
python scripts/run_blink_experiment.py --resume --check --output-dir /path/to/blink-run
python scripts/run_blink_experiment.py --resume --execute --output-dir /path/to/blink-run
```

续跑加载运行目录中的原源码和原预算，不要求再次手填预算。
显式传入的配置或预算与原记录不一致时拒绝执行。
已完成题不重复调用；请求意向存在而没有结果、服务失败或结果身份改变时停止，
保留请求记录，不自动重发。

## 输出

| 路径（相对于新运行目录） | 内容 |
|---|---|
| `manifest.json` | 冻结源码、配置、公开题单、缓存、初始 Skill、宿主标签文件身份 |
| `results/<stage>/` | 逐题预测、原始回答、推理内容、解析方式、完整 usage、程序及身份 |
| `requests/<operation>/` | 每次请求的意向、实际响应及内容哈希 |
| `budget/state.json` | 总请求预算及安全计数，原始响应不放入预算账本 |
| `skills/attempt-*/` | 修订、配对 Gate 和晋升决定 |
| `skills/frozen.json` | 最终 Skill、修订次数、尾部轨迹数 |
| `completion.json` | 四臂完整分母得分、成本、初始最终差值及全部结果文件哈希 |

非法答案和程序失败留在分母中；训练和 Gate 只用训练反馈。
原 `BLINK/evaluate.py` 和历史直答结果保留原用途，新入口不接管旧运行。
客户端实现有更新，历史运行的冻结源码哈希仍以原 manifest 为准。
