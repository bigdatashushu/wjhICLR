# v11 episode 执行池：16 / 32 路验证

`SystemVariantRunner` 新增可选 `episode_pool` 调度器，最多启动 32 个 episode 执行进程。Skill library 的 owner 数量单独冻结，默认 4 个；owner 是协调器中的版本管理对象，不再限制在线解题并发数。

## 执行与学习边界

- 每题由执行池中的进程在主线程执行，继续使用原有在线 runner、工具授权和 SIGALRM sandbox。
- 每题派发前冻结 Skill 内容 hash 和 snapshot identity，结果、trace、work、recon 使用原有独立 variant 目录。
- 同一 lineage 可并行执行当前批次的剩余题目。结果按 `scene_id`、`variant_id` 顺序入库。
- 第 12 条新经验触发既有 reviser、双 seed parent/candidate A/B、严格准入、原子发布和正常加载验证。完成后才能派发该 lineage 的下一批。
- 协调器串行处理修订和发布。此时暂停新的在线题目派发，已启动题目继续完成，gate episode 使用共享 episode permit 并行执行。因此这版不承诺所有阶段都持续占满 32 路。
- 在线 executor 不接收 owner、label metadata、revision factory 或 gate factory。原有 online label isolation 和 prepared-binding 检查继续生效。

此改动保留 fit-set 入口已有的槽位格式；没有为并发新增 Skill slots。

## 全局请求额度

`RequestGovernor` 对 solver、reviser 和所有 gate arms 使用同一个运行目录。请求发出前先持有 OS file-lock permit，再持久化额度预留记录。

| 参数 | 含义 |
| --- | --- |
| `--max-requests` | 整个 run 的请求预留上限；新调度器必须显式提供 |
| `--request-concurrency` | 全局正在请求 API 的上限；0 使用 workers |
| `--requests-per-second` | 全局请求启动频率；0 不限频率 |
| `--episode-concurrency` | 在线和 gate episode 的共享上限；新调度器中 0 使用 workers |
| `--skill-owners` | 固定 library owner 数，独立于 workers |

预算耗尽按取消运行处理，绕过模型故障恢复，不进入 Skill 方法失败或 A/B 拒绝统计。进程死亡后 OS 自动释放 permit，但未收到完成记录的请求仍占用累计额度，不自动重放。摘要区分 active、completed、error 和 unknown，并汇总可测的 token usage。日志不记录 prompt、凭据或异常正文。

## 恢复

使用相同代码、输入、scheduler、workers、owner 数和请求策略才能 `--resume`。学习选择必须是每条 lineage 的前缀，不能跳过早先题目后继续积累。

完成结果不重新执行。乱序完成的结果在缺失的早先题目完成后按冻结顺序入库。其他 lineage 推进共享 owner snapshot 时，未完成派发仍绑定原来的 snapshot，且必须证明方法 hash 未变。

已有 incomplete trace 或无完成记录的模型请求沿用既有 fail-closed 规则。失败和中断后停止新派发并排空当前进程，保留 `scheduler_receipts/` 历史；`episode_pool_receipt.json` 仅作为最新摘要。

## 使用

在 `scripts/run_fitset_v11.py run-system` 现有输入参数之外，16 路验证使用：

```text
--scheduler episode_pool
--workers 16
--skill-owners 4
--episode-concurrency 16
--request-concurrency 16
--max-requests 1000
```

先用 `--preflight-only` 校验冻结输入，再对同一输出目录使用 `--resume` 运行。32 路必须使用新的输出目录和 manifest，将 workers 和两个 concurrency 参数一起改为 32。上述 1000 是额度示例，应按冻结 panel 的请求上界配置。

新一轮真实学习验证应预先冻结 induction-only panel、prepared bindings、32 帧输入、Skill seed、模型、工具、预算和 A/B 配置。旧 fit-set 入口允许同题反馈，不代表本次真实验证可以把 validation/test labels 加入 revision。

## 已验证与待验证

本地测试用真实 `RestrictedNamespaceKernel` 和 spawned processes，输入与模型响应采用确定性夹具。16、32 路均实测达到目标进程并发；每种配置完成 39 条学习题目和 96 条双 seed gate 题目，三条 lineage 均在第 13 题使用经严格准入的新版本。中断排空、恢复去重、配置变化拒绝、失败停止和全局额度竞争有独立测试。

这证明调度、版本边界和工程隔离，不证明真实模型质量或 API 稳定性。2026-10-23 的三节点只读核验显示操作环境未加载 `DASHSCOPE_API_KEY`，也没有运行中的 API 网关；本轮真实 Agent/API 请求数为 0。真实 16 → 32 路验证仍待接通授权 API 凭据及冻结输入。

在已安装项目依赖的 Python >=3.11 环境中运行：

```text
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest -q \
  tests/unit/test_episode_pool_v11.py \
  tests/unit/test_request_governor.py \
  tests/unit/test_fitset_v11.py \
  tests/unit/test_fitset_stream_v11.py \
  tests/unit/test_streaming_v11_cli.py \
  tests/unit/test_v11_streaming_evolution.py \
  tests/integration/test_streaming_evolution_contract_v11.py
```
