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
| `--request-concurrency` | 全局 client.chat 调用上限，RPC 模式指网关调用；0 使用 workers |
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

先用 `--preflight-only` 校验冻结输入，再对同一输出目录使用 `--resume` 运行。32 路必须使用新的输出目录和 manifest，将 workers 和 episode-concurrency 改为 32。request-concurrency 按 direct API 或 RPC 网关的冻结容量配置。上述 1000 是额度示例，应按冻结 panel 的请求上界配置。

新一轮真实学习验证应预先冻结 induction-only panel、prepared bindings、32 帧输入、Skill seed、模型、工具、预算和 A/B 配置。旧 fit-set 入口允许同题反馈，不代表本次真实验证可以把 validation/test labels 加入 revision。

## 已发布 r04 输入

`--published-inputs MANIFEST` 使用 `v11-published-fitset-inputs/1` manifest，
包含原 catalog entry、独立解包目录和已核验的 `scene.json` hash。
它与旧的 `--bindings`、`--direct-run-root`、视频来源参数互斥。
正常 frozen-scene loader 会检查 32 张实际 PNG/RGB、帧身份及旧几何依赖；
binding loader 通过原 migration receipt 检查完整逻辑身份和全部产物。
panel 的 `artifact_ref` 继续保存原 sealed path，实际访问位置单独记录。
只有 runtime artifact 对象使用经过验证的本地依赖引用；不改写 artifact JSON 或 sealed hash。

该 provider 始终返回空 ground truth，不读取离线 metadata。
在线和 gate 执行进程均不接收 coordinator 的 label metadata；
metadata 仅由 coordinator 用于既有评分与 sanitized feedback。
M5 binding seed 从 sealed request 读取，独立于 solver 和双 seed gate 的采样 seed。

induction panel 每槽位至少保留 13 题和 8 个不同场景。
`--scheduling-mode batch_v1` 保证第 12 条新经验到达修订边界；
reviser 仍可返回 no_change，严格 A/B 未提升仍正常拒绝。

## RPC 请求容量

solver、reviser 和 gate factory 都可显式使用 `--perception-endpoint ORIGIN`，
只要求执行节点的既有 RPC token；上游 API key 留在服务节点。
`--perception-capacity` 冻结真实网关容量，在线启动前检查健康、模型身份和容量。
现有网关硬上限是 8，因此经该网关执行 16/32 路 episode 时应冻结
`--request-concurrency 8 --perception-capacity 8`。
run governor 记录 RPC 请求并发；它不能单独证明上游 API 的实测并发。
真实报告应分别提供 episode 执行区间、RPC 请求区间和服务节点上游调用证据；
缺少上游证据时必须记录为未测量。

网关在私有 SQLite 的 `upstream_audit` 表记录 `client.chat` 开始/结束、
active/peak、成功/错误/unknown 和安全 token usage；health 暴露汇总计数。
缓存回读和重复 RPC 不增加上游调用计数，重启时未完成记录转为 unknown，
不补造结束时间或重放模型调用。该记录衡量服务端 client 调用区间，
不证明提供商内部执行区间；调用前持久化 start 后崩溃也只能记 unknown。

## 已验证与待验证

本地测试用真实 `RestrictedNamespaceKernel` 和 spawned processes，输入与模型响应采用确定性夹具。16、32 路均实测达到目标进程并发；每种配置完成 39 条学习题目和 96 条双 seed gate 题目，三条 lineage 均在第 13 题使用经严格准入的新版本。中断排空、恢复去重、配置变化拒绝、失败停止和全局额度竞争有独立测试。

这证明调度、版本边界和工程隔离，不证明真实模型质量或 API 稳定性。2026-10-23 的三节点只读核验显示操作环境未加载 `DASHSCOPE_API_KEY`，也没有运行中的 API 网关；本轮真实 Agent/API 请求数为 0。

真实输入回放发现原 39 题分散在六条 lineage，每条均不足 12 条。
在任何模型请求前，按公开 binding 的 `m4_unavailable` 诊断及场景/QA 顺序，
调整为计数、距离、物体尺寸各 13 个不同场景的 induction/base 输入。
39 题已通过正常 loader、实际 32 帧字节、逻辑输入身份及原 binding migration 校验，
均为 `g0m0`；调整过程不使用答案或分数。
该清单具备触发三条学习批次的输入条件，验证范围限定为 3D 不可用时的视觉路径。
独立运行目录还需通过正常 CLI preflight；真实 16 → 32 路执行仍待授权 API 环境。

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
