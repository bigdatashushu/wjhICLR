# 逐轮快照实施状态

用户已批准方案一，并在实施期间要求所有更改尽可能集中在本目录。上下文预算见 `IMPLEMENTATION_CONTEXT_PLAN.md`。

2026-11-05 后续 live 工作验收完成：正式入口、Linux 隔离、三环境 × 三臂的真实首动作联调均通过。综合证据见 [action-live-acceptance-20261105.json](checks/action-live-acceptance-20261105.json)。下文 P0–P4 的零 live/SSH/部署计数记录的是当时离线验收范围，最新结果在本页末尾。

2026-11-06 第四环境 EB-Manipulation 也完成真实首动作接入：Direct/Full **2/2**，当前实现支持四环境。最新验收见 [manipulation-acceptance-20261106.json](checks/manipulation-acceptance-20261106.json)，具体公开输入、七维动作与感知覆盖见 [接入说明](MANIPULATION_IMPLEMENTATION_PLAN.md)。

## 当前目录调整

- 实现入口改为 `/Users/bytedance/wangjh/wjhICLR/embodiedRunner`。
- `action_runtime/` 是任务专用适配层；共享 `src/skill3d` 不修改。
- 前期隔离worktree保留来源参考，不作为新代码的最终运行入口。
- 当前文件均已在本目录重新实现、导入和测试；以本页最近的验收结果为准。

## 已核对的边界

- 正常step，包括官方动作失败，三个worker均返回RGB和反馈。
- 家务invalid只更新invalid计数，复用旧RGB，不作为真实step。
- 默认single_action：执行合法计划的第一个动作，再重新感知/生成程序。
- 独立动作提交槽，不复用AnswerPayload；动作后撤销旧current结果。
- 本节记录最初离线边界验收：当时没有调用真实模型、感知、仿真或部署。后续正式 live 结果见末节；没有 commit/push。

## 当前已实现并重新验证

- `action_contract.py`：当前RGB/像素/动作表/反馈/记忆身份、不可变快照、独立执行envelope、typed ActionPayload、严格JSON/动作/记忆/证据依赖检查。
- `state.py`：独立step/mutation/observation/evidence版本，合法FSM迁移，mutation后按decision撤销，append-only链式hash迁移记录。
- `action_runtime/`：task-local AST/control/action检查、动作kernel、segment trace、FrozenMethod完整交付和独立动作execution contract。数值安全和工具授权复用共享代码。
- 提交动作后立即终结；不执行后续工具/第二提交；生成程序不能改写host提交槽。每段清空用户变量，保留当前decision工具账本；动作kernel不允许retarget。
- 当前测试：任务合同/执行与共享AST/数值边界/assembler合计83 passed（0.95s）；旧planner选定假输入6 passed、4 subtests passed（0.67s）。一次错误的AST-only测试已改为真实runtime拒绝测试，重跑通过。
- 原preflight保护：230个source hash与103个旧baseline hash全部一致；`git diff -- src/skill3d`为空。没有旧QA源码改动。
- 本目录为实际import/测试来源，不继续使用隔离worktree作为实现路径。

## 进程隔离与预算

- kernel仅是独立受限worker内部的执行组件；OS隔离由worker内seal与父进程资源监测承担。
- `sandbox_wire.py`：不使用pickle；从字段白名单重建快照并复核identity。只接纳当前原图对齐的数值数组和候选公开字段；候选表面/几何标志强制未验证。
- `sandbox_process.py` / `sandbox_worker.py`：每段全新、无凭据的Python进程，关闭继承及运行期非通信FD；可信依赖加载后、生成代码执行前使用macOS seatbelt封锁文件读写、网络和fork/exec。封锁失败不执行程序，不返回动作。
- `sandbox-exec`与二次`sandbox_init`在本机无法叠加（EPERM）；最终使用worker内单次seatbelt。依赖加载阶段只运行可信代码，不能把该阶段描述为已禁止文件访问。
- 父进程以20ms采样监测驻留/footprint内存并终止owned进程组，CPU/墙钟/64KiB输出限制已实测。内存采样允许瞬时超额，不声称逐次分配的硬内存上限。
- `action_tools.py`：结果绑定episode/decision/observation/evidence/snapshot；有界ReadObservation，完整实际依赖复核，成功/拒绝工具调用均保留去参数化授权回执，类别确认不能授权candidate表面深度。
- `budget.py`：run/episode共享permit-before-call，Qwen普通/修复/类别解析共用子cap，类别核验计入全部模型cap，DINO/SAM/MoGe逐次预留且每当前view至多一次。失败不退回请求额度，usage仅保留数值并注明缺失，token观察阈值是软上限。
- sandbox执行前预留完整剩余工具/reader tranche；仅验证完整worker响应后退回未用部分。无法取得完整执行结果时保留全额。AST拒绝也保留前段结果，父进程复核之前账本没有丢失。
- timeout、OOM、输出及工具cap立即终止；普通语法/运行/工具参数错误与yield可在同decision预算内继续。各段新进程、同decision账本累积。
- 本阶段最初验收没有真实服务、仿真或部署；当时 Linux 入口拒绝执行。后续已实现并验证 Linux 隔离，见末节。

## 宿主、V2与三个环境

- `mutation_protocol.py`：严格UUID/digest/session/episode/revision/step/action table/参数校验均在mutation前；同UUID相同内容返回原ACK，内容变化拒绝。缓存仅覆盖当前worker存活期间；异常发生在mutation边界后即poison，不自动重发或reset。
- `action_bridge.py`：单个在途请求、非阻塞有界NDJSON读写、整条命令deadline、逐ACK身份复核。ACK丢失/类型或hash不符时停止，保留已预留动作预算。
- `action_adapters.py`：新Navigation/Household adapter复用未改动的 `OfficialNavigation` / `OfficialHousehold` 接口；原worker/Direct基线入口未更改。统一私有评分packet，经白名单转为公开snapshot。
- `action_worker_v2.py`：独立opt-in V2入口。离线fake支持真实NDJSON子进程；官方模式还需部署preflight，不默认启动。
- `action_host.py`：三个环境、Direct/B01/B11共用single_action FSM；合法计划只执行首动作，连官方失败或RGB重复也撤销旧current结果并产生新decision。家务格式invalid只增加mutation/invalid，旧RGB仍创建新观测修订。
- 新增P2测试62 passed（34.17s）；P1及共享AST/v11 execution boundary/assembler回归97 passed（24.35s），合计159 passed。包括三个真实fake worker进程、三环境×三臂真实sandbox闭环、重复/stale/丢ACK、非法尾动作、变量清空、yield/reader、执行cap和账本保留。
- 上述fake请求使用同permit计数，但不连接真实模型；0真实模型请求、0感知服务调用、0官方仿真启动、0部署。

## P3当前证据、完整方法与入口

- 新增 `evidence.py`：共享SmallImage内存decode接线，固定公开词表与单候选配置，DINO组合prompt一次请求，backend每次内部调用先预留，服务故障立即停止；当前scope与最终证据修订/hash全绑定，数组只读，mutation后重新感知。
- 新增 `methods.py` / `methods/manifest.json` / 三个完整正文，B11普通/修复请求全文交付；B01同工具合同无方法。
- 新增 `action_solver.py`：一次Qwen completion、严格sole-program_source JSON和既有assembler，Direct复用official prompt/examples/parser。没有构造真实client。
- `action_run.py`支持零请求check与三个环境×三臂fake。正常/修复/yield均同budget；yield新增有界公开结果页回传，同agent可见，宿主reader读取也计费。
- P3接线检查47 passed（44.66s），含18项新增测试和29项宿主/预算回归。三环境三臂真实prompt/sandbox闭环全部完成；Direct家务image callback参数名兼容问题已修复并重跑。
- `--check`通过：0真实模型/感知请求、0官方仿真启动、0部署。详见 `P3_INTERFACE.md`。

## P4固定分母与最终离线验收

- `action_identity.py`：开跑前冻结355个显式代码/方法/manifest文件身份，包含任务实现、共享源码与官方代码；另外冻结协议、执行合同、模型/预算/感知/方法配置及运行时版本。不开模型、服务或仿真。
- `action_audit.py`：独立只读CLI，核对固定选择与三臂分母、缺题/重复题、源码/配置身份、初始公开观测、完整方法交付、request/usage/cap、FSM链、mutation/ACK/版本、逐轮撤销和Direct零工具/感知。每题与总报告均有完整性hash。
- 合法端点故障、服务故障和未知动作ACK保留在原分母；服务偏离与错误usage拒绝审计。新增端点启动失败报告，记录失败而不跳过题目。准备失败的snapshot也保留并撤销，失败请求仍计费。格式解析失败保留已返回的数值token usage。
- 最终全套：本目录tests加共享AST/v11 execution boundary/assembler回归 **207 passed（88.22s）**；其中P4新增30项审计/故障测试。原planner选定假输入回归 **6 passed（0.35s）**。
- 真实NDJSON fake worker综合：三个环境 × Direct/B01/B11共 **9/9** 到达 `official_done`；18个fake环境动作、18次fake completion、12段真实隔离程序执行。保存后用独立CLI复核，**3对/9次审计通过**。fake成功率仅证明闭环，不代表benchmark质量。
- 保护检查：**230个原源码、103个旧基线文件hash均不变，active pointer不变**；共享 `src/skill3d` diff为空。没有commit、push、清理、SSH、部署或真实请求。
- 使用说明：[ACTION_USAGE.md](ACTION_USAGE.md)。综合收据：[action-v2-p4-acceptance-20261105.json](checks/action-v2-p4-acceptance-20261105.json)。完整fake报告：[action-v2-p4-process-run-20261105.json](checks/action-v2-p4-process-run-20261105.json)。独立审计：[action-v2-p4-paired-audit-20261105.json](checks/action-v2-p4-paired-audit-20261105.json)。原文件保护：[action-v2-p4-preservation-20261105.json](checks/action-v2-p4-preservation-20261105.json)。

## 交付范围

P0–P4离线实现和验收完成，所有新增实现、测试、方法库、说明和最终收据位于本目录。正式入口另增加 `--live-check` / `--live`；Linux namespace/seccomp 已在服务器实测。当前几何仅是当前视角相对证据，不提供跨视角米制距离或完整地图。真实联调的结果与任务成功分别报告，见下一节。

## Linux 与正式联调

- `sandbox_linux.py` 和进程树/峰值内存验收已实现。最新 r06 服务器隔离复验 **14 passed、1 skipped**（macOS 专用项），[测试回执](checks/linux-isolation-final-20261105.xml)。
- `action_live.py`、`action_live_launch.py` 和 `config/action-live-4090.json` 接通真实 Qwen、受保护检测、SAM2、Intern 类别审核、MoGe2 与官方 V2 worker；无自动重试、无 fake 回退。
- 三个官方 runtime probe、真实服务和隔离 seal 的零推理预检通过；本地独立核对当前源码、配置及身份也通过：[预检回执](checks/action-live-final-preflight-20261105.json)。
- 首批 r03 失败历史保留在 [action-live-first-attempt-20261105.json](checks/action-live-first-attempt-20261105.json) 和原独立部署。已修正官方 invalid 与 host 格式-invalid 的计数语义，并补充预注入 API、扁平 runtime ctx、字符串动作表说明和有限语法拒绝分类；冻结的新版本部署到 `/home/cvailab/wangjh/embodied-action-v2-20261105-r06`。
- r06 真实首动作联调 **9/9 通过**：Navigation、ALFRED、Habitat 各自的 Direct/B01/B11 均执行一个官方 step，全部取得 ACK，0 abort、0 程序错误。9 次 Qwen 返回 3 个合法 Direct 计划和 6 个合法程序；6 个生成程序均在新的 Linux 隔离进程中执行。实际推理共 **25 次**：Qwen 9、DINO 6、SAM2 2、Intern 类别审核 2、MoGe2 6。
- [真实运行回执](checks/action-live-final-run-20261105.json) 在服务器审计通过，下载后由本地 [独立审计](checks/action-live-final-audit-20261105.json) 再次通过，核对了当前完整源码、配置、方法、请求/预算、mutation/ACK 和固定 3 对 / 9 次分母。
- 本次到达的是首动作联调上限，完整任务成功率尚未评估。ALFRED 三臂均取得官方进度 1/3；Habitat 的 B01/B11 各有一次官方动作失败，仍正常返回新 RGB/反馈并记录 official invalid，未误计为 host 格式-invalid。生成程序在本次 smoke 中没有调用工具或 reader，不能据此声称这些分支完成了真实模型调用验收；对应执行路径由离线闭环和隔离测试覆盖。
- Navigation 的独立初始渲染 hash 不同；ALFRED/Habitat 的初始观测一致。本次 `paired_comparison_eligible=false`，不用于方法效果比较。当前 pilot 资产范围仍是 ALFRED base 前三题和每个全新 Habitat worker 的 index 0。
- 最终完整离线回归 **245 passed、1 skipped、4 subtests passed**，[测试回执](checks/action-live-final-regression-20261105.xml)。原源码/旧基线保护再次通过：230 / 103 个文件无变化，active pointer 不变，[保护回执](checks/action-live-final-preservation-20261105.json)。没有 commit/push。
- 最终验收收据：[action-live-acceptance-20261105.json](checks/action-live-acceptance-20261105.json)，绑定上述证据文件的字节 hash 和本次实现源码身份。

## 第四环境 EB-Manipulation

- 独立固定官方源码、`base:000` 资产及 CoppeliaSim/PyRep runtime；原三个环境和共享源码保持原样。七维体素位置/Euler/夹爪动作与官方匿名坐标、官方标注图已接入统一观测、mutation、沙盒和审计。
- 0 模型真实 reset/step 成功；最终 r04 零推理预检通过。Direct/Full 各用 1 次 Qwen 并执行 1 个真实官方动作，均返回成功反馈与新观测，Full 的程序隔离 seal 和执行通过。共 3 次真实推理（Qwen 2、MoGe 1），0 abort、0 程序错误。
- [真实运行](runs/manipulation-smoke-r02.json)、[本地运行审计](checks/manipulation-live-audit-r04.json)、[完整轨迹审计](checks/manipulation-trajectory-audit-r04.json) 全部通过；[轨迹入口](runs/manipulation-smoke-r02-episodes/trajectories.json) 保存完整公开消息、响应、程序、RGB 和预测数组。
- 最新本目录回归 **182 passed、1 skipped**；Linux 七维/进程隔离回归 **30 passed、1 skipped**。当前源码四环境三臂 NDJSON fake 闭环 **12/12** 及其独立审计通过。相关 XML 和 hash 在最终验收收据中绑定。
- 真实验证范围仅为 `base:000` 首动作，任务进度为 0，尚未评测完整抓取放置或全量性能。初始输入相同，首动作预算仍不具备方法效果比较资格。
- 实际检查发现固定词表未覆盖 `star` / `container`，本题检测类别为空：DINO/SAM/Intern 和工具/reader 均未调用。相对深度可读，检测/分割质量未获得本轮验证。[实际图像](checks/manipulation-visual-review-r04.png) 已保留，动态检测留作下一步。
- 本轮所有开发与证据位于 `embodiedRunner/`，原 230 源码、103 旧基线及 active pointer 保护通过，没有 commit/push。
