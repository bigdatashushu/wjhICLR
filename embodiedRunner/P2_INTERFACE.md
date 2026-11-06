# 宿主与V2接口

当前入口均为本目录task-local Python；不修改旧worker、planner、QA代码或旧run。三臂采用相同 `current_only` / `single_action`。

## 宿主接线

1. `RunBudget(run_id, environments)` 的environments必须绑定预选固定任务分母，每arm独立instance与后续budget目录。`episode(episode_id, environment)`仅注册一次。真实Qwen接线可附加该arm自己的 `RequestGovernor`，max_requests等于该run的Qwen cap、max_inflight=1。
2. `MutationSession(adapter)`是唯一worker mutation入口。`LocalTransport`用于离线假环境；`NDJSONTransport`连接独立 `action_worker_v2.py`。`ActionBridge`拿到初始handshake后只允许一个open；每题新worker/session。
3. `ActionHost(bridge, solver, run_id, arm, evidence_preparer, method_sha256)`接收可信solver，后者仅看到不可变snapshot与有界纠错feedback，不得到bridge/scorer。B11必须有冻结方法hash，B01/Direct为空；Direct禁止准备器。
4. `solver.generate(snapshot, feedback=..., usage=...)`每次只允许一个completion，返回 `Proposal(source=assembled_python)` 或Direct的 `Proposal(actions=tuple)`。Host在调用前统一预留synthesis/repair permit；usage槽只接纳prompt/completion/total tokens数值。后续模型client不得重复套用另一份额度，backend内部的真实请求也必须逐次从同EpisodeBudget预留。
5. `evidence_preparer(snapshot, binding, budget)`返回 `PreparedEvidence(prepared, manifest_sha256, capabilities)`；host在synthesis前升级同decision的证据修订，随后整个决策的snapshot identity固定。准备器只从snapshot当前RGB/公开文本取输入，不读method改变类别或配置。P3将实现此接线。
6. Host以独立 `ExecutionEnvelope` 复核payload及所有实际成功工具依赖。每段执行前预留完整工具/reader tranche；完整worker响应的身份、scope、之前账本和计数均通过后才退回未用额度。fatal/未知执行的额度不退回。
7. 程序提交后只执行合法计划首动作。任何step/家务invalid后，旧ResultStore立即revoke，旧decision登记撤销，新观测和新程序从零开始。官方状态仅在host报告中出现。

## 终止与故障

格式/计划错误在家务环境调用一次官方invalid；它不增加step，不产生新RGB，但产生新mutation/观测/decision。Navigation在同快照、同预算中恢复格式错误。AST/普通程序/工具参数错误和yield保留同decision账本，至多三个请求/三个执行段；纠错feedback仅含固定code、line和yielded IDs，不回传源码/stdout/服务异常。

执行timeout/OOM/输出/tool cap停止，不再申请模型补答。模型/感知服务故障、V2断连/ACK不确定或worker异常立即终止；没有自动retry/reset/replay。总tokens阈值只在收到usage后阻止后续请求，最终一次允许跨过软阈值。

V2去重仅限当前worker存活期间。缓存同UUID/digest ACK不会再次mutation；不同digest或新UUID下的过期state会在写之前拒绝。宿主不自动利用重发取ACK，ACK未知直接poison/stop。缓存不支持进程崩溃后的exactly-once恢复。

## 离线证据

2026-11-05新增P2 62 passed，P1/共享边界97 passed，合计159 passed。真实独立进程和OS封锁测试使用假RGB/假工具数值；fake solver占用模拟permit但0真实模型/感知请求，0官方仿真启动。Linux代码执行仍拒绝，等待后续部署隔离preflight。
