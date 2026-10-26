# FullQA 服务器无人值守与固定数据快照

运行对象为 `precompute-fullqa-20261024-r01`，288 场景，基础版与米制版各 5,130 题。操作层独立于冻结算法源码；方法、GPU worker、队列源码保持原哈希。

Titan 的 `fullqa_supervisor.py` 保管派发记录，通过受认证 HTTP 将完整原生租约交付给 3090 的 `fullqa_gateway.py`。3090 在独立子进程调用原 `cluster_worker.run_job()`。输入下载、start/finish 与输出发布遇到临时网络异常时重试同一租约。派发回复丢失根据持久 intent 与当前队列恢复，避免再次 claim。ACK 保留原始字节；完成并审计通过后才领取下一场景。

单个 RAM 槽使用物理 GPU 3。领取前要求 MemAvailable 至少 172 GiB、RAM 文件系统可用空间至少 92 GiB；原生 worker 保留 52 GiB。其余七个网络暂存槽继续使用原服务器控制器。进程守护只重启已绑定源码身份的操作服务，运行中的模型子进程不受影响。

计算进程死亡且没有完整 pending checkpoint、输入或产物身份改变、租约失效、历史 retry、审计错误及归档 attempt 异常均保留现场并隔离。无人值守支持正常运行与临时传输恢复；它不承诺在所有硬件或模型故障下自动成功，也不会靠隐式增加模型尝试来维持进度。

全量完成后原采集器的完整 `main()` 以固定服务端传输适配执行，源码与最终回执字节保持原样；另检查正常缓存、RAM 缓存和网络缓存为空。SQLite 副本使用原 backup API，原采集器在 Titan 操作目录保存精确副本，binder 使用回执内同一绝对路径。归档调用现有 capacity parent 和原 binder，保持完整原验收 gate、归档锁、成功调用边界 anchor、当前文件身份复核。完整 tar 留在 Titan。

运行状态在 Titan：

```text
/home/wangy/wangjh/harness3d/fullqa-server-custody-20261025-v3/current.json
```

3090 custody 状态：

```text
/home/zhouky/wangjh/harness3d/fullqa-server-custody-20261025-v2/gateway_current.json
```

最终成功正文：

```text
/home/wangy/wangjh/harness3d/precompute-fullqa-20261024-r01/receipts/fullqa_unattended_final_binding.json
```

## 已有数据继续实验

快照目录：

```text
/home/wangy/wangjh/harness3d/experiment-snapshots/fullqa-audited-20261025-v1/
```

新建快照时，`scripts/fullqa_snapshot.py` 在创建输出目录前读取 checkout 中的
`src/skill3d/evaluation/published_fitset_v11.py`，并核对逐场景审计回执。
发布身份引用真实 provider 源码路径；缺失源码会在任何快照文件写入前拒绝。
该修复不改写上述已发布快照。

`source_catalog.json` 保存原始已审计目录字节；`source_audit.json` 保存同一轮审计；`published_inputs.json` 保存与其绑定的原引用和实际输入目录；`preflight.json` 成功后说明 32 帧输入与正常 prepared binding loader 实际重放通过。快照不会随着后续预计算自动扩充。

`selected_catalog.json` 同时提供基础版、米制版各 749 题的正常 loader 引用；读取米制版可调用 `cluster_audit.load_catalog_entry(catalog["variants"]["metric"][qa_id])`。在线学习的 input provider 使用基础版。

此快照包含 induction 490 题、inner_validation 259 题，共 13 场景。学习时只选 induction，验证保留 inner_validation 的原分组；最终测试和 outer_holdout 不进入此学习/验证输入 manifest。完整原始目录由操作员保存作为 provenance，不能把它整体交付在线模型。

`PublishedFitSetInputProvider` 的 v2 schema 从已绑定的 `scene_scopes` 推导每场景题数上限，最大 128，并核对原始 catalog 条目。它保持空 ground truth、真实 source split、原 artifact_ref、绑定 manifest 与 relocation receipt。旧 v1 schema 仍限制 induction/base、每场景最多 8 题。

在 Titan 使用已绑定的 provider（不需要改动 source_frozen）：

```python
import sys
from pathlib import Path

sys.path.insert(0, "/home/wangy/wangjh/harness3d/experiment-snapshots/tools")
from inspect_fullqa_snapshot import open_provider

snapshot = Path("/home/wangy/wangjh/harness3d/experiment-snapshots/fullqa-audited-20261025-v1")
provider = open_provider(snapshot)
entries = provider.entries
learning = {qa: row for qa, row in entries.items() if row["split"] == "induction"}
validation = {qa: row for qa, row in entries.items() if row["split"] == "inner_validation"}
```

只检查快照或重放指定题目（零模型请求）：

```bash
LD_PRELOAD=/home/wangy/wangjh/harness3d/precompute-20261022-r01/cpu_sqlite/libsqlite3.so.0 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python \
/home/wangy/wangjh/harness3d/experiment-snapshots/tools/inspect_fullqa_snapshot.py \
/home/wangy/wangjh/harness3d/experiment-snapshots/fullqa-audited-20261025-v1
```

加 `--qa-id 10` 可重读该题的 32 帧与基础版 binding；启动脚本只输出身份与验证信息。题目必须来自快照 entries。该命令不连接在线模型。

进行 A/B 或演化前需另外冻结 panel、父/候选版本、模型、采样参数、seed、请求与 token 预算。默认 gate_size 为 8 个不同场景，此快照只有 4 个 inner_validation 场景，不能当作满足默认晋级 gate 的数据。可先用于诊断和有界验证；扩大正式晋级评测应等待更多已审计验证场景。r04 的 1,154 题属于另一冻结方法身份，单独使用并报告，不能和 fullQA 结果静默混合。

原始 reference answers 只进入单独评分器；在线读取、工具和 sandbox 继续不接收答案或 GT 几何。发布审计是工程完整性验证，不代表几何准确率：`m4_unavailable`、测量 unavailable 与米制失败保持原状态。
