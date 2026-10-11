# 2026-10-11 核心修复质量规则确认

当前合同：`072fd4de48d6ae608487d3fbafa67419152862cba1f20d64250ce4a069bac363`。
冻结运行时源码：`21100aea37a594c84fa1e31c948c9d7bd15620ca283c6adb2d869795f608f7ff`。

确认范围是质量规则、固定阈值、版本、输入身份与工具授权行为。
`confirmed=true` 与场景 `main_gate_passed` 独立。两个固定场景均未通过 AND 门，
依赖几何的工具被拒绝，图像工具和视觉作答仍可用，全部样本保留评分分母。
本确认不宣称 TODO 默认阈值已经统计校准，开发冒烟不取得正式结果资格。

固定输入为 induction 3107/3108 和 inner_validation 2166/2167，
原32帧、原Qwen/VGGT/SAM2/DINO权重，4096 tokens、最多6轮；
没有运行 Skill 演化或正式512次实验。

本地完整离线回归：1129通过、2跳过。GPU完成2条induction和4次B01/B11求解，
共25次模型请求（16次perception、9次program）；两次断流usage未知，
已取得vLLM真实abort和空运行队列证据。另有1次开发断流探针，不计生产成本。

`quality_confirmation_v11.json` 可传给现有GPU/配对入口。
场景确认仅覆盖这里两份artifact；其他正式面板场景仍须冻结输入、重建并确认身份。
原始数据、全请求、日志及重建数组保留在服务器：
`/home/cvailab/experiments/harness3d-v11/campaigns/core-repair-20261011`。

`quality_gate_receipt.json` 是原确认收据的简版索引；
scene内 `evidence_ref/evidence_sha256` 指向服务器原收据，未伪造或改写旧hash。
首条induction完成后的验收脚本序列化失败已从完整trace恢复，没有重复模型请求。

待正式启动：中英文快照包含相同修复基线、完整面板FrameSet/artifact身份、
场景确认、环境/服务/磁盘复检。低几何通过率本身不阻断正式实验。
