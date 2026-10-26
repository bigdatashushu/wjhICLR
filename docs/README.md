# Harness3D 文档索引

当前系统从 [项目 README](../README.md) 和 [系统架构 v11](../系统架构v11.md) 阅读。专题说明与历史资料集中在 `docs/`；正文中的命令和代码路径默认以仓库根目录为起点。

## 当前实验与运行说明

| 文档 | 用途 |
|---|---|
| [配对评测与消融合同](skill_ablation_v11.md) | B01/B11 面板、共享准备、配对身份与评分 |
| [同题 fit-set 槽位流](v11_fitset_slot_stream.md) | 同题学习、修订、晋升及恢复 |
| [Episode 并发池](v11_episode_pool.md) | spawn worker、并发和谱系边界 |
| [19 槽位流式演化](v11_streaming_19_slots.md) | 完整槽位范围和演化入口 |
| [米制融合链路](metric_fusion_v11.md) | MoGe-2、尺度证据和米制工具准入 |
| [FullQA 无人值守运行](fullqa_unattended.md) | 预计算、监督、服务和收尾 |
| [SSD 发布传输](cluster_ssd_publication_transport.md) | SSD 发布、背压、验证和切换 |
| [M5 题目对象验证](m5_question_validation_20261015.md) | 对象解析、候选绑定和核查记录 |

组件说明见 [方法库 README](../skill_library/README.md)、[集群 README](../src/skill3d/cluster/README.md) 和 [Intern 服务 README](../scripts/intern_decision/README.md)。运行期 `SKILL.md` 按方法库版本目录存放。

## 历史架构与实现记录

历史资料中的版本、环境和运行结论以其记录时间为准。

| 文档 | 范围 |
|---|---|
| [v3 实现对照](架构v3实现对照.md) | 早期架构落地记录 |
| [v4 实现对照](架构v4实现对照.md) | 早期架构落地记录 |
| [v5 实施报告](architecture_v5_implementation_report.md) | 标定、程序路径和 v5 实现 |
| [v6 实施报告](architecture_v6_implementation_report.md) | v6 几何与工具合同 |
| [v7 实现与实测报告](architecture_v7_implementation_report.md) | v7 实现对照与实测 |
| [系统架构 v9](系统架构v9.md) | v9 规格 |
| [v9 方法库实现记录](skill_library_v9_implementation.md) | v9 方法库与观察链 |
| [系统架构 v10](系统架构v10.md) | v10 规格 |
| [v10 方法库实现记录](skill_library_v10_implementation.md) | v10 P0–P3 落地 |
| [v10 实现与验收报告](系统架构v10_实现与验收报告.md) | v10 正式运行与验收 |

## 历史实验、环境与诊断

- [v6 inner32 实验报告](实验报告v6_inner32.md)
- [v6 MoGe-2 PoC 报告](实验报告v6_moge2_poc.md)
- [本机环境实测](本机环境实测.md)
- [2026-09-24 系统现状](系统现状0924.md)
- [供架构修订的问题报告](问题报告_供架构修订.md)
- [架构决策记录目录](decisions/)
- [阶段报告目录](reports/)
