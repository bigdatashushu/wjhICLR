# legacy/ — 原 Skill-3D 参考代码（保留，不参与新系统）

本目录存放改造前的原始 codebase：**Skill-3D**（*Evolving Scene-Aware Skills for Agentic 3D
Spatial Reasoning*，arXiv:2606.07436，Apache-2.0）。原项目说明见本目录 `README.md`
（未改动），许可证见仓库根 `LICENSE`。

## 为什么移到这里

新系统按 `系统架构.md` 以 `src/skill3d/` 为单一事实源（§2）。原代码顶层也有一个
`skill3d/` 包，与 `src/skill3d/` **同名**；由于 `sys.path` 中 cwd 优先，从仓库根运行时
旧包会遮蔽新包：`import skill3d` 拿到旧代码，`pytest` 出现 16 个 collection error
（`No module named 'skill3d.reconstruction_gate'`）。

因此把旧顶层项一次性 `git mv` 到 `legacy/` 下（内容零改动，git 记录为 rename）：

| 原路径 | 现路径 | 说明 |
|---|---|---|
| `skill3d/` | `legacy/skill3d/` | SPAgent、core（skill_learning / skill_retrieval / prompts）、tools、models、external_experts |
| `test/` | `legacy/test/` | 旧脚本式测试（需外部工具服务） |
| `examples/` | `legacy/examples/` | 旧评测示例 |
| `train/` | `legacy/train/` | 旧 SFT / GRPO 训练 |
| `statics/` | `legacy/statics/` | 旧 memory / progressive_skills 目录骨架 |
| `third_party/` | `legacy/third_party/` | GroundingDINO / Orient-Anything / sam3 等 |
| `checkpoints/` | `legacy/checkpoints/` | 旧权重骨架（真实权重不入库） |
| `assets/` | `legacy/assets/` | 旧 README 图片 |
| `README.md` | `legacy/README.md` | 旧项目说明（原样保留） |
| `requirements.txt` | `legacy/requirements.txt` | 旧依赖清单（新系统用 `pyproject.toml`） |
| `scripts/*.sh` | `legacy/scripts/` | `common_env.sh`、`run_skill3d_*_inference.sh`、`vllm_start.sh` |

## 怎么使用

新系统**不 import** 本目录任何代码（`src/skill3d` 对 `external_experts` / `third_party`
无任何依赖）。需要对照旧实现时按旧方式独立运行：

```bash
cd legacy
PYTHONPATH=. python skill3d/quick_start.py     # 旧 SPAgent demo
```

旧代码依赖 `legacy/requirements.txt`（vLLM、transformers、groundingdino_py 等），
与新系统的依赖环境互不影响；跑旧代码请另建环境。

## 与新系统的对应关系（仅供对照）

| 旧实现 | 新系统（`系统架构.md`） |
|---|---|
| `core/skill_learning.py`、`core/skill.py` | M15 Skill Registry（§4 M15）+ M16 GPT-6 离线归纳 |
| `core/skill_retrieval.py` | M7 Skill Retriever（§4 M7） |
| `core/tool.py`、`tools/*` | M6 Tool Registry（§4 M6） |
| `external_experts/Pi3` | M3 3D 重建——新系统主线为 **VGGT**（§4 M3、ADR-3），不是 Pi3 |
| `external_experts/SAM3`、`GroundingDINO` | M5 对象绑定——新系统为 **SAM2 + 实例关联**（§4 M5） |
| `statics/skill3d_shared/` | M14 Memory Store（LanceDB）+ M15 |
| 在线由 GPT-4o / GPT-5 类模型交织驱动 | 在线 **Qwen3-VL-8B 确定性执行**、GPT-6 **仅离线**（硬约束 1/2） |

对照表只说明"从哪来"，不代表新系统会复用旧实现；技术选型一律以 `系统架构.md` 为准。
