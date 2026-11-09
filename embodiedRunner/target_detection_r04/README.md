# 按问题检测目标、候选编号与裁剪复核

当前入口：`target_detection_r04`，配置 `experiment-20261108-r04.json`。
源码和结果独立，前三次接入版本及旧实验均保留。

同一 Qwen 依次完成目标提取、检测请求、裁剪请求、明确裁剪判断和官方完整动作计划。
目标来自公开任务指令，英文类别、数量、颜色属性和关系分别记录；
原文片段必须匹配指令，1–4 个根目标在整题内冻结。

GroundingDINO 逐类别检测同一帧未标注 RGB，阈值固定 0.15，
保留全部原始框、空标签、分数和查询词。IoU ≥ 0.8 用完整链接检查分组，
防止重叠链误并。分组不等于物体实例或物体数量。

G 编号每帧递增、换观测失效，同帧缓存稳定。每页 12 组，全部后续页可取，
编号在图外侧栏并由颜色/引线关联，不删除低分候选。
每次裁剪最多 2 组，使用原始 RGB 像素并保留少量上下文、坐标和哈希。
模型用 `review_candidates` 明确给出 match/not_match/uncertain、视觉理由和所选目标，
只有当前已裁剪并判为 match 的候选能被选择。

之后输出官方完整计划。离散动作兼容明确的整数 ID、`[整数 ID, 描述]` 和
`action_id` 字典；不推测 ID、不修正非法 ID、不补动作。Manipulation 仍需合法 7D 向量。
原计划前缀、失败中断、终止和 ACK 语义保持；工具不 reset/step/observe 仿真。
全部 Qwen 轮次计入原模型请求预算，无重试、无动作重发。

Manipulation 的官方匿名编号图、坐标轴和动作坐标保留；检测另用同帧未标注 RGB。
G 编号不等于官方 object 编号。匿名坐标投影仅提供可能对应，需在官方图中复核。
投影、检测框和 Qwen 判断均不表示类别或表面几何已经验证。

## 模型接口

```json
{
  "targets":[{"id":"T1","role":"object","category":"star","source_phrase":"star",
              "quantity":1,"attributes":[],"relation":"","aliases":["star shaped block"]}],
  "tool_request":{"name":"detect_targets","target_ids":["T1"]}
}
```

```json
{"tool_request":{"name":"inspect_candidates","group_ids":["G002"]}}
```

```json
{"tool_request":{"name":"review_candidates",
 "perception_review":[{"group_id":"G002","verdict":"match","reason":"Visible star-shaped object."}],
 "selected_targets":[{"target_id":"T1","group_id":"G002"}]}}
```

```json
{"executable_plan":[{"action_id":0,"action_name":"Move forward by 0.25"}]}
```

## 运行入口

工作目录 `/Users/bytedance/wangjh/wjhICLR/embodiedRunner`。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m unittest target_detection_r04.test_contracts -v
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_r04 check --manipulation-catalog /Users/bytedance/wangjh/wjhICLR/embodiedRunner/official_direct/assets/manipulation_assets.json
```

首次准备依次用 `python -B -m target_detection_r04.lifecycle freeze`、`deploy`、`probe`。
已冻结、部署或开始的 run-id 不可覆盖。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_r04.launch --run-id target-detection-20261108-r04
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_r04.audit --run-id target-detection-20261108-r04
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_r04.lifecycle after
```

新运行需独立 run-id，也可加 `--environment navigation` 等筛选。
Qwen key 从此前授权的私有权限 `qwenkey.md` 仅在 coordinator 内存加载，
SSH、worker 和 detector 采用环境白名单，不接收模型 key。

模型固定 `qwen3.8-flash`、temperature=0、seed=137、max_tokens=4096、
enable_thinking=false；请求上限 Navigation 20、ALFRED/Habitat 40、Manipulation 15，
包含全部工具和复核轮次。四题为 Navigation base0、ALFRED base0、Habitat base0、
Manipulation base1，串行运行。检测复用已加载的物理 GPU1/SwinB/端口 20030，
逐次核验模型哈希、启动回执、PID/创建时间、boot_id、参数哈希和真实 GPU 映射。
仿真 Navigation/Habitat 用 GPU7，ALFRED/Manipulation 软件渲染。

八项离线检查覆盖目标来源/冻结、逐类别查询、分组链、缓存、编号失效、
原始裁剪像素、图外标号、明确阶段、四轮工具零仿真步、动作形式与完整计划 ACK。
实际验收和截图见 `acceptance-report.md`。四题只用于接入验收，不是整体收益评估。
