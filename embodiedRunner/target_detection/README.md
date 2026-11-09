# 目标检测与裁剪复核

可运行入口：`target_detection`；配置：`experiment-20261108-r03.json`。
旧实验及前两次接入版本全部保留。

同一个 Qwen 从公开任务指令提取 1–4 个英文类别，单独记录数量、颜色和关系，
并用原文片段绑定目标。根目标整题冻结，按需使用已声明 alias。
模型逐类别调用 GroundingDINO，阈值固定 0.15，保留全部原始框、空标签、
分数与查询词；IoU ≥ 0.8 的框用完整链接检查分组，防止重叠链误合并。

每组采用 G 编号，换观测后失效，同帧缓存稳定。图片每页最多 12 组，
编号放在图外侧栏并通过引线关联，所有后续页可取，不删除低分候选。
模型每次可请求 1–2 组原始 RGB 裁剪；裁剪保留上下文、原图坐标和哈希。
随后独立调用 `review_candidates`，明确判定 match/not_match/uncertain，
记录视觉理由和所选目标，之后返回官方完整动作计划。

独立复核阶段确保每次判断都有明确记录，动作 JSON 保持官方字段。
工具循环不 reset、step、observe 仿真；目标提取、检测、裁剪、复核和动作
模型轮次均计入原请求预算。只有当前已裁剪并判为 match 的候选可选。
分组、分数和视觉判断均不是 GT，组数也不等于物体数。

Manipulation 检测使用同一观测的未标注 RGB，官方模型图仍保留匿名编号、
坐标轴和动作坐标。G 编号不等于官方 object 编号。工具只提供匿名坐标
投影作为可能对应，模型须在官方图中核对，不能据此宣称表面几何已验证。

模型阶段格式：

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

确认后输出官方 `executable_plan`；复核记录按当前观测保留，无需在动作响应中重复。

## 运行

工作目录：`/Users/bytedance/wangjh/wjhICLR/embodiedRunner`。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m unittest target_detection.test_contracts -v
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection check --manipulation-catalog /Users/bytedance/wangjh/wjhICLR/embodiedRunner/official_direct/assets/manipulation_assets.json
```

首次冻结、部署和预检依次执行 `python -B -m target_detection.lifecycle freeze`、
`deploy`、`probe`。已经冻结或部署的目录不能覆盖。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection.launch --run-id target-detection-20261108-r03
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection.audit --run-id target-detection-20261108-r03
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection.lifecycle after
```

新运行需独立 run-id，可加 `--environment navigation` 等筛选。
key 由本地 coordinator 从此前授权的私有权限 `qwenkey.md` 仅内存加载。
SSH、worker、detector 采用环境白名单，不接收 Qwen key，配置和日志不保存 key。

模型固定 `qwen3.8-flash`、temperature=0、seed=137、max_tokens=4096、
enable_thinking=false；请求上限 Navigation 20、ALFRED/Habitat 40、
Manipulation 15，包含全部工具/复核轮次。四题为 Navigation base0、
ALFRED base0、Habitat base0、Manipulation base1，串行运行。

检测使用已加载的物理 GPU1、端口 20030、固定 SwinB 哈希，逐次验证启动
回执、PID/创建时间、boot_id、进程参数哈希和真实 GPU 映射。仿真 GPU7
用于 Navigation/Habitat，ALFRED/Manipulation 采用软件渲染。服务异常关闭
suite，不重试、不重发动作；模型格式错误终止该题。原计划前缀、失败中断、
终止和 ACK 逻辑保留官方语义。

七项离线测试覆盖目标来源与冻结、分组链、逐类别请求、缓存、编号失效、
裁剪原像素、图外标号、明确阶段、四轮工具零仿真步和完整计划 ACK。
真实结果与实际截图见 `acceptance-report.md`，仅用于小样本接入验收。
