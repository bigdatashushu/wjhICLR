# 按问题检测、编号与局部裁剪复核

当前可运行包：`target_detection_v2`，四环境独立配置
`experiment-20261108-r02.json`。保留旧实验和首版 `target_detection_trial`。

同一 Qwen 在系统指令规定的阶段依次完成：

1. 根据公开指令提取 1–4 个目标。短英文类别、数量、颜色属性和关系分别记录；
   原文 `source_phrase` 必须与指令匹配，根目标在整题内冻结。
2. 按需调用 `detect_targets`，每类别单独向已加载的 GroundingDINO SwinB
   发送当前原始 RGB。阈值为 0.15，保留所有框、空标签、原始分数和查询词。
   已冻结 alias 只有模型显式选择时才使用。
3. IoU ≥ 0.8 的框用完整链接检查分组，防止重叠链误并相邻物体。每组保留
   全部提议，组数不等于物体数。G 编号随观测更新失效，同帧结果缓存稳定。
4. 每页 12 组，编号在图外侧栏，用颜色和引线对应框。`view_candidates`
   可以查看全部后续页，没有 Top-K 删除。
5. `inspect_candidates` 每次最多复核 2 组，从未标注 RGB 截取带少量上下文
   的原像素裁剪。模型给出 match/not_match/uncertain 和理由。只有当前已
   裁剪并判为 match 的组可作为 `selected_targets`。
6. 输出官方完整 `executable_plan`，保留原动作前缀、失败中断、终止和 ACK。
   工具轮次不 reset、step 或 observe 仿真，所有 Qwen 轮次计入原请求预算。

Manipulation 的官方图和匿名动作坐标保持原格式；检测另用未标注同帧 RGB。
工具 G 编号不等于官方 object 编号。匿名坐标投影只给出当前图的可能对应，
模型必须结合官方编号图核对，投影和检测分数都不能证明类别或物体表面准确。

首个响应同时提取目标并请求检测：

```json
{
  "targets": [
    {"id":"T1","role":"object","category":"star","source_phrase":"star",
     "quantity":1,"attributes":[],"relation":"","aliases":["star shaped block"]}
  ],
  "tool_request":{"name":"detect_targets","target_ids":["T1"]}
}
```

```json
{"tool_request":{"name":"inspect_candidates","group_ids":["G002"]}}
```

准备行动时，保留官方计划字段，并增加：

```json
{
  "perception_review":[{"group_id":"G002","verdict":"match","reason":"Visible star-shaped object."}],
  "selected_targets":[{"target_id":"T1","group_id":"G002"}]
}
```

## 入口

工作目录 `/Users/bytedance/wangjh/wjhICLR/embodiedRunner`。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m unittest target_detection_v2.test_contracts -v
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_v2 check --manipulation-catalog /Users/bytedance/wangjh/wjhICLR/embodiedRunner/official_direct/assets/manipulation_assets.json
```

首次冻结和部署依次执行 `python -B -m target_detection_v2.lifecycle freeze`、
`deploy`、`probe`。已经冻结、部署或开始的 run-id 均不能覆盖。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_v2.launch --run-id target-detection-20261108-r02
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_v2.audit --run-id target-detection-20261108-r02
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_v2.lifecycle after
```

新一次运行需要独立 run-id；也可加 `--environment navigation` 等筛选。
模型 key 仅由本地 coordinator 从原有授权 `qwenkey.md` 内存加载，文件须为
私有权限。SSH、worker 和 detector 子进程用环境白名单，不接收 Qwen 凭证。

模型参数固定为 `qwen3.8-flash`、temperature=0、seed=137、max_tokens=4096、
enable_thinking=false。Navigation 20、ALFRED/Habitat 40、Manipulation 15 次
模型请求上限包含目标提取、工具和动作。四题为 Navigation base0、ALFRED base0、
Habitat base0、Manipulation base1，串行运行。

检测使用已加载的物理 GPU1、端口 20030 和固定模型哈希，逐次核验启动回执、
PID/创建时间、boot_id、进程参数哈希和真实 GPU 映射。仿真 Navigation/Habitat
使用 GPU7，ALFRED/Manipulation 使用软件渲染。服务异常关闭 suite，不重试；
动作不重发。模型目标/工具格式错误终止该题并记录 planner_error。

七项离线检查覆盖目标来源/冻结、逐类别请求、缓存、完整链接分组、编号失效、
裁剪像素、图外编号、系统阶段指令和完整计划 ACK。真实验收及原截图见
`acceptance-report.md`。小样本只验证接入，不能据此判断成功率提升。
