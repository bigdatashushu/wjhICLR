# 按问题检测目标并裁剪复核

独立包 `target_detection_trial`，基于现有官方完整动作计划和同步 RGB-D 观测。
仅此包产生新源码和结果。旧包、结果和上游快照保留原样。

1. 同一 `qwen3.8-flash` 根据公开任务指令提取 1–4 个目标，类别用短英文名词，
   数量、颜色、关系独立记录。`source_phrase` 必须来自指令原文，根目标在整题内冻结。
2. 模型请求 `detect_targets`。GroundingDINO 使用同一观测的原始 RGB，
   每个类别独立请求，阈值固定 0.15。保留所有框、空标签、原分数和查询词。
   只有模型显式选择已冻结 alias 时才更换查询词，没有自动重试或自动扩展。
3. IoU ≥ 0.8 的框采用完整链接检查分组，新增成员必须与组内每个成员满足条件，
   防止 A–B–C 链式相交把相邻物体合成一组。分组仍保留每个原始框，
   不代表已经确认物体身份，也不能直接作为物体数量。
4. 每帧分配递增 G 编号，换观测后全部旧编号失效。同帧缓存复用检测结果。
   检测图按 12 组分页，编号始终放在图外侧栏，以颜色和引线关联对应框。
   `view_candidates` 可以取全部后续页；数据没有被 Top-K 删除。
5. 模型用 `inspect_candidates` 请求最多 2 组局部裁剪，裁剪取自未标注 RGB，
   保留少量上下文、原图坐标和哈希。模型随后给出 match/not_match/uncertain
   及视觉理由；只有当前已裁剪并判为 match 的组才能作为 `selected_targets`。
6. 模型输出官方完整 `executable_plan`。原有动作前缀、失败中断、终止和 ACK
   语义保持原版。工具循环不 reset、step 或 observe 仿真。

初次返回示例：

```json
{
  "targets": [
    {
      "id": "T1", "role": "object", "category": "star", "source_phrase": "star",
      "quantity": 1, "attributes": [], "relation": "", "aliases": ["star shaped block"]
    }
  ],
  "tool_request": {"name": "detect_targets", "target_ids": ["T1"]}
}
```

裁剪请求：

```json
{"tool_request": {"name": "inspect_candidates", "group_ids": ["G002"]}}
```

动作 JSON 保留官方字段，同时增加：

```json
{
  "perception_review": [{"group_id": "G002", "verdict": "match", "reason": "Visible star-shaped object."}],
  "selected_targets": [{"target_id": "T1", "group_id": "G002"}]
}
```

Manipulation 的原模型图保留官方匿名物体坐标和编号。检测图另用 G 编号，
两者不等价。工具只提供官方匿名坐标在当前图上的投影作为可能对应，
模型仍须核对原图，投影和检测框都不能证明语义或表面几何准确性。

## 运行

工作目录：`/Users/bytedance/wangjh/wjhICLR/embodiedRunner`
Python：`/Users/bytedance/wangjh/wjhICLR/.venv/bin/python`

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m unittest target_detection_trial.test_contracts -v
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_trial check --manipulation-catalog /Users/bytedance/wangjh/wjhICLR/embodiedRunner/official_direct/assets/manipulation_assets.json
```

首次冻结：`python -B -m target_detection_trial.lifecycle freeze`。
部署和状态检查用已修正进程匹配的辅助入口：
`python -B target_detection_trial/verification/lifecycle_control.py deploy`，
再执行同一入口的 `probe`。已经冻结或部署的目录不能覆盖。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_trial.launch --run-id target-detection-20261108-r01
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_trial.audit --run-id target-detection-20261108-r01
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B target_detection_trial/verification/lifecycle_control.py after
```

新一次运行需独立 `--run-id`，也可用 `--environment navigation` 等筛选。
同一个 run-id 不能再次启动。模型 key 仅由本地 coordinator 从原有授权
`qwenkey.md` 在内存加载；文件必须限制权限。worker 和检测 SSH 子进程用
环境白名单，不接收模型凭证。不要把 key 写入配置或命令行。

Qwen 参数固定为 temperature=0、seed=137、max_tokens=4096、
enable_thinking=false。模型预算为 Navigation 20、ALFRED/Habitat 40、
Manipulation 15 次，包含目标提取、工具复核和行动轮次。此轮四环境各一题
串行验收：Navigation base0、ALFRED base0、Habitat base0、Manipulation base1。

GroundingDINO 使用已加载的物理 GPU1，端口 20030，SwinB 模型身份固定。
每次连接核验启动回执、PID 创建时间、boot_id、进程参数哈希和真实 GPU 映射。
不经过共享检测池、不新加载检测模型。Navigation/Habitat 仿真使用 GPU7，
ALFRED/Manipulation 使用软件渲染。服务异常不重试，动作不重发。

模型生成的目标/工具格式错误记录为该题 planner_error；服务、仿真和观测
完整性异常关闭 suite。真实验收结果见此目录的 `acceptance-report.md`。
四题仅验证工作流程，不能用于判断整体成功率收益。
