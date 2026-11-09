# 按问题检测、候选编号与局部裁剪复核

当前实现入口为 `target_detection_r05`。真实四环境试跑的冻结源码、原始请求、
响应、动作与截图保存在 `../target_detection_r04`；修正版通过离线检查与真实
失败响应回放，尚未执行新的在线对照实验。

## 工作过程

1. 同一个 Qwen 从公开任务指令提取 1–4 个目标，包括物体、目的地或支撑物。
   类别用短英文名，数量、颜色等属性、空间关系分别记录；`source_phrase`
   必须出现在公开指令中。目标在整题内冻结，模型可选择事先记录的类别别名。
2. GroundingDINO 对当前未标注 RGB 逐类别检测，阈值为 0.15。
   保留全部原始框、分数、空标签和查询词，不将检测结果当作确认过的实例。
3. 对 IoU ≥ 0.8 的框做完整链接分组：新框必须与组内每个框都达到阈值，
   避免一串重叠框把相邻物体并在一起。原始成员全部保留。
4. 每页显示 12 组，全部后续页可以请求。G 编号放在原图外侧栏，通过
   颜色和引线关联检测框。同帧编号稳定，换观测后旧编号失效。
5. 模型每次请求最多两组原始 RGB 裁剪，保留少量周围像素；返回原始坐标与
   哈希。模型明确判断 `match`、`not_match` 或 `uncertain`，并给出视觉理由。
   只有当前已裁剪、判为 `match` 且来自相应目标查询的候选可以被选择。
6. 接受复核后，模型输出官方完整动作计划，执行器按原顺序执行，
   保留动作失败中断、终止与 ACK 语义。全部工具轮次计入模型请求预算，
   工具内循环不会 `reset`、`observe` 或 `step` 仿真。

Manipulation 保留原官方匿名编号图、坐标轴与 7D 动作坐标；检测和裁剪使用同帧
未标注 RGB。G 编号和官方 object 编号不同，匿名投影只表示可能对应。
各环境仍使用此前接入的同步 RGB 与原生深度显示。

## R05 的两处修正

- 裁剪文件名使用单独变量，结果中的 `tool` 始终为 `inspect_candidates`。
- 模型若只返回明确的 `perception_review` 和 `selected_targets`，按复核响应处理，
  执行与标准 `review_candidates` 相同的校验，再进入动作规划阶段。
  不生成、补充或修改任何动作；其他模糊或非法响应仍拒绝。

真实 R04 Navigation 的第 7 次响应触发了第二种情况。修正版离线重放该题
全部 7 次响应，确认复核能被接收。此回放没有新模型答案、检测请求或仿真运行，
不能视为该题已经在线成功。

## 检查与运行

工作目录为 `/Users/bytedance/wangjh/wjhICLR/embodiedRunner`，Python 为
`/Users/bytedance/wangjh/wjhICLR/.venv/bin/python`。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m unittest target_detection_r05.test_contracts -v
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_r05 check --manipulation-catalog /Users/bytedance/wangjh/wjhICLR/embodiedRunner/official_direct/assets/manipulation_assets.json
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B target_detection_r05/verification/replay_navigation_review.py
```

当前源码已冻结并部署到 `cvailab@10.10.0.198` 的独立 R05 目录，
四环境预检完成，预检没有启动仿真或请求模型。使用当前部署的新在线运行
须采用独立 run-id：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_r05.launch --run-id <new-run-id>
```

配置为 `experiment-20261108-r05.json`。Qwen 参数保持 `qwen3.8-flash`、
temperature=0、seed=137、max_tokens=4096、enable_thinking=false；
请求上限为 Navigation 20、ALFRED/Habitat 40、Manipulation 15。
检测复用固定物理 GPU1 的 SwinB 服务，每次核验服务与模型身份。
Navigation/Habitat 使用 GPU7，ALFRED/Manipulation 使用软件渲染，串行运行。
服务失败停止 suite，不重试模型请求，不重发动作。

模型 key 从此前授权的私有权限文件仅加载到 coordinator 内存，
worker、SSH 子进程和 detector 不接收该 key。

## 验收材料

- [真实四环境试跑报告](../target_detection_r04/acceptance-report.md)
- [真实 RGB、检测编号与裁剪对照](../target_detection_r04/detection-review-preview.png)
- [真实失败响应的修正版离线回放](verification/navigation-review-replay.json)
- [实现与验证状态](implementation-report.md)

这四题用于工程接入检查，无法据此判断完整数据集上的成功率收益。
