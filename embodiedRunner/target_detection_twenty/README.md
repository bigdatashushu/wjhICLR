# 原 20 题问题定向感知实验

复用此前四环境官方 `base 0–4` 的 20 题。方法为已冻结、已部署的
`target_detection_r05`，不改写四题验收的源码、配置或结果。
新模型请求和观测记录保存于 R05 的独立 `target-detection-twenty-20261108-r01`
运行目录；本目录保存完整题目清单、实验身份、历史对照和最终报告。

模型、RGB-D 输入、官方完整动作计划与原请求预算保持；所有检测、裁剪和复核
轮次均计入原预算。仿真串行，GroundingDINO 复用 GPU1，Navigation/Habitat 用 GPU7，
ALFRED/Manipulation 用软件渲染。服务故障停止 suite，不重发请求或动作。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_twenty.control prepare
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_twenty.control probe
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_twenty.control run
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_twenty.control status
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_twenty.control after
PYTHONDONTWRITEBYTECODE=1 /Users/bytedance/wangjh/wjhICLR/.venv/bin/python -B -m target_detection_twenty.report
```

准备和运行身份不可覆盖。若运行因服务故障停止，先处理已观察到的故障和自有资源；
仍未启动的题目可用独立 run-id 和 `--remaining` 执行，已启动题目不会再次运行。

模型 key 仅通过此前授权的私有权限文件在 coordinator 内存加载，不向 SSH、
worker 或 detector 传递，不保存原始值。

历史 RGB-D 20 题有完整观测与模型请求，成功 15/20。旧 Full 的感知工具轨迹另作
补充参考；它的子目标和执行协议与当前完整计划不同，不能视为相同协议的消融。

最终交付为 [报告](report.md)、[逐题结果](results.json) 和四环境的实际截图。
