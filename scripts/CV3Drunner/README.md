# CV3Drunner

CV3D 的新实现集中在此目录。独立协议为 `cv3d-box-review-program/3`，
只接收一张原始图片、两个选项的 CV3D Depth / Distance 题。

## 代码边界

| 文件 | 职责 |
| --- | --- |
| `grounding.py` | 从原始 RGB 的完整矩形边框提取红、蓝、绿框，记录唯一性及问题颜色绑定 |
| `tools.py` | 每题独立注册框内取点工具，复用原有 5×5 深度中值与局部中值点距离计算 |
| `runner.py` / `context.py` | 五次程序请求加一次 finalization；新增测量必须在下一次模型请求中复核 |
| `protocol.py` | 原图、可选同网格深度图、取点标注图及冻结 Skill 的请求内容 |
| `program.py` / `worker.py` | 输入身份检查、轨迹记录及独立子进程主线程沙箱 |
| `run.py` | 零模型/GPU请求预检、冻结清单、双臂运行和主机侧评分 |
| `runner.yaml` / `skills/` | 本地配置及有内容哈希的 Depth / Distance 策略 |
| `tests/` | 本目录显式运行的离线测试 |

共享 `src/skill3d` 提供数值采样、输入校验、沙箱、答案解析和请求预算；
`tests/CV3D` 提供已有数据回放、传输客户端、被动轨迹记录、调度和预算接口。
这些文件没有被改写，旧入口仍使用自己的配置和工具面。新逻辑不向全局注册表
注册工具，不对共享 solver 类进行替换。当前是源码目录入口，部署需同时携带
依赖的 `src` 和 `tests/CV3D`。

## 修复行为

- 按四边覆盖、角点和闭合性识别矩形；不使用所有同色像素的最大最小坐标。
  缺框、多个同色框或候选过多时，隐藏测量工具，模型仍可依据原图答题。
- `point_depth_in_box(view_index, box_id, u, v)` 和
  `pair_distance_in_boxes(view_index, a_box_id, a_uv, b_box_id, b_uv)` 仅接受
  问题绑定的框，归一化参数必须有限且位于 `[0,1]`。
  参数映射到扣除笔画、取样半径和舍入余量的 `safe_xyxy`。
  旧的任意像素测量接口不在新模型工具面中。
- 同一程序内测量后 `ReturnAnswer` 会记为暂定答案。
  下一次请求提供全部有效测量、准确的取样像素及标注图；模型可以重新取点
  或调整答案。重新测量会再次要求下一轮复核。第五轮测量交给第六轮
  finalization 复核，预算不增加。
- 框内点不等于物体表面。取样标注图展示 5×5 区域，并要求结合遮挡、支撑、
  接触关系判断测量冲突。主机强制的是复核机会，语义判断由同一模型完成。
  所有值保持原始预测尺度，不改深度或点云，不硬编码任何样本坐标/答案。
- 安全违规、工具失败、超时和服务失败终止该 episode；有限的格式/API/Python
  修复使用原有五个程序请求槽。失败后不自动重试模型请求。

## 入口

在仓库根运行：

```bash
.venv/bin/python -B scripts/CV3Drunner/run.py --config /absolute/path/to/new-run.json
```

默认只预检。运行配置采用原 paired 配置的字段，并将输出指向一个新目录：

```json
{
  "output": "/absolute/path/to/new-output",
  "dataset_root": "/absolute/path/to/dataset",
  "public_manifest": "/absolute/path/to/public-inputs.json",
  "official_annotations": "/absolute/path/to/official-3d.jsonl",
  "prepared_root": "/absolute/path/to/existing-prepared",
  "answers_path": "/absolute/path/to/host-only-answers.json",
  "concurrency": 8,
  "max_requests": 5908,
  "timeout_s": 180
}
```

`runner_config` 和 `skills_manifest` 可省略，默认使用本目录文件。
默认预检 4 道训练协议题及 840 道作者测试题。运行明确冻结的小集合时增加：

```json
{
  "selection": {
    "train_ids": ["depth-train-id", "distance-train-id"],
    "test_ids": ["test-id"]
  }
}
```

每道题预算至少 7 次请求：Skill 最多 6 次，直答 1 次。训练协议需同时覆盖
Depth / Distance；测试选择及顺序写入 manifest，预检后不能改变。
十道题的诊断实验使用 `run_kind: "bounded_diagnostic_trial"`，
`selection: {"test_ids": [...]}`，选择五道 Depth、五道 Distance，并将
`max_requests` 固定为 70。这个入口不增加训练题或额外模型请求。
只跑完整作者测试集时使用 `run_kind: "full_test_paired"`，不传 `selection`，
并将 `max_requests` 固定为 5880。该入口严格核验 Depth / Distance 各 420 题，
共 840 题、1680 个 episode，不增加训练题；求解逻辑与十题诊断入口一致。
预检哈希绑定输入、原始图片、prepared 产物、框目录、工具面、配置、Skill
及源码。重复预检保留首次 receipt，检测到身份变化时拒绝继续。

正式运行使用同一命令加 `--execute`，并在运行配置中提供：
`gateway`、`gateway_source_sha256`、`upstream_endpoint_sha256`、
`auth_module_root`。后者包含现有 `transport_auth.load_transport_token`。
配置不保存密钥。运行前检查网关身份及 8 路容量。两臂各 4 个 episode，
不借用对方空闲槽；Skill 在子进程主线程执行，标签只由父进程评分。
直答请求复用原来精确的官方文本和原始 RGB 字节。

输出包含冻结 manifest、预检 receipt、请求/响应及工具轨迹，
后续请求的 `*.sampling_overlay.png`、episode 结果和 paired 汇总。
请求预算账本只保存允许的身份字段和数值 token 用量。

## 离线验证

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts:src:tests \
  .venv/bin/python -B -m pytest scripts/CV3Drunner/tests -q \
  -p no:cacheprovider --basetemp=/private/tmp/cv3d-local-validation
```

测试使用合成 RGB/geometry 和本机假 HTTP 服务，无真实模型或 GPU 调用。
如果 `/tmp/cv3d-disagreement-000775-rgb.png` 存在，还会检查该原图的三个框及
历史错误像素确实在相应安全区域之外。这项检查证明取点约束，不能代替
真实模型实验对准确率的验证。
