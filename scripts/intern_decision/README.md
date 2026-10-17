# Intern-Decision M5 实验服务

默认 M5 仍使用百炼 Qwen。此目录的 GPU 依赖仅安装到仓库外的独立 Python 3.12 环境。
`serve.py` 导入所选模型包的原版 `inference.py`，使用官方 compiler、masked assistant
skeleton、marker 前一位置 logits 和合法单 token 选项，保留未校准及 release 温度结果。

已冻结模型 revision 为 `1e65a988dacbcf00b84aee7366f9e47563407000`。
官方仓库参考 commit 为 `3572c8a68b5df5dafe02d0e093989ba8ec0183bc`；本服务无需该整仓。
HF 模型包的温度为 `1.99241824`，与同包实现一起固定；不应用仓库 XTuner 预设，
不拟合本任务温度，也不声称复现论文 ECE。

```bash
python3.12 -m venv "$DEPLOY_ROOT/venv"
python3 scripts/intern_decision/download.py "$DEPLOY_ROOT/model"
"$DEPLOY_ROOT/venv/bin/python" -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu126
"$DEPLOY_ROOT/venv/bin/python" -m pip install transformers==5.14.1 'Pillow>=10,<13'
bash scripts/intern_decision/start.sh "$DEPLOY_ROOT" 2 18445
```

下载失败可在联网节点运行 download.py，将完整目录及 manifest 传到服务器。
依赖也可下载与 Linux/Python 3.12/CUDA 匹配的 wheels，再用 `--no-index --find-links`
安装。本次使用 Torch fallback，没有可选 FLA/causal-conv1d 加速，时延单独报告。
实际环境锁与每个 wheel 的哈希在外部实验报告目录中。

服务仅监听 `127.0.0.1`，单次前向并发为 1，拒绝忙请求，限制请求和图片尺寸，
8192 token 超限拒绝，120 秒推理 watchdog 终止挂起服务。图片只接收内联字节；
临时路径不含客户端文件名，compiler 仅看到图片占位与像素，不接收目标和检测标签。
`/health` 保存 PID、冷启动、权重、processor、tokenizer、backend、温度与实现身份。
实际响应另含三个图像 grid、图像 token 数、合法 token IDs 和 marker 读出位置。

三图调用：

```bash
.venv/bin/python scripts/intern_decision/call.py context.jpg crop.jpg overlay.jpg \
  --identity '<health identity_sha256>' --output receipt.json
```

从 Mac 调用可建 SSH 隧道：

```bash
ssh -N -L 18445:127.0.0.1:18445 cvailab@10.10.0.198
```

在 `online_model.mask_review` 中选择 `provider: intern-decision`、loopback endpoint、
已验证的 `expected_identity_sha256`。关系默认为 `qwen`，仅替换视觉审核。
`relation: frozen-rules` 是独立实验：在冻结观察后做固定同义/类别规则，明确记录 identity。
parser、M8 和补答继续由原来的 online_model 控制。普通 runner 和 paired preparation
共用实际 M5 入口。实验缓存禁用；不复用 Qwen 收据。

未知类别、未知质量、格式/概率/身份异常或服务失败均关闭相关几何。Intern 观察不
生成 free-form alternate_object；名称来自固定 60 项选项映射。规则未覆盖的自由名词
保留 uncertain。完整性只判断可见轮廓，实验没有概率准入阈值。

停止前核对 startup PID；Linux 上可用：

```bash
python3 scripts/intern_decision/stop.py "$DEPLOY_ROOT"
```

官方来源：[模型包](https://huggingface.co/internlm/Intern-Decision-4B)、
[实现](https://github.com/InternLM/Intern-Decision)、
[数据合同](https://github.com/InternLM/Intern-Decision/blob/main/docs/DATA.md)、
[评测合同](https://github.com/InternLM/Intern-Decision/blob/main/docs/EVALUATION.md)。

真实 M5 复跑：设置现有检测器 endpoint、SAM2 checkpoint/config 环境变量，使用原有
冻结目录内的 episode、32 帧和 artifact。此入口共用生产 `bind_objects_for_scene`，
实际扫描检测器、SAM2、HTTP 审核、关系、点云与绑定校验。

```bash
PYTHONPATH=src /path/to/skill3d-exp/bin/python scripts/intern_decision/run_m5.py \
  --frozen /path/to/frozen/input --output /path/to/new/result \
  --identity '<service identity_sha256>' --relation frozen-rules \
  --historical-parser-result /path/to/actual-flash-result.json
```

历史复用只允许真实成功的 Flash parser 收据，严格匹配问题输入；不能用它回放视觉审核。
有 `DASHSCOPE_API_KEY` 时省略历史 parser 参数，默认关系为 Qwen。

## Target-conditioned yes/no experiment

`run_binary_experiment.py` runs P0/P1/P2 on the exact same three JPEGs. Native
`noul` returns both probabilities; it does not generate an object name. The
fixed model-package temperature is retained and T=1 probabilities are saved.
Threshold search starts at .50, brackets with .70/.90/.95/.99, then bisects to
.01. It uses saved probabilities, not repeated inference for each threshold.

`mode: binary` is opt-in in `online_model.mask_review`, with explicit
`prompt_version`, `yes_threshold`, `quality_manifest` and
`quality_manifest_sha256`. The quality manifest must refer to the exact same
candidate ID and JPEG hashes. This first experiment reuses explicitly pinned
old target-blind quality receipts, so semantic and final admission metrics must
be reported separately. A below-threshold score establishes uncertainty, not
physical absence. Missing/changed quality closes geometry admission.

Development scenes are chair/fridge (112); validation scenes are paper bag and
computer tower/heater (98). Prompt/threshold selection must finish before the
validation stage. `score_binary_experiment.py` produces candidate-level results
and pessimistic repeat summaries. A Codex visual review is required before
selection; it is not substituted for human reference labels.

A native binary request can be made with the same three-image CLI:

```bash
.venv/bin/python scripts/intern_decision/call.py context.jpg crop.jpg overlay.jpg \
  --target 'chair' --prompt-version P1 --identity '<service identity_sha256>' \
  --output binary-receipt.json
```

Do not infer binary probabilities from the earlier 60-category softmax. The
binary protocol issues a new native `noul` request with only `target_match`.
`P(yes) >= yes_threshold` is the explicit acceptance predicate. The operating
point is chosen on development scenes and frozen before validation. Missing or
changed candidates/images in the frozen quality manifest fail closed.

The M5 CLI accepts `--binary-operating-point` and
`--frozen-quality-manifest` for this same optional protocol. A historical parser
receipt can still be supplied explicitly if the DashScope environment key is
absent. It is never used to answer a new binary vision request.
