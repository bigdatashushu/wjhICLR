# Intern-Decision M5 实验服务

默认 M5 使用下文的 Intern P1 / .77 类别准入；Qwen 保留题目解析、M8 与视觉补答。
此目录的 GPU 依赖仅安装到仓库外的独立 Python 3.12 环境。
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

新版服务身份使用 `intern-decision-service/2`，额外记录实际加载的 Qwen3.5
逐层注意力、卷积和 normalization 实现，以及 fla-core、flash-linear-attention、
causal-conv1d、triton、einops、ninja 的存在状态、版本和实际文件哈希（包括 native
extension）。安装目录、bytecode 和安装时间不进入身份；同版本但字节改变仍会改变
identity。`--kernel-mode torch|fla|full` 分别要求原生算子、FLA 注意力加原生卷积、
FLA 加 causal-conv1d，任一层不匹配便拒绝启动。省略参数保留 upstream 自动选择；
不会改写算子。总括的 “fast path unavailable” 日志不足以判断 FLA 是否启用。

既有冻结服务的身份不会追溯升级。新增内核、修改 service.py 或改变算子模式时，
必须重新冻结实验版本、四个副本 identity pins 和 prepared-binding 方法身份，
不得把旧版与新版副本放进同一个审核池，也不得复用旧审核输出。
本机编译成功与可导入只证明 ABI 兼容，实际 GPU 请求、输入哈希和吞吐仍需独立验证。

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

下述历史 observation 模式中，未知类别、未知质量、格式/概率/身份异常或服务失败均关闭相关几何。Intern 观察不
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

The historical `frozen-quality` experiment uses `mode: binary` in `online_model.mask_review`, with explicit
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

## Main M5 category-only admission

The main YAML, `OnlineModelSettings` and `OnlineRunConfig` select P1 / .77
with `admission_policy: category-only`. The image compiler and prompt are unchanged.
Only the native `target_match` noul question is sent. No category name is
fabricated as an independent observation; no quality question is sent.
`p(yes)` uses the pinned model-package temperature, not the saved T=1 diagnostic.

This admission policy has its own version and identity. It does not read a
quality manifest. Receipts contain `quality_assessed: false`, empty observation,
exact candidate/frame/bbox and JPEG hashes, plus the original native response.
Pointcloud admission requires `p(yes) >= .77` and a valid native response; existing
depth, pointcloud, instance and binding validation still apply. These checks do
not establish mask purity/completeness. Unknown/failed service responses close
admission. Qwen remains responsible for question parsing, M8 and visual fallback.

Run the actual configured M5 on 32 frozen real frames with live GroundingDINO,
frame-independent SAM2 and live Intern:

```bash
SKILL3D_DETECTOR_ENDPOINT=http://127.0.0.1:20022 \
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src /path/to/skill3d-exp/bin/python \
  scripts/intern_decision/run_m5.py --main-config configs/config.yaml \
  --frozen /path/to/frozen/input --output /path/to/new/result \
  --historical-parser-result /path/to/actual-flash-result.json
```

The parser argument is only needed when `DASHSCOPE_API_KEY` is absent and must
match the exact question. It never substitutes for live detection, masks or
Intern decisions. Choose a free GPU before starting SAM2/Intern. Set
`online_model.mask_review: null` (or `OnlineRunConfig(semantic_review_config=None)`)
to explicitly use the Qwen path. Old prepared bindings/caches cannot serve the
new policy because configuration and implementation identities differ.
