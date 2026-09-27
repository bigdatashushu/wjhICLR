#!/usr/bin/env bash
# v7 inner 档主实验：C1（工具+程序+yield 多轮）与 C0（直答 VLM）对照。
#
# 与 v6 的 run_inner128_batch.sh 同帧集、同样本、同模型，只改被 v7 修正的部分，
# 使配对比较成立（硬约束 21：同一帧集、同一顺序、同一分辨率政策）。
#
# 输出目录用 data/v7_inner128：v6 的结果一律保留原样，不覆盖、不改标签。
# 注意：v7 扩了检测器词表（40→83），M5 场景清单缓存键随之变化 ——
# 所以 seed0 会重跑全部 scene 的 M5（VLM 清单 + GroundingDINO + SAM2 传播），
# seed1/2 复用同一份新缓存。
set -uo pipefail
cd "$(dirname "$0")/.."

source /home/cvailab/anaconda3/etc/profile.d/conda.sh
conda activate skill3d-exp
export PYTHONPATH=third_party/vggt:src
export HF_HUB_OFFLINE=1
export SKILL3D_DETECTOR_ENDPOINT="${SKILL3D_DETECTOR_ENDPOINT:-http://127.0.0.1:20022}"

ROOT="data/v7_inner128"
RECON="data/v6_scoped/recon_moge2"
ENDPOINT="http://127.0.0.1:8100"
MODEL="qwen3vl-8b-r0"
LOG="$ROOT/logs"
mkdir -p "$LOG"

one() {                        # one <baseline> <seed> <gpu> <tag>
  local baseline="$1" seed="$2" gpu="$3" tag="$4"
  local moge="--moge2"
  [ "$baseline" = "C0_direct_vlm" ] && moge=""
  local cmd=(python -m skill3d.online.eval
    --mode real --source vsi_bench --split inner_validation
    --datasets scannet,scannetpp --sampling-per-task 16 --seed "$seed"
    --baseline "$baseline" $moge
    --vllm-endpoint "$ENDPOINT" --vllm-model "$MODEL"
    --recon-dir "$RECON"
    --trace-dir "$ROOT/$tag/traces"
    --memory-dir "$ROOT/$tag/memory"
    --run-manifest "$ROOT/$tag/run_manifest.json")
  echo "[batch] $tag start gpu=$gpu $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$LOG/$tag.log" 2>&1
  echo "[batch] $tag exit=$? $(date '+%F %T')"
}

# ---- Stage A：C1 seed0（预热 v7 词表的 M5 场景清单缓存）----
one C1_tools_program 0 0 c1_seed0

# ---- Stage B：C1 seed1/2 并行（复用缓存）----
one C1_tools_program 1 1 c1_seed1 &
one C1_tools_program 2 3 c1_seed2 &
wait

# ---- Stage C：C0 直答 seed0/1/2 并行 ----
one C0_direct_vlm 0 2 c0_seed0 &
one C0_direct_vlm 1 5 c0_seed1 &
one C0_direct_vlm 2 6 c0_seed2 &
wait

echo "[batch] 全部结束 $(date '+%F %T')"
