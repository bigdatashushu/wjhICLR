#!/usr/bin/env bash
# inner 档主实验（§18.2：16 题/题型 × 3 seed）+ 噪声底协议（§18.3：同配置 ≥3 seed）。
#
# 样本：inner_validation × scannet+scannetpp，`--sampling-per-task 16` → 8 题型 × 16 = 128 题。
# 抽样是"按 meta 行序先到先得"（确定性，与 seed 无关）→ 6 个 run 看到**同一批 128 题**，
# 且是既有 32 题（4/题型）的**超集** → 可与既有结论直接对比。
#
# 三个臂：
#   C1_tools_program （程序路径，带 MoGe-2 米制尺度）
#   C0_direct_vlm    （直答基线；按规格不带 --moge2）
# 每个臂 3 个 seed = 噪声底；逐题一致率 / per-task 波动由 scripts/noise_floor.py 汇总。
#
# 分阶段跑的理由：seed0 先把 M5 逐场景清单缓存写进 recon 目录，seed1/2 复用缓存
# （对象清单是 scene 级产物，不该按 seed 重算）；同时避免 6 个进程同时写同一份缓存。
#
# 用法：bash scripts/run_inner128_batch.sh [--dry]
set -uo pipefail
cd "$(dirname "$0")/.."

source /home/cvailab/anaconda3/etc/profile.d/conda.sh
conda activate skill3d-exp
export PYTHONPATH=third_party/vggt:src
export HF_HUB_OFFLINE=1

ROOT="data/v6_inner128"
RECON="data/v6_scoped/recon_moge2"
ENDPOINT="http://127.0.0.1:8100"
MODEL="qwen3vl-8b-r0"
LOG="$ROOT/logs"
mkdir -p "$LOG"

DRY=0
[ "${1:-}" = "--dry" ] && DRY=1

one() {                        # one <baseline> <seed> <gpu> <tag>
  local baseline="$1" seed="$2" gpu="$3" tag="$4"
  local moge="--moge2"
  # C0 直答不需要米制尺度（不使用任何 Tool），按 §16.1 不带 --moge2
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
  if [ "$DRY" = "1" ]; then
    echo "CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"
    return 0
  fi
  echo "[batch] $tag start gpu=$gpu $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$LOG/$tag.log" 2>&1
  echo "[batch] $tag exit=$? $(date '+%F %T')"
}

# ---- Stage A：C1 seed0（单独跑，预热 M5 场景清单缓存）----
one C1_tools_program 0 0 c1_seed0

# ---- Stage B：C1 seed1/2 并行 ----
one C1_tools_program 1 1 c1_seed1 &
p1=$!
one C1_tools_program 2 3 c1_seed2 &
p2=$!
wait $p1 $p2

# ---- Stage C：C0 直答 seed0/1/2 并行 ----
one C0_direct_vlm 0 0 c0_seed0 &
q0=$!
one C0_direct_vlm 1 5 c0_seed1 &
q1=$!
one C0_direct_vlm 2 6 c0_seed2 &
q2=$!
wait $q0 $q1 $q2

echo "[batch] 全部结束 $(date '+%F %T')"
