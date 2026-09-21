#!/usr/bin/env bash
# v5 分层子集实验：8 题型 × N episodes，真实链（无 BA）
# 纪律：产物只用于"管道级真实结果"，不是论文主表结论（单 seed / 小样本，HC34）
set -uo pipefail
# 冻结环境（与 RunManifest 的 inference_env 一致；不得用 base env）
source /home/cvailab/anaconda3/etc/profile.d/conda.sh
conda activate skill3d-exp
cd "$(dirname "$0")/.."
TASKS="object_counting object_abs_distance object_size_estimation room_size_estimation object_rel_distance object_rel_direction route_planning obj_appearance_order"
N="${N:-4}"
RECON_DIR="${RECON_DIR:-data/v5_subset/recon}"
TRACE_DIR="${TRACE_DIR:-data/v5_subset/traces}"
LOG="${LOG:-data/v5_subset/sweep.log}"
export PYTHONPATH=third_party/vggt:src
export HF_HUB_OFFLINE=1
mkdir -p "$RECON_DIR" "$TRACE_DIR" "$(dirname "$LOG")"

{
echo "=== P1 重建（GPU5，method=vggt，无 BA）==="
for T in $TASKS; do
  echo "--- P1 task=$T limit=$N ---"
  CUDA_VISIBLE_DEVICES=5 python -m skill3d.reconstruction.run \
    --split inner_validation --method vggt --question-types "$T" --limit "$N" \
    --gpus 0 --recon-dir "$RECON_DIR" 2>&1 | grep -E "^  (done|failed|skipped)|^\[warn\]|manifest"
done

echo "=== P2 在线评测（vLLM=GPU4 / SAM2=GPU5）==="
for T in $TASKS; do
  echo "--- P2 task=$T limit=$N ---"
  CUDA_VISIBLE_DEVICES=5 python -m skill3d.online.eval \
    --mode real --source vsi_bench --split inner_validation \
    --question-types "$T" --limit "$N" \
    --recon-dir "$RECON_DIR" --recon-method vggt \
    --vllm-endpoint http://127.0.0.1:8100 --vllm-model qwen3vl-8b-r0 \
    --trace-dir "$TRACE_DIR" --memory-dir data/v5_subset/memory \
    --run-manifest "data/v5_subset/run_manifest_${T}.json" 2>&1 \
    | grep -E "per-task|^  [a-z_]+ +n=|主表口径|C1_tools_program|coverage=|refusal|tool_contract:|尺度能力|confidence 分布|synthesis_source"
done
echo "=== 子集实验完成 ==="
} 2>&1 | tee -a "$LOG"
