#!/usr/bin/env bash
# Qwen3-VL-8B DP×8 启动脚本（系统架构.md §13.1，ADR-2：DP 不 TP）
# 每卡一个 vLLM 副本，绑不同 CUDA_VISIBLE_DEVICES，port 8100..8107
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
QUANT="${QUANT:-awq}"            # 关键结果用官方 FP8 交叉验证：QUANT=fp8 MODEL=Qwen/Qwen3-VL-8B-Instruct-FP8
DP_WORLD_SIZE="${DP_WORLD_SIZE:-8}"
BASE_PORT="${BASE_PORT:-8100}"

for ((i=0; i<DP_WORLD_SIZE; i++)); do
  CUDA_VISIBLE_DEVICES=$i nohup vllm serve "$MODEL" \
    --quantization "$QUANT" \
    --served-model-name "qwen3vl-8b-r$i" \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --port $((BASE_PORT+i)) \
    > "vllm_r$i.log" 2>&1 &
  echo "rank $i -> port $((BASE_PORT+i)) (pid $!)"
done
