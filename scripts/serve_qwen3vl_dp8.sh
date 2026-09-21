#!/usr/bin/env bash
# Qwen3-VL-8B DP×N 启动脚本（系统架构2.md §13.1，ADR-2：DP 不 TP）
#
# 本脚本的参数已在本机 8×RTX4090 上**实测**（2026-09-18）：
#   - BF16 权重（17GB）在 24GB 卡上装不下（另有用卡时 OOM）→ 默认用官方 FP8（9.9GB）
#   - 32 帧均匀采样（§1.2）需要 --limit-mm-per-prompt image>=32
#   - 32 帧 @ max_pixels=131072 → prompt_tokens≈9703（≈303/帧），延迟 ~1.4s（单卡 FP8）
#   - 实测命令：Qwen/Qwen3-VL-8B-Instruct-FP8 + vllm 0.19.1（qwen3 conda env）
#
# 用法：
#   bash scripts/serve_qwen3vl_dp8.sh                            # FP8，8 卡 DP，port 8100..8107
#   DP_WORLD_SIZE=1 GPUS=3 bash scripts/serve_qwen3vl_dp8.sh     # 单卡（测试用）
#   MODEL=Qwen/Qwen3-VL-8B-Instruct QUANT=none ...              # BF16（需每卡 ≥20GB 空闲）
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
QUANT="${QUANT:-none}"           # FP8 权重自带量化，无需 --quantization；AWQ 时设 QUANT=awq
DP_WORLD_SIZE="${DP_WORLD_SIZE:-8}"
BASE_PORT="${BASE_PORT:-8100}"
GPUS="${GPUS:-}"                 # 空 = 用 0..DP_WORLD_SIZE-1；非空 = 逗号分隔的物理卡号
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.90}"     # 实际可用值取决于他人占用：先 nvidia-smi 看空闲显存
MAX_PIXELS="${MAX_PIXELS:-131072}"   # 单帧像素上限（TODO_CALIBRATE；实测 32 帧≈9.7k tokens）
N_IMAGES="${N_IMAGES:-32}"       # 与 §4 M1 的 32 帧均匀采样对齐
VLLM_BIN="${VLLM_BIN:-vllm}"     # 例：/home/cvailab/anaconda3/envs/qwen3/bin/vllm（vllm>=0.11 才支持 Qwen3-VL）
LOG_DIR="${LOG_DIR:-.}"

# 卡号列表
if [[ -n "$GPUS" ]]; then
  IFS=',' read -ra DEVICES <<< "$GPUS"
  DP_WORLD_SIZE="${#DEVICES[@]}"
else
  DEVICES=()
  for ((i=0; i<DP_WORLD_SIZE; i++)); do DEVICES+=("$i"); done
fi

QUANT_ARGS=()
[[ "$QUANT" != "none" ]] && QUANT_ARGS=(--quantization "$QUANT")

for idx in "${!DEVICES[@]}"; do
  dev="${DEVICES[$idx]}"
  port=$((BASE_PORT+idx))
  # HF_HUB_OFFLINE=1：本地已有缓存时不访问网络（HF 直连在本机不可达）
  CUDA_VISIBLE_DEVICES="$dev" HF_HUB_OFFLINE=1 nohup "$VLLM_BIN" serve "$MODEL" \
    "${QUANT_ARGS[@]}" \
    --served-model-name "qwen3vl-8b-r$idx" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --limit-mm-per-prompt "{\"image\": $N_IMAGES, \"video\": 1}" \
    --mm-processor-kwargs "{\"max_pixels\": $MAX_PIXELS}" \
    --port "$port" \
    > "$LOG_DIR/vllm_r$idx.log" 2>&1 &
  echo "rank $idx -> GPU $dev, port $port (pid $!)"
done

echo
echo "等待就绪后检查： curl -s http://127.0.0.1:$BASE_PORT/health"
echo -n "在线评测： python -m skill3d.online.eval --mode real --source vsi_bench"
for ((i=0;i<DP_WORLD_SIZE;i++)); do echo -n " --vllm-endpoint http://127.0.0.1:$((BASE_PORT+i))"; done
echo
