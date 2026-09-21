#!/usr/bin/env bash
# scannet + scannetpp 限定重建（ARKitScenes 标定数据到位前的口径）
# 用法：GPU=<卡号> SPLITS=<逗号分隔> bash scripts/v5_scoped_recon.sh
set -uo pipefail
# 冻结环境（与 RunManifest 的 inference_env 一致；不得用 base env）
source /home/cvailab/anaconda3/etc/profile.d/conda.sh
conda activate skill3d-exp
GPU="${GPU:-0}"
SPLITS="${SPLITS:-inner_validation,outer_holdout}"
# 分层小样本：每题型 N 条（与评测批**必须**同参数，否则样本对不上）；0 = 全量
PER_TASK="${PER_TASK:-0}"
SHARD="${SHARD:-}"
DATASETS="${DATASETS:-scannet,scannetpp}"
RECON_DIR="${RECON_DIR:-data/v5_scoped/recon}"
export PYTHONPATH=third_party/vggt:src HF_HUB_OFFLINE=1
mkdir -p "$RECON_DIR" data/v5_scoped/logs
TAG="$(echo "$SPLITS" | tr ',' '_')"
LOG="data/v5_scoped/logs/recon_gpu${GPU}_${TAG}.log"
echo "=== recon GPU=$GPU splits=$SPLITS datasets=$DATASETS per_task=$PER_TASK shard=${SHARD:-none} ===" | tee "$LOG"
CUDA_VISIBLE_DEVICES="$GPU" python -m skill3d.reconstruction.run \
  --split "$SPLITS" --method vggt --datasets "$DATASETS" \
  --sampling-per-task "$PER_TASK" ${SHARD:+--scene-shard "$SHARD"} \
  --gpus 0 --recon-dir "$RECON_DIR" 2>&1 \
  | grep -vE "FutureWarning|autocast|^Loading weights" | tee -a "$LOG" | tail -3
echo "=== done $SPLITS ===" | tee -a "$LOG"
