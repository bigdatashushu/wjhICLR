#!/usr/bin/env bash
# inner 档（§18.2：16 题/题型）重建：把 inner_validation × scannet+scannetpp 需要的
# 全部 scene 用 VGGT + MoGe-2 重建进同一个 artifact 目录。
#
# 为什么与 32 题实验共用 `data/v6_scoped/recon_moge2`：artifact 是**按 scene** 落的
# （`--episodes-per-scene 1`），而 frame set 只由视频决定（`video_path_for` 只看
# dataset/scene_name），与"哪个题型/哪一行 QA 选中该 scene"无关 → 同 scene 复用是
# 同一份计算。共用后 32 题（4/题型）是 128 题（16/题型）的**子集**，可直接比。
#
# 用法：bash scripts/recon_inner128.sh [GPU 列表,默认 0,1,3,5,6,7]
set -uo pipefail
cd "$(dirname "$0")/.."

source /home/cvailab/anaconda3/etc/profile.d/conda.sh
conda activate skill3d-exp
export PYTHONPATH=third_party/vggt:src
export HF_HUB_OFFLINE=1

GPUS="${1:-0,1,3,5,6,7}"
RECON_DIR="${RECON_DIR:-data/v6_scoped/recon_moge2}"
LOG_DIR="data/v6_scoped/logs"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a gpu_arr <<< "$GPUS"
n=${#gpu_arr[@]}
echo "[recon_inner128] shards=$n gpus=$GPUS recon_dir=$RECON_DIR"

pids=()
for i in "${!gpu_arr[@]}"; do
  g="${gpu_arr[$i]}"
  log="$LOG_DIR/recon128_shard${i}of${n}_gpu${g}.log"
  CUDA_VISIBLE_DEVICES="$g" python -m skill3d.reconstruction.run \
    --split inner_validation --method vggt --datasets scannet,scannetpp \
    --sampling-per-task 16 --episodes-per-scene 1 --moge2 \
    --recon-dir "$RECON_DIR" --scene-shard "${i}/${n}" \
    > "$log" 2>&1 &
  pids+=($!)
  echo "  shard ${i}/${n} gpu=${g} pid=${pids[-1]} log=$log"
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[recon_inner128] shard $i 失败（pid=${pids[$i]}）"
    fail=1
  fi
done
echo "[recon_inner128] 全部 shard 结束 fail=$fail"

# 合并分片 manifest → 一份整批 manifest（分片各自写了自己的那份，不再互相覆盖）
python - "$RECON_DIR" "$n" <<'PY'
import glob, json, sys
recon_dir, n = sys.argv[1], int(sys.argv[2])
shards = sorted(glob.glob(f"{recon_dir}/vggt/manifest_shard*of*.json"))
merged = {"method": "vggt", "recon_dir": recon_dir, "schema_version": "6.0",
          "merged_from_shards": len(shards), "jobs": []}
for s in shards:
    d = json.load(open(s))
    for k in ("quality_metric_version", "n_frames", "dataset_scope", "sampling_per_task",
              "sampling_strategy", "question_type_scope"):
        merged.setdefault(k, d.get(k))
    merged["jobs"].extend(d.get("jobs", []))
# scene 去重（同一 scene 不应出现在两个 shard；出现即说明分片有过重叠）
seen, uniq = set(), []
for j in merged["jobs"]:
    if j["scene_name"] in seen:
        continue
    seen.add(j["scene_name"])
    uniq.append(j)
merged["jobs"] = sorted(uniq, key=lambda j: j["scene_name"])
merged["n_scenes"] = len(uniq)
for st in ("done", "skipped", "failed"):
    merged[f"n_{st}"] = sum(1 for j in uniq if j["status"] == st)
out = f"{recon_dir}/vggt/manifest.json"
json.dump(merged, open(out, "w"), ensure_ascii=False, indent=2)
print(f"[recon_inner128] manifest 合并 → {out}（shards={len(shards)} "
      f"scenes={merged['n_scenes']} done={merged['n_done']} failed={merged['n_failed']}）")
PY
exit $fail
