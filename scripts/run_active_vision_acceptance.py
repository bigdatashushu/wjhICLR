#!/usr/bin/env python3
"""§18 阶段 1 硬性证据：**真实** crop → 下一模型请求 → 下一程序 → 答案链。

规范原文（§18 阶段 1 交付证据）："八题型真实模型求解覆盖、实际方法正文输入；
**至少一条真实 crop→下一模型请求→下一程序→答案链**；S0 冻结记录。"

本脚本跑**真实**链路（真实 vLLM endpoint + 真实视频帧 + 真实重建 artifact）：

1. 取一条真实 episode（真实 VSI-Bench video + 复用既有 VGGT artifact）；
2. 第 1 轮：程序调用 `inspect_frames` 裁剪真实像素 → `YieldObservations` 让出；
3. 第 2 轮：**真实模型请求**（本轮请求里真的带上那张裁剪图）→ 模型写下一段程序 →
   提交答案；
4. 落盘图像账本（produced / delivered / observed 三态、布局、token 成本）。

两种模式：

- `--model-rounds real`：第 1 轮**也**由真实模型生成（最忠实，但模型不一定会去裁剪）；
- `--model-rounds scripted-first`：第 1 轮用一段**固定的**程序强制触发裁剪链，
  第 2 轮起全部真实（模型真实收到裁剪图、真实写程序、真实作答）。
  脚本会打印两种模式的差别，"第一条链由谁写"必须如实标注。

用法：
  EXP=/home/cvailab/anaconda3/envs/skill3d-exp/bin/python
  PYTHONPATH=src $EXP scripts/run_active_vision_acceptance.py \
      --endpoint http://127.0.0.1:8100 --qa-id 2487 --model-rounds scripted-first
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# 第 1 轮的固定程序（scripted-first 模式）：真实调用 inspect_frames 裁剪真实像素，
# 再把裁剪图的 image_id 交给 YieldObservations —— 这正是 §9.4 的主动观察入口。
SCRIPTED_FIRST_PROGRAM = (
    'crop = inspect_frames([8], [[0, 0, 96, 96]])\n'
    'img = crop["images"][0]["image_id"]\n'
    'print("crop:", img)\n'
    'return YieldObservations([img], "需要放大看清这一块区域")\n'
)


def _unmark_scripted_round(ledger: dict, round_index: int, note: str) -> int:
    """把"脚本轮"的交付/观察标记按事实撤回（**不谎报**）。

    `scripted-first` 模式下第 1 轮的程序由脚本给出、**没有发出真实请求**；runner 在
    chat 调用周围做的 delivered/observed 标记因此是替身的产物。receipt 必须体现真相：
    这一轮的图既没进过请求、也没被模型看过 —— 撤回标记并写明原因，而不是让它自称"看过"。
    """
    cleared = 0
    rid = int(round_index)
    for img in ledger.get("images") or []:
        delivered = [r for r in (img.get("delivered_rounds") or []) if r != rid]
        observed = [r for r in (img.get("observed_rounds") or []) if r != rid]
        missed = (len(delivered) != len(img.get("delivered_rounds") or [])
                  or len(observed) != len(img.get("observed_rounds") or []))
        if missed:
            cleared += 1
        img["delivered_rounds"] = delivered
        img["observed_rounds"] = observed
        if not observed:
            img["unobserved_reason"] = note
    for rnd in ledger.get("rounds") or []:
        if int(rnd.get("round_index", 0)) == rid:
            rnd["delivered"] = False
            rnd["observed"] = False
            rnd["prompt_tokens"] = None
            rnd["scripted_no_request"] = True
    imgs = ledger.get("images") or []
    ledger["stats"] = {
        "n_images_produced": len(imgs),
        "n_images_delivered": sum(1 for i in imgs if i.get("delivered_rounds")),
        "n_images_observed": sum(1 for i in imgs if i.get("observed_rounds")),
        "n_images_unobserved": sum(1 for i in imgs if not i.get("observed_rounds")),
        "n_derived_produced": sum(1 for i in imgs if i.get("kind") != "frame"),
        "n_derived_observed": sum(1 for i in imgs if i.get("kind") != "frame"
                                  and i.get("observed_rounds")),
    }
    return cleared


def _n_images(messages) -> int:
    return sum(1 for m in messages if isinstance(m.get("content"), list)
               for p in m["content"]
               if isinstance(p, dict) and p.get("type") == "image_url")


class ScriptedFirstClient:
    """第 1 轮返回固定程序，之后**全部转真实** VLLMClient（含图像的真实请求）。"""

    def __init__(self, real_client, first_program: str = SCRIPTED_FIRST_PROGRAM) -> None:
        self._real = real_client
        self._first = first_program
        # 只统计**程序合成**调用：M5 对象清单阶段也会走同一个客户端，
        # 把那些调用算进来会让"第一轮"错位（实测踩过一次：脚本程序没被用上）。
        self.n_program_calls = 0
        self.scripted_rounds: list[int] = []
        self.last_usage: dict = {}
        self.log: list[str] = []

    def chat(self, messages, max_tokens: int = 4096, seed=None) -> str:
        blob = json.dumps(messages, ensure_ascii=False)
        is_program_call = ("ReturnAnswer" in blob or "YieldObservations" in blob)
        if is_program_call:
            self.n_program_calls += 1
            if self.n_program_calls == 1:
                self.scripted_rounds.append(1)
                self.log.append("program#1 → 脚本程序（未发请求）")
                # 第 1 轮**没有请求** → 不能留下任何"真实用量"（否则会被记成真实 token 成本）
                self.last_usage = {}
                try:
                    self._real.last_usage = {}
                except Exception:  # noqa: BLE001
                    pass
                return self._first
            self.log.append(f"program#{self.n_program_calls} → 真实模型"
                            f"（图 {_n_images(messages)} 张）")
        out = self._real.chat(messages, max_tokens=max_tokens, seed=seed)
        self.last_usage = dict(getattr(self._real, "last_usage", None) or {})
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="qwen3vl-8b-r0")
    ap.add_argument("--qa-id", default="2487", help="VSI-Bench meta 的题目 id")
    ap.add_argument("--artifact", default="data/reconstructions/vggt/7b6477cb95.json")
    ap.add_argument("--model-rounds", choices=["real", "scripted-first"],
                    default="scripted-first")
    ap.add_argument("--video-root", default="")
    ap.add_argument("--out", default="data/p4_acceptance.json")
    ap.add_argument("--no-video", action="store_true",
                    help="只打印将要用到的 episode 信息，不跑链路")
    args = ap.parse_args(argv)

    from skill3d.adapters import vsibench_loader as vl
    from skill3d.adapters.episode_source import load_vsi_bench_items
    from skill3d.online.config import load_config, paths_from
    from skill3d.online.runner import OnlineRunConfig, run_episode
    from skill3d.synthesis.vllm_client import VLLMClient, probe_service

    cfg_yaml = load_config(REPO / "configs/config.yaml")
    paths = paths_from(cfg_yaml)

    # §15.2 preflight：服务必须真的在（不在就明确报错，不静默降级）
    probe = probe_service(args.endpoint, args.model)
    print(f"[preflight] {json.dumps(probe, ensure_ascii=False)}")
    if not probe.get("healthy"):
        print("[错误] 模型服务不可用：§18 阶段 1 的证据不能由 mock 代替", file=sys.stderr)
        return 2

    rows = vl.load_meta()
    row = next((r for r in rows if str(r.get("id")) == str(args.qa_id)), None)
    if row is None:
        print(f"[错误] meta 里没有 qa_id={args.qa_id}", file=sys.stderr)
        return 2
    scene = str(row["scene_name"])
    qtype = str(row["question_type"])
    print(f"[episode] qa_id={args.qa_id} scene={scene} type={qtype}\n"
          f"          question={row.get('question')}\n"
          f"          ground_truth={row.get('ground_truth')}")

    # 取该 scene 的 episode（真实视频 + 复用既有 artifact；A/B 同源）
    items = load_vsi_bench_items(
        "inner_validation", {"inner_validation_scene_ids": [scene]},
        video_root=args.video_root or paths.raw_videos,
        video_fallback_roots=paths.raw_video_fallbacks,
        cache_dir=paths.vsi_bench_meta)
    item = next((i for i in items if str(i.episode.qa_id) == str(args.qa_id)), None)
    if item is None:
        print(f"[错误] 该 scene 下没装配出 qa_id={args.qa_id} 的条目", file=sys.stderr)
        return 2
    print(f"[frames] n={len(item.pixels)} frame_set_hash="
          f"{item.episode.frame_set.frame_set_hash[:16]}…")
    if args.no_video:
        return 0

    real = VLLMClient([args.endpoint], args.model)
    client = real if args.model_rounds == "real" else ScriptedFirstClient(real)
    cfg = OnlineRunConfig(
        mode="real", vllm_endpoints=[args.endpoint], vllm_model=args.model,
        reuse_artifact=str(REPO / args.artifact) if args.artifact else None,
        seed=0, trace_dir=str(REPO / "data/traces_real"),
        max_images=int((cfg_yaml.get("vllm") or {}).get("n_frames", 32) or 32),
        max_pixels=int((cfg_yaml.get("vllm") or {}).get("max_pixels", 131072) or 131072),
        max_model_len=int((cfg_yaml.get("vllm") or {}).get("max_model_len", 32768) or 32768),
        active_snapshot_ref="S0-seed-20260925-v1",
    )
    out = run_episode(item.episode, item.pixels, cfg, llm=client)

    ledger = out.image_ledger or {}
    scripted = list(getattr(client, "scripted_rounds", []) or [])
    if scripted:
        n_cleared = _unmark_scripted_round(
            ledger, scripted[0],
            f"第 {scripted[0]} 轮由脚本程序触发（未发出真实请求）→ 不计交付/观察")
        print(f"[honesty] 撤回第 {scripted[0]} 轮（脚本轮）的交付/观察标记：{n_cleared} 张图")
    stats = ledger.get("stats") or {}
    derived = [i for i in (ledger.get("images") or []) if i.get("kind") != "frame"]
    report = {
        "qa_id": args.qa_id, "scene": scene, "question_type": qtype,
        "model_rounds_mode": args.model_rounds,
        "scripted_rounds": scripted,
        "honesty_note": ("第 1 轮由脚本程序触发（未发真实请求）→ 该轮交付/观察标记已撤回；"
                         "第 2 轮起的请求、图像与程序均为真实"
                         if scripted else "全部轮次均由真实模型生成"),
        "endpoint": args.endpoint, "model": args.model,
        "final_state": out.final_state, "answer": out.answer,
        "ground_truth": row.get("ground_truth"),
        "correct": out.correct, "mra_value": out.mra_value,
        "synthesis_source": out.synthesis_source,
        "agent_rounds": out.agent_rounds, "yield_count": out.yield_count,
        "n_images_per_request": [
            {"round": r["round_index"], "layout": r["layout"], "n_images": r["n_images"],
             "derived": len(r["derived_image_ids"]),
             "originals": len(r["original_image_ids"]),
             "omitted_originals": len(r["omitted_originals"]),
             "delivered": r["delivered"], "observed": r["observed"],
             "prompt_tokens": r.get("prompt_tokens"),
             "tokens_estimate": r.get("token_estimate_total")}
            for r in (ledger.get("rounds") or [])],
        "image_stats": stats,
        "derived_images": [
            {k: i.get(k) for k in ("image_id", "kind", "source_frame_id",
                                   "source_frame_index", "box_xyxy",
                                   "source_hw", "sent_hw", "scale", "content_sha256",
                                   "delivered_rounds", "observed_rounds",
                                   "token_estimate", "unobserved_reason")}
            for i in derived],
        "notes_tail": [n for n in out.notes if "派生图" in n or "yield" in n][:5],
    }
    if getattr(client, "log", None):
        print("[client log] " + " | ".join(client.log))
    print("[notes]")
    for n in out.notes:
        print("   -", n[:160])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    Path(REPO / args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[receipt] {args.out}")

    # §18 阶段 1 的判据：至少一条真实 crop → 下一模型请求 → 下一程序 → 答案链。
    # 脚本轮（若有）的交付/观察已被撤回，因此这里的 observed 必然是**真实请求**留下的。
    ok = bool(derived) and all(d.get("observed_rounds") for d in derived) \
        and any(r["delivered"] and r["observed"] and r["derived_image_ids"]
                for r in (ledger.get("rounds") or []))
    print(f"[chain] crop→下一请求→下一程序→答案：{'PASS' if ok else 'FAIL'}"
          f"（derived={len(derived)}，answer={out.answer!r}）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
