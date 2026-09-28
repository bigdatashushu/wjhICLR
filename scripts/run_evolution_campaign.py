#!/usr/bin/env python3
"""v10 §11/§14 两代演化 campaign 的**真实**运行入口（P2/P3）。

它做四件事（其余都在 `skill3d.evolution.campaign` 里）：

1. **冻结面板**：按规范题型从 learning / inner_validation 两个 split 取题，并把
   inner 子面板按 **scene 整块**切成互不重叠的 `inner_g0` / `inner_g1`（§8.5）；
2. **建 campaign 自己的 Skill 库**：从仓库的 `skill_library/` 复制 S0 快照与 manifest
   （仓库 S0 库的 `--check` 不变量保持不变，演化只发生在副本里）；
3. **接线**在线模型（本地 vLLM）与离线归纳模型（DeepSeek；只有显式给了
   `--offline-*` 覆盖时才换别的 endpoint，且会如实记进收据）；
4. 调 `EvolutionCampaignRunner.run()`，把 §14.1 的十类收据写进
   `<run-root>/<campaign-id>/gen<N>/`。

用法：

```
EXP=/home/cvailab/anaconda3/envs/skill3d-exp/bin/python
PYTHONPATH=src $EXP scripts/run_evolution_campaign.py --dry-run          # 只冻结面板与库
PYTHONPATH=src $EXP scripts/run_evolution_campaign.py --campaign-id v10-001
```

**不谎报**：`--mode mock_light` 只做管道验证（不可 promote，§5.6b）；离线模型不可用
时 campaign 进 `blocked`，不会用 mock 顶替（§3.4）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
# 真实重建走 `vggt_runner`（lazy import `vggt`）；源码在 third_party/vggt，与仓库既有
# 重建脚本一致（scripts/recon_inner128.sh 等同样用 `PYTHONPATH=third_party/vggt:src`）。
if (REPO / "third_party" / "vggt").is_dir():
    sys.path.insert(0, str(REPO / "third_party" / "vggt"))

from skill3d.adapters.episode_source import (  # noqa: E402 - 脚本入口
    EpisodeSourceError,
    load_vsi_bench_items,
)
from skill3d.evolution.campaign import (  # noqa: E402
    CampaignConfig,
    EvolutionCampaignRunner,
    seed_campaign_library,
)
from skill3d.online.config import load_config, load_yaml, paths_from  # noqa: E402
from skill3d.routing.retrieval_policy import retrieval_policy_from_config  # noqa: E402
from skill3d.tools import REGISTRY  # noqa: E402


def _canonical_of(item) -> str:
    from skill3d.routing.task_classifier import canonical_task

    return canonical_task(item.episode.question_type)


class FrozenPanels:
    """§8.5：面板清单在**看候选结果之前**冻结（scene 整块、inner 两代互不重叠）。"""

    def __init__(self, cfg_yaml: dict, args) -> None:
        self.cfg_yaml = cfg_yaml
        self.args = args
        self._cache: dict[str, list] = {}
        self._sampling: dict[str, dict] = {}
        self.manifest: dict = {}

    def _load(self, split: str, limit: int) -> list:
        """按 **scene 为单位**的确定性抽样取条（§5.3；禁止按文件行序取前 N 题）。

        `_sample_by_scene(per_task_cap)` 在一个 scene 内会连续取多条（直到配额），而
        `max_per_scene=1` 只保留每 scene 的第一条 —— 两者叠加会让"想要的 N 条"缩水成
        "前几个 scene"。因此这里把配额放宽到 3N 再按 scene 去重截取：结果 = N 个
        **不同 scene** 的题目（经验包的跨 scene 计数才有意义），抽样仍由 seed 决定。
        抽样收据（seed/qa_id/hash）与面板清单一并落盘。
        """
        paths = paths_from(self.cfg_yaml)
        split_cfg = load_yaml(self.cfg_yaml.get("split_config", "configs/vsi_bench_split.yaml"))
        receipt: dict = {}
        items = load_vsi_bench_items(
            split, split_cfg, video_root=self.args.video_root or paths.raw_videos,
            video_fallback_roots=paths.raw_video_fallbacks,
            cache_dir=paths.vsi_bench_meta,
            question_types=[self.args.question_type],
            stratified_per_task=max(int(limit) * 3, int(limit)),
            sampling_seed=int(self.args.seed),
            max_per_scene=1, seed=self.args.seed, sampling_receipt=receipt)
        seen: set[str] = set()
        picked: list = []
        for item in items:
            scene = item.episode.scene_name
            if scene in seen:
                continue
            seen.add(scene)
            picked.append(item)
            if len(picked) >= int(limit):
                break
        receipt["picked_scenes"] = sorted(seen)
        self._sampling[split] = receipt
        return picked

    def freeze(self) -> dict:
        """冻结 learning 池与 inner 两代子面板（互不重叠，按 scene 整块切）。"""
        learn_limit = int(self.args.learning_limit)
        inner_limit = int(self.args.inner_panel_limit)
        learning = self._load(self.args.learning_split, learn_limit)
        inner_all = self._load(self.args.inner_split, (inner_limit * 2) + 8)
        # scene 整块切分（硬约束 19：不共享 scene）；scene 顺序由 seed 决定，可复现
        by_scene: dict[str, list] = {}
        for item in inner_all:
            by_scene.setdefault(item.episode.scene_name, []).append(item)
        import random

        scenes = sorted(by_scene)
        random.Random(self.args.seed).shuffle(scenes)
        half = len(scenes) // 2
        g0_scenes, g1_scenes = scenes[:half], scenes[half:]
        g0 = [it for s in g0_scenes for it in by_scene[s]][:inner_limit]
        g1 = [it for s in g1_scenes for it in by_scene[s]][:inner_limit]
        used = {it.episode.qa_id for it in g0 + g1}
        probe = [it for it in inner_all if it.episode.qa_id not in used][
            : int(self.args.post_publish_limit)]
        self._cache = {"learning": learning, "inner_g0": g0, "inner_g1": g1,
                       "post_publish": probe}
        self.manifest = {
            "question_type": self.args.question_type,
            "learning_split": self.args.learning_split,
            "inner_split": self.args.inner_split,
            "seed": int(self.args.seed),
            "learning_qa_ids": [it.episode.qa_id for it in learning],
            "learning_scenes": sorted({it.episode.scene_name for it in learning}),
            "inner_g0_qa_ids": [it.episode.qa_id for it in g0],
            "inner_g1_qa_ids": [it.episode.qa_id for it in g1],
            "inner_g0_scenes": sorted({it.episode.scene_name for it in g0}),
            "inner_g1_scenes": sorted({it.episode.scene_name for it in g1}),
            "post_publish_qa_ids": [it.episode.qa_id for it in probe],
            "inner_panels_disjoint": not (
                {it.episode.scene_name for it in g0}
                & {it.episode.scene_name for it in g1}),
            # §5.3：抽样算法、scene/qa_id 清单、seed 与 hash 必须落盘
            "sampling_receipts": dict(self._sampling),
        }
        return self.manifest

    def provider(self, kind: str, generation: int):
        if kind == "learning":
            return list(self._cache.get("learning") or [])
        if kind == "inner":
            key = "inner_g0" if int(generation) == 1 else "inner_g1"
            return list(self._cache.get(key) or [])
        if kind == "post_publish":
            return list(self._cache.get("post_publish") or [])
        raise KeyError(f"未知面板 {kind}")

    def qa_ids(self, kind: str) -> list[str]:
        if kind == "learning":
            return [it.episode.qa_id for it in self._cache.get("learning") or []]
        return []


def _source_env_file(path: Path) -> None:
    """把本地凭据文件里的 `export K=V` 注入当前进程环境（**值不打印、不落盘**）。"""
    if not path.is_file():
        raise SystemExit(f"凭据文件不存在: {path}")
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, _, value = line[len("export "):].partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)
            count += 1
    print(f"[凭据] 从 {path} 注入 {count} 个环境变量（值不显示、不落盘）")


def _make_online_llm(args):
    if args.mode != "real":
        return None
    from skill3d.synthesis.vllm_client import VLLMClient

    endpoints = [e for e in (args.vllm_endpoint or [f"http://127.0.0.1:8100"])]
    client = VLLMClient(endpoints, model=args.vllm_model)
    ok, reason = client.check_service()
    if not ok:
        raise SystemExit(f"在线模型不可用（§5.6b：真实运行必须有真实模型）: {reason}")
    print(f"[在线模型] {reason}")
    return client


def _make_offline(args):
    """离线归纳模型：默认冻结的 DeepSeek 配置；只有显式覆盖才换 endpoint。"""
    from skill3d.governance.deepseek_client import (
        DeepSeekClient,
        OfflineAuthError,
        OfflineServiceUnavailable,
    )

    # 只在**显式给出**覆盖时才传参：`DeepSeekClient` 的参数注解是 `str`，显式传 None
    # 会被 `str(None)` 变成字面量 "None"（endpoint 就废了），而缺省参数才是 §3.4 的
    # 冻结值。因此这里用 kwargs 逐项决定，不用 `or None`。
    kwargs: dict = {}
    if args.offline_api_key_env:
        kwargs["api_key"] = os.environ.get(args.offline_api_key_env, "")
    if args.offline_base_url:
        kwargs["base_url"] = args.offline_base_url
    if args.offline_model:
        kwargs["model"] = args.offline_model
    client = DeepSeekClient(**kwargs)
    if not client.api_key_configured:
        print(f"[离线模型] 未配置密钥（环境变量 {args.offline_api_key_env or 'DEEPSEEK_API_KEY'}）"
              "→ campaign 会在 GENERATE_CANDIDATE 进 blocked（§3.4：不用 mock 顶替）")
        return client
    try:
        client.require_service()
        print(f"[离线模型] provider={client.provider} model={client.model_id} "
              f"endpoint_hash={client.endpoint_hash}")
    except (OfflineAuthError, OfflineServiceUnavailable) as exc:
        print(f"[离线模型] 健康检查失败: {type(exc).__name__}: {exc} "
              "→ campaign 会进 blocked")
    return client


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default="")
    parser.add_argument("--question-type", default="object_counting",
                        help="首个规范题型（v10 §12.1 暂定 object_counting）")
    parser.add_argument("--target-skill-id", default="S01")
    parser.add_argument("--mode", default="real", choices=["real", "mock_light"])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--source-library", default="skill_library",
                        help="S0 库（只读；campaign 在 --library-root 的副本里演进）")
    parser.add_argument("--library-root", default="data/v10_campaign/library")
    parser.add_argument("--run-root", default="data/v10_campaign/runs")
    parser.add_argument("--learning-split", default="induction")
    parser.add_argument("--inner-split", default="inner_validation")
    parser.add_argument("--learning-limit", type=int, default=12)
    parser.add_argument("--inner-panel-limit", type=int, default=8)
    parser.add_argument("--post-publish-limit", type=int, default=3)
    parser.add_argument("--min-eligible", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-root", default="")
    parser.add_argument("--vllm-endpoint", action="append", default=[])
    parser.add_argument("--vllm-model", default="qwen3vl-8b-r0")
    parser.add_argument("--offline-base-url", default="",
                        help="**显式覆盖**冻结的离线 endpoint（会被如实记进收据）")
    parser.add_argument("--offline-model", default="")
    parser.add_argument("--offline-api-key-env", default="",
                        help="从该环境变量取离线密钥（默认 DEEPSEEK_API_KEY）")
    parser.add_argument("--offline-env-file", default="",
                        help="从该文件 source DEEPSEEK_* 环境变量（本机私有凭据文件；"
                             "内容不进收据、不落日志）")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-generations", type=int, default=0,
                        help="覆盖配置里的 max_evolution_rounds（0 = 用配置值）；"
                             "先跑 1 代验证、之后用 2 代 resume 是允许的")
    parser.add_argument("--dry-run", action="store_true",
                        help="只冻结面板与库，不跑任何 episode")
    parser.add_argument("--panels-json", default="",
                        help="冻结后的面板清单落盘路径（默认 <run-root>/<id>/panels.json）")
    args = parser.parse_args(argv)
    if args.offline_env_file:
        _source_env_file(Path(args.offline_env_file))

    cfg_yaml = load_config(args.config)
    admission = load_yaml(cfg_yaml.get("admission_thresholds",
                                       "configs/admission_thresholds.yaml"))
    # §14.2/§14.3：轮数与 seed 集合**从配置真读**（v9 的"死配置"问题）
    max_generations = int(args.max_generations or cfg_yaml.get("max_evolution_rounds", 2))
    seeds = tuple(int(s) for s in (cfg_yaml.get("candidate_validation_seeds") or [0, 1]))
    policy = retrieval_policy_from_config(cfg_yaml)

    campaign_id = args.campaign_id or "camp-v10-001"
    panels = FrozenPanels(cfg_yaml, args)
    manifest = panels.freeze()
    panels_path = Path(args.panels_json or (Path(args.run_root) / campaign_id / "panels.json"))
    panels_path.parent.mkdir(parents=True, exist_ok=True)
    panels_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                      sort_keys=True) + "\n", encoding="utf-8")
    print(f"[面板冻结] {panels_path}")
    print(f"  learning n={len(manifest['learning_qa_ids'])} "
          f"scenes={len(manifest['learning_scenes'])}")
    print(f"  inner_g0 n={len(manifest['inner_g0_qa_ids'])} "
          f"inner_g1 n={len(manifest['inner_g1_qa_ids'])} "
          f"互不重叠={manifest['inner_panels_disjoint']}")
    print(f"  post_publish n={len(manifest['post_publish_qa_ids'])}")
    print(f"  配置：max_evolution_rounds={max_generations} "
          f"candidate_validation_seeds={list(seeds)} "
          f"top_k={policy.top_k} method_context_max_chars="
          f"{policy.method_context_max_chars}")

    seeded = seed_campaign_library(args.source_library, args.library_root)
    print(f"[Skill 库] 复制 {len(seeded['copied'])} 项 → {seeded['target']}")

    if args.dry_run:
        print("[dry-run] 不跑任何 episode。")
        return 0
    if not manifest["inner_panels_disjoint"]:
        print("[警告] inner_g0 / inner_g1 的 scene 有重叠：按 §8.5 必须**降低结论等级**"
              "（收据里如实记录）")

    runner = EvolutionCampaignRunner(
        CampaignConfig(
            campaign_id=campaign_id,
            target_question_type=args.question_type,
            target_skill_id=args.target_skill_id,
            max_generations=max_generations,
            validation_seeds=seeds,
            mode=args.mode,
            library_root=args.library_root,
            run_root=args.run_root,
            learning_split=args.learning_split,
            inner_split=args.inner_split,
            learning_limit=args.learning_limit,
            inner_panel_limit=args.inner_panel_limit,
            post_publish_probe_limit=args.post_publish_limit,
            min_eligible_experiences=int(args.min_eligible),
            method_context_max_chars=int(policy.method_context_max_chars),
            retrieval_policy=policy,
            resume=not args.no_resume),
        offline_client=_make_offline(args) if args.mode == "real" else None,
        llm=_make_online_llm(args),
        panels_provider=panels.provider,
        tool_names=REGISTRY.names(),
        qa_ids_by_split={"learning": panels.qa_ids("learning")},
        print_fn=print)
    campaign = runner.run()
    print("\n" + "=" * 78)
    print(json.dumps(campaign.model_dump(mode="json"), ensure_ascii=False, indent=2))
    print(f"收据目录: {Path(args.run_root) / campaign_id}")
    return 0 if campaign.completion_status.startswith("completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
