#!/usr/bin/env python3
"""G-09 四层 split 构建 CLI（§4 M1 / §13.3）。

```bash
# 用官方 test.jsonl（data/vsi_bench_meta/test.jsonl）切分并落盘
python scripts/build_splits.py

# 显式指定 meta 与留出比例
python scripts/build_splits.py --meta data/vsi_bench_meta/test.jsonl \
    --final-ratio 0.2 --seed 0 --out configs/vsi_bench_split.yaml
```

产出 `configs/vsi_bench_split.yaml`（四列表 + 分层证据表）与
`configs/contamination_check.log`（断言记录）。分层算法与打断言见
`skill3d.adapters.split_builder`。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:  # 免安装直接跑
    sys.path.insert(0, str(_SRC))

from skill3d.adapters.split_builder import (  # noqa: E402
    FINAL_RATIO,
    SPLIT_RATIOS,
    SplitBuildError,
    build_split_config,
    load_local_meta,
    meta_sha256,
    write_contamination_log,
    write_split_yaml,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts/build_splits.py",
        description="构建 VSI-Bench induction/inner/outer/final 四层 split（G-09）",
    )
    p.add_argument("--meta", default="data/vsi_bench_meta/test.jsonl",
                   help="QA meta 路径（.jsonl/.json/.csv；官方 test.jsonl 为 5130 行/288 scene）")
    p.add_argument("--seed", type=int, default=0, help="分层随机种子（与 split_version 绑定）")
    p.add_argument("--final-ratio", type=float, default=FINAL_RATIO,
                   help=f"final_test 留出 scene 比例（TODO_CALIBRATE，默认 {FINAL_RATIO}）")
    p.add_argument("--final-scenes-file", default="",
                   help="显式 final_test scene 清单（一行一个）；给出则忽略 --final-ratio")
    p.add_argument("--induction-ratio", type=float, default=SPLIT_RATIOS["induction"],
                   help="非 final pool 内 induction 份额（TODO_CALIBRATE）")
    p.add_argument("--inner-ratio", type=float, default=SPLIT_RATIOS["inner_validation"],
                   help="非 final pool 内 inner_validation 份额（TODO_CALIBRATE）")
    p.add_argument("--out", default="configs/vsi_bench_split.yaml")
    p.add_argument("--log", default="configs/contamination_check.log")
    p.add_argument("--dry-run", action="store_true", help="只打印分层结果，不落盘")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        rows = load_local_meta(args.meta)
    except SplitBuildError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        print("        下载官方 meta: curl -L -o data/vsi_bench_meta/test.jsonl "
              "https://huggingface.co/datasets/nyu-visionx/vsi-bench/resolve/main/test.jsonl",
              file=sys.stderr)
        return 2

    final_scenes = None
    if args.final_scenes_file:
        p = Path(args.final_scenes_file)
        if not p.is_file():
            print(f"[错误] final scene 清单不存在: {p}", file=sys.stderr)
            return 2
        final_scenes = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
                        if ln.strip()]

    ratios = {"induction": args.induction_ratio,
              "inner_validation": args.inner_ratio,
              "outer_holdout": 1.0 - args.induction_ratio - args.inner_ratio}
    try:
        cfg, report = build_split_config(
            rows,
            seed=args.seed,
            final_ratio=args.final_ratio,
            final_scene_ids=final_scenes,
            ratios=ratios,
            meta_source=args.meta,
            meta_hash=meta_sha256(args.meta),
            log_ref=args.log,
        )
    except SplitBuildError as exc:
        print(f"[错误] split 构建失败: {exc}", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f"meta={args.meta}  rows={report.n_rows}  scenes={report.n_scenes}  "
          f"seed={report.seed}  split_version={cfg.split_version}")
    print(f"dataset scenes: {dict(sorted(report.dataset_scene_counts.items()))}")
    print("-" * 78)
    print(f"{'split':18s}{'scenes':>8s}{'qa':>8s}")
    for k in ("induction", "inner_validation", "outer_holdout", "final_test"):
        print(f"{k:18s}{report.per_split_scenes[k]:8d}{report.per_split_qa[k]:8d}")
    print("-" * 78)
    print("题型 × split（scene 数）:")
    keys = ("induction", "inner_validation", "outer_holdout", "final_test")
    print(f"{'question_type':28s}" + "".join(f"{k[:9]:>11s}" for k in keys))
    for t in sorted(report.per_task_scenes):
        print(f"{t:28s}" + "".join(f"{report.per_task_scenes[t].get(k, 0):11d}" for k in keys))
    print("=" * 78)

    if args.dry_run:
        print("[dry-run] 未落盘")
        return 0

    command = "scripts/build_splits.py " + " ".join(sys.argv[1:])
    out_yaml = write_split_yaml(cfg, report, args.out)
    out_log = write_contamination_log(cfg, report, args.log, command=command)
    print(f"已写 {out_yaml}")
    print(f"已写 {out_log}")
    print(f"final_test 隔离断言通过：{len(cfg.final_test_scene_ids)} 个 scene 留出，"
          "不进任何在线/离线流程（硬约束 9）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
