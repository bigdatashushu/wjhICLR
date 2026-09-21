"""G-09 四层 split 构建：scene-whole 分层切分 + 污染断言 + 落盘（§4 M1 / §13.3）。

离线一次性脚本的实现体（CLI 见 `scripts/build_splits.py`）。产出：

- `configs/vsi_bench_split.yaml`：`DataSplitConfig` 四列表 + 分层证据表；
- `configs/contamination_check.log`：断言记录与分层表，供论文复现引用。

纪律：
- 硬约束 19：按 `scene_id + task_type` 分层切分，**不共享 scene**；
- 硬约束 9：`final_test` 完全隔离，仅记录存在，不进任何在线/离线流程；
- 起点是**整 scene 分配**（不是整 QA 行分配）：否则同一 scene 会横跨两层，
  重建产物与场景先验即构成跨层泄漏。

分层算法（确定性，与 seed 绑定）：
1. 稀有 `question_type` 先行：按其 scene 数升序处理各任务类型，保证
   `object_rel_direction_easy`（仅 76 scene）这类稀缺题型在每层都有代表；
2. 每个 scene 归属"目标缺口最大"的桶：`缺口 = 目标份额×该题型 scene 数 − 已分配数`，
   平票按"当前总分配数最少"再按桶名破（避免系统性偏向第一个桶）；
3. 两趟复用同一算法：先切出 `final_test`（留出比例），再在剩余 pool 上按
   induction:inner:outer 切分。
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from skill3d.schemas.episode import DataSplitConfig

# final_test 留出比例（TODO_CALIBRATE，§4 M1 字段 12；"5:3:2:留出"中的"留出"）
FINAL_RATIO: float = 0.2
# induction:inner:outer 在非 final pool 内的起始比例（TODO_CALIBRATE，§4 M1）
SPLIT_RATIOS: dict[str, float] = {
    "induction": 0.5,
    "inner_validation": 0.3,
    "outer_holdout": 0.2,
}
SPLIT_KEYS: dict[str, str] = {
    "induction": "induction_scene_ids",
    "inner_validation": "inner_validation_scene_ids",
    "outer_holdout": "outer_holdout_scene_ids",
    "final_test": "final_test_scene_ids",
}
SPLIT_ORDER = ("induction", "inner_validation", "outer_holdout", "final_test")


class SplitBuildError(RuntimeError):
    """split 构建失败（meta 缺失、scene 冲突、分层不满足断言等）。"""


@dataclass
class SceneInfo:
    """单个 scene 的分层信息。"""

    scene_name: str
    dataset: str
    question_types: set[str] = field(default_factory=set)
    n_qa: int = 0


@dataclass
class SplitReport:
    """分层证据表（写进 YAML 与 contamination log，供论文引用）。"""

    seed: int
    final_ratio: float
    n_rows: int
    n_scenes: int
    dataset_scene_counts: dict[str, int]
    per_split_scenes: dict[str, int]
    per_split_qa: dict[str, int]
    per_task_scenes: dict[str, dict[str, int]]      # task_type -> split -> scene 数
    per_task_qa: dict[str, dict[str, int]]          # task_type -> split -> QA 数
    ratios: dict[str, float]
    meta_source: str
    meta_sha256: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "final_ratio": self.final_ratio,
            "n_rows": self.n_rows,
            "n_scenes": self.n_scenes,
            "dataset_scene_counts": dict(sorted(self.dataset_scene_counts.items())),
            "per_split_scenes": dict(self.per_split_scenes),
            "per_split_qa": dict(self.per_split_qa),
            "per_task_scenes": {k: dict(v) for k, v in sorted(self.per_task_scenes.items())},
            "per_task_qa": {k: dict(v) for k, v in sorted(self.per_task_qa.items())},
            "ratios": dict(self.ratios),
            "meta_source": self.meta_source,
            "meta_sha256": self.meta_sha256,
            "warnings": list(self.warnings),
        }


# ------------------------------------------------------------------ meta 读取 ----

def load_local_meta(path: str | Path) -> list[dict]:
    """读本地导出的 VSI-Bench meta（jsonl / json / csv），免依赖 `datasets`。

    官方 HF 仓库可直接下载 `test.jsonl`（5130 行 / 288 scene，§1.2 已核验规模）。
    """
    p = Path(path)
    if not p.is_file():
        raise SplitBuildError(f"meta 文件不存在: {p}")
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    elif suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("rows", [])
    elif suffix == ".csv":
        import csv

        with p.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise SplitBuildError(f"不支持的 meta 格式: {p.suffix}（用 .jsonl/.json/.csv）")
    if not rows:
        raise SplitBuildError(f"meta 为空: {p}")
    return rows


def meta_sha256(path: str | Path) -> str:
    """meta 内容哈希（写进 split_version / run manifest，保证切分可追溯）。"""
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def scene_inventory(rows: Iterable[dict]) -> dict[str, SceneInfo]:
    """按 `scene_name` 去重聚合题型与 QA 数（§4 M1：按 scene_name 去重）。

    scene 名跨数据集冲突时抛错而不是静默合并：合并会让两层共用同一 scene 名，
    破坏硬约束 19 的"不共享 scene"。
    """
    inv: dict[str, SceneInfo] = {}
    for r in rows:
        name = str(r["scene_name"])
        ds = str(r.get("dataset", "unknown"))
        info = inv.get(name)
        if info is None:
            info = inv[name] = SceneInfo(scene_name=name, dataset=ds)
        elif info.dataset != ds:
            raise SplitBuildError(
                f"scene 名跨数据集冲突: {name} ∈ {{{info.dataset}, {ds}}}；"
                "请改用 dataset/scene 复合键并同步 video_path_for 约定"
            )
        info.question_types.add(str(r["question_type"]))
        info.n_qa += 1
    if not inv:
        raise SplitBuildError("meta 中无有效 scene")
    return inv


def qa_counts_per_scene(rows: Iterable[dict]) -> dict[tuple[str, str], int]:
    """(scene, question_type) -> QA 行数，用于分层证据表的 QA 侧统计。"""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        counts[(str(r["scene_name"]), str(r["question_type"]))] += 1
    return dict(counts)


# ------------------------------------------------------------------ 分层算法 ----

def _stratified_assign(
    inventory: dict[str, SceneInfo],
    scenes: Sequence[str],
    fractions: dict[str, float],
    rng: random.Random,
) -> dict[str, str]:
    """把 `scenes` 整 scene 分配到 `fractions` 指定的各桶（scene-whole 分层贪心）。

    见模块 docstring 的算法说明。返回 scene -> bucket。
    """
    buckets = list(fractions)
    if len(buckets) < 2:
        raise SplitBuildError("至少需要两个桶")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise SplitBuildError(f"桶份额之和须为 1.0，当前 {sum(fractions.values())}")

    by_type: dict[str, list[str]] = defaultdict(list)
    for s in scenes:
        for t in inventory[s].question_types:
            by_type[t].append(s)
    if not by_type:
        raise SplitBuildError("无可用 question_type（分层依据缺失）")

    shuffled: dict[str, list[str]] = {}
    for t, lst in by_type.items():
        copy = sorted(lst)
        rng.shuffle(copy)
        shuffled[t] = copy

    assigned: dict[str, str] = {}
    total_in_bucket: dict[str, int] = {b: 0 for b in buckets}
    counts: dict[str, dict[str, int]] = {t: {b: 0 for b in buckets} for t in by_type}

    # 稀有题型先行（scene 数少 → 代表性地板低，先分才不会被大题型挤掉）
    for t in sorted(by_type, key=lambda k: (len(by_type[k]), k)):
        cand = shuffled[t]
        n_t = len(cand)
        target = {b: fractions[b] * n_t for b in buckets}
        for s in cand:
            if s in assigned:
                continue
            pending = [b for b in buckets if target[b] - counts[t][b] > 1e-9]
            pool = pending or buckets
            # 缺口最大者优先；平票取总分配数最少者；再平票按桶名（确定性）
            bucket = min(
                pool,
                key=lambda b: (-(target[b] - counts[t][b]), total_in_bucket[b], b),
            )
            assigned[s] = bucket
            total_in_bucket[bucket] += 1
            for t2 in inventory[s].question_types:
                if bucket in counts.get(t2, {}):
                    counts[t2][bucket] += 1
    missing = [s for s in scenes if s not in assigned]
    if missing:
        raise SplitBuildError(f"分层未覆盖 {len(missing)} 个 scene，例如 {missing[:5]}")
    return assigned


def build_split_config(
    rows: Sequence[dict],
    *,
    seed: int = 0,
    final_ratio: float = FINAL_RATIO,
    final_scene_ids: Optional[Sequence[str]] = None,
    ratios: Optional[dict[str, float]] = None,
    meta_source: str = "",
    meta_hash: str = "",
    log_ref: str = "",
) -> tuple[DataSplitConfig, SplitReport]:
    """构建四层 split（§4 M1）。返回 (DataSplitConfig, 分层证据表)。

    `final_scene_ids` 显式给出时按用户指定留出；否则按 `final_ratio` 分层留出。
    `final_ratio=0` 且不给显式列表 → final_test 为空（由用户在后续版本单独划入）。
    """
    ratio_map = dict(ratios or SPLIT_RATIOS)
    if abs(sum(ratio_map.values()) - 1.0) > 1e-9:
        raise SplitBuildError(f"induction/inner/outer 份额之和须为 1.0，当前 {sum(ratio_map.values())}")
    inventory = scene_inventory(rows)
    all_scenes = sorted(inventory)
    rng = random.Random(seed)
    warnings: list[str] = []

    # ---- 第一趟：留出 final_test ----
    final_scenes: list[str] = []
    if final_scene_ids is not None:
        unknown = sorted(set(final_scene_ids) - set(all_scenes))
        if unknown:
            raise SplitBuildError(f"显式 final scene 不在 meta 中: {unknown[:5]}")
        final_scenes = sorted(set(final_scene_ids))
    elif final_ratio > 0:
        picked = _stratified_assign(
            inventory, all_scenes, {"final_test": final_ratio, "rest": 1.0 - final_ratio}, rng)
        final_scenes = sorted(s for s, b in picked.items() if b == "final_test")

    # ---- 第二趟：在剩余 pool 上切 induction/inner/outer ----
    remaining = [s for s in all_scenes if s not in set(final_scenes)]
    if not remaining:
        raise SplitBuildError("final_test 占满全部 scene，无法切分其余三层")
    picked = _stratified_assign(inventory, remaining, ratio_map, rng)
    by_split: dict[str, list[str]] = {k: [] for k in SPLIT_KEYS}
    by_split["final_test"] = list(final_scenes)
    for s, b in picked.items():
        by_split[b].append(s)
    for k in by_split:
        by_split[k] = sorted(by_split[k])

    cfg = DataSplitConfig(
        induction_scene_ids=by_split["induction"],
        inner_validation_scene_ids=by_split["inner_validation"],
        outer_holdout_scene_ids=by_split["outer_holdout"],
        final_test_scene_ids=by_split["final_test"],
        task_type_stratification=True,
        split_version="",
        contamination_check_log_ref=log_ref,
    )

    # ---- 证据表 ----
    qa = qa_counts_per_scene(rows)
    per_task_scenes: dict[str, dict[str, int]] = {}
    per_task_qa: dict[str, dict[str, int]] = {}
    for s, info in inventory.items():
        owner = next((k for k in SPLIT_KEYS if s in set(by_split[k])), None)
        if owner is None:  # pragma: no cover - validate_split 会先报错
            raise SplitBuildError(f"scene {s} 未被分配")
        for t in sorted(info.question_types):
            per_task_scenes.setdefault(t, {k: 0 for k in SPLIT_KEYS})[owner] += 1
            per_task_qa.setdefault(t, {k: 0 for k in SPLIT_KEYS})[owner] += qa.get((s, t), 0)

    ds_counts: dict[str, int] = defaultdict(int)
    for info in inventory.values():
        ds_counts[info.dataset] += 1

    report = SplitReport(
        seed=seed,
        final_ratio=final_ratio if final_scene_ids is None else -1.0,
        n_rows=len(rows),
        n_scenes=len(all_scenes),
        dataset_scene_counts=dict(ds_counts),
        per_split_scenes={k: len(by_split[k]) for k in SPLIT_ORDER},
        per_split_qa={
            k: sum(qa.get((s, t), 0) for s in by_split[k] for t in inventory[s].question_types)
            for k in SPLIT_ORDER
        },
        per_task_scenes=per_task_scenes,
        per_task_qa=per_task_qa,
        ratios=dict(ratio_map),
        meta_source=meta_source,
        meta_sha256=meta_hash,
        warnings=warnings,
    )
    validate_split(cfg, inventory, report)
    cfg = cfg.model_copy(update={"split_version": split_version(cfg, report)})
    report.warnings = warnings
    return cfg, report


def split_version(cfg: DataSplitConfig, report: SplitReport) -> str:
    """split 版本指纹：四列表 + seed + meta 哈希（§16.4 split version 全记录）。"""
    key = repr((
        cfg.induction_scene_ids, cfg.inner_validation_scene_ids,
        cfg.outer_holdout_scene_ids, cfg.final_test_scene_ids,
        report.seed, report.meta_sha256,
    ))
    return "v1-" + hashlib.sha256(key.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ 断言 ----

def validate_split(cfg: DataSplitConfig, inventory: dict[str, SceneInfo],
                   report: Optional[SplitReport] = None) -> None:
    """四层切分硬断言（§4 M1 验收条件 c、硬约束 9/19）。

    1. 四列表两两不相交；
    2. 并集 == 全部 scene（无遗漏、无多余）；
    3. final_test 与其余三层不相交（`assert_final_test_isolation`）；
    4. 8 题型在 induction/inner/outer 三层各有代表（分层有效性）。
    """
    lists = {k: list(getattr(cfg, v)) for k, v in SPLIT_KEYS.items()}

    # 1) 两两不相交
    for i, a in enumerate(SPLIT_ORDER):
        for b in SPLIT_ORDER[i + 1:]:
            overlap = set(lists[a]) & set(lists[b])
            if overlap:
                raise SplitBuildError(f"split 相交: {a} ∩ {b} = {sorted(overlap)[:5]}")

    # 2) 并集完整
    union = set().union(*(set(v) for v in lists.values()))
    expected = set(inventory)
    missing, extra = expected - union, union - expected
    if missing or extra:
        raise SplitBuildError(
            f"split 覆盖不完整: 缺 {sorted(missing)[:5]}，多 {sorted(extra)[:5]}"
        )
    if sum(len(v) for v in lists.values()) != len(expected):
        raise SplitBuildError("scene 总数与分片数之和不一致")

    # 3) final_test 隔离（硬约束 9）
    from skill3d.adapters.vsibench_loader import assert_final_test_isolation

    assert_final_test_isolation(cfg)

    # 4) 分层有效性：每个题型在 induction/inner/outer 都有代表
    for t, per_split in (report.per_task_scenes.items() if report else []):
        empty = [k for k in ("induction", "inner_validation", "outer_holdout")
                 if per_split.get(k, 0) == 0]
        if empty:
            raise SplitBuildError(f"题型 {t} 在 {empty} 无 scene 代表（分层失效）")


# ------------------------------------------------------------------ 落盘 ----

def to_yaml_dict(cfg: DataSplitConfig, report: SplitReport) -> dict[str, Any]:
    """§13.3 YAML 结构 + 分层证据表（阈值一律标 TODO_CALIBRATE）。"""
    return {
        "split_version": cfg.split_version,
        "seed": report.seed,
        "final_ratio": report.final_ratio,          # TODO_CALIBRATE（负值 = 显式给定）
        "ratios": dict(report.ratios),              # TODO_CALIBRATE
        "task_type_stratification": cfg.task_type_stratification,
        "contamination_check_log_ref": cfg.contamination_check_log_ref,
        "meta_source": report.meta_source,
        "meta_sha256": report.meta_sha256,
        "n_rows": report.n_rows,
        "total_scenes": report.n_scenes,
        "dataset_scene_counts": dict(sorted(report.dataset_scene_counts.items())),
        "per_split_scenes": dict(report.per_split_scenes),
        "per_split_qa": dict(report.per_split_qa),
        "induction_scene_ids": cfg.induction_scene_ids,
        "inner_validation_scene_ids": cfg.inner_validation_scene_ids,
        "outer_holdout_scene_ids": cfg.outer_holdout_scene_ids,
        "final_test_scene_ids": cfg.final_test_scene_ids,
        "per_task_scenes": {k: dict(v) for k, v in sorted(report.per_task_scenes.items())},
        "per_task_qa": {k: dict(v) for k, v in sorted(report.per_task_qa.items())},
    }


def _yaml_dump(data: dict[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False,
                          width=100)


def write_split_yaml(cfg: DataSplitConfig, report: SplitReport, out_path: str | Path) -> Path:
    """写 `configs/vsi_bench_split.yaml`（§13.3）。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# 由 scripts/build_splits.py 生成（G-09，§4 M1 / §13.3）——请勿手工编辑。\n"
        "# 纪律：按 scene 整块切分、四层不共享 scene（硬约束 19）；final_test 完全隔离\n"
        "#（硬约束 9，仅记录存在）。所有比例/阈值 TODO_CALIBRATE。\n"
    )
    out.write_text(header + _yaml_dump(to_yaml_dict(cfg, report)), encoding="utf-8")
    return out


def write_contamination_log(cfg: DataSplitConfig, report: SplitReport,
                            log_path: str | Path, *, command: str = "") -> Path:
    """写 `configs/contamination_check.log`：断言记录 + 分层表（论文复现引用）。"""
    out = Path(log_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add("VSI-Bench 四层 split 污染检查（G-09）")
    add("=" * 78)
    add(f"command            : {command or 'scripts/build_splits.py'}")
    add(f"meta_source        : {report.meta_source}")
    add(f"meta_sha256        : {report.meta_sha256}")
    add(f"split_version      : {cfg.split_version}")
    add(f"seed               : {report.seed}")
    add(f"final_ratio        : {report.final_ratio}  # TODO_CALIBRATE")
    add(f"ratios(ind/in/out) : {report.ratios}  # TODO_CALIBRATE")
    add(f"rows / scenes      : {report.n_rows} / {report.n_scenes}")
    add(f"dataset scenes     : {dict(sorted(report.dataset_scene_counts.items()))}")
    add("")
    add("-- 断言（全部通过才会写出本文件）".ljust(78, "-"))
    for i, (name, ok, detail) in enumerate(_checks(cfg, report), 1):
        add(f"  [{i}] {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    add("")
    add("-- 每层规模".ljust(78, "-"))
    add(f"{'split':18s}{'scenes':>8s}{'qa':>8s}")
    for k in SPLIT_ORDER:
        add(f"{k:18s}{report.per_split_scenes[k]:8d}{report.per_split_qa[k]:8d}")
    add(f"{'TOTAL':18s}{sum(report.per_split_scenes.values()):8d}"
        f"{sum(report.per_split_qa.values()):8d}")
    add("")
    add("-- 题型 × split（scene 数 / QA 数）".ljust(78, "-"))
    header = f"{'question_type':28s}" + "".join(f"{k[:9]:>12s}" for k in SPLIT_ORDER)
    add(header)
    for t in sorted(report.per_task_scenes):
        row = f"{t:28s}"
        for k in SPLIT_ORDER:
            row += f"{report.per_task_scenes[t].get(k, 0):5d}/{report.per_task_qa[t].get(k, 0):<6d}"
        add(row)
    add("")
    add("-- final_test 隔离（硬约束 9）".ljust(78, "-"))
    final = set(cfg.final_test_scene_ids)
    for name in ("induction_scene_ids", "inner_validation_scene_ids", "outer_holdout_scene_ids"):
        inter = final & set(getattr(cfg, name))
        add(f"  final_test ∩ {name:28s} = {sorted(inter) if inter else '∅'}")
    add(f"  结论: final_test 与 induction/inner/outer 无交集；"
        f"仅 {len(final)} 个 scene 留出，不进任何在线/离线流程")
    add("")
    add("-- scene 清单（按层）".ljust(78, "-"))
    for k in SPLIT_ORDER:
        ids = getattr(cfg, SPLIT_KEYS[k])
        add(f"  {k} ({len(ids)}): {', '.join(ids)}")
    add("")
    if report.warnings:
        add("-- 警告".ljust(78, "-"))
        for w in report.warnings:
            add(f"  ! {w}")
        add("")
    add("=" * 78)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _checks(cfg: DataSplitConfig, report: SplitReport) -> list[tuple[str, bool, str]]:
    """污染检查条目：(名称, 是否通过, 说明)。"""
    lists = {k: getattr(cfg, v) for k, v in SPLIT_KEYS.items()}
    checks: list[tuple[str, bool, str]] = []

    pairwise = True
    detail = "四列表两两不相交"
    for i, a in enumerate(SPLIT_ORDER):
        for b in SPLIT_ORDER[i + 1:]:
            inter = set(lists[a]) & set(lists[b])
            if inter:
                pairwise = False
                detail = f"{a} ∩ {b} = {sorted(inter)[:5]}"
    checks.append(("四层互斥", pairwise, detail))

    total = sum(len(v) for v in lists.values())
    checks.append(("scene 总数", total == report.n_scenes,
                   f"{total} == {report.n_scenes}"))
    checks.append(("每层非空", all(len(v) > 0 for v in lists.values()),
                   str({k: len(v) for k, v in lists.items()})))
    checks.append(("final_test 隔离",
                   not (set(cfg.final_test_scene_ids)
                        & (set(cfg.induction_scene_ids)
                           | set(cfg.inner_validation_scene_ids)
                           | set(cfg.outer_holdout_scene_ids))),
                   f"final_test n={len(cfg.final_test_scene_ids)}"))
    strat_ok, strat_detail = True, "8 题型在 ind/in/out 三层均有 scene"
    for t, per in report.per_task_scenes.items():
        empty = [k for k in ("induction", "inner_validation", "outer_holdout")
                 if per.get(k, 0) == 0]
        if empty:
            strat_ok = False
            strat_detail = f"题型 {t} 缺 {empty}"
            break
    checks.append(("task_type 分层", strat_ok, strat_detail))
    return checks


def load_split_config(path: str | Path) -> DataSplitConfig:
    """从 YAML 读回 `DataSplitConfig`（忽略证据表等附加键）。"""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SplitBuildError(f"split 配置格式错误（应为 mapping）: {path}")
    known = set(DataSplitConfig.model_fields)
    return DataSplitConfig(**{k: v for k, v in data.items() if k in known})
