"""HC34 Experiment Readiness Gate：四级证据门 + `readiness_manifest.json`（§11.1）。

**为什么需要它**：本项目最容易犯的错误是把"代码存在"当成"能力已具备"，进而把
未跑通/未标定/mock 的输出写进论文主表。本模块把这件事变成可判定的数据：

| 层级 | 必要证据 | 不足时禁止声称 |
|---|---|---|
| `implemented` | 实码存在、非 TODO 桩，单测覆盖核心合同 | "已实现" |
| `connected` | 存在非自身、非测试的生产调用者，配置开关实际生效 | "系统已具备" |
| `real_poc_verified` | 冻结真实环境/模型/数据上通过预定义 PoC，保存 receipt | "已跑通" |
| `paper_eligible` | 数据隔离、统计门槛、≥3 seed、无 mock/泄漏、复现信息齐全 | 进主表 |

四级是**单调布尔**（HC34）：`real_poc_verified ⇒ connected ⇒ implemented`。
状态只能由证据推进，且每次变化都要落 `readiness_manifest.json`（含 evidence refs、
时间、代码提交、数据 split hash、blockers），并可被 RunManifest 引用。

本模块**只读事实**：它不跑实验、不判定能力好坏，只把当前实况与阻断项结构化，
使"不得进主表"这件事在代码层面可查（`assert_paper_eligible`）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from skill3d.schemas.readiness import ExperimentReadiness

READINESS_MANIFEST_VERSION = "readiness-manifest-v1"
DEFAULT_MANIFEST_PATH = "data/readiness_manifest.json"
# §7 主表要求：≥3 个 seed（mean±std）
PAPER_MIN_SEEDS = 3


@dataclass
class ReadinessManifest:
    """一次项目级 readiness 快照（§11.1）。"""

    capabilities: list[ExperimentReadiness] = field(default_factory=list)
    generated_at: str = ""
    code_commit: str = ""
    data_split_hash: str = ""
    scale_calibration_id: str = ""
    notes: list[str] = field(default_factory=list)

    # ---- 查询 ----
    def get(self, capability: str) -> Optional[ExperimentReadiness]:
        for c in self.capabilities:
            if c.capability == capability:
                return c
        return None

    def set(self, readiness: ExperimentReadiness) -> None:
        self.capabilities = [c for c in self.capabilities
                             if c.capability != readiness.capability] + [readiness]

    def paper_eligible(self) -> list[str]:
        return sorted(c.capability for c in self.capabilities if c.paper_eligible)

    def blocked(self) -> list[dict]:
        """尚未 paper_eligible 的能力及其阻断项（论文"限制"小节的直接来源）。"""
        return [{"capability": c.capability,
                 "implemented": c.implemented, "connected": c.connected,
                 "real_poc_verified": c.real_poc_verified,
                 "paper_eligible": c.paper_eligible,
                 "blockers": list(c.blockers)}
                for c in sorted(self.capabilities, key=lambda x: x.capability)
                if not c.paper_eligible]

    def assert_paper_eligible(self, capability: str) -> None:
        """写论文主表前的硬门（HC34）：未达四级一律抛错。"""
        c = self.get(capability)
        if c is None:
            raise ReadinessGateError(
                f"能力 {capability!r} 未登记 readiness 状态 —— "
                "未登记 = 未验证，不得写入主表（HC34）")
        if not c.paper_eligible:
            raise ReadinessGateError(
                f"能力 {capability!r} 未达 paper_eligible（"
                f"implemented={c.implemented} connected={c.connected} "
                f"real_poc_verified={c.real_poc_verified}）；"
                f"blockers={c.blockers} —— 按 HC34 不得写入主表")

    def seed_requirement_ok(self, n_seeds: int) -> bool:
        """主表需 ≥3 seed（§7）；不足即不得报 mean±std 主结果。"""
        return int(n_seeds) >= PAPER_MIN_SEEDS

    # ---- 序列化 ----
    def to_dict(self) -> dict:
        return {
            "schema_version": READINESS_MANIFEST_VERSION,
            "generated_at": self.generated_at,
            "code_commit": self.code_commit,
            "data_split_hash": self.data_split_hash,
            "scale_calibration_id": self.scale_calibration_id,
            "notes": list(self.notes),
            "capabilities": [c.model_dump() for c in self.capabilities],
            "summary": {
                "n_capabilities": len(self.capabilities),
                "paper_eligible": self.paper_eligible(),
                "blocked": [b["capability"] for b in self.blocked()],
                "seed_requirement_met": self.seed_requirement_ok(PAPER_MIN_SEEDS),
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReadinessManifest":
        ver = str(data.get("schema_version", ""))
        if ver != READINESS_MANIFEST_VERSION:
            raise ReadinessGateError(
                f"readiness manifest schema 版本不符: {ver!r} != "
                f"{READINESS_MANIFEST_VERSION!r}")
        return cls(
            capabilities=[ExperimentReadiness.model_validate(c)
                          for c in data.get("capabilities") or []],
            generated_at=str(data.get("generated_at", "")),
            code_commit=str(data.get("code_commit", "")),
            data_split_hash=str(data.get("data_split_hash", "")),
            scale_calibration_id=str(data.get("scale_calibration_id", "")),
            notes=list(data.get("notes") or []),
        )


class ReadinessGateError(RuntimeError):
    """Readiness 门未过（HC34）：不得写主表 / manifest 版本不符。"""


def write_readiness_manifest(manifest: ReadinessManifest,
                             path: str | Path = DEFAULT_MANIFEST_PATH) -> Path:
    """原子落盘 `readiness_manifest.json`（§11.1：状态变化必须留档）。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(out)
    return out


def load_readiness_manifest(path: str | Path = DEFAULT_MANIFEST_PATH
                            ) -> Optional[ReadinessManifest]:
    """读 `readiness_manifest.json`；不存在返回 None（不伪造空 manifest）。"""
    p = Path(path)
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return ReadinessManifest.from_dict(data)


def split_hash(scene_ids: Iterable[str]) -> str:
    """数据 split 的内容哈希（进 manifest：证明用的是哪批数据）。"""
    payload = json.dumps(sorted({str(i) for i in scene_ids}), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def capability_readiness(
    capability: str,
    *,
    implemented: bool,
    connected: bool = False,
    real_poc_verified: bool = False,
    paper_eligible: bool = False,
    evidence_refs: Optional[Sequence[str]] = None,
    blockers: Optional[Sequence[str]] = None,
    code_commit: str = "",
    data_split_hash: str = "",
    updated_at: str = "",
    notes: str = "",
) -> ExperimentReadiness:
    """构造一项能力的 readiness（单调性由 Schema 强制）。"""
    return ExperimentReadiness(
        capability=capability, implemented=implemented, connected=connected,
        real_poc_verified=real_poc_verified, paper_eligible=paper_eligible,
        evidence_refs=list(evidence_refs or []),
        blockers=list(blockers or []),
        code_commit=code_commit, data_split_hash=data_split_hash,
        updated_at=updated_at, notes=notes,
    )


# ------------------------------------------------- v5 当前实况（§10.5 基线）----
# 以下是**当前实况快照**，不是目标态。任何升级都必须附真实 evidence_refs；
# 不得为了让代码"看起来完成"而手动置真（§11.1：状态只允许由证据推进）。
CURRENT_BASELINE: tuple[tuple[str, bool, bool, bool, bool, tuple[str, ...]], ...] = (
    # capability, implemented, connected, real_poc_verified, paper_eligible, blockers
    ("vggt_feed_forward_mainline", True, True, False, False,
     ("单 episode 跑通；需 ≥3 seed 真实端到端 + 与基线统计比较",)),
    # ---- v6 §20：官方 VGGSfM BA 已退出生产（历史失败实验，保留 implemented=true）----
    # 状态纪律：connected/real_poc_verified/paper_eligible 全 false，理由码
    # `rejected_on_24g_oom`；代码只保留在 skill3d/legacy/retired/legacy_vggsfm_ba/
    # （只读 + 复现，运行时代码不得 import）。
    ("official_vggsfm_ba", True, False, False, False,
     ("rejected_on_24g_oom（HC35）：官方 VGGSfM tracker 在 predict_tracks 内峰值 ~21–23 GiB "
      "且与输入无关（分辨率 280²–518²、帧数 4–32、query 256–2048 均命中同一峰值），"
      "24 GiB 卡装不下；v5 起不得作为生产 route / 完成条件 / 主结果前置条件。"
      "历史码与失败 receipt 保留在 skill3d/legacy/retired/legacy_vggsfm_ba/（v6 §20 废止，"
      "任何运行时代码 import 即实现错误）",)),
    # ---- v6 §20：`vggt_sparse_ba` 已按止损纪律整体废止，替代物为「无」（G5 永久 not_available）----
    ("vggt_sparse_ba", True, False, False, False,
     ("实现进度：pair graph / track merge / receipt / L0 合同测试 + 前端"
      "（SuperPoint 或 ALIKED + LightGlue）与 PyCOLMAP 后端均已落码 → implemented=true；"
      "但 **L1 单 episode 实测 `rejected_on_l1_gate`**：LightGlue 在 pair=(0,1) 断言失败"
      "（`skip_reason=matching: …`），未产出有限 G5。v6 §20 已把该机制整体废止"
      "（代码归档于 skill3d/legacy/retired/sparse_ba/，无替代物，G5 永久 not_available），"
      "connected/real_poc_verified/paper_eligible 恒 false",)),
    # ---- v5 HC39：版本隔离基础设施 ----
    ("v5_schema_and_legacy_isolation", True, True, False, False,
     ("Schema 5.0（schema_version/quality_metric_version/reprojection_status/G5 三态）与 "
      "legacy readers 已实现并有负向测试；real_poc_verified 需真实 GPU 端到端产物按 v5 "
      "Schema 落盘后复核",)),
    ("v5_golden", True, True, False, False,
     ("v5 golden 已由 v5 pipeline 重生成（原 `tests/golden/v5/`，golden_version=v5-golden-1，"
      "无 G8、G5=None 不入分母），旧 golden 只读归档于 `tests/golden/archive_v4/` 并标 "
      "incomparable_with_v5=true（HC39）；混用旧 golden 有 hard-fail 测试。"
      "**v6 §20：v5 golden 与 v4 归档均已移入 `tests/archive_v5/`（不参与收集）。**"
      "real_poc_verified 待真实统计对比后推进",)),
    ("scale_recovery", True, True, False, False,
     ("v5 多锚点 + log-scale 融合 + conformal 校准池：缺 ARKitScenes 非重叠场景 GT 位姿"
      "（TODO_USER_INPUT）→ 无冻结校准器 → 未标定。**v6 §20 已整体废止该路线**，"
      "替代物为 `reconstruction/metric_fusion.py`（零样本度量深度跨帧融合，[待实验]）；"
      "本条能力在 v6 仅作历史登记，不再推进 real_poc_verified",)),
    ("metric_tasks_authorization", True, True, False, False,
     ("逐题型授权已接线，但 upper 层（scale_recovery）未达 real_poc_verified，"
      "故当前恒收回米制题型；v6 改由 MetricEvidenceGate + direct_vlm_routed 显式回退"
      "（§20 / D3）",)),
    ("tool_contract_replay", True, False, False, False,
     ("仅机制 + 集成测试；无真实 Qwen3-VL-8B 回灌修复率数字（门槛 ≥50%），默认关闭",)),
    ("g5_reprojection_semantics", True, True, False, False,
     ("HC37 / v6 §20：正式 vggt 主线固定 reprojection_status=not_available + G5=None"
      "（不聚合、无代理值），Schema 层 fail-closed 且有负向测试；待真实主线产物复核",)),
    ("dust3r_mast3r_baseline", False, False, False, False,
     ("v6 §20：DUSt3R/MASt3R 对照重建基线随 `recon_method` 收窄为 `Literal[\"vggt\"]` 一并"
      "废止（代码归档于 skill3d/legacy/retired/dust32_mast3r_fallback.py），无替代物；"
      "该项恒 implemented=false、不推进",)),
    ("gsplat_densification", False, False, False, False,
     ("桩：恒 return None",)),
    ("docker_sandbox", False, False, False, False,
     ("模板未接线；M10 走 in-process kernel（镜像 digest 未锁 TODO_USER_INPUT）",)),
    ("multi_seed_main_table", True, True, False, False,
     ("CLI 入口已接（evaluation/main_table.py：≥3 seed + HC24 门禁 + split 一致性）；"
      "real_poc_verified 待真实 ≥3 seed 结果",)),
    ("gpt6_offline_governance", True, False, False, False,
     ("GPT-6 endpoint/model_id/auth 缺失（TODO_USER_INPUT）→ 真实模式只能 QUARANTINE",)),
)


def baseline_manifest(*, generated_at: str = "", code_commit: str = "",
                      data_split_hash: str = "",
                      scale_calibration_id: str = "") -> ReadinessManifest:
    """v4 §10.5 实况基线的结构化版本（代码侧可查的"哪些还不能进主表"）。"""
    m = ReadinessManifest(generated_at=generated_at, code_commit=code_commit,
                          data_split_hash=data_split_hash,
                          scale_calibration_id=scale_calibration_id)
    for cap, impl, conn, poc, paper, blockers in CURRENT_BASELINE:
        m.set(capability_readiness(
            cap, implemented=impl, connected=conn, real_poc_verified=poc,
            paper_eligible=paper, blockers=list(blockers),
            code_commit=code_commit, data_split_hash=data_split_hash,
            updated_at=generated_at))
    m.notes.append(
        "本 manifest 为 §10.5 实况基线：四项全 true 才可进论文主表（HC34）。"
        "升级必须附真实 evidence_refs，不得手工置真。")
    return m


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI：写 / 打印 readiness manifest（`python -m skill3d.readiness.manifest`）。"""
    import argparse
    import subprocess

    p = argparse.ArgumentParser(description="Experiment Readiness Gate（HC34 / §11.1）")
    p.add_argument("--out", default=DEFAULT_MANIFEST_PATH,
                   help="readiness_manifest.json 落盘路径")
    p.add_argument("--check", default="",
                   help="断言某能力已 paper_eligible（不过则退出码 1）")
    p.add_argument("--code-commit", default="")
    args = p.parse_args(list(argv) if argv is not None else None)

    commit = args.code_commit
    if not commit:
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, timeout=30, check=True).stdout.strip()
        except Exception:  # noqa: BLE001 - 非 git 环境不阻断
            commit = "unknown"
    m = baseline_manifest(code_commit=commit)
    out = write_readiness_manifest(m, args.out)
    print(f"readiness manifest: {out}")
    print(f"{'capability':32s} impl conn poc paper  blockers")
    for c in sorted(m.capabilities, key=lambda x: x.capability):
        print(f"{c.capability:32s} {'Y' if c.implemented else '-':4s} "
              f"{'Y' if c.connected else '-':4s} {'Y' if c.real_poc_verified else '-':5s} "
              f"{'Y' if c.paper_eligible else '-':5s}  "
              f"{'；'.join(c.blockers)[:70]}")
    print(f"\npaper_eligible: {m.paper_eligible() or '（无）'}")
    if args.check:
        try:
            m.assert_paper_eligible(args.check)
        except ReadinessGateError as exc:
            print(f"[FAIL] {exc}", file=__import__("sys").stderr)
            return 1
        print(f"[OK] {args.check} 已达 paper_eligible")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())
