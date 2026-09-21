"""G-62/G-63/G-64 消融矩阵：C0–C5（在线）/ E0–E5（演进沙箱）/ G0–G2（治理）。

§16.1 的三张消融表在此成为**可执行的配置**：每档解析为一组特性开关，
由 `online.eval`（C 档）与 `evolution.offline_driver`（E/G 档）消费。

C 档（在线链，§16.1）：
| 档 | 说明 | 落地方式 |
|---|---|---|
| C0 | direct VLM 直答，无 Tool | `--baseline C0_direct_vlm` |
| C1 | 生成 program 编排 Tool，无 Skill | `--baseline C1_tools_program` |
| C2 | 静态人工 Skill（无归纳） | `--skill-spec path.json` |
| C3 | 归纳后用，无 paired A/B 准入 | driver `--single-arm` |
| C4 | 完整归纳 + paired 验证（主线） | driver 默认 |
| C5 | 注入已知错误 Skill，测回退/退化 | `--inject-wrong-skill` |

E 档（演进沙箱，§8.3）：E0 仅隔离 / E1 +replay / E2 +反例 / E3 +MR / E4 +paired / E5 全开。

G 档（治理，§8.4）：G0 全治理 / G1 无 review / G2 无监控。

纪律：消融只改"机制开关"，不改模型、split、seed 与重建产物；
C3/C5 与 E 档的结果**不得**用作主结果，只用于消融表（§16.1）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ------------------------------------------------------------------ C 档（在线）----

C_ABLATIONS: dict[str, dict] = {
    "C0_direct_vlm": {
        "baseline": "C0_direct_vlm", "skills": "none",
        "desc": "direct VLM 直答（无 Tool、无 program）"},
    "C1_tools_program": {
        "baseline": "C1_tools_program", "skills": "none",
        "desc": "+Tools+Program，无 Skill 模板"},
    "C2_static_skill": {
        "baseline": "C1_tools_program", "skills": "static_spec",
        "desc": "静态人工 Skill（手写 SkillSpec，无自动归纳）"},
    "C3_induction_no_ab": {
        "baseline": "C1_tools_program", "skills": "induced",
        "desc": "GPT-6 归纳后直接使用，无 paired A/B 准入",
        "admission": "single_arm"},
    "C4_full": {
        "baseline": "C1_tools_program", "skills": "induced",
        "desc": "完整归纳 + paired 验证 + 硬门准入（主线）",
        "admission": "paired"},
    "C5_wrong_skill": {
        "baseline": "C1_tools_program", "skills": "wrong_injected",
        "desc": "注入已知错误 Skill，观察回退/退化（可证伪 §17.5 #5）"},
}

# ------------------------------------------------------------------ E 档（演进）----

E_ABLATIONS: dict[str, dict] = {
    "E0": {"isolation": True, "replay": False, "counterexample": False,
           "metamorphic": False, "paired": False,
           "desc": "普通隔离执行，无 fork/replay/反例/配对"},
    "E1": {"isolation": True, "replay": True, "counterexample": False,
           "metamorphic": False, "paired": False, "desc": "+state replay"},
    "E2": {"isolation": True, "replay": True, "counterexample": True,
           "metamorphic": False, "paired": False, "desc": "+反例自动挖掘"},
    "E3": {"isolation": True, "replay": True, "counterexample": False,
           "metamorphic": True, "paired": False, "desc": "+5 类空间 MR 门"},
    "E4": {"isolation": True, "replay": True, "counterexample": False,
           "metamorphic": False, "paired": True, "desc": "+paired 分支"},
    "E5": {"isolation": True, "replay": True, "counterexample": True,
           "metamorphic": True, "paired": True, "desc": "完整 Evolution Sandbox（全开）"},
}

# ------------------------------------------------------------------ G 档（治理）----

G_ABLATIONS: dict[str, dict] = {
    "G0_full": {"review": True, "monitoring": True, "rollback": True,
                "desc": "全治理（归纳 review + 运行期监控 + 回滚）"},
    "G1_no_review": {"review": False, "monitoring": True, "rollback": True,
                     "desc": "无归纳 review"},
    "G2_no_monitoring": {"review": True, "monitoring": False, "rollback": False,
                         "desc": "无运行期监控与回滚"},
}

# 治理档 → driver 的 governance 参数名（driver 已实现 G0/G1/G2）
GOVERNANCE_TO_DRIVER = {"G0": "G0_full", "G1": "G1_no_review", "G2": "G2_no_monitoring"}


@dataclass
class AblationSpec:
    """一档消融的解析结果（可直接映射到 CLI/配置）。"""

    name: str
    kind: str                       # online | evolution | governance
    features: dict = field(default_factory=dict)
    desc: str = ""

    def enabled(self, feature: str) -> bool:
        return bool(self.features.get(feature, False))

    def cli_args(self) -> list[str]:
        """该档对应的 CLI 参数（供 driver/eval 直接拼接）。"""
        if self.kind == "online":
            args = ["--baseline", str(self.features["baseline"])]
            if self.features.get("skills") == "static_spec":
                args += ["--skill-spec", "<path>"]
            elif self.features.get("skills") == "wrong_injected":
                args += ["--inject-wrong-skill"]
            return args
        if self.kind == "evolution":
            return ["--ablation", self.name]
        return ["--governance", str(self.features["driver_name"])]

    def to_row(self) -> dict:
        return {"config": self.name, "kind": self.kind, **self.features, "desc": self.desc}


def resolve_ablation(name: str) -> AblationSpec:
    """解析消融档名（C0–C5 / E0–E5 / G0–G2 及别名）。"""
    key = str(name).strip()
    if key in C_ABLATIONS:
        return AblationSpec(key, "online", dict(C_ABLATIONS[key]), C_ABLATIONS[key]["desc"])
    ekey = key.upper() if key.upper() in E_ABLATIONS else key
    if ekey in E_ABLATIONS:
        f = dict(E_ABLATIONS[ekey])
        return AblationSpec(ekey, "evolution", f, f.pop("desc", ""))
    gkey = {"G0": "G0_full", "G1": "G1_no_review", "G2": "G2_no_monitoring"}.get(
        key.upper(), key)
    if gkey in G_ABLATIONS:
        f = dict(G_ABLATIONS[gkey])
        desc = f.pop("desc", "")
        f["driver_name"] = gkey
        return AblationSpec(gkey, "governance", f, desc)
    raise KeyError(f"未知消融档: {name}（可选 {list(C_ABLATIONS) + list(E_ABLATIONS) + list(G_ABLATIONS)}）")


def e_ablation_features(name: str) -> dict:
    """E 档特性开关（E0–E5，含 desc）；未知名抛错。"""
    key = str(name).upper()
    if key not in E_ABLATIONS:
        raise KeyError(f"未知 E 档: {name}（可选 {list(E_ABLATIONS)}）")
    return dict(E_ABLATIONS[key])


def format_ablation_markdown(kind: Optional[str] = None) -> str:
    """生成消融表 Markdown（论文表格骨架；结果列留空由实验填写）。"""
    out: list[str] = []
    if kind in (None, "online"):
        out.append("| 配置 | 说明 | MCA ↑ | MRA ↑ | 备注 |")
        out.append("|---|---|---|---|---|")
        for name, f in C_ABLATIONS.items():
            out.append(f"| {name} | {f['desc']} | | | |")
    if kind in (None, "evolution"):
        out.append("| 配置 | 隔离 | replay | 反例 | MR | paired A/B | MCA Δ | 样本效率 |")
        out.append("|---|---|---|---|---|---|---|---|")
        for name, f in E_ABLATIONS.items():
            mark = lambda k: "✅" if f.get(k) else "❌"  # noqa: E731
            out.append(f"| {name} | {mark('isolation')} | {mark('replay')} | "
                       f"{mark('counterexample')} | {mark('metamorphic')} | "
                       f"{mark('paired')} | | |")
    if kind in (None, "governance"):
        out.append("| 配置 | 归纳 review | 运行期监控 | 回滚 | MCA ↑ | MRA ↑ | 回归率 |")
        out.append("|---|---|---|---|---|---|---|")
        for name, f in G_ABLATIONS.items():
            mark = lambda k: "✅" if f.get(k) else "❌"  # noqa: E731
            out.append(f"| {name} | {mark('review')} | {mark('monitoring')} | "
                       f"{mark('rollback')} | | | |")
    return "\n".join(out)


def describe_ablations() -> dict:
    """所有消融档的结构化描述（落 JSON 供论文与审计）。"""
    return {
        "online": {k: v for k, v in C_ABLATIONS.items()},
        "evolution": {k: v for k, v in E_ABLATIONS.items()},
        "governance": {k: v for k, v in G_ABLATIONS.items()},
        "note": "消融只改机制开关；模型/split/seed/重建产物保持一致（§16.1）",
    }
