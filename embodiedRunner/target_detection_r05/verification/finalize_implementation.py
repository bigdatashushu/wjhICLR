"""Record current source/deployment identity separately from observed R04 outcomes."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from target_detection_r05.lifecycle import preserved_local
from target_detection_r05.sources import canonical, file_digest, implementation_hashes

OUT = ROOT / "target_detection_r05"
OBSERVED = ROOT / "target_detection_r04"
RUN = "target-detection-20261108-r04"


def load(path):
    return json.loads(path.read_text())


def main():
    audit = load(OBSERVED / "runs" / RUN / "audit.json")
    after = load(OUT / "verification/server-after.json")
    frozen = load(OUT / "verification/frozen.json")
    replay = load(OUT / "verification/navigation-review-replay.json")
    assert audit["status"] == "verified" and after["status"] == "ok"
    assert not after["owned_trial_workers"]
    assert after["old_evidence_unchanged"] and after["local_old_files_unchanged"]
    assert frozen["implementation_hashes"] == implementation_hashes() == replay["implementation_hashes"]
    assert preserved_local() == load(OUT / "verification/preserved-local.json")
    assert replay["live_model_requests"] == replay["live_detector_requests"] == replay["simulator_launches"] == 0
    for path in OUT.glob("*.py"):
        ast.parse(path.read_text(), feature_version=(3, 9))
    old_config = load(OBSERVED / "experiment-20261108-r04.json")
    new_config = load(OUT / "experiment-20261108-r05.json")
    equivalent = json.loads(json.dumps(new_config).replace("r05", "r04"))
    assert old_config == equivalent, "scientific_or_budget_configuration_changed"
    scanned = []
    possible_keys = 0
    for folder in (OUT, OBSERVED):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".md", ".json", ".jsonl"):
                scanned.append(path)
                possible_keys += len(re.findall(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_.+/-]{16,}", path.read_bytes()))
    assert possible_keys == 0, "possible_raw_key_in_deliverables"
    successes = sum(r["status"] == "official_done" and r["official_success"] == 1 for r in audit["episodes"])
    lines = [
        "# 目标检测与裁剪复核：实现交付",
        "",
        "当前源码入口为 `target_detection_r05`。问题提目标、逐类别 GroundingDINO、"
        "完整链接重叠分组、图外 G 编号、原 RGB 裁剪复核和官方完整动作执行已接入。"
        "修正版已部署并完成四环境零模型请求预检。",
        "",
        "九项离线检查通过，覆盖公开来源与目标冻结、重叠链、类别缓存、编号失效、"
        "裁剪像素、工具名、阶段推进、动作形式、工具阶段零仿真步和完整计划 ACK。"
        "真实 Navigation 的七次响应已离线重放，修正后的复核处理无需新模型请求。",
        "",
        "## 真实运行结果",
        "",
        f"真实试跑使用冻结的 R04，四环境各一题，官方成功 {successes}/{audit['selected']}。"
        f"合计 {audit['model_requests']} 次 Qwen 请求、{audit['tool_calls']} 次工具调用、"
        f"{audit['detector_requests']} 次逐类别检测，{audit['actions']} 个动作与 {audit['acks']} 个 ACK。",
        "",
        "| 环境 | 结束状态 | 官方成功 | Qwen 请求 | 检测请求 | 裁剪 | 动作/ACK |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in audit["episodes"]:
        lines.append(f'| {row["task"]["environment"]} | {row["status"]} | {int(row["official_success"])} | '
                     f'{row["model_requests"]} | {row["detector_requests"]} | {row["crops"]} | '
                     f'{row["actions"]}/{row["acks"]} |')
    lines.extend([
        "",
        "Navigation 在第二次观测后的纯复核响应省略 tool_request 外层，R04 提前结束。"
        "R05 按明确复核处理，保留原视觉判断并执行相同校验。裁剪工具名被文件名覆盖的"
        "问题也已修正。R05 尚未进行新的在线试跑，上述真实结果均属于 R04。",
        "",
        "Habitat 和 Manipulation 分别达到固定的 40 和 15 次模型请求预算后结束；"
        "检测、裁剪和复核轮次消耗同一预算。"
        "这是本次观察到的成本限制，不能把更多工具调用直接当作表现改进。",
        "",
        "## 证据与入口",
        "",
        "- [源码与运行说明](README.md)",
        "- [实际截图与逐环境说明](../target_detection_r04/acceptance-report.md)",
        "- [RGB、编号与裁剪对照 PNG](../target_detection_r04/detection-review-preview.png)",
        f"- [真实 R04 完整离线审计](../target_detection_r04/runs/{RUN}/audit.json)",
        "- [真实失败响应的修正版回放](verification/navigation-review-replay.json)",
        "- [修正版四环境运行预检](verification/preflight.json)",
        "- [修正版源码与部署身份](verification/frozen.json)",
        "- [最终服务器与原文件保留检查](verification/server-after.json)",
        "",
        "当前原实验、前四版源码和运行证据保持字节一致，已无本轮仿真 worker。"
        "检测服务仍为固定 GPU1/SwinB 身份。模型 key 仅由原授权入口在内存加载，"
        "交付文件扫描未发现原始 key。",
        "",
        "四题用于工程接入检查，不代表完整数据集上的成功率收益。检测框、分组"
        "和模型裁剪判断不等于已核实的实例或表面几何。",
        "",
    ])
    report = OUT / "implementation-report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    receipt = {
        "status": "verified",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "implementation_package": "target_detection_r05",
        "implementation_hashes": implementation_hashes(),
        "config_sha256": file_digest(OUT / "experiment-20261108-r05.json"),
        "observed_run_id": RUN,
        "observed_run_package": "target_detection_r04",
        "observed_successes": successes,
        "observed_selected": audit["selected"],
        "online_r05_retest_completed": False,
        "preflight_model_requests": 0,
        "replay_model_requests": 0,
        "source_and_previous_evidence_unchanged": True,
        "credential_scan": {"files": len(scanned), "possible_raw_keys": possible_keys},
        "report_sha256": file_digest(report),
        "observed_audit_sha256": file_digest(OBSERVED / "runs" / RUN / "audit.json"),
        "preview_sha256": file_digest(OBSERVED / "detection-review-preview.png"),
    }
    (OUT / "verification/deliverables.json").write_bytes(canonical(receipt) + b"\n")
    print(canonical({k: v for k, v in receipt.items() if k != "implementation_hashes"}).decode())


if __name__ == "__main__":
    main()
