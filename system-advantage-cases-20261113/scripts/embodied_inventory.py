"""Inventory existing successful EmbodiedBench runs without executing episodes."""
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = pathlib.Path("/Users/bytedance/wangjh/具身最小提示词剩余104题并发合并156题-20261109-r02/results.json")
combined = json.loads(SOURCE.read_text())
inventory = []
for pair in combined["pairs"]:
    if pair["current_success"] != 1:
        continue
    directory = pathlib.Path(pair["directory"])
    result = json.loads((directory / "result.json").read_text())
    assert result["official"]["done"] and result["official"]["task_success"] == 1
    responses = []
    for path in sorted(directory.glob("response-*.json")):
        response = json.loads(path.read_text())
        text = response.get("text") or response.get("message", {}).get("content", "")
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            parsed = None
        responses.append({"file": path.name, "text": text, "parsed": parsed})
    inventory.append({
        **pair,
        "instruction": result["metrics"]["instruction"],
        "official": result["official"],
        "metrics": result["metrics"],
        "targets": result["perception"]["targets"],
        "responses": responses,
        "request_count": len(list(directory.glob("request-*.json"))),
        "rgb_count": len(list(directory.glob("observation-*-0.png"))),
        "result_hash_matches": hashlib.sha256((directory / "result.json").read_bytes()).hexdigest() == pair["audit"]["result_sha256"],
    })
(ROOT / "review/embodied-candidate-inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2))
lines = ["# EmbodiedBench 成功候选盘点", "", "原始合并结果：99/156 成功。下列全部为已有成功轨迹。", ""]
for env in ["alfred", "habitat", "manipulation", "navigation"]:
    lines += ["## " + env, ""]
    for c in inventory:
        if c["task"]["environment"] != env:
            continue
        lines.append(f'- **{c["identity"]}** · {c["instruction"]} · {c["audit"]["actions"]} 动作 / {c["audit"]["planner_requests"]} 规划 / {c["audit"].get("invalid_actions", 0)} 无效')
(ROOT / "review/embodied-candidates.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
