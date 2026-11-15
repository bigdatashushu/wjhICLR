"""Write a compact source index and the final human-readable selection ledger."""
import hashlib
import json
import pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = pathlib.Path("/Users/bytedance/wangjh")
cases = json.loads((ROOT / "build/reviewed-cases.json").read_text())
source_index = []
for c in cases:
    ds, qid = c["dataset"], c["id"]
    if ds == "VSIbench":
        paths = [BASE / c["evidence"]["outcome_source"]]
    elif ds == "MindCube":
        paths = [BASE / "outputs/harness3d_four_datasets_20261031" / c["run"] / "experiment_trace.jsonl"]
    elif ds == "CV3D":
        paths = list((BASE / "outputs/harness3d_four_datasets_20261031" / c["run"]).glob(f"*/{qid}/case.json"))
    elif ds == "BLINK":
        paths = [BASE / "wjhICLR/GPTexp/BLINKGPT/runs" / c["run"] / "final/manifest.json"]
    elif ds == "MMSI":
        paths = [BASE / c["run"] / "manifest.json", BASE / "MMSI六类四十题原图直答-20261109-r01/manifest.json"]
    else:
        paths = [BASE / "outputs/harness3d_four_datasets_20261031" / c["run"] / "scores.json"]
    sources = []
    for p in paths:
        assert p.exists(), p
        entry = {"path": str(p), "bytes": p.stat().st_size}
        if p.stat().st_size < 100_000_000:
            entry["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        else:
            entry["identity_note"] = "大型逐题 JSONL 已按 sample_id 流式提取；本次未重复对整文件计算哈希。"
        sources.append(entry)
    source_index.append({"dataset": ds, "id": qid, "sources": sources,
                         "normalized_record": "build/reviewed-cases.json",
                         "asset_refs": [g["asset"] for g in c["gallery"]],
                         "request_mode": {a: [r["request_mode"] for r in v["rounds"]] for a, v in c["arms"].items()}})
(ROOT / "source/selected-case-source-index.json").write_text(json.dumps(source_index, ensure_ascii=False, indent=2))

counts = Counter(c["audit"]["mechanism"] for c in cases)
text = [
    "# 30 题入选复核",
    "",
    "六个数据集各五题。所有入选题均有系统正确与直答实际回答错误的原始评分记录。",
    "入选依据是具体的测量采用、位姿提示匹配、参照物绑定或可追踪的方向推导，"
    "而不是仅凭分数差异推断机制有效。原始记录与本次中文解读在 HTML 中分别呈现。",
    "",
    "证据类型：" + "；".join(f"{k} {v} 题" for k, v in counts.items()) + "。",
    "",
    "图片：226 个独立资源。VSIbench 五题共 160 张冻结帧，逐张 RGB 像素哈希与原 image_ledger 一致；"
    "MMSI 使用 10 次图片输入，含一张跨题复用原图，独立图片为 9 张。",
    "",
    "保留范围：BLINK、Omni3D、MindCube 的请求逐轮原始保存；CV3D、MMSI、VSIbench 的请求正文按已有记录恢复并显式标注。"
    "CV3D、Omni3D、MindCube 的返回 thinking 保留；BLINK 的 JSON 显式 reasoning 保留，隐藏 thinking 未返回；"
    "MMSI 和 VSIbench 的隐藏 thinking 未保存在回收记录。",
    "",
    "Omni3D 标签正文未归档，展示评分为 1.0 的系统匹配作答和两边原评分。"
    "VSIbench effort 为 max / low，官方 low 兼容映射 high；这里展示案例行为，不将其称为同预算独立工具消融。",
]
for ds in ["VSIbench", "CV3D", "MindCube", "MMSI", "BLINK", "Omni3D"]:
    text += ["", f"## {ds}", ""]
    for c in [c for c in cases if c["dataset"] == ds]:
        a = c["audit"]
        text += [
            f"### {c['id']} · {a['title']}", "",
            f"系统：{c['arms']['system']['answer']}；直答：{c['arms']['direct']['answer']}。类型：{a['mechanism']}。",
            "", a["why"], "", "直答差异：" + a["direct_error"], "",
            "复核：" + "；".join(a["support"]) + "。", "",
            "记录范围：" + a["limits"], "",
        ]
text += ["## 排除与替换", "",
         "MindCube 的石拱门/垃圾桶题：垃圾桶采样为 unreliable，无法与可靠拱墙单点构成数值比较，"
         "已替换为 rotation_group005_q1_4。",
         "",
         "早期 BLINK 多视图获胜题：仅有深度可用性和裸答案，改用有显式 reasoning 的 SR Skill 记录。",
         "",
         "Omni3D 高度比题 0：缺少实际稳定尺寸测量，未选。",
         "",
         "本次未发送新的模型请求；全部案例来自历史实验。",
         "",
         "来源与哈希：source/selected-case-source-index.json；逐题机器核对：review/selected-case-audit.json；"
         "离线验证：checks/offline-browser-validation.json。"]
(ROOT / "review/selection-audit.md").write_text("\n".join(text))
print("Archived source identities and thirty-case selection ledger.")
