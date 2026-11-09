"""Build the acceptance report and a montage using actual simulator PNGs only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from target_detection.sources import canonical, file_digest, implementation_hashes
from PIL import Image, ImageDraw, ImageFont, ImageOps

OUT = ROOT / "target_detection"


def font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="target-detection-20261108-r03")
    args = parser.parse_args()
    directory = OUT / "runs" / args.run_id
    audit = json.loads((directory / "audit.json").read_text())
    after = json.loads((OUT / "verification/server-after.json").read_text())
    assert audit["status"] == "verified" and not after["owned_trial_workers"]
    assert after["old_evidence_unchanged"] and after["local_old_files_unchanged"]
    rows = audit["episodes"]
    canvas = Image.new("RGB", (1480, 120 + len(rows) * 445 + 44), "#E2E8F0")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 25), "Question-directed detection and crop review", fill="#0F172A", font=font(30))
    draw.text((28, 70), "Actual simulator frames | raw RGB -> grouped proposals -> unannotated crops",
              fill="#475569", font=font(20))
    details = []
    for i, row in enumerate(rows):
        task = row["task"]
        folder = directory / f'{task["environment"]}-{task["subset"]}-{task["index"]:03d}'
        events = [json.loads(line) for line in (folder / "trajectory.jsonl").read_text().splitlines()]
        result = json.loads((folder / "result.json").read_text())
        tools = [e for e in events if e["type"] == "perception_tool_result"]
        reviews = [e for e in events if e["type"] == "perception_action_review"]
        matched = next((e for e in reviews if e["selected_targets"]), None)
        serial = matched["observation_serial"] if matched else tools[0]["result"]["observation_serial"]
        page = next(e for e in tools if e["result"]["observation_serial"] == serial and
                    e["result"]["tool"] == "detect_targets")
        crop_events = [e for e in tools if e["result"]["observation_serial"] == serial and
                       e["result"]["tool"] == "inspect_candidates"]
        crops = [(c["group_id"], img) for e in crop_events
                 for c, img in zip(e["result"]["crops"], e["images"])]
        seen, unique = set(), []
        selected = {r["group_id"] for r in matched["selected_targets"]} if matched else set()
        for gid, image in sorted(crops, key=lambda p: p[0] not in selected):
            if gid not in seen:
                unique.append((gid, image))
                seen.add(gid)
        y = 120 + i * 445
        draw.rectangle((16, y, 1464, y + 427), fill="white")
        title = (f'{task["environment"].upper()} base {task["index"]} | '
                 f'{page["result"]["proposal_count"]} proposals / {page["result"]["group_count"]} groups | '
                 f'official success={int(row["official_success"])}')
        draw.text((30, y + 12), title, font=font(22), fill="#0F172A")
        targets = ", ".join(t["category"] for t in row["targets"])
        draw.text((30, y + 44), "Targets from instruction: " + targets, font=font(17), fill="#475569")
        paths = [(folder / f"observation-{serial:03d}-raw.png", (30, y + 88, 354, 324), "Raw RGB"),
                 (folder / page["images"][0]["file"], (411, y + 88, 574, 324), "Numbered groups")]
        for p, (x, top, w, h), caption in paths:
            draw.text((x, y + 66), caption, font=font(16), fill="#475569")
            image = ImageOps.contain(Image.open(p).convert("RGB"), (w, h))
            canvas.paste(image, (x, top))
        for j, (gid, meta) in enumerate(unique[:2]):
            x, top = 1009 + j * 224, y + 88
            draw.text((x, y + 66), gid + " raw crop", font=font(16), fill="#475569")
            image = ImageOps.contain(Image.open(folder / meta["file"]).convert("RGB"), (204, 290))
            canvas.paste(image, (x + (204 - image.width) // 2, top + (290 - image.height) // 2))
            if gid in selected:
                draw.text((x, y + 384), "model verdict: match", font=font(14), fill="#047857")
        details.append({"task": task, "displayed_observation_serial": serial,
                        "first_displayed_proposal_count": page["result"]["proposal_count"],
                        "first_displayed_group_count": page["result"]["group_count"],
                        "raw_image": str((folder / f"observation-{serial:03d}-raw.png").relative_to(OUT)),
                        "overlay_image": str((folder / page["images"][0]["file"]).relative_to(OUT)),
                        "displayed_crop_groups": [g for g, _ in unique[:2]]})
    draw.text((28, canvas.height - 32), "Proposal groups and model crop judgments are not ground-truth object identities.",
              font=font(16), fill="#475569")
    preview = OUT / "detection-review-preview.png"
    canvas.save(preview)
    summary = json.loads((directory / "summary.json").read_text())
    successes = sum(r["official_success"] == 1 and r["status"] == "official_done" for r in rows)
    lines = [
        "# 目标检测与局部裁剪复核：真实验收",
        "",
        f"最终入口为 `target_detection`，冻结配置 `experiment-20261108-r03.json`，"
        f"运行 `{args.run_id}`。查询与验收完成时间：{after['checked_at']}，节点 "
        "`cvailab@10.10.0.198`。",
        "",
        f"四题中官方判定成功 {successes}/{len(rows)}。本次实际记录 "
        f"{audit['model_requests']} 次 Qwen 请求、{audit['tool_calls']} 次工具调用、"
        f"{audit['detector_requests']} 次逐类别 GroundingDINO 推理，"
        f"{audit['actions']} 个动作与 {audit['acks']} 个 ACK。",
        "",
        "| 环境与题目 | 结束状态 | 官方成功 | Qwen 请求 | 检测推理 | 工具调用 | 裁剪 | 动作/ACK |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        t = r["task"]
        lines.append(f'| {t["environment"]} {t["subset"]}{t["index"]} | {r["status"]} | '
                     f'{int(r["official_success"])} | {r["model_requests"]} | {r["detector_requests"]} | '
                     f'{r["tool_calls"]} | {r["crops"]} | {r["actions"]}/{r["acks"]} |')
    lines.extend([
        "", "## 实现行为", "",
        "同一个 Qwen 从公开指令提取类别，数量、属性和关系单独记录；原文片段必须匹配公开指令，"
        "根目标整题冻结。模型按需逐类别请求检测，低阈值 0.15 保留全部原始提议和空标签。"
        "IoU ≥ 0.8 完整链接分组保留每个成员，避免重叠链误并。编号放在图外侧栏，"
        "每页 12 组，后续页均可取。",
        "",
        "模型请求原始 RGB 局部裁剪后，独立调用 `review_candidates` 给出 match/not_match/uncertain "
        "及理由，再输出官方完整动作计划。复核记录绑定当前观测；换帧后旧 G 编号失效。"
        "全部模型轮次计入原预算，工具阶段没有仿真 step/observe。",
        "",
        "Manipulation 的原官方匿名编号图和动作坐标保留，检测使用同帧未标注 RGB。"
        "匿名坐标投影只提供可能对应，G 编号与官方 object 编号不等价。",
        "", "## 真实截图", "",
        "[四环境原图、检测编号及局部裁剪对照](detection-review-preview.png)",
        "",
    ])
    for r, d in zip(rows, details):
        lines.append(f'- {r["task"]["environment"]}：类别 '
                     + "、".join(t["category"] for t in r["targets"])
                     + f'；所示同一帧为 {d["first_displayed_proposal_count"]} 个提议 / '
                     f'{d["first_displayed_group_count"]} 组。'
                     + f' [原图]({d["raw_image"]}) · [检测图]({d["overlay_image"]})')
    lines.extend([
        "", "## 审计", "",
        "七项离线测试通过。真实验收中全部模型消息和工具结果已离线重放，"
        "目标冻结、候选编号/失效和当前裁剪判断均复核；检测请求绑定原始 RGB 哈希，"
        "裁剪与原 RGB 对应区域逐像素一致。原图、深度 PNG 和原始数值深度文件均校验。"
        "动作顺序与完整计划前缀一致，已发送动作与 ACK 数量一致。",
        "",
        f"旧本地 {json.loads((OUT / 'verification/frozen.json').read_text())['preserved_local_files']} "
        "个文件与前两版结果保持字节一致；服务器旧证据和最终源码身份一致。"
        "当前已无本轮 worker 或工作目录内原生仿真进程，检测服务仍为固定 GPU1/SwinB 身份。",
        "",
        f"[完整审计 JSON](runs/{args.run_id}/audit.json) · "
        f"[运行摘要](runs/{args.run_id}/summary.json) · "
        "[源码与运行说明](README.md) · [最终服务器检查](verification/server-after.json)",
        "", "## 接入版本记录", "",
    ])
    for package, run in (("target_detection_trial", "target-detection-20261108-r01"),
                         ("target_detection_v2", "target-detection-20261108-r02")):
        p = ROOT / package / "runs" / run
        previous = json.loads((p / "summary.json").read_text())
        lines.append(f"- `{package}`：{previous['http_requests']} 次模型请求，"
                     f"{previous['status']}；原源码、请求、响应和终态均保留。")
    lines.extend([
        "",
        "首版的 Navigation/ALFRED 未输出目标，记录为 planner_error；Habitat 已检测和裁剪，"
        "第二动作等待 ACK 时连接关闭，结果未知，没有重发，suite 当时停止。"
        "第二版将目标和工具阶段放入系统指令，ALFRED 完成了 11 个动作并成功；"
        "Navigation 动作响应遗漏裁剪判断字段。最终版采用独立复核工具阶段。",
        "",
        "这四题验证接入和动作链，不能据此判断完整数据集上的成功率收益。"
        "检测框、分组和 Qwen 判断也不能视为已验证的物体实例或几何表面。",
        "",
    ])
    (OUT / "acceptance-report.md").write_text("\n".join(lines), encoding="utf-8")
    receipt = {"status": "verified", "run_id": args.run_id, "created_at": datetime.now(timezone.utc).isoformat(),
               "preview_sha256": file_digest(preview), "report_sha256": file_digest(OUT / "acceptance-report.md"),
               "audit_sha256": file_digest(directory / "audit.json"), "screenshots": details,
               "implementation_hashes": implementation_hashes(), "selected": len(rows), "official_successes": successes}
    (OUT / "verification/deliverables.json").write_bytes(canonical(receipt) + b"\n")
    print(canonical({"status": "verified", "official_successes": successes, "selected": len(rows),
                     "report": str(OUT / "acceptance-report.md"), "preview": str(preview)}).decode())


if __name__ == "__main__":
    main()
