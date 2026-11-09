"""Keep the same three boxes, change label placement, and inspect one Qwen reply."""
import base64
import copy
import json
import os
from pathlib import Path
import re
import sys

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT))
from run_probe import canonical, font, save, sha, call_service
from rgbd_three_envs.client import ModelClient


def main():
    source = json.loads((OUT / "qwen-density-results.json").read_text())
    row = next(r for r in source["density_pairs"]
               if r["environment"] == "manipulation" and r["condition"] == "filtered")
    image = Image.open(OUT / "inputs/manipulation-rgb.png").convert("RGB")
    draw = ImageDraw.Draw(image)
    locations = {3: (316, 291), 4: (281, 359), 5: (42, 391)}
    for item in row["candidate_legend"]:
        draw.rectangle(item["bbox"], outline="#E02875", width=2)
        x, y = locations[item["id"]]
        label = str(item["id"])
        width = draw.textbbox((0, 0), label, font=font(14))[2] + 6
        if item["id"] == 3:
            draw.line((x + width / 2, y + 19, 273, 318), fill="#E02875", width=2)
        draw.rectangle((x, y, x + width, y + 19), fill="white")
        draw.text((x + 3, y), label, font=font(14), fill="#A00044")
    corrected = OUT / "images/manipulation-clear-ids.png"
    image.save(corrected)
    canvas = Image.new("RGB", (1000, 550), "white")
    canvas.paste(Image.open(OUT / "images/manipulation-target_filtered.png"), (0, 0))
    canvas.paste(image, (500, 0))
    cd = ImageDraw.Draw(canvas)
    cd.text((10, 512), "Same 3 boxes: original ID positions", font=font(18), fill="black")
    cd.text((510, 512), "Same 3 boxes: separated ID positions", font=font(18), fill="black")
    canvas.save(OUT / "images/id-layout-comparison.png")
    payload = json.loads((OUT / "qwen-request-03.json").read_text())
    previous = copy.deepcopy(payload)
    images = [p for p in payload["messages"][0]["content"] if p["type"] == "image_url"]
    assert len(images) == 3
    images[2]["image_url"]["url"] = "data:image/png;base64," + base64.b64encode(corrected.read_bytes()).decode()
    assert payload["messages"][0]["content"][0] == previous["messages"][0]["content"][0]
    assert images[:2] == [p for p in previous["messages"][0]["content"] if p["type"] == "image_url"][:2]
    assert not (OUT / "layout-frozen.json").exists(), "layout_probe_already_started"
    save(OUT / "layout-frozen.json", {
        "maximum_qwen_requests": 1, "same_three_candidate_boxes_and_legend": True,
        "only_change": "image-3 numeric label placement and one leader line",
        "original_request_sha256": sha(canonical(previous)),
        "new_request_sha256": sha(canonical(payload)), "image_sha256": sha(corrected.read_bytes()),
        "automatic_retry": False, "environment_actions": 0,
    })
    save(OUT / "qwen-request-layout.json", payload)
    config = source["model"]
    key_file = ROOT.parent / "qwenkey.md"
    assert not key_file.stat().st_mode & 0o077
    keys = set(re.findall(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_.+/-]{16,}", key_file.read_text()))
    assert len(keys) == 1
    os.environ["DASHSCOPE_API_KEY"] = keys.pop()
    os.environ["DASHSCOPE_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    try:
        response = ModelClient(config).complete(payload)
        assert response["model"] == config["name"]
        save(OUT / "qwen-response-layout.json", response)
        value = json.loads(response["text"])
        for role in ("object", "destination"):
            assert all(type(x) is int and x in locations for x in value[role]["candidate_ids"])
    finally:
        os.environ.pop("DASHSCOPE_API_KEY", None)
        os.environ.pop("DASHSCOPE_BASE_URL", None)
    save(OUT / "layout-result.json", {"status": "verified", "qwen_requests": 1,
                                    "same_boxes_and_prompt": True, "result": value})
    identity = call_service({"mode": "status"})
    original = json.loads((OUT / "detections.json").read_text())
    assert identity["model_sha256"] == original["model_sha256"]
    save(OUT / "detector-final-identity.json", identity)
    frozen = json.loads((ROOT / "rgbd_twenty/frozen.json").read_text())
    preserved_packages = {}
    import importlib
    for package, pin in frozen["packages"].items():
        hashes = importlib.import_module(package + ".sources").implementation_hashes()
        assert hashes == pin["implementation_hashes"]
        preserved_packages[package] = True
    for case in original["inputs"]:
        kind = case["environment"]
        assert sha((OUT / "inputs" / f"{kind}-rgb.png").read_bytes()) == case["rgb_sha256"]
    summary = {
        "status": "verified", "detector_inference_requests": 14, "qwen_static_requests": 5,
        "environment_actions": 0, "automatic_retry": False,
        "counts": {k: {n: v[n]["count"] for n in ("broad_low", "target_low", "target_high")}
                   for k, v in original["results"].items()},
        "layout_result": value, "frozen_rgbd_packages_unchanged": preserved_packages,
        "manual_visual_check": {
            "manipulation_target_low": {"3": "teal star", "4": "green container", "5": "blue container",
                                        "8": "overlapping green-container region"},
            "star_only": {"true_star_score": 0.271, "robot_arm_star_false_positive_score": 0.437},
            "habitat": "Both density conditions rejected the toy-airplane label as a car/truck appearance.",
        },
        "scope": "Four screenshots; two static density pairs and one layout probe. Not task success evaluation.",
    }
    save(OUT / "summary.json", summary)
    lines = [
        "# GroundingDINO 环境截图与标框实测\n",
        "完成 14 次 GroundingDINO-SwinB 推理和 5 次 Qwen3.8-Flash 静态识别。"
        "四张真实 500×500 截图；其中 Manipulation 单独采集无标注 RGB/深度，采集期间动作数为 0。"
        "检测使用现有物理 GPU 1 副本；原 RGB-D 源码保持原哈希；没有自动重试。\n",
        "| 环境 | 宽泛类别 0.15 | 问题类别 0.15 | 问题类别 0.35 |",
        "|---|---:|---:|---:|",
    ]
    for kind, item in summary["counts"].items():
        lines.append(f"| {kind} | {item['broad_low']} | {item['target_low']} | {item['target_high']} |")
    lines += [
        "\n前 12 次是四截图 × 三配置；后 2 次是在同一 Manipulation 原图上分别查询 `star.` 和 `container.`。\n",
        "框数本身不足以预测模型判断。真实风险包括重复区域、错误类别、空标签以及编号和物体的视觉对应关系。\n",
        "## 实际观察\n",
        "Navigation：问题只查 bread 时，从宽泛扫描的 18 框降到 1 框，且保留了面包区域。\n",
        "ALFRED：问题查纸卷和马桶时，从 15 框降到 3 框，提高阈值后为 2 框。"
        "这一帧可见一卷纸，检测框数量不代表任务要求的两卷都已处理。\n",
        "Habitat：6 框的低阈值结果中，机械臂被标为 toy airplane；0.35 去掉此框，剩 2 框。"
        "但两次 Qwen 都认为黄色玩具外形是车而不是飞机，并拒认 toy airplane。"
        "减框并没有解决任务名称与视觉外形之间的类别歧义。\n",
        "Manipulation：复合查询 `star shaped block. container.` 返回 9 框，"
        "有 6 对 IoU≥0.75 的高重叠框；0.35 后剩 3 框，却把两个容器也标成 shaped block。"
        "单独查询 container 能给两个容器正确标签；单独查询 star 时，"
        "真正星形块分数约 0.271，而机械臂误检为 star 的分数约 0.437。"
        "因此不能一律提高阈值，也不能把最高置信度当作正确目标。\n",
        "## Qwen 静态对照\n",
        "同一原始 RGB 和深度始终保留，第三张图提供候选框；模型参数固定 "
        "temperature=0、seed=137、max_tokens=4096、enable_thinking=false。"
        "密集和过滤条件使用同一问题、相同候选来源，过滤是分数≥0.35和相同标签 IoU≥0.8 的 NMS。\n",
        "Manipulation：9 框时，Qwen 说是青绿色星形块，却选择了 IDs 3 和 8；"
        "8 实际覆盖绿色容器。3 框时，它选择 ID 4，而 4 实际是绿色容器；"
        "可见星形块数量还从 1 报成 2。两种条件下蓝色容器均选 ID 5。\n",
        "图像检查发现，原图编号 4 画在绿色容器框的上方，恰好落到星形块附近。"
        "随后保留同样 3 个框、相同候选表和原始提示，只移开编号并增加一条引线，"
        "额外调用 Qwen 1 次。以下是这次真实结果：\n",
        "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```\n",
        "这些是每条件单次静态识别结果，不能将变化全部归因于框数量或编号位置，"
        "也不代表环境任务成功率。\n",
        "## 接入建议\n",
        "按问题限制类别，但保留足够候选；按查询保留原始标签与置信度，明确未验证。"
        "高重叠区域可以分组展示，但不要把相邻物体或部件关系一律删除。"
        "避免编号落在其他目标上，必要时用引线和局部裁剪核对框与实例。"
        "采用类别/任务相关的阈值和语义复核，避免全局最高分、固定 Top-K 或一律 0.35。\n",
        f"[四环境原图与检测图](computer://{OUT / 'groundingdino-comparison.png'}) · "
        f"[同样三框的编号布局对比](computer://{OUT / 'images/id-layout-comparison.png'})\n",
        f"[12 次主检测完整记录](computer://{OUT / 'detections.json'}) · "
        f"[2 次单类别补测](computer://{OUT / 'single-category-probes.json'}) · "
        f"[4 次密度对照](computer://{OUT / 'qwen-density-results.json'}) · "
        f"[1 次编号布局对照](computer://{OUT / 'layout-result.json'})\n",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "verified", "detector_requests": 14, "qwen_requests": 5,
                      "layout_result": value}, ensure_ascii=False))


if __name__ == "__main__":
    main()
