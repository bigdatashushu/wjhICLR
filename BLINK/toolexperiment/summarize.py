#!/usr/bin/env python3
"""Verify raw measurements and produce the final tool report and RGB evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def stats(values):
    a = np.asarray(values, dtype=float)
    assert np.isfinite(a).all()
    return {"n": len(a), "median": float(np.median(a)),
            "p95": float(np.quantile(a, .95)), "max": float(a.max()),
            "fraction_over_20pct": float(np.mean(a > .2))} if len(a) else {"n": 0}


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                    encoding="utf-8")


def figure(rows, source, destination, font_path):
    annotated = [r for r in rows if r["visual_ordinals"]]
    annotated.sort(key=lambda r: int(r["sample_id"].rsplit("_", 1)[1]))
    width, step = 1096, 384
    canvas = Image.new("RGB", (width, 110+step*len(annotated)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(font_path), 21)
    title = ImageFont.truetype(str(font_path), 28)
    draw.text((28, 16), "BLINK: RGB-first near/far annotations and cached depth", fill="#172b39", font=title)
    draw.text((28, 54), "N: nearer surface   F: farther surface   |   Depth color range is independent per view",
              fill="#425362", font=font)
    for i, row in enumerate(annotated):
        y = 102+i*step
        official_id = int(row["sample_id"].rsplit("_", 1)[1])+1
        draw.text((28, y), f"Official val {official_id}  /  both views passed", fill="#172b39", font=font)
        for view in (0, 1):
            image = Image.open(source[row["sample_id"]][view]).convert("RGB").resize((240, 320))
            annotation = next(a for a in row["visual_ordinals"] if a["view_index"] == view)
            result = annotation["measurements"][0]
            depth = np.array(row["preview_depth"][view], dtype=float)
            good = np.isfinite(depth)
            lo, hi = np.quantile(depth[good], [.02, .98])
            norm = np.clip((depth-lo)/(hi-lo), 0, 1)
            colored = cv2.cvtColor(cv2.applyColorMap(
                np.nan_to_num(norm*255).astype("uint8"), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
            depth_image = Image.fromarray(colored).resize((240, 320), Image.Resampling.NEAREST)
            x = 28+view*536
            for im, px in ((image, x), (depth_image, x+252)):
                canvas.paste(im, (px, y+30))
                for kind, label, color in (("near", "N", "#20efb0"), ("far", "F", "#ffac44")):
                    xx, yy = annotation[kind]
                    cx, cy = px+xx/2, y+30+yy/2
                    draw.ellipse((cx-6, cy-6, cx+6, cy+6), fill=color, outline="black", width=2)
                    draw.text((cx+9, cy-12), label, fill=color, stroke_width=1, stroke_fill="black", font=font)
            draw.text((x, y+352), f"View {view}: N={result['near_depth']:.3f}, F={result['far_depth']:.3f}",
                      fill="#243b4d", font=font)
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    results = root/"results"
    execution = json.loads((results/"execution.json").read_text())
    assert execution["complete"] and execution["returncode"] == 0
    assert sha(results/"raw.jsonl") == execution["raw_sha256"]
    for name, key in (("run_tools.py", "measurement_source_sha256"),
                      ("visual_annotations.json", "annotation_file_sha256"),
                      ("rgb_correspondences.json", "rgb_correspondences_file_sha256")):
        assert sha(root/name) == execution[key], name
    records = [json.loads(s) for s in (results/"raw.jsonl").read_text().splitlines()]
    assert records[0]["record_type"] == "manifest" and records[-1]["record_type"] == "complete"
    manifest, completion = records[0], records[-1]
    rows = [r["result"] for r in records if r["record_type"] == "sample"]
    assert len(rows) == completion["questions"] == manifest["questions"] == 133
    assert [r["sample_id"] for r in rows] == manifest["sample_ids"]
    assert len(set(manifest["sample_ids"])) == len(rows)
    for row in rows:
        assert digest({k: v for k, v in row.items() if k != "record_sha256"}) == row["record_sha256"]
    assert manifest["annotation_sha256"] == digest(json.loads((root/"visual_annotations.json").read_text()))
    assert manifest["rgb_correspondences_sha256"] == digest(json.loads((root/"rgb_correspondences.json").read_text()))
    checks, calls, failed, cross_status = Counter(), Counter(), Counter(), Counter()
    point_jitter, pair_jitter, ordinals, cross = [], [], [], []
    for row in rows:
        checks.update(row["checks"])
        calls.update(row["calls_by_tool"])
        failed.update(row["failed_calls_by_tool_including_expected_rejections"])
        ordinals.extend(row["visual_ordinals"])
        cross_status[row["cross_view"]["status"]] += 1
        if row["cross_view"]["status"] == "measured":
            cross.append(row["cross_view"])
        for sample in row["grid"]:
            for jitter in sample["jitter"]:
                if jitter["relative_change"] is not None:
                    point_jitter.append({**jitter, "sample_id": row["sample_id"],
                                         "view": sample["view"], "xy": sample["xy"],
                                         "original_depth": sample["depth"]})
        for pair in row["pairs"]:
            for jitter in pair["jitter"]:
                if jitter["relative_change"] is not None:
                    pair_jitter.append({**jitter, "sample_id": row["sample_id"],
                                        "view": pair["view"], "a": pair["a"], "b": pair["b"],
                                        "original_distance": pair["distance"]})
    summary = {
        "schema": "blink-real-tools-experiment-summary/1",
        "questions": len(rows), "roles": dict(Counter(r["source_split"] for r in rows)),
        "images": sum(len(r["view_geometry"]) for r in rows),
        "completion": completion, "checks": dict(checks), "calls_by_tool": dict(calls),
        "registry_calls": sum(calls.values()), "sandbox_tool_calls": len(rows)*4,
        "total_calls": sum(calls.values())+len(rows)*4,
        "empty_candidate_questions": sum(r["n_candidates"] == 0 for r in rows),
        "point_jitter": stats([v["relative_change"] for v in point_jitter]),
        "distance_jitter": stats([v["relative_change"] for v in pair_jitter]),
        "point_jitter_largest": sorted(point_jitter, key=lambda r:r["relative_change"], reverse=True)[:5],
        "distance_jitter_largest": sorted(pair_jitter, key=lambda r:r["relative_change"], reverse=True)[:5],
        "visual_ordinal_pairs": len(ordinals),
        "visual_ordinal_pairs_agree": sum(v["measurements"][0]["near_is_closer"] is True for v in ordinals),
        "visual_ordinal_checks_with_perturbations": sum(len(v["measurements"]) for v in ordinals),
        "visual_ordinal_checks_agree": sum(m["near_is_closer"] is True for v in ordinals for m in v["measurements"]),
        "cross_view_status": dict(cross_status),
        "cross_view_pair_relative_errors": stats([p["scale_normalized_error"] for v in cross for p in v["measured_pairs"]]),
        "cross_view_per_question_medians": stats([v["relative_error"]["median"] for v in cross]),
        "cross_view_scale_ratios": [v["scale_1_over_0"] for v in cross],
        "ordinal_annotations_are_sensor_gt": False,
        "raw_and_record_hashes_verified": True,
        "source_unchanged": completion["source_unchanged"],
        "model_requests": 0, "gpu_requests": 0, "benchmark_answer_access": False,
        "raw_sha256": execution["raw_sha256"],
    }
    write(results/"summary.json", summary)
    source = {}
    for split in ("train", "test"):
        for text in (args.dataset_root/"skill3d_splits"/f"blink_multiview_{split}.jsonl").read_text().splitlines():
            row = json.loads(text)
            source[row["id"]] = [args.dataset_root/p for p in row["image"]]
    figure(rows, source, results/"visual_ordinals.png", args.font)
    p, d, c = summary["point_jitter"], summary["distance_jitter"], summary["cross_view_pair_relative_errors"]
    started = datetime.fromtimestamp(execution["started_at_unix"], timezone.utc).astimezone().isoformat()
    text = f"""# BLINK 四工具实测结果

结论：三个几何工具能忠实读取和计算现有 MoGe2 缓存；局部相对深度得到有限的图像证据支持。
`view_candidates` 在全部 133 题上返回空列表，目前不能提供目标定位信息。
这些工具没有跨图位姿或运动估计，尚未验证它们能提高 BLINK 左右方向答题准确率。

## 实际执行

- 时间：{started}；Titan CPU 单进程，耗时 {completion['elapsed_seconds']:.1f} 秒。
- 数据：原 40 题训练、93 题测试，共 133 题、266 张原图；保留原划分。
- 调用：原 `ToolRegistry.call_tool` 共 {summary['registry_calls']:,} 次，原沙箱 {summary['sandbox_tool_calls']:,} 次，合计 {summary['total_calls']:,} 次。
- 源码：四数据集实验原冻结源码，前后哈希一致。没有修改工具、缓存或主实验。
- 每题经原 `load_prepared` 校验 NPZ 哈希、内容身份和公开输入；266 张缓存 RGB 与原图解码像素一致。
- Qwen 请求 0、GPU 推理 0、BLINK 答案读取 0。

## 四工具分别表现

| 工具 | 实测证据 | 使用建议 |
|---|---|---|
| `view_candidates` | 399 次正常查询与缓存一致，133/133 题候选均为空 | 接口正常，但本批没有检测信息；空列表不等于场景没有物体 |
| `view_depth_summary` | 266/266 张图的有效像素数和 10/50/90% 分位数与独立重算一致 | 可查看预测深度分布，不能用汇总分位数直接回答相机方向 |
| `point_depth` | 6,650/6,650 个网格点与独立 5×5 中位数计算一致 | 可辅助单图前后关系，避开边界并检查坐标扰动 |
| `pair_point_distance` | 3,192/3,192 对与独立点图计算一致；交换对称与同点为零均通过 | 返回单图相对预测距离；几何真值准确性未由本实验确定 |

全部 133 题均在原沙箱中实际调用四工具并成功 `YieldObservations`，没有提交 BLINK 答案。
1,463 次无效坐标、错误视图或额外跨视图参数均被拒绝。
本批几何有效 mask 全为有效，未遇到自然缺失的深度区域，因此这轮不声称测试过该类缺失。

## 数值稳定性与几何诊断

查询坐标上下左右移动 2 像素，按每次变化相对原查询值计算：

| 诊断 | 次数 | 中位变化 | P95 | 最大变化 |
|---|---:|---:|---:|---:|
| 单点深度 | {p['n']:,} | {100*p['median']:.3f}% | {100*p['p95']:.3f}% | {100*p['max']:.1f}% |
| 同图两点距离（只移动一个端点） | {d['n']:,} | {100*d['median']:.3f}% | {100*d['p95']:.3f}% | {100*d['max']:.1f}% |

绝大多数查询变化较小，但最大变化很大。扰动可能跨越物体边界；这说明单次像素查询不能保证稳定。
逐个大变化样本、坐标和实际值保留在 `summary.json`，未据此修改工具或筛掉原样本。

在查看任何深度结果前，从数字编号最小的六道训练题的两张原图标注 12 组明显前后关系：
**12/12 与深度排序一致**；原点加四个 2 像素扰动，共 **60/60 一致**。
这些是助手依据 RGB 的定性标注，范围较小、没有独立第二位标注者，也不是传感器深度真值。

![全部原图标注及预测深度](results/visual_ordinals.png)

为检查单图点图在视角变化下的相对距离，使用 RGB-only SIFT 双向比率匹配和基础矩阵 RANSAC。
133 对图像中 79 对达到至少 12 个内点；其余 54 对因匹配不足保留为未测。
在可测图对上，工具实际计算 6,098 组同名端点距离，给每图对拟合一个独立相对尺度后：

- 距离比偏差中位数 **{100*c['median']:.2f}%**，P95 **{100*c['p95']:.2f}%**，最大 **{100*c['max']:.1f}%**。
- 每题自身中位偏差的总体中位数 **{100*summary['cross_view_per_question_medians']['median']:.2f}%**。
- 偏差定义：`abs((d_view1 / d_view0) / median(d_view1 / d_view0) - 1)`。

这表明部分可匹配区域的相对形状一致，但尾部存在明显不一致。
特征匹配可能出错，纹理充分样本也有选择偏差；尺度使用同批点对拟合。
不能把以上数值写成距离真值误差，或据此认证跨视图定位。

## 对 BLINK 实验的含义

1. 保留 `point_depth` 和 `pair_point_distance` 作为带视图索引的辅助证据，要求同视图使用。
2. `view_depth_summary` 可提供可用性和范围信息。`view_candidates` 当前不增加有效证据。
3. 这轮证明实际工具执行和缓存数值一致，提供有限几何诊断；没有进行有/无工具的答题对比。
4. 要判断 BLINK 答题是否受益，需要单独冻结相同模型、thinking、输入、预算的配对对比。

## 文件与复现

- `run_tools.py`：服务端实测程序；`run_remote.py`：SSH 调用与本地证据保存。
- `visual_annotations.json`：先于深度查询冻结的原图标注。
- `prepare_rgb_checks.py`、`rgb_correspondences.json`：RGB 匹配方法和冻结坐标。
- `results/raw.jsonl`：逐题调用结果、原沙箱返回、缓存身份和诊断数据。
- `results/summary.json`：汇总及完整大变化案例。
- `results/execution.json`：程序/输入/结果哈希与完成状态。
- `4090_environment.json`、`4090_cache.json`：4090 环境与缓存核查。4090 原图和冻结源码存在，已发布缓存当前在 Titan。

原始记录及每题内容哈希已重新核验。大预计算缓存保留原服务器位置，不纳入 Git。
"""
    (root/"实验结果.md").write_text(text, encoding="utf-8")
    print(json.dumps({"questions": len(rows), "calls": summary["total_calls"],
                      "summary": str(results/"summary.json"), "report": str(root/"实验结果.md")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
