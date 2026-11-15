"""Build independently shareable HTML readers with exact embedded evidence."""
import base64
import hashlib
import json
import pathlib
import re
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
AUTHOR = ROOT / "build/case-reader"
cases = json.loads((ROOT / "build/cases.json").read_text())
assets = json.loads((ROOT / "build/assets.json").read_text())
annotations = json.loads((ROOT / "review/annotations.json").read_text())
anchors = json.loads((ROOT / "review/vsi-anchor-records.json").read_text())
ORDER = ["VSIbench", "CV3D", "MindCube", "MMSI", "BLINK", "Omni3D"]
EB_ORDER = ["EB-Manipulation", "EB-Navigation", "EB-ALFRED", "EB-Habitat"]

def images(value):
    found = []
    def walk(v):
        if isinstance(v, dict):
            if "asset" in v:
                found.append(v["asset"])
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(value)
    return list(dict.fromkeys(found))

def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")

def normalize_ws(text):
    return re.sub(r"\s+", " ", text).strip()

checks = []
for c in cases:
    c["audit"] = annotations[c["id"]]
    c["reference_kind"] = "scored_prediction" if c["dataset"] == "Omni3D" else "reference"
    if c["reference_kind"] == "scored_prediction":
        c["audit"]["limits"] += " 原评分系统=1.0、直答=0.0；原标签正文未归档，页面显示评测匹配作答。"
    sysimgs = set(images(c["arms"]["system"]["rounds"][0]["messages"]))
    dirimgs = set(images(c["arms"]["direct"]["rounds"][0]["messages"]))
    if c["dataset"] == "VSIbench":
        assert c["evidence"]["same_frozen_rgb_identity"]
        c["audit"]["anchor_rows"] = anchors[c["id"]]
        used = c["evidence"]["derivation_replay"]["input_result_ids"]
        c["audit"]["deciding_results"] = [
            {k: t.get(k) for k in ("result_id", "tool", "args", "payload", "status")}
            for t in c["evidence"]["tool_results"] if t["result_id"] in used]
    else:
        assert dirimgs <= sysimgs, (c["dataset"], c["id"], "original RGB mismatch")
    assert str(c["arms"]["system"]["answer"]).lower() == str(c["reference"]).lower()
    assert str(c["arms"]["direct"]["answer"]).lower() != str(c["reference"]).lower()
    for arm in ("system", "direct"):
        text = "\n".join(r["response"] + "\n" + r["reasoning"] for r in c["arms"][arm]["rounds"])
        quote = c["audit"].get(arm + "_quote", "")
        c["audit"][arm + "_quote_verbatim"] = bool(quote and normalize_ws(quote) in normalize_ws(text))
    checks.append({"dataset": c["dataset"], "id": c["id"], "system_answer": c["arms"]["system"]["answer"],
                   "direct_answer": c["arms"]["direct"]["answer"], "reference": c["reference"],
                   "original_images_shared": True, "reference_kind": c["reference_kind"],
                   "request_modes": {k: list({r["request_mode"] for r in a["rounds"]})
                                     for k, a in c["arms"].items()},
                   "thinking_saved": {k: any(r["reasoning"] for r in a["rounds"])
                                      for k, a in c["arms"].items()}})

(ROOT / "review/selected-case-audit.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2))
(ROOT / "review/excluded-cases.json").write_text(json.dumps([
    {"dataset": "MindCube", "id": "around_0b328f5b774157256ac3851a51b544039f5a3dfdff2f2c901beee768e6e41fcf_q2_1",
     "reason": "垃圾桶的 point_depth 被标为 edge_sampling / unreliable；只有拱墙的单点可靠值，不足以建立深度排序。最终正确视觉答案不作为可靠测量优势。",
     "replacement": "rotation_group005_q1_4"},
    {"dataset": "BLINK", "source": "early multiview winning cases",
     "reason": "只有 view_depth_summary 的可用性和裸答案，缺乏可追踪的决定性关系证据，改选保留显式 JSON reasoning 的 SR Skill 实验。"},
    {"dataset": "Omni3D", "id": "0",
     "reason": "高度比答案来自视觉量级估计，缺少实际测量与稳定尺寸绑定，不用于本组展示。"}
], ensure_ascii=False, indent=2))

css = (AUTHOR / "assets/reader.css").read_text()
js = (AUTHOR / "assets/embodied.js").read_text() + "\n" + (AUTHOR / "assets/reader.js").read_text()
shell = (AUTHOR / "reader-template.html").read_text()
qa_cases = cases
embodied = json.loads((ROOT / "build/embodied-reviewed-cases.json").read_text())
cases = embodied + cases
assets.update(json.loads((ROOT / "build/embodied-assets.json").read_text()))
full_order = EB_ORDER + ORDER
manifest = []
for name in ["index", "EmbodiedBench"] + EB_ORDER + ORDER:
    selected = [c for ds in full_order for c in cases if c["dataset"] == ds and
                (name == "index" or ds == name or name == "EmbodiedBench" and c.get("kind") == "embodied")]
    used = images(selected)
    packed_assets = {}
    for h in used:
        a = assets[h]
        raw = (ROOT / a["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == h
        packed_assets[h] = {"mime": a["mime"], "width": a["width"], "height": a["height"],
                            "bytes": a["bytes"], "base64": base64.b64encode(raw).decode()}
    data = {"cases": selected, "assets": packed_assets, "datasets": [ds for ds in full_order if any(c["dataset"] == ds for c in selected)],
            "generated": "2026-11-13", "single_dataset": name not in ("index", "EmbodiedBench")}
    title = "空间智能 · 系统优势案例" + ("" if name == "index" else " · " + name)
    result = shell.replace("__TITLE__", title).replace("__CSS__", css).replace("__DATA__", compact(data)).replace("__JS__", js)
    has_eb = any(c.get("kind") == "embodied" for c in selected)
    has_qa = any(c.get("kind") != "embodied" for c in selected)
    result = result.replace("__HEADLINE__", "从空间理解<br>到实际完成" if has_eb else "从空间线索<br>到正确答案")
    result = result.replace("__INTRO__", (
        "具身任务优先：沿着实际画面、动作和环境反馈，查看系统如何完成任务。其余六个数据集保留系统与 Direct 的完整对比。"
        if has_eb and has_qa else
        "沿着实际画面、动作和环境反馈，查看系统如何抓放、堆叠、搬运、清洗和导航。点选案例即可查看过程。"
        if has_eb else
        "逐题对照系统与直接回答，沿着原始图片、参照物和推导记录，查看答案差异从何而来。"))
    result = result.replace("__NOTE__", "具身四子集各 5 条 · 其他六集各 5 题" if has_eb and has_qa else
                            "四个子集各 5 条成功轨迹" if name == "EmbodiedBench" else
                            "精选 5 条成功轨迹" if has_eb else "每个数据集精选 5 题")
    scope = ("具身部分来自同一 EmbodiedBench 实验的已有系统成功轨迹，四个环境子集各 5 条；未设置 Direct 对照。"
             if has_eb else "")
    scope += ("问答部分均为系统作答正确、Direct 实际作答且错误的入选案例，保留双方上下文和图片。"
              if has_qa else "")
    scope += "每条案例的未成功查询与修正过程按原记录呈现。案例用于展示具体行为，全集表现见相应完整评测。"
    result = result.replace("__SCOPE__", scope)
    record_notes = ("具身请求原文及图片逐条核验；环境步与观察序号分别标注。页面包含原始 RGB、同帧深度、检测候选、全部动作与反馈。原始数值深度另保存在制作资料中。"
                    if has_eb else "")
    record_notes += ("问答原文逐轮保留；恢复的请求标为“据记录恢复”，未归档的 thinking 明确注明。Omni3D 原标签正文未归档，显示原评分为 1.0 的匹配作答。VSIbench 两臂 effort 标签不同，原帧像素逐张核验。"
                     if has_qa else "")
    result = result.replace("__RECORD_NOTES__", record_notes)
    result = result.replace("__FOOTER_COUNT__", f"{len(selected)} 条精选案例 · 图片与记录全部内嵌")
    path = ROOT / (name + ".html")
    path.write_text(result)
    manifest.append({"file": path.name, "bytes": path.stat().st_size, "cases": len(selected),
                     "unique_images": len(used), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

readme = """空间智能系统优势案例 · 2026-11-13 · 具身优先版

打开 index.html 可查看全部 50 条案例，首先呈现具身任务。
EmbodiedBench.html：20 条系统成功轨迹，四个环境子集各 5 条，不设 Direct 对照。
EB-Manipulation.html / EB-Navigation.html / EB-ALFRED.html / EB-Habitat.html：每份独立包含该子集的 5 条成功轨迹。
其余 VSIbench / CV3D / MindCube / MMSI / BLINK / Omni3D：每份 5 题，继续保留系统正确、Direct 错误的双边对比。
所有成品 HTML 都是单文件：图片、上下文、样式和交互已经内嵌。
接收方不需要服务器、VPN、网络、assets 文件夹或 Python，使用 Chrome、Edge、Safari 等现代浏览器直接打开即可。
若聊天应用只显示预览，请先下载 HTML，再选择用浏览器打开。

具身使用：点击案例即可看到原始任务、初始/关键/最终画面和实际动作；拖动滑块或点动作查看对应 RGB、深度与环境反馈；展开逐轮输入和完整模型输出。
问答使用：按数据集切换；搜索题号或关键词；点击图片放大；展开每轮记录查看两边输入、thinking、原始回答和工具返回。
具身的20条均核对官方 task_success=1、done=true、动作回执、初始未成功、原始结果/轨迹/请求哈希；筛选解读见 review/embodied-selection-audit.md。
同帧原始数值深度保存在 source/embodied 中的 trajectory.jsonl 和 depth-meters.npy，HTML 呈现同步深度图、统计和来源。
这是经过上下文复核的获胜案例集，各选五题，不是完整测试集准确率报告。
几何测量、位姿提示与 Skill 视觉/坐标推理按实际证据标注。
归档图片按原字节嵌入；VSIbench 视频帧按经 SHA-256 核验的原始 RGB 像素保存。
BLINK / MMSI / VSIbench 的隐藏 thinking 没有归档；CV3D / MMSI / VSIbench 的请求正文按已保存记录恢复，页面逐轮标注。
Omni3D 原标签正文未归档，页面显示评分为 1.0 的系统匹配作答，并保留两边原始评分。
VSIbench 两臂 effort 为 max 与 low（官方 low 兼容映射 high），不能将单个案例解释为独立控制的工具因果消融。

新的50条ZIP和具身20条ZIP仅含成品 HTML 和本说明。旧30题ZIP保留为之前版本。
完整制作资料保留在 system-advantage-cases-20261113：
build/ 归一化上下文、成品作者源文件及本地验证环境
assets/ 原始实验图片
review/ 逐题解读、候选取舍、图像审查与参照物记录
source/ 来源索引、回收的冻结帧与原视频；embodied/ 含原始20条入选轨迹及被替换候选资料
scripts/ 提取、审查、构建与验证脚本
checks/ 离线浏览器验证与截图
"""
(ROOT / "README.txt").write_text(readme)
(ROOT / "build/delivery-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
(ROOT / "build/reviewed-cases.json").write_text(json.dumps(qa_cases, ensure_ascii=False, indent=2))
with zipfile.ZipFile(ROOT / "空间智能系统优势案例-50条-具身优先-离线分享.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for m in manifest:
        z.write(ROOT / m["file"], m["file"])
    z.write(ROOT / "README.txt", "README.txt")
with zipfile.ZipFile(ROOT / "EmbodiedBench-20条成功轨迹-离线分享.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for m in manifest:
        if m["file"] == "EmbodiedBench.html" or m["file"].startswith("EB-"):
            z.write(ROOT / m["file"], m["file"])
    z.write(ROOT / "README.txt", "README.txt")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
