from pathlib import Path
import difflib
import hashlib
import html
import json
import shutil

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[2]
OUT = Path("/Users/bytedance/wangjh/vsi-direction-review-20261111")
SKILL = Path("/Users/bytedance/.trae-cn/plugins/trae-remote-official/html-report-presentation/0.1.0/skills/html-report")
ASSETS = OUT / "assets"
intake = json.loads((WORK / "intake.json").read_text())
manifest = json.loads((WORK / "images/manifest.json").read_text())
findings = json.loads((WORK / "visual_findings.json").read_text())

findings.update({
    "3711": dict(title="直答左右错误；工具组是零工具视觉正确", roles="窗 → 桌子 → 毛巾；easy",
        finding="工具组R1视觉A/left，零工具。直答只保存B/right，不能还原内部决策。",
        pixels="第5个Agent核验frame-5/6窗旁青绿色长布、13楼梯旁桌子、30金属货架。存在多块布与多处桌面，left与一组局部布局相容，但实例身份未独立认证。",
        certainty="高：两组结果和零工具路径。低：直答因果；没有保存thinking。",
        action="保留零工具视觉路线；不从最终字母反推模型沿用了相机左右。", frames=[5, 6, 13, 30]),
    "1572": dict(title="直答选back；真实椅子是黑椅，蓝色物体是切纸器", roles="天花板灯 → 房门 → 椅子；medium",
        finding="工具组R1视觉C/left，零工具；直答A/back。没有可重放角度或实际实例选择记录。",
        pixels="第5个Agent核验frame-0长条灯、7/22黑色穿孔靠背椅与蓝色切纸器、31白门。物体外观有支持，不能从原图量出135°转角。",
        certainty="高：left/back最终分类不同。低：直答是错实例还是角度判断；不能认定为左右翻转。",
        action="保留水平观察者定义与135°，不强制几何，不添加灯位固定等场景特例。", frames=[0, 7, 22, 31]),
    "2425": dict(title="工具采纳的正确对照，两个垃圾桶是相邻实物", roles="电话 → 垃圾桶 → 杯子；easy",
        finding="R3两个垃圾桶候选均left，R4引用真实result_id映射A；直答B/right错。R1-R4没有裁剪或原帧身份注释。",
        pixels="第5个Agent核验frame-9电话、31白色带把杯；0/16红圆桶、24相邻白方桶。两个桶是不同实物；审查bbox支持所选物体外观，无显见错类。",
        certainty="高：工具确实决定正确输出、bbox外观有支持。mask纯度、完整质心和world_up仍未认证；直答thinking未知。",
        action="保留三角色、双候选与option_map；不因成本直接删除有意义的歧义覆盖。", frames=[0, 9, 16, 24, 31]),
    "4325": dict(title="三轮成本控制：无语法错误，补跑视觉正确", roles="床头柜 → 镜子 → 窗；按实际题面查看",
        finding="r15补跑R1清单、R2分页、R3视觉B正确；只1次工具调用，无方向查询、无语法错误。原超时请求Token未知，未计入成本。",
        pixels="第5个Agent核验frame-1床头柜上的红色灯与棕色灯罩，支持最终注释的外观；没有独立认证窗/镜几何。",
        certainty="高：73088已知Token、三轮路径、无语法错误。不能证明删除前两轮保留正确答案。",
        action="保留传输中断独立统计；精简重复接口规则的收益需独立消融。", frames=[1])
})

old_skill = (REPO / "skill_library/versions/S06-g1m0/1.0.0/judge-relative-direction-g1m0/SKILL.md").read_text()
candidate_lines = {
    "2.": "2. Fix each role to a visible instance before geometry. Record `role=obj_id @ image_id + region` in a brief program comment. Check that candidate's box/mask itself, not another object in the same frame. Use modifiers and fixed structures to distinguish instances across rooms or sofas.",
    "4.": "4. With authorized geometry, valid local world conventions and usable bindings for the confirmed instances, call `relative_direction_of(observer_id, facing_at_id, target_id, difficulty)`. Check `direction`, `difficulty`, `world_up_used` and `handedness_used`. Never borrow another object's centroid for a missing role.",
    "6.": "6. For visual reasoning, keep O, F and T in one local horizontal layout and compare O->F with O->T. For medium, back requires a turn >=135 degrees; opposite room ends alone do not establish it. Preserve the same instances across views; do not reuse camera left/right."
}
new_lines = []
for line in old_skill.splitlines():
    prefix = line[:2]
    if prefix in candidate_lines:
        line = candidate_lines[prefix]
    elif line.startswith("- Near direction boundaries"):
        line = "- If plausible instances change the option, or the layout is mirrored or near a boundary, inspect the distinguishing views before accepting a direction. Do not vote over candidates or select the first successful result."
    elif line.startswith("- M5 provides candidates"):
        line = "- M5 offers candidates and technical checks, not referent certification. Missing bindings use visual locations."
    new_lines.append(line)
new_skill = "\n".join(new_lines) + "\n"
(ASSETS / "S06-g1m0-original.md").write_text(old_skill)
(ASSETS / "S06-g1m0-candidate.md").write_text(new_skill)
(ASSETS / "S06-g1m0-candidate.diff").write_text("".join(difflib.unified_diff(
    old_skill.splitlines(True), new_skill.splitlines(True),
    fromfile="S06-g1m0@1.0.0/original", tofile="review-only/S06-g1m0-candidate")))
prompt_source = (REPO / "src/skill3d/synthesis/prompt_builder.py").read_text()
old_prompt = next(x for x in prompt_source.splitlines() if x.startswith("- M5 candidates and resolved tool IDs"))
new_prompt = "- M5/Tool IDs identify candidates, not question referents or scene completeness. Match depicted regions and spatial qualifiers before selecting IDs; keep missing bindings visual. M11 checks technical validity/derivation, not intended identity."
prompt_draft = f"""# 共享提示词单行替换候选（未应用）

位置：src/skill3d/synthesis/prompt_builder.py:72。

原句（{len(old_prompt)}字符）：

```text
{old_prompt}
```

候选（{len(new_prompt)}字符）：

```text
{new_prompt}
```

具体三锚点注释与射线动作留在S06；共享规则不再复制整套方法。
当前S06 {len(old_skill)}字符，候选 {len(new_skill)}字符，净差 {len(new_skill)-len(old_skill):+d}字符。
字符数不是token数，未验证模型成本或准确率收益。
这些候选没有写入活跃Skill、prompt或运行配置。
"""
(ASSETS / "prompt-candidate.md").write_text(prompt_draft)
shutil.copy2("/Users/bytedance/wangjh/VSI相对方向30题五Agent复核-20261111.md", ASSETS / "完整复核报告.md")
for i in range(1, 6):
    shutil.copy2(WORK / f"agent{i}_findings.md", ASSETS / f"agent{i}_findings.md")
shutil.copy2(WORK / "html-plan.md", ASSETS / "report-plan.md")
shutil.copy2(SKILL / "assets/js/echarts.min.js", OUT / "_shared/js/echarts.min.js")
for name in ["WorkSans-Regular.ttf", "WorkSans-Bold.ttf", "JetBrainsMono-Regular.ttf"]:
    shutil.copy2(SKILL / "canvas-fonts" / name, OUT / "_shared/fonts" / name)

def selected_fields(d, keys):
    return {k: d[k] for k in keys if k in d}

def summarize_object(obj):
    fields = selected_fields(obj, [
        "obj_id", "category_name", "source_label", "visible_frame_ids", "frame_ids",
        "duplicate_suspect", "mask_available", "binding_validation", "flags"
    ])
    fields["candidate_regions"] = [selected_fields(p, [
        "frame_slot", "bbox", "candidate_id", "category", "semantic_admission"
    ]) for p in obj.get("binding_provenance", [])]
    return fields

def summarize_result(result):
    item = selected_fields(result, [
        "tool", "source_tool", "args", "result_id", "status", "error", "error_code",
        "failure_cause", "failure_guidance", "resolved_object_ids"
    ])
    value = result.get("payload")
    if value is None:
        value = result.get("value")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
    if isinstance(value, list) and value and isinstance(value[0], dict) and "obj_id" in value[0]:
        item["candidate_summary"] = [summarize_object(x) for x in value]
    elif isinstance(value, dict) and "detections" in value:
        item["payload_summary"] = {
            "n_detections": len(value["detections"]),
            "detections": [selected_fields(x, [
                "frame_id", "frame_slot", "category", "category_name", "bbox", "bbox_xyxy",
                "obj_id", "score"
            ]) for x in value["detections"]]
        }
    else:
        item["payload"] = value
    return item

cases = []
copied_images = {}
source_manifest = {"model_requests": 0, "cases": {}}
for pair in intake["all_pairs"]:
    qa = pair["qa_id"]
    if qa not in findings:
        continue
    outcome = json.loads(Path(pair["system_files"]["outcome_file"]).read_text())
    direct = json.loads(Path(pair["direct_files"]["response_file"]).read_text())
    frames = []
    image_case = manifest["cases"][qa]
    assert image_case["all_jpegs_equal_actual_model_request"] is True
    for frame in image_case["frames"] + [image_case["overview"]]:
        source = WORK / "images" / frame["path"]
        target = ASSETS / "images" / frame["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if str(target) not in copied_images:
            shutil.copy2(source, target)
            copied_images[str(target)] = hashlib.sha256(target.read_bytes()).hexdigest()
        assert copied_images[str(target)] == frame["sha256"]
        frames.append({**frame, "path": target.relative_to(OUT).as_posix()})
    contacts = []
    for name in ["frames_0_15.jpg", "frames_16_31.jpg"]:
        source = WORK / "images/contacts" / qa / name
        target = ASSETS / "contacts" / qa / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        contacts.append(target.relative_to(OUT).as_posix())
    source_manifest["cases"][qa] = {
        "scene_id": pair["scene_id"],
        "all_jpegs_equal_actual_model_request": True,
        "frames": frames
    }
    rounds = []
    for r in outcome["rounds"]:
        rounds.append({
            "index": r["index"], "executed": r["executed"],
            "tool_calls": r["tool_calls"], "error_code": r.get("error_code"),
            "program_source": r.get("program_source", ""),
            "feedback": r.get("feedback", ""),
            "results": [summarize_result(x) for x in r.get("results", [])],
            "observation_reads": r.get("observation_reads", []),
            "usage": r.get("generation", {}).get("usage", {})
        })
    system_wrong = pair["system_score"] == 0
    direct_wrong = pair["direct_score"] == 0
    entry = {
        "qa_id": qa, "scene_id": pair["scene_id"],
        "question": pair["question"], "options": pair["options"],
        "system_answer": pair["system_answer"], "direct_answer": pair["direct_answer"],
        "reference_answer": pair["reference_answer"],
        "system_wrong": system_wrong, "direct_wrong": direct_wrong,
        "category": "both" if system_wrong and direct_wrong else "tool" if system_wrong else "direct" if direct_wrong else "control",
        "finding": findings[qa], "frames": frames, "contacts": contacts,
        "rounds": rounds, "answer_payload": outcome.get("answer_payload"),
        "verify": outcome.get("verify"),
        "candidate_coverage": outcome.get("m5_summary", {}).get("question_binding", {}).get("candidate_coverage"),
        "direct_response_text": direct.get("response_text"),
        "direct_usage": direct.get("usage"),
        "source": {"outcome": pair["system_files"]["outcome_file"], "direct": pair["direct_files"]["response_file"]},
        "source_pointers": ["/rounds/<i>/program_source", "/rounds/<i>/results", "/answer_payload", "/verify", "/m5_summary/question_binding"]
    }
    cases.append(entry)
    (ASSETS / "evidence").mkdir(exist_ok=True)
    (ASSETS / "evidence" / f"{qa}.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2))

cases.sort(key=lambda c: ({"tool": 0, "both": 0, "direct": 1, "control": 2}[c["category"]], int(c["qa_id"])))
payload = {
    "cases": cases,
    "aggregate": {"tool_correct": 23, "direct_correct": 25, "n": 30, "tool_tokens": 1738590, "direct_tokens": 592280},
    "draft": {"original_chars": len(old_skill), "candidate_chars": len(new_skill), "prompt_old_chars": len(old_prompt), "prompt_candidate_chars": len(new_prompt)},
    "skill_sha256": hashlib.sha256(old_skill.encode()).hexdigest()
}
(ASSETS / "review-data.js").write_text("window.REVIEW = " + json.dumps(payload, ensure_ascii=False) + ";\n")
(ASSETS / "image-manifest.json").write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2))
(ASSETS / "visual-findings.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2))
page = """<!-- Generated by Trae Work -->
<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VSI相对方向30题 · 五Agent证据复核</title>
<style>
@font-face{font-family:WorkSans;src:url('./_shared/fonts/WorkSans-Regular.ttf')}@font-face{font-family:WorkSans;src:url('./_shared/fonts/WorkSans-Bold.ttf');font-weight:700}
@font-face{font-family:JetBrains;src:url('./_shared/fonts/JetBrainsMono-Regular.ttf')}
:root{--bg:#f5f3ee;--bg2:#ffffff;--ink:#182b35;--muted:#52646d;--rule:#d9e0e0;--accent:#17645d;--accent2:#a44e35;--body:WorkSans,"PingFang SC","Microsoft YaHei",sans-serif;--mono:JetBrains,monospace}
*{box-sizing:border-box}body{background:var(--bg);color:var(--ink);font:16px/1.7 var(--body);margin:0}main{max-width:1220px;margin:auto;padding:44px 28px}h1{font-size:38px;line-height:1.25;letter-spacing:-.03em;margin:14px 0 22px}h2{font-size:25px;margin:35px 0 15px}h3{font-size:21px;margin:12px 0}p{margin:10px 0}a{color:var(--accent)}.eyebrow{color:var(--accent);font-weight:700;font-size:14px;letter-spacing:.08em}
.muted,.meta{color:var(--muted)}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:25px 0}.metric{background:var(--bg2);padding:21px;border:1px solid var(--rule)}.metric strong{display:block;font-size:32px;color:var(--accent);line-height:1.3}.metric span{font-size:14px;color:var(--muted)}
.panel,.case{background:var(--bg2);border:1px solid var(--rule);padding:24px;margin:22px 0;border-radius:9px}.case header{display:flex;justify-content:space-between;gap:15px;align-items:start}.badge{font-size:14px;padding:5px 12px;border:1px solid var(--rule);border-radius:30px;white-space:nowrap;color:var(--accent2)}
.table-wrap{overflow-x:auto;overflow-y:auto;max-height:600px}table{width:100%;border-collapse:collapse}th,td{padding:12px;text-align:left;border-bottom:1px solid var(--rule)}th{font-size:14px;color:var(--muted)}.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:30px 0}input,select,button{font:inherit;color:var(--ink);background:var(--bg2);border:1px solid var(--rule);padding:9px 13px;border-radius:6px}input{width:250px}button{cursor:pointer}button:hover{border-color:var(--accent)}select:focus,input:focus,button:focus,summary:focus{outline:2px solid var(--accent);outline-offset:2px}
.question{padding:14px 18px;background:var(--bg);border-left:3px solid var(--accent);white-space:pre-wrap}.options{font-size:14px;color:var(--muted)}.answers{display:flex;gap:22px;flex-wrap:wrap;margin:14px 0}.wrong{color:var(--accent2)}.correct{color:var(--accent)}.gallery{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}.gallery figure{margin:0;min-width:0}.gallery button{display:block;width:100%;padding:0;overflow:hidden;border-radius:5px}.gallery img{display:block;width:100%;height:auto;aspect-ratio:4/3;object-fit:contain;background:var(--bg)}figcaption{font-size:12px;color:var(--muted);padding-top:5px}.contact{width:100%;height:auto}.image-contact{margin:18px 0}.image-contact a{display:block}.chart{height:300px}.chart-figure{margin:24px 0}.chart-figure figcaption{font-size:15px;font-weight:700;color:var(--ink)}
details{margin:14px 0;border-top:1px solid var(--rule);padding-top:12px}summary{cursor:pointer;color:var(--accent);font-weight:700}pre{font:13px/1.65 var(--mono);background:var(--bg);padding:16px;overflow:auto;max-height:580px;white-space:pre-wrap;overflow-wrap:anywhere}code{font-family:var(--mono);font-size:.9em}.round-meta{font-size:14px;color:var(--muted);overflow-wrap:anywhere}.small{font-size:14px}.citation{font-size:12px}.sources ol{padding-left:1.2rem;font-size:14px;color:var(--muted)}.sources li{margin:10px 0;overflow-wrap:break-word}.sources a{word-break:break-all}.sources .src-title{color:var(--ink)}sup a{color:var(--accent);text-decoration:none;font-size:.75em;font-weight:700}
dialog{background:var(--bg2);color:var(--ink);border:1px solid var(--rule);border-radius:8px;width:min(1080px,96vw);max-height:96vh;padding:16px}dialog::backdrop{background:var(--ink);opacity:.85}dialog img{display:block;width:auto;max-width:100%;max-height:73vh;margin:10px auto}dialog .modal-head{display:flex;justify-content:space-between;gap:15px}.hash{font-family:var(--mono);font-size:12px;overflow-wrap:anywhere}.case[hidden]{display:none}
@media(max-width:768px){main{padding:25px 14px}h1{font-size:30px}.metrics{grid-template-columns:repeat(2,1fr)}.gallery{grid-template-columns:repeat(2,1fr)}.case,.panel{padding:17px}.case header{display:block}.badge{display:inline-block}table{min-width:650px}}
@media(max-width:450px){.gallery{grid-template-columns:1fr}.metrics{gap:8px}.metric{padding:14px}.metric strong{font-size:27px}}
@media print{body{background:var(--bg2)}.controls,button,.chart-figure,dialog{display:none}.case{break-before:page}.gallery{grid-template-columns:repeat(4,1fr)}details:not([open])>*:not(summary){display:block}.case[hidden]{display:block}pre{max-height:none}.table-wrap{max-height:none;overflow:visible}}
</style></head>
<body><main>
<div class="eyebrow">VSI BENCH · 2026 / 11 / 11</div>
<h1>相对方向30题<br>五Agent错题证据复核</h1>
<p><strong>建议小改实例检查与视觉射线步骤，优先修候选数据和呈现。</strong> 当前S06已有主要规则；实际错误集中在实例替换、跨视角布局和未经支持的角度分类。活跃Skill、提示词与配置尚未改动。<sup><a href="#cite-1">[1]</a></sup></p>
<div class="metrics"><div class="metric"><strong>23 / 30</strong><span>工具组正确 · 76.7%</span></div><div class="metric"><strong>25 / 30</strong><span>Qwen直答正确 · 83.3%</span></div><div class="metric"><strong>11道</strong><span>错题覆盖 · 12条错误输出</span></div><div class="metric"><strong>2.94×</strong><span>合并已知Token成本</span></div></div>
<figure class="chart-figure"><figcaption>两批与合并准确率对照</figcaption><div id="accuracy-chart" class="chart"></div></figure>
<p class="small muted">两组32张原始RGB同序同哈希。工具组另加全景图、Skill与工具/程序协议，完整上下文不相同。原10题直答复用r11，工具组取r13；新增取r14，QA4325工具补跑取r15。超时未知Token未计入。此查看器没有发模型请求。</p>
<section class="panel"><h2>调整候选</h2><p><strong>采纳几何前记录三锚点的ID、图像与区域，核对候选本身；视觉回退保留同一组三个实例，比较O→F与O→T。</strong> medium的back仍按至少135°，不从“房间对端”直接推后方。合理实例会改变选项时先消歧，不取首成功或多数投票。</p>
<p>不放宽confidence_gate、不翻转左右、不强制每题工具化或三者同框。数据接口优先呈现所选候选区域与逐实例绑定状态；保存真实thinking，补方向角度与边界余量供审查。</p>
<p class="small">候选未应用，准确率与Token收益尚未测试。程序短注释提供可追溯依据，不能自动认证语义身份。</p>
<p><a href="assets/S06-g1m0-candidate.md">S06候选全文</a> · <a href="assets/S06-g1m0-candidate.diff">逐行差异</a> · <a href="assets/prompt-candidate.md">共享prompt单行替换</a> · <a href="assets/完整复核报告.md">完整Markdown报告</a></p>
<p id="draft-sizes" class="small muted"></p></section>
<h2>原帧与逐轮实际证据</h2><p class="muted">先看每题的关键原帧；展开查看全部32帧、全景、实际程序与反馈。点击图片可放大并核对image_id、物理帧号和SHA256。下列图像复核是审查结果，不能回填成模型当时已完成的核对。</p>
<div class="controls"><label>筛选 <select id="filter"><option value="all">全部11错题＋成本控制</option><option value="system">工具组错误（7题）</option><option value="direct">直答错误（5题）</option><option value="both">两组都错（1题）</option><option value="control">成本控制（1题）</option></select></label><label>题号/关键词 <input id="search" placeholder="例如 1860、沙发、135"></label><span id="count" class="meta"></span></div>
<div id="cases"></div>
<footer class="sources"><h2>证据与来源</h2><ol>
<li id="cite-1"><span class="src-title">主Agent：完整复核报告、像素结论与最小改稿。</span><br><a href="assets/完整复核报告.md" target="_blank">完整复核报告.md</a> · <a href="assets/visual-findings.json" target="_blank">逐题像素补核</a></li>
<li id="cite-2"><span class="src-title">真实冻结图像：逐张匹配实际模型请求JPEG SHA256。</span><br><a href="assets/image-manifest.json" target="_blank">图像身份清单</a></li>
<li id="cite-3"><span class="src-title">5个Agent的独立分工报告；早期待原帧标记以主Agent补核和Agent5最终追加为准。</span><br>
<a href="assets/agent1_findings.md" target="_blank">Agent1 · 1556/1558</a> · <a href="assets/agent2_findings.md" target="_blank">Agent2 · 3575/3588</a> · <a href="assets/agent3_findings.md" target="_blank">Agent3 · 1536/1860</a> · <a href="assets/agent4_findings.md" target="_blank">Agent4 · 3663/4259</a> · <a href="assets/agent5_findings.md" target="_blank">Agent5 · 直答错误与正确控制</a></li>
<li id="cite-4"><span class="src-title">实际S06送达完整，3530字符；比较候选只保存在本交付目录。</span><br><a href="assets/S06-g1m0-original.md" target="_blank">原Skill</a> · <a href="assets/S06-g1m0-candidate.diff" target="_blank">候选diff</a></li>
</ol><p class="small muted">所有结果摘录均可由每题证据JSON中的源路径和JSON pointer追溯。候选摘要省略校准日志等非核心字段；反馈与程序保留实际文本。passed/validated不认证题目实例。直答缺thinking，内部错因保留未知。</p></footer>
</main>
<dialog id="image-dialog"><div class="modal-head"><strong id="modal-title"></strong><button id="close-modal" aria-label="关闭图片">关闭</button></div><img id="modal-image" alt=""><p id="modal-hash" class="hash"></p></dialog>
<script src="./_shared/js/echarts.min.js"></script><script src="assets/review-data.js"></script><script src="assets/charts.js"></script><script src="assets/review.js"></script>
</body></html>
"""
(OUT / "vsi-direction-review-20261111.html").write_text(page)
print(json.dumps({
    "cases": len(cases), "system_errors": sum(c["system_wrong"] for c in cases),
    "direct_errors": sum(c["direct_wrong"] for c in cases),
    "verified_original_images": len(copied_images), "draft": payload["draft"],
    "output": str(OUT)
}, ensure_ascii=False, indent=2))
