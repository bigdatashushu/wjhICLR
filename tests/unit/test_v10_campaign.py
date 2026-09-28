"""v10 §11 / §16：EvolutionCampaign 两代闭环的**结构性**测试（合成运行）。

纪律（§16.3）：本文件的运行全部是合成 / 注入的 —— 它只证明**结构与不变量**成立
（状态机、收据、经验绑定、谱系竞争、发布后验证、恢复不重复消费），**不**构成首期
验收证据。真实验收见 `data/v10_campaign/` 的收据与报告。

测试用的 `_FakeRuntime` 仍然遵守真实合同的形状：learning 运行把 trace 写进
trace_dir（经验包从落盘 trace 构建，而不是从内存对象），固定注入臂把"该臂正文 hash"
写进检索记录，发布后的 learning 运行只在 active 快照里出现新版本时才交付新版本。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from skill3d.evolution.campaign import (
    CampaignConfig,
    CampaignRuntime,
    EvolutionCampaignRunner,
    seed_campaign_library,
)
from skill3d.evolution.experience import build_experience_bundle, build_experience_events
from skill3d.evolution.panel import fixed_injection_binding
from skill3d.online.runner import OnlineRunConfig
from skill3d.schemas import SkillSpec
from skill3d.skills.delivery import skill_content_sha256
from skill3d.skills.promote_atomic import read_active_snapshot
from skill3d.skills.registry import load_active_skills

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# 合成运行实现（形状与真实一致：写 trace、写交付 hash）
# --------------------------------------------------------------------------- #

class _FakeRuntime(CampaignRuntime):
    """确定性的合成运行：不调模型、不跑沙箱，但**落盘**与真实路径同形。"""

    def __init__(self, *, parent_qa: int = 4, inner_items: int = 4,
                 post_items: int = 2, parent_score: float = 0.25,
                 candidate_score: float = 0.75):
        self.parent_qa = parent_qa
        self.inner_n = inner_items
        self.post_n = post_items
        self.parent_score = parent_score
        self.candidate_score = candidate_score
        self.learning_calls = 0
        self.inner_calls = 0
        self.promoted_version: str = ""

    # ---- learning：正常检索（用 active 快照里的最新版本作"被交付版本"）----
    def run_learning(self, *, items, run_cfg: OnlineRunConfig, trace_dir: str,
                     llm=None):
        self.learning_calls += 1
        root = Path(trace_dir)
        root.mkdir(parents=True, exist_ok=True)
        target = [s for s in run_cfg.skills if s.skill_id == "S01"]
        target.sort(key=lambda s: tuple(int(p) for p in s.version.split(".")))
        delivered = target[-1] if target else None
        # "新版本更好"：合成口径是"不是 1.0.0 的版本答得更好"，与快照加载结果绑定
        # （真实路径里交付的就是 active 快照里的代表版本）。
        newest = bool(delivered is not None and delivered.version != "1.0.0")
        lines = []
        eval_lines = []
        for idx, item in enumerate(items):
            qa = str(item.episode.qa_id)
            correct = idx % 2 == 0 if newest else (idx % 4 == 0)
            key = f"{delivered.skill_id}@{delivered.version}" if delivered else ""
            record = {
                "retrieval_index": 1, "trigger": "initial",
                "canonical_question_type": "object_counting",
                "n_skills_offered": len(run_cfg.skills),
                "eligible_skill_versions": [key] if key else [],
                "retrieved_skill_versions": [key] if key else [],
                "delivered_skill_versions": [key] if key else [],
                "declared_selected_skill_versions": [key] if key else [],
                "candidates": [{
                    "skill_id": "S01", "version": delivered.version if delivered else "",
                    "skill_version": key, "hard_filter_passed": True, "selected": True,
                    "reason_code": "hit", "delivered": bool(key),
                    "delivery_reason": "delivered" if key else "not_selected",
                    "score": 0.9, "rank": 1,
                }],
                "delivered_content_sha256": ({key: skill_content_sha256(delivered)}
                                             if key else {}),
                "usage_clues": ([{"skill_version": key, "clue_source": "static_scan",
                                  "declared_in_program": True,
                                  "template_tool_overlap": ["detect_objects"]}]
                                if key else []),
                "lineage_selections": [],
            }
            lines.append(json.dumps({
                "qa_id": qa, "episode_id": qa, "schema_version": "9.0",
                "active_snapshot_ref": run_cfg.active_snapshot_ref,
                "retrieval_records": [record],
                "failure": (None if correct else {"categories": ["perception"]}),
                "failure_code": None if correct else "grounding_recall_miss",
                "skill_mapping_misses": [],
            }, ensure_ascii=False))
            eval_lines.append(json.dumps({
                "qa_id": qa, "correct": bool(correct), "mra_value": None,
            }, ensure_ascii=False))
        (root / "episode_trace.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (root / "evaluation_result.jsonl").write_text("\n".join(eval_lines) + "\n",
                                                      encoding="utf-8")
        return [_FakeOutcome(f"{item.episode.qa_id}", correct=False) for item in items]

    # ---- inner：固定注入 A/B ----
    def run_inner(self, *, campaign_id, generation, seed, panel_id, items,
                  parent_spec: SkillSpec, candidate_spec: SkillSpec, base_cfg,
                  trace_store=None, llm=None):
        from skill3d.schemas import PairedPanelReceipt

        self.inner_calls += 1
        binding_a = fixed_injection_binding("parent", parent_spec)
        binding_b = fixed_injection_binding("candidate", candidate_spec)
        n = len(items)
        per_item = []
        for idx, item in enumerate(items):
            per_item.append({
                "qa_id": str(item.episode.qa_id), "scene_id": str(item.episode.scene_name),
                "score_a": 0.0, "score_b": 1.0, "final_state_a": "answer",
                "final_state_b": "answer", "run_error_a": False, "run_error_b": False,
                "valid_answer_a": True, "valid_answer_b": True,
                "delivered_sha_a": binding_a.content_sha256,
                "delivered_sha_b": binding_b.content_sha256,
            })
        return PairedPanelReceipt(
            campaign_id=campaign_id, generation=int(generation), seed=int(seed),
            panel_id=panel_id, n_items=n,
            arm_a_skill_version=binding_a.skill_version,
            arm_b_skill_version=binding_b.skill_version,
            arm_a_content_sha256=binding_a.content_sha256,
            arm_b_content_sha256=binding_b.content_sha256,
            mean_a=self.parent_score, mean_b=self.candidate_score,
            delta=self.candidate_score - self.parent_score,
            n_run_error_a=0, n_run_error_b=0,
            valid_answer_rate_a=1.0, valid_answer_rate_b=1.0,
            delivered_a=n, delivered_b=n, per_item=per_item), {"a": [], "b": []}


class _FakeOutcome:
    def __init__(self, qa_id: str, *, correct: bool):
        self.qa_id = qa_id
        self.correct = correct
        self.final_state = "answer"
        self.retrieval_records: list = []
        self.answer_payload = {"value": 1}


# --------------------------------------------------------------------------- #

class _FakeOffline:
    """离线归纳器替身：输出**完整候选 SkillSpec**（§5.4 推荐路径的形状）。"""

    def __init__(self):
        self.calls = 0
        self.last_prompt = ""

    def chat(self, prompt: str) -> str:
        self.calls += 1
        self.last_prompt = prompt
        parent = json.loads(prompt.split("## 父 Skill 完整 SkillSpec", 1)[1]
                            .split("\n", 1)[1].split("\n## ", 1)[0])
        major, minor, _patch = (int(p) for p in str(parent["version"]).split("."))
        parent["version"] = f"{major}.{minor + 1}.0"
        parent["description"] = (str(parent.get("description", ""))
                                 + f" 修订 {parent['version']}：先锁定目标类别再跨帧覆盖")
        parent["call_graph_template"] = (str(parent.get("call_graph_template", ""))
                                         + "\n- 空检出错题时要复核")
        return json.dumps({
            "spec": parent,
            "hypothesis": "检测空检出时先回退到视觉复核",
            "expected_effect": "object_counting 得分提高",
            "known_risks": ["可能增加一轮模型调用"],
            "diff_summary": "细化空检出分支",
        }, ensure_ascii=False)


def _object_counting_spec(version: str = "1.0.0") -> SkillSpec:
    return SkillSpec(
        skill_id="S01", version=version,
        applicable_question_types=["object_counting"], skill_family="counting",
        description="数出场景中出现的目标对象数量",
        call_graph_template="1) detect_objects(img, category) 统计实例\n2) 跨帧去重",
        supported_coordinate_frames=["world"],
        source="real")


def _seed_library(tmp_path: Path) -> Path:
    """建一个 S0 形状的 campaign 库（只有 S01，避免与仓库 S0 库耦合）。"""
    lib = tmp_path / "lib"
    store = lib / "snapshots"
    store.mkdir(parents=True)
    spec = _object_counting_spec()
    entry = {
        "root_candidate_id": "S01", "candidate_type": "skill",
        "skill_version": "S01@1.0.0",
        "spec_content": json.dumps(spec.model_dump(mode="json"), ensure_ascii=False,
                                   indent=2, sort_keys=True),
        "parent_version": None, "created_by": "s0_import",
    }
    snapshot = {
        "schema_version": "runtime-skill-snapshot/1.0", "document_schema_version": "8.0",
        "snapshot_id": "S0-seed", "parent_snapshot_id": None, "generation": 0,
        "skill_versions": ["S01@1.0.0"], "manifest_hash": "seed",
        "entries": {"S01@1.0.0": entry},
    }
    (store / "snapshot_S0-seed.json").write_text(json.dumps(snapshot), encoding="utf-8")
    (store / "active_snapshot.json").write_text(
        json.dumps({"snapshot_id": "S0-seed"}), encoding="utf-8")
    return lib


class _Item:
    def __init__(self, qa_id: str, scene: str):
        self.episode = type("E", (), {"qa_id": qa_id, "scene_name": scene,
                                     "question_type": "object_counting"})()
        self.pixels = []
        self.geometry = None


def _panels(n_learning: int, n_inner: int, n_post: int):
    def provider(kind: str, generation: int):
        if kind == "learning":
            # §8.5：learning_g0 / learning_g1 可以来自**同一** learning 场景池，
            # 但必须是两次独立真实运行（因此这里 qa_id 与 generation 无关）。
            return [_Item(f"qa-lear-{i}", f"scene-{i}") for i in range(n_learning)]
        if kind == "inner":
            base = 0 if generation == 1 else 100  # inner_g0 / inner_g1 互不重叠
            return [_Item(f"inner-{generation}-{i}", f"iscene-{base + i}")
                    for i in range(n_inner)]
        if kind == "post_publish":
            return [_Item(f"post-{generation}-{i}", f"pscene-{i}") for i in range(n_post)]
        raise AssertionError(f"未知面板 {kind}")
    return provider


def _runner(tmp_path: Path, *, runtime=None, offline=None, generations: int = 2):
    lib = _seed_library(tmp_path)
    cfg = CampaignConfig(
        campaign_id="camp-test", max_generations=generations, mode="real",
        library_root=str(lib), run_root=str(tmp_path / "runs"),
        learning_limit=8, inner_panel_limit=8, min_eligible_experiences=3)
    runtime = runtime or _FakeRuntime()
    runtime_offline = offline or _FakeOffline()
    runner = EvolutionCampaignRunner(
        cfg, runtime=runtime, offline_client=runtime_offline,
        panels_provider=_panels(4, 4, 2),
        tool_names=["detect_objects"], print_fn=lambda *_: None)
    return runner, runtime, runtime_offline, lib


# --------------------------------------------------------------------------- #
# §16.1 / §16.2 单元与集成断言
# --------------------------------------------------------------------------- #

def test_two_generations_promote_and_receipts_complete(tmp_path):
    """两代都晋升：状态机走完、收据齐全、谱系按 §5.3 竞争与历史化。"""
    runner, runtime, offline, lib = _runner(tmp_path)
    campaign = runner.run()

    assert campaign.completion_status == "completed_two_generations"
    assert campaign.current_generation == 2
    assert offline.calls == 2                      # 每代一次归纳调用
    gen1 = tmp_path / "runs" / "camp-test" / "gen1"
    for name in ("parent_run_manifest.json", "experience_events.jsonl",
                 "experience_bundle.json", "candidate.json", "static_validation.json",
                 "candidate_snapshot.json", "paired_seed_0.json", "paired_seed_1.json",
                 "decision.json", "promotion.json", "post_publish_use.json"):
        assert (gen1 / name).is_file(), f"缺收据 {name}"
    bundle = json.loads((gen1 / "experience_bundle.json").read_text(encoding="utf-8"))
    assert bundle["parent_skill_version"] == "1.0.0"
    assert bundle["canonical_question_type"] == "object_counting"
    assert bundle["scene_count"] >= 3

    gen2 = tmp_path / "runs" / "camp-test" / "gen2"
    bundle2 = json.loads((gen2 / "experience_bundle.json").read_text(encoding="utf-8"))
    # §11.3：第二代从**第一代之后**的新运行经验出发（父版本已变成 1.1.0）
    assert bundle2["parent_skill_version"] == "1.1.0"
    assert bundle2["generation"] == 2
    assert json.loads((gen1 / "experience_bundle.json").read_text(encoding="utf-8"))[
        "bundle_id"] != bundle2["bundle_id"]

    active = read_active_snapshot(lib / "snapshots")
    assert active["generation"] == 2
    assert active["active_skill_versions"] == ["S01@1.1.0", "S01@1.2.0"]
    # §5.3-4/5：第三个版本进入时最旧版本转 historical，但仍在快照里
    assert active["historical_skill_versions"] == ["S01@1.0.0"]
    assert active["entries"]["S01@1.0.0"]["state"] == "historical"
    skills, warnings, _sid = load_active_skills(lib / "snapshots")
    assert [f"{s.skill_id}@{s.version}" for s in skills] == ["S01@1.1.0", "S01@1.2.0"]
    assert any("historical" in w for w in warnings)


def test_post_publish_use_is_observed_and_points_at_new_version(tmp_path):
    """§10.2：promote 后新 learning episode 必须真的检索+交付新版本，且 hash 一致。"""
    runner, runtime, _offline, lib = _runner(tmp_path)
    runner.run()
    receipt = json.loads((tmp_path / "runs" / "camp-test" / "gen1" /
                          "post_publish_use.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "observed"
    assert receipt["retrieved"] and receipt["delivered"]
    assert receipt["request_content_sha256_match"] is True
    assert receipt["delivered_content_sha256"] == receipt["snapshot_content_sha256"]
    assert receipt["experience_event_refs"]                      # 新经验指向新版本
    assert receipt["episodes_run"] == 2


def test_fixed_injection_records_are_not_retrieval_hits(tmp_path):
    """§8.3：固定注入不得记录成正常检索命中（两臂各带自己的正文 hash）。"""
    runner, runtime, _offline, _lib = _runner(tmp_path, generations=1)
    runner.run()
    for seed in (0, 1):
        receipt = json.loads((tmp_path / "runs" / "camp-test" / "gen1" /
                              f"paired_seed_{seed}.json").read_text(encoding="utf-8"))
        assert receipt["arm_a_skill_version"] == "S01@1.0.0"
        assert receipt["arm_b_skill_version"] == "S01@1.1.0"
        assert receipt["arm_a_content_sha256"] != receipt["arm_b_content_sha256"]
        assert receipt["delivered_a"] == receipt["n_items"]
        assert receipt["delivered_b"] == receipt["n_items"]


def test_rejection_keeps_active_pointer_and_next_generation_retries(tmp_path):
    """§13：任一 seed 不提高即 reject；§11.3：第二代基于新经验重试同一父版本。"""
    runtime = _FakeRuntime(candidate_score=0.25, parent_score=0.25)   # 持平 → reject
    runner, runtime, offline, lib = _runner(tmp_path, runtime=runtime)
    before = read_active_snapshot(lib / "snapshots")["snapshot_id"]
    campaign = runner.run()
    assert campaign.completion_status == "completed_with_rejection"
    assert read_active_snapshot(lib / "snapshots")["snapshot_id"] == before  # 指针未动
    gen1 = tmp_path / "runs" / "camp-test" / "gen1"
    decision = json.loads((gen1 / "decision.json").read_text(encoding="utf-8"))
    assert decision["promote"] is False
    assert decision["conditions"]["panel_score_strictly_improved"] is False
    promotion = json.loads((gen1 / "promotion.json").read_text(encoding="utf-8"))
    # §14.1 的 `promotion.json` 在 reject 时是 RejectionReceipt 形态（`outcome` 字段
    # 区分 promote/reject）——"没发布"与"发布了"必须能一眼分开。
    assert promotion["outcome"] == "rejected"
    assert promotion["reasons"]
    # 第二代仍然从同一父快照出发，但有自己的 generation / 运行身份
    gen2_bundle = json.loads((tmp_path / "runs" / "camp-test" / "gen2" /
                              "experience_bundle.json").read_text(encoding="utf-8"))
    assert gen2_bundle["generation"] == 2
    assert gen2_bundle["parent_skill_version"] == "1.0.0"
    assert offline.calls == 2
    gen1_run = json.loads((gen1 / "parent_run_manifest.json").read_text(encoding="utf-8"))
    gen2_run = json.loads((tmp_path / "runs" / "camp-test" / "gen2" /
                           "parent_run_manifest.json").read_text(encoding="utf-8"))
    assert gen1_run["run_id"] != gen2_run["run_id"]        # 两次独立真实运行
    assert gen1_run["panel_hash"] == gen2_run["panel_hash"]  # 同一 learning 池


def test_resume_does_not_reconsume_panels_or_model_calls(tmp_path):
    """§11.1：中断恢复不得重复消费已完成的模型调用 / 面板 / 发布动作。"""
    lib = _seed_library(tmp_path)
    runtime = _FakeRuntime()
    offline = _FakeOffline()
    cfg = CampaignConfig(campaign_id="camp-test", max_generations=2, mode="real",
                         library_root=str(lib), run_root=str(tmp_path / "runs"),
                         learning_limit=8, inner_panel_limit=8,
                         min_eligible_experiences=3)
    first = EvolutionCampaignRunner(cfg, runtime=runtime, offline_client=offline,
                                    panels_provider=_panels(4, 4, 2),
                                    tool_names=["detect_objects"],
                                    print_fn=lambda *_: None)
    first.run()
    learning_calls = runtime.learning_calls
    inner_calls = runtime.inner_calls
    model_calls = offline.calls
    active = read_active_snapshot(lib / "snapshots")["snapshot_id"]

    second = EvolutionCampaignRunner(
        CampaignConfig(**{**cfg.__dict__, "resume": True}),
        runtime=runtime, offline_client=offline, panels_provider=_panels(4, 4, 2),
        tool_names=["detect_objects"], print_fn=lambda *_: None)
    campaign = second.run()
    assert runtime.learning_calls == learning_calls       # 面板未重跑
    assert runtime.inner_calls == inner_calls
    assert offline.calls == model_calls                   # 模型调用未重发
    assert read_active_snapshot(lib / "snapshots")["snapshot_id"] == active  # 未重复发布
    assert campaign.current_generation == 2


def test_experience_bundle_excludes_inner_split_and_unbound_traces():
    """§16.1：非 eligible 轨迹不能进入经验包；不同 Skill 版本不混桶。"""
    from skill3d.evolution.experience import EpisodeEvidence

    def ev(qa, split, delivered=True):
        record = {
            "retrieval_index": 1,
            "candidates": [{"skill_version": "S01@1.0.0", "hard_filter_passed": True,
                            "selected": True, "reason_code": "hit",
                            "delivered": delivered,
                            "delivery_reason": "delivered" if delivered else "not_selected"}],
            "delivered_content_sha256": ({"S01@1.0.0": "sha"} if delivered else {}),
            "usage_clues": ([{"skill_version": "S01@1.0.0", "declared_in_program": True}]
                            if delivered else []),
        }
        return EpisodeEvidence(episode_id=qa, scene_id=f"s-{qa}", split=split,
                               snapshot_id="S0-seed",
                               trace={"retrieval_records": [record]},
                               answer_correct=True, outcome_ref=f"evaluation_result:{qa}")

    events = build_experience_events(
        campaign_id="C", generation=1, parent_snapshot_id="S0-seed",
        skills=["S01@1.0.0"], episodes=[ev("a", "learning"), ev("b", "inner_validation")])
    by_id = {e.episode_id: e for e in events}
    assert by_id["a"].eligible_for_induction is True
    assert by_id["b"].eligible_for_induction is False
    assert "split_not_learning" in by_id["b"].exclusion_reasons

    bundle = build_experience_bundle(
        campaign_id="C", generation=1, parent_snapshot_id="S0-seed",
        parent_skill_key="S01@1.0.0", canonical_question_type="object_counting",
        events=events)
    assert bundle.n_eligible == 1
    assert bundle.exclusion_summary.get("split_not_learning") == 1
    assert all(ref == by_id["a"].experience_id for ref in bundle.eligible_experience_refs)

    with pytest.raises(Exception):
        build_experience_bundle(
            campaign_id="C", generation=1, parent_snapshot_id="S0-seed",
            parent_skill_key="S02@1.0.0", canonical_question_type="object_counting",
            events=events)


def test_seed_campaign_library_copies_snapshot_without_touching_source(tmp_path):
    """演化发生在 campaign 自己的库里；仓库 S0 库（--check 不变量）不被改动。"""
    src = _seed_library(tmp_path / "src")
    dst = tmp_path / "dst"
    result = seed_campaign_library(src, dst)
    assert (dst / "snapshots" / "snapshot_S0-seed.json").is_file()
    assert (dst / "snapshots" / "active_snapshot.json").is_file()
    assert "snapshots/snapshot_S0-seed.json" in result["copied"]
    before = (src / "snapshots" / "active_snapshot.json").read_text(encoding="utf-8")
    (dst / "snapshots" / "active_snapshot.json").write_text('{"snapshot_id": "changed"}',
                                                            encoding="utf-8")
    assert (src / "snapshots" / "active_snapshot.json").read_text(encoding="utf-8") == before


def test_mra_panel_reports_mra_mean_and_never_calls_it_zero_percent_correct():
    """MRA 题型（`object_counting`）没有对错布尔量：

    `answer_correct_rate` 的分母只能是**有判分布尔量**的题（`n_answer_graded`），
    MRA 另记 `mra_mean`。把"没测过对错"记成 0% 会把 MRA 题型说成全错 —— 这是
    本轮真实运行发现的口径缺陷（gen1 经验包 `answer_correct_rate=0.0` 而全部题
    的 `answer_correct` 都是 None）。
    """
    from skill3d.evolution.experience import EpisodeEvidence

    record = {
        "retrieval_index": 1,
        "candidates": [{"skill_version": "S01@1.0.0", "hard_filter_passed": True,
                        "selected": True, "reason_code": "hit", "delivered": True,
                        "delivery_reason": "delivered"}],
        "delivered_content_sha256": {"S01@1.0.0": "sha"},
        "usage_clues": [{"skill_version": "S01@1.0.0", "declared_in_program": True}],
    }
    evidence = [
        EpisodeEvidence(episode_id=f"m{i}", scene_id=f"ms{i}", split="learning",
                        snapshot_id="S0", trace={"retrieval_records": [record]},
                        answer_correct=None, mra_value=0.5 + 0.1 * i,
                        outcome_ref=f"evaluation_result:m{i}")
        for i in range(3)
    ] + [
        EpisodeEvidence(episode_id="c0", scene_id="cs0", split="learning",
                        snapshot_id="S0", trace={"retrieval_records": [record]},
                        answer_correct=True, outcome_ref="evaluation_result:c0"),
        EpisodeEvidence(episode_id="c1", scene_id="cs1", split="learning",
                        snapshot_id="S0", trace={"retrieval_records": [record]},
                        answer_correct=False, outcome_ref="evaluation_result:c1"),
    ]
    events = build_experience_events(campaign_id="C", generation=1,
                                     parent_snapshot_id="S0",
                                     skills=["S01@1.0.0"], episodes=evidence)
    bundle = build_experience_bundle(
        campaign_id="C", generation=1, parent_snapshot_id="S0",
        parent_skill_key="S01@1.0.0", canonical_question_type="object_counting",
        events=events,
        evidence_mra={ev.episode_id: ev.mra_value for ev in evidence
                      if ev.mra_value is not None})
    behavior = bundle.behavior_summary
    assert behavior["n_eligible"] == 5
    assert behavior["n_answer_graded"] == 2          # 只有两条有对错布尔量
    assert behavior["answer_correct_rate"] == 0.5    # 1/2，而不是 1/5 或 0.0
    assert behavior["n_mra_graded"] == 3
    assert abs(behavior["mra_mean"] - (0.5 + 0.6 + 0.7) / 3) < 1e-9


def test_behavior_summary_keys_are_the_declared_controlled_vocabulary():
    """摘要键必须在受控词表内（§6.3 `behavior_summary`）：不得随手加键绕过口径。"""
    from skill3d.schemas.experience import BEHAVIOR_SUMMARY_KEYS

    for key in ("n_answer_graded", "mra_mean", "n_mra_graded", "answer_correct_rate",
                "run_error_rate", "n_usage_supported"):
        assert key in BEHAVIOR_SUMMARY_KEYS


def test_static_validation_failure_retries_with_structured_feedback(tmp_path):
    """§7.4："Schema 不合法：生成结构化错误反馈，产生新 revision；超过最大 revision 数：reject"。

    归纳器第一次给出坏候选（引用未知工具）时必须**回灌问题清单再试**，而不是一次就停；
    尝试次数用尽才 blocked，且每次尝试的收据都在。
    """
    class _BadThenGoodOffline(_FakeOffline):
        def __init__(self):
            super().__init__()
            self.bad_calls = 0

        def chat(self, prompt: str) -> str:
            if "# 上一版候选未通过静态检查" in prompt:
                return super().chat(prompt)          # 收到反馈后给合法候选
            self.bad_calls += 1
            parent = json.loads(prompt.split("## 父 Skill 完整 SkillSpec", 1)[1]
                                .split("\n", 1)[1].split("\n## ", 1)[0])
            parent["version"] = "1.1.0"
            parent["call_graph_template"] = "n = magic_tool(img)\nReturnAnswer(str(n))"
            return json.dumps({"spec": parent, "hypothesis": "坏候选",
                               "expected_effect": "", "known_risks": [],
                              "diff_summary": "引用未知工具"}, ensure_ascii=False)

    offline = _BadThenGoodOffline()
    runner, runtime, _off, lib = _runner(tmp_path, offline=offline, generations=1)
    campaign = runner.run()
    gen1 = tmp_path / "runs" / "camp-test" / "gen1"
    assert (gen1 / "static_validation_attempt1.json").is_file()   # 失败尝试留档
    assert offline.bad_calls == 1                                  # 只坏了一次
    assert (gen1 / "candidate.json").is_file()                     # 反馈后拿到合法候选
    assert (gen1 / "static_validation.json").is_file()
    assert campaign.completion_status in ("running", "completed_two_generations",
                                          "completed_with_rejection")


def test_tool_reference_check_ignores_method_calls():
    """假阳性修复：`tracks.append(...)` 是属性访问，不是"未知工具"。

    本轮真实运行：候选正文里的 `append(` 被旧规则判成未知工具 → 整个候选被拒。
    """
    from skill3d.governance.induce import validate_candidate_against_parent
    from skill3d.schemas import SkillSpec

    parent_payload = {
        "skill_id": "S01", "version": "1.0.0",
        "applicable_question_types": ["object_counting"], "skill_family": "counting",
        "source": "real", "description": "数对象",
        "call_graph_template": "n = detect_objects(img, 'chair')",
    }
    parent = SkillSpec.model_validate(parent_payload)
    child = SkillSpec.model_validate({
        **parent_payload, "version": "1.1.0",
        "call_graph_template": ("tracks = []\ntracks.append(detect_objects(img, 'chair'))\n"
                                "counts = {}\ncounts.get('chair', 0)\n"
                                "ReturnAnswer(str(len(tracks)))")})
    receipt = validate_candidate_against_parent(
        child, parent,
        parent_content_sha256="h", claimed_parent_content_sha256="h",
        canonical_question_type="object_counting",
        tool_names=["detect_objects"])
    assert receipt.checks["tools_known"] is True, receipt.problems
    # 真正的裸调用仍然要被抓到
    bad = SkillSpec.model_validate({
        **parent_payload, "version": "1.1.0",
        "call_graph_template": "n = magic_tool(img)"})
    receipt2 = validate_candidate_against_parent(
        bad, parent, parent_content_sha256="h", claimed_parent_content_sha256="h",
        canonical_question_type="object_counting", tool_names=["detect_objects"])
    assert receipt2.checks["tools_known"] is False


def test_offline_failure_blocks_instead_of_crashing(tmp_path):
    """§3.4：离线模型不可用时 campaign 记 blocked + 结局码，**不**崩、**不**降级。

    本轮真实运行里一次 DeepSeek 超时（`OfflineServiceUnavailable`）曾把进程打崩 ——
    这条锁住"异常必须变成显式结局码 + 收据"。
    """
    from skill3d.governance.deepseek_client import OfflineServiceUnavailable

    class _TimeoutOffline:
        model_id = "mock-deepseek"

        def chat(self, prompt: str) -> str:
            raise OfflineServiceUnavailable("mock：重试 3 次后仍失败")

    runner, runtime, _off, lib = _runner(tmp_path, offline=_TimeoutOffline(),
                                         generations=1)
    campaign = runner.run()
    assert campaign.completion_status == "blocked"
    gen1 = tmp_path / "runs" / "camp-test" / "gen1"
    receipt = json.loads((gen1 / "offline_failure.json").read_text(encoding="utf-8"))
    assert receipt["failure_code"] == "service_unavailable"
    assert receipt["exception"] == "OfflineServiceUnavailable"
    assert not (gen1 / "candidate.json").is_file()      # 不创建伪候选


def test_label_access_true_only_for_eligible_learning_experiences():
    """§6.4："learning 的答对/答错和**误差**可由离线归纳器使用，但必须记录
    `label_access=true`"。

    - 合格经验（进了经验包）且有标签信号（布尔对错或 MRA 分值）→ `label_access=true`；
    - 被排除的经验不进经验包 → 归纳器没看到它的标签 → `false`；
    - 没有标签信号（既无对错也无 MRA）→ `false`。
    """
    from skill3d.evolution.experience import EpisodeEvidence

    record = {
        "retrieval_index": 1,
        "candidates": [{"skill_version": "S01@1.0.0", "hard_filter_passed": True,
                        "selected": True, "reason_code": "hit", "delivered": True,
                        "delivery_reason": "delivered"}],
        "delivered_content_sha256": {"S01@1.0.0": "s"},
        "usage_clues": [{"skill_version": "S01@1.0.0", "declared_in_program": True}],
    }
    undelivered = {
        "retrieval_index": 1,
        "candidates": [{"skill_version": "S01@1.0.0", "hard_filter_passed": True,
                        "selected": True, "reason_code": "hit"}],
        "delivered_content_sha256": {}, "usage_clues": [],
    }
    episodes = [
        EpisodeEvidence(episode_id="elig", scene_id="s1", split="learning",
                        snapshot_id="S0", trace={"retrieval_records": [record]},
                        answer_correct=None, mra_value=0.5,
                        outcome_ref="evaluation_result:elig"),
        EpisodeEvidence(episode_id="excluded", scene_id="s2", split="learning",
                        snapshot_id="S0", trace={"retrieval_records": [undelivered]},
                        answer_correct=None, mra_value=0.5,
                        outcome_ref="evaluation_result:excluded"),
        EpisodeEvidence(episode_id="unlabeled", scene_id="s3", split="learning",
                        snapshot_id="S0", trace={"retrieval_records": [record]},
                        answer_correct=None, mra_value=None,
                        outcome_ref="evaluation_result:unlabeled"),
    ]
    events = {e.episode_id: e for e in build_experience_events(
        campaign_id="C", generation=1, parent_snapshot_id="S0",
        skills=["S01@1.0.0"], episodes=episodes)}
    assert events["elig"].eligible_for_induction and events["elig"].label_access
    assert not events["excluded"].eligible_for_induction
    assert events["excluded"].label_access is False
    # 既无对错又无 MRA = 结果身份不完整（§6.2-5）→ 本身就不合格
    assert events["unlabeled"].eligible_for_induction is False
    assert "result_identity_missing" in events["unlabeled"].exclusion_reasons
    assert events["unlabeled"].label_access is False    # 没有标签信号可访问
