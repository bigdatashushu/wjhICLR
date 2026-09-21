"""G-26 在线 episodic 记忆单测（§4 M14、§7.1 G-26、硬约束 9/19）。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from skill3d.memory.online_memory import (
    EpisodicMemory,
    MemoryWriteForbiddenError,
    recency_score,
    relevance_score,
    three_factor_score,
)
from skill3d.schemas import MemoryEntry

NOW = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


def _entry(content: str, *, strength: float = 1.0, age_days: float = 0.0,
           mid: str = "m1") -> MemoryEntry:
    return MemoryEntry(
        memory_id=mid, layer="episodic", content=content,
        provenance=["episode_trace:qa-1"], evidence_strength=strength,
        contradiction_group_id=None,
        created_at=(NOW - timedelta(days=age_days)).isoformat(),
    )


# ------------------------------------------------------------------ 三因子 ----

def test_recency_decays_exponentially_with_halflife():
    assert recency_score(NOW.isoformat(), now=NOW) == pytest.approx(1.0)
    half = recency_score((NOW - timedelta(days=7)).isoformat(), now=NOW)
    assert half == pytest.approx(0.5, abs=0.01)
    older = recency_score((NOW - timedelta(days=30)).isoformat(), now=NOW)
    assert older < half < 1.0


def test_recency_tolerates_bad_timestamp():
    assert recency_score("not-a-date", now=NOW) == 1.0


def test_relevance_is_deterministic_and_bounded():
    a = relevance_score("object counting", "task=object_counting final_state=answer")
    b = relevance_score("object counting", "task=object_counting final_state=answer")
    assert a == b and 0.0 <= a <= 1.0
    assert relevance_score("object counting", "unrelated text about chairs") < a
    assert relevance_score("", "x") == 0.0


def test_three_factor_prefers_recent_relevant_strong():
    query = "task=object_counting"
    good = _entry("[scene=s1] task=object_counting verdict=answer_correct",
                  strength=1.0, age_days=0, mid="good")
    weak = _entry("[scene=s1] task=object_counting verdict=unanswerable",
                  strength=0.2, age_days=20, mid="weak")
    assert three_factor_score(good, query, now=NOW) > three_factor_score(weak, query, now=NOW)


# ------------------------------------------------------------------ 写入门禁 ----

def test_final_test_write_is_forbidden(tmp_path):
    mem = EpisodicMemory(path=str(tmp_path / "m.jsonl"))
    with pytest.raises(MemoryWriteForbiddenError, match="硬约束 9"):
        mem.record(scene_name="s", content="x", provenance=["t"], split="final_test")
    # record_episode 对 final_test 静默跳过（调用方无需分支）
    assert mem.record_episode(qa_id="q", scene_name="s", task="object_counting",
                              final_state="answer", correct=True,
                              split="final_test") is None
    assert mem.entries == []


def test_provenance_required(tmp_path):
    mem = EpisodicMemory(path=str(tmp_path / "m.jsonl"))
    with pytest.raises(MemoryWriteForbiddenError, match="provenance"):
        mem.record(scene_name="s", content="x", provenance=[])


def test_leakage_blocked_on_write(tmp_path):
    mem = EpisodicMemory(path=str(tmp_path / "m.jsonl"))
    with pytest.raises(MemoryWriteForbiddenError, match="LEAKAGE"):
        mem.record(scene_name="s", content="ground truth 是 3",
                   provenance=["t"])
    with pytest.raises(MemoryWriteForbiddenError, match="LEAKAGE"):
        mem.record(scene_name="s", content="样本 qa_0a1b2c3d 记录",
                   provenance=["t"])


def test_forbidden_sample_ids_are_scanned(tmp_path):
    mem = EpisodicMemory(path=str(tmp_path / "m.jsonl"))
    with pytest.raises(MemoryWriteForbiddenError):
        mem.record(scene_name="s", content="scene 41069025 的统计",
                   provenance=["t"], forbidden_sample_ids={"41069025"})
    # 短纯数字 id 不逐字扫描（避免误报），但样式正则仍兜底
    mem.record(scene_name="s", content="task=object_counting 计数为 3",
               provenance=["t"], forbidden_sample_ids={"3"})


# ------------------------------------------------------------------ 持久化 ----

def test_roundtrip_jsonl_and_scene_aggregation(tmp_path):
    path = tmp_path / "m.jsonl"
    mem = EpisodicMemory(path=str(path))
    mem.record_episode(qa_id="q1", scene_name="s1", task="object_counting",
                       final_state="answer", correct=True, route="full_3d")
    mem.record_episode(qa_id="q2", scene_name="s2", task="room_size_estimation",
                       final_state="unanswerable", correct=False, route="fallback_2d_only",
                       flags=["g11_measurement_2d_only"])
    assert path.is_file()

    back = EpisodicMemory(path=str(path))
    assert len(back.entries) == 2
    assert len(back.list_scene("s1")) == 1
    assert "room_size_estimation" in back.list_scene("s2")[0].content
    assert back.list_scene("nope") == []


def test_evidence_strength_by_outcome(tmp_path):
    mem = EpisodicMemory(path=str(tmp_path / "m.jsonl"))
    e_ok = mem.record_episode(qa_id="a", scene_name="s", task="object_counting",
                              final_state="answer", correct=True)
    e_bad = mem.record_episode(qa_id="b", scene_name="s", task="object_counting",
                               final_state="answer", correct=False)
    e_best = mem.record_episode(qa_id="c", scene_name="s", task="object_counting",
                                final_state="answer_best_effort", correct=None)
    e_none = mem.record_episode(qa_id="d", scene_name="s", task="object_counting",
                                final_state="unanswerable", correct=None)
    assert e_ok.evidence_strength > e_bad.evidence_strength > e_best.evidence_strength \
        > e_none.evidence_strength


def test_content_has_no_question_or_answer_text(tmp_path):
    """episodic 内容只含聚合摘要（题型/终态/路由），不含问题文本与答案（硬约束 19）。"""
    mem = EpisodicMemory(path=str(tmp_path / "m.jsonl"))
    e = mem.record_episode(qa_id="q1", scene_name="kitchen", task="object_counting",
                           final_state="answer", correct=True, route="full_3d")
    assert "kitchen" in e.content           # scene 用于聚合
    assert "task=object_counting" in e.content
    for banned in ("ground_truth", "How many", "answer:", "选项"):
        assert banned not in e.content


# ------------------------------------------------------------------ 检索 ----

def test_retrieve_top_k_ordering_and_scene_filter(tmp_path):
    mem = EpisodicMemory(path=str(tmp_path / "m.jsonl"))
    for i in range(6):
        mem.record(scene_name="s1", content=f"task=object_counting note{i}",
                   provenance=[f"t{i}"], evidence_strength=1.0 - i * 0.1)
    mem.record(scene_name="s2", content="task=room_size_estimation", provenance=["t9"])

    top = mem.retrieve("object_counting", top_k=3)
    assert len(top) == 3
    assert all("object_counting" in e.content for _s, e in top)
    assert top[0][0] >= top[-1][0]                    # 降序
    only_s2 = mem.retrieve("object_counting", scene_name="s2", top_k=5)
    assert len(only_s2) == 1 and "s2" in only_s2[0][1].content


def test_stats_reports_scenes_and_strength(tmp_path):
    mem = EpisodicMemory(path=str(tmp_path / "m.jsonl"))
    mem.record(scene_name="s1", content="a", provenance=["t"], evidence_strength=1.0)
    mem.record(scene_name="s2", content="b", provenance=["t"], evidence_strength=0.0)
    st = mem.stats()
    assert st["n_entries"] == 2 and st["n_scenes"] == 2
    assert st["mean_evidence_strength"] == pytest.approx(0.5)


# ------------------------------------------------------------------ 在线链集成 ----

def test_online_chain_writes_episodic_and_skips_final_test(tmp_path):
    """端到端（mock_light）：run_split 写 episodic；final_test 不写（硬约束 9）。"""
    from dataclasses import replace

    from skill3d.adapters.episode_source import load_synthetic_items
    from skill3d.online.runner import OnlineRunConfig, run_split

    items = load_synthetic_items("inner_validation", question_types=["object_counting"],
                                 seed=0, out_dir=str(tmp_path / "recons"),
                                 n_frames=6, frame_size=(64, 96))
    cfg = OnlineRunConfig(mode="mock_light", seed=0, trace_dir=str(tmp_path / "traces"),
                          recon_dir=str(tmp_path / "recons"),
                          memory_dir=str(tmp_path / "memory_episodic"))
    outcomes, _run = run_split(items, cfg)
    assert len(outcomes) == 1
    mem = EpisodicMemory(path=f"{cfg.memory_dir}/episodic_seed0_C1_tools_program.jsonl")
    assert len(mem.entries) == 1
    assert mem.entries[0].layer == "episodic"                 # 在线不写 semantic
    assert mem.entries[0].provenance == [f"episode_trace:{outcomes[0].qa_id}"]
    assert any("M14 episodic 记忆已写" in n for n in outcomes[0].notes)

    # final_test：不写 Memory（硬约束 9）
    ft_items = load_synthetic_items("final_test", question_types=["object_counting"],
                                   seed=0, out_dir=str(tmp_path / "recons"),
                                   n_frames=6, frame_size=(64, 96))
    ft_cfg = replace(cfg, allow_final_test=True)
    ft_outcomes, _ = run_split(ft_items, ft_cfg)
    ft_mem = EpisodicMemory(path=f"{ft_cfg.memory_dir}/episodic_seed0_C1_tools_program.jsonl")
    assert len(ft_mem.entries) == 1                           # 仍只有前面那条
    assert not (ft_mem.entries[0].content.startswith("[scene=synthetic-final_test"))
    del ft_outcomes


def test_memory_write_failure_does_not_break_online_chain(tmp_path):
    """记忆写入异常只记 note，不阻断在线链（记忆是增强项）。"""
    from skill3d.adapters.episode_source import load_synthetic_items
    from skill3d.online.runner import OnlineRunConfig, run_split

    items = load_synthetic_items("inner_validation", question_types=["object_counting"],
                                 seed=0, out_dir=str(tmp_path / "recons"),
                                 n_frames=6, frame_size=(64, 96))
    # 用一个必失败路径（父级是文件而非目录）触发写入异常
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    cfg = OnlineRunConfig(mode="mock_light", seed=0, trace_dir=str(tmp_path / "traces"),
                          recon_dir=str(tmp_path / "recons"),
                          memory_dir=str(blocker / "sub"))
    outcomes, _ = run_split(items, cfg)
    assert outcomes[0].final_state in ("answer", "answer_best_effort", "unanswerable")
    assert any("episodic 记忆写入失败" in n or "M14" in n for n in outcomes[0].notes)
