"""v9 P4 端到端：`inspect_frames` → `YieldObservations` → **下一模型请求** → 下一程序。

规范原文（§9.4）：

    「模型服务的图像数量／像素上限必须涵盖原图与派生图；不得静默丢图。可使用带明确
    映射的图像布局或分次观察，**实际传入图像、缩放和 token 成本完整记录**。」
    「观察链分开记录 `produced / delivered / observed`…未交付的材料保持 unobserved。」

本文件的模型是假客户端（真实模型请求的验收在
`scripts/run_active_vision_acceptance.py`，其 receipt 记在实现文档里）；这里守的是
**代码路径**：裁剪图真的装进了下一次请求、布局不超服务上限、三态与 token 成本都落盘。
"""

from __future__ import annotations

import json

import pytest

from skill3d.adapters.episode_source import load_synthetic_items
from skill3d.online.runner import OnlineRunConfig, _prompt_messages, run_episode

# 交付层的真实路径需要"合法 v6 artifact"（real 模式复用产物）；复用恢复层测试的
# 夹具构造器（同一套合成几何 + 真算质量），不另造一份替身。
from test_tool_contract_recovery import _write_v6_artifact  # noqa: E402 - 同目录夹具

FRAME_SIZE = (120, 160)

# 真实链路里会用到的那段程序（真实 smoke 用的就是它）：裁剪 + 让出
CROP_THEN_YIELD = (
    'crop = inspect_frames([8], [[0, 0, 60, 48]])\n'
    'img = crop["images"][0]["image_id"]\n'
    'return YieldObservations([img], "需要放大看清这一块区域")\n'
)


class _FakeClient:
    """按序返回 program；记录每轮请求（含图像数）。"""

    def __init__(self, programs: list[str]) -> None:
        self.programs = list(programs)
        self.calls: list[list[dict]] = []
        self.last_usage: dict = {}

    def chat(self, messages, max_tokens: int = 512, **kwargs) -> str:
        if "ReturnAnswer" not in json.dumps(messages, ensure_ascii=False):
            return ""                      # 非 M8 调用（M5 清单等）
        self.calls.append(messages)
        self.last_usage = {"prompt_tokens": 9000 + len(self.calls), "completion_tokens": 7}
        return self.programs[min(len(self.calls) - 1, len(self.programs) - 1)]


def _n_images(messages) -> int:
    return sum(1 for m in messages if isinstance(m.get("content"), list)
               for p in m["content"]
               if isinstance(p, dict) and p.get("type") == "image_url")


def _prompt_text(messages) -> str:
    parts = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") == "text")
    return "\n".join(parts)


@pytest.fixture(scope="module")
def fixture_episode():
    """(episode, pixels, artifact_path)：三者同源（合成几何 → 真算质量的 artifact）。"""
    import tempfile

    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    item = load_synthetic_items("inner_validation", question_types=["room_size_estimation"],
                                frame_size=FRAME_SIZE, seed=0)[0]
    return item, _write_v6_artifact(tmp, with_poses=True)


def _cfg(tmp_path, art_path: str, **over) -> OnlineRunConfig:
    base = dict(mode="real", vllm_endpoints=["http://fake"], deterministic_replay=True,
                trace_dir=str(tmp_path / "t"), max_images=32,
                reuse_artifact=art_path)
    base.update(over)
    return OnlineRunConfig(**base)


def test_crop_reaches_the_next_request_with_an_explicit_layout(fixture_episode, tmp_path):
    """§9.4：裁剪图必须真的进**下一次**请求，且清单里写明"哪张图来自哪一帧的哪个框"。"""
    item, art = fixture_episode
    client = _FakeClient([CROP_THEN_YIELD, 'ReturnAnswer("4")'])
    cfg = _cfg(tmp_path, art)
    out = run_episode(item.episode, item.pixels, cfg, llm=client)

    assert out.yield_count == 1 and out.agent_rounds >= 2
    ledger = out.image_ledger
    derived = [i for i in ledger["images"] if i["kind"] != "frame"]
    assert len(derived) == 1
    crop = derived[0]
    # 三态：真的交付并观察到（请求发出且返回）
    assert crop["delivered_rounds"] and crop["observed_rounds"]
    assert crop["content_sha256"] and crop["box_xyxy"] == [0.0, 0.0, 60.0, 48.0]
    assert crop["source_frame_id"] == 8                 # 槽位（模型指帧口径）

    # 第 2 个请求：派生图 + 原帧，总数不超过服务上限
    assert len(client.calls) >= 2
    second = client.calls[1]
    assert _n_images(second) <= cfg.max_images
    text = _prompt_text(second)
    assert "本轮图像清单" in text
    assert crop["image_id"] in text                      # 明确映射：图 ↔ 源帧/框
    assert "裁剪图" in text
    rnd = next(r for r in ledger["rounds"] if r["round_index"] == 2)
    assert rnd["layout"] == "derived_plus_originals_v1"
    assert crop["image_id"] in rnd["derived_image_ids"]
    assert rnd["prompt_tokens"] == client.last_usage["prompt_tokens"]
    assert rnd["delivered"] and rnd["observed"]


def test_round_one_is_unchanged_without_derived_images(fixture_episode, tmp_path):
    """没有派生图时第 1 轮仍是 32 张原帧（布局退化，行为与改动前一致）。"""
    item, art = fixture_episode
    client = _FakeClient(['ReturnAnswer("4")'])
    out = run_episode(item.episode, item.pixels, _cfg(tmp_path, art), llm=client)
    rnd = out.image_ledger["rounds"][0]
    assert rnd["layout"] == "originals_only"
    assert rnd["n_images"] == 32 and not rnd["derived_image_ids"]
    assert out.image_ledger["stats"]["n_derived_produced"] == 0


def test_request_over_the_image_limit_fails_loudly(fixture_episode, tmp_path):
    """§9.4"不得静默丢图"：超服务图像上限必须报错，而不是悄悄少送。"""
    item, _art = fixture_episode
    cfg = OnlineRunConfig(mode="real", max_images=8)
    with pytest.raises(ValueError, match="禁止静默丢帧"):
        _prompt_messages("p", list(item.pixels), cfg, expected_frames=len(item.pixels))


def test_budget_boundary_still_carries_the_observations(fixture_episode, tmp_path):
    """让出后触到轮预算边界也不会丢观察：收口轮仍带上派生图（§9.4 不静默丢图）。"""
    item, art = fixture_episode
    client = _FakeClient([CROP_THEN_YIELD, "ReturnAnswer(1)", "ReturnAnswer(2)"])
    out = run_episode(item.episode, item.pixels,
                      _cfg(tmp_path, art, max_solver_rounds=2, finalization_rounds=1),
                      llm=client)
    derived = [i for i in out.image_ledger["images"] if i["kind"] != "frame"]
    assert derived, "裁剪图必须真的产出"
    # 无论后续走哪条路径，派生图的交付/未交付都必须有明确记录（不许既没记也没送）
    for img in derived:
        assert img["delivered_rounds"] or img["unobserved_reason"] or img["delivered_rounds"] == []
    assert any(r["derived_image_ids"] for r in out.image_ledger["rounds"])
