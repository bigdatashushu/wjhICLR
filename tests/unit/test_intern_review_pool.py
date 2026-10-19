from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

from skill3d.segmentation.intern_decision_client import make_review_client, review_config
from skill3d.segmentation.intern_review_pool import InternReviewPool
from skill3d.synthesis.request_context import current_request_phase, request_phase


def clients(n=3):
    result = []
    for i in range(n):
        identity = {"model": "same-model", "runtime": {"torch": "same-version"},
                    "calibration": {"temperature": 2}, "device": {"visible_devices": str(i)}}
        result.append(SimpleNamespace(
            endpoint=f"http://127.0.0.1:{18445 + i}", provider="test", target_binary=True,
            identity=lambda identity=identity: identity,
            audit_identity=lambda **_: {"protocol": "same", "cache_eligible": False},
        ))
    return result


def test_ordered_pool_overlaps_replicas_without_overlapping_each_endpoint():
    members = clients()
    barrier = Barrier(3, timeout=3)
    lock = Lock()
    active = set()
    routes = []
    for index, member in enumerate(members):
        def observe(candidate, frame, mask, index=index, **kwargs):
            with lock:
                assert index not in active
                active.add(index)
            barrier.wait()
            assert current_request_phase() == "test-phase"
            with lock:
                routes.append((candidate["candidate_id"], index))
                active.remove(index)
            return {"candidate_id": candidate["candidate_id"], "status": "observed"}
        member.observe_candidate = observe
    pool = InternReviewPool(members)
    observations = [{"candidate": {"candidate_id": str(i), "frame_slot": 0},
                     "raw_mask": None} for i in range(6)]
    with request_phase("test-phase"):
        results = list(pool.observe_candidates(observations, [None]))
    assert [row["candidate_id"] for row in results] == [str(i) for i in range(6)]
    assert sorted(routes) == [(str(i), i % 3) for i in range(6)]
    assert [row["review_route"]["replica_index"] for row in results] == [0, 1, 2, 0, 1, 2]


@pytest.mark.parametrize("field", ["runtime", "calibration", "weights", "service_sha256"])
def test_pool_rejects_method_identity_difference_before_any_request(field):
    members = clients(2)
    members[1].identity()[field] = "different"
    for member in members:
        member.observe_candidate = lambda *_, **__: pytest.fail("must fail before inference")
    with pytest.raises(ValueError, match="identities differ"):
        InternReviewPool(members).observe_candidate({}, None, None)


def test_unavailable_result_keeps_its_position_and_no_retry():
    members = clients(2)
    counts = [0, 0]
    for index, member in enumerate(members):
        def observe(candidate, frame, mask, index=index, **kwargs):
            counts[index] += 1
            return {"candidate_id": candidate["candidate_id"],
                    "status": "unavailable" if index == 0 else "observed"}
        member.observe_candidate = observe
    rows = [{"candidate": {"candidate_id": str(i), "frame_slot": 0}, "raw_mask": None}
            for i in range(3)]
    results = list(InternReviewPool(members).observe_candidates(rows, [None]))
    assert [r["status"] for r in results] == ["unavailable", "observed", "unavailable"]
    assert counts == [2, 1]


def test_pool_factory_requires_distinct_pinned_replicas():
    config = {"provider": "intern-decision", "mode": "binary", "prompt_version": "P1",
              "yes_threshold": .77, "admission_policy": "category-only",
              "endpoints": [{"endpoint": f"http://127.0.0.1:{18445 + i}",
                             "expected_identity_sha256": str(i) * 64} for i in range(3)]}
    pool = make_review_client(config)
    assert isinstance(pool, InternReviewPool) and len(pool.clients) == 3 and pool.target_binary
    for bad in [
        {**config, "endpoint": "http://127.0.0.1:18445"},
        {**config, "endpoints": []},
        {**config, "endpoints": [config["endpoints"][0]] * 2},
        {**config, "endpoints": [{"endpoint": "http://127.0.0.1:18445",
                                  "expected_identity_sha256": ""}]},
    ]:
        with pytest.raises(ValueError):
            review_config(bad)


def test_yaml_pool_replaces_singular_default_instead_of_merging_it(tmp_path):
    import yaml
    from skill3d.online.config import load_config
    review = {"provider": "intern-decision", "mode": "binary", "prompt_version": "P1",
              "yes_threshold": .77, "admission_policy": "category-only",
              "endpoints": [{"endpoint": "http://127.0.0.1:18446",
                             "expected_identity_sha256": "a" * 64}]}
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump({"online_model": {"mask_review": review}}))
    assert load_config(path)["online_model"]["mask_review"] == review
