"""Selection depends only on slot and prepared size, never model outcomes."""
import importlib.util
from pathlib import Path


def test_representative_cohort_uses_per_slot_median_and_stable_tiebreak():
    path = Path(__file__).resolve().parents[2] / "scripts/run_reviser_successor_v11.py"
    spec = importlib.util.spec_from_file_location("cohort_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {"slot_key": slot, "campaign_id": campaign,
         "editor_preflight": {"budget": {"batch_count": batches}}}
        for slot, campaign, batches in [
            ("a/g0m0", "c", 30), ("a/g0m0", "b", 5), ("a/g0m0", "a", 5),
            ("b/g0m0", "e", 8), ("b/g0m0", "d", 4)]]
    assert [r["campaign_id"] for r in module.representative_selection(rows)] == ["d", "b"]
