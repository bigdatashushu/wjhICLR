import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def test_nonfinite_diagnostics_remain_explicit_in_strict_json():
    path = Path(__file__).resolve().parents[1] / "scripts/pilot40.py"
    spec = importlib.util.spec_from_file_location("pilot_json_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    value = module.json_value({
        "quality": float("nan"),
        "points": np.array([1., float("inf"), float("-inf")]),
    })
    json.dumps(value, allow_nan=False)
    assert value["quality"] == {"$nonfinite": "nan"}
    assert value["points"]["data"] == [
        1., {"$nonfinite": "positive_infinity"}, {"$nonfinite": "negative_infinity"}]
