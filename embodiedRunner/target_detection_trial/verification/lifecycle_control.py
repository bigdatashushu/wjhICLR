"""Versioned deployment/status helper; frozen evaluated modules remain byte-identical."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from target_detection_trial import lifecycle

# Match actual argv entries. A read-only `python -c` script can contain the worker
# path in its source text without being a simulator worker.
lifecycle.REMOTE_STATE = lifecycle.REMOTE_STATE.replace(
    'if b"target_detection_trial/worker.py" not in raw:',
    'if not any(a.endswith(b"/target_detection_trial/worker.py") for a in raw.split(b"\\0")):')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("deploy", "probe", "after"))
    phase = parser.parse_args().phase
    getattr(lifecycle, phase)()
