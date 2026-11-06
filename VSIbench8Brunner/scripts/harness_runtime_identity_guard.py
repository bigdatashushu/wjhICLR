#!/usr/bin/env python3
"""Stop the owned scheduler if a native solver response changes model identity."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source_frozen" / "scripts"))
sys.path.insert(0, str(ROOT / "source_frozen" / "src"))

import pilot40 as pilot
import pilot40_parallel as scheduler


def install():
    original = pilot.RecordingClient.record

    def record(self, request, timeout):
        result = original(self, request, timeout)
        body = json.loads(result.getvalue())
        if body.get("model") != self.model:
            receipt = self.requests[-1]
            receipt.update(status="unavailable", error_type="ResponseModelIdentityChanged")
            pilot.write(self.root / f"{receipt['request_index']:03d}.json", receipt)
            pilot.export_question(self.run_root, self.qa_id, status="failed")
            raise RuntimeError("native solver response model identity changed")
        return result

    pilot.RecordingClient.record = record


if __name__ == "__main__":
    install()
    scheduler.main()
