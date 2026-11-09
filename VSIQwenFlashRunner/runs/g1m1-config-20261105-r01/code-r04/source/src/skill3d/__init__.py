"""VSI1000 modules first, with an explicit read-only shared package fallback."""

import json
from pathlib import Path

_package = Path(__file__).resolve().parent
_runner = _package.parents[1]
_layout = _runner / "source_layout.json"
__path__ = [str(_package)]
if _layout.is_file():
    _value = json.loads(_layout.read_text())
    if _value != {"schema": "vsi-qwen-flash-source-layout/1", "mode": "frozen-merged"}:
        raise ImportError("Invalid VSI runner source layout")
else:
    _shared = _runner.parent / "src/skill3d"
    if not _shared.is_dir():
        raise ImportError("VSI runner requires the sibling shared src/skill3d package")
    __path__.append(str(_shared))
