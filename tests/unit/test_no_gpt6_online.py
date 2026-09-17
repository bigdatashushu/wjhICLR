"""硬约束 1 静态守卫：在线链模块源码中不得 import governance / gpt6（大小写不敏感）。

扫描目录：adapters/gates/reconstruction/reconstruction_gate/segmentation/tools/
routing/synthesis/sandbox/verifier/evaluation/trace/online + fsm/online_fsm.py。
仅检查 import 语句行（注释中提及 GPT-6 的设计说明不算违规）。
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "skill3d"

ONLINE_DIRS = [
    "adapters", "gates", "reconstruction", "reconstruction_gate", "segmentation",
    "tools", "routing", "synthesis", "sandbox", "verifier", "evaluation", "trace",
    "online",  # 在线链编排层（M1-M13 driver，§6.1）
]
ONLINE_FILES = ["fsm/online_fsm.py"]

_IMPORT_LINE = re.compile(r"^\s*(import|from)\s+\S+", re.IGNORECASE)
_FORBIDDEN = ("governance", "gpt6", "gpt-6", "gpt_6")


def _violations(path: Path) -> list[str]:
    hits = []
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        if _IMPORT_LINE.match(line):
            low = line.lower()
            if any(tok in low for tok in _FORBIDDEN):
                hits.append(f"{path}:{lineno}: {line.strip()}")
    return hits


def test_online_modules_never_import_gpt6():
    all_hits: list[str] = []
    for d in ONLINE_DIRS:
        dpath = SRC / d
        if not dpath.is_dir():
            continue
        for py in dpath.rglob("*.py"):
            all_hits.extend(_violations(py))
    for f in ONLINE_FILES:
        fpath = SRC / f
        if fpath.is_file():
            all_hits.extend(_violations(fpath))
    assert not all_hits, "在线链出现 governance/gpt6 import（违反硬约束 1）:\n" + \
        "\n".join(all_hits)
