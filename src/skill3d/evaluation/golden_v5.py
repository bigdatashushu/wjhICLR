"""`evaluation/golden_v5`：v5 golden 的版本注册与加载门（HC39 / §7）。

**为什么必须版本化**（v5 HC39）：

- 旧 golden 的 `overall_quality` 分母含已退役的 G8、且 G5 是必填 NaN 占位，
  与 v5 活动指标集合（G5 条件项、G8 不存在）**不可比**；
- 统计数值随 `numpy/scipy` 版本变化（E-1），环境不符即不可比；
- 因此：当前统计只接受 `schema_version="5.0"` + `quality_metric_version=
  "v5-no-g8-g5-optional"` + 同 `env_versions` 的 golden；其余一律 hard fail，
  **不得**自动重算、也不得静默比较（HC39）。

旧 golden 只读归档在 `tests/golden/archive_v4/`，其 manifest 标
`incomparable_with_v5=true`。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Union

GOLDEN_VERSION = "v5-golden-1"
GOLDEN_SCHEMA_VERSION = "5.0"
GOLDEN_QUALITY_METRIC_VERSION = "v5-no-g8-g5-optional"


class GoldenVersionError(RuntimeError):
    """golden 版本/环境不符 → hard fail（不自动重算、不静默比较）。"""


def golden_dir(root: Union[str, Path]) -> Path:
    return Path(root)


def _env_versions() -> dict:
    out: dict[str, str] = {}
    for mod in ("scipy", "numpy"):
        try:
            import importlib

            out[mod] = str(getattr(importlib.import_module(mod), "__version__", "?"))
        except Exception:  # noqa: BLE001
            out[mod] = "absent"
    return out


def load_golden_stats(path: Union[str, Path], *,
                      strict_env: bool = True) -> dict:
    """加载 v5 golden 统计基线；版本或环境不符即 hard fail。

    `strict_env=True`（默认）时要求 `env_versions` 与当前环境逐项一致（E-1）：
    `numpy/scipy` 变了，bootstrap/Wilcoxon 的数值就可能变，比较结论不再成立。
    """
    p = Path(path)
    if not p.is_file():
        raise GoldenVersionError(f"golden 统计文件不存在: {p}（需由 v5 pipeline 生成）")
    data = json.loads(p.read_text(encoding="utf-8"))
    ver = data.get("golden_version")
    if ver != GOLDEN_VERSION:
        raise GoldenVersionError(
            f"golden_version={ver!r}（需要 {GOLDEN_VERSION!r}）。旧 golden 与 v5 不可比"
            "（HC39）：请用 tests/golden/v5/make_golden.py 重新生成，"
            "旧数据只读归档在 tests/golden/archive_v4/")
    if data.get("schema_version") != GOLDEN_SCHEMA_VERSION:
        raise GoldenVersionError(
            f"golden schema_version={data.get('schema_version')!r} "
            f"（需要 {GOLDEN_SCHEMA_VERSION!r}）")
    if data.get("quality_metric_version") != GOLDEN_QUALITY_METRIC_VERSION:
        raise GoldenVersionError(
            "golden quality_metric_version="
            f"{data.get('quality_metric_version')!r}"
            f"（需要 {GOLDEN_QUALITY_METRIC_VERSION!r}）")
    if strict_env:
        want = data.get("env_versions") or {}
        have = _env_versions()
        mismatch = {k: (want.get(k), have.get(k))
                    for k in set(want) | set(have) if want.get(k) != have.get(k)}
        if mismatch:
            raise GoldenVersionError(
                f"golden env_versions 与当前环境不符 {mismatch}（E-1）："
                "golden 数值随统计库版本变化，环境不符即不可比")
    return data


def write_golden_manifest(path: Union[str, Path], *, artifact_version: str,
                          artifact_content_sha256: str) -> Path:
    """写 golden 版本清单（谁生成的、哪个版本、哪个 commit）。"""
    from skill3d.infra.version_lock import git_head

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "golden_version": GOLDEN_VERSION,
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "quality_metric_version": GOLDEN_QUALITY_METRIC_VERSION,
        "artifact_version": artifact_version,
        "artifact_content_sha256": artifact_content_sha256,
        "env_versions": _env_versions(),
        "pip_freeze_hash": _pip_hash(),
        "code_commit": _safe_commit(),
        "generated_by": "tests/golden/v5/make_golden.py",
        "incomparable_with_v5": False,
        "notes": ("v5 golden 由 v5 Schema/pipeline 重新生成：G5 为条件项（本夹具无真 BA → "
                  "None，不入 overall_quality 分母）、不含 G8 字段。"),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _pip_hash() -> str:
    try:
        from skill3d.infra.version_lock import pip_freeze_hash

        return pip_freeze_hash()
    except Exception:  # noqa: BLE001
        return ""


def _safe_commit() -> str:
    try:
        from skill3d.infra.version_lock import git_head

        return git_head(".")
    except Exception:  # noqa: BLE001
        return ""


def golden_payload_sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def legacy_archive_manifest(root: Union[str, Path]) -> Optional[dict]:
    """读取旧 golden 归档清单（用于断言"只读且不可比"）。"""
    p = Path(root) / "ARCHIVE.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def assert_not_legacy_golden(path: Union[str, Path]) -> None:
    """混用旧 golden 的调用点护栏：hard fail 而不是自动比较。"""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("incomparable_with_v5") is True or "golden_version" not in data:
        raise GoldenVersionError(
            f"{p} 是 v5 之前的 golden（incomparable_with_v5=true 或缺 golden_version）。"
            "HC39：旧 golden 不得与 v5 统计直接比较，也不得自动重算；"
            "请重跑 v5 pipeline 生成新 golden。")


def golden_versions() -> dict[str, Any]:
    """当前 golden 版本三元组（进 RunManifest / 论文复现清单）。"""
    return {"golden_version": GOLDEN_VERSION,
            "schema_version": GOLDEN_SCHEMA_VERSION,
            "quality_metric_version": GOLDEN_QUALITY_METRIC_VERSION}
