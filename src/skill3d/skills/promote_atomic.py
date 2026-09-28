"""M15 / v10 §10.1：promote = **原子发布完整候选快照**（硬约束 12 + §5.3 谱系竞争）。

规范原文（§10.1）：

    「promotion 不再直接向当前 JSON 的 `entries[revision_id]` 追加。发布过程为：

    1. 读取并锁定父快照；
    2. 根据候选 operation 构建完整候选快照；
    3. 重算 entries、竞争集合、`skill_versions`、generation、manifest hash；
    4. 校验所有 SkillSpec 和谱系关系；
    5. 写入不可变 snapshot 文件；
    6. 原子切换 active pointer；
    7. 新 episode 才读取新 active；
    8. 保存父快照以供回滚。」

规范原文（§5.3，新旧版本共同竞争）：

    「1. 一个 episode 对同一 `skill_id` 最多交付一个版本；
    2. 检索先按题型过滤，再按谱系分组，最后在谱系内选版本；
    3. `top_k` 针对方法谱系，而不是任意版本条目；
    4. 同一谱系默认最多保留两个在线竞争版本；
    5. 第三个版本晋升时，最旧版本转为 `historical`，仍保留在历史快照但不参加新 episode；
    6. 不允许通过版本号大小直接获得排序加分；
    7. 检索分数、稳定 tie-break 和版本选择原因必须落盘。」

本模块因此提供两条发布路径：

- `publish_candidate_snapshot()`：v10 路径 —— 输入完整候选 `SkillCandidate`（含父版本
  声明），产出**完整快照** + `PromotionReceipt`（父子关系、manifest hash、竞争集合、
  历史版本、回滚点）；
- `promote()` / `apply_candidate()`：v9 路径（`CandidateRevision` → 单条目追加），
  保留给既有 CLI 与历史测试；两者共用同一套 `validate_references` 与原子指针写入。

快照条目键统一为 `skill_id@version`（与 S0 一致）。v9 路径的历史行为是按
`revision_id` 建键，因此本模块保留 `entry_key_mode` 参数并在 v10 路径上固定为
`skill_version` —— 不是静默改口径，而是显式声明。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from skill3d.schemas import CandidateRevision, PromotionReceipt, SkillCandidate, SkillSpec

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

# 快照条目的在线状态（§9.3 生命周期）：只有 `active_competing` 参加新 episode 的检索。
ENTRY_STATE_ACTIVE = "active_competing"
ENTRY_STATE_HISTORICAL = "historical"

# 同一谱系默认最多保留两个在线竞争版本（§5.3-4；首期已由用户确认）。
DEFAULT_MAX_COMPETING_PER_LINEAGE = 2

MANIFEST_REL_PATH = Path("manifests") / "library_manifest.json"


class SnapshotValidationError(ValueError):
    """新 snapshot 引用校验失败：不得切换。"""


class PublishLockedError(RuntimeError):
    """发布锁被占用（§10.1-1 读取并锁定父快照）。"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8")


def canonical_digest(value: object) -> str:
    """规范字节序列的 sha256（快照 / manifest 摘要口径的唯一实现）。"""
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@contextmanager
def publish_lock(store_dir: Path):
    """§10.1-1 的"锁定父快照"：以 O_EXCL 独占锁文件阻止并发发布。

    锁只是**发布动作**的互斥（防两次 promote 交错写指针）；它不阻塞在线读取 ——
    在线链读的是原子替换后的指针，见 `_write_pointer_atomic`。
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    lock = store_dir / ".publish.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise PublishLockedError(
            f"发布锁已被占用（{lock}）；确认没有并发 publish 后手动清理") from exc
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "at": _now_iso()}))
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _snapshot_path(store_dir: Path, snapshot_id: str) -> Path:
    return store_dir / f"snapshot_{snapshot_id}.json"


def read_snapshot(store_dir: str | Path, snapshot_id: str) -> dict:
    """读**指定**快照（发布时读父快照，§10.1-1）。"""
    path = _snapshot_path(Path(store_dir), snapshot_id)
    if not path.exists():
        raise FileNotFoundError(f"快照不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_active_snapshot(store_dir: str | Path) -> dict:
    """读 active 指针指向的 snapshot；无 active 时返回空初始 snapshot。"""
    store_dir = Path(store_dir)
    pointer = store_dir / "active_snapshot.json"
    if not pointer.exists():
        return {"snapshot_id": "genesis", "entries": {}, "parent_snapshot_id": None,
                "created_at": _now_iso()}
    ref = json.loads(pointer.read_text(encoding="utf-8"))
    return read_snapshot(store_dir, ref["snapshot_id"])


def apply_candidate(active: dict, candidate: CandidateRevision) -> dict:
    """v9 路径：在 active 的副本上应用候选，产出新 snapshot（不修改 active 本体）。

    条目键沿用 v9 的 `revision_id`（历史 CLI `manage_skill_library.py` 与既有测试依赖
    该口径）。v10 路线见 `publish_candidate_snapshot()`：它按 `skill_id@version` 建键，
    并维护谱系竞争状态 —— v9 路径**没有**这些字段，因此不得用于 v10 campaign。
    """
    new = json.loads(json.dumps(active))  # 深拷贝，物理隔离
    new["parent_snapshot_id"] = active["snapshot_id"]
    new["snapshot_id"] = f"snap-{uuid.uuid4().hex[:12]}"
    new["created_at"] = _now_iso()
    new.setdefault("entries", {})[candidate.revision_id] = {
        "root_candidate_id": candidate.root_candidate_id,
        "candidate_type": candidate.candidate_type,
        "spec_content": candidate.spec_content,
        "parent_version": candidate.parent_version,
        "created_by": candidate.created_by,
        # v9 provenance is audit metadata; spec_content remains the only
        # runtime payload consumed by the online loader.
        "source_split": candidate.source_split,
        "experience_relation": candidate.experience_relation,
        "source_path": candidate.source_path,
        "source_sha256": candidate.source_sha256,
        "generated_spec_path": candidate.generated_spec_path,
        "generated_sha256": candidate.generated_sha256,
        "manifest_ref": candidate.manifest_ref,
        "candidate_record_ref": candidate.candidate_record_ref,
    }
    return new


def validate_references(snapshot: dict, known_roots: set[str] | None = None,
                        *, strict_skill_specs: bool = False,
                        method_context_max_chars: int | None = None) -> None:
    """校验 snapshot 结构、候选根引用和可选的严格 SkillSpec 内容。

    Historical unit tests exercise this low-level writer with opaque template
    strings.  Production Skill-library promotion passes ``strict_skill_specs``
    so an invalid runtime SkillSpec cannot enter the active pointer; the
    compatibility default keeps the old generic candidate API unchanged.

    v9（§13.5）：严格模式下还做**服务限制**静态检查 —— 完整方法正文超过
    `method_context_max_chars` 的候选永远无法完整交付（§13.5 禁止截断正文后仍称
    "完整 Skill 已交付"），因此在发布前拒绝。

    v10 追加（§10.1-4）：严格模式下校验**谱系关系** —— 每条 SkillSpec 的
    `skill_id@version` 必须与条目键一致（否则"哪一条被检索到"无法回答）。
    """
    from skill3d.skills.delivery import DEFAULT_METHOD_CONTEXT_MAX_CHARS

    limit = int(DEFAULT_METHOD_CONTEXT_MAX_CHARS if method_context_max_chars is None
                else method_context_max_chars)
    if not snapshot.get("snapshot_id"):
        raise SnapshotValidationError("缺少 snapshot_id")
    entries = snapshot.get("entries")
    if not isinstance(entries, dict):
        raise SnapshotValidationError("entries 必须为 dict")
    for rid, e in entries.items():
        if not e.get("spec_content"):
            raise SnapshotValidationError(f"条目 {rid} 缺少 spec_content")
        if e.get("root_candidate_id") is None:
            raise SnapshotValidationError(f"条目 {rid} 缺少 root_candidate_id 引用")
        if known_roots is not None and e["root_candidate_id"] not in known_roots:
            raise SnapshotValidationError(
                f"条目 {rid} 引用未知 root_candidate_id={e['root_candidate_id']}"
            )
        if strict_skill_specs and e.get("candidate_type") == "skill":
            try:
                from skill3d.schemas import SkillSpec
                spec = SkillSpec.model_validate(json.loads(e["spec_content"]))
            except Exception as exc:  # noqa: BLE001 - publish must fail closed
                raise SnapshotValidationError(
                    f"条目 {rid} 不是合法运行 SkillSpec: {type(exc).__name__}: {exc}"
                ) from exc
            key = e.get("skill_version") or f"{spec.skill_id}@{spec.version}"
            if e.get("skill_version") and key != rid:
                raise SnapshotValidationError(
                    f"条目键 {rid} 与 SkillSpec 身份 {key} 不一致（§10.1-4 谱系校验）")
            from skill3d.skills.library import static_check_skill_spec

            problems = static_check_skill_spec(
                spec, method_context_max_chars=limit)
            if problems:
                raise SnapshotValidationError(
                    f"条目 {rid} 未通过静态检查（§14.4/§13.5）: {problems}")


# --------------------------------------------------------------------------- #
# v10：完整快照发布
# --------------------------------------------------------------------------- #

def _parse_semver(value: str) -> tuple[int, int, int]:
    m = _SEMVER_RE.match(str(value or ""))
    if not m:
        raise SnapshotValidationError(f"非法语义版本 {value!r}（应为 MAJOR.MINOR.PATCH）")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def lineage_of(key: str) -> str:
    """条目键 `skill_id@version` → 谱系 id。"""
    return key.split("@", 1)[0]


def _entry_of(skill_version: str, spec: SkillSpec, *, parent_version: str | None,
              created_by: str, state: str, extra: dict | None = None) -> dict:
    entry = {
        "root_candidate_id": spec.skill_id,
        "candidate_type": "skill",
        "skill_version": skill_version,
        "spec_content": json.dumps(spec.model_dump(mode="json"), ensure_ascii=False,
                                   indent=2, sort_keys=True),
        "parent_version": parent_version,
        "created_by": created_by,
        "state": state,
        "content_sha256": hashlib.sha256(
            json.dumps(spec.model_dump(mode="json"), ensure_ascii=False,
                       indent=2, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    if extra:
        entry.update(extra)
    return entry


def _spec_of_entry(entry: dict) -> SkillSpec | None:
    """条目 → SkillSpec（解析失败返回 None：损坏条目不得让发布崩在无关位置）。"""
    try:
        return SkillSpec.model_validate(json.loads(entry.get("spec_content") or "{}"))
    except Exception:  # noqa: BLE001
        return None


def _skill_key_of(key: str, entry: dict) -> str:
    """条目键 → `skill_id@version`（S0 与 v10 快照已按此建键；历史 v9 条目按内容规范化）。"""
    if "@" in key and _SEMVER_RE.match(key.split("@", 1)[1]):
        return key
    spec = _spec_of_entry(entry)
    if spec is None:
        return key
    return f"{spec.skill_id}@{spec.version}"


def build_candidate_snapshot(parent_snapshot: dict,
                             candidate: SkillCandidate,
                             *,
                             max_competing_per_lineage: int = DEFAULT_MAX_COMPETING_PER_LINEAGE,
                             created_at: str | None = None) -> tuple[dict, dict]:
    """§10.1-2/3：构建完整候选快照（返回 `(snapshot, 谱系变化摘要)`）。

    确定性：同一父快照 + 同一候选 → 同一快照内容（snapshot_id 也由内容派生，
    因此重复发布会撞到"文件已存在且内容不同"而不是静默覆盖）。
    """
    spec: SkillSpec = candidate.full_skill_spec
    key = f"{spec.skill_id}@{spec.version}"
    if candidate.candidate_skill_version != key:
        raise SnapshotValidationError(
            f"候选声明版本 {candidate.candidate_skill_version} 与 SkillSpec 身份 {key} 不一致")
    lineage = spec.skill_id
    if lineage != candidate.parent_skill_version.split("@", 1)[0]:
        raise SnapshotValidationError(
            f"候选跨谱系：parent={candidate.parent_skill_version} candidate={key}（§7.2）")

    # 1) 父快照的条目统一规范化成 `skill_id@version` 键（含历史 v9 的 revision_id 键）。
    entries: dict[str, dict] = {}
    for rid, entry in (parent_snapshot.get("entries") or {}).items():
        row = dict(entry)
        if row.get("candidate_type") != "skill":
            entries[rid] = row
            continue
        norm = _skill_key_of(str(rid), row)
        if norm in entries:
            raise SnapshotValidationError(
                f"父快照中 {norm} 出现两次（条目 {rid} 与既有条目冲突）")
        row["skill_version"] = norm
        row.setdefault("state", ENTRY_STATE_ACTIVE)
        entries[norm] = row

    if key in entries:
        raise SnapshotValidationError(
            f"版本 {key} 已存在于父快照（硬约束 11：候选不可变，必须 bump 版本）")
    parent_key = candidate.parent_skill_version
    if parent_key not in entries:
        raise SnapshotValidationError(
            f"父版本 {parent_key} 不在父快照 {parent_snapshot.get('snapshot_id')} 中"
            "（§5.2：新版本必须有父版本）")

    # 2) 新候选进入竞争集合。
    entries[key] = _entry_of(
        key, spec, parent_version=parent_key, created_by="offline_induction",
        state=ENTRY_STATE_ACTIVE,
        extra={
            "campaign_id": candidate.campaign_id,
            "generation": int(candidate.generation),
            "candidate_id": candidate.candidate_id,
            "inducer_receipt_ref": candidate.inducer_receipt_ref,
            "source_experience_bundle_ref": candidate.source_experience_bundle_ref,
            "hypothesis": candidate.hypothesis,
            "expected_effect": candidate.expected_effect,
        })

    # 3) 谱系内最多两个在线竞争版本：第三个晋升时最旧转 historical（§5.3-5）。
    by_lineage: dict[str, list[str]] = {}
    for rid, row in entries.items():
        if row.get("candidate_type") != "skill":
            continue
        if row.get("state") != ENTRY_STATE_ACTIVE:
            continue
        by_lineage.setdefault(lineage_of(str(row.get("skill_version") or rid)), []).append(
            str(row.get("skill_version") or rid))

    historized: list[str] = []
    limit = max(1, int(max_competing_per_lineage))
    for lin, keys in by_lineage.items():
        if len(keys) <= limit:
            continue
        ordered = sorted(keys, key=lambda k: _parse_semver(k.split("@", 1)[-1]))
        for old in ordered[: len(keys) - limit]:
            entries[old]["state"] = ENTRY_STATE_HISTORICAL
            entries[old]["historized_at"] = created_at or _now_iso()
            historized.append(old)

    # 4) 重算 skill_versions / generation / 谱系视图（§10.1-3）。
    skill_versions = sorted(rid for rid, row in entries.items()
                            if row.get("candidate_type") == "skill")
    active = sorted(rid for rid in skill_versions
                    if entries[rid].get("state") != ENTRY_STATE_HISTORICAL)
    historical = sorted(set(skill_versions) - set(active))
    lineages: dict[str, dict] = {}
    for rid in skill_versions:
        lin = lineage_of(rid)
        slot = lineages.setdefault(lin, {"active_competing": [], "historical": []})
        slot["historical" if rid in historical else "active_competing"].append(rid)
    generation = int(parent_snapshot.get("generation", 0)) + 1
    snapshot_id = f"S{generation}-{lineage}-{spec.version}-{candidate.candidate_id}"

    snapshot = {
        "schema_version": "runtime-skill-snapshot/1.0",
        "document_schema_version": parent_snapshot.get("document_schema_version", "8.0"),
        "snapshot_id": snapshot_id,
        "parent_snapshot_id": parent_snapshot.get("snapshot_id"),
        "generation": generation,
        "skill_versions": skill_versions,
        "active_skill_versions": active,
        "historical_skill_versions": historical,
        "competing_lineages": lineages,
        "manifest_hash": "",              # 由 publish 填（先算 manifest 再回填）
        "validation_receipt_refs": [],
        "compatibility": dict(parent_snapshot.get("compatibility") or {}),
        "entries": entries,
    }
    change = {
        "lineage": lineage,
        "added": [key],
        "historized": historized,
        "active_competing": sorted(lineages.get(lineage, {}).get("active_competing", [])),
        "historical": sorted(lineages.get(lineage, {}).get("historical", [])),
    }
    return snapshot, change


def _build_library_manifest(snapshot: dict, *, created_at: str) -> dict:
    """§10.1-3 的 manifest（与 S0 的 `manifests/library_manifest.json` 同构）。"""
    entries = []
    for rid in sorted(snapshot.get("entries") or {}):
        row = snapshot["entries"][rid]
        if row.get("candidate_type") != "skill":
            continue
        entries.append({
            "skill_version": rid,
            "skill_id": lineage_of(rid),
            "version": rid.split("@", 1)[1],
            "state": row.get("state", ENTRY_STATE_ACTIVE),
            "parent_version": row.get("parent_version"),
            "content_sha256": row.get("content_sha256", ""),
        })
    return {
        "library_format": "harness3d-skill-library/1.0",
        "snapshot_id": snapshot.get("snapshot_id"),
        "parent_snapshot_id": snapshot.get("parent_snapshot_id"),
        "generation": snapshot.get("generation"),
        "skill_versions": list(snapshot.get("skill_versions") or []),
        "active_skill_versions": list(snapshot.get("active_skill_versions") or []),
        "historical_skill_versions": list(snapshot.get("historical_skill_versions") or []),
        "competing_lineages": dict(snapshot.get("competing_lineages") or {}),
        "entries": entries,
        "runtime_verified": False,
        "created_at": created_at,
    }


def _write_pointer_atomic(store_dir: Path, snapshot_id: str) -> None:
    """原子写 active 指针：临时文件 + os.replace（硬约束 12）。"""
    pointer = store_dir / "active_snapshot.json"
    tmp = store_dir / f".active_snapshot.{uuid.uuid4().hex[:8]}.tmp"
    tmp.write_text(json.dumps({"snapshot_id": snapshot_id}, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, pointer)  # 同目录原子替换


def publish_candidate_snapshot(
    store_dir: str | Path,
    candidate: SkillCandidate,
    *,
    max_competing_per_lineage: int = DEFAULT_MAX_COMPETING_PER_LINEAGE,
    method_context_max_chars: int | None = None,
    expected_parent_snapshot_id: str | None = None,
) -> tuple[dict, PromotionReceipt]:
    """§10.1：发布完整候选快照并原子切换 active 指针。返回 `(snapshot, receipt)`。

    - `expected_parent_snapshot_id` 非空时校验当前 active 仍是该父快照（**防止在别人
      发布过的快照上继续发布**：那会让本代候选的父版本声明与实际父快照不符）；
    - 校验失败（结构 / SkillSpec / 谱系 / 服务限制）→ 抛 `SnapshotValidationError`，
      active 指针**不变**；
    - 父快照文件与 `<store>/../manifests/library_manifest.json` 一并保留 / 更新，
      供回滚与 provenance 复算（§10.1-8）。
    """
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    with publish_lock(store_dir):
        before = read_active_snapshot(store_dir)
        if expected_parent_snapshot_id and \
                before.get("snapshot_id") != expected_parent_snapshot_id:
            raise SnapshotValidationError(
                f"active 快照不是本代父快照：期望 {expected_parent_snapshot_id}，"
                f"实际 {before.get('snapshot_id')}")
        if candidate.parent_snapshot_id and \
                str(before.get("snapshot_id")) != str(candidate.parent_snapshot_id):
            raise SnapshotValidationError(
                f"候选声明的父快照 {candidate.parent_snapshot_id} ≠ active "
                f"{before.get('snapshot_id')}（§5.2：父快照必须与实际一致）")

        created_at = _now_iso()
        new, change = build_candidate_snapshot(
            before, candidate,
            max_competing_per_lineage=max_competing_per_lineage, created_at=created_at)
        validate_references(new, strict_skill_specs=True,
                            method_context_max_chars=method_context_max_chars)

        manifest = _build_library_manifest(new, created_at=created_at)
        manifest_digest = canonical_digest(manifest)
        manifest["manifest_sha256"] = manifest_digest
        new["manifest_hash"] = manifest_digest
        new["validation_receipt_refs"] = [
            f"promotion/{candidate.candidate_id}.json"]

        # 5) 写不可变快照（内容已存在的同一快照 → 幂等返回，不覆盖）
        snapshot_path = _snapshot_path(store_dir, new["snapshot_id"])
        payload = _canonical_bytes(new)
        if snapshot_path.exists():
            if snapshot_path.read_bytes() != payload:
                raise SnapshotValidationError(
                    f"快照 {new['snapshot_id']} 已存在且内容不同（拒绝覆盖）")
        else:
            snapshot_path.write_bytes(payload)

        library_root = store_dir.parent
        manifest_path = library_root / MANIFEST_REL_PATH
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(_canonical_bytes(manifest))
        receipt_dir = library_root / "validation" / "promotion"
        receipt_dir.mkdir(parents=True, exist_ok=True)

        # 6) 原子切换 active 指针（新 episode 才读新 active，§10.1-7）
        _write_pointer_atomic(store_dir, new["snapshot_id"])

        receipt = PromotionReceipt(
            campaign_id=candidate.campaign_id,
            generation=int(candidate.generation),
            candidate_id=candidate.candidate_id,
            skill_version=candidate.candidate_skill_version,
            snapshot_before=str(before.get("snapshot_id")),
            snapshot_after=str(new["snapshot_id"]),
            manifest_hash_before=str(before.get("manifest_hash", "") or ""),
            manifest_hash_after=manifest_digest,
            parent_snapshot_id=before.get("snapshot_id"),
            competing_versions=list(change["active_competing"]),
            historical_versions=list(change["historical"]),
            rollback_ref=str(_snapshot_path(store_dir, str(before.get("snapshot_id")))),
            created_at=created_at,
        )
        (receipt_dir / f"{candidate.candidate_id}.json").write_bytes(
            _canonical_bytes(receipt.model_dump(mode="json")))
    return new, receipt


def promote(store_dir: str | Path, candidate: CandidateRevision,
            known_roots: set[str] | None = None,
            promotion_log: list | None = None,
            *, strict_skill_specs: bool = False,
            method_context_max_chars: int | None = None) -> dict:
    """promote(candidate)：原子切换 active snapshot，返回新 snapshot（v9 路径）。

    validate 失败 → 不切换（抛 SnapshotValidationError，active 保持不变）。
    v10 演化链请用 `publish_candidate_snapshot()`。
    """
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    before = read_active_snapshot(store_dir)
    new = apply_candidate(before, candidate)
    validate_references(new, known_roots=known_roots,
                        strict_skill_specs=strict_skill_specs,
                        method_context_max_chars=method_context_max_chars)  # 失败即不切换
    _snapshot_path(store_dir, new["snapshot_id"]).write_text(
        json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_pointer_atomic(store_dir, new["snapshot_id"])
    if promotion_log is not None:
        promotion_log.append({
            "revision_id": candidate.revision_id,
            "snapshot_before": before["snapshot_id"],  # 记录 snapshot_before 供回滚
            "snapshot_after": new["snapshot_id"],
            "timestamp": _now_iso(),
        })
    return new


def rollback(store_dir: str | Path, snapshot_before: str,
             promotion_log: list | None = None) -> dict:
    """一键回滚到旧 snapshot（旧 snapshot 文件必须仍在）。"""
    store_dir = Path(store_dir)
    path = _snapshot_path(store_dir, snapshot_before)
    if not path.exists():
        raise FileNotFoundError(f"回滚目标 snapshot 不存在: {snapshot_before}")
    snap = json.loads(path.read_text(encoding="utf-8"))
    _write_pointer_atomic(store_dir, snapshot_before)
    if promotion_log is not None:
        promotion_log.append({"rollback_to": snapshot_before, "timestamp": _now_iso()})
    return snap


__all__ = [
    "DEFAULT_MAX_COMPETING_PER_LINEAGE",
    "ENTRY_STATE_ACTIVE",
    "ENTRY_STATE_HISTORICAL",
    "MANIFEST_REL_PATH",
    "PublishLockedError",
    "SnapshotValidationError",
    "apply_candidate",
    "build_candidate_snapshot",
    "canonical_digest",
    "lineage_of",
    "promote",
    "publish_candidate_snapshot",
    "publish_lock",
    "read_active_snapshot",
    "read_snapshot",
    "rollback",
    "validate_references",
]
