"""§17.1「数据访问」记录行 + §14.1 `label_access`（v9）。

规范原文（§17.1 最小真实记录表）：

    数据访问 | split、用途、组件角色、运行／候选身份、输入清单 hash、时间；生产入口实际记录

规范原文（§14.1 数据权限与经验资格）：

    "数据按 scene 分为 learning／induction、inner_validation、outer_holdout、
    final_test，同场景的题、图片、缓存和派生产物跟随划分。"
    "学习题标签允许用于离线分析并记 `label_access`；GT 3D 框、位姿和尺度默认不提供，
    不进入在线测量或标定。"
    "inner 的逐题题目、图片、标签、轨迹和错误分析只留在验证器，不交给归纳器修订。
    归纳器只获得聚合得分／失败统计及接受或拒绝决定；不能通过检索记录、调试日志或
    候选报告绕过该限制。"

三条纪律：

1. **记录必须在真实读取点产生**（"生产入口实际记录"）——不得事后补写一条看起来
   完整的访问记录；`at` 是**实际读取发生的时间**，不是报告生成时间。
2. **拒收也必须留痕**：隔离不是"悄悄过滤掉其它划分的轨迹"，被拒条目要逐条进
   `refused`（数量、原因、id 集合 hash），否则审计无法回答"到底读没读 inner"。
3. **`label_access` 只对学习划分成立**：§14.1 允许学习题标签用于离线分析（记
   `label_access`）；验证器读 inner 标签用于评分是另一回事，不套用这个字段
   （由 validator 角色与 `purpose` 表达）。
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Mapping, Sequence

from pydantic import model_validator

from . import Spec

# 数据划分（§14.1 / §16.2 四层）
DATA_SPLITS = ("induction", "inner_validation", "outer_holdout", "final_test")

# 组件角色（谁在读）——只登记当前真实存在的读取方，新增读取方须先登记名字
COMPONENT_ROLES = ("inducer", "validator", "evaluator")

# 用途（读来做什么）
ACCESS_PURPOSES = ("skill_induction", "candidate_admission_panel", "learning_run")

# 拒收原因码（§14.1 硬隔离；当前只有一类：不属于本次读取允许的学习划分）
REFUSAL_REASON_CODES = frozenset({"not_in_learning_split"})

EXPERIMENT_QA_ROLE_SCHEMA = "harness3d-experiment-qa-roles/1.0"
TRAIN_TEST_PARTITION_SCHEMA = "harness3d-train-test-split/1"


def is_train_test_partition(value: Mapping | None) -> bool:
    return bool(value and value.get("protocol") == TRAIN_TEST_PARTITION_SCHEMA
                and value.get("partition_schema") == TRAIN_TEST_PARTITION_SCHEMA)


def _identity_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def normalize_experiment_split(value: Mapping) -> dict:
    """Freeze a QA-role overlay without changing any catalog split."""
    body = deepcopy(dict(value))
    claimed = body.pop("identity_sha256", None)
    roles_hash = body.pop("qa_roles_sha256", None)
    body.setdefault("schema_version", EXPERIMENT_QA_ROLE_SCHEMA)
    if body["schema_version"] != EXPERIMENT_QA_ROLE_SCHEMA:
        raise ValueError("unknown experiment QA role schema")
    if not isinstance(body.get("protocol"), str) or not body["protocol"].strip():
        raise ValueError("experiment_split requires an explicit protocol")
    roles = body.get("qa_roles")
    if not isinstance(roles, dict) or not roles or any(
        not isinstance(key, str) or not key.strip()
        or role not in {"learning", "validation"} for key, role in roles.items()
    ):
        raise ValueError("qa_roles must map nonempty QA IDs to learning/validation")
    if (body.get("protocol") == TRAIN_TEST_PARTITION_SCHEMA
            or body.get("partition_schema") == TRAIN_TEST_PARTITION_SCHEMA):
        sources = body.get("source_split_by_qa")
        if (not is_train_test_partition(body)
                or body.get("source_split_preserved") is not True
                or not isinstance(sources, dict) or set(sources) != set(roles)
                or any(source not in DATA_SPLITS for source in sources.values())):
            raise ValueError("train/test protocol requires the complete preserved source role map")
    manifest = body.get("manifest_identity")
    if not isinstance(manifest, dict) or not any(
        re.fullmatch(r"[0-9a-f]{64}", str(manifest.get(key, "")))
        for key in ("sha256", "identity_sha256")
    ):
        raise ValueError("experiment_split requires a hashed manifest_identity")
    actual_roles = _identity_sha256(roles)
    if roles_hash is not None and roles_hash != actual_roles:
        raise ValueError("experiment QA role map hash changed")
    body["qa_roles_sha256"] = actual_roles
    actual = _identity_sha256(body)
    if claimed is not None and claimed != actual:
        raise ValueError("experiment_split identity hash changed")
    return {**body, "identity_sha256": actual}


def learning_authorization_for(experiment_split: Mapping, qa_ids: Sequence[str]) -> dict:
    overlay = normalize_experiment_split(experiment_split)
    ids = sorted(set(qa_ids))
    if (overlay.get("execution_role") == "validation" or not ids
            or any(overlay["qa_roles"].get(qa) != "learning" for qa in ids)):
        raise ValueError("validation or unmapped QA cannot enter learning")
    receipt = {
        "gate_mode": "train_sample" if is_train_test_partition(overlay) else "qa_disjoint",
        "experiment_role": "learning", "qa_ids": ids,
        "protocol": overlay["protocol"], "manifest_identity": overlay["manifest_identity"],
        "qa_roles_sha256": overlay["qa_roles_sha256"],
        "experiment_split_sha256": overlay["identity_sha256"],
    }
    if is_train_test_partition(overlay):
        receipt.update({
            "partition_schema": TRAIN_TEST_PARTITION_SCHEMA,
            "source_split_by_qa": {qa: overlay["source_split_by_qa"][qa] for qa in ids},
            "source_split_by_qa_sha256": _identity_sha256(overlay["source_split_by_qa"]),
        })
    return receipt


def validate_learning_authorization(
    authorization: Mapping, qa_ids: Sequence[str], experiment_split: Mapping | None = None,
) -> None:
    """Check the scoped receipt; consumers also bind it to the frozen overlay."""
    ids = sorted(set(qa_ids))
    if (
        not ids or authorization.get("qa_ids") != ids
        or authorization.get("gate_mode") != (
            "train_sample" if is_train_test_partition(authorization) else "qa_disjoint")
        or authorization.get("experiment_role") != "learning"
        or not authorization.get("protocol")
        or not isinstance(authorization.get("manifest_identity"), dict)
        or any(not re.fullmatch(r"[0-9a-f]{64}", str(authorization.get(key, "")))
               for key in ("qa_roles_sha256", "experiment_split_sha256"))
    ):
        raise ValueError("missing or invalid explicit learning authorization")
    if authorization.get("gate_mode") == "train_sample":
        sources = authorization.get("source_split_by_qa")
        if (not isinstance(sources, dict) or set(sources) != set(ids)
                or any(source not in DATA_SPLITS for source in sources.values())
                or not re.fullmatch(r"[0-9a-f]{64}", str(
                    authorization.get("source_split_by_qa_sha256", "")))):
            raise ValueError("train/test learning authorization lacks scoped source identity")
    elif (authorization.get("protocol") == TRAIN_TEST_PARTITION_SCHEMA
          or authorization.get("partition_schema") == TRAIN_TEST_PARTITION_SCHEMA):
        raise ValueError("train/test learning requires train_sample authorization")
    manifest = authorization["manifest_identity"]
    if not any(re.fullmatch(r"[0-9a-f]{64}", str(manifest.get(key, "")))
               for key in ("sha256", "identity_sha256")):
        raise ValueError("learning authorization lacks manifest identity")
    if experiment_split is not None and dict(authorization) != learning_authorization_for(
        experiment_split, ids,
    ):
        raise ValueError("learning authorization differs from frozen QA role map")


def validate_learning_source(authorization: Mapping, qa_id: str, source_split: str) -> None:
    """Keep legacy split guards; only the new scoped receipt admits all four."""
    validate_learning_authorization(authorization, authorization.get("qa_ids", []))
    if is_train_test_partition(authorization):
        if authorization["source_split_by_qa"].get(qa_id) != source_split:
            raise ValueError("authorized learning source_split changed")
    elif source_split not in {"induction", "inner_validation"}:
        raise ValueError("legacy learning authorization requires induction/inner_validation")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DataAccessRecord(Spec):
    """§17.1「数据访问」一行：谁（角色／运行身份）在什么时候读了哪个 split 的什么。

    `input_manifest_sha256` 由**实际读取到的条目清单**确定（不是目录哈希）：
    同一清单复算必须一致，从而让"读的是哪一批材料"可复核。
    """

    record_id: str
    at: str = ""
    split: str
    purpose: str
    component_role: str
    # 运行身份：演化 run 的 run_id（§14.2 每轮冻结一次）
    run_id: str = ""
    # 候选身份：该访问属于某个候选验证时填 revision/candidate id，否则留空
    candidate_id: str = ""
    input_manifest_sha256: str = ""
    n_items: int = 0
    # §14.1：本次读取是否用了学习题标签（成功/失败等派生标签）做离线分析
    label_access: bool = False
    # 被拒条目（隔离证据）：{"n": int, "by_reason": {code: n}, "ids_sha256": str}
    refused: dict = {}
    # 实际读取的路径/glob（生产入口传入，不猜）
    source_refs: list[str] = []
    notes: list[str] = []
    learning_authorization: dict = {}

    @model_validator(mode="after")
    def _check(self) -> "DataAccessRecord":
        if self.split not in DATA_SPLITS:
            raise ValueError(f"未登记的数据划分: {self.split!r}（可选 {DATA_SPLITS}）")
        if self.component_role not in COMPONENT_ROLES:
            raise ValueError(
                f"未登记的组件角色: {self.component_role!r}（可选 {COMPONENT_ROLES}）")
        if self.purpose not in ACCESS_PURPOSES:
            raise ValueError(
                f"未登记的用途: {self.purpose!r}（可选 {ACCESS_PURPOSES}）")
        # §14.1：只有学习划分的标签可以记 label_access
        if self.learning_authorization:
            validate_learning_authorization(
                self.learning_authorization, self.learning_authorization.get("qa_ids", []))
            for qa_id in self.learning_authorization["qa_ids"]:
                validate_learning_source(self.learning_authorization, qa_id, self.split)
            if (self.component_role != "inducer"
                    or self.purpose != "skill_induction"
                    or not self.label_access
                    or len(self.learning_authorization["qa_ids"]) != self.n_items):
                raise ValueError("invalid experiment learning label access")
        if self.label_access and self.split != "induction" and not self.learning_authorization:
            raise ValueError(
                "label_access=True 只对学习划分（induction）成立（§14.1）")
        if self.refused:
            bad = set(self.refused.get("by_reason", {})) - REFUSAL_REASON_CODES
            if bad:
                raise ValueError(f"未登记的拒收原因码: {sorted(bad)}")
            if int(self.refused.get("n", 0)) != sum(
                    int(v) for v in self.refused.get("by_reason", {}).values()):
                raise ValueError("refused.n 与 by_reason 合计不一致")
        return self


__all__ = [
    "ACCESS_PURPOSES",
    "COMPONENT_ROLES",
    "DATA_SPLITS",
    "REFUSAL_REASON_CODES",
    "DataAccessRecord",
    "EXPERIMENT_QA_ROLE_SCHEMA",
    "TRAIN_TEST_PARTITION_SCHEMA",
    "is_train_test_partition",
    "learning_authorization_for",
    "normalize_experiment_split",
    "validate_learning_authorization",
    "validate_learning_source",
    "utcnow_iso",
]
