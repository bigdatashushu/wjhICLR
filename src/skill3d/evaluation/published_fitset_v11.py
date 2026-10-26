"""Label-free replay of audited publications and independently extracted inputs.

The input manifest uses ``v11-published-fitset-inputs/1``. Its ``entries`` map
QA IDs to original ``cluster_audit`` catalog entries plus ``input_root`` and
``frozen_scene_sha256``. ``source_catalog_sha256`` and ``selection_policy`` are
required provenance anchors. No catalog locator or sealed identity is rewritten.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
import json
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Mapping

from skill3d.cluster import scene_compute

if TYPE_CHECKING:
    from skill3d.adapters.episode_source import EpisodeItem


PUBLISHED_INPUT_SCHEMA = "v11-published-fitset-inputs/1"
FULLQA_INPUT_SCHEMA = "v11-published-fitset-inputs/2"
TRANSPORT_RECEIPT_SCHEMA = "v11-published-fitset-transport/1"
_CATALOG_FIELDS = {
    "qa_id", "scene_id", "split", "variant", "reference", "scene_artifact",
    "scene_input_identity_file", "scene_input_identity_file_sha256",
    "migration_receipt_file", "migration_receipt_file_sha256",
    "publication_root", "publication_receipt_sha256",
}
_INPUT_FIELDS = {"input_root", "frozen_scene_sha256"}
_PROVENANCE_FIELDS = {
    "input_archive_sha256", "prepackage_scene_sha256",
    "storage_migration_receipt_sha256",
}
_PRIVATE_FIELDS = {
    "ground_truth", "answer", "answers", "reference_answer", "reference_answers",
    "correct_answer", "label", "labels", "score", "scores", "accuracy",
    "outcome", "outcomes", "predicted", "prediction",
}


class PublishedFitSetInputError(ValueError):
    """The frozen input, catalog, or publication no longer matches its anchor."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublishedFitSetInputError(message)


def _public_json(value) -> None:
    if isinstance(value, dict):
        for name, child in value.items():
            _require(name not in _PRIVATE_FIELDS,
                     f"published input contains forbidden label/score field: {name}")
            _public_json(child)
    elif isinstance(value, list):
        for child in value:
            _public_json(child)


def _public_entry(entry: dict) -> dict:
    public = copy.deepcopy(entry)
    if "outcome" in public:
        diagnostic = public.pop("outcome")
        _require(
            isinstance(diagnostic, dict)
            and set(diagnostic) == {
                "status", "materialized", "n_objects", "binding_error",
            }
            and isinstance(diagnostic["status"], str)
            and type(diagnostic["materialized"]) is bool
            and type(diagnostic["n_objects"]) is int
            and diagnostic["n_objects"] >= 0
            and (diagnostic["binding_error"] is None
                 or isinstance(diagnostic["binding_error"], str)),
            "catalog outcome must contain only unlabelled binding diagnostics",
        )
    _public_json(public)
    return public


def _file(path: str | Path, expected_sha256: str | None = None) -> dict:
    from skill3d.infra.m5_identity import file_identity

    path = Path(path).expanduser().absolute()
    scene_compute._no_symlinks(path)
    identity = file_identity(path)
    if expected_sha256 is not None:
        _require(identity["sha256"] == expected_sha256,
                 f"published input file integrity changed: {path}")
    return {"path": str(path), **identity}


def _read_anchored(path: str | Path, sha256: str) -> tuple[dict, dict]:
    identity = _file(path, sha256)
    value = scene_compute._read_json(Path(identity["path"]))
    _require(_file(path, sha256) == identity,
             "published input file changed during reading")
    return value, identity


def _json_fingerprint(value) -> tuple:
    """Compare loaded JSON without serialization or mutable snapshot aliases."""
    kind = type(value)
    if kind is dict:
        _require(all(type(name) is str for name in value),
                 "cached frozen JSON keys changed")
        return (kind, tuple((name, _json_fingerprint(value[name]))
                            for name in sorted(value)))
    if kind is list:
        return (kind, tuple(_json_fingerprint(child) for child in value))
    if kind is float:
        return (kind, value.hex())
    _require(kind in (str, int, bool, type(None)),
             "cached frozen JSON structure changed")
    return (kind, value)


def _scene_file_closure(loaded) -> tuple:
    files = {}

    def add(target, identity=None):
        if identity is None:
            files.setdefault(target, None)
            return
        expected = (identity["sha256"], identity["size"])
        _require(files.get(target) in (None, expected),
                 "conflicting frozen file integrity anchors")
        files[target] = expected

    for target in loaded.manifest["files"].values():
        add(target)
    for target, identity in (loaded.manifest.get("file_identities") or {}).items():
        add(target, identity)
    for frame in loaded.manifest["frames"]:
        add(loaded.manifest["files"][frame["source_path"]],
            {"sha256": frame["png_sha256"], "size": frame["size"]})
    for receipt in loaded.receipts.values():
        for record in (receipt["artifact"], *receipt["dependencies"].values()):
            add(record["target_ref"], record["identity"])
    return tuple(files.items())


class PublishedFitSetInputProvider:
    """Decode at most ``max_scenes`` frozen scenes, without metadata or labels.

    ``provider(variant)`` returns an isolated ``EpisodeItem`` with empty GT.
    ``receipt(variant, item)`` replays transport integrity and returns safe JSON.
    ``prepare_published(variant, item)`` additionally loads the original sealed
    binding and returns a runtime artifact and validated relocation capability.
    Public ``entries`` omit only the catalog's unlabelled binding diagnostics;
    the original catalog entry hashes and full manifest file remain anchored.
    """

    def __init__(self, manifest_path: str | Path, max_scenes: int = 2) -> None:
        from skill3d.infra.m5_identity import content_sha256

        _require(type(max_scenes) is int and max_scenes > 0,
                 "scene cache size must be a positive integer")
        self.manifest_path = Path(manifest_path).expanduser().absolute()
        self._manifest_identity = _file(self.manifest_path)
        manifest, _ = _read_anchored(
            self.manifest_path, self._manifest_identity["sha256"])
        self.schema = manifest.get("schema")
        _require(self.schema in {PUBLISHED_INPUT_SCHEMA, FULLQA_INPUT_SCHEMA},
                 "published fit-set input manifest schema changed")
        scene_compute._digest(manifest.get("source_catalog_sha256"))
        _require(bool(manifest.get("selection_policy")),
                 "published inputs require a selection_policy")
        entries = manifest.get("entries")
        _require(isinstance(entries, dict) and bool(entries),
                 "published inputs require entries keyed by qa_id")
        self.scene_scopes = {}
        source_catalog = None
        if self.schema == FULLQA_INPUT_SCHEMA:
            source_catalog, _ = _read_anchored(
                manifest["source_catalog_file"], manifest["source_catalog_sha256"])
            _require(source_catalog.get("schema") == "harness3d-published-binding-catalog/1",
                     "fullQA source catalog schema changed")
            scopes = manifest.get("scene_scopes")
            _require(isinstance(scopes, dict) and bool(scopes),
                     "fullQA requires frozen scene scopes")
            for scene_id, scope in scopes.items():
                _require(isinstance(scope, dict) and set(scope) == {
                    "qa_ids", "split", "frozen_scene_sha256", "input_root"},
                    "invalid fullQA scene scope")
                ids = scope["qa_ids"]
                _require(isinstance(ids, list) and 1 <= len(ids) <= 128
                         and len(set(ids)) == len(ids)
                         and all(isinstance(q, str) and q for q in ids)
                         and scope["split"] in {"induction", "inner_validation"},
                         "invalid fullQA bounded question/split scope")
                self.scene_scopes[scene_id] = copy.deepcopy(scope)
        public_entries = {}
        for qa_id, entry in entries.items():
            _require(
                isinstance(qa_id, str) and bool(qa_id)
                and isinstance(entry, dict)
                and _CATALOG_FIELDS | _INPUT_FIELDS <= set(entry)
                and set(entry) <= _CATALOG_FIELDS | _INPUT_FIELDS | _PROVENANCE_FIELDS | {"outcome"}
                and entry["qa_id"] == qa_id,
                "published entry differs from the original catalog schema/qa_id",
            )
            allowed_splits = ({"induction", "inner_validation"} if
                              self.schema == FULLQA_INPUT_SCHEMA else {"induction"})
            _require(entry["split"] in allowed_splits and entry["variant"] == "base",
                     f"{qa_id}: published fit-set requires induction/base")
            if self.schema == FULLQA_INPUT_SCHEMA:
                original = {k: v for k, v in entry.items()
                            if k not in _INPUT_FIELDS | _PROVENANCE_FIELDS}
                _require(source_catalog["variants"]["base"].get(qa_id) == original,
                         "fullQA catalog entry changed")
                scope = self.scene_scopes.get(entry["scene_id"])
                _require(scope is not None and qa_id in scope["qa_ids"]
                         and all(scope[k] == entry[k] for k in
                                 ("split", "input_root", "frozen_scene_sha256")),
                         "fullQA entry differs from frozen scene scope")
            _require(isinstance(entry["scene_id"], str) and bool(entry["scene_id"])
                     and isinstance(entry["reference"], dict),
                     f"{qa_id}: missing catalog scene/reference")
            reference = entry["reference"]
            _require(set(reference) == {
                "schema_version", "store_root", "key", "request_identity",
                "scene_artifact_ref", "scene_input_identity_sha256", "manifest_sha256",
            } and isinstance(reference["request_identity"], dict),
                     f"{qa_id}: original catalog reference schema changed")
            for name in ("key", "scene_input_identity_sha256", "manifest_sha256"):
                scene_compute._digest(reference[name])
            for name in ("store_root", "scene_artifact_ref"):
                scene_compute._reference(reference[name], absolute=True)
            for name in (
                "scene_input_identity_file_sha256", "migration_receipt_file_sha256",
                "publication_receipt_sha256", "frozen_scene_sha256",
            ):
                scene_compute._digest(entry[name])
            for name in _PROVENANCE_FIELDS & set(entry):
                scene_compute._digest(entry[name])
            for name in (
                "input_root", "scene_artifact", "scene_input_identity_file",
                "migration_receipt_file", "publication_root",
            ):
                scene_compute._reference(entry[name], absolute=True)
            public_entries[qa_id] = _public_entry(entry)
        _public_json({**manifest, "entries": public_entries})
        self._entries = copy.deepcopy(entries)
        self._input_contract_identity = {
            "schema": self.schema,
            "manifest": copy.deepcopy(self._manifest_identity),
            "source_catalog_sha256": manifest["source_catalog_sha256"],
            "selection_policy": copy.deepcopy(manifest["selection_policy"]),
            "entries_sha256": content_sha256(entries),
            "real_labels_available": False,
        }
        self.max_scenes = max_scenes
        self.cache: OrderedDict[tuple[str, str], object] = OrderedDict()
        self._scene_checks = {}
        self.cache_lock = threading.RLock()

    @property
    def entries(self) -> dict[str, dict]:
        return {qa_id: _public_entry(entry) for qa_id, entry in self._entries.items()}

    @property
    def input_contract_identity(self) -> dict:
        return copy.deepcopy(self._input_contract_identity)

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["cache"] = OrderedDict()
        state["_scene_checks"] = {}
        state.pop("cache_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.cache = OrderedDict()
        self._scene_checks = {}
        self.cache_lock = threading.RLock()

    def _entry(self, variant: Mapping) -> dict:
        _require(_file(self.manifest_path) == self._manifest_identity,
                 "published input manifest integrity changed")
        qa_id = variant.get("qa_id")
        entry = self._entries.get(qa_id)
        _require(entry is not None, f"{qa_id}: published catalog entry is missing")
        _require(
            variant.get("split") == entry["split"]
            and (variant.get("source_split", entry["split"]) == entry["split"])
            and variant.get("scene_id") == entry["scene_id"]
            and entry["variant"] == "base"
            and variant.get("artifact_ref")
            == entry["reference"].get("scene_artifact_ref"),
            f"{qa_id}: panel/catalog identity changed (original artifact_ref required)",
        )
        return entry

    def _scene(self, entry: dict):
        from skill3d.infra.m5_identity import file_identity

        root = Path(entry["input_root"])
        key = (str(root), entry["frozen_scene_sha256"])
        with self.cache_lock:
            _file(root / "scene.json", entry["frozen_scene_sha256"])
            if key not in self.cache:
                scope = self.scene_scopes.get(entry["scene_id"])
                cap = len(scope["qa_ids"]) if scope is not None else 8
                loaded = scene_compute._load_scene(root, question_cap=cap)
                if scope is not None:
                    _require(loaded.manifest["qa_ids"] == scope["qa_ids"]
                             and loaded.manifest["split"] == scope["split"],
                             "fullQA frozen scene question/split scope changed")
                _file(root / "scene.json", entry["frozen_scene_sha256"])
                # The normal cold loader validates every receipt and its anchors.
                self._scene_checks[key] = (
                    loaded, loaded.root, _json_fingerprint(loaded.manifest),
                    _json_fingerprint(loaded.receipts), _scene_file_closure(loaded),
                )
                self.cache[key] = loaded
                while len(self.cache) > self.max_scenes:
                    evicted, _ = self.cache.popitem(last=False)
                    self._scene_checks.pop(evicted, None)
            else:
                loaded = self.cache[key]
                trusted, checked_root, manifest, receipts, files = self._scene_checks[key]
                _require(loaded is trusted and loaded.root == checked_root
                         and scene_compute._root(root) == checked_root,
                         "cached frozen scene root changed")
                _require(_json_fingerprint(loaded.manifest) == manifest
                         and _json_fingerprint(loaded.receipts) == receipts,
                         "cached frozen scene structure/receipt changed")
                # Trust only unchanged cold-validated structures, not receipt hashes.
                for target, expected in files:
                    path = scene_compute._local_file(checked_root, target)
                    if expected is not None:
                        actual = file_identity(path)
                        _require((actual["sha256"], actual["size"]) == expected,
                                 "frozen transferred file integrity changed")
                self.cache.move_to_end(key)
            return loaded

    def _validate_item(self, variant: Mapping, entry: dict, item: EpisodeItem) -> None:
        from skill3d.adapters.episode_source import EpisodeItem
        from skill3d.adapters.frame_set import with_pixel_identity
        from skill3d.routing.task_classifier import canonical_task

        _require(isinstance(item, EpisodeItem) and item.input_error is None
                 and item.geometry is None,
                 "published provider requires a normal, error-free EpisodeItem")
        scene = self._scene(entry)
        _require(scene.manifest["scene_id"] == entry["scene_id"]
                 and scene.manifest["split"] == entry["split"],
                 "frozen scene/catalog identity changed")
        original = next(
            (candidate for candidate in scene.items
             if candidate.episode.qa_id == entry["qa_id"]), None)
        _require(original is not None, "catalog QA is missing from frozen scene")
        ep = item.episode
        _require(
            ep.ground_truth == ""
            and ep.model_dump(mode="json") == original.episode.model_dump(mode="json")
            and item.source == original.source and item.video_path == original.video_path
            and canonical_task(ep.question_type) == variant.get("question_type"),
            "published public episode/source/task identity changed",
        )
        fs = ep.frame_set
        _require(
            fs is not None and fs.n_frames == 32 and fs.decode_status == "ok"
            and len(fs.frame_ids) == len(fs.readable_frame_ids) == len(ep.frames) == 32
            and fs.readable_frame_ids == fs.frame_ids and len(item.pixels) == 32,
            "published input requires exactly 32 readable frames",
        )
        _require(with_pixel_identity(fs, item.pixels).pixel_sha256 == fs.pixel_sha256,
                 "published frozen RGB pixel identity changed")

    def __call__(self, variant: Mapping) -> EpisodeItem:
        entry = self._entry(variant)
        scene = self._scene(entry)
        item = next(
            (candidate for candidate in scene.items
             if candidate.episode.qa_id == entry["qa_id"]), None)
        _require(item is not None, "catalog QA is missing from frozen scene")
        item = copy.deepcopy(item)
        self._validate_item(variant, entry, item)
        return item

    def _transport(self, variant: Mapping, item: EpisodeItem) -> dict:
        from skill3d.infra.m5_identity import content_sha256
        from skill3d.segmentation.prepared_binding_store import (
            PreparedBindingStore, PreparedBindingStoreError,
        )

        entry = self._entry(variant)
        self._validate_item(variant, entry, item)
        identity, identity_file = _read_anchored(
            entry["scene_input_identity_file"],
            entry["scene_input_identity_file_sha256"],
        )
        validation, migration_file = _read_anchored(
            entry["migration_receipt_file"], entry["migration_receipt_file_sha256"])
        try:
            migration = validation["variants"][entry["variant"]]["migration_receipts"][
                entry["qa_id"]]
        except (KeyError, TypeError) as exc:
            raise PublishedFitSetInputError("catalog migration receipt is missing") from exc
        reference = entry["reference"]
        store = PreparedBindingStore(reference["store_root"])
        _require(identity.get("artifact_path") == reference["scene_artifact_ref"]
                 and identity.get("sha256") == reference["scene_input_identity_sha256"],
                 "catalog logical scene identity changed")
        try:
            _require(reference == store.reference(reference["key"]),
                     "original catalog binding reference/manifest changed")
            resolver = store.validate_migration_receipt(
                reference["key"], migration,
                scene_artifact=entry["scene_artifact"], scene_input_identity=identity,
                destination_root=entry["publication_root"],
            )
        except PreparedBindingStoreError as exc:
            raise PublishedFitSetInputError(f"published binding replay rejected: {exc}") from exc
        actual = resolver.recompute_scene_input_identity(
            episode=item.episode.model_dump(mode="json"), pixels=item.pixels,
            source=item.source, video_path=item.video_path,
        )
        _require(actual == identity, "published logical input identity changed")
        fs = item.episode.frame_set
        original_catalog = {k: v for k, v in entry.items()
                            if k not in _INPUT_FIELDS | _PROVENANCE_FIELDS}
        receipt = {
            "schema": TRANSPORT_RECEIPT_SCHEMA,
            "variant_id": variant.get("variant_id"),
            "qa_id": entry["qa_id"], "scene_id": entry["scene_id"],
            "split": entry["split"], "question_type": variant["question_type"],
            "input_manifest": copy.deepcopy(self._manifest_identity),
            "source_catalog_sha256": self._input_contract_identity["source_catalog_sha256"],
            "catalog_entry_sha256": content_sha256(original_catalog),
            "frozen_scene": _file(
                Path(entry["input_root"]) / "scene.json", entry["frozen_scene_sha256"]),
            "scene_input_identity_file": identity_file,
            "migration_receipt_file": migration_file,
            "scene_input_identity_sha256": identity["sha256"],
            "migration_receipt_sha256": migration["receipt_sha256"],
            "publication_receipt_sha256": entry["publication_receipt_sha256"],
            "original_scene_artifact_ref": reference["scene_artifact_ref"],
            "binding_key": reference["key"],
            "binding_manifest_sha256": reference["manifest_sha256"],
            "n_images": 32, "frame_set_hash": fs.frame_set_hash,
            "frame_identity": fs.cache_identity(),
            "source_frame_indices": list(fs.source_frame_indices),
            "pixel_sha256": list(fs.pixel_sha256),
            "public_episode_sha256": content_sha256(
                item.episode.model_dump(mode="json", exclude={"ground_truth"})),
            "real_labels_available": False, "reference_answer_visible": False,
            "archive_provenance": {
                name: entry[name] for name in sorted(_PROVENANCE_FIELDS & set(entry))
            },
        }
        receipt["receipt_sha256"] = content_sha256(receipt)
        json.dumps(receipt, allow_nan=False)
        return {
            "entry": entry, "identity": identity, "migration": migration,
            "resolver": resolver, "receipt": receipt,
        }

    def receipt(self, variant: Mapping, item: EpisodeItem) -> dict:
        """Return JSON-only evidence after checking actual inputs and copied bytes."""
        return self._transport(variant, item)["receipt"]

    def prepare_published(self, variant: Mapping, item: EpisodeItem) -> dict:
        """Load an unchanged sealed binding and return its runtime resolver.

        Keys: ``artifact_path``, ``artifact``, ``identity``, ``reference``,
        ``prepared``, ``artifact_relocation``, ``transport_receipt``.
        Only the runtime artifact has local dependency refs; the catalog,
        original logical identity, binding manifest, and panel are unchanged.
        """
        from skill3d.segmentation.prepared_binding_store import (
            PreparedBindingStoreError, load_prepared_binding_reference,
        )

        transport = self._transport(variant, item)
        entry, resolver = transport["entry"], transport["resolver"]
        try:
            prepared = load_prepared_binding_reference(
                entry["reference"], scene_input_identity=transport["identity"],
                migration_receipt=transport["migration"],
                scene_artifact=entry["scene_artifact"],
                migration_destination_root=entry["publication_root"],
            )
        except PreparedBindingStoreError as exc:
            raise PublishedFitSetInputError(f"published binding replay rejected: {exc}") from exc
        return {
            "artifact_path": resolver.artifact_path,
            "artifact": resolver.resolved_artifact(),
            "identity": transport["identity"],
            "reference": copy.deepcopy(entry["reference"]),
            "prepared": prepared,
            "artifact_relocation": resolver,
            "transport_receipt": transport["receipt"],
        }


__all__ = [
    "PUBLISHED_INPUT_SCHEMA", "FULLQA_INPUT_SCHEMA", "PublishedFitSetInputError",
    "PublishedFitSetInputProvider",
]
