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
    ``transport_locator`` accepts an approved ``PublishedTransportLocator`` or
    its descriptor. Physical lookups are separate from every catalog locator;
    the manifest itself may also be an exact pinned file mapping.
    """

    _reuse_verified_publications = False

    def __init__(
        self, manifest_path: str | Path, max_scenes: int = 2, *,
        transport_locator=None,
    ) -> None:
        from skill3d.infra.m5_identity import content_sha256
        from skill3d.storage.published_transport import PublishedTransportLocator

        self._reuse_verified_publications = False
        _require(type(max_scenes) is int and max_scenes > 0,
                 "scene cache size must be a positive integer")
        self.transport_locator = (
            PublishedTransportLocator(transport_locator)
            if isinstance(transport_locator, Mapping) else transport_locator
        )
        _require(self.transport_locator is None or isinstance(
            self.transport_locator, PublishedTransportLocator),
            "transport_locator requires an approved physical transport descriptor")
        self.manifest_path = self._physical_file(
            Path(manifest_path).expanduser().absolute(), None)
        self._manifest_transport_sources = (
            self.transport_locator.sources_for_file(self.manifest_path)
            if self.transport_locator is not None else ()
        )
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
        self._source_catalog_file = manifest.get("source_catalog_file")
        source_catalog = None
        if self.schema == FULLQA_INPUT_SCHEMA:
            source_catalog, _ = _read_anchored(
                self._physical_file(
                    manifest["source_catalog_file"], manifest["source_catalog_sha256"]),
                manifest["source_catalog_sha256"])
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
        scene_qa_ids = {}
        for qa_id, entry in self._entries.items():
            scene_qa_ids.setdefault(entry["scene_id"], []).append(qa_id)
        self._scene_qa_ids = {scene: tuple(ids) for scene, ids in scene_qa_ids.items()}
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
        self._publication_files = OrderedDict()
        self._scene_checks = {}
        self._scene_transport_scopes = {}
        self.cache_lock = threading.RLock()

    @property
    def entries(self) -> dict[str, dict]:
        return {qa_id: _public_entry(entry) for qa_id, entry in self._entries.items()}

    @property
    def input_contract_identity(self) -> dict:
        return copy.deepcopy(self._input_contract_identity)

    def public_entry(self, variant):
        """Project one catalog entry without copying the entire QA catalog."""
        return _public_entry(self._entry(variant))

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["cache"] = OrderedDict()
        state["_publication_files"] = OrderedDict()
        state["_scene_checks"] = {}
        state["_scene_transport_scopes"] = {}
        state.pop("cache_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.cache = OrderedDict()
        self._publication_files = OrderedDict()
        self._scene_checks = {}
        self._scene_transport_scopes = {}
        self.cache_lock = threading.RLock()

    def _add_spawn_migration_files(self, files, migration, source_root):
        if isinstance(migration, (str, Path)):
            files.add(str(migration))
            migration, _ = self._publication_file(migration, None)
        for record in (migration["artifact"], *migration["dependencies"].values()):
            files.add(str(Path(source_root) / record["target_ref"]))

    def _spawn_transport_sources(self, scene_id):
        files = set(self._manifest_transport_sources)
        if self._source_catalog_file is not None:
            files.add(self._source_catalog_file)
        roots, trees = set(), set()
        for qa_id in self._scene_qa_ids[scene_id]:
            entry = self._entries[qa_id]
            files.update((entry["scene_input_identity_file"],
                          entry["migration_receipt_file"], entry["scene_artifact"]))
            reference = entry["reference"]
            roots.update((entry["input_root"], entry["publication_root"],
                          reference["store_root"]))
            trees.add(entry["input_root"])
            key = reference["key"]
            trees.add(str(Path(reference["store_root"]) / "entries" / key[:2] / key))
            validation, _ = self._publication_file(
                entry["migration_receipt_file"], entry["migration_receipt_file_sha256"])
            try:
                migration = validation["variants"][entry["variant"]]["migration_receipts"][qa_id]
                self._add_spawn_migration_files(files, migration, entry["publication_root"])
            except (KeyError, TypeError) as exc:
                raise PublishedFitSetInputError("catalog migration receipt is missing") from exc
        return files, roots, trees

    def _spawn_state(self, entry):
        state = PublishedFitSetInputProvider.__getstate__(self)
        if self._reuse_verified_publications:
            scene_id = entry["scene_id"]
            ids = self._scene_qa_ids[scene_id]
            scope = self.scene_scopes.get(scene_id)
            _require(scope is None or set(ids) == set(scope["qa_ids"]),
                     "spawn replay requires the complete original scene QA group")
            state["_entries"] = copy.deepcopy({qa: self._entries[qa] for qa in ids})
            state["scene_scopes"] = (
                {scene_id: copy.deepcopy(scope)} if scope is not None else {})
            state["_scene_qa_ids"] = {scene_id: ids}
            # Keep global manifest/catalog/entry hashes, not hashes of the subset.
            state["_spawn_input_scene"] = scene_id
            if scene_id not in self._scene_transport_scopes:
                files, roots, trees = self._spawn_transport_sources(scene_id)
                state_locator = self.transport_locator.for_sources(
                    files=files, roots=roots, trees=trees)
                self._scene_transport_scopes[scene_id] = state_locator
            state["transport_locator"] = self._scene_transport_scopes[scene_id]
        return state

    def for_spawned_episode(self, variant):
        """Cold clone; verified mode retains exactly the complete current scene."""
        entry = self._entry(variant)
        with self.cache_lock:
            clone = type(self).__new__(type(self))
            clone.__setstate__(self._spawn_state(entry))
            return clone

    def reuse_verified_publications(self):
        """Use production/transport verification for this immutable input store.

        Only an approved exact physical descriptor can enable this path. The
        runtime holds the scene lease through solving; it verifies downloads.
        """
        _require(self.transport_locator is not None,
                 "verified replay requires an approved physical transport descriptor")
        if not self._reuse_verified_publications:
            self.cache.clear()
            self._scene_checks.clear()
        self._reuse_verified_publications = True

    def _publication_file(self, path, sha256):
        if not self._reuse_verified_publications:
            return _read_anchored(self._physical_file(path, sha256), sha256)
        physical = self._physical_file(path, sha256)
        key = (str(physical), sha256)
        with self.cache_lock:
            if key not in self._publication_files:
                self._publication_files[key] = (
                    scene_compute._read_json(physical),
                    {"path": str(physical), "sha256": sha256, "size": physical.stat().st_size})
                while len(self._publication_files) > 2 * self.max_scenes + 2:
                    self._publication_files.popitem(last=False)
            self._publication_files.move_to_end(key)
            return self._publication_files[key]

    def _publication_identity(self, path, sha256):
        physical = self._physical_file(path, sha256)
        if not self._reuse_verified_publications:
            return _file(physical, sha256)
        return {"path": str(physical), "sha256": sha256, "size": physical.stat().st_size}

    def _entry(self, variant: Mapping) -> dict:
        if not self._reuse_verified_publications:
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

    def _physical_file(self, source: str | Path, sha256: str | None) -> Path:
        from skill3d.storage.relocation import RelocationError

        try:
            lookup = (self.transport_locator.verified_file
                      if self._reuse_verified_publications else self.transport_locator.file
                      ) if self.transport_locator is not None else None
            return (lookup(source, sha256)
                    if self.transport_locator is not None else Path(source))
        except RelocationError as exc:
            raise PublishedFitSetInputError(f"published physical file rejected: {exc}") from exc

    def _physical_root(self, source: str | Path) -> Path:
        from skill3d.storage.relocation import RelocationError

        try:
            lookup = (self.transport_locator.verified_root
                      if self._reuse_verified_publications else self.transport_locator.root
                      ) if self.transport_locator is not None else None
            return (lookup(source)
                    if self.transport_locator is not None else Path(source))
        except RelocationError as exc:
            raise PublishedFitSetInputError(f"published physical root rejected: {exc}") from exc

    def _rebase_migration(self, store, key, migration, identity, source_root):
        """Validate the old receipt against a new full, binding-anchored receipt."""
        from skill3d.infra.m5_identity import content_sha256
        from skill3d.storage.relocation import RelocationError, read_relocation_receipt

        try:
            if self._reuse_verified_publications:
                original = (scene_compute._read_json(self._physical_file(migration, None))
                            if isinstance(migration, (str, Path))
                            else copy.deepcopy(migration))
            else:
                original = read_relocation_receipt(
                    self._physical_file(migration, None)
                    if isinstance(migration, (str, Path)) else migration)
            _require(original.get("destination_root") == source_root,
                     "original migration destination changed")
            root = self._physical_root(source_root)
            records = {"artifact": original["artifact"], **original["dependencies"]}
            refs = {}
            for name, record in records.items():
                lookup = (self.transport_locator.verified_member
                          if self._reuse_verified_publications
                          else self.transport_locator.member)
                path = lookup(
                    source_root, record["target_ref"], record["identity"])
                refs[name] = path.relative_to(root).as_posix()
            if self._reuse_verified_publications:
                fresh = copy.deepcopy(original)
                fresh["destination_root"] = str(root)
                fresh["artifact"]["target_ref"] = refs.pop("artifact")
                for name, ref in refs.items():
                    fresh["dependencies"][name]["target_ref"] = ref
                fresh["receipt_sha256"] = content_sha256({
                    k: v for k, v in fresh.items() if k != "receipt_sha256"})
                return fresh, root
            fresh = store.create_migration_receipt(
                key, scene_input_identity=identity, destination_root=root,
                scene_artifact_ref=refs.pop("artifact"), dependency_refs=refs)
            # Location alone may differ. Replay all original record fields,
            # including the full logical identity and immutable manifest anchor.
            replay = copy.deepcopy(fresh)
            replay["destination_root"] = source_root
            replay["artifact"]["target_ref"] = original["artifact"]["target_ref"]
            for name in replay["dependencies"]:
                replay["dependencies"][name]["target_ref"] = (
                    original["dependencies"][name]["target_ref"])

            replay["receipt_sha256"] = content_sha256({
                k: v for k, v in replay.items() if k != "receipt_sha256"})
            _require(replay == original,
                     "original migration full identity/member/binding anchor changed")
            return fresh, root
        except (RelocationError, KeyError, TypeError, ValueError) as exc:
            raise PublishedFitSetInputError(f"published migration relocation rejected: {exc}") from exc

    def _scene(self, entry: dict):
        from skill3d.infra.m5_identity import file_identity

        root = self._physical_root(entry["input_root"])
        key = (str(root), entry["frozen_scene_sha256"])
        with self.cache_lock:
            if self._reuse_verified_publications:
                if key not in self.cache:
                    from skill3d.evaluation.verified_publication_v11 import load_verified_scene
                    scope = self.scene_scopes.get(entry["scene_id"])
                    loaded = load_verified_scene(
                        root, question_cap=len(scope["qa_ids"]) if scope is not None else 8)
                    if scope is not None:
                        _require(loaded.manifest["qa_ids"] == scope["qa_ids"]
                                 and loaded.manifest["split"] == scope["split"],
                                 "fullQA frozen scene question/split scope changed")
                    self.cache[key] = loaded
                    while len(self.cache) > self.max_scenes:
                        self.cache.popitem(last=False)
                self.cache.move_to_end(key)
                return self.cache[key]
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
        if not self._reuse_verified_publications:
            try:
                actual_pixels = with_pixel_identity(fs, item.pixels).pixel_sha256
            except ValueError as exc:
                raise PublishedFitSetInputError("published frozen RGB pixel identity changed") from exc
            _require(actual_pixels == fs.pixel_sha256, "published frozen RGB pixel identity changed")

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

    def frozen_frame_bytes(self, variant):
        """Reuse the original PNG encoding already decoded for this scene."""
        if not self._reuse_verified_publications:
            return None
        return self._scene(self._entry(variant)).png_bytes

    def _transport(self, variant: Mapping, item: EpisodeItem) -> dict:
        from skill3d.infra.m5_identity import content_sha256
        from skill3d.segmentation.prepared_binding_store import (
            PreparedBindingStore, PreparedBindingStoreError,
        )

        entry = self._entry(variant)
        self._validate_item(variant, entry, item)
        identity, identity_file = self._publication_file(
            entry["scene_input_identity_file"],
            entry["scene_input_identity_file_sha256"],
        )
        if self._reuse_verified_publications:
            # Shared parsed publication JSON stays private; episode consumers
            # receive their own logical identity and receipt records.
            identity, identity_file = copy.deepcopy(identity), copy.deepcopy(identity_file)
        validation, migration_file = self._publication_file(
            entry["migration_receipt_file"],
            entry["migration_receipt_file_sha256"])
        if self._reuse_verified_publications:
            migration_file = copy.deepcopy(migration_file)
        try:
            migration = validation["variants"][entry["variant"]]["migration_receipts"][
                entry["qa_id"]]
        except (KeyError, TypeError) as exc:
            raise PublishedFitSetInputError("catalog migration receipt is missing") from exc
        reference = entry["reference"]
        store_root = self._physical_root(reference["store_root"])
        store = PreparedBindingStore(store_root)
        _require(identity.get("artifact_path") == reference["scene_artifact_ref"]
                 and identity.get("sha256") == reference["scene_input_identity_sha256"],
                 "catalog logical scene identity changed")
        try:
            if not self._reuse_verified_publications:
                store.validate_reference(reference)
            original_migration = copy.deepcopy(migration)
            destination = Path(entry["publication_root"])
            artifact_path = Path(entry["scene_artifact"])
            if self.transport_locator is not None:
                _require(artifact_path == destination / migration["artifact"]["target_ref"],
                         "catalog published artifact location changed")
                migration, destination = self._rebase_migration(
                    store, reference["key"], migration, identity, entry["publication_root"])
                artifact_path = destination / migration["artifact"]["target_ref"]
            if self._reuse_verified_publications:
                from skill3d.storage.relocation import ProducerVerifiedRelocation
                resolver = ProducerVerifiedRelocation(migration)
            else:
                resolver = store.validate_migration_receipt(
                    reference["key"], migration,
                    scene_artifact=str(artifact_path), scene_input_identity=identity,
                    destination_root=str(destination),
                )
        except PreparedBindingStoreError as exc:
            raise PublishedFitSetInputError(f"published binding replay rejected: {exc}") from exc
        if not self._reuse_verified_publications:
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
            "frozen_scene": (
                self._publication_identity(
                    Path(entry["input_root"]) / "scene.json", entry["frozen_scene_sha256"])
                if self._reuse_verified_publications else _file(
                    self._physical_root(entry["input_root"]) / "scene.json",
                    entry["frozen_scene_sha256"])),
            "scene_input_identity_file": identity_file,
            "migration_receipt_file": migration_file,
            "scene_input_identity_sha256": identity["sha256"],
            "migration_receipt_sha256": original_migration["receipt_sha256"],
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
        if self.transport_locator is not None:
            receipt["physical_transport"] = {
                "descriptor_sha256": self.transport_locator.descriptor_sha256,
                "store_root": str(store_root), "geometry_root": str(destination),
                "migration_receipt_sha256": migration["receipt_sha256"],
            }
        receipt["receipt_sha256"] = content_sha256(receipt)
        json.dumps(receipt, allow_nan=False)
        return {
            "entry": entry, "identity": identity, "migration": migration,
            "resolver": resolver, "receipt": receipt,
            "original_migration": original_migration,
            "store_root": store_root, "destination_root": destination,
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
            if self._reuse_verified_publications:
                from skill3d.segmentation.prepared_binding_store import PreparedBindingStore
                prepared = PreparedBindingStore(transport["store_root"]).load_verified(
                    entry["reference"]["key"])
            else:
                prepared = load_prepared_binding_reference(
                    entry["reference"], scene_input_identity=transport["identity"],
                    migration_receipt=transport["migration"],
                    scene_artifact=str(resolver.artifact_path),
                    migration_destination_root=str(transport["destination_root"]),
                    physical_store_root=(transport["store_root"]
                                         if self.transport_locator is not None else None),
                )
        except PreparedBindingStoreError as exc:
            raise PublishedFitSetInputError(f"published binding replay rejected: {exc}") from exc
        result = {
            "artifact_path": resolver.artifact_path,
            "artifact": resolver.resolved_artifact(),
            "identity": transport["identity"],
            "reference": copy.deepcopy(entry["reference"]),
            "prepared": prepared,
            "artifact_relocation": resolver,
            "transport_receipt": transport["receipt"],
        }
        if self._reuse_verified_publications:
            from skill3d.evaluation.verified_publication_v11 import ProducerVerifiedBindingContext
            result["binding_derivation_context"] = ProducerVerifiedBindingContext(
                prepared, copy.deepcopy(entry["reference"]["request_identity"]))
        return result


__all__ = [
    "PUBLISHED_INPUT_SCHEMA", "FULLQA_INPUT_SCHEMA", "PublishedFitSetInputError",
    "PublishedFitSetInputProvider",
]
