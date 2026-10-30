"""Caller-approved physical locations, separate from published logical inputs.

Descriptors use ``published-physical-transport-v1`` with exact absolute source
keys. ``files`` rows contain ``path`` and ``identity`` (SHA256/size); ``roots``
rows contain ``path``, ``members`` (relative file names to identities), and
``members_sha256``. The caller pins these descriptors, not hashes discovered
from untrusted copied files. Root members describe the entire physical copy.
No ancestor/prefix lookup, source-host access, or symlink resolution is allowed.
"""

from __future__ import annotations

from bisect import bisect_left
from copy import deepcopy
from pathlib import Path
from typing import Mapping

from skill3d.infra.m5_identity import content_sha256
from skill3d.storage.relocation import (
    RelocationError, _check_file_identity, _identity, _no_symlinks, _ref, _root,
    _sha256, _target,
)

PUBLISHED_TRANSPORT_SCHEMA = "published-physical-transport-v1"


class PublishedTransportLocator:
    """Validate exact declared copies on every lookup; never fall back on error."""

    def __init__(self, descriptor: Mapping) -> None:
        self._descriptor = deepcopy(dict(descriptor))
        if set(self._descriptor) != {"schema_version", "files", "roots"} or (
            self._descriptor["schema_version"] != PUBLISHED_TRANSPORT_SCHEMA
        ):
            raise RelocationError("unsupported published physical transport descriptor")
        for section, fields in (
            ("files", {"path", "identity"}),
            ("roots", {"path", "members", "members_sha256"}),
        ):
            rows = self._descriptor[section]
            if not isinstance(rows, dict):
                raise RelocationError("physical transport mappings must be objects")
            for source, row in rows.items():
                _ref(source, absolute=True)
                if not isinstance(row, dict) or set(row) != fields:
                    raise RelocationError("physical transport requires pinned file/member identities")
                _ref(row["path"], absolute=True)
                if section == "files":
                    _check_file_identity(row["identity"])
                else:
                    members = row["members"]
                    if not isinstance(members, dict) or not members:
                        raise RelocationError("physical root requires a complete member closure")
                    if _sha256(row["members_sha256"]) != content_sha256(members):
                        raise RelocationError("physical root member anchor changed")
                    for name, identity in members.items():
                        if Path(_ref(name)).is_absolute():
                            raise RelocationError("physical root members must be root-relative")
                        _check_file_identity(identity)
        self._descriptor_sha256 = content_sha256(self._descriptor)
        self._scoped = False
        self._verified_root_paths = {
            source: row["path"] for source, row in self._descriptor["roots"].items()
        }
        self._verified_files = {}
        for source, row in self._descriptor["roots"].items():
            for name, identity in row["members"].items():
                old = str(Path(source) / name)
                candidate = {"path": str(Path(row["path"]) / name), "identity": identity}
                previous = self._verified_files.get(old)
                if previous is not None and previous != candidate:
                    raise RelocationError("conflicting exact verified member mappings")
                self._verified_files[old] = candidate
        self._verified_files.update(self._descriptor["files"])
        self._verified_file_sources = tuple(sorted(self._verified_files))
        self._explicit_file_sources = frozenset(self._descriptor["files"])
        physical_counts = {}
        for row in self._descriptor["files"].values():
            physical_counts[row["path"]] = physical_counts.get(row["path"], 0) + 1
        self._ambiguous_explicit_paths = frozenset(
            path for path, count in physical_counts.items() if count > 1)

    @property
    def descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def sources_for_file(self, path: str | Path) -> tuple[str, ...]:
        """Return existing exact source aliases, not newly inferred mappings."""
        path = _ref(str(path), absolute=True)
        return tuple(source for source, row in self._verified_files.items()
                     if row["path"] == path)

    def explicit_sources_for_file(self, path: str | Path) -> tuple[str, ...]:
        """Keep reverse lookup restricted to the original explicit files rows."""
        path = _ref(str(path), absolute=True)
        return tuple(source for source in self._explicit_file_sources
                     if self._verified_files[source]["path"] == path)

    def has_ambiguous_explicit_source(self, path: str | Path) -> bool:
        """An original ambiguity remains invalid after pruning other sources."""
        return _ref(str(path), absolute=True) in self._ambiguous_explicit_paths

    def for_sources(self, *, files=(), roots=(), trees=()):
        """Select verified tables while retaining the original descriptor anchor.

        ``files`` and ``roots`` are exact logical sources. ``trees`` selects
        already indexed file mappings below logical directories; it does not
        authorize prefix root lookup. A subset is not a new descriptor or a
        complete root closure, so strict root/member audits are unavailable.
        ``_descriptor`` is absent on the selected capability. Consumers must
        reuse this locator, not construct or hash a replacement descriptor.
        Missing selections fail closed, including on a previously scoped copy.
        """
        selected = {}
        for source in files:
            source = _ref(str(source), absolute=True)
            row = self._verified_files.get(source)
            if row is None:
                raise RelocationError("verified replay requires an explicit file mapping")
            selected[source] = row
        for source in trees:
            prefix = _ref(str(source), absolute=True) + "/"
            start = bisect_left(self._verified_file_sources, prefix)
            # '/' sorts immediately before '0'; directory siblings are excluded.
            stop = bisect_left(self._verified_file_sources, prefix[:-1] + "0")
            if start == stop:
                raise RelocationError("verified replay requires explicit tree member mappings")
            for name in self._verified_file_sources[start:stop]:
                selected[name] = self._verified_files[name]
        root_paths = {}
        for source in roots:
            source = _ref(str(source), absolute=True)
            path = self._verified_root_paths.get(source)
            if path is None:
                raise RelocationError("verified replay requires an explicit root mapping")
            root_paths[source] = path
        clone = type(self).__new__(type(self))
        clone._descriptor = None
        clone._descriptor_sha256 = self._descriptor_sha256
        clone._scoped = True
        clone._verified_root_paths = root_paths
        clone._verified_files = selected
        clone._verified_file_sources = tuple(sorted(selected))
        clone._explicit_file_sources = self._explicit_file_sources.intersection(selected)
        clone._ambiguous_explicit_paths = self._ambiguous_explicit_paths.intersection(
            row["path"] for row in selected.values())
        return clone

    def file(self, source: str | Path, expected_sha256: str | None = None) -> Path:
        source = _ref(str(source), absolute=True)
        row = (self._verified_files if self._scoped
               else self._descriptor["files"]).get(source)
        if row is None:
            if self._scoped:
                raise RelocationError("file is outside the selected transport scope")
            return Path(source)
        path = Path(row["path"])
        _no_symlinks(path)
        if (expected_sha256 is not None and row["identity"]["sha256"] != expected_sha256) or (
            not path.is_file() or _identity(path) != row["identity"]
        ):
            raise RelocationError("physical transport file identity changed")
        return path

    def root(self, source: str | Path) -> Path:
        if self._scoped:
            raise RelocationError("scoped transport cannot audit a complete root closure")
        source = _ref(str(source), absolute=True)
        row = self._descriptor["roots"].get(source)
        if row is None:
            return _root(source)
        root = _root(row["path"])
        members = row["members"]
        actual = set()
        for path in root.rglob("*"):
            _no_symlinks(path)
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())
            elif not path.is_dir():
                raise RelocationError("unsupported physical root member")
        if actual != set(members):
            raise RelocationError("physical root member closure changed")
        for name, identity in members.items():
            if _identity(_target(root, name)) != identity:
                raise RelocationError(f"physical root member identity changed: {name}")
        return root

    def member(self, source_root: str | Path, name: str, identity: Mapping) -> Path:
        """Resolve one explicitly named member, with optional exact file mapping."""
        if self._scoped:
            raise RelocationError("scoped transport requires verified member lookup")
        source_root = _ref(str(source_root), absolute=True)
        row = self._descriptor["roots"].get(source_root)
        root = _root(row["path"] if row is not None else source_root)
        old_path = Path(source_root) / _ref(name)
        if Path(name).is_absolute():
            raise RelocationError("physical member must be root-relative")
        if str(old_path) in self._descriptor["files"]:
            path = self.file(old_path, identity["sha256"])
            if not path.is_relative_to(root):
                raise RelocationError("physical member escapes its approved root")
        else:
            path = _target(root, name)
        if _identity(path) != dict(identity):
            raise RelocationError("physical member differs from source receipt")
        return path

    def verified_file(self, source, expected_sha256=None):
        """Resolve producer/transport-verified immutable bytes without rehashing."""
        source = _ref(str(source), absolute=True)
        row = self._verified_files.get(source)
        if row is None:
            raise RelocationError("verified replay requires an explicit file mapping")
        if expected_sha256 is not None and row["identity"]["sha256"] != expected_sha256:
            raise RelocationError("verified file differs from its publication anchor")
        path = Path(row["path"])
        _no_symlinks(path)
        if not path.is_file():
            raise RelocationError("verified publication file is missing")
        return path

    def verified_root(self, source):
        """Resolve an exact approved root without scanning all of its members."""
        source = _ref(str(source), absolute=True)
        path = self._verified_root_paths.get(source)
        if path is None:
            raise RelocationError("verified replay requires an explicit root mapping")
        return _root(path)

    def verified_member(self, source_root, name, identity):
        source_root = _ref(str(source_root), absolute=True)
        name = _ref(name)
        if Path(name).is_absolute():
            raise RelocationError("physical member must be root-relative")
        source = str(Path(source_root) / name)
        row = self._verified_files.get(source)
        if row is None or row["identity"] != dict(identity):
            raise RelocationError("verified member differs from its publication anchor")
        root = self.verified_root(source_root)
        path = self.verified_file(source, identity["sha256"])
        if not path.is_relative_to(root):
            raise RelocationError("verified member escapes its approved root")
        return path


__all__ = ["PUBLISHED_TRANSPORT_SCHEMA", "PublishedTransportLocator"]
