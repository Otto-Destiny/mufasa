"""Offline, pinned authority snapshots.

This module performs no network access.  Acquisition is a separate build-plane
step; resolution accepts only files whose hashes match a local manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .contracts import ContractError, EntityType, parse_json_list


@dataclass(frozen=True)
class AuthorityRecord:
    authority: str
    external_id: str
    preferred_label: str
    entity_type: EntityType
    aliases: tuple[str, ...]
    rank: str | None = None
    parent_external_id: str | None = None
    country_code: str | None = None
    attributes_json: str = "{}"

    def __post_init__(self) -> None:
        authority = str(self.authority).strip().upper()
        external_id = str(self.external_id).strip()
        label = str(self.preferred_label).strip()
        if not authority or not external_id or not label:
            raise ContractError("authority records require nonblank authority, external_id, and preferred_label")
        aliases = tuple(sorted(set(str(item).strip() for item in self.aliases)))
        if any(not item for item in aliases):
            raise ContractError(f"authority record {authority}:{external_id} contains a blank alias")
        try:
            attributes = json.loads(self.attributes_json or "{}")
        except json.JSONDecodeError as exc:
            raise ContractError(f"authority record {authority}:{external_id} attributes_json is invalid") from exc
        if not isinstance(attributes, Mapping):
            raise ContractError(f"authority record {authority}:{external_id} attributes_json must be an object")
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "external_id", external_id)
        object.__setattr__(self, "preferred_label", label)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(
            self,
            "attributes_json",
            json.dumps(attributes, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )


@dataclass(frozen=True)
class AuthoritySnapshot:
    version: str
    manifest_hash: str
    records: tuple[AuthorityRecord, ...]
    trusted_crosswalks: tuple[tuple[str, str, str, str], ...]
    licences: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not str(self.version).strip():
            raise ContractError("authority snapshot version is blank")
        identifiers = [(item.authority, item.external_id) for item in self.records]
        if len(identifiers) != len(set(identifiers)):
            raise ContractError("authority snapshot contains duplicate authority/external_id pairs")
        by_id = {(item.authority, item.external_id): item for item in self.records}

        authorities = {item.authority for item in self.records}
        licence_rows = [(str(authority).strip().upper(), str(licence).strip()) for authority, licence in self.licences]
        if any(not authority or not licence for authority, licence in licence_rows):
            raise ContractError("authority snapshot licences require nonblank authority and licence")
        if len(licence_rows) != len({authority for authority, _licence in licence_rows}):
            raise ContractError("authority snapshot must contain exactly one licence row per authority")
        if {authority for authority, _licence in licence_rows} != authorities:
            raise ContractError("authority snapshot licence authorities must exactly match record authorities")

        seen_edges: set[tuple[tuple[str, str], tuple[str, str]]] = set()
        directed: dict[tuple[str, str, str], str] = {}
        normalized_crosswalks = []
        for raw in self.trusted_crosswalks:
            if len(raw) != 4:
                raise ContractError("trusted crosswalk rows require four fields")
            left = (str(raw[0]).strip().upper(), str(raw[1]).strip())
            right = (str(raw[2]).strip().upper(), str(raw[3]).strip())
            if not all((*left, *right)):
                raise ContractError("trusted crosswalk endpoints must be nonblank")
            if left == right or left[0] == right[0]:
                raise ContractError("trusted crosswalks must connect distinct identifiers from different authorities")
            if left not in by_id or right not in by_id:
                raise ContractError(f"trusted crosswalk endpoint is absent from records: {left} <-> {right}")
            if by_id[left].entity_type != by_id[right].entity_type:
                raise ContractError(f"trusted crosswalk endpoints have incompatible types: {left} <-> {right}")
            edge = tuple(sorted((left, right)))
            if edge in seen_edges:
                raise ContractError(f"duplicate/reversed trusted crosswalk: {left} <-> {right}")
            seen_edges.add(edge)
            for source, target in ((left, right), (right, left)):
                key = (source[0], source[1], target[0])
                previous = directed.setdefault(key, target[1])
                if previous != target[1]:
                    raise ContractError(
                        f"conflicting trusted crosswalk targets for {source} into {target[0]}"
                    )
            normalized_crosswalks.append((left[0], left[1], right[0], right[1]))
        object.__setattr__(self, "records", tuple(sorted(self.records, key=lambda item: (item.authority, item.external_id))))
        object.__setattr__(self, "trusted_crosswalks", tuple(sorted(normalized_crosswalks)))
        object.__setattr__(self, "licences", tuple(sorted(licence_rows)))

    @classmethod
    def empty(cls) -> "AuthoritySnapshot":
        payload = b'{"version":"none"}'
        return cls("none", hashlib.sha256(payload).hexdigest(), (), (), ())

    @cached_property
    def by_identifier(self) -> dict[tuple[str, str], AuthorityRecord]:
        return {(r.authority.upper(), r.external_id): r for r in self.records}

    @cached_property
    def crosswalk_set(self) -> frozenset[tuple[str, str, str, str]]:
        values = set(self.trusted_crosswalks)
        values.update((right_authority, right_id, left_authority, left_id) for left_authority, left_id, right_authority, right_id in self.trusted_crosswalks)
        return frozenset(values)

    def get(self, authority: str, external_id: str) -> AuthorityRecord | None:
        return self.by_identifier.get((authority.upper(), external_id))

    def are_crosswalked(self, left_authority: str, left_id: str, right_authority: str, right_id: str) -> bool:
        left = (left_authority.upper(), left_id, right_authority.upper(), right_id)
        right = (right_authority.upper(), right_id, left_authority.upper(), left_id)
        return left in self.crosswalk_set or right in self.crosswalk_set


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_file(root: Path, entry: Mapping[str, Any]) -> Path:
    relative = Path(str(entry["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ContractError(f"authority manifest path must be relative: {relative}")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents and path != root.resolve():
        raise ContractError(f"authority file escapes snapshot root: {relative}")
    if not path.is_file():
        raise ContractError(f"authority file is missing: {path}")
    actual = _sha256(path)
    expected = str(entry["sha256"]).lower()
    if actual != expected:
        raise ContractError(f"authority file hash mismatch for {relative}: {actual} != {expected}")
    return path


def load_authority_snapshot(path: str | Path | None) -> AuthoritySnapshot:
    if path is None:
        return AuthoritySnapshot.empty()
    root = Path(path)
    manifest_path = root / "authority_manifest.json" if root.is_dir() else root
    root = manifest_path.parent
    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid authority manifest: {exc}") from exc
    if manifest.get("network_required") not in (None, False):
        raise ContractError("authority snapshot declares a runtime network dependency")
    version = str(manifest.get("version") or "").strip()
    if not version:
        raise ContractError("authority manifest version is required")
    records: list[AuthorityRecord] = []
    for entry in manifest.get("record_files", []):
        file_path = _verified_file(root, entry)
        frame = pd.read_parquet(file_path)
        required = {"authority", "external_id", "preferred_label", "entity_type", "aliases_json"}
        missing = required - set(frame.columns)
        if missing:
            raise ContractError(f"authority file {file_path.name} missing columns {sorted(missing)}")
        for row in frame.to_dict("records"):
            aliases = tuple(sorted(set(str(item).strip() for item in parse_json_list(row["aliases_json"], "aliases_json") if str(item).strip())))
            records.append(
                AuthorityRecord(
                    authority=str(row["authority"]).strip().upper(),
                    external_id=str(row["external_id"]).strip(),
                    preferred_label=str(row["preferred_label"]).strip(),
                    entity_type=EntityType(str(row["entity_type"])),
                    aliases=aliases,
                    rank=_optional(row.get("rank")),
                    parent_external_id=_optional(row.get("parent_external_id")),
                    country_code=_optional(row.get("country_code")),
                    attributes_json=str(row.get("attributes_json") or "{}"),
                )
            )
    crosswalks: list[tuple[str, str, str, str]] = []
    for item in manifest.get("trusted_crosswalks", []):
        if not isinstance(item, Mapping):
            raise ContractError("trusted_crosswalks entries must be objects")
        crosswalks.append(
            (
                str(item["left_authority"]).upper(),
                str(item["left_id"]),
                str(item["right_authority"]).upper(),
                str(item["right_id"]),
            )
        )
    licences = tuple(
        sorted((str(item["authority"]).upper(), str(item["licence"])) for item in manifest.get("licences", []))
    )
    if records and not licences:
        raise ContractError("non-empty authority snapshots must record licences")
    return AuthoritySnapshot(
        version=version,
        manifest_hash=hashlib.sha256(raw).hexdigest(),
        records=tuple(sorted(records, key=lambda r: (r.authority, r.external_id))),
        trusted_crosswalks=tuple(crosswalks),
        licences=licences,
    )


def _optional(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip()
    return text or None


def write_authority_manifest_template(path: str | Path) -> None:
    """Write a deliberately empty, offline manifest template."""

    payload = {
        "version": "replace-with-pinned-snapshot-version",
        "network_required": False,
        "record_files": [],
        "trusted_crosswalks": [],
        "licences": [],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
