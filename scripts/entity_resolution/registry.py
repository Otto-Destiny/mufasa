"""Immutable entity registry snapshots and reviewed lineage operations."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Iterable, Mapping, Sequence

from .authorities import AuthoritySnapshot
from .contracts import (
    AliasTrust,
    AuthorityLink,
    CanonicalAlias,
    CanonicalEntity,
    ContractError,
    EntityInstance,
    EntityRedirect,
    EntityRelation,
    EntityType,
    EventType,
    LifecycleStatus,
    ResolutionConstraint,
    ResolutionEvent,
    ConstraintType,
)


_CONCEPT_NAMESPACE = uuid.UUID("634d4b92-f659-5a3b-b4dd-d01b5cb9de75")
_INSTANCE_NAMESPACE = uuid.UUID("1816853f-7352-5bc9-a30a-35aa969223f3")
_EVENT_NAMESPACE = uuid.UUID("280490b3-1fbc-5bcf-8492-0a47a7db1359")
_ALIAS_NAMESPACE = uuid.UUID("926afae9-33a6-5e6b-aa80-6c8bd62c6397")
_LINK_NAMESPACE = uuid.UUID("8d17de14-7227-532e-97fb-b50887281631")


def stable_concept_id(seed_mention_id: str) -> str:
    return f"CON-{uuid.uuid5(_CONCEPT_NAMESPACE, seed_mention_id).hex}"


def stable_instance_id(seed_mention_id: str, paper_id: str, context_id: str) -> str:
    return f"INS-{uuid.uuid5(_INSTANCE_NAMESPACE, f'{paper_id}\x1f{context_id}\x1f{seed_mention_id}').hex}"


def stable_alias_id(concept_id: str, normalized_key: str, entity_type: EntityType, source: str) -> str:
    value = f"{concept_id}\x1f{entity_type.value}\x1f{normalized_key}\x1f{source}"
    return f"ALS-{uuid.uuid5(_ALIAS_NAMESPACE, value).hex}"


def stable_authority_link_id(concept_id: str, authority: str, external_id: str) -> str:
    value = f"{concept_id}\x1f{authority.upper()}\x1f{external_id}"
    return f"AUT-{uuid.uuid5(_LINK_NAMESPACE, value).hex}"


def stable_event_id(event_type: EventType, run_id: str, subject_id: str, object_id: str | None = None) -> str:
    value = f"{event_type.value}\x1f{run_id}\x1f{subject_id}\x1f{object_id or ''}"
    return f"EVT-{uuid.uuid5(_EVENT_NAMESPACE, value).hex}"


def proposal_id(kind: str, members: Sequence[str], policy_version: str) -> str:
    payload = json.dumps(
        {"kind": kind, "members": sorted(members), "policy_version": policy_version},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"PRP-{hashlib.sha256(payload.encode()).hexdigest()[:32]}"


@dataclass(frozen=True)
class RegistrySnapshot:
    version: str
    canonical_entities: tuple[CanonicalEntity, ...] = ()
    entity_instances: tuple[EntityInstance, ...] = ()
    canonical_aliases: tuple[CanonicalAlias, ...] = ()
    authority_links: tuple[AuthorityLink, ...] = ()
    entity_relations: tuple[EntityRelation, ...] = ()
    redirects: tuple[EntityRedirect, ...] = ()
    constraints: tuple[ResolutionConstraint, ...] = ()
    events: tuple[ResolutionEvent, ...] = ()
    manifest_hash: str = ""

    @classmethod
    def empty(cls) -> "RegistrySnapshot":
        return cls(version="registry-v000000")

    @cached_property
    def concept_by_id(self) -> dict[str, CanonicalEntity]:
        return {item.concept_id: item for item in self.canonical_entities}

    @cached_property
    def active_concepts(self) -> tuple[CanonicalEntity, ...]:
        return tuple(item for item in self.canonical_entities if item.lifecycle_status == LifecycleStatus.ACTIVE)

    @cached_property
    def instance_by_id(self) -> dict[str, EntityInstance]:
        return {item.instance_id: item for item in self.entity_instances}

    @cached_property
    def resolved_redirects(self) -> dict[str, str]:
        raw = {item.retired_id: item.active_id for item in self.redirects}
        resolved: dict[str, str] = {}
        for source in raw:
            seen: set[str] = set()
            target = source
            while target in raw:
                if target in seen:
                    raise ContractError(f"redirect cycle contains {target}")
                seen.add(target)
                target = raw[target]
            resolved[source] = target
        return resolved

    def active_id(self, target_id: str) -> str:
        return self.resolved_redirects.get(target_id, target_id)

    def next_version(self) -> str:
        try:
            number = int(self.version.rsplit("v", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ContractError(f"registry version is not registry-vNNNNNN: {self.version}") from exc
        return f"registry-v{number + 1:06d}"

    def validate(self, authority_snapshot: AuthoritySnapshot | None = None) -> None:
        _unique(self.canonical_entities, "concept_id", "canonical_entities")
        _unique(self.entity_instances, "instance_id", "entity_instances")
        _unique(self.canonical_aliases, "alias_id", "canonical_aliases")
        _unique(self.authority_links, "authority_link_id", "authority_links")
        _unique(self.entity_relations, "relation_id", "entity_relations")
        _unique(self.redirects, "retired_id", "redirects")
        _unique(self.constraints, "constraint_id", "constraints")
        _unique(self.events, "event_id", "events")
        concepts = self.concept_by_id
        active = {item.concept_id for item in self.active_concepts}
        authority_aliases = [
            item for item in self.canonical_aliases if item.trust_level == AliasTrust.AUTHORITY
        ]
        if (self.authority_links or authority_aliases) and (
            authority_snapshot is None or not authority_snapshot.records
        ):
            raise ContractError(
                "authority-backed registry records require a pinned, non-empty authority snapshot"
            )
        for alias in self.canonical_aliases:
            if alias.concept_id not in concepts:
                raise ContractError(f"alias {alias.alias_id} references missing concept {alias.concept_id}")
            if alias.entity_type != concepts[alias.concept_id].entity_type:
                raise ContractError(f"alias {alias.alias_id} has incompatible type")
        authority_owners: dict[tuple[str, str], str] = {}
        per_concept_authority: dict[tuple[str, str], str] = {}
        links_by_concept: dict[str, list[AuthorityLink]] = {}
        for link in self.authority_links:
            if link.concept_id not in concepts:
                raise ContractError(f"authority link references missing concept {link.concept_id}")
            identifier = (link.authority.upper(), link.external_id)
            previous = authority_owners.setdefault(identifier, link.concept_id)
            if previous != link.concept_id:
                raise ContractError(f"authority identifier {identifier} belongs to multiple concepts")
            key = (link.concept_id, link.authority.upper())
            previous_id = per_concept_authority.setdefault(key, link.external_id)
            if previous_id != link.external_id:
                raise ContractError(f"concept {link.concept_id} has conflicting {link.authority} identifiers")
            if authority_snapshot:
                record = authority_snapshot.get(link.authority, link.external_id)
                if link.authority_snapshot_version != authority_snapshot.version:
                    raise ContractError(
                        f"authority link {link.authority_link_id} uses snapshot "
                        f"{link.authority_snapshot_version}, expected {authority_snapshot.version}"
                    )
                if record is None:
                    raise ContractError(
                        f"authority link {link.authority_link_id} is absent from pinned snapshot"
                    )
                if record.entity_type != concepts[link.concept_id].entity_type:
                    raise ContractError(f"authority type conflict for {link.authority}:{link.external_id}")
            links_by_concept.setdefault(link.concept_id, []).append(link)
        linked_concepts = set(links_by_concept)
        for alias in authority_aliases:
            if alias.concept_id not in linked_concepts:
                raise ContractError(
                    f"authority alias {alias.alias_id} has no pinned authority link for its concept"
                )
        for concept_id, links in links_by_concept.items():
            authorities = {(item.authority.upper(), item.external_id) for item in links}
            if len({authority for authority, _external_id in authorities}) <= 1:
                continue
            if authority_snapshot is None:
                raise ContractError(
                    f"concept {concept_id} has cross-authority identifiers but no pinned snapshot/crosswalk validator"
                )
            values = sorted(authorities)
            for index, left in enumerate(values):
                for right in values[index + 1 :]:
                    if left[0] == right[0]:
                        continue
                    if not authority_snapshot.are_crosswalked(left[0], left[1], right[0], right[1]):
                        raise ContractError(
                            f"concept {concept_id} has untrusted cross-authority links {left} and {right}"
                        )
        for instance in self.entity_instances:
            try:
                source_ids = json.loads(instance.source_mention_ids_json)
            except json.JSONDecodeError as exc:
                raise ContractError(f"instance {instance.instance_id} has invalid source_mention_ids_json") from exc
            if not isinstance(source_ids, list) or not source_ids or any(not str(item).strip() for item in source_ids):
                raise ContractError(f"instance {instance.instance_id} requires non-empty source mention IDs")
            if len(source_ids) != len(set(source_ids)):
                raise ContractError(f"instance {instance.instance_id} has duplicate source mention IDs")
            if instance.concept_id and instance.concept_id not in active:
                raise ContractError(f"instance {instance.instance_id} targets non-active concept {instance.concept_id}")
        for relation in self.entity_relations:
            if relation.source_concept_id not in concepts or relation.target_concept_id not in concepts:
                raise ContractError(f"relation {relation.relation_id} references missing concepts")
            if relation.source_concept_id == relation.target_concept_id:
                raise ContractError(f"relation {relation.relation_id} is a self-loop")
        self.resolved_redirects
        for redirect in self.redirects:
            if redirect.retired_id not in concepts or redirect.active_id not in active:
                raise ContractError(f"invalid redirect {redirect.retired_id} -> {redirect.active_id}")
        instances = self.instance_by_id
        for constraint in self.constraints:
            if constraint.target_id not in concepts and constraint.target_id not in instances:
                raise ContractError(
                    f"constraint {constraint.constraint_id} references missing target {constraint.target_id}"
                )


def _unique(records: Iterable[object], field: str, label: str) -> None:
    values = [getattr(record, field) for record in records]
    if len(values) != len(set(values)):
        raise ContractError(f"{label} contains duplicate {field} values")


@dataclass(frozen=True)
class RegistryChange:
    snapshot: RegistrySnapshot
    events: tuple[ResolutionEvent, ...]
    diff: Mapping[str, object]


@dataclass(frozen=True)
class SplitPartition:
    seed_mention_id: str
    preferred_label: str
    identity_qualifiers_json: str
    alias_ids: tuple[str, ...] = ()
    authority_link_ids: tuple[str, ...] = ()
    instance_ids: tuple[str, ...] = ()


def merge_concepts(
    snapshot: RegistrySnapshot,
    concept_ids: Sequence[str],
    survivor_id: str,
    run_id: str,
    reviewer: str,
    reason: str,
    authority_snapshot: AuthoritySnapshot | None = None,
) -> RegistryChange:
    """Apply an explicit merge plan; fuzzy similarity alone never calls this."""

    ids = tuple(sorted(set(concept_ids)))
    if len(ids) < 2 or survivor_id not in ids:
        raise ContractError("merge requires at least two concepts and a survivor among them")
    concepts = snapshot.concept_by_id
    missing = set(ids) - set(concepts)
    if missing:
        raise ContractError(f"merge references missing concepts {sorted(missing)}")
    records = [concepts[item] for item in ids]
    if any(record.lifecycle_status != LifecycleStatus.ACTIVE for record in records):
        raise ContractError("only active concepts may be merged")
    if len({record.entity_type for record in records}) != 1:
        raise ContractError("concepts of different types cannot be merged")
    merged_identity = _merge_identity_qualifiers(records)
    links_by_authority: dict[str, str] = {}
    for link in snapshot.authority_links:
        if link.concept_id in ids:
            previous = links_by_authority.setdefault(link.authority.upper(), link.external_id)
            if previous != link.external_id:
                raise ContractError(f"merge would combine conflicting {link.authority} identifiers")
    new_version = snapshot.next_version()
    updated_concepts = []
    for entity in snapshot.canonical_entities:
        if entity.concept_id in ids and entity.concept_id != survivor_id:
            updated_concepts.append(
                replace(entity, lifecycle_status=LifecycleStatus.MERGED, updated_run_id=run_id, registry_version=new_version)
            )
        elif entity.concept_id == survivor_id:
            updated_concepts.append(
                replace(
                    entity,
                    identity_qualifiers_json=merged_identity,
                    updated_run_id=run_id,
                    registry_version=new_version,
                )
            )
        else:
            updated_concepts.append(entity)
    events: list[ResolutionEvent] = []
    redirects = list(snapshot.redirects)
    for retired in ids:
        if retired == survivor_id:
            continue
        event_id = stable_event_id(EventType.MERGE, run_id, retired, survivor_id)
        events.append(
            ResolutionEvent(event_id, EventType.MERGE, retired, survivor_id, run_id, new_version, (reason,), reviewer)
        )
        redirects.append(EntityRedirect(retired, survivor_id, event_id, new_version))
    aliases = tuple(
        replace(alias, concept_id=survivor_id, registry_version=new_version)
        if alias.concept_id in ids and alias.concept_id != survivor_id
        else alias
        for alias in snapshot.canonical_aliases
    )
    links = tuple(
        replace(link, concept_id=survivor_id, registry_version=new_version)
        if link.concept_id in ids and link.concept_id != survivor_id
        else link
        for link in snapshot.authority_links
    )
    instances = tuple(
        replace(instance, concept_id=survivor_id, updated_run_id=run_id, registry_version=new_version)
        if instance.concept_id in ids and instance.concept_id != survivor_id
        else instance
        for instance in snapshot.entity_instances
    )
    changed = RegistrySnapshot(
        version=new_version,
        canonical_entities=tuple(sorted(updated_concepts, key=lambda item: item.concept_id)),
        entity_instances=tuple(sorted(instances, key=lambda item: item.instance_id)),
        canonical_aliases=_dedupe_aliases(aliases),
        authority_links=_dedupe_links(links),
        entity_relations=snapshot.entity_relations,
        redirects=tuple(sorted(redirects, key=lambda item: item.retired_id)),
        constraints=snapshot.constraints,
        events=tuple(sorted(snapshot.events + tuple(events), key=lambda item: item.event_id)),
    )
    changed.validate(authority_snapshot)
    return RegistryChange(changed, tuple(events), {"merged": list(ids), "survivor": survivor_id})


def _merge_identity_qualifiers(records: Sequence[CanonicalEntity]) -> str:
    """Union compatible identity context; reject only explicit unequal values."""

    merged: dict[str, str] = {}
    for record in records:
        try:
            parsed = json.loads(record.identity_qualifiers_json or "[]")
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid identity qualifier JSON on {record.concept_id}") from exc
        if not isinstance(parsed, list):
            raise ContractError(f"identity qualifiers on {record.concept_id} must be a list")
        for item in parsed:
            if not isinstance(item, Mapping) or set(item) != {"kind", "value_text"}:
                raise ContractError(f"malformed identity qualifier on {record.concept_id}")
            kind, value = str(item["kind"]), str(item["value_text"])
            previous = merged.setdefault(kind, value)
            if previous != value:
                raise ContractError(f"concepts have conflicting identity qualifier {kind}")
    return json.dumps(
        [{"kind": kind, "value_text": value} for kind, value in sorted(merged.items())],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _dedupe_aliases(items: Iterable[CanonicalAlias]) -> tuple[CanonicalAlias, ...]:
    # Alias records with different language/kind/provenance are distinct audit
    # evidence even when their normalized lookup key is the same.
    chosen: dict[str, CanonicalAlias] = {}
    for item in sorted(items, key=lambda value: value.alias_id):
        chosen.setdefault(item.alias_id, item)
    return tuple(sorted(chosen.values(), key=lambda item: item.alias_id))


def _dedupe_links(items: Iterable[AuthorityLink]) -> tuple[AuthorityLink, ...]:
    chosen: dict[tuple[str, str, str], AuthorityLink] = {}
    for item in sorted(items, key=lambda value: value.authority_link_id):
        key = (item.concept_id, item.authority.upper(), item.external_id)
        chosen.setdefault(key, item)
    return tuple(sorted(chosen.values(), key=lambda item: item.authority_link_id))


def split_concept(
    snapshot: RegistrySnapshot,
    source_concept_id: str,
    partitions: Sequence[SplitPartition],
    run_id: str,
    reviewer: str,
    reason: str,
    authority_snapshot: AuthoritySnapshot | None = None,
) -> RegistryChange:
    """Apply an explicit reviewed split with complete artifact reassignment."""

    source = snapshot.concept_by_id.get(source_concept_id)
    if source is None or source.lifecycle_status != LifecycleStatus.ACTIVE:
        raise ContractError("split source must be an active concept")
    if len(partitions) < 2:
        raise ContractError("a split requires at least two reviewed partitions")
    incident_relations = [
        item.relation_id
        for item in snapshot.entity_relations
        if source_concept_id in {item.source_concept_id, item.target_concept_id}
    ]
    if incident_relations:
        raise ContractError(
            "split has incident relations and requires an explicit reviewed relation-reassignment plan; "
            f"relations={sorted(incident_relations)}"
        )
    targeting_constraints = [
        item.constraint_id
        for item in snapshot.constraints
        if item.active and snapshot.active_id(item.target_id) == source_concept_id
    ]
    if targeting_constraints:
        raise ContractError(
            "split has active constraints and requires an explicit reviewed constraint-reassignment plan; "
            f"constraints={sorted(targeting_constraints)}"
        )
    new_ids = [stable_concept_id(item.seed_mention_id) for item in partitions]
    if len(new_ids) != len(set(new_ids)) or source_concept_id in new_ids:
        raise ContractError("split partitions produce duplicate/existing concept IDs")

    expected_aliases = {item.alias_id for item in snapshot.canonical_aliases if item.concept_id == source_concept_id}
    expected_links = {item.authority_link_id for item in snapshot.authority_links if item.concept_id == source_concept_id}
    expected_instances = {item.instance_id for item in snapshot.entity_instances if item.concept_id == source_concept_id}
    _assert_complete_partition("aliases", expected_aliases, [value for item in partitions for value in item.alias_ids])
    _assert_complete_partition("authority links", expected_links, [value for item in partitions for value in item.authority_link_ids])
    _assert_complete_partition("instances", expected_instances, [value for item in partitions for value in item.instance_ids])

    new_version = snapshot.next_version()
    entities = [
        replace(source, lifecycle_status=LifecycleStatus.SPLIT, updated_run_id=run_id, registry_version=new_version)
        if item.concept_id == source_concept_id else item
        for item in snapshot.canonical_entities
    ]
    events: list[ResolutionEvent] = []
    alias_target: dict[str, str] = {}
    link_target: dict[str, str] = {}
    instance_target: dict[str, str] = {}
    for partition, concept_id in zip(partitions, new_ids):
        entities.append(
            CanonicalEntity(
                concept_id=concept_id,
                preferred_label=partition.preferred_label,
                entity_type=source.entity_type,
                identity_qualifiers_json=partition.identity_qualifiers_json,
                lifecycle_status=LifecycleStatus.ACTIVE,
                seed_mention_id=partition.seed_mention_id,
                created_run_id=run_id,
                updated_run_id=run_id,
                registry_version=new_version,
                provenance_json=json.dumps(
                    {"split_from": source_concept_id, "reviewer": reviewer, "reason": reason},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )
        event_id = stable_event_id(EventType.SPLIT, run_id, source_concept_id, concept_id)
        events.append(ResolutionEvent(event_id, EventType.SPLIT, source_concept_id, concept_id, run_id, new_version, (reason,), reviewer))
        alias_target.update({item: concept_id for item in partition.alias_ids})
        link_target.update({item: concept_id for item in partition.authority_link_ids})
        instance_target.update({item: concept_id for item in partition.instance_ids})
    aliases = tuple(
        replace(item, concept_id=alias_target[item.alias_id], registry_version=new_version)
        if item.alias_id in alias_target else item
        for item in snapshot.canonical_aliases
    )
    links = tuple(
        replace(item, concept_id=link_target[item.authority_link_id], registry_version=new_version)
        if item.authority_link_id in link_target else item
        for item in snapshot.authority_links
    )
    instances = []
    for item in snapshot.entity_instances:
        if item.instance_id in instance_target:
            new_target = instance_target[item.instance_id]
            instances.append(replace(item, concept_id=new_target, updated_run_id=run_id, registry_version=new_version))
            event_id = stable_event_id(EventType.REASSIGN, run_id, item.instance_id, new_target)
            events.append(ResolutionEvent(event_id, EventType.REASSIGN, item.instance_id, new_target, run_id, new_version, (reason,), reviewer))
        else:
            instances.append(item)
    changed = RegistrySnapshot(
        version=new_version,
        canonical_entities=tuple(sorted(entities, key=lambda item: item.concept_id)),
        entity_instances=tuple(sorted(instances, key=lambda item: item.instance_id)),
        canonical_aliases=tuple(sorted(aliases, key=lambda item: item.alias_id)),
        authority_links=tuple(sorted(links, key=lambda item: item.authority_link_id)),
        entity_relations=snapshot.entity_relations,
        redirects=snapshot.redirects,
        constraints=snapshot.constraints,
        events=tuple(sorted(snapshot.events + tuple(events), key=lambda item: item.event_id)),
    )
    changed.validate(authority_snapshot)
    return RegistryChange(changed, tuple(events), {"split": source_concept_id, "created": sorted(new_ids)})


def _assert_complete_partition(label: str, expected: set[str], assigned: Sequence[str]) -> None:
    assigned_set = set(assigned)
    if len(assigned) != len(assigned_set):
        raise ContractError(f"split {label} are assigned to more than one partition")
    if assigned_set != expected:
        raise ContractError(
            f"split {label} must be assigned exactly once; missing={sorted(expected-assigned_set)}, extra={sorted(assigned_set-expected)}"
        )


def reassign_instance_concept(
    snapshot: RegistrySnapshot,
    instance_id: str,
    new_concept_id: str | None,
    run_id: str,
    reviewer: str,
    reason: str,
    *,
    compatible_types: Mapping[EntityType, Sequence[EntityType]] | None = None,
    authority_snapshot: AuthoritySnapshot | None = None,
) -> RegistryChange:
    instance = snapshot.instance_by_id.get(instance_id)
    if instance is None:
        raise ContractError(f"missing instance {instance_id}")
    if new_concept_id is not None:
        concept = snapshot.concept_by_id.get(snapshot.active_id(new_concept_id))
        if concept is None or concept.lifecycle_status != LifecycleStatus.ACTIVE:
            raise ContractError(f"new instance target is not active: {new_concept_id}")
        allowed = set((compatible_types or {}).get(instance.entity_type, (instance.entity_type,)))
        if concept.entity_type not in allowed:
            raise ContractError(f"instance type {instance.entity_type} cannot target {concept.entity_type}")
        new_concept_id = concept.concept_id
    new_version = snapshot.next_version()
    updated = tuple(
        replace(item, concept_id=new_concept_id, updated_run_id=run_id, registry_version=new_version)
        if item.instance_id == instance_id else item
        for item in snapshot.entity_instances
    )
    event_id = stable_event_id(EventType.REASSIGN, run_id, instance_id, new_concept_id)
    event = ResolutionEvent(event_id, EventType.REASSIGN, instance_id, new_concept_id, run_id, new_version, (reason,), reviewer)
    changed = replace(snapshot, version=new_version, entity_instances=updated, events=tuple(sorted(snapshot.events + (event,), key=lambda item: item.event_id)), manifest_hash="")
    changed.validate(authority_snapshot)
    return RegistryChange(changed, (event,), {"instance_id": instance_id, "from": instance.concept_id, "to": new_concept_id})


def record_human_constraint(
    snapshot: RegistrySnapshot,
    mention_id: str,
    target_id: str,
    constraint_type: ConstraintType,
    run_id: str,
    reviewer: str,
    reason: str,
    authority_snapshot: AuthoritySnapshot | None = None,
) -> RegistryChange:
    if target_id not in snapshot.concept_by_id and target_id not in snapshot.instance_by_id:
        raise ContractError(f"constraint target does not exist: {target_id}")
    opposite = ConstraintType.CANNOT_LINK if constraint_type == ConstraintType.MUST_LINK else ConstraintType.MUST_LINK
    if any(
        item.active and item.mention_id == mention_id and item.target_id == target_id and item.constraint_type == opposite
        for item in snapshot.constraints
    ):
        raise ContractError(
            f"cannot add {constraint_type} while active {opposite} exists for {mention_id} -> {target_id}"
        )
    new_version = snapshot.next_version()
    stable = hashlib.sha256(
        f"{constraint_type.value}\x1f{mention_id}\x1f{target_id}\x1f{reviewer}".encode()
    ).hexdigest()[:32]
    constraint_id = f"CST-{stable}"
    constraints = []
    events: list[ResolutionEvent] = []
    for item in snapshot.constraints:
        if item.active and item.mention_id == mention_id and item.constraint_type == constraint_type and item.target_id != target_id:
            constraints.append(replace(item, active=False, registry_version=new_version))
            event_id = stable_event_id(EventType.SUPERSEDE, run_id, item.constraint_id, constraint_id)
            events.append(ResolutionEvent(event_id, EventType.SUPERSEDE, item.constraint_id, constraint_id, run_id, new_version, (reason,), reviewer))
        else:
            constraints.append(item)
    if not any(item.constraint_id == constraint_id and item.active for item in constraints):
        constraints.append(
            ResolutionConstraint(
                constraint_id, constraint_type, mention_id, target_id, reviewer, reason, True, run_id, new_version
            )
        )
    event_id = stable_event_id(EventType.REVIEW, run_id, constraint_id, target_id)
    events.append(ResolutionEvent(event_id, EventType.REVIEW, constraint_id, target_id, run_id, new_version, (reason,), reviewer))
    changed = replace(
        snapshot,
        version=new_version,
        constraints=tuple(sorted(constraints, key=lambda item: item.constraint_id)),
        events=tuple(sorted(snapshot.events + tuple(events), key=lambda item: item.event_id)),
        manifest_hash="",
    )
    changed.validate(authority_snapshot)
    return RegistryChange(changed, tuple(events), {"constraint_id": constraint_id, "type": constraint_type.value})
