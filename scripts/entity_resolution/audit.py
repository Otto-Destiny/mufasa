"""Audit tables, review queues, candidate hashes, and registry diffs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Iterable

from .contracts import Candidate, DecisionStatus, Mention, ResolutionDecision, ResolutionRun
from .registry import RegistrySnapshot


def candidate_set_hash(candidates: Iterable[Candidate]) -> str:
    payload = [
        {
            "target_kind": item.target_kind,
            "target_id": item.target_id,
            "method": item.method,
            "score": item.score,
            "conflicts": list(item.conflicts),
            "features": list(item.features),
        }
        for item in sorted(candidates, key=lambda c: (c.target_kind, c.target_id, c.method))
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def decision_memo_key(
    mention: Mention,
    candidate_hash: str,
    policy_version: str,
    policy_hash: str,
    registry_version: str,
    registry_manifest_hash: str,
    authority_manifest_hash: str,
    resolver_code_version: str,
) -> str:
    payload = {
        "mention_id": mention.mention_id,
        "source_mention_id": mention.source_mention_id,
        "source_evidence_id": mention.source_evidence_id,
        "paper_id": mention.paper_id,
        "owner_kind": mention.owner_kind.value,
        "owner_id": mention.owner_id,
        "context_id": mention.context_id,
        "role": mention.role,
        "surface_text": mention.surface_text,
        "atom_text": mention.atom_text,
        "type": mention.entity_type.value,
        "scope": mention.identity_scope.value,
        "qualifiers": [(q.kind, q.value_text) for q in mention.qualifiers],
        "source_page": mention.source_page,
        "source_char_start": mention.source_char_start,
        "source_char_end": mention.source_char_end,
        "source_occurrence_count": mention.source_occurrence_count,
        "source_occurrences_json": mention.source_occurrences_json,
        "source_alignment_status": mention.source_alignment_status.value,
        "provenance_scope": mention.provenance_scope.value,
        "qualifier_vocab_version": mention.qualifier_vocab_version,
        "extraction_schema_version": mention.extraction_schema_version,
        "assertion_status": mention.assertion_status.value,
        "domain": mention.domain,
        "language": mention.language,
        "country_code": mention.country_code,
        "aliases_json": mention.aliases_json,
        "source_flags": list(mention.source_flags),
        "metadata": list(mention.metadata),
        "candidate_set_hash": candidate_hash,
        "policy_version": policy_version,
        "policy_hash": policy_hash,
        "registry_version": registry_version,
        "registry_manifest_hash": registry_manifest_hash,
        "authority_manifest_hash": authority_manifest_hash,
        "resolver_code_version": resolver_code_version,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_review_rows(run: ResolutionRun) -> list[dict[str, object]]:
    """Deduplicate review by mention signature and candidate set, not occurrence."""

    mentions = {item.mention_id: item for item in run.mentions}
    groups: dict[tuple[object, ...], list[ResolutionDecision]] = defaultdict(list)
    for decision in run.decisions:
        post_merge_audit = any(
            code.startswith("AUTO_MERGE_REVIEW:") for code in decision.reason_codes
        )
        if (
            decision.status not in {DecisionStatus.REVIEW_REQUIRED, DecisionStatus.UNRESOLVED}
            and not post_merge_audit
        ):
            continue
        mention = mentions[decision.mention_id]
        key = (
            mention.entity_type.value,
            mention.identity_scope.value,
            mention.atom_text.casefold(),
            tuple((q.kind, q.value_text.casefold()) for q in mention.qualifiers),
            decision.candidate_set_hash,
            decision.reason_codes,
        )
        if mention.identity_scope.value == "STUDY_INSTANCE":
            key += (mention.paper_id, mention.context_id, mention.source_mention_id)
        else:
            key += (
                (mention.language or "").casefold(),
                (mention.country_code or "").upper(),
                (mention.domain or "").upper(),
            )
        groups[key].append(decision)
    rows: list[dict[str, object]] = []
    for index, (key, decisions) in enumerate(sorted(groups.items(), key=lambda item: repr(item[0]))):
        mention_ids = sorted(item.mention_id for item in decisions)
        sample = mentions[mention_ids[0]]
        sample_decision = decisions[0]
        blocking = sample_decision.status in {
            DecisionStatus.REVIEW_REQUIRED,
            DecisionStatus.UNRESOLVED,
        }
        rows.append(
            {
                "review_task_id": f"REV-{hashlib.sha256(repr(key).encode()).hexdigest()[:24]}",
                "entity_type": sample.entity_type.value,
                "identity_scope": sample.identity_scope.value,
                "atom_text": sample.atom_text,
                "occurrence_count": len(mention_ids),
                "mention_ids_json": json.dumps(mention_ids, separators=(",", ":")),
                "candidate_set_hash": decisions[0].candidate_set_hash,
                "reason_codes_json": json.dumps(list(decisions[0].reason_codes), separators=(",", ":")),
                "review_mode": "BLOCKING_CONFLICT" if blocking else "POST_MERGE_AUDIT",
                "blocking": blocking,
                "resolved_concept_id": sample_decision.concept_id or "",
                "proposal_id": sample_decision.proposal_id or "",
                "priority": _review_priority(sample.entity_type.value, len(mention_ids), decisions[0]),
                "propagation_policy": (
                    "LOCKED_SOURCE_MENTION_GROUP"
                    if sample.identity_scope.value == "STUDY_INSTANCE"
                    else "REVIEWER_CONFIRM_EACH_MENTION"
                    if blocking
                    else "POST_MERGE_AUDIT_ONLY"
                ),
                "propagation_approved": False,
                "reviewer_decision": "",
                "reviewer_target_id": "",
                "reviewer_notes": "",
            }
        )
    return sorted(rows, key=lambda row: (-float(row["priority"]), str(row["review_task_id"])))


def _review_priority(entity_type: str, recurrence: int, decision: ResolutionDecision) -> float:
    risk = 2.0 if entity_type in {"PLACE", "ORGANISM", "CHEMICAL", "HEALTH_CONDITION"} else 1.0
    ambiguity = 2.0 if decision.candidate_count > 1 else 1.0
    return round(risk * ambiguity * (1.0 + min(recurrence, 100) / 10.0), 3)


def registry_diff(before: RegistrySnapshot, after: RegistrySnapshot) -> dict[str, object]:
    def ids(items: Iterable[object], field: str) -> set[str]:
        return {str(getattr(item, field)) for item in items}

    mapping = {
        "concepts": (before.canonical_entities, after.canonical_entities, "concept_id"),
        "instances": (before.entity_instances, after.entity_instances, "instance_id"),
        "aliases": (before.canonical_aliases, after.canonical_aliases, "alias_id"),
        "authority_links": (before.authority_links, after.authority_links, "authority_link_id"),
        "relations": (before.entity_relations, after.entity_relations, "relation_id"),
        "redirects": (before.redirects, after.redirects, "retired_id"),
        "constraints": (before.constraints, after.constraints, "constraint_id"),
        "events": (before.events, after.events, "event_id"),
    }
    result: dict[str, object] = {
        "from_registry_version": before.version,
        "to_registry_version": after.version,
    }
    for label, (old_items, new_items, field) in mapping.items():
        old_ids, new_ids = ids(old_items, field), ids(new_items, field)
        result[label] = {
            "added_count": len(new_ids - old_ids),
            "removed_count": len(old_ids - new_ids),
            "added_ids": sorted(new_ids - old_ids),
            "removed_ids": sorted(old_ids - new_ids),
        }
    return result


def run_summary(run: ResolutionRun) -> dict[str, object]:
    status = Counter(item.status.value for item in run.decisions)
    methods = Counter(item.method.value for item in run.decisions)
    return {
        "run_id": run.run_id,
        "mentions": len(run.mentions),
        "status_counts": dict(sorted(status.items())),
        "method_counts": dict(sorted(methods.items())),
        "proposals": len(run.proposals),
        "candidates_generated": run.generated_candidate_count,
        "candidates_retained": len(run.candidates),
        "invalid_inputs": len(run.invalid_inputs),
        "review_tasks": len(build_review_rows(run)),
    }
