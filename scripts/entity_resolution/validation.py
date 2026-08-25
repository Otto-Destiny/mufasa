"""Fail-closed contract and cross-table validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from .contracts import AlignmentStatus, AssertionStatus, ContractError, IdentityScope, Mention, ProvenanceScope
from .policy import ResolverPolicy


@dataclass(frozen=True)
class ValidationIssue:
    mention_id: str
    code: str
    detail: str


@dataclass(frozen=True)
class ValidationResult:
    valid_mentions: tuple[Mention, ...]
    issues: tuple[ValidationIssue, ...]

    @property
    def invalid_ids(self) -> frozenset[str]:
        return frozenset(issue.mention_id for issue in self.issues)


def validate_mentions(mentions: Iterable[Mention], policy: ResolverPolicy) -> ValidationResult:
    ordered = tuple(sorted(mentions, key=lambda item: item.mention_id))
    issues: list[ValidationIssue] = []
    counts = Counter(item.mention_id for item in ordered)
    for mention_id, count in counts.items():
        if count > 1:
            issues.append(ValidationIssue(mention_id, "DUPLICATE_MENTION_ID", f"appears {count} times"))

    groups: dict[str, list[Mention]] = defaultdict(list)
    for mention in ordered:
        groups[mention.source_mention_id].append(mention)
        if mention.role not in policy.allowed_roles:
            issues.append(ValidationIssue(mention.mention_id, "OFF_LIST_ROLE", mention.role))
        if mention.entity_type not in policy.allowed_types:
            issues.append(ValidationIssue(mention.mention_id, "OFF_LIST_ENTITY_TYPE", str(mention.entity_type)))
        if mention.identity_scope not in policy.allowed_scopes:
            issues.append(ValidationIssue(mention.mention_id, "OFF_LIST_IDENTITY_SCOPE", str(mention.identity_scope)))
        if mention.owner_kind not in policy.allowed_owner_kinds:
            issues.append(ValidationIssue(mention.mention_id, "OFF_LIST_OWNER_KIND", str(mention.owner_kind)))
        if mention.identity_scope == IdentityScope.STUDY_INSTANCE and not mention.context_id:
            issues.append(ValidationIssue(mention.mention_id, "INSTANCE_CONTEXT_MISSING", "study instance requires context_id"))
        if mention.source_alignment_status == AlignmentStatus.EXACT_UNIQUE:
            if mention.source_char_start is None or mention.source_char_end is None:
                issues.append(ValidationIssue(mention.mention_id, "VERIFIED_SOURCE_OFFSETS_MISSING", "unique raw source offsets are required"))
            if mention.source_occurrence_count != 1:
                issues.append(ValidationIssue(mention.mention_id, "UNIQUE_OCCURRENCE_COUNT_INVALID", "EXACT_UNIQUE requires one occurrence"))
        elif mention.source_alignment_status == AlignmentStatus.EXACT_AMBIGUOUS:
            if mention.source_char_start is not None or mention.source_char_end is not None:
                issues.append(ValidationIssue(mention.mention_id, "AMBIGUOUS_OFFSET_MUST_BE_NULL", "ambiguous evidence cannot select an occurrence"))
            if mention.source_occurrence_count < 2:
                issues.append(ValidationIssue(mention.mention_id, "AMBIGUOUS_OCCURRENCE_COUNT_INVALID", "EXACT_AMBIGUOUS requires at least two occurrences"))
        if mention.source_page is None or mention.source_page < 1:
            issues.append(ValidationIssue(mention.mention_id, "VERIFIED_SOURCE_PAGE_MISSING", "raw source page is required"))
        if mention.assertion_status != AssertionStatus.REPORTED:
            issues.append(
                ValidationIssue(
                    mention.mention_id,
                    "NON_REPORTED_ASSERTION",
                    "source entity resolution accepts paper-reported extraction only",
                )
            )
        if mention.provenance_scope == ProvenanceScope.STUDY_CONTEXT and not mention.source_evidence_id:
            issues.append(ValidationIssue(mention.mention_id, "CONTEXT_PROVENANCE_MISSING", "study-context provenance requires context evidence"))
        for qualifier in mention.qualifiers:
            rule = policy.qualifier_rule(qualifier.kind)
            if rule is None:
                issues.append(ValidationIssue(mention.mention_id, "OFF_LIST_QUALIFIER_KIND", qualifier.kind))
                continue
            if mention.entity_type not in rule.allowed_types:
                issues.append(
                    ValidationIssue(
                        mention.mention_id,
                        "QUALIFIER_TYPE_CONFLICT",
                        f"{rule.kind} is not allowed for {mention.entity_type}",
                    )
                )
            if mention.identity_scope not in rule.allowed_scopes:
                issues.append(
                    ValidationIssue(
                        mention.mention_id,
                        "QUALIFIER_SCOPE_CONFLICT",
                        f"{rule.kind} is not allowed for {mention.identity_scope}",
                    )
                )

    for source_id, members in groups.items():
        signatures = {
            (
                member.paper_id,
                member.source_evidence_id,
                member.source_page,
                member.source_char_start,
                member.source_char_end,
                member.source_occurrence_count,
                member.source_occurrences_json,
                member.surface_text,
            )
            for member in members
        }
        if len(signatures) != 1:
            for member in members:
                issues.append(
                    ValidationIssue(
                        member.mention_id,
                        "INCONSISTENT_SOURCE_MENTION_GROUP",
                        f"source_mention_id {source_id} spans different source phrases",
                    )
                )

    invalid = {issue.mention_id for issue in issues}
    valid = tuple(item for item in ordered if item.mention_id not in invalid)
    return ValidationResult(valid, tuple(sorted(issues, key=lambda issue: (issue.mention_id, issue.code, issue.detail))))


def assert_referential_integrity(
    mentions: Iterable[Mention],
    owner_ids: Mapping[str, frozenset[str]],
    evidence_ids: frozenset[str],
    eligible_paper_ids: frozenset[str] | None = None,
) -> None:
    errors: list[str] = []
    for mention in mentions:
        known_owners = owner_ids.get(mention.owner_kind.value, frozenset())
        if mention.owner_id not in known_owners:
            errors.append(f"{mention.mention_id}: unknown {mention.owner_kind} owner {mention.owner_id}")
        if mention.source_evidence_id not in evidence_ids:
            errors.append(f"{mention.mention_id}: unknown evidence {mention.source_evidence_id}")
        if eligible_paper_ids is not None and mention.paper_id not in eligible_paper_ids:
            errors.append(f"{mention.mention_id}: paper {mention.paper_id} is ineligible")
    if errors:
        preview = "; ".join(errors[:20])
        suffix = f"; and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ContractError(f"referential integrity failed: {preview}{suffix}")


def automatic_evidence_is_safe(mention: Mention) -> bool:
    return mention.source_alignment_status == AlignmentStatus.EXACT_UNIQUE
