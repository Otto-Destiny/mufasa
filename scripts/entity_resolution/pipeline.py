"""Deterministic dry-run resolution and explicit atomic registry commits."""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.metadata
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from .audit import candidate_set_hash, decision_memo_key, registry_diff
from .authorities import AuthoritySnapshot
from .contracts import (
    AliasTrust,
    AuthorityHint,
    AuthorityLink,
    CanonicalAlias,
    CanonicalEntity,
    ConceptMergePlan,
    ContractError,
    DecisionMethod,
    DecisionStatus,
    EntityInstance,
    EventType,
    InvalidMentionRecord,
    LifecycleStatus,
    Mention,
    Proposal,
    ProposalKind,
    ResolutionConflict,
    ResolutionDecision,
    ResolutionEvent,
    ResolutionRun,
)
from .matching import EmbeddingRecall, MatchResult, ResolverIndex, resolve_against_registry
from .normalization import (
    NORMALIZATION_IMPLEMENTATION,
    QUALIFIER_NORMALIZATION_VERSION,
    build_keys,
    primary_key,
    qualifier_signature,
)
from .policy import ResolverPolicy
from .registry import (
    RegistrySnapshot,
    merge_concepts,
    proposal_id,
    stable_alias_id,
    stable_authority_link_id,
    stable_concept_id,
    stable_event_id,
    stable_instance_id,
)
from .validation import ValidationIssue, validate_mentions


RESOLVER_CODE_VERSION = "mufasa-entity-resolution-0.2.0-candidate.1"
_AUTOMATIC_BLOCKING_FLAGS = frozenset(
    {
        "UNMODELED_CONDITION",
        "SUSPICIOUS_COMPOUND",
        "WEAK_SOURCE",
        "OWNER_REVIEW_REQUIRED",
        "PARSER_WARNING",
        "OCR_WARNING",
        "LOW_TEXT_PAGE",
        "SOURCE_EVIDENCE_EXACT_AMBIGUOUS",
    }
)


@dataclass(frozen=True)
class ResolutionExecution:
    run: ResolutionRun
    conflicts: tuple[ResolutionConflict, ...]
    capability_manifest: Mapping[str, Any]
    merge_plans: tuple[ConceptMergePlan, ...] = ()


@dataclass(frozen=True)
class CommitResult:
    registry: RegistrySnapshot
    run: ResolutionRun
    events: tuple[ResolutionEvent, ...]
    diff: Mapping[str, Any]


def preflight_capabilities(
    policy: ResolverPolicy,
    registry: RegistrySnapshot,
    authority_snapshot: AuthoritySnapshot,
    authority_hints: Sequence[AuthorityHint],
    embedding: EmbeddingRecall | None,
) -> dict[str, Any]:
    """Verify active capabilities. There are no hidden fallback matchers."""

    package_versions = {}
    python_version = ".".join(map(str, sys.version_info[:3]))
    if Version(python_version) not in SpecifierSet(policy.python_specifier):
        raise ContractError(
            f"Python {python_version} is incompatible; active policy requires {policy.python_specifier}"
        )
    for distribution, specifier_text in policy.dependency_specifiers:
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ContractError(
                f"required resolver package {distribution} is unavailable; install the declared dependency"
            ) from exc
        try:
            compatible = Version(installed) in SpecifierSet(specifier_text)
        except (InvalidVersion, InvalidSpecifier) as exc:
            raise ContractError(
                f"cannot validate dependency {distribution}={installed!r} against {specifier_text!r}: {exc}"
            ) from exc
        if not compatible:
            raise ContractError(
                f"resolver dependency {distribution}=={installed} is incompatible; active policy requires {specifier_text}"
            )
        package_versions[distribution] = {"installed": installed, "required": specifier_text}
    if policy.code_version != RESOLVER_CODE_VERSION:
        raise ContractError(
            f"policy code_version {policy.code_version} is incompatible with resolver {RESOLVER_CODE_VERSION}"
        )
    if policy.normalization_version != "mufasa-normalization-v1":
        raise ContractError(f"unsupported normalization policy {policy.normalization_version}")
    registry.validate(authority_snapshot if authority_snapshot.records else None)
    if authority_hints and not authority_snapshot.records:
        raise ContractError("authority hints require a pinned, non-empty authority snapshot")
    for hint in authority_hints:
        if hint.snapshot_version != authority_snapshot.version:
            raise ContractError(
                f"authority hint {hint.mention_id} uses {hint.snapshot_version}, expected {authority_snapshot.version}"
            )
        if authority_snapshot.get(hint.authority, hint.external_id) is None:
            raise ContractError(
                f"authority hint {hint.authority}:{hint.external_id} is absent from the pinned snapshot"
            )
    embedding_automatic_types = sorted(
        entity_type.value
        for entity_type, type_policy in policy.type_policies
        if type_policy.embedding.automatic
    )
    if embedding_automatic_types and embedding is None:
        raise ContractError(
            "active policy requires embedding adjudication for "
            f"{embedding_automatic_types}, but no pinned embedding backend was supplied"
        )
    embedding_manifest: dict[str, Any]
    if embedding is None:
        embedding_manifest = {
            "enabled": False,
            "reason": "explicitly disabled for this run; no embedding lane was executed",
        }
    else:
        for name in (
            "model_id", "model_hash", "implementation_version", "mention_vectors_hash",
            "target_vectors_hash", "vector_set_fingerprint",
        ):
            if not getattr(embedding, name, None):
                raise ContractError(f"embedding backend does not declare {name}")
        embedding_manifest = {
            "enabled": True,
            "model_id": embedding.model_id,
            "model_hash": embedding.model_hash,
            "implementation_version": embedding.implementation_version,
            "mention_vectors_sha256": embedding.mention_vectors_hash,
            "target_vectors_sha256": embedding.target_vectors_hash,
            "vector_set_fingerprint": embedding.vector_set_fingerprint,
        }
    return {
        "python": {"installed": python_version, "required": policy.python_specifier},
        "policy_contract": {
            "policy_version": policy.version,
            "policy_hash": policy.content_hash,
            "schema_version": policy.schema_version,
            "qualifier_vocab_version": policy.qualifier_vocab_version,
            "condition_vocab_version": policy.condition_vocab_version,
        },
        "packages": dict(sorted(package_versions.items())),
        "lexical": {
            "enabled": True,
            "implementation": "type-scoped-inverted-blocking+rapidfuzz.WRatio",
            "blocking": {
                "char_ngram_size": policy.lexical_blocking.char_ngram_size,
                "max_candidate_pool": policy.lexical_blocking.max_candidate_pool,
                "max_postings_per_key": policy.lexical_blocking.max_postings_per_key,
                "max_query_blockers": policy.lexical_blocking.max_query_blockers,
                "max_labels_per_target": policy.lexical_blocking.max_labels_per_target,
                "all_pairs_fallback": False,
            },
        },
        "embedding": embedding_manifest,
        "authorities": {
            "enabled": bool(authority_snapshot.records),
            "version": authority_snapshot.version,
            "manifest_hash": authority_snapshot.manifest_hash,
            "records": len(authority_snapshot.records),
            "network_required": False,
        },
        "normalization": {
            "policy_version": policy.normalization_version,
            "implementation": NORMALIZATION_IMPLEMENTATION,
            "qualifier_normalization_version": QUALIFIER_NORMALIZATION_VERSION,
            "policy_hash": policy.content_hash,
        },
        "no_fallbacks_used": True,
    }


def resolve_batch(
    mentions: Iterable[Mention],
    registry: RegistrySnapshot,
    policy: ResolverPolicy,
    *,
    authority_snapshot: AuthoritySnapshot | None = None,
    authority_hints: Iterable[AuthorityHint] = (),
    invalid_mentions: Iterable[InvalidMentionRecord] = (),
    input_fingerprint: str | None = None,
    embedding: EmbeddingRecall | None = None,
    memo: Mapping[str, ResolutionDecision] | None = None,
    workers: int = 1,
) -> ResolutionExecution:
    """Resolve against a frozen registry and return a mutation-free dry run."""

    mention_tuple = tuple(sorted(mentions, key=lambda item: item.mention_id))
    invalid_tuple = tuple(sorted(invalid_mentions, key=lambda item: item.mention_id))
    hint_tuple = tuple(sorted(authority_hints, key=lambda item: (item.mention_id, item.authority, item.external_id)))
    authority = authority_snapshot or AuthoritySnapshot.empty()
    if workers < 1:
        raise ContractError("workers must be positive")
    if len({item.mention_id for item in mention_tuple}) != len(mention_tuple):
        raise ContractError("duplicate mention_id values are not allowed")
    if {item.mention_id for item in mention_tuple} & {item.mention_id for item in invalid_tuple}:
        raise ContractError("a mention cannot be both parsed and adapter-invalid")
    known_ids = {item.mention_id for item in mention_tuple}
    unknown_hints = {item.mention_id for item in hint_tuple} - known_ids
    if unknown_hints:
        raise ContractError(f"authority hints reference unknown mentions {sorted(unknown_hints)[:10]}")
    capability_manifest = preflight_capabilities(policy, registry, authority, hint_tuple, embedding)

    fingerprint = input_fingerprint or _mention_fingerprint(mention_tuple, invalid_tuple)
    effective_controls = {
        "workers": workers,
        "top_k_candidates": policy.top_k_candidates,
        "score_round_digits": policy.score_round_digits,
        "embedding_enabled": embedding is not None,
        "lexical_max_candidate_pool": policy.lexical_blocking.max_candidate_pool,
        "lexical_max_postings_per_key": policy.lexical_blocking.max_postings_per_key,
        "lexical_max_query_blockers": policy.lexical_blocking.max_query_blockers,
        "lexical_max_labels_per_target": policy.lexical_blocking.max_labels_per_target,
    }
    run_id = _run_id(fingerprint, registry, policy, authority, capability_manifest, effective_controls)
    validation = validate_mentions(mention_tuple, policy)
    issues_by_id: dict[str, list[ValidationIssue]] = defaultdict(list)
    for issue in validation.issues:
        issues_by_id[issue.mention_id].append(issue)
    decisions: dict[str, ResolutionDecision] = {}
    for mention_id, issues in issues_by_id.items():
        decisions[mention_id] = ResolutionDecision(
            mention_id=mention_id,
            status=DecisionStatus.INVALID_INPUT,
            method=DecisionMethod.INVALID_CONTRACT,
            reason_codes=tuple(sorted({item.code for item in issues})),
        )
    for invalid in invalid_tuple:
        decisions[invalid.mention_id] = ResolutionDecision(
            mention_id=invalid.mention_id,
            status=DecisionStatus.INVALID_INPUT,
            method=DecisionMethod.INVALID_CONTRACT,
            reason_codes=invalid.error_codes,
        )

    index = ResolverIndex(registry, policy)
    hints_by_id: dict[str, list[AuthorityHint]] = defaultdict(list)
    for hint in hint_tuple:
        hints_by_id[hint.mention_id].append(hint)

    def resolve_one(mention: Mention) -> tuple[str, MatchResult]:
        return (
            mention.mention_id,
            resolve_against_registry(
                mention,
                index,
                policy,
                authority,
                hints_by_id.get(mention.mention_id, ()),
                embedding,
            ),
        )

    match_results: dict[str, MatchResult] = {}
    valid_for_matching = list(validation.valid_mentions)
    if workers == 1:
        for mention in valid_for_matching:
            mention_id, result = resolve_one(mention)
            match_results[mention_id] = result
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(resolve_one, mention): mention.mention_id for mention in valid_for_matching}
            for future in concurrent.futures.as_completed(futures):
                mention_id, result = future.result()
                match_results[mention_id] = result

    all_candidates = []
    generated_candidate_count = 0
    conflicts: list[ResolutionConflict] = []
    needs_bootstrap: list[Mention] = []
    bootstrap_reasons: dict[str, tuple[str, ...]] = {}
    merge_requests: list[tuple[str, tuple[str, ...]]] = []
    mention_by_id = {item.mention_id: item for item in mention_tuple}
    for mention_id in sorted(match_results):
        result = match_results[mention_id]
        generated_candidate_count += result.generated_candidate_count
        all_candidates.extend(result.candidates)
        for target_id, code, detail in result.conflict_rows:
            conflicts.append(
                ResolutionConflict(
                    conflict_id=_conflict_id(run_id, mention_id, target_id, code),
                    mention_id=mention_id,
                    target_id=target_id,
                    conflict_code=code,
                    severity="HARD" if "UNKNOWN" not in code else "UNCERTAINTY",
                    detail=detail,
                    run_id=run_id,
                )
            )
        if result.decision is not None:
            decisions[mention_id] = result.decision
        elif result.needs_bootstrap:
            needs_bootstrap.append(mention_by_id[mention_id])
            bootstrap_reasons[mention_id] = result.bootstrap_reason_codes
        else:
            raise AssertionError(f"match result for {mention_id} has no terminal state")
        if result.merge_concept_ids:
            merge_requests.append((mention_id, result.merge_concept_ids))

    merge_plans, blocked_merges = _coalesce_merge_plans(merge_requests, registry)
    for mention_id, reason_codes in blocked_merges.items():
        decision = decisions[mention_id]
        decisions[mention_id] = replace(
            decision,
            reason_codes=tuple(
                dict.fromkeys(
                    decision.reason_codes
                    + tuple(f"AUTO_MERGE_REVIEW:{item}" for item in reason_codes)
                )
            ),
        )
        for code in reason_codes:
            conflicts.append(
                ResolutionConflict(
                    conflict_id=_conflict_id(run_id, mention_id, None, code),
                    mention_id=mention_id,
                    target_id=None,
                    conflict_code=code,
                    severity="HARD",
                    detail="transitive exact-match reconciliation contains a hard contradiction",
                    run_id=run_id,
                )
            )
    for plan in merge_plans:
        for mention_id in plan.trigger_mention_ids:
            decision = decisions.get(mention_id)
            if decision and decision.status == DecisionStatus.MATCHED:
                decisions[mention_id] = replace(
                    decision,
                    concept_id=plan.survivor_concept_id,
                    reason_codes=tuple(dict.fromkeys(decision.reason_codes + plan.reason_codes)),
                )

    bootstrap_proposals, bootstrap_decisions = _bootstrap(
        needs_bootstrap, policy, authority, hints_by_id
    )
    if bootstrap_reasons:
        bootstrap_proposals = [
            replace(
                proposal,
                reason_codes=tuple(
                    dict.fromkeys(
                        proposal.reason_codes
                        + tuple(
                            code
                            for mention_id in proposal.member_mention_ids
                            for code in bootstrap_reasons.get(mention_id, ())
                        )
                    )
                ),
            )
            for proposal in bootstrap_proposals
        ]
        bootstrap_decisions = {
            mention_id: decision
            for proposal in bootstrap_proposals
            for mention_id, decision in _proposal_decisions(proposal).items()
        }
    decisions.update(bootstrap_decisions)
    proposals = _link_instance_proposals(
        bootstrap_proposals, decisions, mention_by_id, registry, policy
    )

    candidates_by_id: dict[str, list[Any]] = defaultdict(list)
    for candidate in all_candidates:
        candidates_by_id[candidate.mention_id].append(candidate)
    finalized = []
    memo = memo or {}
    for mention_id in sorted(decisions):
        decision = decisions[mention_id]
        candidate_hash = candidate_set_hash(candidates_by_id.get(mention_id, ()))
        mention = mention_by_id.get(mention_id)
        if mention is None:
            finalized.append(replace(decision, candidate_set_hash=candidate_hash))
            continue
        key = decision_memo_key(
            mention,
            candidate_hash,
            policy.version,
            policy.content_hash,
            registry.version,
            registry.manifest_hash,
            authority.manifest_hash,
            RESOLVER_CODE_VERSION,
        )
        cached = memo.get(key)
        if cached is not None:
            if cached.mention_id != mention_id or cached.candidate_set_hash != candidate_hash:
                raise ContractError(f"memo entry {key} is inconsistent")
            finalized.append(replace(cached, memo_key=key, memo_reused=True))
        else:
            finalized.append(
                replace(decision, candidate_set_hash=candidate_hash, memo_key=key, memo_reused=False)
            )
    run = ResolutionRun(
        run_id=run_id,
        input_fingerprint=fingerprint,
        base_registry_version=registry.version,
        result_registry_version=registry.version,
        policy_version=policy.version,
        normalization_version=policy.normalization_version,
        authority_manifest_hash=authority.manifest_hash,
        resolver_code_version=RESOLVER_CODE_VERSION,
        mentions=mention_tuple,
        decisions=tuple(finalized),
        proposals=tuple(sorted(proposals, key=lambda item: item.proposal_id)),
        candidates=tuple(sorted(all_candidates, key=lambda item: (item.mention_id, item.rank, item.target_id, item.method))),
        generated_candidate_count=generated_candidate_count,
        invalid_inputs=invalid_tuple,
        effective_controls=tuple(sorted(effective_controls.items())),
    )
    return ResolutionExecution(
        run,
        tuple(sorted(conflicts, key=lambda item: item.conflict_id)),
        capability_manifest,
        merge_plans,
    )


def _coalesce_merge_plans(
    requests: Sequence[tuple[str, tuple[str, ...]]],
    registry: RegistrySnapshot,
) -> tuple[tuple[ConceptMergePlan, ...], dict[str, tuple[str, ...]]]:
    """Union overlapping duplicate-concept requests, then re-check the whole component."""

    if not requests:
        return (), {}
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for _mention_id, concept_ids in requests:
        for concept_id in concept_ids:
            find(concept_id)
        for concept_id in concept_ids[1:]:
            union(concept_ids[0], concept_id)
    components: dict[str, set[str]] = defaultdict(set)
    for concept_id in parent:
        components[find(concept_id)].add(concept_id)
    triggers: dict[str, set[str]] = defaultdict(set)
    for mention_id, concept_ids in requests:
        triggers[find(concept_ids[0])].add(mention_id)

    plans: list[ConceptMergePlan] = []
    blocked: dict[str, tuple[str, ...]] = {}
    concepts = registry.concept_by_id
    for root, component in sorted(components.items()):
        ids = tuple(sorted(component))
        records = [concepts[item] for item in ids]
        reason_codes: set[str] = set()
        if len({item.entity_type for item in records}) != 1:
            reason_codes.add("AUTO_MERGE_ENTITY_TYPE_CONFLICT")
        try:
            _merged_identity_json_values(item.identity_qualifiers_json for item in records)
        except ContractError:
            reason_codes.add("AUTO_MERGE_IDENTITY_QUALIFIER_CONFLICT")
        links: dict[str, set[str]] = defaultdict(set)
        for link in registry.authority_links:
            if registry.active_id(link.concept_id) in component:
                links[link.authority.upper()].add(link.external_id)
        for authority_name, identifiers in links.items():
            if len(identifiers) > 1:
                reason_codes.add(f"AUTO_MERGE_AUTHORITY_CONFLICT:{authority_name}")
        mention_ids = tuple(sorted(triggers[root]))
        if reason_codes:
            reasons = tuple(sorted(reason_codes))
            blocked.update({mention_id: reasons for mention_id in mention_ids})
            continue
        plans.append(
            ConceptMergePlan(
                concept_ids=ids,
                survivor_concept_id=min(ids),
                trigger_mention_ids=mention_ids,
                reason_codes=(
                    "AUTO_MERGE_EXACT_DUPLICATE_CONCEPTS",
                    "AUTO_MERGE_REVIEW:EXISTING_CONCEPT_RECONCILIATION",
                ),
            )
        )
    return tuple(plans), blocked



def _alias_components(mentions, policy) -> dict[str, str]:
    """Map each mention to a shared key when its names overlap another's.

    Extraction emits several names for one entity: onugbu also arrives as
    Vernonia amygdalina, bitter leaf and Gymnanthemum amygdalinum. Two mentions
    describe the same thing when any of their names coincide, so identity is the
    connected component over shared names rather than agreement on one preferred
    label. Names are compared within an entity type, so `spring` the water source
    never joins `spring` the season.
    """
    parent: dict[tuple, tuple] = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    names_by_mention: dict[str, list[tuple]] = {}
    for mention in mentions:
        if mention.identity_scope.value != "CANONICAL":
            continue
        names = [primary_key(mention.atom_text, mention.entity_type)]
        for alias in mention.aliases:
            text = str(alias).strip()
            if text:
                names.append(primary_key(text, mention.entity_type))
        names = [(mention.entity_type, name) for name in names if name]
        if len(names) < 2:
            # A mention with no aliases still anchors its own name, so another
            # mention listing that name as an alias can reach it.
            names = names or []
        names_by_mention[mention.mention_id] = names
        for other in names[1:]:
            union(names[0], other)

    component: dict[str, str] = {}
    alias_backed: set[str] = set()
    for mention_id, names in names_by_mention.items():
        if not names:
            continue
        roots = sorted({find(name) for name in names})
        key = "ALIASGROUP:" + roots[0][1]
        component[mention_id] = key
        if len(names) > 1:
            # this mention contributed an alias link, so the component rests on
            # stated alternative names rather than a bare string coincidence
            alias_backed.add(key)
    return component, alias_backed

def _extraction_aliases(members) -> dict[str, dict]:
    """Alias records the extraction model supplied for a set of mentions.

    Keyed by text, so the same name contributed by several mentions yields one
    registry alias. A name the paper itself stated wins over the same name
    supplied from the model's own knowledge, because the stronger provenance is
    the one worth recording.
    """
    records: dict[str, dict] = {}
    for mention in members:
        try:
            parsed = json.loads(mention.aliases_json or "[]")
        except json.JSONDecodeError:
            continue
        for record in parsed if isinstance(parsed, list) else []:
            if not isinstance(record, Mapping):
                continue
            text = str(record.get("text", "")).strip()
            if not text:
                continue
            existing = records.get(text)
            if existing is None or (record.get("stated_in_paper")
                                    and not existing.get("stated_in_paper")):
                records[text] = dict(record)
    return records


def _bootstrap(
    mentions: Sequence[Mention],
    policy: ResolverPolicy,
    authority: AuthoritySnapshot,
    hints_by_id: Mapping[str, Sequence[AuthorityHint]],
) -> tuple[list[Proposal], dict[str, ResolutionDecision]]:
    proposals: list[Proposal] = []
    decisions: dict[str, ResolutionDecision] = {}
    alias_component, _alias_backed = _alias_components(mentions, policy)
    canonical_base_groups: dict[tuple[Any, ...], list[Mention]] = defaultdict(list)
    instance_groups: dict[tuple[Any, ...], list[Mention]] = defaultdict(list)
    audit_reasons: dict[str, tuple[str, ...]] = {}

    for mention in sorted(mentions, key=lambda item: item.mention_id):
        rules = [policy.qualifier_rule(item.kind) for item in mention.qualifiers]
        mention_audit: list[str] = []
        if any(rule and rule.semantic == "REVIEW_ONLY" for rule in rules):
            mention_audit.append("AUTO_MERGE_REVIEW:UNMODELED_QUALIFIER")
        blocking = sorted(set(mention.source_flags) & _AUTOMATIC_BLOCKING_FLAGS)
        mention_audit.extend(f"AUTO_MERGE_REVIEW:SOURCE_FLAG:{item}" for item in blocking)
        audit_reasons[mention.mention_id] = tuple(mention_audit)
        if mention.entity_type.value == "OTHER":
            decisions[mention.mention_id] = ResolutionDecision(
                mention.mention_id,
                DecisionStatus.REVIEW_REQUIRED,
                DecisionMethod.NO_SAFE_DECISION,
                reason_codes=("OTHER_TYPE_REQUIRES_REVIEW",),
            )
            continue
        keys = build_keys(mention, policy)
        qualifier_key = keys.qualifier_signature
        if mention.identity_scope.value == "STUDY_INSTANCE":
            key = (
                mention.paper_id,
                mention.context_id,
                mention.instance_local_id or mention.source_mention_id,
                mention.entity_type,
                keys.primary,
                qualifier_key,
            )
            instance_groups[key].append(mention)
            continue
        hints = hints_by_id.get(mention.mention_id, ())
        if hints:
            component = _authority_component_key(hints, authority)
            key = ("AUTHORITY", mention.entity_type, component)
            canonical_base_groups[key].append(mention)
        else:
            component = alias_component.get(mention.mention_id, keys.primary)
            canonical_base_groups[("NORMALIZED", mention.entity_type, component)].append(mention)

    for key, base_members in sorted(canonical_base_groups.items(), key=lambda item: repr(item[0])):
        for members, qualifier_audit in _partition_compatible_mentions(base_members, policy):
            if len(members) == 1 and not policy.type_policy(members[0].entity_type).self_seed:
                mention = members[0]
                decisions[mention.mention_id] = ResolutionDecision(
                    mention.mention_id,
                    DecisionStatus.UNRESOLVED,
                    DecisionMethod.NO_SAFE_DECISION,
                    reason_codes=("SELF_SEED_DISABLED_BY_TYPE_POLICY",),
                )
                continue
            method = (
                DecisionMethod.CLEAN_SINGLETON_BOOTSTRAP
                if len(members) == 1
                else DecisionMethod.EXACT_BOOTSTRAP_GROUP
            )
            reasons: list[str] = [
                "EXACT_AUTHORITY_BOOTSTRAP" if key[0] == "AUTHORITY" else "TYPE_SAFE_NORMALIZED_BOOTSTRAP"
            ]
            reasons.extend(qualifier_audit)
            for member in members:
                reasons.extend(audit_reasons.get(member.mention_id, ()))
            if key[0] == "NORMALIZED" and _homonym_audit_type(members[0]):
                reasons.append("AUTO_MERGE_REVIEW:HOMONYM_PRONE_EXACT_NAME")
            proposal = _concept_proposal(
                members,
                policy,
                hints_by_id,
                method,
                tuple(dict.fromkeys(reasons)),
                auto=True,
            )
            proposals.append(proposal)
            decisions.update(_proposal_decisions(proposal))
    for _key, members in sorted(instance_groups.items(), key=lambda item: repr(item[0])):
        proposal_identifier = proposal_id("INSTANCE", [item.mention_id for item in members], policy.version)
        proposal = Proposal(
            proposal_id=proposal_identifier,
            proposal_kind=ProposalKind.INSTANCE,
            member_mention_ids=tuple(sorted(item.mention_id for item in members)),
            preferred_label=_preferred_label(members),
            entity_type=members[0].entity_type,
            paper_id=members[0].paper_id,
            context_id=members[0].context_id,
            primary_concept_id=None,
            primary_concept_proposal_id=None,
            authority_keys=(),
            auto_approved=True,
            method=DecisionMethod.EXACT_INSTANCE_GROUP,
            reason_codes=("PAPER_CONTEXT_SCOPED_EXACT_GROUP",),
        )
        proposals.append(proposal)
        decisions.update(_proposal_decisions(proposal))
    return proposals, decisions


def _partition_compatible_mentions(
    members: Sequence[Mention], policy: ResolverPolicy
) -> list[tuple[list[Mention], tuple[str, ...]]]:
    """Group exact names without using missing qualifiers as a hard separator.

    An unqualified mention joins a single compatible qualified identity.  When
    two explicit values conflict (for example Niger river and Niger basin), an
    unqualified bridge is kept separate rather than arbitrarily choosing one or
    collapsing both.
    """

    by_signature: dict[tuple[tuple[str, str], ...], list[Mention]] = defaultdict(list)
    for mention in sorted(members, key=lambda item: item.mention_id):
        by_signature[qualifier_signature(mention.qualifiers, policy)].append(mention)
    clusters = [
        (dict(signature), list(values))
        for signature, values in sorted(by_signature.items(), key=lambda item: repr(item[0]))
        if signature
    ]

    # Merge only mutually unique compatible clusters. This admits complementary
    # context while preventing an underspecified record from bridging two
    # explicitly contradictory identities.
    changed = True
    while changed and len(clusters) > 1:
        changed = False
        compatibility = {
            index: [
                other
                for other in range(len(clusters))
                if other != index and _qualifier_maps_compatible(clusters[index][0], clusters[other][0])
            ]
            for index in range(len(clusters))
        }
        pair = next(
            (
                (index, others[0])
                for index, others in compatibility.items()
                if len(others) == 1 and compatibility.get(others[0]) == [index]
            ),
            None,
        )
        if pair is None:
            break
        left, right = sorted(pair)
        merged_map = {**clusters[left][0], **clusters[right][0]}
        merged_members = clusters[left][1] + clusters[right][1]
        clusters = [item for index, item in enumerate(clusters) if index not in {left, right}]
        clusters.append((merged_map, sorted(merged_members, key=lambda item: item.mention_id)))
        clusters.sort(key=lambda item: repr(sorted(item[0].items())))
        changed = True

    unqualified = list(by_signature.get((), ()))
    if not clusters:
        clusters = [({}, unqualified)]
        unqualified = []
    elif len(clusters) == 1:
        clusters[0][1].extend(unqualified)
        clusters[0][1].sort(key=lambda item: item.mention_id)
        unqualified = []
    elif unqualified:
        clusters.append(({}, unqualified))

    result: list[tuple[list[Mention], tuple[str, ...]]] = []
    for qualifier_map, grouped in clusters:
        signatures = {qualifier_signature(item.qualifiers, policy) for item in grouped}
        audit: list[str] = []
        kinds = set(qualifier_map)
        if len(signatures) > 1:
            for kind in sorted(kinds):
                if any(kind not in dict(signature) for signature in signatures):
                    audit.append(f"AUTO_MERGE_REVIEW:MISSING_IDENTITY_QUALIFIER:{kind}")
        if not qualifier_map and len(clusters) > 1:
            audit.append("AUTO_MERGE_REVIEW:AMBIGUOUS_MISSING_IDENTITY_QUALIFIER")
        result.append((sorted(grouped, key=lambda item: item.mention_id), tuple(audit)))
    return result


def _qualifier_maps_compatible(left: Mapping[str, str], right: Mapping[str, str]) -> bool:
    return all(left[kind] == right[kind] for kind in set(left) & set(right))


def _homonym_audit_type(mention: Mention) -> bool:
    if mention.entity_type.value in {"PLACE", "ORGANISM"}:
        return True
    if mention.entity_type.value != "ENVIRONMENTAL_FEATURE":
        return False
    generic = {"air", "groundwater", "rainfall", "river water", "soil", "surface water", "wastewater", "water"}
    return primary_key(mention.atom_text, mention.entity_type) not in generic


def _concept_proposal(
    members: Sequence[Mention],
    policy: ResolverPolicy,
    hints_by_id: Mapping[str, Sequence[AuthorityHint]],
    method: DecisionMethod,
    reasons: tuple[str, ...],
    auto: bool,
) -> Proposal:
    authority_keys = tuple(
        sorted(
            {
                (hint.authority.upper(), hint.external_id, hint.snapshot_version, hint.source)
                for member in members
                for hint in hints_by_id.get(member.mention_id, ())
            }
        )
    )
    return Proposal(
        proposal_id=proposal_id("CONCEPT", [item.mention_id for item in members], policy.version),
        proposal_kind=ProposalKind.CONCEPT,
        member_mention_ids=tuple(sorted(item.mention_id for item in members)),
        preferred_label=_preferred_label(members),
        entity_type=members[0].entity_type,
        paper_id=None,
        context_id=None,
        primary_concept_id=None,
        primary_concept_proposal_id=None,
        authority_keys=authority_keys,
        auto_approved=auto,
        method=method,
        reason_codes=reasons,
    )


def _proposal_decisions(proposal: Proposal) -> dict[str, ResolutionDecision]:
    status = (
        DecisionStatus.NEW_CONCEPT_PROPOSED
        if proposal.proposal_kind == ProposalKind.CONCEPT
        else DecisionStatus.NEW_INSTANCE_PROPOSED
    )
    return {
        mention_id: ResolutionDecision(
            mention_id,
            status,
            proposal.method,
            proposal_id=proposal.proposal_id,
            reason_codes=proposal.reason_codes,
        )
        for mention_id in proposal.member_mention_ids
    }


def _link_instance_proposals(
    proposals: Sequence[Proposal],
    decisions: Mapping[str, ResolutionDecision],
    mentions: Mapping[str, Mention],
    registry: RegistrySnapshot,
    policy: ResolverPolicy,
) -> list[Proposal]:
    proposal_by_id = {item.proposal_id: item for item in proposals}
    canonical_refs_by_source: dict[str, set[tuple[str, str, Any]]] = defaultdict(set)
    for mention_id, decision in decisions.items():
        mention = mentions.get(mention_id)
        if mention is None or mention.identity_scope.value != "CANONICAL":
            continue
        if decision.concept_id:
            concept = registry.concept_by_id.get(registry.active_id(decision.concept_id))
            if concept:
                canonical_refs_by_source[mention.source_mention_id].add(
                    ("CONCEPT", concept.concept_id, concept.entity_type)
                )
        elif decision.proposal_id:
            proposal = proposal_by_id.get(decision.proposal_id)
            if proposal and proposal.proposal_kind == ProposalKind.CONCEPT:
                canonical_refs_by_source[mention.source_mention_id].add(
                    ("PROPOSAL", proposal.proposal_id, proposal.entity_type)
                )
    linked = []
    for proposal in proposals:
        if proposal.proposal_kind != ProposalKind.INSTANCE:
            linked.append(proposal)
            continue
        refs = set()
        allowed = set(policy.instance_concept_map.get(proposal.entity_type, ()))
        for mention_id in proposal.member_mention_ids:
            source_id = mentions[mention_id].source_mention_id
            refs.update(item for item in canonical_refs_by_source.get(source_id, ()) if item[2] in allowed)
        if len(refs) == 1:
            kind, identifier, _entity_type = next(iter(refs))
            linked.append(
                replace(
                    proposal,
                    primary_concept_id=identifier if kind == "CONCEPT" else None,
                    primary_concept_proposal_id=identifier if kind == "PROPOSAL" else None,
                    reason_codes=proposal.reason_codes + ("UNAMBIGUOUS_SOURCE_GROUP_INSTANCE_OF",),
                )
            )
        elif len(refs) > 1:
            linked.append(replace(proposal, reason_codes=proposal.reason_codes + ("AMBIGUOUS_INSTANCE_OF_LEFT_NULL",)))
        else:
            linked.append(proposal)
    return linked


def commit_resolution_run(
    execution: ResolutionExecution,
    registry: RegistrySnapshot,
    policy: ResolverPolicy,
    *,
    authority_snapshot: AuthoritySnapshot | None = None,
    approved_proposal_ids: Iterable[str] = (),
) -> CommitResult:
    """Commit approved proposals with one deterministic writer."""

    run = execution.run
    authority = authority_snapshot or AuthoritySnapshot.empty()
    if run.base_registry_version != registry.version:
        raise ContractError(
            f"run targets {run.base_registry_version}, but commit registry is {registry.version}"
        )
    if run.policy_version != policy.version or run.authority_manifest_hash != authority.manifest_hash:
        raise ContractError("run policy/authority fingerprint does not match commit inputs")
    registry.validate(authority if authority.records else None)
    approved = set(approved_proposal_ids)
    known_proposals = {item.proposal_id for item in run.proposals}
    unknown = approved - known_proposals
    if unknown:
        raise ContractError(f"approved proposal IDs do not exist in run: {sorted(unknown)}")
    accepted = [item for item in run.proposals if item.auto_approved or item.proposal_id in approved]

    # Reconcile duplicate existing concepts first. Plans are produced only from
    # aggregated exact compatible evidence and were revalidated as complete
    # transitive components during resolve_batch.
    working_registry = registry
    merge_events: list[ResolutionEvent] = []
    for plan in execution.merge_plans:
        active_ids = tuple(sorted({working_registry.active_id(item) for item in plan.concept_ids}))
        if len(active_ids) < 2:
            continue
        survivor = min(active_ids)
        change = merge_concepts(
            working_registry,
            active_ids,
            survivor,
            run.run_id,
            "SYSTEM:EXACT_CONNECTIVITY",
            "AUTO_MERGE_EXACT_DUPLICATE_CONCEPTS",
            authority if authority.records else None,
        )
        working_registry = change.snapshot
        merge_events.extend(change.events)

    new_version = working_registry.next_version()
    mention_by_id = {item.mention_id: item for item in run.mentions}
    concepts = list(working_registry.canonical_entities)
    instances = list(working_registry.entity_instances)
    aliases = list(working_registry.canonical_aliases)
    links = list(working_registry.authority_links)
    events: list[ResolutionEvent] = []
    proposal_targets: dict[str, str] = {}

    for proposal in sorted((item for item in accepted if item.proposal_kind == ProposalKind.CONCEPT), key=lambda item: item.proposal_id):
        seed = min(proposal.member_mention_ids)
        concept_id = stable_concept_id(seed)
        if concept_id in working_registry.concept_by_id:
            raise ContractError(f"new concept ID collides with registry: {concept_id}")
        members = [mention_by_id[item] for item in proposal.member_mention_ids]
        identity_json = _merged_identity_qualifiers_json(members, policy)
        concepts.append(
            CanonicalEntity(
                concept_id=concept_id,
                preferred_label=proposal.preferred_label,
                entity_type=proposal.entity_type,
                identity_qualifiers_json=identity_json,
                lifecycle_status=LifecycleStatus.ACTIVE,
                seed_mention_id=seed,
                created_run_id=run.run_id,
                updated_run_id=run.run_id,
                registry_version=new_version,
                provenance_json=json.dumps(
                    {
                        "proposal_id": proposal.proposal_id,
                        "mention_ids": list(proposal.member_mention_ids),
                        "source_evidence_ids": sorted({item.source_evidence_id for item in members}),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )
        proposal_targets[proposal.proposal_id] = concept_id
        event_id = stable_event_id(EventType.CREATE_CONCEPT, run.run_id, concept_id, proposal.proposal_id)
        events.append(ResolutionEvent(event_id, EventType.CREATE_CONCEPT, concept_id, proposal.proposal_id, run.run_id, new_version, proposal.reason_codes))
        for text in sorted({proposal.preferred_label} | {item.surface_text for item in members} | {item.atom_text for item in members}):
            key = primary_key(text, proposal.entity_type)
            source = "BOOTSTRAP_PRIMARY" if text == proposal.preferred_label else "BOOTSTRAP_VARIANT"
            alias_id = stable_alias_id(concept_id, key, proposal.entity_type, source)
            aliases.append(
                CanonicalAlias(
                    alias_id, concept_id, text, key, proposal.entity_type,
                    None, None, source, AliasTrust.MODEL_SUPPLIED, run.run_id, new_version,
                    provenance_json=_alias_provenance_json(members, "BOOTSTRAP_NAME"),
                )
            )
        # The names extraction supplied for these mentions become registry
        # aliases too. Without this the alternative names are used once, inside
        # the batch that produced them, and then lost: a later run meeting
        # "bitter leaf" would have no way back to the concept it named, and the
        # corpus would never accumulate what the model already knew.
        for text, record in sorted(_extraction_aliases(members).items()):
            key = primary_key(text, proposal.entity_type)
            if not key:
                continue
            alias_identity = (
                f"EXTRACTION|{record.get('kind') or ''}|{record.get('language') or '-'}|"
                f"{int(bool(record.get('stated_in_paper')))}"
            )
            alias_id = stable_alias_id(concept_id, key, proposal.entity_type, alias_identity)
            aliases.append(
                CanonicalAlias(
                    alias_id, concept_id, text, key, proposal.entity_type,
                    record.get("language") or None, None, "EXTRACTION",
                    AliasTrust.MODEL_SUPPLIED, run.run_id, new_version,
                    alias_kind=str(record.get("kind") or ""),
                    stated_in_paper=bool(record.get("stated_in_paper")),
                    provenance_json=_alias_provenance_json(
                        [item for item in members if text in item.aliases], "MODEL_ALIAS"
                    ),
                )
            )
        for authority_name, external_id, snapshot_version, source in proposal.authority_keys:
            record = authority.get(authority_name, external_id)
            if record is None or record.entity_type != proposal.entity_type:
                raise ContractError(f"proposal authority is absent or type-incompatible: {authority_name}:{external_id}")
            links.append(
                AuthorityLink(
                    stable_authority_link_id(concept_id, authority_name, external_id),
                    concept_id, authority_name, external_id, snapshot_version, source, run.run_id, new_version,
                )
            )
            for text in (record.preferred_label,) + record.aliases:
                key = primary_key(text, proposal.entity_type)
                aliases.append(
                    CanonicalAlias(
                        stable_alias_id(concept_id, key, proposal.entity_type, f"AUTHORITY:{authority_name}"),
                        concept_id, text, key, proposal.entity_type, None, None,
                        f"AUTHORITY:{authority_name}", AliasTrust.AUTHORITY, run.run_id, new_version,
                    )
                )

    for proposal in sorted((item for item in accepted if item.proposal_kind == ProposalKind.INSTANCE), key=lambda item: item.proposal_id):
        seed = min(proposal.member_mention_ids)
        if not proposal.paper_id or not proposal.context_id:
            raise ContractError(f"instance proposal {proposal.proposal_id} lacks paper/context scope")
        instance_id = stable_instance_id(seed, proposal.paper_id, proposal.context_id)
        if instance_id in registry.instance_by_id:
            raise ContractError(f"new instance ID collides with registry: {instance_id}")
        members = [mention_by_id[item] for item in proposal.member_mention_ids]
        identity_json = _merged_identity_qualifiers_json(members, policy)
        concept_id = proposal.primary_concept_id
        if proposal.primary_concept_proposal_id:
            concept_id = proposal_targets.get(proposal.primary_concept_proposal_id)
            if concept_id is None:
                # The concept proposal was not approved. Never create a dangling edge.
                concept_id = None
        instances.append(
            EntityInstance(
                instance_id=instance_id,
                paper_id=proposal.paper_id,
                context_id=proposal.context_id,
                local_label=proposal.preferred_label,
                entity_type=proposal.entity_type,
                concept_id=concept_id,
                identity_qualifiers_json=identity_json,
                source_mention_ids_json=json.dumps(
                    sorted({mention_by_id[item].source_mention_id for item in proposal.member_mention_ids}),
                    separators=(",", ":"),
                ),
                created_run_id=run.run_id,
                updated_run_id=run.run_id,
                registry_version=new_version,
            )
        )
        proposal_targets[proposal.proposal_id] = instance_id
        event_id = stable_event_id(EventType.CREATE_INSTANCE, run.run_id, instance_id, proposal.proposal_id)
        events.append(ResolutionEvent(event_id, EventType.CREATE_INSTANCE, instance_id, proposal.proposal_id, run.run_id, new_version, proposal.reason_codes))

    # A matched mention teaches the registry every name it carried. This is a
    # deterministic enrichment step, not a new concept, and keeps the full
    # paper/evidence/model provenance needed to audit a later alias-driven hit.
    matched_aliases_before = _dedupe_alias_rows(aliases)
    for decision in sorted(run.decisions, key=lambda item: item.mention_id):
        if decision.status != DecisionStatus.MATCHED or not decision.concept_id:
            continue
        mention = mention_by_id[decision.mention_id]
        concept_id = working_registry.active_id(decision.concept_id)
        target = working_registry.concept_by_id.get(concept_id)
        if target is None or target.entity_type != mention.entity_type:
            raise ContractError(f"matched decision {mention.mention_id} targets a missing/incompatible concept")
        aliases.extend(_learned_alias_rows(mention, concept_id, run.run_id, new_version))

    deduped_aliases = _dedupe_alias_rows(aliases)
    has_registry_writes = bool(accepted) or deduped_aliases != matched_aliases_before
    if not has_registry_writes:
        committed_run = replace(run, result_registry_version=working_registry.version)
        return CommitResult(
            working_registry,
            committed_run,
            tuple(merge_events),
            registry_diff(registry, working_registry),
        )

    new_registry = RegistrySnapshot(
        version=new_version,
        canonical_entities=tuple(sorted(concepts, key=lambda item: item.concept_id)),
        entity_instances=tuple(sorted(instances, key=lambda item: item.instance_id)),
        canonical_aliases=deduped_aliases,
        authority_links=_dedupe_authority_rows(links),
        entity_relations=working_registry.entity_relations,
        redirects=working_registry.redirects,
        constraints=working_registry.constraints,
        events=tuple(sorted(working_registry.events + tuple(events), key=lambda item: item.event_id)),
    )
    new_registry.validate(authority if authority.records else None)

    committed_decisions = []
    for decision in run.decisions:
        target = proposal_targets.get(decision.proposal_id or "")
        if target is None:
            committed_decisions.append(decision)
            continue
        proposal = next(item for item in accepted if item.proposal_id == decision.proposal_id)
        if proposal.proposal_kind == ProposalKind.CONCEPT:
            committed_decisions.append(
                replace(
                    decision,
                    status=DecisionStatus.MATCHED,
                    concept_id=target,
                    instance_id=None,
                    proposal_id=None,
                    reason_codes=decision.reason_codes + ("COMMITTED",),
                )
            )
        else:
            concept_id = proposal.primary_concept_id
            if proposal.primary_concept_proposal_id:
                concept_id = proposal_targets.get(proposal.primary_concept_proposal_id)
            committed_decisions.append(
                replace(
                    decision,
                    status=DecisionStatus.MATCHED,
                    concept_id=concept_id,
                    instance_id=target,
                    proposal_id=None,
                    reason_codes=decision.reason_codes + ("COMMITTED",),
                )
            )
    committed_run = replace(
        run,
        decisions=tuple(committed_decisions),
        result_registry_version=new_registry.version,
    )
    return CommitResult(
        new_registry,
        committed_run,
        tuple(merge_events + events),
        registry_diff(registry, new_registry),
    )


def _authority_component_key(hints: Sequence[AuthorityHint], authority: AuthoritySnapshot) -> tuple[str, str]:
    nodes = {(item.authority.upper(), item.external_id) for item in hints}
    frontier = list(nodes)
    graph: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for left_authority, left_id, right_authority, right_id in authority.trusted_crosswalks:
        left, right = (left_authority, left_id), (right_authority, right_id)
        graph[left].add(right)
        graph[right].add(left)
    seen = set(nodes)
    while frontier:
        current = frontier.pop()
        for neighbor in graph.get(current, ()):
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return min(seen)


def _preferred_label(members: Sequence[Mention]) -> str:
    seed = min(members, key=lambda item: item.mention_id)
    return seed.atom_text.strip()


def _identity_qualifiers_json(mention: Mention, policy: ResolverPolicy) -> str:
    signature = qualifier_signature(mention.qualifiers, policy)
    value = [{"kind": kind, "value_text": text} for kind, text in signature]
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _learned_alias_rows(
    mention: Mention, concept_id: str, run_id: str, registry_version: str
) -> list[CanonicalAlias]:
    rows: list[CanonicalAlias] = []
    provenance = _alias_provenance_json([mention], "MATCHED_NAME")
    for text in sorted({mention.surface_text, mention.atom_text}):
        key = primary_key(text, mention.entity_type)
        if not key:
            continue
        source = "MATCHED_MENTION"
        rows.append(
            CanonicalAlias(
                stable_alias_id(concept_id, key, mention.entity_type, source),
                concept_id,
                text,
                key,
                mention.entity_type,
                mention.language,
                mention.country_code,
                source,
                AliasTrust.MODEL_SUPPLIED,
                run_id,
                registry_version,
                provenance_json=provenance,
            )
        )
    records = json.loads(mention.aliases_json)
    for record in records:
        text = record["text"]
        key = primary_key(text, mention.entity_type)
        if not key:
            continue
        kind = record["kind"]
        language = record["language"] or None
        stated = record["stated_in_paper"]
        identity = f"EXTRACTION|{kind}|{language or '-'}|{int(stated)}"
        rows.append(
            CanonicalAlias(
                stable_alias_id(concept_id, key, mention.entity_type, identity),
                concept_id,
                text,
                key,
                mention.entity_type,
                language,
                mention.country_code,
                "EXTRACTION",
                AliasTrust.MODEL_SUPPLIED,
                run_id,
                registry_version,
                alias_kind=kind,
                stated_in_paper=stated,
                provenance_json=_alias_provenance_json([mention], "MODEL_ALIAS"),
            )
        )
    return rows


def _alias_provenance_json(mentions: Sequence[Mention], origin: str) -> str:
    records = [
        {
            "mention_id": item.mention_id,
            "paper_id": item.paper_id,
            "source_evidence_id": item.source_evidence_id,
            "source_mention_id": item.source_mention_id,
        }
        for item in sorted(mentions, key=lambda value: value.mention_id)
    ]
    return json.dumps(
        {"origin": origin, "records": records},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _merged_identity_qualifiers_json(
    mentions: Sequence[Mention], policy: ResolverPolicy
) -> str:
    return _merged_identity_json_values(_identity_qualifiers_json(item, policy) for item in mentions)


def _merged_identity_json_values(values: Iterable[str]) -> str:
    merged: dict[str, str] = {}
    for raw in values:
        parsed = json.loads(raw or "[]")
        if not isinstance(parsed, list):
            raise ContractError("identity qualifiers must be a JSON list")
        for item in parsed:
            if not isinstance(item, Mapping) or set(item) != {"kind", "value_text"}:
                raise ContractError("identity qualifier record is malformed")
            kind, value = str(item["kind"]), str(item["value_text"])
            existing = merged.setdefault(kind, value)
            if existing != value:
                raise ContractError(f"conflicting identity qualifier {kind}: {existing!r} != {value!r}")
    return json.dumps(
        [{"kind": kind, "value_text": value} for kind, value in sorted(merged.items())],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _mention_fingerprint(
    mentions: Sequence[Mention], invalid_mentions: Sequence[InvalidMentionRecord]
) -> str:
    payload = {
        "mentions": [
            {
                "mention_id": item.mention_id,
                "paper_id": item.paper_id,
                "source_mention_id": item.source_mention_id,
                "source_evidence_id": item.source_evidence_id,
                "atom_text": item.atom_text,
                "entity_type": item.entity_type.value,
                "identity_scope": item.identity_scope.value,
                "qualifiers": [(q.kind, q.value_text) for q in item.qualifiers],
                "aliases_json": item.aliases_json,
                "source_flags": list(item.source_flags),
            }
            for item in mentions
        ],
        "invalid": [(item.mention_id, item.raw_json, item.error_codes) for item in invalid_mentions],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _run_id(
    input_fingerprint: str,
    registry: RegistrySnapshot,
    policy: ResolverPolicy,
    authority: AuthoritySnapshot,
    capabilities: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> str:
    payload = {
        "input": input_fingerprint,
        "registry": registry.version,
        "registry_manifest": registry.manifest_hash,
        "policy": policy.version,
        "policy_hash": policy.content_hash,
        "authority": authority.manifest_hash,
        "capabilities": capabilities,
        "controls": controls,
        "code": RESOLVER_CODE_VERSION,
    }
    return f"RUN-{hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:32]}"


def _conflict_id(run_id: str, mention_id: str, target_id: str | None, code: str) -> str:
    payload = f"{run_id}\x1f{mention_id}\x1f{target_id or ''}\x1f{code}"
    return f"CNF-{hashlib.sha256(payload.encode()).hexdigest()[:32]}"


def _dedupe_alias_rows(items: Sequence[CanonicalAlias]) -> tuple[CanonicalAlias, ...]:
    chosen: dict[tuple[Any, ...], CanonicalAlias] = {}
    for item in sorted(items, key=lambda value: value.alias_id):
        key = (
            item.concept_id,
            item.normalized_key,
            item.entity_type,
            item.source,
            item.language,
            item.region,
            item.alias_kind,
            item.stated_in_paper,
        )
        existing = chosen.get(key)
        if existing is None:
            chosen[key] = item
        else:
            chosen[key] = replace(
                existing,
                provenance_json=_merge_alias_provenance(
                    existing.provenance_json, item.provenance_json
                ),
            )
    return tuple(sorted(chosen.values(), key=lambda item: item.alias_id))


def _merge_alias_provenance(left: str, right: str) -> str:
    records: dict[tuple[str, str, str, str], dict[str, str]] = {}
    origins: set[str] = set()
    for raw in (left, right):
        try:
            value = json.loads(raw or "{}")
        except json.JSONDecodeError:
            value = {}
        if not isinstance(value, Mapping):
            continue
        origin = value.get("origin")
        if isinstance(origin, str) and origin:
            origins.add(origin)
        origins_value = value.get("origins")
        if isinstance(origins_value, list):
            origins.update(str(item) for item in origins_value if str(item))
        for record in value.get("records", []) if isinstance(value.get("records", []), list) else []:
            if not isinstance(record, Mapping):
                continue
            clean = {
                "mention_id": str(record.get("mention_id", "")),
                "paper_id": str(record.get("paper_id", "")),
                "source_evidence_id": str(record.get("source_evidence_id", "")),
                "source_mention_id": str(record.get("source_mention_id", "")),
            }
            key = tuple(clean[name] for name in ("mention_id", "paper_id", "source_evidence_id", "source_mention_id"))
            records[key] = clean
    return json.dumps(
        {"origins": sorted(origins), "records": [records[key] for key in sorted(records)]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _dedupe_authority_rows(items: Sequence[AuthorityLink]) -> tuple[AuthorityLink, ...]:
    chosen = {}
    for item in sorted(items, key=lambda value: value.authority_link_id):
        key = (item.concept_id, item.authority.upper(), item.external_id)
        chosen.setdefault(key, item)
    return tuple(sorted(chosen.values(), key=lambda item: item.authority_link_id))
