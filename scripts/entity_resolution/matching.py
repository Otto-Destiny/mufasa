"""Type-safe candidate generation and conservative decisions."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np
from rapidfuzz import fuzz

from .authorities import AuthoritySnapshot
from .contracts import (
    AliasTrust,
    AuthorityHint,
    Candidate,
    ContractError,
    DecisionMethod,
    DecisionStatus,
    EntityType,
    IdentityScope,
    Mention,
    Qualifier,
    ResolutionDecision,
)
from .normalization import build_keys, character_ngrams, primary_key, qualifier_signature, token_key
from .policy import ResolverPolicy
from .registry import RegistrySnapshot
from .validation import automatic_evidence_is_safe


@dataclass(frozen=True)
class Target:
    target_kind: str
    target_id: str
    label: str
    entity_type: EntityType
    identity_qualifiers: tuple[Qualifier, ...]
    paper_id: str | None = None
    context_id: str | None = None
    primary_concept_id: str | None = None
    source_mention_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LexicalPoolEntry:
    target_id: str
    token_hits: int
    ngram_hits: int
    rarity_score: float


class EmbeddingRecall(Protocol):
    """Real embedding recall backend supplied by the build plane."""

    model_id: str
    model_hash: str
    implementation_version: str
    mention_vectors_hash: str
    target_vectors_hash: str
    vector_set_fingerprint: str

    def search(self, mention: Mention, eligible_target_ids: Sequence[str], top_k: int) -> Sequence[tuple[str, float]]:
        """Return target IDs with cosine similarity in [0, 1]."""


class PrecomputedEmbeddingRecall:
    """Deterministic cosine search over pinned, precomputed vectors.

    This class never downloads a model or substitutes lexical scores.  Missing
    query/target vectors fail explicitly when the backend is requested.
    """

    implementation_version = "numpy-exact-cosine-v1"

    def __init__(
        self,
        mention_vectors: Mapping[str, Sequence[float]],
        target_vectors: Mapping[str, Sequence[float]],
        *,
        model_id: str,
        model_hash: str,
        mention_vectors_hash: str,
        target_vectors_hash: str,
        vector_set_fingerprint: str,
    ) -> None:
        if not all((model_id, model_hash, mention_vectors_hash, target_vectors_hash, vector_set_fingerprint)):
            raise ContractError("embedding backend requires pinned model and vector artifact identities")
        self.model_id = model_id
        self.model_hash = model_hash
        self.mention_vectors_hash = mention_vectors_hash
        self.target_vectors_hash = target_vectors_hash
        self.vector_set_fingerprint = vector_set_fingerprint
        self._mention_vectors = {key: _unit_vector(value, f"mention {key}") for key, value in mention_vectors.items()}
        self._target_vectors = {key: _unit_vector(value, f"target {key}") for key, value in target_vectors.items()}
        dimensions = {len(value) for value in self._mention_vectors.values()} | {
            len(value) for value in self._target_vectors.values()
        }
        if len(dimensions) > 1:
            raise ContractError(f"embedding vectors have inconsistent dimensions {sorted(dimensions)}")

    def search(self, mention: Mention, eligible_target_ids: Sequence[str], top_k: int) -> Sequence[tuple[str, float]]:
        try:
            query = self._mention_vectors[mention.mention_id]
        except KeyError as exc:
            raise ContractError(f"embedding vector missing for mention {mention.mention_id}") from exc
        missing = [target for target in eligible_target_ids if target not in self._target_vectors]
        if missing:
            raise ContractError(f"embedding vectors missing for {len(missing)} eligible targets; first={missing[0]}")
        scored = []
        for target_id in eligible_target_ids:
            cosine = float(np.dot(query, self._target_vectors[target_id]))
            score = max(0.0, min(1.0, (cosine + 1.0) / 2.0)) * 100.0
            scored.append((target_id, score))
        return tuple(sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k])


def _unit_vector(value: Sequence[float], label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ContractError(f"{label} vector must be a finite non-empty one-dimensional array")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ContractError(f"{label} vector has zero norm")
    return vector / norm


def _qualifiers_from_json(raw: str) -> tuple[Qualifier, ...]:
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise ContractError(f"registry identity_qualifiers_json is invalid: {exc}") from exc
    if not isinstance(value, list):
        raise ContractError("registry identity_qualifiers_json must be a list")
    qualifiers: list[Qualifier] = []
    for item in value:
        if isinstance(item, dict):
            qualifiers.append(Qualifier.from_mapping(item))
        elif isinstance(item, list) and len(item) == 2:
            qualifiers.append(Qualifier(item[0], item[1]))
        else:
            raise ContractError("registry identity qualifier must be {kind,value_text}")
    return tuple(sorted(set(qualifiers)))


class ResolverIndex:
    def __init__(self, registry: RegistrySnapshot, policy: ResolverPolicy) -> None:
        self.registry = registry
        self.policy = policy
        self.targets: dict[str, Target] = {}
        self.primary: dict[tuple[EntityType, str], list[str]] = {}
        self.aliases: dict[tuple[EntityType, str], list[tuple[str, AliasTrust]]] = {}
        self.authorities: dict[tuple[str, str], list[str]] = {}
        self.instances: dict[tuple[str, str, EntityType, str], list[str]] = {}
        self.instance_sources: dict[tuple[str, str], list[str]] = {}
        self.concepts_by_type: dict[EntityType, list[str]] = defaultdict(list)
        self.instances_by_scope: dict[tuple[str, str, EntityType], list[str]] = defaultdict(list)
        labels_by_target: dict[str, list[str]] = defaultdict(list)

        for entity in registry.active_concepts:
            target = Target(
                "CONCEPT",
                entity.concept_id,
                entity.preferred_label,
                entity.entity_type,
                _qualifiers_from_json(entity.identity_qualifiers_json),
            )
            self.targets[target.target_id] = target
            self.concepts_by_type[target.entity_type].append(target.target_id)
            labels_by_target[target.target_id].append(target.label)
            self.primary.setdefault(
                (target.entity_type, primary_key(target.label, target.entity_type)), []
            ).append(target.target_id)
        for alias in registry.canonical_aliases:
            concept_id = registry.active_id(alias.concept_id)
            if concept_id not in self.targets:
                continue
            labels_by_target[concept_id].append(alias.alias_text)
            self.aliases.setdefault((alias.entity_type, alias.normalized_key), []).append(
                (concept_id, alias.trust_level)
            )
        for link in registry.authority_links:
            concept_id = registry.active_id(link.concept_id)
            if concept_id in self.targets:
                self.authorities.setdefault((link.authority.upper(), link.external_id), []).append(concept_id)
        for instance in registry.entity_instances:
            target = Target(
                "INSTANCE",
                instance.instance_id,
                instance.local_label,
                instance.entity_type,
                _qualifiers_from_json(instance.identity_qualifiers_json),
                instance.paper_id,
                instance.context_id,
                instance.concept_id,
                tuple(sorted(set(json.loads(instance.source_mention_ids_json)))),
            )
            self.targets[target.target_id] = target
            self.instances_by_scope[(instance.paper_id, instance.context_id, instance.entity_type)].append(
                instance.instance_id
            )
            labels_by_target[target.target_id].append(target.label)
            key = (
                instance.paper_id,
                instance.context_id,
                instance.entity_type,
                primary_key(instance.local_label, instance.entity_type),
            )
            self.instances.setdefault(key, []).append(instance.instance_id)
            for source_mention_id in target.source_mention_ids:
                self.instance_sources.setdefault((instance.paper_id, source_mention_id), []).append(instance.instance_id)
        for index in (self.primary, self.instances):
            for key in index:
                index[key] = sorted(set(index[key]))
        for key in self.instance_sources:
            self.instance_sources[key] = sorted(set(self.instance_sources[key]))
        for key in self.aliases:
            self.aliases[key] = sorted(set(self.aliases[key]), key=lambda value: (value[0], value[1].value))
        for key in self.authorities:
            self.authorities[key] = sorted(set(self.authorities[key]))
        for key in self.concepts_by_type:
            self.concepts_by_type[key] = sorted(set(self.concepts_by_type[key]))
        for key in self.instances_by_scope:
            self.instances_by_scope[key] = sorted(set(self.instances_by_scope[key]))

        self.labels_by_target: dict[str, tuple[str, ...]] = {}
        token_postings: dict[tuple[str, EntityType, str | None, str | None, str], set[str]] = defaultdict(set)
        ngram_postings: dict[tuple[str, EntityType, str | None, str | None, str], set[str]] = defaultdict(set)
        ngram_size = policy.lexical_blocking.char_ngram_size
        for target_id, raw_labels in sorted(labels_by_target.items()):
            target = self.targets[target_id]
            labels = tuple(
                sorted(
                    set(raw_labels),
                    key=lambda label: (
                        label != target.label,
                        primary_key(label, target.entity_type),
                        label,
                    ),
                )
            )
            self.labels_by_target[target_id] = labels
            kind, paper_id, context_id = self._target_block_scope(target)
            for label in labels:
                for token in token_key(label, target.entity_type):
                    token_postings[(kind, target.entity_type, paper_id, context_id, token)].add(target_id)
                for ngram in character_ngrams(label, target.entity_type, ngram_size):
                    ngram_postings[(kind, target.entity_type, paper_id, context_id, ngram)].add(target_id)
        self.token_postings = {
            key: tuple(sorted(value)) for key, value in token_postings.items()
        }
        self.ngram_postings = {
            key: tuple(sorted(value)) for key, value in ngram_postings.items()
        }

    def eligible_concept_ids(self, mention: Mention, *, instance_link: bool = False) -> tuple[str, ...]:
        if instance_link:
            allowed = set(self.policy.instance_concept_map.get(mention.entity_type, ()))
        else:
            allowed = {mention.entity_type}
        return tuple(
            sorted(
                target_id
                for entity_type in allowed
                for target_id in self.concepts_by_type.get(entity_type, ())
            )
        )

    def eligible_instance_ids(self, mention: Mention) -> tuple[str, ...]:
        return tuple(
            self.instances_by_scope.get(
                (mention.paper_id, mention.context_id or "", mention.entity_type), ()
            )
        )

    def lexical_candidate_pool(self, mention: Mention) -> tuple[LexicalPoolEntry, ...]:
        """Return a bounded deterministic pool; never scan every eligible target."""

        if mention.identity_scope == IdentityScope.CANONICAL:
            prefix = ("CONCEPT", mention.entity_type, None, None)
        else:
            prefix = ("INSTANCE", mention.entity_type, mention.paper_id, mention.context_id)
        blockers: list[tuple[int, int, str, tuple[str, ...]]] = []
        for token in token_key(mention.atom_text, mention.entity_type):
            posting = self.token_postings.get(prefix + (token,), ())
            if 0 < len(posting) <= self.policy.lexical_blocking.max_postings_per_key:
                blockers.append((len(posting), 0, token, posting))
        for ngram in character_ngrams(
            mention.atom_text, mention.entity_type, self.policy.lexical_blocking.char_ngram_size
        ):
            posting = self.ngram_postings.get(prefix + (ngram,), ())
            if 0 < len(posting) <= self.policy.lexical_blocking.max_postings_per_key:
                blockers.append((len(posting), 1, ngram, posting))
        blockers.sort(key=lambda item: (item[0], item[1], item[2]))
        blockers = blockers[: self.policy.lexical_blocking.max_query_blockers]

        hits: dict[str, list[float]] = {}
        for posting_size, kind, _key, posting in blockers:
            for target_id in posting:
                values = hits.setdefault(target_id, [0.0, 0.0, 0.0])
                values[kind] += 1.0
                values[2] += 1.0 / posting_size
        ordered = sorted(
            hits.items(),
            key=lambda item: (-item[1][0], -item[1][2], -item[1][1], item[0]),
        )[: self.policy.lexical_blocking.max_candidate_pool]
        return tuple(
            LexicalPoolEntry(target_id, int(values[0]), int(values[1]), round(values[2], 12))
            for target_id, values in ordered
        )

    def scoring_labels(self, mention: Mention, target_id: str) -> tuple[str, ...]:
        query_tokens = set(token_key(mention.atom_text, mention.entity_type))
        query_ngrams = set(
            character_ngrams(
                mention.atom_text,
                mention.entity_type,
                self.policy.lexical_blocking.char_ngram_size,
            )
        )
        target = self.targets[target_id]

        def label_rank(label: str) -> tuple[int, int, bool, str, str]:
            label_tokens = set(token_key(label, target.entity_type))
            label_ngrams = set(
                character_ngrams(label, target.entity_type, self.policy.lexical_blocking.char_ngram_size)
            )
            return (
                -len(query_tokens & label_tokens),
                -len(query_ngrams & label_ngrams),
                label != target.label,
                primary_key(label, target.entity_type),
                label,
            )

        return tuple(
            sorted(self.labels_by_target.get(target_id, (target.label,)), key=label_rank)[
                : self.policy.lexical_blocking.max_labels_per_target
            ]
        )

    @staticmethod
    def _target_block_scope(target: Target) -> tuple[str, str | None, str | None]:
        if target.target_kind == "CONCEPT":
            return "CONCEPT", None, None
        return "INSTANCE", target.paper_id, target.context_id


@dataclass(frozen=True)
class MatchResult:
    decision: ResolutionDecision | None
    candidates: tuple[Candidate, ...]
    conflict_rows: tuple[tuple[str | None, str, str], ...]
    needs_bootstrap: bool
    instance_concept_id: str | None = None
    generated_candidate_count: int = 0
    merge_concept_ids: tuple[str, ...] = ()
    bootstrap_reason_codes: tuple[str, ...] = ()


def resolve_against_registry(
    mention: Mention,
    index: ResolverIndex,
    policy: ResolverPolicy,
    authority_snapshot: AuthoritySnapshot,
    authority_hints: Sequence[AuthorityHint] = (),
    embedding: EmbeddingRecall | None = None,
) -> MatchResult:
    """Resolve one mention against a frozen snapshot without mutating it."""

    constraints = [item for item in index.registry.constraints if item.active and item.mention_id == mention.mention_id]
    must = [item for item in constraints if item.constraint_type.value == "MUST_LINK"]
    cannot: set[str] = set()
    for constraint in constraints:
        if constraint.constraint_type.value != "CANNOT_LINK":
            continue
        target_id = index.registry.active_id(constraint.target_id)
        target = index.targets.get(target_id)
        if target is None:
            decision = _decision(
                mention,
                DecisionStatus.REVIEW_REQUIRED,
                DecisionMethod.HARD_CONFLICT,
                ("CANNOT_LINK_TARGET_MISSING",),
            )
            return MatchResult(
                decision,
                (),
                ((target_id, "CANNOT_LINK_TARGET_MISSING", "active human cannot-link target is absent"),),
                False,
            )
        compatibility = target_conflicts(mention, target, policy)
        if compatibility:
            decision = _decision(
                mention,
                DecisionStatus.REVIEW_REQUIRED,
                DecisionMethod.HARD_CONFLICT,
                ("CANNOT_LINK_TARGET_INCOMPATIBLE",) + compatibility,
            )
            return MatchResult(
                decision,
                (),
                tuple(
                    (target_id, code, "human cannot-link target violates mention scope/type")
                    for code in ("CANNOT_LINK_TARGET_INCOMPATIBLE",) + compatibility
                ),
                False,
            )
        cannot.add(target_id)
    if len({item.target_id for item in must}) > 1:
        decision = _decision(mention, DecisionStatus.REVIEW_REQUIRED, DecisionMethod.HARD_CONFLICT, ("CONFLICTING_MUST_LINKS",))
        return MatchResult(decision, (), ((None, "CONFLICTING_MUST_LINKS", "multiple active human must-links"),), False)
    if must:
        target_id = index.registry.active_id(must[0].target_id)
        if target_id in cannot:
            decision = _decision(
                mention,
                DecisionStatus.REVIEW_REQUIRED,
                DecisionMethod.HARD_CONFLICT,
                ("CONFLICTING_MUST_AND_CANNOT_LINK",),
            )
            return MatchResult(
                decision,
                (),
                ((target_id, "CONFLICTING_MUST_AND_CANNOT_LINK", "active constraints converge after redirect"),),
                False,
            )
        target = index.targets.get(target_id)
        if not target:
            decision = _decision(mention, DecisionStatus.REVIEW_REQUIRED, DecisionMethod.HARD_CONFLICT, ("MUST_LINK_TARGET_MISSING",))
            return MatchResult(decision, (), ((target_id, "MUST_LINK_TARGET_MISSING", "human target is absent"),), False)
        conflicts = target_conflicts(mention, target, policy)
        if conflicts:
            decision = _decision(mention, DecisionStatus.REVIEW_REQUIRED, DecisionMethod.HARD_CONFLICT, tuple(conflicts))
            return MatchResult(decision, (), tuple((target_id, code, "human must-link violates hard invariant") for code in conflicts), False)
        return MatchResult(_matched(mention, target, DecisionMethod.HUMAN_OVERRIDE, ("ACTIVE_HUMAN_MUST_LINK",)), (), (), False)

    candidates: list[Candidate] = []
    conflicts_out: list[tuple[str | None, str, str]] = []
    keys = build_keys(mention, policy)
    unsafe_reasons = []
    if not automatic_evidence_is_safe(mention):
        unsafe_reasons.append("SOURCE_NOT_EXACT_UNIQUE")
    if any(
        policy.qualifier_rule(item.kind) and policy.qualifier_rule(item.kind).semantic == "REVIEW_ONLY"
        for item in mention.qualifiers
    ):
        unsafe_reasons.append("UNMODELED_QUALIFIER_REQUIRES_REVIEW")
    if "UNMODELED_CONDITION" in mention.source_flags:
        unsafe_reasons.append("UNMODELED_CONDITION_REQUIRES_REVIEW")
    for flag in sorted(
        set(mention.source_flags)
        & {"SUSPICIOUS_COMPOUND", "WEAK_SOURCE", "OWNER_REVIEW_REQUIRED", "PARSER_WARNING", "OCR_WARNING", "LOW_TEXT_PAGE", "SOURCE_EVIDENCE_EXACT_AMBIGUOUS"}
    ):
        unsafe_reasons.append(f"SOURCE_FLAG_REQUIRES_REVIEW:{flag}")
    safe_source = not unsafe_reasons
    # Exact evidence is aggregated before any decision. Returning on the first
    # hit silently made the outcome depend on name/alias iteration order and
    # could hide that trusted aliases connected two existing registry concepts.
    exact_groups: list[tuple[DecisionMethod, list[str], str]] = []

    # 1. Exact authority ID. Conflicting hints never pick a target arbitrarily.
    hint_keys = sorted({(hint.authority.upper(), hint.external_id) for hint in authority_hints})
    incompatible_hint_pairs = []
    for index_a, left in enumerate(hint_keys):
        for right in hint_keys[index_a + 1 :]:
            if left == right or authority_snapshot.are_crosswalked(left[0], left[1], right[0], right[1]):
                continue
            incompatible_hint_pairs.append((left, right))
    if incompatible_hint_pairs:
        decision = _decision(mention, DecisionStatus.REVIEW_REQUIRED, DecisionMethod.HARD_CONFLICT, ("CONFLICTING_AUTHORITY_HINTS",))
        return MatchResult(decision, (), ((None, "CONFLICTING_AUTHORITY_HINTS", repr(incompatible_hint_pairs)),), False)
    authority_target_ids: set[str] = set()
    for hint in authority_hints:
        record = authority_snapshot.get(hint.authority, hint.external_id)
        if record and record.entity_type != mention.entity_type:
            code = "AUTHORITY_TYPE_CONFLICT"
            conflicts_out.append((None, code, f"{hint.authority}:{hint.external_id} is {record.entity_type}"))
            decision = _decision(mention, DecisionStatus.REVIEW_REQUIRED, DecisionMethod.HARD_CONFLICT, (code,))
            return MatchResult(decision, (), tuple(conflicts_out), False)
        authority_target_ids.update(index.authorities.get((hint.authority.upper(), hint.external_id), []))
    if authority_target_ids:
        exact_groups.append(
            (DecisionMethod.EXACT_AUTHORITY_ID, sorted(authority_target_ids), "|".join(f"{a}:{i}" for a, i in hint_keys))
        )

    if mention.identity_scope == IdentityScope.CANONICAL:
        # Query the registry with every name this mention carries, not just its
        # own. A mention of "onugbu" that supplies "Vernonia amygdalina" as an
        # alias should reach the existing Vernonia amygdalina concept; keying
        # only on the mention's own text meant alias agreement worked inside a
        # batch and never against the registry, so each run re-proposed the same
        # concept under a different name.
        lookup_keys: list[tuple[str, bool]] = [(keys.primary, False)]
        seen_lookup = {keys.primary}
        for alias in mention.aliases:
            alias_key = primary_key(alias, mention.entity_type)
            if alias_key and alias_key not in seen_lookup:
                lookup_keys.append((alias_key, True))
                seen_lookup.add(alias_key)
        for lookup_key, supplied_alias in lookup_keys:
            alias_entries = index.aliases.get((mention.entity_type, lookup_key), [])
            trusted_ids = sorted({target_id for target_id, trust in alias_entries
                                  if trust in policy.trusted_alias_levels})
            if trusted_ids:
                exact_groups.append((DecisionMethod.TRUSTED_ALIAS, trusted_ids, lookup_key))
            primary_ids = index.primary.get((mention.entity_type, lookup_key), [])
            if primary_ids:
                # A model-supplied alias hitting a registry preferred label is
                # alias evidence, not a weak bare-primary coincidence.
                method = DecisionMethod.TRUSTED_ALIAS if supplied_alias else DecisionMethod.NORMALIZED_PRIMARY
                exact_groups.append((method, primary_ids, lookup_key))
    else:
        instance_ids = index.instance_sources.get((mention.paper_id, mention.source_mention_id), [])
        if instance_ids:
            exact_groups.append((DecisionMethod.NORMALIZED_PRIMARY, instance_ids, mention.source_mention_id))

    if exact_groups:
        allowed_by_id: dict[str, set[DecisionMethod]] = defaultdict(set)
        evidence_by_id: dict[str, set[str]] = defaultdict(set)
        candidate_keys: set[tuple[str, str, str]] = set()
        for method, target_ids, evidence in exact_groups:
            for target_id in sorted(set(target_ids)):
                if target_id in cannot:
                    conflicts_out.append((target_id, "HUMAN_CANNOT_LINK", "active human cannot-link"))
                    continue
                target = index.targets[target_id]
                conflicts = target_conflicts(mention, target, policy)
                if conflicts:
                    conflicts_out.extend((target_id, code, "hard compatibility check") for code in conflicts)
                else:
                    allowed_by_id[target_id].add(method)
                    evidence_by_id[target_id].add(evidence)
                candidate_key = (target_id, method.value, evidence)
                if candidate_key not in candidate_keys:
                    candidates.append(
                        _candidate(
                            mention,
                            target,
                            method.value,
                            100.0,
                            {"exact_evidence": evidence},
                            conflicts,
                        )
                    )
                    candidate_keys.add(candidate_key)

        allowed = sorted(allowed_by_id)
        if allowed:
            merge_conflicts = _existing_concept_merge_conflicts(allowed, index)
            if len(allowed) > 1 and merge_conflicts:
                conflicts_out.extend((None, code, "existing exact-hit concepts are incompatible") for code in merge_conflicts)
                if _can_seed_separate(mention, safe_source):
                    return MatchResult(
                        None,
                        _rank_candidates(candidates, policy),
                        tuple(conflicts_out),
                        True,
                        generated_candidate_count=len(candidates),
                        bootstrap_reason_codes=(
                            "AUTO_MERGE_REVIEW:SEPARATE_IDENTITY_HARD_CONFLICT",
                            "AUTO_MERGE_REVIEW:POTENTIAL_DUPLICATE_CANDIDATE",
                        ),
                    )
                decision = _decision(
                    mention,
                    DecisionStatus.REVIEW_REQUIRED,
                    DecisionMethod.HARD_CONFLICT,
                    tuple(merge_conflicts),
                    candidates=candidates,
                )
                return MatchResult(
                    decision,
                    _rank_candidates(candidates, policy),
                    tuple(conflicts_out),
                    False,
                    generated_candidate_count=len(candidates),
                )

            target_id = min(allowed)
            target = index.targets[target_id]
            methods = {method for values in allowed_by_id.values() for method in values}
            method = (
                DecisionMethod.EXACT_AUTHORITY_ID
                if DecisionMethod.EXACT_AUTHORITY_ID in methods
                else DecisionMethod.TRUSTED_ALIAS
                if DecisionMethod.TRUSTED_ALIAS in methods
                else DecisionMethod.NORMALIZED_PRIMARY
            )
            exact_permitted = (
                mention.entity_type != EntityType.OTHER
                and (
                    method != DecisionMethod.NORMALIZED_PRIMARY
                    or policy.type_policy(mention.entity_type).normalized_primary_auto
                )
            )
            if exact_permitted:
                reasons = ["COLLISION_FREE", "HARD_CHECKS_PASSED"]
                if len(allowed) > 1:
                    reasons.extend(
                        [
                            "AUTO_MERGE_EXACT_DUPLICATE_CONCEPTS",
                            "AUTO_MERGE_REVIEW:EXISTING_CONCEPT_RECONCILIATION",
                        ]
                    )
                reasons.extend(_exact_audit_reasons(mention, target, policy, unsafe_reasons, method))
                return MatchResult(
                    _matched(mention, target, method, tuple(dict.fromkeys(reasons)), candidates),
                    _rank_candidates(candidates, policy) if len(allowed) > 1 else (),
                    tuple(conflicts_out),
                    False,
                    generated_candidate_count=len(candidates),
                    merge_concept_ids=tuple(allowed) if len(allowed) > 1 else (),
                )

            decision = _decision(
                mention,
                DecisionStatus.REVIEW_REQUIRED,
                DecisionMethod.AMBIGUOUS_CANDIDATES,
                ("TYPE_POLICY_REQUIRES_REVIEW",),
                candidates=candidates,
            )
            return MatchResult(
                decision,
                _rank_candidates(candidates, policy),
                tuple(conflicts_out),
                False,
                generated_candidate_count=len(candidates),
            )

    # No compatible exact hit survived. Approximate recall may still surface a
    # candidate, but explicit conflicts remain in the audit table.

    # 2. Approximate recall.  It may auto-decide only if this exact method/type
    # combination has calibrated thresholds in the active policy.
    if mention.identity_scope == IdentityScope.CANONICAL:
        eligible_ids = index.eligible_concept_ids(mention)
    else:
        eligible_ids = index.eligible_instance_ids(mention)
    lexical_policy = policy.type_policy(mention.entity_type).lexical
    lexical_candidates = _lexical_candidates(mention, index, lexical_policy.candidate_min_score, policy)
    candidates.extend(item for item in lexical_candidates if item.target_id not in cannot)
    if embedding is not None and eligible_ids:
        embedding_policy = policy.type_policy(mention.entity_type).embedding
        for target_id, score in embedding.search(mention, eligible_ids, policy.top_k_candidates):
            score = round(float(score), policy.score_round_digits)
            if score < embedding_policy.candidate_min_score or target_id in cannot:
                continue
            target = index.targets[target_id]
            conflicts = target_conflicts(mention, target, policy)
            if conflicts:
                conflicts_out.extend((target_id, code, "embedding candidate hard check") for code in conflicts)
            candidates.append(_candidate(mention, target, "EMBEDDING", score, {}, conflicts))
    ranked = _rank_candidates(candidates, policy)
    compatible = [item for item in ranked if not item.conflicts]
    if compatible and safe_source:
        top = compatible[0]
        runner = compatible[1] if len(compatible) > 1 else None
        method_policy = lexical_policy if top.method == "LEXICAL" else policy.type_policy(mention.entity_type).embedding
        if method_policy.automatic:
            margin = top.score - (runner.score if runner else 0.0)
            if top.score >= float(method_policy.accept_score) and margin >= float(method_policy.min_margin):
                method = DecisionMethod.LEXICAL_CALIBRATED if top.method == "LEXICAL" else DecisionMethod.EMBEDDING_CALIBRATED
                return MatchResult(
                    _matched(mention, index.targets[top.target_id], method, ("CALIBRATED_TYPE_POLICY",), ranked),
                    ranked,
                    tuple(conflicts_out),
                    False,
                    generated_candidate_count=len(candidates),
                )
    if compatible:
        reason = "APPROXIMATE_CANDIDATES_REQUIRE_REVIEW"
        if not safe_source:
            reason = unsafe_reasons[0]
        if _can_seed_separate(mention, safe_source):
            return MatchResult(
                None,
                ranked,
                tuple(conflicts_out),
                True,
                generated_candidate_count=len(candidates),
                bootstrap_reason_codes=(
                    "AUTO_MERGE_REVIEW:UNCALIBRATED_APPROXIMATE_CANDIDATE",
                    "AUTO_MERGE_REVIEW:POTENTIAL_DUPLICATE_CANDIDATE",
                ),
            )
        return MatchResult(
            _decision(
                mention,
                DecisionStatus.REVIEW_REQUIRED,
                DecisionMethod.AMBIGUOUS_CANDIDATES,
                (reason,),
                candidates=ranked,
            ),
            ranked,
            tuple(conflicts_out),
            False,
            generated_candidate_count=len(candidates),
        )
    if ranked:
        reason_codes = tuple(sorted({code for item in ranked for code in item.conflicts})) or (
            "ALL_CANDIDATES_BLOCKED",
        )
        if _can_seed_separate(mention, safe_source):
            return MatchResult(
                None,
                ranked,
                tuple(conflicts_out),
                True,
                generated_candidate_count=len(candidates),
                bootstrap_reason_codes=(
                    "AUTO_MERGE_REVIEW:SEPARATE_IDENTITY_HARD_CONFLICT",
                    "AUTO_MERGE_REVIEW:POTENTIAL_DUPLICATE_CANDIDATE",
                ),
            )
        return MatchResult(
            _decision(
                mention,
                DecisionStatus.REVIEW_REQUIRED,
                DecisionMethod.HARD_CONFLICT,
                reason_codes,
                candidates=ranked,
            ),
            ranked,
            tuple(conflicts_out),
            False,
            generated_candidate_count=len(candidates),
        )
    if not safe_source:
        return MatchResult(
            _decision(mention, DecisionStatus.REVIEW_REQUIRED, DecisionMethod.NO_SAFE_DECISION, tuple(unsafe_reasons), candidates=ranked),
            ranked,
            tuple(conflicts_out),
            False,
            generated_candidate_count=len(candidates),
        )
    return MatchResult(
        None, ranked, tuple(conflicts_out), True,
        generated_candidate_count=len(candidates),
    )


def _can_seed_separate(mention: Mention, safe_source: bool) -> bool:
    return (
        safe_source
        and mention.identity_scope == IdentityScope.CANONICAL
        and mention.entity_type != EntityType.OTHER
    )


def _existing_concept_merge_conflicts(
    concept_ids: Sequence[str], index: ResolverIndex
) -> tuple[str, ...]:
    """Return only genuine contradictions, never ordinary missing context."""

    targets = [index.targets[item] for item in sorted(set(concept_ids))]
    conflicts: set[str] = set()
    if any(item.target_kind != "CONCEPT" for item in targets):
        conflicts.add("AUTO_MERGE_IDENTITY_SCOPE_CONFLICT")
    if len({item.entity_type for item in targets}) > 1:
        conflicts.add("AUTO_MERGE_ENTITY_TYPE_CONFLICT")
    qualifier_maps = [dict(qualifier_signature(item.identity_qualifiers, index.policy)) for item in targets]
    for left_index, left in enumerate(qualifier_maps):
        for right in qualifier_maps[left_index + 1 :]:
            for kind in sorted(set(left) & set(right)):
                if left[kind] != right[kind]:
                    conflicts.add(f"AUTO_MERGE_IDENTITY_QUALIFIER_CONFLICT:{kind}")
    links_by_authority: dict[str, set[str]] = defaultdict(set)
    ids = {item.target_id for item in targets}
    for link in index.registry.authority_links:
        active = index.registry.active_id(link.concept_id)
        if active in ids:
            links_by_authority[link.authority.upper()].add(link.external_id)
    for authority, external_ids in links_by_authority.items():
        if len(external_ids) > 1:
            conflicts.add(f"AUTO_MERGE_AUTHORITY_CONFLICT:{authority}")
    return tuple(sorted(conflicts))


def _exact_audit_reasons(
    mention: Mention,
    target: Target,
    policy: ResolverPolicy,
    unsafe_reasons: Sequence[str],
    method: DecisionMethod,
) -> tuple[str, ...]:
    reasons = [f"AUTO_MERGE_REVIEW:{item}" for item in unsafe_reasons]
    left = dict(qualifier_signature(mention.qualifiers, policy))
    right = dict(qualifier_signature(target.identity_qualifiers, policy))
    for kind in sorted(set(left) ^ set(right)):
        reasons.append(f"AUTO_MERGE_REVIEW:MISSING_IDENTITY_QUALIFIER:{kind}")
    if method == DecisionMethod.NORMALIZED_PRIMARY:
        if mention.entity_type in {EntityType.PLACE, EntityType.ORGANISM}:
            reasons.append("AUTO_MERGE_REVIEW:HOMONYM_PRONE_EXACT_NAME")
        elif mention.entity_type == EntityType.ENVIRONMENTAL_FEATURE and _looks_named_feature(
            mention.atom_text
        ):
            reasons.append("AUTO_MERGE_REVIEW:HOMONYM_PRONE_EXACT_NAME")
    return tuple(dict.fromkeys(reasons))


_GENERIC_ENVIRONMENTAL_FEATURES = frozenset(
    {
        "air",
        "groundwater",
        "rainfall",
        "river water",
        "soil",
        "surface water",
        "wastewater",
        "water",
    }
)


def _looks_named_feature(text: str) -> bool:
    key = " ".join(re.findall(r"[\w]+", text.casefold()))
    return key not in _GENERIC_ENVIRONMENTAL_FEATURES


def target_conflicts(mention: Mention, target: Target, policy: ResolverPolicy) -> tuple[str, ...]:
    conflicts: list[str] = []
    if mention.identity_scope == IdentityScope.CANONICAL and target.target_kind != "CONCEPT":
        conflicts.append("IDENTITY_SCOPE_CONFLICT")
    if mention.identity_scope == IdentityScope.STUDY_INSTANCE and target.target_kind != "INSTANCE":
        conflicts.append("IDENTITY_SCOPE_CONFLICT")
    if mention.identity_scope == IdentityScope.STUDY_INSTANCE and target.target_kind == "INSTANCE":
        if mention.paper_id != target.paper_id:
            conflicts.append("CROSS_PAPER_INSTANCE")
        if mention.context_id != target.context_id:
            conflicts.append("INSTANCE_CONTEXT_CONFLICT")
    if mention.entity_type != target.entity_type:
        conflicts.append("ENTITY_TYPE_CONFLICT")
    left = dict(qualifier_signature(mention.qualifiers, policy))
    right = dict(qualifier_signature(target.identity_qualifiers, policy))
    for kind in sorted(set(left) & set(right)):
        if left[kind] != right[kind]:
            conflicts.append(f"IDENTITY_QUALIFIER_CONFLICT:{kind}")
    # Missing context is uncertainty, not contradictory identity. Exact matches
    # still connect and carry a post-merge audit flag; only two explicit unequal
    # values separate concepts.
    return tuple(conflicts)


def _lexical_candidates(
    mention: Mention,
    index: ResolverIndex,
    minimum: float,
    policy: ResolverPolicy,
) -> list[Candidate]:
    scored: list[Candidate] = []
    mention_key = build_keys(mention, policy)
    for pool_entry in index.lexical_candidate_pool(mention):
        target_id = pool_entry.target_id
        target = index.targets[target_id]
        labels = index.scoring_labels(mention, target_id)
        best = 0.0
        best_label = target.label
        for label in labels:
            target_key = primary_key(label, target.entity_type)
            value = float(fuzz.WRatio(mention_key.primary, target_key))
            if value > best:
                best, best_label = value, label
        score = round(best, policy.score_round_digits)
        if score < minimum:
            continue
        conflicts = target_conflicts(mention, target, policy)
        scored.append(
            _candidate(
                mention,
                target,
                "LEXICAL",
                score,
                {
                    "matched_label": best_label,
                    "block_token_hits": pool_entry.token_hits,
                    "block_ngram_hits": pool_entry.ngram_hits,
                    "block_rarity_score": pool_entry.rarity_score,
                },
                conflicts,
            )
        )
    return sorted(scored, key=lambda item: (-item.score, item.target_id))[: policy.top_k_candidates]


def _candidate(
    mention: Mention,
    target: Target,
    method: str,
    score: float,
    features: Mapping[str, object],
    conflicts: Sequence[str],
) -> Candidate:
    return Candidate(
        mention_id=mention.mention_id,
        target_kind=target.target_kind,
        target_id=target.target_id,
        target_label=target.label,
        target_entity_type=target.entity_type,
        method=method,
        score=score,
        features=tuple(sorted(features.items())),
        conflicts=tuple(sorted(set(conflicts))),
    )


def _rank_candidates(candidates: Iterable[Candidate], policy: ResolverPolicy) -> tuple[Candidate, ...]:
    # Retain separate method traces but rank deterministically across them.
    ordered = sorted(candidates, key=lambda item: (-item.score, item.target_id, item.method))
    return tuple(
        Candidate(
            mention_id=item.mention_id,
            target_kind=item.target_kind,
            target_id=item.target_id,
            target_label=item.target_label,
            target_entity_type=item.target_entity_type,
            method=item.method,
            score=round(item.score, policy.score_round_digits),
            features=item.features,
            conflicts=item.conflicts,
            rank=index + 1,
        )
        for index, item in enumerate(ordered[: policy.top_k_candidates])
    )


def _matched(
    mention: Mention,
    target: Target,
    method: DecisionMethod,
    reasons: Sequence[str],
    candidates: Sequence[Candidate] = (),
) -> ResolutionDecision:
    best_by_target: dict[str, float] = {}
    for item in candidates:
        if not item.conflicts:
            best_by_target[item.target_id] = max(best_by_target.get(item.target_id, 0.0), item.score)
    scores = sorted(best_by_target.values(), reverse=True)
    top = scores[0] if scores else 100.0
    runner = scores[1] if len(scores) > 1 else None
    return ResolutionDecision(
        mention_id=mention.mention_id,
        status=DecisionStatus.MATCHED,
        method=method,
        concept_id=target.target_id if target.target_kind == "CONCEPT" else target.primary_concept_id,
        instance_id=target.target_id if target.target_kind == "INSTANCE" else None,
        reason_codes=tuple(reasons),
        top_score=top,
        runner_up_score=runner,
        margin=None if runner is None else top - runner,
        candidate_count=len({item.target_id for item in candidates}),
    )


def _decision(
    mention: Mention,
    status: DecisionStatus,
    method: DecisionMethod,
    reasons: Sequence[str],
    *,
    candidates: Sequence[Candidate] = (),
) -> ResolutionDecision:
    scores = sorted((item.score for item in candidates if not item.conflicts), reverse=True)
    top = scores[0] if scores else None
    runner = scores[1] if len(scores) > 1 else None
    return ResolutionDecision(
        mention_id=mention.mention_id,
        status=status,
        method=method,
        reason_codes=tuple(reasons),
        top_score=top,
        runner_up_score=runner,
        margin=None if top is None or runner is None else top - runner,
        candidate_count=len({item.target_id for item in candidates}),
    )
