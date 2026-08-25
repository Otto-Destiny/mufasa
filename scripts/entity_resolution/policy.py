"""Versioned policy loading and validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import (
    AliasTrust,
    ContractError,
    EntityType,
    IdentityScope,
    OwnerKind,
)


EXPECTED_QUALIFIER_KINDS = frozenset({
    "ADMINISTRATIVE_LEVEL", "AGE_GROUP", "CHEMICAL_FORM", "COUNTRY",
    "DEPTH_CLASS", "FEATURE_CLASS",
    "DEVELOPMENTAL_STAGE", "GENETIC_VARIANT_STRAIN", "MATERIAL_FORM",
    "PROTECTION_STATUS", "QUALITY_GRADE", "SEX_GENDER", "SIZE_CLASS",
    "SOURCE_ORIGIN", "URBAN_RURAL_CLASS", "VERSION_VARIANT", "UNMODELED_QUALIFIER",
})
EXPECTED_CONDITION_NAMES = frozenset({
    "BASELINE_STATUS", "DISEASE_STAGE", "DOSE_EXPOSURE", "DURATION",
    "ENVIRONMENTAL_STATE", "EXPERIMENTAL_SETTING", "MEASUREMENT_SETTING", "PH",
    "PRESSURE", "SALINITY", "SAMPLING_SETTING", "SEASON", "STATISTICAL_THRESHOLD",
    "TEMPERATURE", "TIME_POINT", "TREATMENT_ARM", "UNMODELED_CONDITION",
})


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{label} keys differ; missing={sorted(expected-actual)}, unknown={sorted(actual-expected)}"
        )


def _strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ContractError(f"{label} must be a YAML boolean, not {value!r}")
    return value


@dataclass(frozen=True)
class MethodPolicy:
    candidate_min_score: float
    automatic: bool
    accept_score: float | None
    min_margin: float | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], label: str) -> "MethodPolicy":
        candidate_min = float(value["candidate_min_score"])
        _exact_keys(
            value,
            {"candidate_min_score", "automatic", "accept_score", "min_margin"},
            label,
        )
        automatic = _strict_bool(value["automatic"], f"{label}.automatic")
        accept = value.get("accept_score")
        margin = value.get("min_margin")
        accept = None if accept is None else float(accept)
        margin = None if margin is None else float(margin)
        if not 0 <= candidate_min <= 100:
            raise ContractError(f"{label}.candidate_min_score must be within [0, 100]")
        if automatic and (accept is None or margin is None):
            raise ContractError(f"{label} automatic matching requires accept_score and min_margin")
        if accept is not None and not candidate_min <= accept <= 100:
            raise ContractError(f"{label}.accept_score must be within [candidate_min_score, 100]")
        if margin is not None and not 0 <= margin <= 100:
            raise ContractError(f"{label}.min_margin must be within [0, 100]")
        return cls(candidate_min, automatic, accept, margin)


@dataclass(frozen=True)
class TypePolicy:
    self_seed: bool
    normalized_primary_auto: bool
    lexical: MethodPolicy
    embedding: MethodPolicy


@dataclass(frozen=True)
class LexicalBlockingPolicy:
    char_ngram_size: int
    max_candidate_pool: int
    max_postings_per_key: int
    max_query_blockers: int
    max_labels_per_target: int


@dataclass(frozen=True)
class QualifierRule:
    kind: str
    semantic: str
    allowed_types: frozenset[EntityType]
    allowed_scopes: frozenset[IdentityScope]


@dataclass(frozen=True)
class ResolverPolicy:
    version: str
    normalization_version: str
    schema_version: str
    qualifier_vocab_version: str
    condition_vocab_version: str
    extraction_prompt_version: str
    code_version: str
    content_hash: str
    allowed_roles: frozenset[str]
    allowed_types: frozenset[EntityType]
    allowed_scopes: frozenset[IdentityScope]
    allowed_owner_kinds: frozenset[OwnerKind]
    condition_names: frozenset[str]
    qualifier_rules: tuple[QualifierRule, ...]
    qualifier_aliases: tuple[tuple[str, str], ...]
    trusted_alias_levels: frozenset[AliasTrust]
    type_policies: tuple[tuple[EntityType, TypePolicy], ...]
    instance_concept_compatibility: tuple[tuple[EntityType, tuple[EntityType, ...]], ...]
    lexical_blocking: LexicalBlockingPolicy
    top_k_candidates: int
    score_round_digits: int
    dependency_specifiers: tuple[tuple[str, str], ...]
    python_specifier: str

    @property
    def qualifier_rule_map(self) -> dict[str, QualifierRule]:
        return {rule.kind: rule for rule in self.qualifier_rules}

    @property
    def qualifier_alias_map(self) -> dict[str, str]:
        return dict(self.qualifier_aliases)

    @property
    def type_policy_map(self) -> dict[EntityType, TypePolicy]:
        return dict(self.type_policies)

    @property
    def instance_concept_map(self) -> dict[EntityType, tuple[EntityType, ...]]:
        return dict(self.instance_concept_compatibility)

    def type_policy(self, entity_type: EntityType) -> TypePolicy:
        try:
            return self.type_policy_map[entity_type]
        except KeyError as exc:
            raise ContractError(f"policy has no entry for entity type {entity_type}") from exc

    def qualifier_rule(self, raw_kind: str) -> QualifierRule | None:
        kind = raw_kind.strip().upper()
        kind = self.qualifier_alias_map.get(kind, kind)
        return self.qualifier_rule_map.get(kind)


def _merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_policy(path: str | Path | None = None) -> ResolverPolicy:
    path = Path(path) if path else Path(__file__).parent / "policies" / "mufasa-v1.yaml"
    raw_bytes = path.read_bytes()
    try:
        data = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise ContractError(f"invalid policy YAML {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ContractError("resolver policy must be a mapping")
    _exact_keys(
        data,
        {
            "version", "normalization_version", "schema_version", "qualifier_vocab_version",
            "condition_vocab_version", "extraction_prompt_version", "code_version",
            "capabilities", "conditions", "vocabularies", "qualifiers", "matching",
        },
        "policy",
    )
    for required in (
        "version",
        "normalization_version",
        "schema_version",
        "qualifier_vocab_version",
        "condition_vocab_version",
        "extraction_prompt_version",
        "code_version",
        "vocabularies",
        "qualifiers",
        "matching",
        "capabilities",
    ):
        if required not in data:
            raise ContractError(f"resolver policy missing {required}")

    vocab = data["vocabularies"]
    _exact_keys(vocab, {"roles", "entity_types", "identity_scopes", "owner_kinds"}, "vocabularies")
    try:
        allowed_types = frozenset(EntityType(value) for value in vocab["entity_types"])
        allowed_scopes = frozenset(IdentityScope(value) for value in vocab["identity_scopes"])
        allowed_owner_kinds = frozenset(OwnerKind(value) for value in vocab["owner_kinds"])
        allowed_roles = frozenset(str(value).upper() for value in vocab["roles"])
    except (KeyError, ValueError) as exc:
        raise ContractError(f"invalid controlled vocabulary: {exc}") from exc
    if allowed_types != frozenset(EntityType):
        missing = frozenset(EntityType) - allowed_types
        raise ContractError(f"policy must define every EntityType; missing {sorted(map(str, missing))}")
    if allowed_scopes != frozenset(IdentityScope):
        raise ContractError("policy identity_scopes must exactly match the code enum")
    if allowed_owner_kinds != frozenset(OwnerKind):
        raise ContractError("policy owner_kinds must exactly match the code enum")

    _exact_keys(data["conditions"], {"names"}, "conditions")
    condition_names = frozenset(str(item).upper() for item in data["conditions"]["names"])
    if condition_names != EXPECTED_CONDITION_NAMES:
        raise ContractError("condition vocabulary does not exactly match the locked extraction contract")

    _exact_keys(data["qualifiers"], {"aliases", "kinds"}, "qualifiers")
    qualifier_kind_names = frozenset(str(item).upper() for item in data["qualifiers"]["kinds"])
    if qualifier_kind_names != EXPECTED_QUALIFIER_KINDS:
        raise ContractError("qualifier vocabulary does not exactly match the locked extraction contract")

    qualifier_rules: list[QualifierRule] = []
    for kind, value in data["qualifiers"]["kinds"].items():
        semantic = str(value["semantic"]).upper()
        if semantic not in {"DESCRIPTIVE", "DISAMBIGUATING", "IDENTITY_BEARING", "INSTANCE_DEFINING", "REVIEW_ONLY"}:
            raise ContractError(f"qualifier {kind} has invalid semantic {semantic}")
        types = frozenset(EntityType(item) for item in value["allowed_types"])
        scopes = frozenset(IdentityScope(item) for item in value["allowed_scopes"])
        qualifier_rules.append(QualifierRule(str(kind).upper(), semantic, types, scopes))
    qualifier_aliases = tuple(
        sorted((str(k).upper(), str(v).upper()) for k, v in data["qualifiers"].get("aliases", {}).items())
    )
    unknown_alias_targets = {target for _source, target in qualifier_aliases} - qualifier_kind_names
    if unknown_alias_targets:
        raise ContractError(f"qualifier aliases target unknown kinds {sorted(unknown_alias_targets)}")

    matching = data["matching"]
    _exact_keys(
        matching,
        {
            "top_k_candidates", "score_round_digits", "trusted_alias_levels",
            "lexical_blocking", "default_type_policy", "type_overrides",
            "instance_concept_compatibility",
        },
        "matching",
    )
    defaults = matching["default_type_policy"]
    overrides = matching.get("type_overrides", {})
    type_policy_keys = {"self_seed", "normalized_primary_auto", "lexical", "embedding"}
    _exact_keys(defaults, type_policy_keys, "matching.default_type_policy")
    expected_type_names = {item.value for item in EntityType}
    if set(overrides) != expected_type_names:
        raise ContractError(
            "matching.type_overrides must explicitly contain every entity type; "
            f"missing={sorted(expected_type_names-set(overrides))}, unknown={sorted(set(overrides)-expected_type_names)}"
        )
    for name, override in overrides.items():
        unknown = set(override) - type_policy_keys
        if unknown:
            raise ContractError(f"matching.type_overrides.{name} has unknown keys {sorted(unknown)}")
        for method in ("lexical", "embedding"):
            if method in override:
                unknown_method = set(override[method]) - {
                    "candidate_min_score", "automatic", "accept_score", "min_margin"
                }
                if unknown_method:
                    raise ContractError(
                        f"matching.type_overrides.{name}.{method} has unknown keys {sorted(unknown_method)}"
                    )
    type_policies: list[tuple[EntityType, TypePolicy]] = []
    for entity_type in sorted(EntityType, key=lambda item: item.value):
        merged = _merge_dict(defaults, overrides.get(entity_type.value, {}))
        type_policies.append(
            (
                entity_type,
                TypePolicy(
                    self_seed=_strict_bool(merged["self_seed"], f"{entity_type}.self_seed"),
                    normalized_primary_auto=_strict_bool(
                        merged["normalized_primary_auto"], f"{entity_type}.normalized_primary_auto"
                    ),
                    lexical=MethodPolicy.from_mapping(merged["lexical"], f"{entity_type}.lexical"),
                    embedding=MethodPolicy.from_mapping(merged["embedding"], f"{entity_type}.embedding"),
                ),
            )
        )
    compatibility_raw = matching["instance_concept_compatibility"]
    if set(compatibility_raw) != expected_type_names:
        raise ContractError("instance_concept_compatibility must explicitly define every entity type")
    compatibility = tuple(
        sorted(
            (
                EntityType(source),
                tuple(sorted((EntityType(item) for item in targets), key=lambda item: item.value)),
            )
            for source, targets in compatibility_raw.items()
        )
    )
    trusted_alias_levels = frozenset(AliasTrust(item) for item in matching["trusted_alias_levels"])
    top_k = int(matching["top_k_candidates"])
    if top_k < 1:
        raise ContractError("top_k_candidates must be positive")
    score_digits = int(matching.get("score_round_digits", 6))
    if not 0 <= score_digits <= 12:
        raise ContractError("score_round_digits must be within [0, 12]")
    blocking_raw = matching.get("lexical_blocking")
    if not isinstance(blocking_raw, Mapping):
        raise ContractError("matching.lexical_blocking must be an explicit versioned mapping")
    lexical_blocking = LexicalBlockingPolicy(
        char_ngram_size=int(blocking_raw["char_ngram_size"]),
        max_candidate_pool=int(blocking_raw["max_candidate_pool"]),
        max_postings_per_key=int(blocking_raw["max_postings_per_key"]),
        max_query_blockers=int(blocking_raw["max_query_blockers"]),
        max_labels_per_target=int(blocking_raw["max_labels_per_target"]),
    )
    if not 2 <= lexical_blocking.char_ngram_size <= 6:
        raise ContractError("lexical_blocking.char_ngram_size must be within [2, 6]")
    for field in (
        "max_candidate_pool", "max_postings_per_key", "max_query_blockers",
        "max_labels_per_target",
    ):
        if getattr(lexical_blocking, field) < 1:
            raise ContractError(f"lexical_blocking.{field} must be positive")
    if lexical_blocking.max_candidate_pool < top_k:
        raise ContractError("lexical_blocking.max_candidate_pool must be >= top_k_candidates")
    return ResolverPolicy(
        version=str(data["version"]),
        normalization_version=str(data["normalization_version"]),
        schema_version=str(data["schema_version"]),
        qualifier_vocab_version=str(data["qualifier_vocab_version"]),
        condition_vocab_version=str(data["condition_vocab_version"]),
        extraction_prompt_version=str(data["extraction_prompt_version"]),
        code_version=str(data["code_version"]),
        content_hash=hashlib.sha256(raw_bytes).hexdigest(),
        allowed_roles=allowed_roles,
        allowed_types=allowed_types,
        allowed_scopes=allowed_scopes,
        allowed_owner_kinds=allowed_owner_kinds,
        condition_names=condition_names,
        qualifier_rules=tuple(sorted(qualifier_rules, key=lambda rule: rule.kind)),
        qualifier_aliases=qualifier_aliases,
        trusted_alias_levels=trusted_alias_levels,
        type_policies=tuple(type_policies),
        instance_concept_compatibility=compatibility,
        lexical_blocking=lexical_blocking,
        top_k_candidates=top_k,
        score_round_digits=score_digits,
        dependency_specifiers=tuple(
            sorted((str(name), str(specifier)) for name, specifier in data["capabilities"]["python_packages"].items())
        ),
        python_specifier=str(data["capabilities"]["python"]),
    )


def policy_manifest(policy: ResolverPolicy) -> dict[str, Any]:
    return {
        "version": policy.version,
        "normalization_version": policy.normalization_version,
        "schema_version": policy.schema_version,
        "qualifier_vocab_version": policy.qualifier_vocab_version,
        "condition_vocab_version": policy.condition_vocab_version,
        "extraction_prompt_version": policy.extraction_prompt_version,
        "code_version": policy.code_version,
        "sha256": policy.content_hash,
    }
