"""Immutable contracts for MUFASA entity resolution.

The resolver never edits extraction records.  It consumes :class:`Mention`
objects and emits separate, immutable decision and registry records.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping


class ContractError(ValueError):
    """Raised when an input violates the resolver contract."""


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class EntityType(StrEnum):
    PLACE = "PLACE"
    ORGANISM = "ORGANISM"
    POPULATION = "POPULATION"
    SAMPLE_SPECIMEN = "SAMPLE_SPECIMEN"
    MATERIAL = "MATERIAL"
    CHEMICAL = "CHEMICAL"
    ENVIRONMENTAL_FEATURE = "ENVIRONMENTAL_FEATURE"
    HEALTH_CONDITION = "HEALTH_CONDITION"
    PROPERTY_METRIC = "PROPERTY_METRIC"
    METHOD = "METHOD"
    MODEL_ALGORITHM = "MODEL_ALGORITHM"
    DATASET = "DATASET"
    INTERVENTION_ACTION = "INTERVENTION_ACTION"
    INFRASTRUCTURE_DEVICE = "INFRASTRUCTURE_DEVICE"
    ORGANIZATION = "ORGANIZATION"
    EVENT_PROCESS = "EVENT_PROCESS"
    HAZARD_RISK = "HAZARD_RISK"
    APPLICATION_USE = "APPLICATION_USE"
    TIME_PERIOD = "TIME_PERIOD"
    STANDARD_POLICY = "STANDARD_POLICY"
    OTHER = "OTHER"


class IdentityScope(StrEnum):
    CANONICAL = "CANONICAL"
    STUDY_INSTANCE = "STUDY_INSTANCE"


class OwnerKind(StrEnum):
    CONTEXT = "CONTEXT"
    OBSERVATION = "OBSERVATION"


class AlignmentStatus(StrEnum):
    EXACT_UNIQUE = "EXACT_UNIQUE"
    EXACT_AMBIGUOUS = "EXACT_AMBIGUOUS"


class ProvenanceScope(StrEnum):
    OWNER_EVIDENCE = "OWNER_EVIDENCE"
    STUDY_CONTEXT = "STUDY_CONTEXT"


class AssertionStatus(StrEnum):
    REPORTED = "REPORTED"
    DERIVED = "DERIVED"


class DecisionStatus(StrEnum):
    MATCHED = "MATCHED"
    NEW_CONCEPT_PROPOSED = "NEW_CONCEPT_PROPOSED"
    NEW_INSTANCE_PROPOSED = "NEW_INSTANCE_PROPOSED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    UNRESOLVED = "UNRESOLVED"
    INVALID_INPUT = "INVALID_INPUT"


class DecisionMethod(StrEnum):
    HUMAN_OVERRIDE = "HUMAN_OVERRIDE"
    EXACT_AUTHORITY_ID = "EXACT_AUTHORITY_ID"
    TRUSTED_ALIAS = "TRUSTED_ALIAS"
    NORMALIZED_PRIMARY = "NORMALIZED_PRIMARY"
    EXACT_BOOTSTRAP_GROUP = "EXACT_BOOTSTRAP_GROUP"
    CLEAN_SINGLETON_BOOTSTRAP = "CLEAN_SINGLETON_BOOTSTRAP"
    EXACT_INSTANCE_GROUP = "EXACT_INSTANCE_GROUP"
    LEXICAL_CALIBRATED = "LEXICAL_CALIBRATED"
    EMBEDDING_CALIBRATED = "EMBEDDING_CALIBRATED"
    AMBIGUOUS_CANDIDATES = "AMBIGUOUS_CANDIDATES"
    HARD_CONFLICT = "HARD_CONFLICT"
    INVALID_CONTRACT = "INVALID_CONTRACT"
    NO_SAFE_DECISION = "NO_SAFE_DECISION"


class AliasTrust(StrEnum):
    AUTHORITY = "AUTHORITY"
    HUMAN_CURATED = "HUMAN_CURATED"
    REVIEWED_CORPUS = "REVIEWED_CORPUS"
    # An alternative name supplied by the extraction model. Trusted by explicit
    # project decision: naming is the one thing the model may draw from its own
    # knowledge, and its accuracy on names is what connects a paper writing
    # "onugbu" to one writing "Vernonia amygdalina". Whether the paper itself
    # stated the equivalence is kept per alias, so the decision stays auditable.
    MODEL_SUPPLIED = "MODEL_SUPPLIED"
    GENERATED = "GENERATED"
    UNVERIFIED = "UNVERIFIED"


class LifecycleStatus(StrEnum):
    ACTIVE = "ACTIVE"
    MERGED = "MERGED"
    SPLIT = "SPLIT"
    RETIRED = "RETIRED"


class ProposalKind(StrEnum):
    CONCEPT = "CONCEPT"
    INSTANCE = "INSTANCE"


class EventType(StrEnum):
    CREATE_CONCEPT = "CREATE_CONCEPT"
    CREATE_INSTANCE = "CREATE_INSTANCE"
    ADD_ALIAS = "ADD_ALIAS"
    ADD_AUTHORITY_LINK = "ADD_AUTHORITY_LINK"
    ASSIGN_MENTION = "ASSIGN_MENTION"
    MERGE = "MERGE"
    SPLIT = "SPLIT"
    REDIRECT = "REDIRECT"
    REASSIGN = "REASSIGN"
    REVIEW = "REVIEW"
    SUPERSEDE = "SUPERSEDE"


class ConstraintType(StrEnum):
    MUST_LINK = "MUST_LINK"
    CANNOT_LINK = "CANNOT_LINK"


TRUSTED_ALIAS_LEVELS = frozenset(
    {AliasTrust.AUTHORITY, AliasTrust.HUMAN_CURATED, AliasTrust.REVIEWED_CORPUS}
)

ALIAS_KINDS = frozenset(
    {
        "SCIENTIFIC",
        "TAXONOMIC_SYNONYM",
        "COMMON_ENGLISH",
        "VERNACULAR",
        "ACRONYM",
        "TRADE_NAME",
        "FORMULA",
        "SPELLING_VARIANT",
    }
)
MAX_ALIASES_PER_MENTION = 10
_ISO_639_1 = re.compile(r"^[a-z]{2}$")


def _required_text(value: Any, name: str) -> str:
    if value is None:
        raise ContractError(f"{name} is required")
    text = str(value).strip()
    if not text:
        raise ContractError(f"{name} is empty")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    # A null Parquet cell arrives as NaN, and str(nan) is the string "nan".
    # Every optional text field here reads from Parquet, so absence has to be
    # absence rather than a four-character label that looks like data.
    if isinstance(value, float) and value != value:
        return None
    text = str(value).strip()
    return text or None


def _enum(enum_type: type[StrEnum], value: Any, name: str) -> StrEnum:
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ContractError(f"{name}={value!r} is invalid; expected one of {allowed}") from exc


def parse_alias_records(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    """Parse the extraction alias contract without legacy coercions.

    Aliases can cause automatic identity merges, so a bare string or a partial
    object is not accepted.  Every record retains the exact four fields emitted
    by extraction and uses the same closed kind vocabulary.
    """

    raw = parse_json_list(value, name)
    if len(raw) > MAX_ALIASES_PER_MENTION:
        raise ContractError(f"{name} exceeds {MAX_ALIASES_PER_MENTION} entries")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    required = {"text", "kind", "language", "stated_in_paper"}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or set(item) != required:
            raise ContractError(
                f"{name}[{index}] must contain exactly text, kind, language, stated_in_paper"
            )
        text_value = item["text"]
        if not isinstance(text_value, str) or not text_value.strip():
            raise ContractError(f"{name}[{index}].text must be a non-empty string")
        text = text_value.strip()
        folded = text.casefold()
        if folded in seen:
            raise ContractError(f"{name}[{index}].text duplicates another alias")
        seen.add(folded)
        kind = item["kind"]
        if not isinstance(kind, str) or kind not in ALIAS_KINDS:
            raise ContractError(
                f"{name}[{index}].kind={kind!r} is invalid; expected one of {sorted(ALIAS_KINDS)}"
            )
        language = item["language"]
        if not isinstance(language, str):
            raise ContractError(f"{name}[{index}].language must be a string")
        if language and not _ISO_639_1.fullmatch(language):
            raise ContractError(
                f"{name}[{index}].language must be an empty string or lowercase ISO 639-1 code"
            )
        stated = item["stated_in_paper"]
        if not isinstance(stated, bool):
            raise ContractError(f"{name}[{index}].stated_in_paper must be true or false")
        records.append(
            {
                "text": text,
                "kind": kind,
                "language": language,
                "stated_in_paper": stated,
            }
        )
    return tuple(records)


def parse_alias_texts(value: Any, name: str) -> tuple[str, ...]:
    return tuple(record["text"] for record in parse_alias_records(value, name))


def parse_json_list(value: Any, name: str) -> list[Any]:
    """Parse a physical Parquet JSON-list field without silently coercing it."""

    if value is None:
        return []
    if isinstance(value, float) and value != value:  # NaN
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{name} is invalid JSON: {exc}") from exc
    if not isinstance(value, list):
        raise ContractError(f"{name} must be a JSON list")
    return value


def freeze_pairs(value: Mapping[str, Any] | Iterable[tuple[str, Any]] | None) -> tuple[tuple[str, Any], ...]:
    if not value:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    return tuple(sorted((str(k), v) for k, v in items))


@dataclass(frozen=True, order=True)
class Qualifier:
    kind: str
    value_text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _required_text(self.kind, "qualifier.kind").upper())
        object.__setattr__(self, "value_text", _required_text(self.value_text, "qualifier.value_text"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Qualifier":
        if not isinstance(value, Mapping) or set(value) != {"kind", "value_text"}:
            raise ContractError("each qualifier must contain exactly kind and value_text")
        return cls(kind=value["kind"], value_text=value["value_text"])


@dataclass(frozen=True)
class Mention:
    mention_id: str
    source_mention_id: str
    source_evidence_id: str
    paper_id: str
    owner_kind: OwnerKind
    owner_id: str
    role: str
    surface_text: str
    atom_text: str
    entity_type: EntityType
    identity_scope: IdentityScope
    qualifiers: tuple[Qualifier, ...] = ()
    context_id: str | None = None
    source_page: int | None = None
    source_char_start: int | None = None
    source_char_end: int | None = None
    source_occurrence_count: int = 1
    source_occurrences_json: str = "[]"
    source_alignment_status: AlignmentStatus = AlignmentStatus.EXACT_UNIQUE
    provenance_scope: ProvenanceScope = ProvenanceScope.OWNER_EVIDENCE
    qualifier_vocab_version: str = ""
    extraction_schema_version: str = ""
    assertion_status: AssertionStatus = AssertionStatus.REPORTED
    domain: str | None = None
    language: str | None = None
    country_code: str | None = None
    # Alternative names for this same entity, supplied by extraction. They are
    # the bridge between papers that use different words for one thing, which is
    # the dominant failure mode in a vernacular-heavy African corpus.
    #
    # `aliases` holds just the names, because matching compares names. The full
    # records - text, kind, language and whether the paper itself stated the
    # equivalence - are preserved verbatim in `aliases_json` and travel into the
    # run output, so any merge these names drove can be audited afterwards
    # against exactly what extraction supplied.
    aliases: tuple[str, ...] = ()
    aliases_json: str = "[]"
    # Extraction's label for one physical thing inside one paper. It is the
    # only reliable way to know that two wordings mean the same sample.
    instance_local_id: str = ""
    source_flags: tuple[str, ...] = ()
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "mention_id",
            "source_mention_id",
            "source_evidence_id",
            "paper_id",
            "owner_id",
            "role",
            "surface_text",
            "atom_text",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(self, "role", self.role.upper())
        if (self.source_char_start is None) != (self.source_char_end is None):
            raise ContractError("source_char_start and source_char_end must both be set or both be null")
        if self.source_char_start is not None:
            if self.source_char_start < 0 or self.source_char_end <= self.source_char_start:
                raise ContractError("source offsets must satisfy 0 <= start < end")
        if self.source_occurrence_count < 1:
            raise ContractError("source_occurrence_count must be positive")
        try:
            occurrences = json.loads(self.source_occurrences_json)
        except json.JSONDecodeError as exc:
            raise ContractError(f"source_occurrences_json is invalid: {exc}") from exc
        if not isinstance(occurrences, list):
            raise ContractError("source_occurrences_json must be a JSON list")
        if tuple(sorted(set(self.qualifiers))) != self.qualifiers:
            raise ContractError("qualifiers must be sorted and duplicate-free")
        alias_records = parse_alias_records(self.aliases_json, "aliases_json")
        record_names = {item["text"].casefold() for item in alias_records}
        alias_names = {str(item).strip().casefold() for item in self.aliases}
        if any(not str(item).strip() for item in self.aliases):
            raise ContractError("aliases cannot contain blank names")
        if len(alias_names) != len(self.aliases):
            raise ContractError("aliases must be duplicate-free")
        if alias_names != record_names:
            raise ContractError("aliases and aliases_json must describe the same names")
        own_names = {self.surface_text.casefold(), self.atom_text.casefold()}
        if own_names & record_names:
            raise ContractError("an alias cannot repeat the mention surface_text or atom_text")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "Mention":
        qualifiers = tuple(
            sorted(
                {
                    Qualifier.from_mapping(item)
                    for item in parse_json_list(row.get("qualifiers_json", row.get("qualifiers")), "qualifiers_json")
                }
            )
        )
        alias_records = parse_alias_records(
            row.get("aliases_json", row.get("aliases")), "aliases_json"
        )
        aliases_json = json.dumps(
            list(alias_records), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return cls(
            mention_id=_required_text(row.get("mention_id"), "mention_id"),
            source_mention_id=_required_text(row.get("source_mention_id"), "source_mention_id"),
            source_evidence_id=_required_text(
                row.get("source_evidence_id", row.get("evidence_id")), "source_evidence_id"
            ),
            paper_id=_required_text(row.get("paper_id"), "paper_id"),
            owner_kind=_enum(OwnerKind, row.get("owner_kind"), "owner_kind"),
            owner_id=_required_text(row.get("owner_id"), "owner_id"),
            role=_required_text(row.get("role"), "role"),
            surface_text=_required_text(row.get("surface_text"), "surface_text"),
            atom_text=_required_text(row.get("atom_text"), "atom_text"),
            entity_type=_enum(EntityType, row.get("entity_type"), "entity_type"),
            identity_scope=_enum(IdentityScope, row.get("identity_scope"), "identity_scope"),
            qualifiers=qualifiers,
            context_id=_optional_text(row.get("context_id")),
            source_page=_optional_int(row.get("source_page", row.get("page"))),
            source_char_start=_optional_int(row.get("source_char_start", row.get("char_start"))),
            source_char_end=_optional_int(row.get("source_char_end", row.get("char_end"))),
            source_occurrence_count=int(row.get("source_occurrence_count", 1)),
            source_occurrences_json=str(row.get("source_occurrences_json", "[]")),
            source_alignment_status=_enum(
                AlignmentStatus,
                row.get("source_alignment_status", row.get("alignment_status", "EXACT_UNIQUE")),
                "source_alignment_status",
            ),
            provenance_scope=_enum(
                ProvenanceScope,
                row.get("provenance_scope", "OWNER_EVIDENCE"),
                "provenance_scope",
            ),
            qualifier_vocab_version=_required_text(
                row.get("qualifier_vocab_version"), "qualifier_vocab_version"
            ),
            extraction_schema_version=_required_text(
                row.get("extraction_schema_version"), "extraction_schema_version"
            ),
            assertion_status=_enum(
                AssertionStatus,
                row.get("assertion_status", "REPORTED"),
                "assertion_status",
            ),
            domain=_optional_text(row.get("domain", row.get("mufasa_domain"))),
            language=_optional_text(row.get("language")),
            country_code=_optional_text(row.get("country_code")),
            source_flags=tuple(sorted(set(parse_json_list(row.get("source_flags_json", []), "source_flags_json")))),
            aliases=tuple(item["text"] for item in alias_records),
            aliases_json=aliases_json,
            instance_local_id=_optional_text(row.get("instance_local_id")) or "",
        )

    @property
    def alignment_status(self) -> AlignmentStatus:
        """Compatibility property; persisted schema uses source_alignment_status."""

        return self.source_alignment_status


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, bool):
        raise ContractError("boolean is not a valid offset")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"invalid integer value {value!r}") from exc


@dataclass(frozen=True)
class AuthorityHint:
    mention_id: str
    authority: str
    external_id: str
    snapshot_version: str
    source: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        for name in ("mention_id", "authority", "external_id", "snapshot_version", "source"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(self, "authority", self.authority.upper())
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ContractError("authority hint confidence must be within [0, 1]")


@dataclass(frozen=True)
class CanonicalEntity:
    concept_id: str
    preferred_label: str
    entity_type: EntityType
    identity_qualifiers_json: str
    lifecycle_status: LifecycleStatus
    seed_mention_id: str
    created_run_id: str
    updated_run_id: str
    registry_version: str
    provenance_json: str = "{}"


@dataclass(frozen=True)
class EntityInstance:
    instance_id: str
    paper_id: str
    context_id: str
    local_label: str
    entity_type: EntityType
    concept_id: str | None
    identity_qualifiers_json: str
    source_mention_ids_json: str
    created_run_id: str
    updated_run_id: str
    registry_version: str


@dataclass(frozen=True)
class CanonicalAlias:
    alias_id: str
    concept_id: str
    alias_text: str
    normalized_key: str
    entity_type: EntityType
    language: str | None
    region: str | None
    source: str
    trust_level: AliasTrust
    created_run_id: str
    registry_version: str
    # Extraction's alias kind (SCIENTIFIC, VERNACULAR, ACRONYM, ...) and whether
    # the paper itself stated the equivalence. Both are blank for aliases that
    # did not come from extraction.
    alias_kind: str = ""
    stated_in_paper: bool | None = None
    provenance_json: str = "{}"

    def __post_init__(self) -> None:
        for name in ("alias_id", "concept_id", "alias_text", "normalized_key", "source"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if self.alias_kind and self.alias_kind not in ALIAS_KINDS:
            raise ContractError(f"canonical alias kind {self.alias_kind!r} is invalid")
        if self.language and not _ISO_639_1.fullmatch(self.language):
            raise ContractError("canonical alias language must be a lowercase ISO 639-1 code")
        if self.stated_in_paper is not None and not isinstance(self.stated_in_paper, bool):
            raise ContractError("canonical alias stated_in_paper must be boolean or null")
        try:
            provenance = json.loads(self.provenance_json or "{}")
        except json.JSONDecodeError as exc:
            raise ContractError("canonical alias provenance_json is invalid") from exc
        if not isinstance(provenance, Mapping):
            raise ContractError("canonical alias provenance_json must be an object")


@dataclass(frozen=True)
class AuthorityLink:
    authority_link_id: str
    concept_id: str
    authority: str
    external_id: str
    authority_snapshot_version: str
    source: str
    created_run_id: str
    registry_version: str


@dataclass(frozen=True)
class EntityRelation:
    relation_id: str
    source_concept_id: str
    relation_type: str
    target_concept_id: str
    provenance: str
    reviewed: bool
    created_run_id: str
    registry_version: str


@dataclass(frozen=True)
class EntityRedirect:
    retired_id: str
    active_id: str
    event_id: str
    registry_version: str


@dataclass(frozen=True)
class ResolutionConstraint:
    constraint_id: str
    constraint_type: ConstraintType
    mention_id: str
    target_id: str
    reviewer: str
    reason: str
    active: bool
    created_run_id: str
    registry_version: str


@dataclass(frozen=True)
class Candidate:
    mention_id: str
    target_kind: str
    target_id: str
    target_label: str
    target_entity_type: EntityType
    method: str
    score: float
    features: tuple[tuple[str, Any], ...] = ()
    conflicts: tuple[str, ...] = ()
    rank: int = 0

    @property
    def feature_map(self) -> dict[str, Any]:
        return dict(self.features)


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    proposal_kind: ProposalKind
    member_mention_ids: tuple[str, ...]
    preferred_label: str
    entity_type: EntityType
    paper_id: str | None
    context_id: str | None
    primary_concept_id: str | None
    primary_concept_proposal_id: str | None
    authority_keys: tuple[tuple[str, str, str, str], ...]
    auto_approved: bool
    method: DecisionMethod
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class ResolutionDecision:
    mention_id: str
    status: DecisionStatus
    method: DecisionMethod
    concept_id: str | None = None
    instance_id: str | None = None
    proposal_id: str | None = None
    reason_codes: tuple[str, ...] = ()
    top_score: float | None = None
    runner_up_score: float | None = None
    margin: float | None = None
    calibrated_probability: float | None = None
    candidate_count: int = 0
    candidate_set_hash: str | None = None
    memo_key: str | None = None
    memo_reused: bool = False
    propagated_review_event_id: str | None = None

    def __post_init__(self) -> None:
        if self.status == DecisionStatus.MATCHED:
            if (self.concept_id is None) == (self.instance_id is None):
                # A study instance may have both its instance ID and an INSTANCE_OF
                # concept. Canonical mentions have only concept_id.
                if self.instance_id is None or self.concept_id is None:
                    raise ContractError("MATCHED must have a concept_id or instance_id")
        elif self.status in {
            DecisionStatus.NEW_CONCEPT_PROPOSED,
            DecisionStatus.NEW_INSTANCE_PROPOSED,
        }:
            if not self.proposal_id or self.concept_id or self.instance_id:
                raise ContractError("proposal decisions carry only proposal_id before commit")
        elif self.concept_id or self.instance_id or self.proposal_id:
            raise ContractError(f"{self.status} cannot carry committed or proposed IDs")


@dataclass(frozen=True)
class ResolutionEvent:
    event_id: str
    event_type: EventType
    subject_id: str
    object_id: str | None
    run_id: str
    registry_version: str
    reason_codes: tuple[str, ...]
    reviewer: str | None = None
    occurred_at: str | None = None


@dataclass(frozen=True)
class ResolutionConflict:
    conflict_id: str
    mention_id: str
    target_id: str | None
    conflict_code: str
    severity: str
    detail: str
    run_id: str


@dataclass(frozen=True)
class ConceptMergePlan:
    """Deterministic, non-blocking reconciliation of duplicate registry concepts."""

    concept_ids: tuple[str, ...]
    survivor_concept_id: str
    trigger_mention_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.concept_ids))) != self.concept_ids or len(self.concept_ids) < 2:
            raise ContractError("concept merge plan requires at least two sorted, unique concept IDs")
        if self.survivor_concept_id not in self.concept_ids:
            raise ContractError("concept merge survivor must be a member of the merge plan")
        if tuple(sorted(set(self.trigger_mention_ids))) != self.trigger_mention_ids:
            raise ContractError("merge trigger mention IDs must be sorted and unique")
        if not self.trigger_mention_ids or not self.reason_codes:
            raise ContractError("concept merge plans require triggers and reason codes")


@dataclass(frozen=True)
class InvalidMentionRecord:
    mention_id: str
    paper_id: str | None
    raw_json: str
    error_codes: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class ResolutionRun:
    run_id: str
    input_fingerprint: str
    base_registry_version: str
    result_registry_version: str
    policy_version: str
    normalization_version: str
    authority_manifest_hash: str
    resolver_code_version: str
    mentions: tuple[Mention, ...]
    decisions: tuple[ResolutionDecision, ...]
    proposals: tuple[Proposal, ...]
    candidates: tuple[Candidate, ...]
    generated_candidate_count: int = 0
    invalid_inputs: tuple[InvalidMentionRecord, ...] = ()
    effective_controls: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.generated_candidate_count < len(self.candidates):
            raise ContractError("generated_candidate_count cannot be smaller than retained candidates")
        ids = [mention.mention_id for mention in self.mentions] + [item.mention_id for item in self.invalid_inputs]
        decision_ids = [decision.mention_id for decision in self.decisions]
        if len(ids) != len(set(ids)):
            raise ContractError("run mentions contain duplicate mention_id values")
        if sorted(ids) != sorted(decision_ids):
            raise ContractError("every mention must have exactly one resolution decision")


def enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [json_value(item) for item in value]
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    return value
