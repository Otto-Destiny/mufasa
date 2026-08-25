"""Fail-closed adapter for the MUFASA extraction Parquets."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..contracts import (
    AlignmentStatus,
    AssertionStatus,
    AuthorityHint,
    ContractError,
    InvalidMentionRecord,
    Mention,
    OwnerKind,
    ProvenanceScope,
)
from ..policy import ResolverPolicy
from ..validation import assert_referential_integrity


SOURCE_FINGERPRINT_FIELDS = (
    "paper_id",
    "openalex_id",
    "pdf_sha256",
    "parser_name",
    "parser_version",
    "parser_config_hash",
    "markdown_sha256",
)


MENTION_COLUMNS = frozenset(
    {
        "mention_id",
        "paper_id",
        "owner_kind",
        "owner_id",
        "source_mention_id",
        "source_evidence_id",
        "source_page",
        "source_char_start",
        "source_char_end",
        "source_occurrence_count",
        "source_occurrences_json",
        "source_alignment_status",
        "provenance_scope",
        "role",
        "surface_text",
        "atom_text",
        "entity_type",
        "identity_scope",
        # Both are required, not optional enrichment: alias overlap is how a
        # paper writing "onugbu" reaches one writing "Vernonia amygdalina", and
        # instance_local_id is the only statement that two wordings in one paper
        # denote one physical sample. Silently defaulting them would leave the
        # resolver matching on bare strings while appearing to work.
        "instance_local_id",
        "aliases_json",
        "qualifiers_json",
        "qualifier_vocab_version",
        "extraction_schema_version",
        "canonical_id",
        "resolution_status",
    }
)

# Where inside a page a quote was taken from. Not the document artifact: that
# is structured_json or markdown_fallback and lives in extraction_status.
EVIDENCE_SOURCE_KINDS = frozenset({"TEXT", "TABLE", "FIGURE"})

EVIDENCE_COLUMNS = frozenset(
    {
        "evidence_id",
        "paper_id",
        "owner_kind",
        "owner_id",
        "page_start",
        "page_end",
        "evidence_text",
        "char_start",
        "char_end",
        "occurrence_count",
        "occurrences_json",
        "alignment_status",
        "source_kind",
        "source_sha256",
    }
)

PUBLISHED_TABLES = frozenset(
    {
        "study_contexts.parquet",
        "observations.parquet",
        "entity_mentions.parquet",
        "evidence_spans.parquet",
        "paper_profiles.parquet",
        "training_pairs.parquet",
        "extraction_status.parquet",
    }
)
PUBLICATION_FORMAT = "mufasa-parquet-generation-v1"


@dataclass(frozen=True)
class MufasaInputs:
    mentions: tuple[Mention, ...]
    invalid_mentions: tuple[InvalidMentionRecord, ...]
    authority_hints: tuple[AuthorityHint, ...]
    input_fingerprint: str
    table_hashes: tuple[tuple[str, str], ...]
    extraction_schema_version: str
    qualifier_vocab_version: str
    condition_vocab_versions: tuple[str, ...]
    model_id: str
    prompt_version: str
    settings_hash: str
    source_fingerprint: str
    structured_content_hashes: tuple[tuple[str, str], ...]
    structured_source_hashes: tuple[tuple[str, str], ...]
    source_paths: tuple[str, ...]
    extraction_generation_id: str


def _read(path: Path, required: frozenset[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise ContractError(f"required {label} is missing: {path}")
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise ContractError(f"cannot read {label} {path}: {type(exc).__name__}: {exc}") from exc
    missing = required - set(frame.columns)
    if missing:
        raise ContractError(f"{label} is missing columns {sorted(missing)}")
    return frame


def _resolve_published_generation(
    root: Path, policy: ResolverPolicy
) -> tuple[dict[str, Path], dict[str, Any], Path]:
    """Resolve the atomic pointer and verify every immutable Parquet before use."""

    marker_path = root / "current-generation.json"
    marker = _read_json_descriptor(marker_path, "current extraction generation")
    required_root = {
        "format",
        "generation_id",
        "created_at",
        "directory",
        "settings_hash",
        "source_fingerprint",
        "schema_version",
        "prompt_version",
        "qualifier_vocab_version",
        "condition_vocab_version",
        "tables",
    }
    if set(marker) != required_root:
        raise ContractError("current-generation.json has an invalid root schema")
    if marker["format"] != PUBLICATION_FORMAT:
        raise ContractError(f"unsupported extraction publication format {marker['format']!r}")
    generation_id = marker["generation_id"]
    if not isinstance(generation_id, str) or not re.fullmatch(r"[0-9a-f]{24}", generation_id):
        raise ContractError("publication generation_id must be 24 lowercase hex characters")
    if not isinstance(marker["created_at"], str) or not marker["created_at"].strip():
        raise ContractError("publication created_at must be a non-empty string")
    expected_directory = f"published-generations/{generation_id}"
    if marker["directory"] != expected_directory:
        raise ContractError("publication directory does not match generation_id")
    generation_dir = (root / marker["directory"]).resolve()
    publication_root = (root / "published-generations").resolve()
    if generation_dir.parent != publication_root or generation_dir.name != generation_id:
        raise ContractError("publication directory escapes the extraction publication root")

    expected_versions = {
        "schema_version": policy.schema_version,
        "prompt_version": policy.extraction_prompt_version,
        "qualifier_vocab_version": policy.qualifier_vocab_version,
        "condition_vocab_version": policy.condition_vocab_version,
    }
    for key, expected in expected_versions.items():
        if marker[key] != expected:
            raise ContractError(f"publication {key}={marker[key]!r}, expected {expected!r}")
    for key in ("settings_hash", "source_fingerprint"):
        value = marker[key]
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ContractError(f"publication {key} must be a lowercase SHA-256 digest")

    tables = marker["tables"]
    if not isinstance(tables, Mapping) or set(tables) != PUBLISHED_TABLES:
        raise ContractError("publication marker does not contain the exact required Parquet set")
    identity = {
        "format": marker["format"],
        "settings_hash": marker["settings_hash"],
        "source_fingerprint": marker["source_fingerprint"],
        "schema_version": marker["schema_version"],
        "prompt_version": marker["prompt_version"],
        "qualifier_vocab_version": marker["qualifier_vocab_version"],
        "condition_vocab_version": marker["condition_vocab_version"],
        "tables": tables,
    }
    recomputed_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    if recomputed_id != generation_id:
        raise ContractError("publication generation_id does not match its canonical identity")

    generation_manifest_path = generation_dir / "generation.json"
    generation_manifest = _read_json_descriptor(
        generation_manifest_path, "immutable extraction generation manifest"
    )
    if generation_manifest != marker:
        raise ContractError("current-generation.json disagrees with immutable generation.json")

    import pyarrow.parquet as pq

    paths: dict[str, Path] = {}
    for filename, metadata in sorted(tables.items()):
        if not isinstance(metadata, Mapping) or set(metadata) != {"sha256", "rows", "columns"}:
            raise ContractError(f"publication metadata is malformed for {filename}")
        digest = metadata["sha256"]
        rows = metadata["rows"]
        columns = metadata["columns"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ContractError(f"publication SHA-256 is malformed for {filename}")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise ContractError(f"publication row count is malformed for {filename}")
        if not isinstance(columns, list) or not all(isinstance(item, str) and item for item in columns):
            raise ContractError(f"publication column list is malformed for {filename}")
        table_path = generation_dir / filename
        if not table_path.is_file() or _file_hash(table_path) != digest:
            raise ContractError(f"published table is missing or corrupt: {filename}")
        try:
            parquet = pq.ParquetFile(table_path)
            actual_rows = parquet.metadata.num_rows
            actual_columns = parquet.schema_arrow.names
        except Exception as exc:
            raise ContractError(f"cannot inspect published table {filename}: {exc}") from exc
        if actual_rows != rows or actual_columns != columns:
            raise ContractError(f"published table metadata drifted: {filename}")
        paths[filename] = table_path
    return paths, marker, generation_manifest_path


def _unique(frame: pd.DataFrame, column: str, label: str) -> None:
    if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
        raise ContractError(f"{label}.{column} contains null/blank IDs")
    duplicated = frame.loc[frame[column].duplicated(keep=False), column].astype(str).unique().tolist()
    if duplicated:
        raise ContractError(f"{label}.{column} is not unique; first duplicates={duplicated[:10]}")


def _conditions(value: Any, label: str, allowed_names: frozenset[str]) -> tuple[tuple[str, str], ...]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} is invalid JSON: {exc}") from exc
    if parsed is None or (isinstance(parsed, float) and parsed != parsed):
        parsed = []
    if not isinstance(parsed, list):
        raise ContractError(f"{label} must be a JSON list")
    result = []
    for index, item in enumerate(parsed):
        if not isinstance(item, Mapping) or set(item) != {"name", "value_text"}:
            raise ContractError(f"{label}[{index}] must contain exactly name and value_text")
        name = str(item["name"]).strip().upper()
        value_text = str(item["value_text"]).strip()
        if name not in allowed_names or not value_text:
            raise ContractError(f"{label}[{index}] has invalid name/value: {item!r}")
        result.append((name, value_text))
    return tuple(result)


def _occurrences(raw: Any, label: str) -> tuple[tuple[int, int, int], ...]:
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(parsed, list):
        raise ContractError(f"{label} must be a JSON list")
    values = []
    for index, item in enumerate(parsed):
        if not isinstance(item, Mapping):
            raise ContractError(f"{label}[{index}] must be an object")
        try:
            page = int(item["page"])
            start = int(item["char_start"])
            end = int(item["char_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"{label}[{index}] requires integer page/char_start/char_end") from exc
        if page < 1 or start < 0 or end <= start:
            raise ContractError(f"{label}[{index}] has invalid page-local span")
        values.append((page, start, end))
    if len(values) != len(set(values)):
        raise ContractError(f"{label} contains duplicate occurrences")
    return tuple(values)


def load_mufasa_inputs(
    extraction_dir: str | Path,
    documents_path: str | Path,
    policy: ResolverPolicy,
    *,
    authority_hints_path: str | Path | None = None,
) -> MufasaInputs:
    """Load and validate the exact extraction seam without coercive fallbacks."""

    root = Path(extraction_dir)
    published_paths, publication_marker, generation_manifest_path = _resolve_published_generation(
        root, policy
    )
    paths = {
        "mentions": published_paths["entity_mentions.parquet"],
        "observations": published_paths["observations.parquet"],
        "contexts": published_paths["study_contexts.parquet"],
        "evidence": published_paths["evidence_spans.parquet"],
        "status": published_paths["extraction_status.parquet"],
        "profiles": published_paths["paper_profiles.parquet"],
        "documents": Path(documents_path),
        "extraction_manifest": root / "manifest.json",
        "run_summary": root / "run_summary.json",
        "current_generation": root / "current-generation.json",
        "generation_manifest": generation_manifest_path,
    }
    extraction_manifest = _read_json_descriptor(paths["extraction_manifest"], "extraction manifest")
    run_summary = _read_json_descriptor(paths["run_summary"], "extraction run summary")
    for field in (
        "settings_hash",
        "source_fingerprint",
        "schema_version",
        "prompt_version",
        "qualifier_vocab_version",
        "condition_vocab_version",
    ):
        if publication_marker[field] != extraction_manifest.get(field) or publication_marker[field] != run_summary.get(field):
            raise ContractError(f"publication marker disagrees with run descriptors on {field}")
    mentions_frame = _read(paths["mentions"], MENTION_COLUMNS, "entity_mentions")
    observations = _read(
        paths["observations"],
        frozenset({
            "observation_id", "paper_id", "context_id", "conditions_json", "assertion_status",
            "evidence_id",
        }),
        "observations",
    )
    contexts = _read(
        paths["contexts"],
        frozenset({"context_id", "paper_id", "conditions_json"}),
        "study_contexts",
    )
    evidence = _read(paths["evidence"], EVIDENCE_COLUMNS, "evidence_spans")
    status = _read(
        paths["status"],
        frozenset({"paper_id", "status", "error", "source_kind",
                   "resolver_eligible", "resolver_excluded_reason",
                   "context_coverage_complete"}),
        "extraction_status",
    )
    profiles = _read(
        paths["profiles"],
        frozenset({"paper_id", "coverage_complete"}),
        "paper_profiles",
    )
    documents = _read(
        paths["documents"],
        frozenset({
            "paper_id", "openalex_id", "pipeline_status", "rights_status", "retraction_status",
            "structured_path", "markdown_path", "pdf_sha256", "parser_name", "parser_version",
            "markdown_sha256",
        }),
        "documents manifest",
    )
    for frame, column, label in (
        (mentions_frame, "mention_id", "entity_mentions"),
        (observations, "observation_id", "observations"),
        (contexts, "context_id", "study_contexts"),
        (evidence, "evidence_id", "evidence_spans"),
        (status, "paper_id", "extraction_status"),
        (profiles, "paper_id", "paper_profiles"),
        (documents, "paper_id", "documents"),
    ):
        _unique(frame, column, label)

    _validate_cross_table_integrity(observations, contexts, evidence, mentions_frame)
    for frame, column, label in (
        (profiles, "coverage_complete", "paper_profiles"),
        (status, "context_coverage_complete", "extraction_status"),
        (status, "resolver_eligible", "extraction_status"),
    ):
        if not pd.api.types.is_bool_dtype(frame[column].dtype) or frame[column].isna().any():
            raise ContractError(f"{label}.{column} must be non-null boolean")
    profile_coverage = dict(
        zip(profiles["paper_id"].astype(str), profiles["coverage_complete"].astype(bool))
    )
    if set(profile_coverage) != set(status["paper_id"].astype(str)):
        raise ContractError("paper_profiles must contain exactly every extraction_status paper")
    for row in status.to_dict("records"):
        paper_id = str(row["paper_id"])
        status_coverage = bool(row["context_coverage_complete"])
        if paper_id not in profile_coverage or profile_coverage[paper_id] != status_coverage:
            raise ContractError(f"paper profile/status coverage disagrees for {paper_id}")
        if not status_coverage and not bool(row["resolver_eligible"]):
            raise ContractError(
                f"partial context profile cannot exclude resolver input for {paper_id}"
            )

    if not observations["assertion_status"].astype(str).eq(AssertionStatus.REPORTED.value).all():
        invalid = observations.loc[
            ~observations["assertion_status"].astype(str).eq(AssertionStatus.REPORTED.value),
            ["observation_id", "assertion_status"],
        ].head(10)
        raise ContractError(f"source extraction observations must be REPORTED: {invalid.to_dict('records')}")

    unmodeled_owners: set[tuple[str, str]] = set()
    condition_versions: set[str] = set()
    for label, frame, id_column, owner_kind in (
        ("study_contexts", contexts, "context_id", "CONTEXT"),
        ("observations", observations, "observation_id", "OBSERVATION"),
    ):
        if "condition_vocab_version" not in frame.columns:
            raise ContractError(f"{label} is missing condition_vocab_version")
        versions = {str(item).strip() for item in frame["condition_vocab_version"] if str(item).strip()}
        if len(versions) != 1:
            raise ContractError(f"{label}.condition_vocab_version must contain exactly one nonblank version")
        condition_versions.update(versions)
        for row in frame.to_dict("records"):
            values = _conditions(
                row["conditions_json"], f"{label}.{row[id_column]}.conditions_json", policy.condition_names
            )
            if any(name == "UNMODELED_CONDITION" for name, _ in values):
                unmodeled_owners.add((owner_kind, str(row[id_column])))
    if len(condition_versions) != 1:
        raise ContractError(f"context and observation condition vocab versions disagree: {sorted(condition_versions)}")
    if next(iter(condition_versions)) != policy.condition_vocab_version:
        raise ContractError(
            f"condition vocabulary {next(iter(condition_versions))} != policy {policy.condition_vocab_version}"
        )

    # Extraction decides which papers feed the concept graph and records it per
    # paper. A gated paper - one the full text says is not real science, or not
    # about Africa - keeps every row it produced; those rows are simply out of
    # scope here rather than an integrity failure. Anything ineligible for any
    # other reason still fails closed. Gating happens after the checks above so
    # a gated paper's data is still validated, and before eligibility below so
    # its rows never reach resolution.
    gated = {
        str(row["paper_id"]): _source_clean_text(row.get("resolver_excluded_reason"))
        for row in status.to_dict("records")
        if not bool(row.get("resolver_eligible"))
        and str(row.get("status")) == "complete"
    }
    if gated:
        def _ungated(frame: pd.DataFrame) -> pd.DataFrame:
            return frame[~frame["paper_id"].astype(str).isin(gated)].copy()

        observations, contexts = _ungated(observations), _ungated(contexts)
        evidence, mentions_frame = _ungated(evidence), _ungated(mentions_frame)

    eligible = documents[
        documents["pipeline_status"].astype(str).eq("ok")
        & documents["rights_status"].astype(str).eq("permitted")
        & ~documents["retraction_status"].astype(str).eq("retracted")
    ].copy()
    eligible = (
        eligible.drop_duplicates("paper_id", keep="last")
        .sort_values("paper_id", kind="stable")
        .reset_index(drop=True)
    )
    completed = status[status["status"].astype(str).eq("complete")]
    eligible_ids = frozenset(
        (set(eligible["paper_id"].astype(str)) & set(completed["paper_id"].astype(str)))
        - set(gated)
    )
    if not eligible_ids:
        raise ContractError("no manifest-eligible papers have complete extraction status")
    if gated:
        print(f"resolver gate: {len(gated)} complete paper(s) excluded from resolution "
              f"and left published; reasons={sorted(set(gated.values()))}")
    # Resolve stale Kaggle paths against the authoritative corpus root and
    # verify immutable parser artifacts before trusting any Parquet offset.
    corpus_root = Path(documents_path).resolve().parent.parent
    structured_pages: dict[str, dict[int, str]] = {}
    structured_hashes: dict[str, str] = {}
    structured_source_hashes: dict[str, str] = {}
    verified_source_paths: list[Path] = []
    enriched_rows: list[dict[str, Any]] = []
    evidence_papers = set(evidence["paper_id"].astype(str))
    # evidence_spans.source_kind says where inside the page a quote came from -
    # TEXT, TABLE or FIGURE - and a paper legitimately mixes all three. Which
    # document artifact the offsets were computed against is a per-paper fact,
    # and extraction records it once, in extraction_status.source_kind.
    document_source_kinds = {
        str(row["paper_id"]): _source_clean_text(row.get("source_kind"))
        for row in status.to_dict("records")
    }
    unexpected = sorted(set(evidence["source_kind"].astype(str)) - EVIDENCE_SOURCE_KINDS)
    if unexpected:
        raise ContractError(f"evidence_spans.source_kind has unknown values: {unexpected}")
    for document_row in eligible.to_dict("records"):
        paper_id = str(document_row["paper_id"])
        structured_candidate = _artifact_path(
            document_row, "structured_path", corpus_root, "structured", ".json"
        )
        if structured_candidate.is_file():
            source_kind = "structured_json"
            source_path = structured_candidate.resolve()
            pages, content_hash, source_hash, parser_config_hash = _load_verified_structured(
                source_path, document_row
            )
        else:
            source_kind = "markdown_fallback"
            source_path = _resolve_markdown_path(document_row, corpus_root)
            pages, content_hash, source_hash = _load_verified_markdown(source_path, document_row)
            parser_config_hash = ""
        recorded_source_kind = document_source_kinds.get(paper_id, "")
        if recorded_source_kind and recorded_source_kind != source_kind:
            raise ContractError(
                f"paper {paper_id} was extracted from {recorded_source_kind} but the "
                f"currently verified source is {source_kind}"
            )
        enriched = dict(document_row)
        existing_config = _source_clean_text(document_row.get("parser_config_hash"))
        if existing_config and existing_config != parser_config_hash:
            raise ContractError(
                f"documents parser_config_hash conflicts with verified sidecar for {paper_id}"
            )
        enriched["parser_config_hash"] = parser_config_hash
        enriched_rows.append(enriched)
        if paper_id in evidence_papers:
            structured_pages[paper_id] = pages
        structured_hashes[paper_id] = content_hash
        structured_source_hashes[paper_id] = source_hash
        verified_source_paths.append(source_path)
    enriched_eligible = pd.DataFrame(enriched_rows, columns=list(eligible.columns) + (
        [] if "parser_config_hash" in eligible.columns else ["parser_config_hash"]
    ))
    descriptor = _validate_run_descriptor(
        extraction_manifest, run_summary, enriched_eligible, status, policy, corpus_root
    )

    for paper_id in evidence_papers:
        if paper_id not in eligible_ids:
            raise ContractError(f"evidence belongs to missing/ineligible paper {paper_id}")

    # Evidence spans themselves are independently checked against raw parser
    # pages; a self-consistent but corrupted extraction Parquet cannot pass.
    for evidence_row in evidence.to_dict("records"):
        _verify_evidence_against_structured(
            evidence_row,
            structured_pages[str(evidence_row["paper_id"])],
            structured_source_hashes[str(evidence_row["paper_id"])],
            f"evidence {evidence_row['evidence_id']}",
        )

    observation_context = dict(zip(observations["observation_id"].astype(str), observations["context_id"].astype(str)))
    context_ids = frozenset(contexts["context_id"].astype(str))
    owner_ids = {
        "CONTEXT": context_ids,
        "OBSERVATION": frozenset(observations["observation_id"].astype(str)),
    }
    evidence_by_id = {str(row["evidence_id"]): row for row in evidence.to_dict("records")}
    valid_mentions: list[Mention] = []
    invalid_mentions: list[InvalidMentionRecord] = []
    qualifier_versions: set[str] = set()
    schema_versions: set[str] = set()
    for row in mentions_frame.to_dict("records"):
        raw_json = json.dumps(_clean_record(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        mention_id = str(row.get("mention_id") or "").strip() or f"INVALID-{len(invalid_mentions):08d}"
        try:
            if pd.notna(row.get("canonical_id")) and str(row.get("canonical_id")).strip():
                raise ContractError("extraction canonical_id placeholder must be null")
            if str(row.get("resolution_status")) != "UNRESOLVED":
                raise ContractError("extraction resolution_status placeholder must be UNRESOLVED")
            occurrences = _occurrences(row["source_occurrences_json"], f"mention {mention_id}.source_occurrences_json")
            if int(row["source_occurrence_count"]) != len(occurrences):
                raise ContractError("source_occurrence_count disagrees with source_occurrences_json")
            status_value = AlignmentStatus(str(row["source_alignment_status"]))
            start = _nullable_int(row.get("source_char_start"))
            end = _nullable_int(row.get("source_char_end"))
            source_page = int(row["source_page"])
            if status_value == AlignmentStatus.EXACT_UNIQUE:
                if len(occurrences) != 1 or occurrences[0] != (source_page, start, end):
                    raise ContractError("EXACT_UNIQUE source fields must equal the sole recorded occurrence")
            else:
                if len(occurrences) < 2 or start is not None or end is not None:
                    raise ContractError("EXACT_AMBIGUOUS requires multiple occurrences and null chosen offsets")
                if any(page != source_page for page, _start, _end in occurrences):
                    raise ContractError("source occurrences must remain within source_page")
            evidence_row = evidence_by_id.get(str(row["source_evidence_id"]))
            if evidence_row is None:
                raise ContractError("source_evidence_id does not exist")
            if str(evidence_row["paper_id"]) != str(row["paper_id"]):
                raise ContractError("mention and evidence belong to different papers")
            _verify_source_occurrences(
                occurrences,
                str(row["surface_text"]),
                evidence_row,
                structured_pages[str(row["paper_id"])],
                f"mention {mention_id}",
            )
            provenance = ProvenanceScope(str(row["provenance_scope"]))
            owner_kind = OwnerKind(str(row["owner_kind"]))
            owner_id = str(row["owner_id"])
            if provenance == ProvenanceScope.OWNER_EVIDENCE:
                if str(evidence_row["owner_kind"]) != owner_kind.value or str(evidence_row["owner_id"]) != owner_id:
                    raise ContractError("OWNER_EVIDENCE does not point to the mention owner")
            else:
                expected_context = owner_id if owner_kind == OwnerKind.CONTEXT else observation_context.get(owner_id)
                if str(evidence_row["owner_kind"]) != "CONTEXT" or str(evidence_row["owner_id"]) != expected_context:
                    raise ContractError("STUDY_CONTEXT evidence does not point to the owner's study context")
            context_id = owner_id if owner_kind == OwnerKind.CONTEXT else observation_context.get(owner_id)
            source_flags = []
            if (owner_kind.value, owner_id) in unmodeled_owners:
                source_flags.append("UNMODELED_CONDITION")
            if str(evidence_row["alignment_status"]) != AlignmentStatus.EXACT_UNIQUE.value:
                source_flags.append("SOURCE_EVIDENCE_EXACT_AMBIGUOUS")
            augmented = dict(row)
            augmented.update(
                context_id=context_id,
                source_flags_json=json.dumps(source_flags),
                assertion_status="REPORTED",
            )
            mention = Mention.from_mapping(augmented)
            valid_mentions.append(mention)
            qualifier_versions.add(mention.qualifier_vocab_version)
            schema_versions.add(mention.extraction_schema_version)
        except Exception as exc:
            invalid_mentions.append(
                InvalidMentionRecord(
                    mention_id=mention_id,
                    paper_id=str(row.get("paper_id") or "").strip() or None,
                    raw_json=raw_json,
                    error_codes=("ADAPTER_CONTRACT_ERROR",),
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
    if len(schema_versions) != 1 or len(qualifier_versions) != 1:
        raise ContractError(
            f"valid mentions require one schema and qualifier vocabulary version; schemas={sorted(schema_versions)}, qualifiers={sorted(qualifier_versions)}"
        )
    schema_version = next(iter(schema_versions))
    qualifier_version = next(iter(qualifier_versions))
    if schema_version != policy.schema_version:
        raise ContractError(f"extraction schema {schema_version} != policy schema {policy.schema_version}")
    if hasattr(policy, "qualifier_vocab_version") and qualifier_version != policy.qualifier_vocab_version:
        raise ContractError(
            f"qualifier vocabulary {qualifier_version} != policy qualifier vocabulary {policy.qualifier_vocab_version}"
        )
    assert_referential_integrity(valid_mentions, owner_ids, frozenset(evidence_by_id), eligible_ids)

    hints: list[AuthorityHint] = []
    if authority_hints_path is not None:
        hint_path = Path(authority_hints_path)
        hint_frame = _read(
            hint_path,
            frozenset({"mention_id", "authority", "external_id", "snapshot_version", "source", "confidence"}),
            "authority_hints",
        )
        known_ids = {item.mention_id for item in valid_mentions}
        for row in hint_frame.to_dict("records"):
            hint = AuthorityHint(
                mention_id=str(row["mention_id"]), authority=str(row["authority"]),
                external_id=str(row["external_id"]), snapshot_version=str(row["snapshot_version"]),
                source=str(row["source"]), confidence=_nullable_float(row.get("confidence")),
            )
            if hint.mention_id not in known_ids:
                raise ContractError(f"authority hint references unknown/invalid mention {hint.mention_id}")
            hints.append(hint)
        paths["authority_hints"] = hint_path
    source_paths = tuple(sorted({str(path.resolve()) for path in paths.values()} | {str(path) for path in verified_source_paths}))
    hashes = tuple(sorted((name, _file_hash(path)) for name, path in paths.items()))
    input_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "table_hashes": hashes,
                "structured_content_hashes": sorted(structured_hashes.items()),
                "structured_source_hashes": sorted(structured_source_hashes.items()),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return MufasaInputs(
        mentions=tuple(sorted(valid_mentions, key=lambda item: item.mention_id)),
        invalid_mentions=tuple(sorted(invalid_mentions, key=lambda item: item.mention_id)),
        authority_hints=tuple(sorted(hints, key=lambda item: (item.mention_id, item.authority, item.external_id))),
        input_fingerprint=input_fingerprint,
        table_hashes=hashes,
        extraction_schema_version=schema_version,
        qualifier_vocab_version=qualifier_version,
        condition_vocab_versions=tuple(sorted(condition_versions)),
        model_id=descriptor["model"],
        prompt_version=descriptor["prompt_version"],
        settings_hash=descriptor["settings_hash"],
        source_fingerprint=descriptor["source_fingerprint"],
        structured_content_hashes=tuple(sorted(structured_hashes.items())),
        structured_source_hashes=tuple(sorted(structured_source_hashes.items())),
        source_paths=source_paths,
        extraction_generation_id=str(publication_marker["generation_id"]),
    )


def _file_hash(path: Path) -> str:
    from ..io import sha256_file

    return sha256_file(path)


def _nullable_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        raise ContractError("boolean is not a valid source offset")
    return int(value)


def _nullable_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _clean_record(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in row.items():
        if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
            result[str(key)] = None
        elif hasattr(value, "item"):
            result[str(key)] = value.item()
        else:
            result[str(key)] = value
    return result


def _read_json_descriptor(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"required {label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _validate_run_descriptor(
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    eligible: pd.DataFrame,
    status: pd.DataFrame,
    policy: ResolverPolicy,
    corpus_root: Path,
) -> dict[str, str]:
    identity_fields = (
        "schema_version",
        "prompt_version",
        "qualifier_vocab_version",
        "condition_vocab_version",
        "model",
        "settings_hash",
        "source_fingerprint",
    )
    for field in identity_fields:
        left, right = str(manifest.get(field, "")).strip(), str(summary.get(field, "")).strip()
        if not left or left != right:
            raise ContractError(f"extraction manifest/run_summary disagree or omit {field}: {left!r} vs {right!r}")
    expected = {
        "schema_version": policy.schema_version,
        "prompt_version": policy.extraction_prompt_version,
        "qualifier_vocab_version": policy.qualifier_vocab_version,
        "condition_vocab_version": policy.condition_vocab_version,
    }
    for field, value in expected.items():
        if str(manifest[field]) != value:
            raise ContractError(f"extraction {field}={manifest[field]!r}, expected policy value {value!r}")
    for field in ("settings_hash", "source_fingerprint"):
        value = str(manifest[field])
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            raise ContractError(f"extraction {field} is not a SHA-256 hex digest")
    selection_contract = {
        "pipeline_status": "ok",
        "rights_status": "permitted",
        "retraction_status_not": "retracted",
        "deduplicate": "paper_id_keep_last",
        "sort": "paper_id_stable",
        "schema_version": policy.schema_version,
    }
    if manifest.get("selection_contract") != selection_contract:
        raise ContractError(
            "extraction manifest selection_contract does not match the active schema and corpus selection policy"
        )
    source_hasher = hashlib.sha256(json.dumps(selection_contract, sort_keys=True).encode())
    for row in eligible.to_dict("records"):
        record = {field: _source_clean_text(row.get(field)) for field in SOURCE_FINGERPRINT_FIELDS}
        record["structured_basename"] = _artifact_path(
            row, "structured_path", corpus_root, "structured", ".json"
        ).name
        source_hasher.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        )
    recomputed_source = source_hasher.hexdigest()
    if recomputed_source != manifest["source_fingerprint"]:
        raise ContractError(
            f"extraction source_fingerprint does not match documents manifest: {manifest['source_fingerprint']} != {recomputed_source}"
        )
    if int(manifest.get("source_rows", -1)) != len(eligible):
        raise ContractError("extraction manifest source_rows does not match eligible documents")
    if int(summary.get("eligible_papers", -1)) != len(eligible):
        raise ContractError("run_summary eligible_papers does not match eligible documents")
    if set(status["paper_id"].astype(str)) != set(eligible["paper_id"].astype(str)):
        raise ContractError("extraction_status must contain exactly every manifest-eligible paper")
    actual_status = {str(key): int(value) for key, value in status["status"].value_counts().items()}
    summary_status = {str(key): int(value) for key, value in dict(summary.get("paper_status", {})).items()}
    if actual_status != summary_status:
        raise ContractError(f"run_summary paper_status disagrees with extraction_status: {summary_status} != {actual_status}")
    batches = manifest.get("batches")
    if not isinstance(batches, Mapping) or not batches:
        raise ContractError("extraction manifest has no completed batch ledger")
    if any(not isinstance(item, Mapping) or item.get("complete") is not True for item in batches.values()):
        raise ContractError("every extraction manifest batch must be marked complete")
    if sum(int(item.get("papers", 0)) for item in batches.values()) != len(eligible):
        raise ContractError("extraction batch paper counts do not cover eligible documents exactly")
    return {field: str(manifest[field]) for field in identity_fields}


def _validate_cross_table_integrity(
    observations: pd.DataFrame,
    contexts: pd.DataFrame,
    evidence: pd.DataFrame,
    mentions: pd.DataFrame,
) -> None:
    """Reject every orphan or cross-paper edge before per-mention parsing."""

    contexts_by_id = {
        str(row["context_id"]): str(row["paper_id"])
        for row in contexts[["context_id", "paper_id"]].to_dict("records")
    }
    observations_by_id = {
        str(row["observation_id"]): str(row["paper_id"])
        for row in observations[["observation_id", "paper_id"]].to_dict("records")
    }
    evidence_by_id = {
        str(row["evidence_id"]): row for row in evidence.to_dict("records")
    }

    for row in observations.to_dict("records"):
        observation_id = str(row["observation_id"])
        paper_id = str(row["paper_id"])
        context_id = str(row["context_id"])
        context_paper = contexts_by_id.get(context_id)
        if context_paper is None:
            raise ContractError(f"observation {observation_id} references absent context {context_id}")
        if context_paper != paper_id:
            raise ContractError(
                f"observation {observation_id} and context {context_id} belong to different papers"
            )
        evidence_id = str(row["evidence_id"])
        evidence_row = evidence_by_id.get(evidence_id)
        if evidence_row is None:
            raise ContractError(f"observation {observation_id} references absent evidence {evidence_id}")
        if (
            str(evidence_row["owner_kind"]) != OwnerKind.OBSERVATION.value
            or str(evidence_row["owner_id"]) != observation_id
            or str(evidence_row["paper_id"]) != paper_id
        ):
            raise ContractError(
                f"observation {observation_id} evidence {evidence_id} does not point back to its owner/paper"
            )

    for row in evidence.to_dict("records"):
        evidence_id = str(row["evidence_id"])
        paper_id = str(row["paper_id"])
        owner_kind = str(row["owner_kind"])
        owner_id = str(row["owner_id"])
        if owner_kind == OwnerKind.CONTEXT.value:
            owner_paper = contexts_by_id.get(owner_id)
        elif owner_kind == OwnerKind.OBSERVATION.value:
            owner_paper = observations_by_id.get(owner_id)
        else:
            raise ContractError(f"evidence {evidence_id} has invalid owner_kind {owner_kind!r}")
        if owner_paper is None:
            raise ContractError(f"evidence {evidence_id} references absent {owner_kind} owner {owner_id}")
        if owner_paper != paper_id:
            raise ContractError(f"evidence {evidence_id} and owner {owner_id} belong to different papers")

    for row in mentions.to_dict("records"):
        mention_id = str(row.get("mention_id") or "<blank>")
        paper_id = str(row.get("paper_id"))
        owner_kind = str(row.get("owner_kind"))
        owner_id = str(row.get("owner_id"))
        if owner_kind == OwnerKind.CONTEXT.value:
            owner_paper = contexts_by_id.get(owner_id)
        elif owner_kind == OwnerKind.OBSERVATION.value:
            owner_paper = observations_by_id.get(owner_id)
        else:
            raise ContractError(f"mention {mention_id} has invalid owner_kind {owner_kind!r}")
        if owner_paper is None:
            raise ContractError(f"mention {mention_id} references absent {owner_kind} owner {owner_id}")
        if owner_paper != paper_id:
            raise ContractError(f"mention {mention_id} and owner {owner_id} belong to different papers")


def _resolve_structured_path(document_row: Mapping[str, Any], corpus_root: Path) -> Path:
    path = _artifact_path(document_row, "structured_path", corpus_root, "structured", ".json")
    if not path.is_file():
        raise ContractError(
            f"structured parser source missing for {document_row['paper_id']}; resolved={path}"
        )
    return path.resolve()


def _resolve_markdown_path(document_row: Mapping[str, Any], corpus_root: Path) -> Path:
    path = _artifact_path(document_row, "markdown_path", corpus_root, "markdown", ".md")
    if not path.is_file():
        raise ContractError(
            f"validated Markdown fallback missing for {document_row['paper_id']}; resolved={path}"
        )
    return path.resolve()


def _artifact_path(
    document_row: Mapping[str, Any],
    field: str,
    corpus_root: Path,
    subfolder: str,
    suffix: str,
) -> Path:
    recorded_text = _source_clean_text(document_row.get(field))
    recorded = Path(recorded_text) if recorded_text else None
    if recorded is not None and recorded.is_file():
        return recorded.resolve()
    return corpus_root / "parsed" / subfolder / f"{document_row['paper_id']}{suffix}"


def _load_verified_structured(
    path: Path, document_row: Mapping[str, Any]
) -> tuple[dict[int, str], str, str, str]:
    structured = _read_json_descriptor(path, f"structured paper {document_row['paper_id']}")
    if str(structured.get("paper_id")) != str(document_row["paper_id"]):
        raise ContractError(f"structured paper_id mismatch in {path}")
    if str(structured.get("pdf_sha256")) != str(document_row["pdf_sha256"]):
        raise ContractError(f"structured PDF hash mismatch for {document_row['paper_id']}")
    if structured.get("parse_status") != "ok":
        raise ContractError(f"structured parse is not successful for {document_row['paper_id']}")
    if document_row.get("openalex_id") and structured.get("openalex_id"):
        if str(document_row["openalex_id"]) != str(structured["openalex_id"]):
            raise ContractError(f"structured OpenAlex identity mismatch for {document_row['paper_id']}")
    for field in ("parser_name", "parser_version", "parser_config_hash"):
        expected = _source_clean_text(document_row.get(field))
        if expected and _source_clean_text(structured.get(field)) != expected:
            raise ContractError(
                f"structured {field} mismatch for {document_row['paper_id']}: "
                f"{structured.get(field)!r} != {expected!r}"
            )
    parser_config_hash = _source_clean_text(structured.get("parser_config_hash"))
    if not parser_config_hash:
        raise ContractError(f"structured parser_config_hash is blank for {document_row['paper_id']}")
    raw_pages = structured.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise ContractError(f"structured paper has no pages: {path}")
    pages: dict[int, str] = {}
    content_pages = []
    for index, item in enumerate(raw_pages):
        if not isinstance(item, Mapping) or isinstance(item.get("page"), bool):
            raise ContractError(f"structured page {index} is malformed: {path}")
        page = int(item["page"])
        text = item.get("text")
        if page < 1 or page in pages or not isinstance(text, str):
            raise ContractError(f"structured page {page} is duplicated/invalid: {path}")
        pages[page] = text
        content_pages.append({"page": page, "text": text})
    payload = {
        "paper_id": structured["paper_id"],
        "pdf_sha256": structured["pdf_sha256"],
        "parser_name": structured.get("parser_name"),
        "parser_version": structured.get("parser_version"),
        "parser_config_hash": structured.get("parser_config_hash"),
        "pages": content_pages,
    }
    content_hash = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return pages, content_hash, _file_hash(path), parser_config_hash


def _source_clean_text(value: Any) -> str:
    """Mirror the extraction notebook's source-fingerprint text coercion."""

    if value is None or (isinstance(value, float) and value != value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


_PAGE_MARKER = re.compile(r"(?m)^<!-- MUFASA_PDF_PAGE: (\d+) -->\s*$")


def _load_verified_markdown(
    path: Path, document_row: Mapping[str, Any]
) -> tuple[dict[int, str], str, str]:
    raw = path.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    expected = _source_clean_text(document_row.get("markdown_sha256"))
    if not expected or source_hash != expected:
        raise ContractError(f"Markdown fallback SHA-256 mismatch for {document_row['paper_id']}")
    try:
        markdown = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError(f"Markdown fallback is not UTF-8 for {document_row['paper_id']}") from exc
    markers = list(_PAGE_MARKER.finditer(markdown))
    if not markers:
        raise ContractError(f"Markdown fallback has no MUFASA PDF page markers for {document_row['paper_id']}")
    pages: dict[int, str] = {}
    content_pages = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(markdown)
        page = int(marker.group(1))
        body = markdown[marker.end() : end].lstrip("\r\n")
        heading = re.match(rf"## PDF page {page}\s*\r?\n", body)
        if heading:
            body = body[heading.end() :]
        text = body.strip()
        if page < 1 or page in pages:
            raise ContractError(f"Markdown fallback has invalid/duplicate page {page}")
        pages[page] = text
        content_pages.append({"page": page, "text": text})
    payload = {
        "paper_id": str(document_row["paper_id"]),
        "source_kind": "markdown_fallback",
        "markdown_sha256": source_hash,
        "pages": content_pages,
    }
    content_hash = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return pages, content_hash, source_hash


def _verify_evidence_against_structured(
    evidence_row: Mapping[str, Any],
    raw_pages: Mapping[int, str],
    structured_source_hash: str,
    label: str,
) -> None:
    if str(evidence_row.get("source_sha256")) != structured_source_hash:
        raise ContractError(f"{label} source_sha256 does not match immutable structured parser file")
    text = str(evidence_row["evidence_text"])
    occurrences = _occurrences(evidence_row["occurrences_json"], f"{label}.occurrences_json")
    if int(evidence_row["occurrence_count"]) != len(occurrences):
        raise ContractError(f"{label} occurrence_count disagrees with occurrences_json")
    for page, start, end in occurrences:
        raw = raw_pages.get(page)
        if raw is None or end > len(raw) or raw[start:end] != text:
            raise ContractError(f"{label} does not equal immutable raw-page slice {(page, start, end)}")
    alignment = str(evidence_row["alignment_status"])
    start = _nullable_int(evidence_row.get("char_start"))
    end = _nullable_int(evidence_row.get("char_end"))
    if alignment == AlignmentStatus.EXACT_UNIQUE.value:
        expected = (int(evidence_row["page_start"]), start, end)
        if len(occurrences) != 1 or None in expected or occurrences[0] != expected:
            raise ContractError(f"{label} EXACT_UNIQUE chosen span is inconsistent")
    elif alignment == AlignmentStatus.EXACT_AMBIGUOUS.value:
        if len(occurrences) < 2 or start is not None or end is not None:
            raise ContractError(f"{label} EXACT_AMBIGUOUS bookkeeping is inconsistent")
    else:
        raise ContractError(f"{label} alignment_status is invalid: {alignment!r}")


def _verify_source_occurrences(
    occurrences: tuple[tuple[int, int, int], ...],
    surface_text: str,
    evidence_row: Mapping[str, Any],
    raw_pages: Mapping[int, str],
    label: str,
) -> None:
    """Verify every page-local mention span against immutable evidence text."""

    evidence_text = str(evidence_row["evidence_text"])
    evidence_occurrences = _occurrences(
        evidence_row["occurrences_json"], f"{label}.source_evidence.occurrences_json"
    )
    if int(evidence_row["occurrence_count"]) != len(evidence_occurrences):
        raise ContractError("source evidence occurrence_count disagrees with occurrences_json")
    alignment = str(evidence_row["alignment_status"])
    if alignment == AlignmentStatus.EXACT_UNIQUE.value:
        evidence_start = _nullable_int(evidence_row.get("char_start"))
        evidence_end = _nullable_int(evidence_row.get("char_end"))
        if len(evidence_occurrences) != 1 or evidence_start is None or evidence_end is None:
            raise ContractError("EXACT_UNIQUE source evidence lacks a unique raw span")
        if evidence_occurrences[0] != (int(evidence_row["page_start"]), evidence_start, evidence_end):
            raise ContractError("source evidence unique occurrence disagrees with chosen span")
    elif alignment == AlignmentStatus.EXACT_AMBIGUOUS.value:
        if len(evidence_occurrences) < 2:
            raise ContractError("EXACT_AMBIGUOUS source evidence lacks multiple occurrences")
        if _nullable_int(evidence_row.get("char_start")) is not None or _nullable_int(evidence_row.get("char_end")) is not None:
            raise ContractError("ambiguous source evidence must not select one span")
    else:
        raise ContractError(f"source evidence alignment_status is invalid: {alignment!r}")
    for page, start, end in occurrences:
        page_text = raw_pages.get(page)
        if page_text is None:
            raise ContractError(f"mention occurrence references absent raw page {page}")
        if end > len(page_text) or page_text[start:end] != surface_text:
            raise ContractError(
                f"immutable raw-page slice at {(page, start, end)} does not equal surface_text {surface_text!r}"
            )
        containing = [
            (ev_start, ev_end)
            for ev_page, ev_start, ev_end in evidence_occurrences
            if ev_page == page and ev_start <= start and end <= ev_end
        ]
        if not containing:
            raise ContractError(
                f"mention occurrence {(page, start, end)} is outside every exact source-evidence occurrence"
            )
        if not any(evidence_text[start - ev_start : end - ev_start] == surface_text for ev_start, _ev_end in containing):
            raise ContractError(
                f"raw source slice at {(page, start, end)} does not equal surface_text {surface_text!r}"
            )
