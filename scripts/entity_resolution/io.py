"""Deterministic Parquet I/O, hashing, manifests, and atomic publication."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .audit import build_review_rows, run_summary
from .authorities import AuthoritySnapshot
from .contracts import (
    AliasTrust,
    AuthorityLink,
    CanonicalAlias,
    CanonicalEntity,
    Candidate,
    ConstraintType,
    DecisionStatus,
    DecisionMethod,
    EntityInstance,
    EntityRedirect,
    EntityRelation,
    EntityType,
    EventType,
    LifecycleStatus,
    ResolutionConflict,
    ResolutionConstraint,
    ResolutionDecision,
    ResolutionEvent,
    ResolutionRun,
    Mention,
    Proposal,
    ProposalKind,
    InvalidMentionRecord,
    json_value,
)
from .registry import RegistrySnapshot


class PublicationError(RuntimeError):
    pass


RESOLUTION_RUN_FORMAT = "mufasa-resolution-run-v1"
RESOLUTION_POINTER_FORMAT = "mufasa-resolution-current-v1"


def _publish_directory(source: Path, destination: Path) -> None:
    """Atomically publish a new directory, tolerating brief Windows file scans."""

    for attempt in range(5):
        try:
            os.rename(source, destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (2**attempt))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def combined_file_hash(paths: Iterable[str | Path]) -> str:
    entries = []
    for path in sorted((Path(item) for item in paths), key=lambda item: item.as_posix()):
        entries.append((path.name, sha256_file(path)))
    return canonical_json_hash(entries)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def atomic_write_json(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, path)


def atomic_write_parquet(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    check = pd.read_parquet(temporary)
    if list(check.columns) != list(frame.columns) or len(check) != len(frame):
        temporary.unlink(missing_ok=True)
        raise PublicationError(f"Parquet verification failed for {path}")
    os.replace(temporary, path)


def records_frame(records: Iterable[Any], columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for record in records:
        value = _jsonable(record)
        if not isinstance(value, dict):
            raise TypeError("records_frame accepts dataclass or mapping records")
        rows.append(value)
    return pd.DataFrame(rows, columns=list(columns))


REGISTRY_SCHEMAS: dict[str, list[str]] = {
    "canonical_entities.parquet": [
        "concept_id", "preferred_label", "entity_type", "identity_qualifiers_json",
        "lifecycle_status", "seed_mention_id", "created_run_id", "updated_run_id",
        "registry_version", "provenance_json",
    ],
    "entity_instances.parquet": [
        "instance_id", "paper_id", "context_id", "local_label", "entity_type", "concept_id",
        "identity_qualifiers_json", "created_run_id", "updated_run_id", "registry_version",
        "source_mention_ids_json",
    ],
    "canonical_aliases.parquet": [
        "alias_id", "concept_id", "alias_text", "normalized_key", "entity_type", "language",
        "region", "source", "trust_level", "created_run_id", "registry_version",
        "alias_kind", "stated_in_paper", "provenance_json",
    ],
    "canonical_authority_links.parquet": [
        "authority_link_id", "concept_id", "authority", "external_id",
        "authority_snapshot_version", "source", "created_run_id", "registry_version",
    ],
    "entity_relations.parquet": [
        "relation_id", "source_concept_id", "relation_type", "target_concept_id",
        "provenance", "reviewed", "created_run_id", "registry_version",
    ],
    "entity_redirects.parquet": ["retired_id", "active_id", "event_id", "registry_version"],
    "resolution_constraints.parquet": [
        "constraint_id", "constraint_type", "mention_id", "target_id", "reviewer", "reason",
        "active", "created_run_id", "registry_version",
    ],
    "resolution_events.parquet": [
        "event_id", "event_type", "subject_id", "object_id", "run_id", "registry_version",
        "reason_codes_json", "reviewer", "occurred_at",
    ],
}


def write_registry_snapshot(
    snapshot: RegistrySnapshot,
    destination: str | Path,
    *,
    manifest_extra: Mapping[str, Any] | None = None,
    authority_snapshot: AuthoritySnapshot | None = None,
) -> Path:
    snapshot.validate(authority_snapshot)
    destination = Path(destination)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        event_rows = [
            {
                **_jsonable(event),
                "reason_codes_json": json.dumps(list(event.reason_codes), ensure_ascii=False, separators=(",", ":")),
            }
            for event in snapshot.events
        ]
        for row in event_rows:
            row.pop("reason_codes", None)
        tables = {
            "canonical_entities.parquet": records_frame(snapshot.canonical_entities, REGISTRY_SCHEMAS["canonical_entities.parquet"]),
            "entity_instances.parquet": records_frame(snapshot.entity_instances, REGISTRY_SCHEMAS["entity_instances.parquet"]),
            "canonical_aliases.parquet": records_frame(snapshot.canonical_aliases, REGISTRY_SCHEMAS["canonical_aliases.parquet"]),
            "canonical_authority_links.parquet": records_frame(snapshot.authority_links, REGISTRY_SCHEMAS["canonical_authority_links.parquet"]),
            "entity_relations.parquet": records_frame(snapshot.entity_relations, REGISTRY_SCHEMAS["entity_relations.parquet"]),
            "entity_redirects.parquet": records_frame(snapshot.redirects, REGISTRY_SCHEMAS["entity_redirects.parquet"]),
            "resolution_constraints.parquet": records_frame(snapshot.constraints, REGISTRY_SCHEMAS["resolution_constraints.parquet"]),
            "resolution_events.parquet": pd.DataFrame(event_rows, columns=REGISTRY_SCHEMAS["resolution_events.parquet"]),
        }
        hashes = {}
        for filename, frame in tables.items():
            path = temporary / filename
            atomic_write_parquet(frame, path)
            hashes[filename] = sha256_file(path)
        manifest = {
            "registry_version": snapshot.version,
            "network_required": False,
            "table_hashes": dict(sorted(hashes.items())),
            **dict(manifest_extra or {}),
        }
        atomic_write_json(manifest, temporary / "registry_manifest.json")
        if destination.exists():
            existing_manifest = destination / "registry_manifest.json"
            if existing_manifest.is_file() and existing_manifest.read_bytes() == (temporary / "registry_manifest.json").read_bytes():
                shutil.rmtree(temporary)
                return destination
            raise PublicationError(f"immutable registry destination already exists: {destination}")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_registry_snapshot(
    path: str | Path | None,
    *,
    authority_snapshot: AuthoritySnapshot | None = None,
) -> RegistrySnapshot:
    if path is None:
        return RegistrySnapshot.empty()
    root = Path(path)
    manifest_path = root / "registry_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("network_required") not in (None, False):
        raise PublicationError("registry manifest declares runtime network dependency")
    for filename, expected in manifest["table_hashes"].items():
        actual = sha256_file(root / filename)
        if actual != expected:
            raise PublicationError(f"registry table hash mismatch: {filename}")

    def rows(filename: str) -> list[dict[str, Any]]:
        return pd.read_parquet(root / filename).where(pd.notna(pd.read_parquet(root / filename)), None).to_dict("records")

    canonical = tuple(
        CanonicalEntity(
            concept_id=row["concept_id"], preferred_label=row["preferred_label"],
            entity_type=EntityType(row["entity_type"]), identity_qualifiers_json=row["identity_qualifiers_json"],
            lifecycle_status=LifecycleStatus(row["lifecycle_status"]), seed_mention_id=row["seed_mention_id"],
            created_run_id=row["created_run_id"], updated_run_id=row["updated_run_id"],
            registry_version=row["registry_version"], provenance_json=row["provenance_json"],
        ) for row in rows("canonical_entities.parquet")
    )
    instances = tuple(EntityInstance(**{**row, "entity_type": EntityType(row["entity_type"])}) for row in rows("entity_instances.parquet"))
    aliases = tuple(
        CanonicalAlias(
            **{**row, "entity_type": EntityType(row["entity_type"]), "trust_level": AliasTrust(row["trust_level"])}
        ) for row in rows("canonical_aliases.parquet")
    )
    links = tuple(AuthorityLink(**row) for row in rows("canonical_authority_links.parquet"))
    relations = tuple(EntityRelation(**row) for row in rows("entity_relations.parquet"))
    redirects = tuple(EntityRedirect(**row) for row in rows("entity_redirects.parquet"))
    constraints = tuple(
        ResolutionConstraint(**{**row, "constraint_type": ConstraintType(row["constraint_type"])})
        for row in rows("resolution_constraints.parquet")
    )
    events = []
    for row in rows("resolution_events.parquet"):
        events.append(
            ResolutionEvent(
                event_id=row["event_id"], event_type=EventType(row["event_type"]),
                subject_id=row["subject_id"], object_id=row["object_id"], run_id=row["run_id"],
                registry_version=row["registry_version"],
                reason_codes=tuple(json.loads(row["reason_codes_json"])), reviewer=row["reviewer"],
                occurred_at=row["occurred_at"],
            )
        )
    snapshot = RegistrySnapshot(
        version=manifest["registry_version"], canonical_entities=canonical, entity_instances=instances,
        canonical_aliases=aliases, authority_links=links, entity_relations=relations,
        redirects=redirects, constraints=constraints, events=tuple(events),
        manifest_hash=sha256_file(manifest_path),
    )
    snapshot.validate(authority_snapshot)
    return snapshot


def write_resolution_run(
    run: ResolutionRun,
    destination: str | Path,
    *,
    conflicts: Sequence[ResolutionConflict] = (),
    capability_manifest: Mapping[str, Any] | None = None,
    registry_diff_value: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    generations_root = destination / "run-generations"
    generations_root.mkdir(parents=True, exist_ok=True)
    staging = generations_root / f".staging-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    mentions = {item.mention_id: item for item in run.mentions}
    invalid_by_id = {item.mention_id: item for item in run.invalid_inputs}
    review_rows = build_review_rows(run)
    review_by_mention: dict[str, dict[str, Any]] = {}
    for review_row in review_rows:
        for mention_id in json.loads(str(review_row["mention_ids_json"])):
            review_by_mention[str(mention_id)] = review_row
    resolution_rows = []
    for decision in run.decisions:
        mention = mentions.get(decision.mention_id)
        raw = {}
        if mention is None:
            invalid = invalid_by_id[decision.mention_id]
            try:
                raw = json.loads(invalid.raw_json)
            except json.JSONDecodeError:
                raw = {}
        review = review_by_mention.get(decision.mention_id)
        review_flags = (
            list(decision.reason_codes)
            if decision.status in {DecisionStatus.REVIEW_REQUIRED, DecisionStatus.UNRESOLVED}
            else [code for code in decision.reason_codes if code.startswith("AUTO_MERGE_REVIEW:")]
        )
        resolution_rows.append(
            {
                "mention_id": decision.mention_id,
                "source_mention_id": mention.source_mention_id if mention else raw.get("source_mention_id"),
                "source_evidence_id": mention.source_evidence_id if mention else raw.get("source_evidence_id"),
                "source_page": mention.source_page if mention else raw.get("source_page"),
                "source_char_start": mention.source_char_start if mention else raw.get("source_char_start"),
                "source_char_end": mention.source_char_end if mention else raw.get("source_char_end"),
                "source_occurrence_count": mention.source_occurrence_count if mention else raw.get("source_occurrence_count"),
                "source_occurrences_json": mention.source_occurrences_json if mention else raw.get("source_occurrences_json"),
                "source_alignment_status": mention.source_alignment_status.value if mention else raw.get("source_alignment_status"),
                "provenance_scope": mention.provenance_scope.value if mention else raw.get("provenance_scope"),
                "paper_id": mention.paper_id if mention else raw.get("paper_id"),
                "owner_kind": mention.owner_kind.value if mention else raw.get("owner_kind"),
                "owner_id": mention.owner_id if mention else raw.get("owner_id"),
                "context_id": mention.context_id if mention else raw.get("context_id"),
                "role": mention.role if mention else raw.get("role"),
                "surface_text": mention.surface_text if mention else raw.get("surface_text"),
                "atom_text": mention.atom_text if mention else raw.get("atom_text"),
                "entity_type": mention.entity_type.value if mention else raw.get("entity_type"),
                "identity_scope": mention.identity_scope.value if mention else raw.get("identity_scope"),
                "qualifiers_json": (
                    json.dumps([_jsonable(item) for item in mention.qualifiers], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    if mention else raw.get("qualifiers_json")
                ),
                # Carried so an alias-driven merge can be audited against the
                # exact records extraction supplied, including whether the paper
                # stated the equivalence or the model supplied it.
                "aliases_json": mention.aliases_json if mention else raw.get("aliases_json", "[]"),
                "instance_local_id": mention.instance_local_id if mention else raw.get("instance_local_id", ""),
                "qualifier_vocab_version": mention.qualifier_vocab_version if mention else raw.get("qualifier_vocab_version"),
                "extraction_schema_version": mention.extraction_schema_version if mention else raw.get("extraction_schema_version"),
                "assertion_status": mention.assertion_status.value if mention else raw.get("assertion_status"),
                "domain": mention.domain if mention else raw.get("domain"),
                "language": mention.language if mention else raw.get("language"),
                "country_code": mention.country_code if mention else raw.get("country_code"),
                "source_flags_json": json.dumps(list(mention.source_flags), separators=(",", ":")) if mention else raw.get("source_flags_json", "[]"),
                "concept_id": decision.concept_id,
                "instance_id": decision.instance_id,
                "proposal_id": decision.proposal_id,
                "decision_status": decision.status.value,
                "decision_method": decision.method.value,
                "reason_codes_json": json.dumps(list(decision.reason_codes), separators=(",", ":")),
                "review_needed": review is not None,
                "review_flags_json": json.dumps(review_flags, separators=(",", ":")),
                "review_priority": float(review["priority"]) if review is not None else 0.0,
                "top_score": decision.top_score,
                "runner_up_score": decision.runner_up_score,
                "margin": decision.margin,
                "calibrated_probability": decision.calibrated_probability,
                "candidate_count": decision.candidate_count,
                "candidate_set_hash": decision.candidate_set_hash,
                "memo_key": decision.memo_key,
                "memo_reused": decision.memo_reused,
                "run_id": run.run_id,
                "policy_version": run.policy_version,
                "registry_version": run.result_registry_version,
                "authority_manifest_hash": run.authority_manifest_hash,
                "resolver_code_version": run.resolver_code_version,
            }
        )
    resolution_columns = list(resolution_rows[0]) if resolution_rows else [
        "mention_id", "source_mention_id", "source_evidence_id", "source_page", "source_char_start",
        "source_char_end", "source_occurrence_count", "source_occurrences_json", "source_alignment_status",
        "provenance_scope", "paper_id", "owner_kind", "owner_id",
        "context_id", "role", "surface_text", "atom_text", "entity_type", "identity_scope",
        "aliases_json", "instance_local_id",
        "qualifiers_json", "qualifier_vocab_version", "extraction_schema_version", "assertion_status",
        "domain", "language", "country_code", "source_flags_json", "concept_id", "instance_id", "proposal_id", "decision_status",
        "decision_method", "reason_codes_json", "review_needed", "review_flags_json",
        "review_priority", "top_score", "runner_up_score", "margin",
        "calibrated_probability", "candidate_count", "candidate_set_hash", "memo_key", "memo_reused",
        "run_id", "policy_version", "registry_version", "authority_manifest_hash", "resolver_code_version",
    ]
    candidate_rows = []
    for item in run.candidates:
        candidate_rows.append(
            {
                "mention_id": item.mention_id, "target_kind": item.target_kind, "target_id": item.target_id,
                "target_label": item.target_label, "target_entity_type": item.target_entity_type.value,
                "method": item.method, "score": item.score,
                "features_json": json.dumps(dict(item.features), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "conflicts_json": json.dumps(list(item.conflicts), separators=(",", ":")), "rank": item.rank,
            }
        )
    proposal_rows = []
    for item in run.proposals:
        row = _jsonable(item)
        row["member_mention_ids_json"] = json.dumps(row.pop("member_mention_ids"), separators=(",", ":"))
        row["reason_codes_json"] = json.dumps(row.pop("reason_codes"), separators=(",", ":"))
        row["authority_keys_json"] = json.dumps(row.pop("authority_keys"), separators=(",", ":"))
        proposal_rows.append(row)
    conflict_rows = [_jsonable(item) for item in conflicts]
    memo_rows = [
        {
            "memo_key": item.memo_key,
            "mention_id": item.mention_id,
            "candidate_set_hash": item.candidate_set_hash,
            "decision_json": json.dumps(_jsonable(item), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "policy_version": run.policy_version,
            "registry_version": run.base_registry_version,
            "authority_manifest_hash": run.authority_manifest_hash,
        }
        for item in run.decisions if item.memo_key
    ]
    tables = {
        "mention_resolutions.parquet": pd.DataFrame(resolution_rows, columns=resolution_columns),
        "resolution_candidates.parquet": pd.DataFrame(candidate_rows, columns=[
            "mention_id", "target_kind", "target_id", "target_label", "target_entity_type", "method",
            "score", "features_json", "conflicts_json", "rank",
        ]),
        "resolution_proposals.parquet": pd.DataFrame(proposal_rows, columns=[
            "proposal_id", "proposal_kind", "member_mention_ids_json", "preferred_label", "entity_type",
            "paper_id", "context_id", "primary_concept_id", "primary_concept_proposal_id",
            "authority_keys_json", "auto_approved", "method", "reason_codes_json",
        ]),
        "resolution_conflicts.parquet": pd.DataFrame(conflict_rows, columns=[
            "conflict_id", "mention_id", "target_id", "conflict_code", "severity", "detail", "run_id",
        ]),
        "resolution_memo.parquet": pd.DataFrame(memo_rows, columns=[
            "memo_key", "mention_id", "candidate_set_hash", "decision_json", "policy_version",
            "registry_version", "authority_manifest_hash",
        ]),
        "invalid_inputs.parquet": pd.DataFrame(
            [_jsonable(item) | {"error_codes_json": json.dumps(list(item.error_codes), separators=(",", ":"))} for item in run.invalid_inputs],
            columns=["mention_id", "paper_id", "raw_json", "error_codes", "detail", "error_codes_json"],
        ).drop(columns=["error_codes"]),
    }
    hashes = {}
    try:
        for filename, frame in tables.items():
            path = staging / filename
            atomic_write_parquet(frame, path)
            hashes[filename] = sha256_file(path)
        review = pd.DataFrame(review_rows, columns=[
        "review_task_id", "entity_type", "identity_scope", "atom_text", "occurrence_count",
        "mention_ids_json", "candidate_set_hash", "reason_codes_json", "priority",
        "review_mode", "blocking", "resolved_concept_id", "proposal_id",
        "propagation_policy", "propagation_approved",
        "reviewer_decision", "reviewer_target_id", "reviewer_notes",
        ])
        atomic_write_csv(review, staging / "resolution_review.csv")
        hashes["resolution_review.csv"] = sha256_file(staging / "resolution_review.csv")
        summary = run_summary(run)
        atomic_write_json(summary, staging / "run_summary.json")
        hashes["run_summary.json"] = sha256_file(staging / "run_summary.json")
        if registry_diff_value is not None:
            atomic_write_json(registry_diff_value, staging / "registry_diff.json")
            hashes["registry_diff.json"] = sha256_file(staging / "registry_diff.json")
        capabilities = dict(capability_manifest or {})
        extraction_input = capabilities.get("extraction_input", {})
        extraction_generation_id = (
            str(extraction_input.get("generation_id", ""))
            if isinstance(extraction_input, Mapping)
            else ""
        )
        if extraction_generation_id and not re.fullmatch(r"[0-9a-f]{24}", extraction_generation_id):
            raise PublicationError("extraction generation ID must be 24 lowercase hex characters")
        manifest_identity = {
            "format": RESOLUTION_RUN_FORMAT,
            "run_id": run.run_id,
            "input_fingerprint": run.input_fingerprint,
            "base_registry_version": run.base_registry_version,
            "result_registry_version": run.result_registry_version,
            "policy_version": run.policy_version,
            "normalization_version": run.normalization_version,
            "authority_manifest_hash": run.authority_manifest_hash,
            "resolver_code_version": run.resolver_code_version,
            "extraction_generation_id": extraction_generation_id,
            "effective_controls": dict(run.effective_controls),
            "candidate_counters": {
                "generated": run.generated_candidate_count,
                "retained": len(run.candidates),
            },
            "capabilities": capabilities,
            "network_required": False,
            "artifact_hashes": dict(sorted(hashes.items())),
        }
        generation_id = canonical_json_hash(manifest_identity)[:24]
        manifest = {**manifest_identity, "generation_id": generation_id}
        # This manifest is the success record and is deliberately written only
        # after every artifact exists and has been hashed.
        atomic_write_json(manifest, staging / "run_manifest.json")
        final_dir = generations_root / generation_id
        if final_dir.exists():
            existing_manifest = final_dir / "run_manifest.json"
            if not existing_manifest.is_file() or json.loads(existing_manifest.read_text(encoding="utf-8")) != manifest:
                raise PublicationError(f"resolution generation-ID collision: {generation_id}")
            _verify_resolution_artifacts(final_dir, manifest)
            shutil.rmtree(staging)
        else:
            _publish_directory(staging, final_dir)
        _verify_resolution_artifacts(final_dir, manifest)
        pointer = {
            "format": RESOLUTION_POINTER_FORMAT,
            "generation_id": generation_id,
            "directory": f"run-generations/{generation_id}",
            "run_id": run.run_id,
            "manifest_sha256": sha256_file(final_dir / "run_manifest.json"),
        }
        # One small atomic pointer is the only mutation visible to readers.
        atomic_write_json(pointer, destination / "current-run.json")
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _verify_resolution_artifacts(root: Path, manifest: Mapping[str, Any]) -> None:
    hashes = manifest.get("artifact_hashes")
    mandatory = {
        "mention_resolutions.parquet",
        "resolution_candidates.parquet",
        "resolution_proposals.parquet",
        "resolution_conflicts.parquet",
        "resolution_memo.parquet",
        "invalid_inputs.parquet",
        "resolution_review.csv",
        "run_summary.json",
    }
    if not isinstance(hashes, Mapping) or not mandatory <= set(hashes):
        raise PublicationError("resolution run manifest has an incomplete artifact set")
    for filename, expected in hashes.items():
        if Path(filename).name != filename or not isinstance(expected, str):
            raise PublicationError(f"unsafe resolution artifact entry: {filename!r}")
        artifact = root / filename
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise PublicationError(f"resolution artifact hash mismatch: {filename}")
    actual_files = {item.name for item in root.iterdir() if item.is_file()}
    expected_files = set(hashes) | {"run_manifest.json"}
    if actual_files != expected_files:
        raise PublicationError("resolution generation contains unmanifested or missing files")


def _resolve_resolution_generation(path: str | Path) -> tuple[Path, dict[str, Any]]:
    requested = Path(path)
    pointer_path = requested / "current-run.json"
    if pointer_path.is_file():
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PublicationError(f"cannot read resolution current pointer: {exc}") from exc
        required = {"format", "generation_id", "directory", "run_id", "manifest_sha256"}
        if not isinstance(pointer, Mapping) or set(pointer) != required:
            raise PublicationError("resolution current pointer has an invalid schema")
        if pointer["format"] != RESOLUTION_POINTER_FORMAT:
            raise PublicationError("unsupported resolution pointer format")
        generation_id = str(pointer["generation_id"])
        if not re.fullmatch(r"[0-9a-f]{24}", generation_id):
            raise PublicationError("invalid resolution generation ID")
        if pointer["directory"] != f"run-generations/{generation_id}":
            raise PublicationError("resolution pointer directory is inconsistent")
        root = (requested / str(pointer["directory"])).resolve()
        expected_parent = (requested / "run-generations").resolve()
        if root.parent != expected_parent or root.name != generation_id:
            raise PublicationError("resolution pointer directory is unsafe")
        manifest_path = root / "run_manifest.json"
        if not manifest_path.is_file() or sha256_file(manifest_path) != pointer["manifest_sha256"]:
            raise PublicationError("resolution success manifest is missing or corrupt")
    elif (requested / "run_manifest.json").is_file():
        # Explicit immutable generation paths are useful for reproducible evals.
        root = requested.resolve()
        manifest_path = root / "run_manifest.json"
        pointer = None
    else:
        raise PublicationError("resolution current pointer or immutable generation manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationError(f"cannot read resolution success manifest: {exc}") from exc
    if not isinstance(manifest, Mapping) or manifest.get("format") != RESOLUTION_RUN_FORMAT:
        raise PublicationError("unsupported resolution run manifest format")
    generation_id = str(manifest.get("generation_id", ""))
    identity = dict(manifest)
    identity.pop("generation_id", None)
    if not re.fullmatch(r"[0-9a-f]{24}", generation_id) or canonical_json_hash(identity)[:24] != generation_id:
        raise PublicationError("resolution generation ID does not match manifest identity")
    if root.name != generation_id:
        raise PublicationError("immutable resolution directory name does not match generation ID")
    if pointer is not None and (pointer["generation_id"] != generation_id or pointer["run_id"] != manifest.get("run_id")):
        raise PublicationError("resolution pointer disagrees with success manifest")
    _verify_resolution_artifacts(root, manifest)
    return root, dict(manifest)


def load_resolution_run(path: str | Path) -> ResolutionRun:
    """Load the current or an explicit immutable run generation, fail closed."""

    root, manifest = _resolve_resolution_generation(path)
    resolutions = pd.read_parquet(root / "mention_resolutions.parquet")
    invalid_frame = pd.read_parquet(root / "invalid_inputs.parquet")
    invalid = tuple(
        InvalidMentionRecord(
            mention_id=str(row["mention_id"]), paper_id=_nullable(row.get("paper_id")),
            raw_json=str(row["raw_json"]), error_codes=tuple(json.loads(row["error_codes_json"])),
            detail=str(row["detail"]),
        ) for row in invalid_frame.to_dict("records")
    )
    invalid_ids = {item.mention_id for item in invalid}
    mentions = []
    decisions = []
    for raw_row in resolutions.to_dict("records"):
        row = {key: _none_if_na(value) for key, value in raw_row.items()}
        if str(row["mention_id"]) not in invalid_ids:
            mentions.append(Mention.from_mapping(row))
        decisions.append(
            ResolutionDecision(
                mention_id=str(row["mention_id"]), status=DecisionStatus(str(row["decision_status"])),
                method=DecisionMethod(str(row["decision_method"])), concept_id=_nullable(row.get("concept_id")),
                instance_id=_nullable(row.get("instance_id")), proposal_id=_nullable(row.get("proposal_id")),
                reason_codes=tuple(json.loads(row["reason_codes_json"])),
                top_score=_nullable_float(row.get("top_score")), runner_up_score=_nullable_float(row.get("runner_up_score")),
                margin=_nullable_float(row.get("margin")), calibrated_probability=_nullable_float(row.get("calibrated_probability")),
                candidate_count=int(row.get("candidate_count") or 0), candidate_set_hash=_nullable(row.get("candidate_set_hash")),
                memo_key=_nullable(row.get("memo_key")), memo_reused=bool(row.get("memo_reused", False)),
            )
        )
    candidate_frame = pd.read_parquet(root / "resolution_candidates.parquet")
    candidates = tuple(
        Candidate(
            mention_id=str(row["mention_id"]), target_kind=str(row["target_kind"]), target_id=str(row["target_id"]),
            target_label=str(row["target_label"]), target_entity_type=EntityType(str(row["target_entity_type"])),
            method=str(row["method"]), score=float(row["score"]),
            features=tuple(sorted(json.loads(row["features_json"]).items())),
            conflicts=tuple(json.loads(row["conflicts_json"])), rank=int(row["rank"]),
        ) for row in candidate_frame.to_dict("records")
    )
    candidate_counters = manifest.get("candidate_counters")
    if not isinstance(candidate_counters, Mapping):
        raise PublicationError("resolution run manifest omits candidate_counters")
    generated_candidates = int(candidate_counters.get("generated", -1))
    retained_candidates = int(candidate_counters.get("retained", -1))
    if retained_candidates != len(candidates) or generated_candidates < retained_candidates:
        raise PublicationError("resolution run candidate counters disagree with retained candidate artifact")
    proposal_frame = pd.read_parquet(root / "resolution_proposals.parquet")
    proposals = tuple(
        Proposal(
            proposal_id=str(row["proposal_id"]), proposal_kind=ProposalKind(str(row["proposal_kind"])),
            member_mention_ids=tuple(json.loads(row["member_mention_ids_json"])),
            preferred_label=str(row["preferred_label"]), entity_type=EntityType(str(row["entity_type"])),
            paper_id=_nullable(row.get("paper_id")), context_id=_nullable(row.get("context_id")),
            primary_concept_id=_nullable(row.get("primary_concept_id")),
            primary_concept_proposal_id=_nullable(row.get("primary_concept_proposal_id")),
            authority_keys=tuple(tuple(item) for item in json.loads(row["authority_keys_json"])),
            auto_approved=bool(row["auto_approved"]), method=DecisionMethod(str(row["method"])),
            reason_codes=tuple(json.loads(row["reason_codes_json"])),
        ) for row in proposal_frame.to_dict("records")
    )
    return ResolutionRun(
        run_id=manifest["run_id"], input_fingerprint=manifest["input_fingerprint"],
        base_registry_version=manifest["base_registry_version"],
        result_registry_version=manifest["result_registry_version"],
        policy_version=manifest["policy_version"],
        normalization_version=manifest["normalization_version"],
        authority_manifest_hash=manifest["authority_manifest_hash"],
        resolver_code_version=manifest["resolver_code_version"], mentions=tuple(mentions),
        decisions=tuple(decisions), proposals=proposals, candidates=candidates,
        generated_candidate_count=generated_candidates,
        invalid_inputs=invalid,
        effective_controls=tuple(sorted(manifest.get("effective_controls", {}).items())),
    )


def _none_if_na(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _nullable(value: Any) -> str | None:
    value = _none_if_na(value)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _nullable_float(value: Any) -> float | None:
    value = _none_if_na(value)
    return None if value is None else float(value)
