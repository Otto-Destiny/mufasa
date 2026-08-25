from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

MUFASA_ROOT = Path(__file__).resolve().parents[2]
if str(MUFASA_ROOT) not in sys.path:
    sys.path.insert(0, str(MUFASA_ROOT))

from scripts.entity_resolution.contracts import (  # noqa: E402
    AlignmentStatus,
    CanonicalEntity,
    EntityType,
    IdentityScope,
    LifecycleStatus,
    Mention,
    OwnerKind,
    ProvenanceScope,
    Qualifier,
)
from scripts.entity_resolution.policy import load_policy  # noqa: E402
from scripts.entity_resolution.registry import RegistrySnapshot  # noqa: E402


@pytest.fixture
def policy():
    return load_policy()


@pytest.fixture
def workspace_tmp():
    root = MUFASA_ROOT / ".test_runtime"
    root.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="entity-resolution-", dir=root))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def mention(
    mention_id: str,
    text: str,
    entity_type: EntityType,
    *,
    scope: IdentityScope = IdentityScope.CANONICAL,
    paper_id: str = "P1",
    context_id: str = "CTX1",
    source_mention_id: str | None = None,
    qualifiers: tuple[Qualifier, ...] = (),
    alignment: AlignmentStatus = AlignmentStatus.EXACT_UNIQUE,
    flags: tuple[str, ...] = (),
    aliases: tuple[str, ...] = (),
    instance_local_id: str = "",
) -> Mention:
    start = 0 if alignment == AlignmentStatus.EXACT_UNIQUE else None
    end = len(text) if alignment == AlignmentStatus.EXACT_UNIQUE else None
    occurrences = (
        [{"page": 1, "char_start": 0, "char_end": len(text)}]
        if alignment == AlignmentStatus.EXACT_UNIQUE
        else [
            {"page": 1, "char_start": 0, "char_end": len(text)},
            {"page": 1, "char_start": len(text) + 1, "char_end": len(text) * 2 + 1},
        ]
    )
    return Mention(
        mention_id=mention_id,
        source_mention_id=source_mention_id or f"SRC-{mention_id}",
        source_evidence_id=f"EVD-{source_mention_id or mention_id}",
        paper_id=paper_id,
        owner_kind=OwnerKind.OBSERVATION,
        owner_id=f"OBS-{mention_id}",
        role="SUBJECT",
        surface_text=text,
        atom_text=text,
        entity_type=entity_type,
        identity_scope=scope,
        qualifiers=tuple(sorted(qualifiers)),
        context_id=context_id,
        source_page=1,
        source_char_start=start,
        source_char_end=end,
        source_occurrence_count=len(occurrences),
        source_occurrences_json=json.dumps(occurrences, separators=(",", ":")),
        source_alignment_status=alignment,
        provenance_scope=ProvenanceScope.OWNER_EVIDENCE,
        qualifier_vocab_version="mufasa-qualifier-v1.1-candidate.1",
        extraction_schema_version="2.3-candidate.1",
        source_flags=flags,
        aliases=aliases,
        aliases_json=json.dumps(
            [
                {
                    "text": alias,
                    "kind": "SPELLING_VARIANT",
                    "language": "",
                    "stated_in_paper": False,
                }
                for alias in aliases
            ],
            separators=(",", ":"),
        ),
        instance_local_id=instance_local_id,
    )


def concept(
    concept_id: str,
    label: str,
    entity_type: EntityType,
    *,
    qualifiers_json: str = "[]",
) -> CanonicalEntity:
    return CanonicalEntity(
        concept_id=concept_id,
        preferred_label=label,
        entity_type=entity_type,
        identity_qualifiers_json=qualifiers_json,
        lifecycle_status=LifecycleStatus.ACTIVE,
        seed_mention_id=f"SEED-{concept_id}",
        created_run_id="RUN-0",
        updated_run_id="RUN-0",
        registry_version="registry-v000001",
        provenance_json="{}",
    )


def snapshot(*concepts: CanonicalEntity) -> RegistrySnapshot:
    return RegistrySnapshot(version="registry-v000001", canonical_entities=tuple(concepts))


def write_extraction_fixture(
    root: Path,
    policy,
    *,
    second_valid_mention: bool = False,
    markdown_fallback: bool = False,
    resolver_eligible: bool = True,
) -> dict[str, Path]:
    extraction = root / "extraction"
    corpus = root / "corpus"
    structured_dir = corpus / "parsed" / "structured"
    markdown_dir = corpus / "parsed" / "markdown"
    manifests_dir = corpus / "manifests"
    extraction.mkdir(parents=True)
    structured_dir.mkdir(parents=True)
    markdown_dir.mkdir(parents=True)
    manifests_dir.mkdir(parents=True)
    paper_id = "P1"
    page_text = "Groundwater nitrate is 12 mg/L."
    quote = page_text
    pdf_hash = "a" * 64
    structured = {
        "schema_version": "1.0",
        "paper_id": paper_id,
        "openalex_id": "https://openalex.org/W1",
        "pdf_sha256": pdf_hash,
        "parser_name": "test-parser",
        "parser_version": "1.0",
        "parser_config_hash": "b" * 64,
        "parse_status": "ok",
        "pages": [{"page": 1, "metadata": {}, "text": page_text}],
    }
    structured_path = structured_dir / f"{paper_id}.json"
    if not markdown_fallback:
        structured_path.write_text(json.dumps(structured), encoding="utf-8")
    markdown_path = markdown_dir / f"{paper_id}.md"
    markdown_path.write_text(
        f"<!-- MUFASA_PDF_PAGE: 1 -->\n## PDF page 1\n{page_text}\n",
        encoding="utf-8",
    )
    markdown_hash = hashlib.sha256(markdown_path.read_bytes()).hexdigest()
    source_path = markdown_path if markdown_fallback else structured_path
    structured_source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    documents = pd.DataFrame(
        [
            {
                "paper_id": paper_id,
                "openalex_id": "https://openalex.org/W1",
                "pipeline_status": "ok",
                "rights_status": "permitted",
                "retraction_status": "clear",
                "structured_path": "/kaggle/working/stale/P1.json",
                "markdown_path": "/kaggle/working/stale/P1.md",
                "pdf_sha256": pdf_hash,
                "parser_name": "test-parser",
                "parser_version": "1.0",
                "markdown_sha256": markdown_hash,
            }
        ]
    )
    documents_path = manifests_dir / "documents.parquet"
    documents.to_parquet(documents_path, index=False)
    contexts = pd.DataFrame(
        [
            {
                "context_id": "CTX1",
                "paper_id": paper_id,
                "conditions_json": "[]",
                "condition_vocab_version": policy.condition_vocab_version,
            }
        ]
    )
    observations = pd.DataFrame(
        [
            {
                "observation_id": "OBS1",
                "paper_id": paper_id,
                "context_id": "CTX1",
                "conditions_json": "[]",
                "condition_vocab_version": policy.condition_vocab_version,
                "assertion_status": "REPORTED",
                "evidence_id": "EVD1",
            }
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "evidence_id": "EVD1",
                "paper_id": paper_id,
                "owner_kind": "OBSERVATION",
                "owner_id": "OBS1",
                "page_start": 1,
                "page_end": 1,
                "evidence_text": quote,
                "char_start": 0,
                "char_end": len(quote),
                "occurrence_count": 1,
                "occurrences_json": json.dumps([{"page": 1, "char_start": 0, "char_end": len(quote)}]),
                "alignment_status": "EXACT_UNIQUE",
                # Where in the page the quote came from. The document artifact
                # it was extracted from is recorded in extraction_status.
                "source_kind": "TEXT",
                "source_sha256": structured_source_hash,
            }
        ]
    )
    mention_rows = [
        {
            "mention_id": "MEN1",
            "paper_id": paper_id,
            "owner_kind": "OBSERVATION",
            "owner_id": "OBS1",
            "source_mention_id": "SRC1",
            "source_evidence_id": "EVD1",
            "source_page": 1,
            "source_char_start": 12,
            "source_char_end": 19,
            "source_occurrence_count": 1,
            "source_occurrences_json": json.dumps([{"page": 1, "char_start": 12, "char_end": 19}]),
            "source_alignment_status": "EXACT_UNIQUE",
            "provenance_scope": "OWNER_EVIDENCE",
            "role": "OUTCOME",
            "surface_text": "nitrate",
            "atom_text": "nitrate",
            "entity_type": "CHEMICAL",
            "identity_scope": "CANONICAL",
            "instance_local_id": "",
            "aliases_json": json.dumps(
                [{"text": "NO3-", "kind": "FORMULA", "language": "",
                  "stated_in_paper": False}]
            ),
            "qualifiers_json": "[]",
            "qualifier_vocab_version": policy.qualifier_vocab_version,
            "extraction_schema_version": policy.schema_version,
            "canonical_id": None,
            "resolution_status": "UNRESOLVED",
        }
    ]
    if second_valid_mention:
        second = dict(mention_rows[0])
        second.update(
            mention_id="MEN2",
            source_mention_id="SRC2",
            surface_text="Groundwater",
            atom_text="groundwater",
            entity_type="ENVIRONMENTAL_FEATURE",
            source_char_start=0,
            source_char_end=11,
            source_occurrences_json=json.dumps([{"page": 1, "char_start": 0, "char_end": 11}]),
        )
        mention_rows.append(second)
    for filename, frame in (
        ("study_contexts.parquet", contexts),
        ("observations.parquet", observations),
        ("evidence_spans.parquet", evidence),
        ("entity_mentions.parquet", pd.DataFrame(mention_rows)),
        ("extraction_status.parquet", pd.DataFrame([{
            "paper_id": paper_id, "status": "complete", "error": "",
            "source_kind": "markdown_fallback" if markdown_fallback else "structured_json",
            "resolver_eligible": resolver_eligible,
            "resolver_excluded_reason": "" if resolver_eligible else "NOT_AFRICA_RELEVANT",
            "context_coverage_complete": True,
        }])),
    ):
        frame.to_parquet(extraction / filename, index=False)
    pd.DataFrame([{"paper_id": paper_id, "coverage_complete": True}]).to_parquet(
        extraction / "paper_profiles.parquet", index=False
    )
    pd.DataFrame(columns=["paper_id"]).to_parquet(extraction / "training_pairs.parquet", index=False)
    selection_contract = {
        "pipeline_status": "ok",
        "rights_status": "permitted",
        "retraction_status_not": "retracted",
        "deduplicate": "paper_id_keep_last",
        "sort": "paper_id_stable",
        "schema_version": policy.schema_version,
    }
    source_hasher = hashlib.sha256(json.dumps(selection_contract, sort_keys=True).encode())
    source_record = {
        "paper_id": paper_id,
        "openalex_id": "https://openalex.org/W1",
        "pdf_sha256": pdf_hash,
        "parser_name": "test-parser",
        "parser_version": "1.0",
        "parser_config_hash": "" if markdown_fallback else "b" * 64,
        "markdown_sha256": markdown_hash,
        "structured_basename": "P1.json",
    }
    source_hasher.update(json.dumps(source_record, sort_keys=True, separators=(",", ":")).encode())
    source_fingerprint = source_hasher.hexdigest()
    settings_hash = "d" * 64
    shared = {
        "schema_version": policy.schema_version,
        "prompt_version": policy.extraction_prompt_version,
        "qualifier_vocab_version": policy.qualifier_vocab_version,
        "condition_vocab_version": policy.condition_vocab_version,
        "model": "test/model",
        "settings_hash": settings_hash,
        "source_fingerprint": source_fingerprint,
    }
    manifest = {
        **shared,
        "selection_contract": selection_contract,
        "source_rows": 1,
        "batches": {"0": {"complete": True, "papers": 1}},
    }
    summary = {
        **shared,
        "eligible_papers": 1,
        "paper_status": {"complete": 1},
    }
    (extraction / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (extraction / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    table_names = (
        "study_contexts.parquet",
        "observations.parquet",
        "entity_mentions.parquet",
        "evidence_spans.parquet",
        "paper_profiles.parquet",
        "training_pairs.parquet",
        "extraction_status.parquet",
    )
    table_meta = {}
    for filename in table_names:
        path = extraction / filename
        frame = pd.read_parquet(path)
        table_meta[filename] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(frame),
            "columns": list(frame.columns),
        }
    identity = {
        "format": "mufasa-parquet-generation-v1",
        "settings_hash": settings_hash,
        "source_fingerprint": source_fingerprint,
        "schema_version": policy.schema_version,
        "prompt_version": policy.extraction_prompt_version,
        "qualifier_vocab_version": policy.qualifier_vocab_version,
        "condition_vocab_version": policy.condition_vocab_version,
        "tables": table_meta,
    }
    generation_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    marker = {
        **identity,
        "generation_id": generation_id,
        "created_at": "2026-08-14T00:00:00+00:00",
        "directory": f"published-generations/{generation_id}",
    }
    generation = extraction / "published-generations" / generation_id
    generation.mkdir(parents=True)
    for filename in table_names:
        (extraction / filename).replace(generation / filename)
    (generation / "generation.json").write_text(json.dumps(marker), encoding="utf-8")
    (extraction / "current-generation.json").write_text(json.dumps(marker), encoding="utf-8")
    return {
        "extraction": extraction,
        "published": generation,
        "documents": documents_path,
        "structured": structured_path,
        "markdown": markdown_path,
    }


def republish_table(paths: dict[str, Path], filename: str, frame: pd.DataFrame) -> None:
    """Publish a changed fixture table as a new immutable generation."""

    frame.to_parquet(paths["published"] / filename, index=False)
    marker_path = paths["extraction"] / "current-generation.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    for table_name, metadata in marker["tables"].items():
        table_path = paths["published"] / table_name
        table = pd.read_parquet(table_path)
        metadata.update(
            sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
            rows=len(table),
            columns=list(table.columns),
        )
    identity = {
        key: marker[key]
        for key in (
            "format",
            "settings_hash",
            "source_fingerprint",
            "schema_version",
            "prompt_version",
            "qualifier_vocab_version",
            "condition_vocab_version",
            "tables",
        )
    }
    new_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    old_dir = paths["published"]
    new_dir = old_dir.parent / new_id
    old_dir.rename(new_dir)
    marker["generation_id"] = new_id
    marker["directory"] = f"published-generations/{new_id}"
    (new_dir / "generation.json").write_text(json.dumps(marker), encoding="utf-8")
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    paths["published"] = new_dir
