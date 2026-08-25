import hashlib
import json

import pandas as pd
import pytest

from conftest import republish_table, write_extraction_fixture
from scripts.entity_resolution.adapters.mufasa import load_mufasa_inputs
from scripts.entity_resolution.contracts import ContractError


def test_adapter_verifies_manifest_and_immutable_raw_source(workspace_tmp, policy):
    paths = write_extraction_fixture(workspace_tmp, policy)
    loaded = load_mufasa_inputs(paths["extraction"], paths["documents"], policy)
    assert len(loaded.mentions) == 1
    assert not loaded.invalid_mentions
    assert loaded.mentions[0].surface_text == "nitrate"
    assert len(loaded.structured_content_hashes) == 1
    marker = json.loads((paths["extraction"] / "current-generation.json").read_text())
    assert loaded.extraction_generation_id == marker["generation_id"]


def test_adapter_requires_atomic_publication_pointer(workspace_tmp, policy):
    paths = write_extraction_fixture(workspace_tmp, policy)
    (paths["extraction"] / "current-generation.json").unlink()
    with pytest.raises(ContractError, match="current extraction generation is missing"):
        load_mufasa_inputs(paths["extraction"], paths["documents"], policy)


def test_adapter_rejects_tampered_published_parquet_before_loading(workspace_tmp, policy):
    paths = write_extraction_fixture(workspace_tmp, policy)
    path = paths["published"] / "entity_mentions.parquet"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ContractError, match="missing or corrupt"):
        load_mufasa_inputs(paths["extraction"], paths["documents"], policy)


def test_partial_context_profile_cannot_exclude_resolver_input(workspace_tmp, policy):
    paths = write_extraction_fixture(workspace_tmp, policy)
    profiles = pd.read_parquet(paths["published"] / "paper_profiles.parquet")
    profiles.loc[0, "coverage_complete"] = False
    republish_table(paths, "paper_profiles.parquet", profiles)
    status = pd.read_parquet(paths["published"] / "extraction_status.parquet")
    status.loc[0, "context_coverage_complete"] = False
    status.loc[0, "resolver_eligible"] = False
    republish_table(paths, "extraction_status.parquet", status)
    with pytest.raises(ContractError, match="partial context profile cannot exclude"):
        load_mufasa_inputs(paths["extraction"], paths["documents"], policy)


def test_parser_config_hash_is_derived_from_sidecar_and_manifest_conflict_fails(workspace_tmp, policy):
    paths = write_extraction_fixture(workspace_tmp, policy)
    documents = pd.read_parquet(paths["documents"])
    assert "parser_config_hash" not in documents.columns
    load_mufasa_inputs(paths["extraction"], paths["documents"], policy)

    documents["parser_config_hash"] = "e" * 64
    documents.to_parquet(paths["documents"], index=False)
    with pytest.raises(ContractError, match="parser_config_hash mismatch|parser_config_hash conflicts"):
        load_mufasa_inputs(paths["extraction"], paths["documents"], policy)


def test_adapter_reconstructs_hash_verified_markdown_fallback_pages(workspace_tmp, policy):
    paths = write_extraction_fixture(workspace_tmp, policy, markdown_fallback=True)
    loaded = load_mufasa_inputs(paths["extraction"], paths["documents"], policy)
    assert [item.surface_text for item in loaded.mentions] == ["nitrate"]
    assert loaded.structured_source_hashes[0][1] == hashlib.sha256(paths["markdown"].read_bytes()).hexdigest()


def test_tampered_markdown_fallback_fails_hash_verification(workspace_tmp, policy):
    paths = write_extraction_fixture(workspace_tmp, policy, markdown_fallback=True)
    paths["markdown"].write_text("tampered", encoding="utf-8")
    with pytest.raises(ContractError, match="Markdown fallback SHA-256 mismatch"):
        load_mufasa_inputs(paths["extraction"], paths["documents"], policy)


def test_tampered_mention_offset_is_recorded_invalid_not_auto_coerced(workspace_tmp, policy):
    paths = write_extraction_fixture(workspace_tmp, policy, second_valid_mention=True)
    path = paths["published"] / "entity_mentions.parquet"
    frame = pd.read_parquet(path)
    frame.loc[frame["mention_id"] == "MEN1", "source_char_start"] = 11
    frame.loc[frame["mention_id"] == "MEN1", "source_occurrences_json"] = json.dumps(
        [{"page": 1, "char_start": 11, "char_end": 19}]
    )
    republish_table(paths, "entity_mentions.parquet", frame)
    loaded = load_mufasa_inputs(paths["extraction"], paths["documents"], policy)
    assert [item.mention_id for item in loaded.invalid_mentions] == ["MEN1"]
    assert [item.mention_id for item in loaded.mentions] == ["MEN2"]


def test_tampered_evidence_span_fails_entire_preflight(workspace_tmp, policy):
    paths = write_extraction_fixture(workspace_tmp, policy)
    path = paths["published"] / "evidence_spans.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "evidence_text"] = "Fabricated text"
    republish_table(paths, "evidence_spans.parquet", frame)
    with pytest.raises(ContractError, match="immutable raw-page slice"):
        load_mufasa_inputs(paths["extraction"], paths["documents"], policy)


def test_evidence_source_hash_must_match_structured_file(workspace_tmp, policy):
    paths = write_extraction_fixture(workspace_tmp, policy)
    path = paths["published"] / "evidence_spans.parquet"
    evidence = pd.read_parquet(path)
    evidence.loc[0, "source_sha256"] = "0" * 64
    republish_table(paths, "evidence_spans.parquet", evidence)

    with pytest.raises(ContractError, match="source_sha256"):
        load_mufasa_inputs(paths["extraction"], paths["documents"], policy)


def test_manifest_summary_mismatch_fails_closed(workspace_tmp, policy):
    paths = write_extraction_fixture(workspace_tmp, policy)
    summary_path = paths["extraction"] / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["prompt_version"] = "wrong"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ContractError, match="prompt_version"):
        load_mufasa_inputs(paths["extraction"], paths["documents"], policy)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("parser_version", "2.0"),
        ("pdf_sha256", "f" * 64),
    ],
)
def test_source_fingerprint_invalidates_changed_parser_or_pdf_metadata(
    workspace_tmp, policy, field, replacement
):
    paths = write_extraction_fixture(workspace_tmp, policy)
    documents = pd.read_parquet(paths["documents"])
    documents.loc[0, field] = replacement
    documents.to_parquet(paths["documents"], index=False)

    with pytest.raises(ContractError, match="source_fingerprint|mismatch"):
        load_mufasa_inputs(paths["extraction"], paths["documents"], policy)


def test_observation_context_cross_paper_edge_fails_preflight(workspace_tmp, policy):
    paths = write_extraction_fixture(workspace_tmp, policy)
    path = paths["published"] / "study_contexts.parquet"
    contexts = pd.read_parquet(path)
    contexts.loc[0, "paper_id"] = "P2"
    republish_table(paths, "study_contexts.parquet", contexts)

    with pytest.raises(ContractError, match="different papers"):
        load_mufasa_inputs(paths["extraction"], paths["documents"], policy)


def test_observation_evidence_must_point_back_to_owner(workspace_tmp, policy):
    paths = write_extraction_fixture(workspace_tmp, policy)
    path = paths["published"] / "evidence_spans.parquet"
    evidence = pd.read_parquet(path)
    evidence.loc[0, "owner_id"] = "OBS-OTHER"
    republish_table(paths, "evidence_spans.parquet", evidence)

    with pytest.raises(ContractError, match="does not point back"):
        load_mufasa_inputs(paths["extraction"], paths["documents"], policy)


def test_orphan_extra_evidence_is_rejected(workspace_tmp, policy):
    paths = write_extraction_fixture(workspace_tmp, policy)
    path = paths["published"] / "evidence_spans.parquet"
    evidence = pd.read_parquet(path)
    orphan = evidence.iloc[0].copy()
    orphan["evidence_id"] = "EVD-ORPHAN"
    orphan["owner_kind"] = "CONTEXT"
    orphan["owner_id"] = "CTX-MISSING"
    evidence = pd.concat([evidence, orphan.to_frame().T], ignore_index=True)
    republish_table(paths, "evidence_spans.parquet", evidence)

    with pytest.raises(ContractError, match="references absent CONTEXT owner"):
        load_mufasa_inputs(paths["extraction"], paths["documents"], policy)


def test_mention_owner_must_share_paper(workspace_tmp, policy):
    paths = write_extraction_fixture(workspace_tmp, policy)
    path = paths["published"] / "entity_mentions.parquet"
    mentions = pd.read_parquet(path)
    mentions.loc[0, "paper_id"] = "P2"
    republish_table(paths, "entity_mentions.parquet", mentions)

    with pytest.raises(ContractError, match="mention MEN1 and owner OBS1 belong to different papers"):
        load_mufasa_inputs(paths["extraction"], paths["documents"], policy)


def test_gated_paper_is_excluded_from_resolution_but_keeps_its_rows(workspace_tmp, policy):
    """A paper the full text says is not real science, or not about Africa, is
    still extracted and still published; it just does not feed the graph. It is
    out of scope here, not an integrity failure."""
    paths = write_extraction_fixture(workspace_tmp, policy, resolver_eligible=False)

    with pytest.raises(ContractError, match="no manifest-eligible papers"):
        load_mufasa_inputs(paths["extraction"], paths["documents"], policy)

    # The rows are untouched on disk - exclusion is a routing decision, not a delete.
    assert len(pd.read_parquet(paths["published"] / "entity_mentions.parquet")) == 1
    assert len(pd.read_parquet(paths["published"] / "observations.parquet")) == 1


def test_alias_provenance_survives_into_the_resolver(workspace_tmp, policy):
    """Model-supplied aliases are trusted and may drive an automatic merge, so
    the evidence for that merge has to survive. `aliases` carries the names that
    matching compares; `aliases_json` keeps kind, language and stated_in_paper
    so a merge can be explained afterwards."""
    paths = write_extraction_fixture(workspace_tmp, policy)
    loaded = load_mufasa_inputs(paths["extraction"], paths["documents"], policy)
    mention = loaded.mentions[0]

    assert mention.aliases == ("NO3-",)
    records = json.loads(mention.aliases_json)
    assert records == [{"text": "NO3-", "kind": "FORMULA", "language": "",
                        "stated_in_paper": False}]


def test_extraction_aliases_and_instance_id_reach_the_resolver(workspace_tmp, policy):
    """Aliases are how a paper writing one name reaches a paper writing another,
    and instance_local_id is the only statement that two wordings mean one
    sample. Both are carried by extraction and both were previously read as
    defaults, which left the resolver matching on bare strings while every test
    still passed."""
    paths = write_extraction_fixture(workspace_tmp, policy)
    path = paths["published"] / "entity_mentions.parquet"
    mentions = pd.read_parquet(path)
    mentions.loc[0, "instance_local_id"] = "SAMPLE_1"
    republish_table(paths, "entity_mentions.parquet", mentions)

    loaded = load_mufasa_inputs(paths["extraction"], paths["documents"], policy)
    assert loaded.mentions[0].aliases == ("NO3-",)
    assert loaded.mentions[0].instance_local_id == "SAMPLE_1"


@pytest.mark.parametrize("column", ["aliases_json", "instance_local_id"])
def test_missing_alias_or_instance_column_fails_closed(workspace_tmp, policy, column):
    paths = write_extraction_fixture(workspace_tmp, policy)
    path = paths["published"] / "entity_mentions.parquet"
    republish_table(paths, "entity_mentions.parquet", pd.read_parquet(path).drop(columns=[column]))
    with pytest.raises(ContractError, match=f"missing columns.*{column}"):
        load_mufasa_inputs(paths["extraction"], paths["documents"], policy)


def test_evidence_source_kind_uses_the_span_vocabulary(workspace_tmp, policy):
    """evidence_spans.source_kind is TEXT/TABLE/FIGURE. It was once compared
    against structured_json/markdown_fallback, which no real extraction output
    can satisfy and which a mixed-source paper would fail anyway."""
    paths = write_extraction_fixture(workspace_tmp, policy)
    path = paths["published"] / "evidence_spans.parquet"
    evidence = pd.read_parquet(path)
    evidence.loc[0, "source_kind"] = "structured_json"
    republish_table(paths, "evidence_spans.parquet", evidence)

    with pytest.raises(ContractError, match="source_kind has unknown values"):
        load_mufasa_inputs(paths["extraction"], paths["documents"], policy)


def test_document_source_kind_must_match_the_verified_artifact(workspace_tmp, policy):
    paths = write_extraction_fixture(workspace_tmp, policy)
    path = paths["published"] / "extraction_status.parquet"
    status = pd.read_parquet(path)
    status.loc[0, "source_kind"] = "markdown_fallback"
    republish_table(paths, "extraction_status.parquet", status)

    with pytest.raises(ContractError, match="was extracted from markdown_fallback"):
        load_mufasa_inputs(paths["extraction"], paths["documents"], policy)
