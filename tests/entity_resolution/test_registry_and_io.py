from dataclasses import replace
import json

import pandas as pd
import pytest

from conftest import concept, mention
from scripts.entity_resolution.authorities import AuthorityRecord, AuthoritySnapshot
from scripts.entity_resolution.contracts import (
    AuthorityLink,
    ConstraintType,
    ContractError,
    EntityInstance,
    EntityRelation,
    EntityType,
    EventType,
)
from scripts.entity_resolution.io import (
    load_registry_snapshot,
    load_resolution_run,
    write_registry_snapshot,
    write_resolution_run,
)
from scripts.entity_resolution.pipeline import commit_resolution_run, preflight_capabilities, resolve_batch
from scripts.entity_resolution.registry import (
    RegistrySnapshot,
    SplitPartition,
    merge_concepts,
    reassign_instance_concept,
    record_human_constraint,
    split_concept,
)


def test_dependency_version_incompatibility_fails_actionably(policy):
    incompatible = replace(policy, dependency_specifiers=(("pandas", "<0"),))
    with pytest.raises(ContractError, match="incompatible"):
        preflight_capabilities(
            incompatible, RegistrySnapshot.empty(), AuthoritySnapshot.empty(), (), None
        )


def test_cross_authority_links_require_trusted_pairwise_crosswalk():
    entity = concept("C1", "Plantus example", EntityType.ORGANISM)
    links = (
        AuthorityLink("A1", "C1", "NCBI", "1", "v1", "test", "R", "registry-v000001"),
        AuthorityLink("A2", "C1", "GBIF", "9", "v1", "test", "R", "registry-v000001"),
    )
    registry = RegistrySnapshot(version="registry-v000001", canonical_entities=(entity,), authority_links=links)
    records = (
        AuthorityRecord("NCBI", "1", "Plantus example", EntityType.ORGANISM, ()),
        AuthorityRecord("GBIF", "9", "Plantus example", EntityType.ORGANISM, ()),
    )
    untrusted = AuthoritySnapshot("v1", "x", records, (), (("NCBI", "x"), ("GBIF", "x")))
    with pytest.raises(ContractError, match="untrusted cross-authority"):
        registry.validate(untrusted)
    trusted = replace(untrusted, trusted_crosswalks=(("NCBI", "1", "GBIF", "9"),))
    registry.validate(trusted)


def test_authority_backed_registry_requires_matching_pinned_snapshot():
    entity = concept("C1", "Plantus example", EntityType.ORGANISM)
    registry = RegistrySnapshot(
        version="registry-v000001",
        canonical_entities=(entity,),
        authority_links=(
            AuthorityLink("A1", "C1", "NCBI", "1", "v0", "test", "R", "registry-v000001"),
        ),
    )
    with pytest.raises(ContractError, match="pinned, non-empty"):
        registry.validate()
    authority = AuthoritySnapshot(
        "v1", "a" * 64,
        (AuthorityRecord("NCBI", "1", "Plantus example", EntityType.ORGANISM, ()),),
        (), (("NCBI", "public"),),
    )
    with pytest.raises(ContractError, match="uses snapshot v0, expected v1"):
        registry.validate(authority)


def test_reviewed_merge_retains_redirect_and_event():
    registry = RegistrySnapshot(
        version="registry-v000001",
        canonical_entities=(
            concept("C1", "nitrate", EntityType.CHEMICAL),
            concept("C2", "nitrate ion", EntityType.CHEMICAL),
        ),
    )
    change = merge_concepts(registry, ["C1", "C2"], "C1", "RUN-M", "reviewer", "same chemical")
    assert change.snapshot.active_id("C2") == "C1"
    assert change.events[0].event_type == EventType.MERGE
    assert change.snapshot.concept_by_id["C2"].lifecycle_status.value == "MERGED"


def test_cannot_link_to_merged_id_blocks_redirect_survivor(policy):
    registry = RegistrySnapshot(
        version="registry-v000001",
        canonical_entities=(
            concept("C1", "nitrate", EntityType.CHEMICAL),
            concept("C2", "nitrate ion", EntityType.CHEMICAL),
        ),
    )
    constrained = record_human_constraint(
        registry, "M1", "C2", ConstraintType.CANNOT_LINK, "RUN-C", "reviewer", "different"
    ).snapshot
    merged = merge_concepts(
        constrained, ["C1", "C2"], "C1", "RUN-M", "reviewer", "registry correction"
    ).snapshot
    execution = resolve_batch([mention("M1", "nitrate", EntityType.CHEMICAL)], merged, policy)
    assert execution.run.decisions[0].concept_id != "C1"


def test_split_refuses_unassigned_incident_relations():
    registry = RegistrySnapshot(
        version="registry-v000001",
        canonical_entities=(
            concept("C0", "spring", EntityType.ENVIRONMENTAL_FEATURE),
            concept("C1", "water", EntityType.ENVIRONMENTAL_FEATURE),
        ),
        entity_relations=(
            EntityRelation(
                "REL1", "C0", "RELATED_TO", "C1", "reviewed", True,
                "RUN-0", "registry-v000001",
            ),
        ),
    )
    with pytest.raises(ContractError, match="relation-reassignment plan"):
        split_concept(
            registry,
            "C0",
            [
                SplitPartition("SEED-A", "water spring", "[]"),
                SplitPartition("SEED-B", "spring season", "[]"),
            ],
            "RUN-S",
            "reviewer",
            "homonym split",
        )


def test_split_refuses_unassigned_active_constraints():
    registry = RegistrySnapshot(
        version="registry-v000001",
        canonical_entities=(concept("C0", "spring", EntityType.ENVIRONMENTAL_FEATURE),),
    )
    constrained = record_human_constraint(
        registry, "M1", "C0", ConstraintType.CANNOT_LINK, "RUN-C", "reviewer", "different"
    ).snapshot
    with pytest.raises(ContractError, match="constraint-reassignment plan"):
        split_concept(
            constrained,
            "C0",
            [
                SplitPartition("SEED-A", "water spring", "[]"),
                SplitPartition("SEED-B", "spring season", "[]"),
            ],
            "RUN-S",
            "reviewer",
            "homonym split",
        )


def test_reviewed_split_reassignment_and_human_constraint_preserve_history():
    base_concept = concept("C0", "spring", EntityType.ENVIRONMENTAL_FEATURE)
    instance = EntityInstance(
        instance_id="I1",
        paper_id="P1",
        context_id="CTX1",
        local_label="spring site",
        entity_type=EntityType.ENVIRONMENTAL_FEATURE,
        concept_id="C0",
        identity_qualifiers_json="[]",
        source_mention_ids_json='["SRC1"]',
        created_run_id="R0",
        updated_run_id="R0",
        registry_version="registry-v000001",
    )
    registry = RegistrySnapshot(
        version="registry-v000001", canonical_entities=(base_concept,), entity_instances=(instance,)
    )
    split = split_concept(
        registry,
        "C0",
        [
            SplitPartition("SEED-A", "water spring", "[]", instance_ids=("I1",)),
            SplitPartition("SEED-B", "spring season", "[]"),
        ],
        "RUN-S",
        "reviewer",
        "homonym split",
    )
    assert len(split.snapshot.active_concepts) == 2
    target = split.snapshot.instance_by_id["I1"].concept_id
    other = next(item.concept_id for item in split.snapshot.active_concepts if item.concept_id != target)
    reassigned = reassign_instance_concept(
        split.snapshot, "I1", other, "RUN-R", "reviewer", "correct target"
    )
    assert reassigned.snapshot.instance_by_id["I1"].concept_id == other
    constrained = record_human_constraint(
        reassigned.snapshot,
        "M1",
        other,
        ConstraintType.CANNOT_LINK,
        "RUN-C",
        "reviewer",
        "reviewed different",
    )
    assert constrained.snapshot.constraints[0].active
    assert {item.event_type for item in constrained.snapshot.events} >= {
        EventType.SPLIT,
        EventType.REASSIGN,
        EventType.REVIEW,
    }
    with pytest.raises(ContractError, match="active CANNOT_LINK"):
        record_human_constraint(
            constrained.snapshot,
            "M1",
            other,
            ConstraintType.MUST_LINK,
            "RUN-C2",
            "reviewer",
            "contradiction",
        )


def test_registry_and_run_parquet_round_trip(workspace_tmp, policy):
    execution = resolve_batch(
        [mention("M1", "nitrate", EntityType.CHEMICAL)], RegistrySnapshot.empty(), policy
    )
    committed = commit_resolution_run(execution, RegistrySnapshot.empty(), policy)
    assert committed.run.base_registry_version == "registry-v000000"
    assert committed.run.result_registry_version == committed.registry.version
    registry_path = workspace_tmp / committed.registry.version
    write_registry_snapshot(committed.registry, registry_path)
    loaded_registry = load_registry_snapshot(registry_path)
    assert loaded_registry.canonical_entities == committed.registry.canonical_entities

    run_path = workspace_tmp / "run"
    write_resolution_run(
        committed.run,
        run_path,
        conflicts=execution.conflicts,
        capability_manifest=execution.capability_manifest,
    )
    loaded_run = load_resolution_run(run_path)
    assert loaded_run.result_registry_version == committed.registry.version
    assert loaded_run.run_id == committed.run.run_id
    assert loaded_run.decisions == committed.run.decisions
    assert loaded_run.proposals == committed.run.proposals
    pointer = json.loads((run_path / "current-run.json").read_text(encoding="utf-8"))
    generation_path = run_path / pointer["directory"]
    assert load_resolution_run(generation_path).run_id == committed.run.run_id
    summary_path = generation_path / "run_summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    with pytest.raises(Exception, match="artifact hash mismatch"):
        load_resolution_run(run_path)


def test_resolution_run_publication_is_idempotent_and_fault_atomic(
    workspace_tmp, policy, monkeypatch
):
    from scripts.entity_resolution import io as resolution_io

    execution = resolve_batch(
        [mention("M1", "nitrate", EntityType.CHEMICAL)], RegistrySnapshot.empty(), policy
    )
    run_path = workspace_tmp / "atomic-run"
    write_resolution_run(execution.run, run_path)
    first_pointer = (run_path / "current-run.json").read_bytes()
    first_directories = sorted(item.name for item in (run_path / "run-generations").iterdir())

    write_resolution_run(execution.run, run_path)
    assert (run_path / "current-run.json").read_bytes() == first_pointer
    assert sorted(item.name for item in (run_path / "run-generations").iterdir()) == first_directories

    original = resolution_io.atomic_write_parquet
    calls = 0

    def fail_during_stage(frame, path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected staged publication failure")
        return original(frame, path)

    monkeypatch.setattr(resolution_io, "atomic_write_parquet", fail_during_stage)
    with pytest.raises(OSError, match="injected"):
        write_resolution_run(execution.run, run_path)
    assert (run_path / "current-run.json").read_bytes() == first_pointer
    assert not any(item.name.startswith(".staging-") for item in (run_path / "run-generations").iterdir())
    assert load_resolution_run(run_path).run_id == execution.run.run_id


def test_persisted_resolution_rows_expose_nonblocking_review_fields(workspace_tmp, policy):
    execution = resolve_batch(
        [
            mention("M1", "Niger", EntityType.PLACE, paper_id="P1"),
            mention("M2", "Niger", EntityType.PLACE, paper_id="P2"),
        ],
        RegistrySnapshot.empty(),
        policy,
    )
    run_path = workspace_tmp / "review-run"
    write_resolution_run(execution.run, run_path)
    pointer = json.loads((run_path / "current-run.json").read_text(encoding="utf-8"))
    rows = pd.read_parquet(run_path / pointer["directory"] / "mention_resolutions.parquet")
    assert {"review_needed", "review_flags_json", "review_priority"} <= set(rows.columns)
    assert rows["review_needed"].all()
    assert rows["review_flags_json"].str.contains("HOMONYM_PRONE_EXACT_NAME").all()
    assert (rows["review_priority"] > 0).all()
    assert set(rows["decision_status"]) == {"NEW_CONCEPT_PROPOSED"}


def test_resolution_manifest_carries_extraction_generation(workspace_tmp, policy):
    execution = resolve_batch(
        [mention("M1", "nitrate", EntityType.CHEMICAL)], RegistrySnapshot.empty(), policy
    )
    run_path = workspace_tmp / "provenance-run"
    capabilities = {
        **execution.capability_manifest,
        "extraction_input": {
            "generation_id": "a" * 24,
            "source_fingerprint": "b" * 64,
            "settings_hash": "c" * 64,
            "schema_version": policy.schema_version,
            "prompt_version": policy.extraction_prompt_version,
        },
    }
    write_resolution_run(execution.run, run_path, capability_manifest=capabilities)
    pointer = json.loads((run_path / "current-run.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (run_path / pointer["directory"] / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["extraction_generation_id"] == "a" * 24
    assert manifest["capabilities"]["extraction_input"]["generation_id"] == "a" * 24
