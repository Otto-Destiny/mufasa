import json
from dataclasses import replace

import pytest

from conftest import concept, mention, snapshot
from scripts.entity_resolution.authorities import AuthorityRecord, AuthoritySnapshot
from scripts.entity_resolution.contracts import (
    AliasTrust,
    AlignmentStatus,
    AssertionStatus,
    AuthorityHint,
    ContractError,
    DecisionMethod,
    DecisionStatus,
    EntityType,
    IdentityScope,
    Qualifier,
)
from scripts.entity_resolution.pipeline import commit_resolution_run, resolve_batch
from scripts.entity_resolution.audit import build_review_rows, decision_memo_key
from scripts.entity_resolution.registry import RegistrySnapshot


def test_assertion_status_is_not_used_as_resolution_workflow_status():
    assert {item.value for item in AssertionStatus} == {"REPORTED", "DERIVED"}


def test_memo_key_changes_when_source_safety_flags_change():
    clean = mention("M1", "nitrate", EntityType.CHEMICAL)
    warned = replace(clean, source_flags=("OCR_WARNING",))
    args = ("c" * 64, "policy-v1", "p" * 64, "registry-v1", "r" * 64, "a" * 64, "code-v1")
    assert decision_memo_key(clean, *args) != decision_memo_key(warned, *args)


def test_cold_start_exact_group_is_deterministic_and_commits_one_concept(policy):
    left = mention("M2", "nitrate", EntityType.CHEMICAL, paper_id="P2")
    right = mention("M1", "Nitrate", EntityType.CHEMICAL, paper_id="P1")
    first = resolve_batch([left, right], RegistrySnapshot.empty(), policy)
    second = resolve_batch([right, left], RegistrySnapshot.empty(), policy)
    assert first.run.run_id == second.run.run_id
    assert first.run.proposals == second.run.proposals
    assert len(first.run.proposals) == 1
    committed = commit_resolution_run(first, RegistrySnapshot.empty(), policy)
    assert len(committed.registry.canonical_entities) == 1
    assert {item.status for item in committed.run.decisions} == {DecisionStatus.MATCHED}
    assert len({item.concept_id for item in committed.run.decisions}) == 1


def test_extraction_aliases_persist_into_the_registry_with_their_provenance(policy):
    """A name the model supplied must become a registry alias, or it is used
    once inside its own batch and then lost."""
    value = mention(
        "M1", "onugbu", EntityType.ORGANISM,
        aliases=("Vernonia amygdalina", "bitter leaf"),
    )
    value = replace(value, aliases_json=json.dumps([
        {"text": "Vernonia amygdalina", "kind": "SCIENTIFIC", "language": "la",
         "stated_in_paper": True},
        {"text": "bitter leaf", "kind": "COMMON_ENGLISH", "language": "en",
         "stated_in_paper": False},
    ]))
    committed = commit_resolution_run(
        resolve_batch([value], RegistrySnapshot.empty(), policy),
        RegistrySnapshot.empty(), policy)

    by_text = {item.alias_text: item for item in committed.registry.canonical_aliases}
    assert "bitter leaf" in by_text, sorted(by_text)
    assert "Vernonia amygdalina" in by_text

    scientific = by_text["Vernonia amygdalina"]
    assert scientific.trust_level is AliasTrust.MODEL_SUPPLIED
    assert scientific.alias_kind == "SCIENTIFIC"
    assert scientific.stated_in_paper is True
    assert scientific.language == "la"
    assert by_text["bitter leaf"].stated_in_paper is False


def test_a_supplied_alias_matches_an_existing_registry_concept(policy):
    """The registry is queried with every name a mention carries, not only its
    own. Here the arriving mention's own text is unknown to the registry and
    only one of its aliases is a match, which is exactly the case that used to
    re-propose a duplicate concept on every run.

    ORGANISM does not auto-merge on a bare normalised name - that is the homonym
    guard - so this also confirms a trusted alias is treated as the stronger
    evidence it is.
    """
    seed = replace(
        mention("SEED", "Vernonia amygdalina", EntityType.ORGANISM, paper_id="P0",
                aliases=("onugbu",)),
        aliases_json=json.dumps([{"text": "onugbu", "kind": "VERNACULAR",
                                  "language": "ig", "stated_in_paper": True}]),
    )
    registry = commit_resolution_run(
        resolve_batch([seed], RegistrySnapshot.empty(), policy),
        RegistrySnapshot.empty(), policy).registry
    existing = {item.concept_id for item in registry.canonical_entities}
    assert len(existing) == 1

    # "bitter leaf" is not in the registry under any label; only its alias is.
    arriving = mention("M9", "bitter leaf", EntityType.ORGANISM, paper_id="P9",
                       aliases=("onugbu",))
    decision = resolve_batch([arriving], registry, policy).run.decisions[0]

    assert decision.status is DecisionStatus.MATCHED, decision.reason_codes
    assert decision.concept_id in existing


def test_agreeing_papers_merge_instead_of_being_punished_for_agreeing(policy):
    """Two papers naming the same material must produce one concept.

    While these types required review for a bare name match, a group of
    agreeing mentions produced NO concept at all, whereas a single isolated
    mention self-seeded one - so the more papers agreed, the less likely the
    concept existed. That is the opposite of what the corpus is for.
    """
    for entity_type, text in ((EntityType.MATERIAL, "rice husk ash"),
                              (EntityType.ORGANISM, "Vernonia amygdalina"),
                              (EntityType.ENVIRONMENTAL_FEATURE, "groundwater")):
        values = [mention("M1", text, entity_type, paper_id="P1"),
                  mention("M2", text, entity_type, paper_id="P2")]
        execution = resolve_batch(values, RegistrySnapshot.empty(), policy)
        statuses = {item.status for item in execution.run.decisions}
        # Against an empty registry there is nothing to match to, so the right
        # outcome is one new concept covering both - not review, and not two.
        assert statuses == {DecisionStatus.NEW_CONCEPT_PROPOSED}, (entity_type, statuses)
        assert len(execution.run.proposals) == 1, entity_type
        assert execution.run.proposals[0].member_mention_ids == ("M1", "M2")
        committed = commit_resolution_run(execution, RegistrySnapshot.empty(), policy)
        assert len(committed.registry.canonical_entities) == 1, entity_type


@pytest.mark.parametrize(
    "entity_type",
    [entity_type for entity_type in EntityType if entity_type is not EntityType.OTHER],
)
def test_every_controlled_canonical_type_connects_on_compatible_exact_name(
    policy, entity_type
):
    """No controlled canonical type may silently fall back to pre-merge review."""

    label = f"shared {entity_type.value.lower()} concept"
    values = [
        mention("M1", label, entity_type, paper_id="P1"),
        mention("M2", label, entity_type, paper_id="P2"),
    ]
    execution = resolve_batch(values, RegistrySnapshot.empty(), policy)
    assert {item.status for item in execution.run.decisions} == {
        DecisionStatus.NEW_CONCEPT_PROPOSED
    }
    assert len(execution.run.proposals) == 1
    committed = commit_resolution_run(execution, RegistrySnapshot.empty(), policy)
    assert len(committed.registry.active_concepts) == 1


def test_identity_bearing_qualifiers_still_separate_same_named_features(policy):
    """Auto-merging on a bare name does not weaken the qualifier guard: the
    river and the state share a name and must stay apart."""
    values = [
        mention("M1", "Niger", EntityType.ENVIRONMENTAL_FEATURE, paper_id="P1",
                qualifiers=(Qualifier("FEATURE_CLASS", "river"),)),
        mention("M2", "Niger", EntityType.ENVIRONMENTAL_FEATURE, paper_id="P2",
                qualifiers=(Qualifier("FEATURE_CLASS", "basin"),)),
    ]
    execution = resolve_batch(values, RegistrySnapshot.empty(), policy)
    committed = commit_resolution_run(execution, RegistrySnapshot.empty(), policy)
    assert len(committed.registry.canonical_entities) == 2


def test_missing_identity_qualifier_connects_and_preserves_known_value(policy):
    values = [
        mention("M1", "Niger", EntityType.ENVIRONMENTAL_FEATURE, paper_id="P1"),
        mention(
            "M2",
            "Niger",
            EntityType.ENVIRONMENTAL_FEATURE,
            paper_id="P2",
            qualifiers=(Qualifier("FEATURE_CLASS", "river"),),
        ),
    ]
    execution = resolve_batch(values, RegistrySnapshot.empty(), policy)
    assert len(execution.run.proposals) == 1
    assert any(
        "AUTO_MERGE_REVIEW:MISSING_IDENTITY_QUALIFIER:FEATURE_CLASS" in item.reason_codes
        for item in execution.run.decisions
    )
    committed = commit_resolution_run(execution, RegistrySnapshot.empty(), policy)
    assert len(committed.registry.active_concepts) == 1
    assert json.loads(committed.registry.active_concepts[0].identity_qualifiers_json) == [
        {"kind": "FEATURE_CLASS", "value_text": "river"}
    ]


def test_supplied_alias_hitting_preferred_label_is_trusted_alias(policy):
    registry = snapshot(
        concept("CON-VERNONIA", "Vernonia amygdalina", EntityType.ORGANISM)
    )
    arriving = mention(
        "M1", "onugbu", EntityType.ORGANISM, aliases=("Vernonia amygdalina",)
    )
    decision = resolve_batch([arriving], registry, policy).run.decisions[0]
    assert decision.status is DecisionStatus.MATCHED
    assert decision.method is DecisionMethod.TRUSTED_ALIAS
    assert decision.concept_id == "CON-VERNONIA"


def test_trusted_aliases_reconcile_duplicate_existing_concepts_and_learn_names(policy):
    registry = snapshot(
        concept("CON-A", "Vernonia amygdalina", EntityType.ORGANISM),
        concept("CON-B", "bitter leaf", EntityType.ORGANISM),
    )
    arriving = mention(
        "M1",
        "onugbu",
        EntityType.ORGANISM,
        aliases=("Vernonia amygdalina", "bitter leaf"),
    )
    execution = resolve_batch([arriving], registry, policy)
    assert len(execution.merge_plans) == 1
    assert execution.merge_plans[0].concept_ids == ("CON-A", "CON-B")
    assert execution.run.decisions[0].status is DecisionStatus.MATCHED
    committed = commit_resolution_run(execution, registry, policy)
    assert len(committed.registry.active_concepts) == 1
    assert committed.registry.active_concepts[0].concept_id == "CON-A"
    assert committed.registry.active_id("CON-B") == "CON-A"
    learned = [
        item
        for item in committed.registry.canonical_aliases
        if item.concept_id == "CON-A" and item.alias_text == "onugbu"
    ]
    assert learned
    provenance = json.loads(learned[0].provenance_json)
    assert provenance["records"][0]["mention_id"] == "M1"
    later = mention("M2", "onugbu", EntityType.ORGANISM, paper_id="P2")
    later_decision = resolve_batch([later], committed.registry, policy).run.decisions[0]
    assert later_decision.status is DecisionStatus.MATCHED
    assert later_decision.concept_id == "CON-A"


def test_alias_records_are_strictly_validated():
    base = mention("M1", "onugbu", EntityType.ORGANISM)
    with pytest.raises(ContractError, match="must contain exactly"):
        replace(base, aliases=("bitter leaf",), aliases_json='["bitter leaf"]')
    with pytest.raises(ContractError, match="kind"):
        replace(
            base,
            aliases=("bitter leaf",),
            aliases_json=json.dumps(
                [
                    {
                        "text": "bitter leaf",
                        "kind": "MADE_UP",
                        "language": "en",
                        "stated_in_paper": False,
                    }
                ]
            ),
        )


def test_homonym_prone_exact_place_group_connects_and_is_audited(policy):
    values = [
        mention("M1", "Niger", EntityType.PLACE, paper_id="P1"),
        mention("M2", "Niger", EntityType.PLACE, paper_id="P2"),
    ]
    execution = resolve_batch(values, RegistrySnapshot.empty(), policy)
    assert len(execution.run.proposals) == 1
    assert {item.status for item in execution.run.decisions} == {DecisionStatus.NEW_CONCEPT_PROPOSED}
    rows = build_review_rows(execution.run)
    assert len(rows) == 1
    assert rows[0]["review_mode"] == "POST_MERGE_AUDIT"
    assert rows[0]["blocking"] is False


def test_single_clean_place_can_seed_without_claiming_an_exact_cross_paper_merge(policy):
    execution = resolve_batch([mention("M1", "Minna", EntityType.PLACE)], RegistrySnapshot.empty(), policy)
    assert len(execution.run.proposals) == 1
    assert execution.run.proposals[0].member_mention_ids == ("M1",)


def test_same_label_instances_in_one_context_do_not_merge_without_same_source_identity(policy):
    values = [
        mention("M1", "water sample", EntityType.SAMPLE_SPECIMEN, scope=IdentityScope.STUDY_INSTANCE, source_mention_id="SRC-A"),
        mention("M2", "water sample", EntityType.SAMPLE_SPECIMEN, scope=IdentityScope.STUDY_INSTANCE, source_mention_id="SRC-B"),
    ]
    execution = resolve_batch(values, RegistrySnapshot.empty(), policy)
    instance_proposals = [item for item in execution.run.proposals if item.proposal_kind.value == "INSTANCE"]
    assert len(instance_proposals) == 2
    committed = commit_resolution_run(execution, RegistrySnapshot.empty(), policy)
    assert len(committed.registry.entity_instances) == 2


def test_instance_atoms_from_same_source_group_stay_together(policy):
    values = [
        mention("M1", "water sample", EntityType.SAMPLE_SPECIMEN, scope=IdentityScope.STUDY_INSTANCE, source_mention_id="SRC-A"),
        mention("M2", "water sample", EntityType.SAMPLE_SPECIMEN, scope=IdentityScope.STUDY_INSTANCE, source_mention_id="SRC-A"),
    ]
    execution = resolve_batch(values, RegistrySnapshot.empty(), policy)
    assert len(execution.run.proposals) == 1
    assert execution.run.proposals[0].member_mention_ids == ("M1", "M2")


def test_review_dedup_never_groups_unrelated_local_instances(policy):
    values = [
        mention(
            "M1", "water sample", EntityType.SAMPLE_SPECIMEN,
            scope=IdentityScope.STUDY_INSTANCE, source_mention_id="SRC-A",
            alignment=AlignmentStatus.EXACT_AMBIGUOUS,
        ),
        mention(
            "M2", "water sample", EntityType.SAMPLE_SPECIMEN,
            scope=IdentityScope.STUDY_INSTANCE, source_mention_id="SRC-B",
            alignment=AlignmentStatus.EXACT_AMBIGUOUS,
        ),
    ]
    run = resolve_batch(values, RegistrySnapshot.empty(), policy).run
    rows = build_review_rows(run)
    assert len(rows) == 2
    assert {row["propagation_policy"] for row in rows} == {"LOCKED_SOURCE_MENTION_GROUP"}
    assert not any(row["propagation_approved"] for row in rows)


def test_unmodeled_qualifier_is_preserved_but_review_only(policy):
    value = mention(
        "M1",
        "groundwater",
        EntityType.ENVIRONMENTAL_FEATURE,
        qualifiers=(Qualifier("UNMODELED_QUALIFIER", "local hydrogeologic class"),),
    )
    execution = resolve_batch([value], RegistrySnapshot.empty(), policy)
    assert execution.run.decisions[0].status == DecisionStatus.REVIEW_REQUIRED
    assert "UNMODELED_QUALIFIER_REQUIRES_REVIEW" in execution.run.decisions[0].reason_codes


def test_ambiguous_source_occurrence_connects_but_is_audited(policy):
    registry = snapshot(concept("CON-GW", "groundwater", EntityType.ENVIRONMENTAL_FEATURE))
    value = mention(
        "M1", "groundwater", EntityType.ENVIRONMENTAL_FEATURE,
        alignment=AlignmentStatus.EXACT_AMBIGUOUS,
    )
    execution = resolve_batch([value], registry, policy)
    assert execution.run.decisions[0].status == DecisionStatus.MATCHED
    rows = build_review_rows(execution.run)
    assert rows[0]["review_mode"] == "POST_MERGE_AUDIT"
    assert rows[0]["blocking"] is False


def test_exact_accepted_match_keeps_compact_decision_not_candidate_row(policy):
    registry = snapshot(concept("CON-N", "nitrate", EntityType.CHEMICAL))
    execution = resolve_batch(
        [mention("M1", "nitrate", EntityType.CHEMICAL)], registry, policy
    )
    assert execution.run.decisions[0].status == DecisionStatus.MATCHED
    assert execution.run.generated_candidate_count == 1
    assert execution.run.candidates == ()


def test_approximate_candidate_seeds_separate_concept_for_later_audit(policy):
    registry = snapshot(concept("CON-1", "nitrate concentration", EntityType.PROPERTY_METRIC))
    value = mention("M1", "nitrate concentrations", EntityType.PROPERTY_METRIC)
    execution = resolve_batch([value], registry, policy)
    assert execution.run.decisions[0].status == DecisionStatus.NEW_CONCEPT_PROPOSED
    assert "AUTO_MERGE_REVIEW:POTENTIAL_DUPLICATE_CANDIDATE" in execution.run.decisions[0].reason_codes
    assert execution.run.candidates


def test_incremental_explicit_identity_conflict_seeds_separate_concept(policy):
    river = mention(
        "M1",
        "Niger",
        EntityType.ENVIRONMENTAL_FEATURE,
        qualifiers=(Qualifier("FEATURE_CLASS", "river"),),
    )
    seeded = commit_resolution_run(
        resolve_batch([river], RegistrySnapshot.empty(), policy),
        RegistrySnapshot.empty(),
        policy,
    )
    basin = mention(
        "M2",
        "Niger",
        EntityType.ENVIRONMENTAL_FEATURE,
        paper_id="P2",
        qualifiers=(Qualifier("FEATURE_CLASS", "basin"),),
    )
    execution = resolve_batch([basin], seeded.registry, policy)
    assert execution.run.decisions[0].status is DecisionStatus.NEW_CONCEPT_PROPOSED
    assert "AUTO_MERGE_REVIEW:SEPARATE_IDENTITY_HARD_CONFLICT" in execution.run.decisions[0].reason_codes
    committed = commit_resolution_run(execution, seeded.registry, policy)
    assert len(committed.registry.active_concepts) == 2


def test_lexical_typo_recall_is_deterministic_and_scoring_is_bounded(policy, monkeypatch):
    from scripts.entity_resolution import matching

    concepts = [concept("CON-GW", "groundwater quality", EntityType.ENVIRONMENTAL_FEATURE)]
    concepts.extend(
        concept(f"CON-{index:05d}", f"soil indicator {index:05d}", EntityType.ENVIRONMENTAL_FEATURE)
        for index in range(2_500)
    )
    registry = snapshot(*concepts)
    value = mention("M1", "groundwatre quality", EntityType.ENVIRONMENTAL_FEATURE)
    calls = 0
    original = matching.fuzz.WRatio

    def counted(left, right):
        nonlocal calls
        calls += 1
        return original(left, right)

    monkeypatch.setattr(matching.fuzz, "WRatio", counted)
    first = resolve_batch([value], registry, policy)
    first_ids = [item.target_id for item in first.run.candidates]
    first_call_count = calls
    calls = 0
    second = resolve_batch([value], registry, policy)
    second_ids = [item.target_id for item in second.run.candidates]

    assert "CON-GW" in first_ids
    assert first_ids == second_ids
    maximum_scores = (
        policy.lexical_blocking.max_candidate_pool
        * policy.lexical_blocking.max_labels_per_target
    )
    assert first_call_count <= maximum_scores
    assert calls <= maximum_scores
    assert first.run.generated_candidate_count <= policy.top_k_candidates


def test_crosswalked_multi_authority_hints_can_resolve_one_concept(policy):
    records = (
        AuthorityRecord("NCBI", "1", "Plantus example", EntityType.ORGANISM, ()),
        AuthorityRecord("GBIF", "9", "Plantus example", EntityType.ORGANISM, ()),
    )
    authority = AuthoritySnapshot(
        "auth-v1", "h" * 64, records, (("NCBI", "1", "GBIF", "9"),), (("NCBI", "public"), ("GBIF", "public")),
    )
    from scripts.entity_resolution.contracts import AuthorityLink
    registry = RegistrySnapshot(
        version="registry-v000001",
        canonical_entities=(concept("CON-1", "Plantus example", EntityType.ORGANISM),),
        authority_links=(
            AuthorityLink("A1", "CON-1", "NCBI", "1", "auth-v1", "test", "R", "registry-v000001"),
            AuthorityLink("A2", "CON-1", "GBIF", "9", "auth-v1", "test", "R", "registry-v000001"),
        ),
    )
    value = mention("M1", "local plant", EntityType.ORGANISM)
    hints = [
        AuthorityHint("M1", "NCBI", "1", "auth-v1", "test"),
        AuthorityHint("M1", "GBIF", "9", "auth-v1", "test"),
    ]
    execution = resolve_batch([value], registry, policy, authority_snapshot=authority, authority_hints=hints)
    assert execution.run.decisions[0].status == DecisionStatus.MATCHED
    assert execution.run.decisions[0].concept_id == "CON-1"


def test_untrusted_multi_authority_hints_require_review(policy):
    records = (
        AuthorityRecord("NCBI", "1", "Plantus example", EntityType.ORGANISM, ()),
        AuthorityRecord("GBIF", "9", "Other plant", EntityType.ORGANISM, ()),
    )
    authority = AuthoritySnapshot(
        "auth-v1", "h" * 64, records, (), (("NCBI", "public"), ("GBIF", "public")),
    )
    value = mention("M1", "local plant", EntityType.ORGANISM)
    hints = [
        AuthorityHint("M1", "NCBI", "1", "auth-v1", "test"),
        AuthorityHint("M1", "GBIF", "9", "auth-v1", "test"),
    ]
    execution = resolve_batch([value], RegistrySnapshot.empty(), policy, authority_snapshot=authority, authority_hints=hints)
    assert execution.run.decisions[0].status == DecisionStatus.REVIEW_REQUIRED
    assert "CONFLICTING_AUTHORITY_HINTS" in execution.run.decisions[0].reason_codes
