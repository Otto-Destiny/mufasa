from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


MODULE_PATH = (
    Path(__file__).parents[1]
    / "01-data-engineering"
    / "data-extraction"
    / "mufasa_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("mufasa_dataset_router", MODULE_PATH)
funnel = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(funnel)


def row(**updates):
    values = {
        "pair_id": "P1:factual:f1",
        "paper_id": "P1",
        "pair_type": "FACTUAL",
        "pair_kind": "",
        "question": "What was the prevalence in Sokoto?",
        "answer": "The prevalence in Sokoto was 17%.",
        "reasoning": "",
        "chosen": "",
        "rejected": "",
        "positive_quote": "",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def write_paper(folder: Path, paper_id: str, body: str) -> None:
    (folder / f"{paper_id}.md").write_text(
        f"---\npaper_id: {paper_id}\n---\n\n"
        "<!-- MUFASA_PDF_PAGE: 1 -->\n\n## PDF page 1\n\n" + body,
        encoding="utf-8",
    )


def test_evidence_bundle_preserves_distinct_pages_and_deduplicates_exact_copy():
    spans = pd.DataFrame(
        [
            {"owner_kind": "TRAINING", "owner_id": "x", "page": 2, "quote": "A"},
            {"owner_kind": "TRAINING", "owner_id": "x", "page": 5, "quote": "B"},
            {"owner_kind": "TRAINING", "owner_id": "x", "page": 2, "quote": "A"},
            {"owner_kind": "OBSERVATION", "owner_id": "x", "page": 6, "quote": "C"},
        ]
    )
    bundle = funnel.evidence_bundles(spans)["x"]
    assert [(item["page"], item["quote"]) for item in bundle] == [(2, "A"), (5, "B")]
    canonical = funnel.dedupe_evidence([
        {"paper_id": "P1", "page": 1, "quote": "same"},
        {"paper_id": "P1", "page": 1.0, "quote": "same"},
    ])
    assert len(canonical) == 1


def test_exact_numeric_boundaries_prevent_32_matching_32_point_7(tmp_path):
    write_paper(tmp_path, "P1", "The recorded prevalence was 32.7% in the survey.")
    assert [item["value"] for item in funnel.quantitative_mentions("32.")] == [32]
    assert funnel.reground("32.", "P1", tmp_path, {}) is None
    assert not funnel.support_report(
        "32.", "The recorded prevalence was 32.7% in the survey.",
        "How many participants were recorded?",
    )["supported"]


def test_numeric_formatting_and_units_are_checked():
    assert funnel.support_report(
        "The sample included 1,000 participants.",
        "A total of 1000 participants were enrolled.",
        "How many participants were enrolled?",
    )["supported"]
    bad = funnel.support_report(
        "The dose was 20 kg.", "The dose was 20 mg.", "What dose was used?"
    )
    assert not bad["supported"]
    assert bad["unit_mismatches"]
    percent = funnel.support_report(
        "Prevalence was 17%.", "Prevalence was 17 mg.", "What was prevalence?"
    )
    assert not percent["supported"]
    assert percent["unit_mismatches"]
    table = funnel.support_report(
        "MLE was 156.6 mm; Bayesian MCMC was 168.6 mm.",
        "|Return period (Years)|MLE (mm)|Bayesian MCMC (mm)|\n|200|156.6|168.6|",
        "What were the MLE and Bayesian MCMC 200-year return levels?",
    )
    assert table["supported"]


def test_modifier_anchor_accepts_reordered_scientific_shorthand():
    report = funnel.support_report(
        "32.",
        "Out of the women that tested positive to HBsAg, 32 opted for breastfeeding.",
        "Out of the HBsAg-positive women, how many opted for breastfeeding?",
    )
    assert report["supported"], report


def test_router_widens_a_too_small_quote_inside_same_paper(tmp_path):
    write_paper(
        tmp_path,
        "P1",
        "The survey was conducted in Sokoto.\n\n"
        "Among sampled households in Sokoto, the measured prevalence was 17%.",
    )
    initial = [{"paper_id": "P1", "page": 1, "quote": "The survey was conducted in Sokoto."}]
    result = funnel.route_pair(row(), initial, tmp_path, {})
    assert result["route"] == "OPEN_WIDENED"
    assert result["paper_verified"]
    assert any("17%" in span["quote"] for span in result["bundle"])


def test_router_quarantines_claim_absent_from_paper(tmp_path):
    write_paper(tmp_path, "P1", "The survey was conducted in Sokoto among households.")
    result = funnel.route_pair(row(), [], tmp_path, {})
    assert result["route"] == "QUARANTINE_UNVERIFIED"
    assert not result["paper_verified"]


def test_sentence_level_check_catches_invented_mechanism():
    report = funnel.support_report(
        "The intervention reduced disease because it inhibited viral replication.",
        "The intervention reduced disease incidence.",
        "What did the intervention do?",
    )
    assert not report["supported"]
    assert "causal relation absent from evidence" in report["direction_conflicts"]


def test_negation_and_direction_are_not_interchangeable():
    report = funnel.support_report(
        "The treatment increased yield.",
        "The treatment decreased yield.",
        "How did treatment affect yield?",
    )
    assert not report["supported"]
    assert report["direction_conflicts"]
    positive = funnel.support_report(
        "The treatment increased yield.",
        "The treatment did not increase yield.",
        "How did treatment affect yield?",
    )
    assert not positive["supported"]
    assert "evidence negates the positive target" in positive["direction_conflicts"]


def test_number_is_bound_to_the_named_place_in_the_question():
    report = funnel.support_report(
        "5%.", "The prevalence in Kano was 5%.",
        "What was the prevalence in Sokoto?",
    )
    assert not report["supported"]
    assert "Sokoto" in report["missing_question_anchors"]
    substring = funnel.support_report(
        "5%.", "The prevalence in Nigeria was 5%.",
        "What was the prevalence in Niger?",
    )
    assert not substring["supported"]
    assert "Niger" in substring["missing_question_anchors"]
    wrong_row = funnel.support_report(
        "5.0%.",
        "|Site|Prevalence|\n|---|---|\n|Dokan Tofa|2.0%|\n|Kano|5.0%|",
        "What was the prevalence in Dokan Tofa?",
    )
    assert not wrong_row["supported"]
    assert "Dokan Tofa" in wrong_row["unbound_question_anchors"]
    swapped = funnel.support_report(
        "Sokoto: 5.0%; Kano: 2.0%.",
        "|Site|Prevalence|\n|---|---|\n|Sokoto|2.0%|\n|Kano|5.0%|",
        "What were the prevalence values in Sokoto and Kano?",
    )
    assert not swapped["supported"]
    wrong_column = funnel.support_report(
        "The dose in Sokoto was 20 mg.",
        "|Site|Dose (mg)|Prevalence (%)|\n|---|---|---|\n|Sokoto|10|20|",
        "What was the dose in Sokoto?",
    )
    assert not wrong_column["supported"]


def test_bag_of_words_does_not_invent_a_relation():
    report = funnel.support_report(
        "Rainfall increased maize yield.",
        "Rainfall was measured. Maize yield increased after fertilizer.",
        "What effect did rainfall have on maize yield?",
    )
    assert not report["supported"]


def test_sentence_starter_is_not_misread_as_a_proper_name():
    report = funnel.support_report(
        "These Dar-Zarrouck parameters combine resistivity and thickness.",
        "Dar-Zarrouck parameters combine layer resistivity and thickness.",
        "What do the parameters combine?",
    )
    assert "These Dar-Zarrouck" not in report["missing_proper_terms"]


def test_explicit_arithmetic_lane_recomputes_difference():
    report = funnel.support_report(
        "The difference was 8 percentage points.",
        "Prevalence increased from 12% at baseline to 20% at follow-up.",
        "What was the difference in prevalence, in percentage points?",
    )
    assert report["supported"]
    assert report["arithmetic"][0]["operation"] == "subtract"
    wrong_units = funnel.support_report(
        "The difference was 8 percentage points.",
        "Participants were aged 12 years and 20 years.",
        "What was the difference in prevalence, in percentage points?",
    )
    assert not wrong_units["supported"]
    wrong_metrics = funnel.support_report(
        "The difference in prevalence was 8 percentage points.",
        "Rainfall was 12% while crop yield was 20%.",
        "What was the difference in prevalence, in percentage points?",
    )
    assert not wrong_metrics["supported"]


def test_conflicting_duplicate_ids_are_all_removed():
    pairs = pd.DataFrame(
        [
            {"pair_id": "x", "paper_id": "P1", "answer": "A"},
            {"pair_id": "x", "paper_id": "P1", "answer": "B is longer"},
            {"pair_id": "y", "paper_id": "P2", "answer": "C"},
        ]
    )
    kept, dropped = funnel.resolve_duplicates(pairs)
    assert kept.pair_id.tolist() == ["y"]
    assert dropped.pair_id.tolist() == ["x", "x"]


def test_verified_factual_numeric_knowledge_can_be_closed_book():
    candidate = row(answer="The measured prevalence was 17%.")
    ok, why = funnel.verified_closed_ready(
        candidate,
        "a cross-sectional survey, reported in 'Prevalence in Sokoto'",
        paper_verified=True,
    )
    assert ok, why
    assert funnel.curriculum_mode(candidate.pair_id) == funnel.curriculum_mode(candidate.pair_id)
    leaked = row(answer="92 (19.3%).")
    ok, why = funnel.verified_closed_ready(
        leaked,
        "a study of tumours with 92 cases (19.3%), reported in 'Tumour survey'",
        paper_verified=True,
    )
    assert not ok
    assert why == "descriptor reveals the target"


def test_wrong_script_gate_checks_the_target_too():
    candidate = row(answer="这是中文回答")
    reasons = funnel.discard_reasons(candidate, {"P1"}, set(), set())
    assert "wrong language" in reasons
    masked = row(
        question="结果是什么？",
        answer="This deliberately long English answer must not hide the Chinese question.",
    )
    assert "wrong language" in funnel.discard_reasons(masked, {"P1"}, set(), set())


def test_corrupt_and_placeholder_targets_are_not_trainable():
    corrupt = row(answer="The value was 17\ufffd%.")
    assert "text encoding corruption" in funnel.discard_reasons(
        corrupt, {"P1"}, set(), set(),
    )
    placeholder = row(answer="24.00 ... 31.00 ... 28.00")
    assert "placeholder ellipsis in target" in funnel.discard_reasons(
        placeholder, {"P1"}, set(), set(),
    )


def test_objective_contract_rejects_identical_or_incomplete_pairs():
    preference = SimpleNamespace(
        **{
            **vars(row()), "pair_type": "PREFERENCE", "chosen": "Same answer.",
            "rejected": "  same   answer. ",
        },
    )
    assert "chosen and rejected are identical" in funnel.discard_reasons(
        preference, {"P1"}, set(), set(),
    )
    reranker = SimpleNamespace(
        **{
            **vars(row()), "pair_type": "RERANKER", "positive_quote": "Useful passage.",
            "hard_negative_quote": "", "answer": "",
        },
    )
    assert "blank hard negative quote" in funnel.discard_reasons(
        reranker, {"P1"}, set(), set(),
    )


def test_provenance_rendering_is_opt_in_and_does_not_change_claim_text():
    candidate = row(answer="The measured prevalence was 17%.")
    evidence = [{"quote": "The measured prevalence was 17%."}]
    assert funnel.assistant_turn(candidate) == "The measured prevalence was 17%."
    legacy = funnel.render_open(candidate, evidence)
    assert legacy["messages"][1]["content"] == funnel.assistant_turn(candidate)

    rendered = funnel.render_open(
        candidate,
        evidence,
        citation_label="Njoku et al. (2022)",
        study_basis="design: cross-sectional survey; location: Sokoto State, Nigeria",
        verification_tier="VERIFIED",
    )["messages"][1]["content"]
    assert rendered.startswith("The measured prevalence was 17%.")
    assert "Provenance: PROVIDED_EVIDENCE — Evidence 1" in rendered
    assert "Citation: (Njoku et al., 2022)" in rendered
    assert "Study basis: design: cross-sectional survey" in rendered


def test_semantic_study_basis_ignores_title_and_identifiers():
    basis = funnel.semantic_study_basis({
        "paper_id": "P-SECRET",
        "profile": {
            "title": "Exact Formal Title That Must Not Be Copied",
            "discipline": "public health",
        },
        "selected_contexts": [{
            "label": "household water quality",
            "study_design": "cross-sectional survey",
            "population_text": "rural households",
            "period_text": "2021 rainy season",
        }],
        "african_innovation": {"place": "Sokoto State, Nigeria"},
        "doi": "10.1000/not-for-recall",
    })
    assert basis == (
        "discipline: public health; design: cross-sectional survey; "
        "population: rural households; location: Sokoto State, Nigeria; "
        "period: 2021 rainy season; focus: household water quality"
    )
    assert "Exact Formal Title" not in basis
    assert "P-SECRET" not in basis
    assert "10.1000" not in basis


def test_citation_label_must_be_supplied_as_author_year():
    assert funnel.normalize_citation_label("(Njoku et al., 2022)") == (
        "Njoku et al., 2022"
    )
    assert funnel.normalize_citation_label("Njoku et al. (2022)") == (
        "Njoku et al., 2022"
    )
    assert not funnel.normalize_citation_label("10.2427/8841")
    assert not funnel.normalize_citation_label("https://openalex.org/W123")
    assert not funnel.normalize_citation_label("Njoku et al.")


def test_unverified_closed_render_is_explicitly_candidate_provenance():
    candidate = row(answer="The measured prevalence was 17%.")
    rendered = funnel.render_closed(
        candidate,
        "Study context (scope metadata):\nLocation: Sokoto State, Nigeria",
        citation_label="Njoku et al., 2022",
        verification_tier="UNVERIFIED",
    )["messages"][1]["content"]
    assert "Provenance: UNVERIFIED_STUDY" in rendered
    assert "Citation: (Njoku et al., 2022) [unverified]" in rendered
    assert "Study basis: location: Sokoto State, Nigeria" in rendered
