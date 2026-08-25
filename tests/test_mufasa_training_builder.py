from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest


MODULE_DIR = (
    Path(__file__).parents[1]
    / "01-data-engineering"
    / "data-extraction"
)
sys.path.insert(0, str(MODULE_DIR))
SPEC = importlib.util.spec_from_file_location(
    "mufasa_training_builder_test", MODULE_DIR / "mufasa_training_builder.py",
)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def config(tmp_path: Path, **updates):
    values = {
        "extraction_root": tmp_path / "extraction",
        "markdown_dir": tmp_path / "markdown",
        "split_manifest": tmp_path / "manifest.parquet",
        "output_root": tmp_path / "training",
        "router_workers": 1,
    }
    values.update(updates)
    return builder.BuildConfig(**values)


def candidate(pair_id: str, split: str, *, closed_ready: bool = True):
    return {
        "pair_id": pair_id,
        "paper_id": f"P-{pair_id}",
        "family_id": f"F-{pair_id}",
        "split": split,
        "pair_type": "FACTUAL",
        "question": "What was found in Sokoto?",
        "question_key": "whatwasfoundinsokoto",
        "extraction_model": "model",
        "pair_kind": "RESULT",
        "tags_json": "[]",
        "support_route": "OPEN_AS_IS",
        "evidence_json": '[{"quote":"The result was 17%."}]',
        "support_report_json": '{"supported":true}',
        "paper_context": "Study: Sokoto result",
        "paper_context_json": '{"paper_id":"P-test"}',
        "verification_tier": "VERIFIED",
        "inclusion_source": "STRICT_SFT",
        "reason_code": "",
        "reason_detail": "",
        "source_row_json": "{}",
        "descriptor": "a study reported in 'Sokoto result'" if closed_ready else "",
        "closed_ready": closed_ready,
        "closed_reason": "" if closed_ready else "no descriptor",
        "open_prompt": "Evidence: The result was 17%. Question: What was found?",
        "open_response": "17%.",
        "open_messages_json": '[{"role":"user","content":"open"},{"role":"assistant","content":"17%."}]',
        "open_tokens": 20,
        "closed_prompt": "From the Sokoto study, what was found?" if closed_ready else "",
        "closed_response": "17%." if closed_ready else "",
        "closed_messages_json": '[{"role":"user","content":"closed"},{"role":"assistant","content":"17%."}]' if closed_ready else "",
        "closed_tokens": 10 if closed_ready else 0,
    }


def _mixed_lane_inputs(tmp_path: Path) -> tuple[builder.BuildConfig, str]:
    extraction = tmp_path / "mixed-extraction"
    raw = extraction / "raw"
    markdown = tmp_path / "mixed-markdown"
    raw.mkdir(parents=True)
    markdown.mkdir()
    paper_id = "P-MIXED"
    source = "Among households in Sokoto, the measured prevalence was 17%."
    (markdown / f"{paper_id}.md").write_text(
        "<!-- MUFASA_PDF_PAGE: 1 -->\n\n## Results\n\n" + source,
        encoding="utf-8",
    )
    base = {
        "paper_id": paper_id,
        "pair_kind": "RESULT",
        "reasoning": "",
        "answer": "",
        "chosen": "",
        "rejected": "",
        "rejection_reason": "",
        "positive_quote": "",
        "hard_negative_quote": "",
        "negative_reason": "",
        "tags_json": '["sokoto"]',
    }
    rows = [
        {
            **base,
            "pair_id": f"{paper_id}:factual:verified",
            "pair_type": "FACTUAL",
            "question": "What prevalence was measured among Sokoto households?",
            "answer": "17%.",
        },
        {
            **base,
            "pair_id": f"{paper_id}:factual:unverified",
            "pair_type": "FACTUAL",
            "question": "What unsupported prevalence was proposed for Sokoto?",
            "answer": "99%.",
        },
        {
            **base,
            "pair_id": f"{paper_id}:reasoning:unverified",
            "pair_type": "REASONING",
            "question": "What unsupported inference was made about Sokoto households?",
            "reasoning": "The survey was extrapolated beyond the reported sample.",
            "answer": "99%.",
        },
        {
            **base,
            "pair_id": f"{paper_id}:preference:verified",
            "pair_type": "PREFERENCE",
            "question": "Which answer accurately reports the Sokoto prevalence?",
            "chosen": source,
            "rejected": "The measured prevalence was 99%.",
            "rejection_reason": "The rejected value is not reported.",
        },
        {
            **base,
            "pair_id": f"{paper_id}:preference:unverified",
            "pair_type": "PREFERENCE",
            "question": "Which answer claims an unsupported Sokoto intervention effect?",
            "chosen": "The intervention reduced symptoms by 42%.",
            "rejected": "The intervention reduced symptoms by 12%.",
            "rejection_reason": "Neither value is established by the supplied span.",
        },
        {
            **base,
            "pair_id": f"{paper_id}:reranker:valid",
            "pair_type": "RERANKER",
            "question": "Which passage contains the reported Sokoto prevalence?",
            "positive_quote": source,
            "hard_negative_quote": "The introduction describes regional public health policy.",
            "negative_reason": "The passage is topical but contains no prevalence result.",
        },
        {
            **base,
            "pair_id": f"{paper_id}:factual:structural",
            "pair_type": "FACTUAL",
            "question": "",
            "answer": "17%.",
        },
    ]
    pd.DataFrame(rows).to_parquet(extraction / "training_pairs.parquet", index=False)

    evidence_rows = []
    for index, row in enumerate(rows[:-1], 1):
        evidence_rows.append({
            "evidence_id": f"E{index}",
            "paper_id": paper_id,
            "owner_kind": "TRAINING",
            "owner_id": row["pair_id"],
            "page": 1,
            "section": "Results",
            "quote": source,
        })
    pd.DataFrame(evidence_rows).to_parquet(
        extraction / "evidence_spans.parquet", index=False,
    )
    pd.DataFrame([{
        "paper_id": paper_id,
        "complete": True,
        "model": "mixed-lane-test-model",
    }]).to_parquet(extraction / "extraction_status.parquet", index=False)
    pd.DataFrame([{
        "paper_id": paper_id,
        "is_real_science": True,
        "is_africa_relevant": True,
        "title": "Household Prevalence in Sokoto",
        "discipline": "public health",
        "key_contribution": "A measured household prevalence estimate.",
    }]).to_parquet(extraction / "paper_profiles.parquet", index=False)
    pd.DataFrame([{
        "paper_id": paper_id,
        "source_task": "context",
        "context_id": "CTX-1",
        "label": "household survey",
        "study_design": "cross-sectional survey",
        "population_text": "households in Sokoto",
        "sample_size_text": "240 households",
        "period_text": "2024",
        "conditions_json": '["community prevalence"]',
    }]).to_parquet(extraction / "study_contexts.parquet", index=False)
    pd.DataFrame([{
        "paper_id": paper_id,
        "is_african_innovation": True,
        "innovation_type": "community surveillance",
        "constraint_addressed": "limited local prevalence data",
        "what_is_distinctive": "household-level sampling",
        "why_it_matters_here": "supports local planning",
        "place": "Sokoto, Nigeria",
        "materials_or_species_json": '["household survey"]',
    }]).to_parquet(extraction / "african_innovation.parquet", index=False)
    manifest = tmp_path / "mixed-manifest.parquet"
    pd.DataFrame([{
        "paper_id": paper_id,
        "family_id": "F-MIXED",
        "split": "train",
        "licence": "not-a-training-gate",
    }]).to_parquet(manifest, index=False)
    return builder.BuildConfig(
        extraction_root=extraction,
        markdown_dir=markdown,
        split_manifest=manifest,
        output_root=tmp_path / "mixed-output",
        router_workers=1,
        progress_every=0,
        open_share=1.0,
        closed_share=0.0,
        dual_share=0.0,
    ), source


def test_frozen_manifest_is_authoritative_and_does_not_filter_licences(tmp_path):
    frame = pd.DataFrame([
        {"paper_id": "P1", "family_id": "F1", "split": "train", "licence": ""},
        {"paper_id": "P2", "family_id": "F2", "split": "evaluate", "licence": "custom"},
        {"paper_id": "P3", "family_id": "F3", "split": "test", "licence": None},
    ])
    path = tmp_path / "manifest.parquet"
    frame.to_parquet(path, index=False)
    loaded = builder.load_frozen_manifest(path)
    assert loaded.paper_id.tolist() == ["P1", "P2", "P3"]
    assert loaded.split.tolist() == ["train", "evaluate", "test"]


def test_cross_split_question_policy_protects_test_then_evaluate():
    train = candidate("train", "train")
    held = candidate("held", "test")
    kept_sft, kept_dpo, removed = builder.suppress_cross_split_questions(
        [train, held], [],
    )
    assert [row["pair_id"] for row in kept_sft] == ["held"]
    assert not kept_dpo
    assert [(lane, row["pair_id"], split) for lane, row, split in removed] == [
        ("SFT", "train", "test"),
    ]

    evaluate = candidate("eval", "evaluate")
    test = candidate("test", "test")
    kept_sft, _, removed = builder.suppress_cross_split_questions(
        [evaluate, test], [],
    )
    assert [row["pair_id"] for row in kept_sft] == ["test"]
    assert removed[0][1]["pair_id"] == "eval"
    assert removed[0][2] == "test"


def test_open_closed_dual_rendering_and_closed_fallback(tmp_path):
    dual = candidate("dual", "train", closed_ready=True)
    forced_open = candidate("forced", "train", closed_ready=False)
    assigned, masses = builder.assign_curriculum(
        [dual, forced_open],
        config(tmp_path, open_share=0.0, closed_share=0.0, dual_share=1.0),
    )
    by_id = {row["pair_id"]: row for row in assigned}
    assert by_id["dual"]["assignment"] == "DUAL"
    assert by_id["forced"]["assignment"] == "OPEN"
    assert masses["DUAL"] == 30
    rows = builder.render_sft(assigned)
    assert {(row["pair_id"], row["mode"]) for row in rows} == {
        ("dual", "OPEN"), ("dual", "CLOSED"), ("forced", "OPEN"),
    }
    assert len({row["example_id"] for row in rows}) == 3


def test_unverified_sft_without_evidence_gets_nonleaking_closed_fallback():
    record = {
        "pair_id": "P-FALLBACK:factual:f1",
        "paper_id": "P-FALLBACK",
        "pair_type": "FACTUAL",
        "pair_kind": "RESULT",
        "question": "What prevalence was reported?",
        "answer": "17%.",
        "reasoning": "",
        "tags_json": "[]",
    }
    route = {
        "route": "QUARANTINE_UNVERIFIED",
        "bundle": [],
        "report": {"supported": False, "reason": "no supporting span found"},
        "paper_verified": False,
    }
    candidate_row = builder._sft_candidate_from_route(
        record,
        route,
        "Study context (scope metadata):\nReported prevalence: 17%.",
        '{"paper_id":"P-FALLBACK"}',
        {"P-FALLBACK": {"family_id": "F-FALLBACK", "split": "train"}},
        {"P-FALLBACK": "test-model"},
    )
    assert candidate_row["closed_ready"]
    assert candidate_row["closed_tokens"] > 0
    assert "Paper ID: P-FALLBACK" in candidate_row["descriptor"]
    assert "17%" not in candidate_row["descriptor"]
    rendered = builder.render_sft_mixed([candidate_row])
    assert [(row["pair_id"], row["mode"]) for row in rendered] == [
        ("P-FALLBACK:factual:f1", "CLOSED"),
    ]


def test_provenance_wiring_keeps_core_routing_target_and_formats_both_dpo_sides():
    split = {"W1": {"family_id": "F1", "split": "train"}}
    models = {"W1": "test-model"}
    context = (
        "Study: 'Sokoto prevalence study'\nDiscipline: public health\n"
        "Location: Sokoto, Nigeria"
    )
    context_json = builder.canonical_json({
        "paper_id": "W1",
        "profile": {"discipline": "public health"},
        "selected_contexts": [{"study_design": "cross-sectional survey"}],
        "african_innovation": {"place": "Sokoto, Nigeria"},
    })
    trace = {
        "openalex_label": "Sam-Wobo et al., 2022",
        "citation_label": "Sam-Wobo et al., 2013",
        "citation_status": "CORRECTED_DOCUMENT",
        "metadata_source": "OPENALEX_PLUS_DOCUMENT",
    }
    evidence = [{"paper_id": "W1", "quote": "The prevalence was 17%."}]
    factual = {
        "pair_id": "W1:factual:1", "paper_id": "W1", "pair_type": "FACTUAL",
        "pair_kind": "RESULT", "question": "What was the prevalence?",
        "answer": "17%.", "reasoning": "", "tags_json": "[]",
    }
    route = {
        "route": "OPEN_AS_IS", "bundle": evidence,
        "report": {"supported": True}, "paper_verified": True,
    }
    candidate_row = builder._sft_candidate_from_route(
        factual, route, context, context_json, split, models,
        citation_trace=trace, provenance_enabled=True,
    )
    assert builder.funnel.assistant_turn(type("Row", (), factual)()) == "17%."
    assert "Provenance: PROVIDED_EVIDENCE" in candidate_row["open_response"]
    assert "Citation: (Sam-Wobo et al., 2013)" in candidate_row["open_response"]
    assert candidate_row["citation_raw_label"] == "Sam-Wobo et al., 2022"

    preference = {
        **factual,
        "pair_id": "W1:preference:1",
        "pair_type": "PREFERENCE",
        "chosen": "The prevalence was 17%.",
        "rejected": "The prevalence was 99%.",
        "rejection_reason": "Wrong figure.",
    }
    rendered = builder._preference_candidate_from_route(
        preference, route, context, context_json, split, models,
        citation_trace=trace, provenance_enabled=True,
    )
    assert "Provenance: PROVIDED_EVIDENCE" in rendered["chosen"]
    assert "Provenance: PROVIDED_EVIDENCE" in rendered["rejected"]
    assert rendered["chosen"].count("Citation:") == rendered["rejected"].count("Citation:") == 1


def test_audit_only_mode_cannot_publish(tmp_path):
    build_config = config(
        tmp_path,
        provenance_mode="AUDIT_ONLY",
        citation_metadata_path=tmp_path / "citation_metadata.parquet",
    )
    with pytest.raises(ValueError, match="cannot publish"):
        builder.build_training_set(build_config, publish=True)


def test_curriculum_assignment_is_isolated_by_frozen_split(tmp_path):
    train = candidate("train", "train")
    build_config = config(
        tmp_path, open_share=0.45, closed_share=0.45, dual_share=0.10,
    )
    alone, _ = builder.assign_curriculum_by_split([train], build_config)
    with_heldout, _ = builder.assign_curriculum_by_split([
        train,
        {**candidate("test", "test"), "open_tokens": 1, "closed_tokens": 200},
    ], build_config)
    assert next(row for row in alone if row["split"] == "train")["assignment"] == next(
        row for row in with_heldout if row["split"] == "train"
    )["assignment"]


def test_parquet_generation_round_trip_and_atomic_pointer(tmp_path):
    sft_rows = builder.render_sft([
        {**candidate("one", "train"), "assignment": "OPEN"},
    ])
    frames = {
        "sft_examples": builder._frame(sft_rows, builder.SFT_COLUMNS),
        "sft_mixed": builder._frame(sft_rows, builder.SFT_COLUMNS),
        "dpo_pairs": builder._frame([], builder.DPO_COLUMNS),
        "preference_mixed": builder._frame([], builder.DPO_COLUMNS),
        "reranker_mixed": builder._frame([], builder.RERANKER_COLUMNS),
        "quarantine": builder._frame([], builder.QUARANTINE_COLUMNS),
        "discarded": builder._frame([], builder.DISCARDED_COLUMNS),
    }
    identity = "a" * 64
    manifest = {
        "format": builder.OUTPUT_FORMAT,
        "run_id": identity[:24],
        "identity_sha256": identity,
    }
    run_dir, marker = builder._publish(frames, tmp_path / "training", manifest)
    assert run_dir.is_dir()
    assert marker["files"]["sft_examples.parquet"]["rows"] == 1
    pointer = json.loads((tmp_path / "training" / "LATEST.json").read_text())
    assert pointer["run_id"] == identity[:24]
    restored = pd.read_parquet(run_dir / "sft_examples.parquet")
    assert restored.example_id.tolist() == ["one:open"]
    for name, schema in builder.OUTPUT_SCHEMAS.items():
        assert pq.read_schema(run_dir / f"{name}.parquet").equals(
            schema, check_metadata=False,
        )
    messages = pq.read_table(run_dir / "sft_examples.parquet")["messages"].to_pylist()[0]
    assert messages == [
        {"role": "user", "content": "open"},
        {"role": "assistant", "content": "17%."},
    ]


def test_mixed_lanes_preserve_strict_outputs_and_keep_objectives_separate(
    tmp_path, monkeypatch,
):
    build_config, source = _mixed_lane_inputs(tmp_path)

    def deterministic_routes(payload):
        records, initial = payload[0], payload[1]
        routes = {}
        for record in records:
            pair_id = record["pair_id"]
            verified = pair_id.endswith(":verified")
            routes[pair_id] = {
                "route": "OPEN_AS_IS" if verified else "QUARANTINE_UNVERIFIED",
                "bundle": initial[pair_id],
                "report": {
                    "supported": verified,
                    "reason": "" if verified else "target is not established by the paper",
                },
                "paper_verified": verified,
            }
        return records, routes, False

    monkeypatch.setattr(builder, "_route_task", deterministic_routes)
    outcome = builder.build_training_set(build_config, publish=False)
    frames = outcome.frames

    strict_sft = frames["sft_examples"].sort_values("example_id").reset_index(drop=True)
    mixed_sft = frames["sft_mixed"].sort_values("example_id").reset_index(drop=True)
    copied_sft = mixed_sft[mixed_sft.example_id.isin(strict_sft.example_id)]
    pd.testing.assert_frame_equal(strict_sft, copied_sft.reset_index(drop=True))
    assert strict_sft.pair_id.tolist() == ["P-MIXED:factual:verified"]
    assert set(mixed_sft.pair_id) == {
        "P-MIXED:factual:verified",
        "P-MIXED:factual:unverified",
        "P-MIXED:reasoning:unverified",
    }
    assert set(mixed_sft.pair_type) == {"FACTUAL", "REASONING"}
    unverified_sft = mixed_sft[mixed_sft.verification_tier.eq("UNVERIFIED")]
    assert set(unverified_sft.pair_id) == {
        "P-MIXED:factual:unverified",
        "P-MIXED:reasoning:unverified",
    }
    assert set(unverified_sft["mode"]) == {"OPEN"}
    assert unverified_sft.reason_code.ne("").all()
    assert not set(unverified_sft.pair_id) & set(strict_sft.pair_id)

    strict_preference = frames["dpo_pairs"].sort_values("pair_id").reset_index(drop=True)
    mixed_preference = frames["preference_mixed"].sort_values("pair_id").reset_index(drop=True)
    copied_preference = mixed_preference[
        mixed_preference.pair_id.isin(strict_preference.pair_id)
    ]
    pd.testing.assert_frame_equal(
        strict_preference, copied_preference.reset_index(drop=True),
    )
    assert strict_preference.pair_id.tolist() == ["P-MIXED:preference:verified"]
    assert set(mixed_preference.pair_id) == {
        "P-MIXED:preference:verified",
        "P-MIXED:preference:unverified",
    }
    assert set(mixed_preference.pair_type) == {"PREFERENCE"}
    preference_unverified = mixed_preference[
        mixed_preference.pair_id.eq("P-MIXED:preference:unverified")
    ].iloc[0]
    assert preference_unverified.verification_tier == "UNVERIFIED"
    assert preference_unverified.reason_code
    assert preference_unverified.pair_id not in set(strict_preference.pair_id)

    reranker = frames["reranker_mixed"]
    assert reranker.pair_id.tolist() == ["P-MIXED:reranker:valid"]
    assert set(reranker.pair_type) == {"RERANKER"}
    assert reranker.iloc[0].verification_tier == "UNVERIFIED"
    assert reranker.iloc[0].reason_code == "HARD_NEGATIVE_NOT_VALIDATED"
    assert frames["sft_examples"].pair_id.is_unique
    assert frames["sft_mixed"].example_id.is_unique
    assert frames["dpo_pairs"].pair_id.is_unique
    assert frames["preference_mixed"].pair_id.is_unique
    assert frames["reranker_mixed"].pair_id.is_unique

    structural_id = "P-MIXED:factual:structural"
    training_pair_ids = set().union(*(
        set(frames[name].pair_id)
        for name in (
            "sft_examples", "sft_mixed", "dpo_pairs",
            "preference_mixed", "reranker_mixed",
        )
    ))
    assert structural_id not in training_pair_ids
    structural = frames["discarded"][frames["discarded"].pair_id.eq(structural_id)]
    assert structural.stage.tolist() == ["STRUCTURAL"]
    assert "blank question" in structural.reason_code.iloc[0]

    enriched = mixed_sft[
        mixed_sft.pair_id.eq("P-MIXED:factual:unverified")
    ].iloc[0]
    assert "Study: 'Household Prevalence in Sokoto'" in enriched.prompt
    assert "design: cross-sectional survey" in enriched.prompt
    assert "population: households in Sokoto" in enriched.prompt
    assert "Location: Sokoto, Nigeria" in enriched.prompt
    context = json.loads(enriched.paper_context_json)
    assert context["paper_id"] == "P-MIXED"
    assert context["profile"]["title"] == "Household Prevalence in Sokoto"
    assert context["selected_contexts"][0]["context_id"] == "CTX-1"
    assert context["african_innovation"]["place"] == "Sokoto, Nigeria"
    evidence = json.loads(enriched.evidence_json)
    assert evidence[0]["paper_id"] == "P-MIXED"
    assert evidence[0]["owner_id"] == enriched.pair_id
    assert evidence[0]["page"] == 1
    assert evidence[0]["section"] == "Results"
    assert evidence[0]["quote"] == source
    source_row = json.loads(enriched.source_row_json)
    assert source_row["pair_id"] == enriched.pair_id
    assert enriched.extraction_model == "mixed-lane-test-model"

    run_dir, marker = builder._publish(
        frames, tmp_path / "mixed-published", outcome.manifest,
    )
    assert set(marker["files"]) == {
        f"{name}.parquet" for name in builder.OUTPUT_SCHEMAS
    }
    for name in ("sft_mixed", "preference_mixed", "reranker_mixed"):
        path = run_dir / f"{name}.parquet"
        assert pq.read_schema(path).equals(
            builder.OUTPUT_SCHEMAS[name], check_metadata=False,
        )
        assert pq.read_table(path).to_pylist() == frames[name].to_dict("records")


def test_tiny_end_to_end_build_routes_every_lane_without_a_licence_gate(tmp_path):
    extraction = tmp_path / "extraction"
    raw = extraction / "raw"
    markdown = tmp_path / "markdown"
    raw.mkdir(parents=True)
    markdown.mkdir()
    paper_id = "P1"
    source = "Among households in Sokoto, the measured prevalence was 17%."
    (markdown / f"{paper_id}.md").write_text(
        "<!-- MUFASA_PDF_PAGE: 1 -->\n\n## Results\n\n" + source,
        encoding="utf-8",
    )
    base = {
        "paper_id": paper_id,
        "pair_kind": "RESULT",
        "reasoning": "",
        "chosen": "",
        "rejected": "",
        "rejection_reason": "",
        "positive_quote": "",
        "hard_negative_quote": "",
        "negative_reason": "",
        "tags_json": "[]",
    }
    pairs = pd.DataFrame([
        {**base, "pair_id": "P1:factual:f1", "pair_type": " factual ",
         "question": "What was the prevalence in Sokoto?", "answer": "17%."},
        {**base, "pair_id": "P1:factual:f2", "pair_type": "FACTUAL",
         "question": "What was the unsupported value in Sokoto?", "answer": "99%."},
        {**base, "pair_id": "P1:preference:p1", "pair_type": "PREFERENCE",
         "question": "What was the prevalence in Sokoto?", "answer": "",
         "chosen": "The prevalence in Sokoto was 17%.",
         "rejected": "The prevalence in Sokoto was 99%."},
        {**base, "pair_id": "P1:reranker:r1", "pair_type": "RERANKER",
         "question": "Which passage reports prevalence?", "answer": "",
         "positive_quote": source, "hard_negative_quote": "A related passage."},
    ])
    pairs.to_parquet(extraction / "training_pairs.parquet", index=False)
    pd.DataFrame([
        {"evidence_id": "E1", "paper_id": paper_id, "owner_kind": "TRAINING",
         "owner_id": "P1:factual:f1", "page": 1, "section": "Results", "quote": source},
        {"evidence_id": "E2", "paper_id": paper_id, "owner_kind": "TRAINING",
         "owner_id": "P1:preference:p1", "page": 1, "section": "Results", "quote": source},
    ]).to_parquet(extraction / "evidence_spans.parquet", index=False)
    pd.DataFrame([{
        "paper_id": paper_id, "complete": True, "model": "local-test",
    }]).to_parquet(extraction / "extraction_status.parquet", index=False)
    pd.DataFrame([{
        "paper_id": paper_id, "is_real_science": True, "is_africa_relevant": True,
        "title": "Prevalence in Sokoto", "discipline": "health",
    }]).to_parquet(extraction / "paper_profiles.parquet", index=False)
    pd.DataFrame([{
        "paper_id": paper_id, "source_task": "context", "study_design": "survey",
        "population_text": "households", "sample_size_text": "", "period_text": "",
    }]).to_parquet(extraction / "study_contexts.parquet", index=False)
    pd.DataFrame([{
        "paper_id": paper_id, "place": "Sokoto, Nigeria",
    }]).to_parquet(extraction / "african_innovation.parquet", index=False)
    split_path = tmp_path / "manifest.parquet"
    pd.DataFrame([{
        "paper_id": paper_id, "family_id": "F1", "split": "train",
        "licence": "deliberately-not-inspected",
    }]).to_parquet(split_path, index=False)

    outcome = builder.build_training_set(builder.BuildConfig(
        extraction_root=extraction,
        markdown_dir=markdown,
        split_manifest=split_path,
        output_root=tmp_path / "output",
        router_workers=1,
        progress_every=0,
    ), publish=False)
    assert set(outcome.frames["sft_examples"].pair_id) == {"P1:factual:f1"}
    assert outcome.frames["dpo_pairs"].pair_id.tolist() == ["P1:preference:p1"]
    assert set(outcome.frames["quarantine"].pair_id) == {
        "P1:factual:f2", "P1:reranker:r1",
    }
    assert outcome.frames["discarded"].empty
    assert len(list((tmp_path / "output" / "router-cache").rglob("*.json"))) == 1
    resumed = builder.build_training_set(builder.BuildConfig(
        extraction_root=extraction,
        markdown_dir=markdown,
        split_manifest=split_path,
        output_root=tmp_path / "output",
        router_workers=1,
        progress_every=0,
    ), publish=False)
    router_stage = next(row for row in resumed.stages if row["stage"] == "4 support router")
    assert "resumed 1/1 paper checkpoints" in router_stage["detail"]
