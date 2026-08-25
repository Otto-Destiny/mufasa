from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest


MODULE_DIR = (
    Path(__file__).parents[1] / "01-data-engineering" / "data-extraction"
)
sys.path.insert(0, str(MODULE_DIR))
import mufasa_dataset as funnel  # noqa: E402
import mufasa_semantic_refiner as refiner  # noqa: E402
import mufasa_training_builder as builder  # noqa: E402


def _sft_row(
    pair_id: str, *, verified: bool, mode: str = "OPEN",
    provenance: bool = False,
) -> dict:
    paper_id = pair_id.split(":", 1)[0]
    question = f"What was reported for {paper_id}?"
    core_response = "The reported prevalence was 17%."
    source_row = {
        "pair_id": pair_id,
        "paper_id": paper_id,
        "pair_type": "FACTUAL",
        "pair_kind": "RESULT",
        "question": question,
        "answer": core_response,
        "reasoning": "",
    }
    prompt = "old open prompt" if mode == "OPEN" else "old closed prompt"
    response = core_response
    if provenance:
        response = funnel.format_provenance_response(
            core_response,
            provenance=(
                funnel.PROVIDED_EVIDENCE if verified and mode == "OPEN"
                else funnel.LEARNED_STUDY if verified
                else funnel.UNVERIFIED_STUDY
            ),
            citation_label="Sam-Wobo et al., 2013",
            study_basis="location: Sokoto, Nigeria",
            evidence_labels=("Evidence 1",) if mode == "OPEN" else (),
        )
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return {
        "example_id": f"{pair_id}:{mode.casefold()}",
        "pair_id": pair_id,
        "paper_id": paper_id,
        "family_id": f"F-{paper_id}",
        "split": "train",
        "pair_type": "FACTUAL",
        "mode": mode,
        "assignment": mode,
        "support_route": "OPEN_AS_IS" if verified else "QUARANTINE_UNVERIFIED",
        "question": question,
        "question_key": funnel.skeleton(question),
        "prompt": prompt,
        "response": response,
        "messages": messages,
        "messages_json": builder.canonical_json(messages),
        "descriptor": "Study: closed fallback" if mode == "CLOSED" else "",
        "evidence_json": "[]",
        "support_report_json": '{"reason":"old evidence was inadequate"}',
        "pair_kind": "RESULT",
        "tags_json": "[]",
        "extraction_model": "test-model",
        "paper_context": f"Study: '{paper_id} prevalence study'\nLocation: Sokoto, Nigeria",
        "paper_context_json": builder.canonical_json({"paper_id": paper_id}),
        "verification_tier": "VERIFIED" if verified else "UNVERIFIED",
        "inclusion_source": "STRICT_SFT" if verified else "SUPPORT_QUARANTINE",
        "reason_code": "" if verified else "QUARANTINE_UNVERIFIED",
        "reason_detail": "" if verified else "old evidence was inadequate",
        "source_row_json": builder.canonical_json(source_row),
        "citation_label": "Sam-Wobo et al., 2013" if provenance else "",
        "citation_raw_label": "Sam-Wobo et al., 2022" if provenance else "",
        "citation_status": "CORRECTED_DOCUMENT" if provenance else "",
        "citation_metadata_source": "OPENALEX_PLUS_DOCUMENT" if provenance else "",
        "citation_metadata_json": '{"source":"audit"}' if provenance else "",
        "token_estimate": 25,
    }


def _source_generation(
    tmp_path: Path, eligible_count: int = 1, paper_limit=None,
    provenance: bool = False,
):
    training = tmp_path / "training_set"
    strict = _sft_row("PV:verified", verified=True, provenance=provenance)
    unverified = [
        _sft_row(
            f"PU{number}:unverified", verified=False, mode="CLOSED",
            provenance=provenance,
        )
        for number in range(eligible_count)
    ]
    sft_strict = builder._frame([strict], builder.SFT_COLUMNS)
    sft_mixed = builder._frame([strict, *unverified], builder.SFT_COLUMNS)
    quarantine_rows = []
    for row in unverified:
        quarantine_rows.append({
            "record_id": f"{row['pair_id']}:SUPPORT",
            "pair_id": row["pair_id"],
            "paper_id": row["paper_id"],
            "family_id": row["family_id"],
            "split": row["split"],
            "pair_type": row["pair_type"],
            "stage": "SUPPORT",
            "reason_code": "QUARANTINE_UNVERIFIED",
            "reason_detail": row["reason_detail"],
            "support_route": row["support_route"],
            "question": row["question"],
            "question_key": row["question_key"],
            "target": row["response"],
            "evidence_json": row["evidence_json"],
            "support_report_json": row["support_report_json"],
            "source_row_json": row["source_row_json"],
            "extraction_model": row["extraction_model"],
        })
    frames = {
        "sft_examples": sft_strict,
        "sft_mixed": sft_mixed,
        "dpo_pairs": builder._frame([], builder.DPO_COLUMNS),
        "preference_mixed": builder._frame([], builder.DPO_COLUMNS),
        "reranker_mixed": builder._frame([], builder.RERANKER_COLUMNS),
        "quarantine": builder._frame(quarantine_rows, builder.QUARANTINE_COLUMNS),
        "discarded": builder._frame([], builder.DISCARDED_COLUMNS),
    }
    identity = "a" * 64
    marker = {
        "format": builder.OUTPUT_FORMAT,
        "run_id": identity[:24],
        "identity_sha256": identity,
        "config": {"paper_limit": paper_limit},
        "selected_papers": eligible_count + 1,
        "inputs": {
            "router_module": builder._file_hash(Path(funnel.__file__)),
            "builder_module": builder._file_hash(Path(builder.__file__)),
        },
    }
    builder._publish(frames, training, marker)
    markdown = tmp_path / "markdown"
    markdown.mkdir()
    locations = {}
    for row in unverified:
        quote = f"In {row['paper_id']}, the reported prevalence was 17%."
        text = f"# Results\n\n{quote}\n"
        (markdown / f"{row['paper_id']}.md").write_text(text, encoding="utf-8")
        start = text.index(quote)
        locations[row["pair_id"]] = (quote, start, start + len(quote))
    return training, markdown, strict, unverified, locations


def _result(row: dict, location: tuple[str, int, int], *, passing=True):
    quote, start, end = location
    span = {
        "paper_id": row["paper_id"],
        "quote": quote,
        "page": 1,
        "section": "Results",
        "source_kind": "TEXT",
        "source_label": "same-paper semantic candidate",
        "char_start": start,
        "char_end": end,
    }
    return {
        "route": (
            "VECTOR_CANDIDATE_DETERMINISTIC_PASS"
            if passing else "STILL_QUARANTINED"
        ),
        "bundle": [span],
        "report": {"supported": passing, "reason": "" if passing else "prose coverage low"},
        "hits": [{
            "chunk_index": 0,
            "score": 0.8125,
            "candidate_origin": "semantic_rank",
            "evidence_token_count": 12,
            "span": span,
        }],
        "query_truncation_count": 0,
        "long_query_split_count": 0,
        "retrieval_limitations": [],
    }


def _retrieval_config(paper_limit=None):
    return {
        "model_id": "BAAI/bge-base-en-v1.5",
        "model_revision": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        "paper_limit": paper_limit,
        "max_tokens": 480,
    }


def test_refines_closed_fallback_and_publishes_training_ready_generation(tmp_path):
    training, markdown, strict, rows, locations = _source_generation(tmp_path)
    source = refiner.load_source_run(training)
    pair_id = rows[0]["pair_id"]
    outcome = refiner.refine_sft_mixed(
        refiner.RefineConfig(training, markdown),
        {pair_id: _result(rows[0], locations[pair_id])},
        retrieval_config=_retrieval_config(),
        source_run=source,
    )

    assert outcome.complete_coverage and outcome.training_ready
    assert outcome.latest_advanced
    assert outcome.source_builder_run_id == source.marker["run_id"]
    refined = outcome.frames["sft_mixed"]
    updated = refined[refined.pair_id.eq(pair_id)].iloc[0]
    assert updated.example_id == f"{pair_id}:open"
    assert updated["mode"] == updated.assignment == "OPEN"
    assert updated.support_route == "VECTOR_CANDIDATE_DETERMINISTIC_PASS"
    assert updated.verification_tier == "UNVERIFIED"
    assert updated.inclusion_source == "SEMANTIC_REFINED"
    assert updated.reason_code == "QUARANTINE_UNVERIFIED"
    assert locations[pair_id][0] in updated.prompt
    assert updated.response == rows[0]["response"]
    verified = refined[refined.pair_id.eq(strict["pair_id"])].iloc[0].to_dict()
    assert builder.canonical_json(verified) == builder.canonical_json(strict)

    assert (outcome.run_dir / "semantic_routes.parquet").is_file()
    assert pq.read_schema(outcome.run_dir / "sft_mixed.parquet").equals(
        builder.OUTPUT_SCHEMAS["sft_mixed"], check_metadata=False,
    )
    candidates = outcome.frames["semantic_candidates"]
    assert candidates.quote_sha256.str.len().tolist() == [64]
    assert "quote" not in candidates.columns
    effective = refiner.resolve_effective_sft_mixed(training)
    assert effective.source == "SEMANTIC_REFINED"
    assert effective.path == outcome.run_dir / "sft_mixed.parquet"


def test_partial_preview_persists_audit_without_advancing_latest(tmp_path):
    training, markdown, _, rows, locations = _source_generation(
        tmp_path, eligible_count=2,
    )
    pair_id = rows[0]["pair_id"]
    outcome = refiner.refine_sft_mixed(
        refiner.RefineConfig(training, markdown, preview=True),
        {pair_id: _result(rows[0], locations[pair_id], passing=False)},
        retrieval_config=_retrieval_config(paper_limit=10),
    )
    assert outcome.run_dir.is_dir()
    assert not outcome.complete_coverage
    assert not outcome.training_ready
    assert not outcome.latest_advanced
    assert not (training / "SEMANTIC_LATEST.json").exists()
    updated = outcome.frames["sft_mixed"].query("pair_id == @pair_id").iloc[0]
    assert updated.support_route == "SEMANTIC_BEST_UNVERIFIED"
    assert updated.verification_tier == "UNVERIFIED"
    untouched_id = rows[1]["pair_id"]
    source_mixed = refiner.load_source_run(training).sft_mixed
    before = source_mixed[source_mixed.pair_id.eq(untouched_id)].iloc[0].to_dict()
    refined_mixed = outcome.frames["sft_mixed"]
    after = refined_mixed[refined_mixed.pair_id.eq(untouched_id)].iloc[0].to_dict()
    assert builder.canonical_json(before) == builder.canonical_json(after)


def test_semantic_closed_to_open_preserves_core_and_rerenders_provenance(tmp_path):
    training, markdown, _, rows, locations = _source_generation(
        tmp_path, provenance=True,
    )
    pair_id = rows[0]["pair_id"]
    outcome = refiner.refine_sft_mixed(
        refiner.RefineConfig(training, markdown),
        {pair_id: _result(rows[0], locations[pair_id])},
        retrieval_config=_retrieval_config(),
        publish=False,
    )
    updated = outcome.frames["sft_mixed"].query("pair_id == @pair_id").iloc[0]
    core = "The reported prevalence was 17%."
    assert updated.response.startswith(core + "\n\nProvenance:")
    assert "Provenance: UNVERIFIED_STUDY" in updated.response
    assert "Citation: (Sam-Wobo et al., 2013) [unverified]" in updated.response
    assert updated.citation_raw_label == "Sam-Wobo et al., 2022"


def test_rejects_nonexact_or_cross_paper_semantic_spans(tmp_path):
    training, markdown, _, rows, locations = _source_generation(tmp_path)
    pair_id = rows[0]["pair_id"]
    bad = _result(rows[0], locations[pair_id])
    bad["bundle"][0]["paper_id"] = "DIFFERENT-PAPER"
    with pytest.raises(ValueError, match="cross-paper"):
        refiner.refine_sft_mixed(
            refiner.RefineConfig(training, markdown),
            {pair_id: bad},
            retrieval_config=_retrieval_config(),
            publish=False,
        )


def test_paper_limited_builder_cannot_advance_semantic_latest(tmp_path):
    training, markdown, _, rows, locations = _source_generation(
        tmp_path, paper_limit=10,
    )
    pair_id = rows[0]["pair_id"]
    with pytest.raises(ValueError, match="cannot advance"):
        refiner.refine_sft_mixed(
            refiner.RefineConfig(training, markdown),
            {pair_id: _result(rows[0], locations[pair_id])},
            retrieval_config=_retrieval_config(),
            advance_latest=True,
        )
