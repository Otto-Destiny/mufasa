from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


MODULE_DIR = (
    Path(__file__).parents[1] / "01-data-engineering" / "data-extraction"
)
sys.path.insert(0, str(MODULE_DIR))
import mufasa_semantic as semantic  # noqa: E402


class WordTokenizer:
    is_fast = True

    def __call__(self, text, **_kwargs):
        offsets = [match.span() for match in re.finditer(r"[A-Za-z0-9.%]+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}

    @staticmethod
    def num_special_tokens_to_add(pair=False):
        return 3 if pair else 2


class PieceTokenizer(WordTokenizer):
    def __call__(self, text, **_kwargs):
        offsets = []
        for match in re.finditer(r"[A-Za-z0-9]+", text):
            offsets.extend(
                (start, min(start + 3, match.end()))
                for start in range(match.start(), match.end(), 3)
            )
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


class HashEncoder:
    tokenizer = WordTokenizer()
    max_seq_length = 256

    def __init__(self, dimensions=512):
        self.dimensions = dimensions

    def encode(self, sentences, **_kwargs):
        output = np.zeros((len(sentences), self.dimensions), dtype=np.float32)
        for row, sentence in enumerate(sentences):
            for token in re.findall(r"[A-Za-z0-9.%]+", sentence.casefold()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                output[row, int.from_bytes(digest, "little") % self.dimensions] += 1
        return output


class ZeroEncoder(HashEncoder):
    def encode(self, sentences, **_kwargs):
        return np.zeros((len(sentences), self.dimensions), dtype=np.float32)


class CountingEncoder(HashEncoder):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def encode(self, sentences, **kwargs):
        self.calls += 1
        return super().encode(sentences, **kwargs)


class BgeCaptureEncoder(HashEncoder):
    max_seq_length = 512
    mufasa_query_prefix = True

    def __init__(self):
        super().__init__()
        self.encoded = []

    def encode(self, sentences, **kwargs):
        self.encoded.append(list(sentences))
        return super().encode(sentences, **kwargs)


def pair(question, answer, paper_id="P1", pair_id="P1:factual:1"):
    return SimpleNamespace(
        paper_id=paper_id,
        pair_id=pair_id,
        pair_type="FACTUAL",
        question=question,
        answer=answer,
        reasoning="",
        chosen="",
    )


def _snapshot(tmp_path):
    root = tmp_path / semantic.PINNED_REVISION
    for name in semantic._REQUIRED_SNAPSHOT_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    (root / "config.json").write_text(json.dumps({
        "model_type": "bert", "hidden_size": 384, "num_hidden_layers": 6,
    }), encoding="utf-8")
    (root / "modules.json").write_text(json.dumps([
        {"type": "sentence_transformers.models.Transformer"},
        {"type": "sentence_transformers.models.Pooling"},
        {"type": "sentence_transformers.models.Normalize"},
    ]), encoding="utf-8")
    return root


def _bge_snapshot(tmp_path):
    root = tmp_path / semantic.PINNED_BGE_REVISION
    for name in semantic._REQUIRED_SNAPSHOT_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    (root / "config.json").write_text(json.dumps({
        "model_type": "bert", "hidden_size": 768, "num_hidden_layers": 12,
    }), encoding="utf-8")
    return root


def test_snapshot_resolver_accepts_only_complete_pinned_local_snapshot(
    tmp_path, monkeypatch,
):
    root = _snapshot(tmp_path)
    monkeypatch.setattr(
        semantic, "PINNED_MODEL_SHA256", hashlib.sha256(b"").hexdigest()
    )
    assert semantic.resolve_minilm_snapshot(root) == root.resolve()
    (root / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA256"):
        semantic.resolve_minilm_snapshot(root)
    (root / "model.safetensors").write_bytes(b"")
    wrong_revision = tmp_path / "main"
    wrong_revision.mkdir()
    with pytest.raises(RuntimeError, match="pinned"):
        semantic.resolve_minilm_snapshot(wrong_revision)
    (root / "tokenizer.json").unlink()
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        semantic.resolve_minilm_snapshot(root)


def test_bge_snapshot_validator_requires_pinned_768d_base(tmp_path):
    root = _bge_snapshot(tmp_path)
    assert semantic._validate_bge_snapshot(root) == root.resolve()
    config = root / "config.json"
    config.write_text(json.dumps({
        "model_type": "bert", "hidden_size": 384, "num_hidden_layers": 12,
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="bge-base-en-v1.5"):
        semantic._validate_bge_snapshot(root)


def test_pinned_local_model_has_expected_weight_digest_when_available():
    try:
        root = semantic.resolve_minilm_snapshot()
    except FileNotFoundError:
        pytest.skip("Pinned MiniLM snapshot is not provisioned on this machine")
    digest = hashlib.sha256((root / "model.safetensors").read_bytes()).hexdigest()
    assert digest == semantic.PINNED_MODEL_SHA256


def test_token_chunks_are_exact_bounded_overlapping_and_exclude_bibliography():
    words = [f"word{index}" for index in range(500)]
    body = " ".join(words)
    text = body + "\n\n# References\n\nThis citation must never be embedded."
    result = semantic.chunk_markdown_tokens("P1", text, WordTokenizer())
    chunks = result["chunks"]
    assert len(chunks) == 3
    assert all(chunk["token_count"] <= 220 for chunk in chunks)
    assert all(chunk["quote"] == text[chunk["char_start"]:chunk["char_end"]] for chunk in chunks)
    assert all("citation" not in chunk["quote"].casefold() for chunk in chunks)
    tokenized = [re.findall(r"word\d+", chunk["quote"]) for chunk in chunks]
    assert tokenized[0][-40:] == tokenized[1][:40]
    assert tokenized[1][-40:] == tokenized[2][:40]


def test_bge_instruction_is_applied_to_queries_only():
    model = BgeCaptureEncoder()
    index = semantic.build_text_semantic_index(
        "P1", "Sokoto households had prevalence of 17%.", model,
        max_tokens=100, overlap_tokens=10,
    )
    semantic.rank_semantic_records(index, [{
        "paper_id": "P1", "pair_id": "P1:factual:1", "pair_type": "FACTUAL",
        "question": "What was the prevalence in Sokoto?", "answer": "17%.",
        "reasoning": "", "chosen": "",
    }], model)
    assert all(
        not text.startswith(semantic.BGE_QUERY_PREFIX) for text in model.encoded[0]
    )
    assert all(
        text.startswith(semantic.BGE_QUERY_PREFIX) for text in model.encoded[1]
    )


@pytest.mark.parametrize("heading", [
    "References", "**References**", "__Bibliography__", "| References |",
])
def test_bibliography_cutoff_recognizes_plain_bold_and_table_headings(heading):
    text = f"Supported body finding.\n\n{heading}\n\nHidden citation text."
    result = semantic.chunk_markdown_tokens("P1", text, WordTokenizer())
    shown = " ".join(chunk["quote"] for chunk in result["chunks"])
    assert "Supported body finding" in shown
    assert "Hidden citation" not in shown


def test_wordpiece_boundaries_are_moved_to_whole_words_without_exceeding_cap():
    text = "extraordinary internationalization electroencephalography microbiology"
    result = semantic.chunk_markdown_tokens(
        "P1", text, PieceTokenizer(), max_tokens=8, overlap_tokens=2
    )
    assert len(result["chunks"]) > 1
    for chunk in result["chunks"]:
        start, end = chunk["char_start"], chunk["char_end"]
        assert chunk["token_count"] <= 8
        assert chunk["quote"] == text[start:end]
        assert not (start and text[start - 1].isalnum() and text[start].isalnum())
        assert not (end < len(text) and text[end - 1].isalnum() and text[end].isalnum())


def test_real_tokenizer_chunks_and_queries_never_split_or_overrun_when_available():
    try:
        root = semantic.resolve_minilm_snapshot()
    except FileNotFoundError:
        pytest.skip("Pinned MiniLM snapshot is not provisioned on this machine")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(root), local_files_only=True, use_fast=True
    )
    text = (
        "Electroencephalographically measured responses improved in Sokoto. " * 180
    )
    result = semantic.chunk_markdown_tokens("P1", text, tokenizer)
    for chunk in result["chunks"]:
        start, end = chunk["char_start"], chunk["char_end"]
        assert len(semantic._token_offsets(tokenizer, chunk["embedding_text"])) <= 220
        assert not (start and text[start - 1].isalnum() and text[start].isalnum())
        assert not (end < len(text) and text[end - 1].isalnum() and text[end].isalnum())
    variants = semantic.query_variants(
        " ".join(["electroencephalographically"] * 300),
        "Sokoto response was improved.",
        tokenizer,
        254,
    )
    assert semantic._query_truncation_count(variants) == 0
    assert all(item["token_count"] <= 254 for item in variants)


def test_table_header_is_embedding_context_only_and_all_chunks_remain_exact():
    header = "| District | Measure | Value |"
    rows = "\n".join(f"| Area {index} | prevalence | {index}% |" for index in range(20))
    text = f"{header}\n|---|---|---|\n{rows}"
    result = semantic.chunk_markdown_tokens(
        "P1", text, WordTokenizer(), max_tokens=16, overlap_tokens=4
    )
    inherited = [
        chunk for chunk in result["chunks"]
        if chunk["embedding_text"].startswith(header + "\n")
        and not chunk["quote"].startswith(header)
    ]
    assert inherited
    assert all(chunk["token_count"] <= 16 for chunk in result["chunks"])
    assert all(chunk["quote"] == text[chunk["char_start"]:chunk["char_end"]] for chunk in result["chunks"])


def test_cosine_ranking_has_stable_ties_and_same_paper_isolation():
    model = ZeroEncoder()
    text = "Alpha result.\n\nBeta result.\n\nGamma result."
    index = semantic.build_text_semantic_index(
        "P1", text, model, max_tokens=10, overlap_tokens=2
    )
    hits = semantic.rank_semantic_chunks(
        index, "Anything?", "Anything.", model, top_k_per_query=3
    )
    assert [hit["chunk_index"] for hit in hits] == [0, 1, 2]
    with pytest.raises(ValueError, match="Same-paper isolation"):
        semantic.route_quarantined_pair(
            pair("Question?", "Answer.", paper_id="P2"), index, model
        )


def test_neighbor_and_supplied_expansion_keeps_only_exact_bounded_same_paper_spans():
    model = ZeroEncoder()
    text = "Alpha result.\n\nBeta result.\n\nGamma result."
    index = semantic.build_text_semantic_index(
        "P1", text, model, max_tokens=10, overlap_tokens=2
    )
    beta = {
        "chunk_index": 1,
        "score": 0.0,
        "query_scores": {},
        "query_truncation_count": 0,
        "long_query_split_count": 0,
        "table_header_embedding_only": False,
        "span": semantic._public_span(index.chunks[1]),
    }
    supplied = semantic._public_span(index.chunks[0])
    expanded = semantic.expand_ranked_hits(
        index, [beta], model.tokenizer, [supplied], max_pool_tokens=6
    )
    assert [item["span"]["quote"].strip() for item in expanded] == [
        "Alpha result.", "Beta result.", "Gamma result.",
    ]
    assert sum(item["evidence_token_count"] for item in expanded) <= 6
    assert all(
        item["span"]["quote"]
        == text[item["span"]["char_start"]:item["span"]["char_end"]]
        for item in expanded
    )
    foreign = dict(supplied, paper_id="P2")
    with pytest.raises(ValueError, match="cross-paper"):
        semantic.expand_ranked_hits(index, [beta], model.tokenizer, [foreign])


@pytest.mark.parametrize(("question", "answer", "evidence"), [
    (
        "What was prevalence among children in Sokoto?",
        "The prevalence was 17%.",
        "Among adults in Sokoto, the prevalence was 17%.",
    ),
    (
        "What was the prevalence in 2022?",
        "The prevalence was 17%.",
        "The prevalence measured in 2019 was 17%.",
    ),
    (
        "What was prevalence in the treatment group?",
        "The prevalence was 5%.",
        "The control group had a prevalence of 5%.",
    ),
    (
        "What was the prevalence?",
        "It was 17.",
        "Annual rainfall was 17 mm.",
    ),
])
def test_deterministic_pass_never_releases_qualifier_ambiguous_candidate(
    question, answer, evidence,
):
    model = HashEncoder()
    index = semantic.build_text_semantic_index("P1", evidence, model)
    routed = semantic.route_quarantined_pair(pair(question, answer), index, model)
    assert routed["route"] in {
        "VECTOR_CANDIDATE_DETERMINISTIC_PASS", "STILL_QUARANTINED",
    }
    assert routed["paper_verified"] is False
    assert routed["release_to_sft"] is False


@pytest.mark.parametrize(("question", "answer", "evidence"), [
    ("What was prevalence?", "Prevalence was 19%.", "Prevalence was 17%."),
    ("Did treatment increase yield?", "Treatment increased yield.", "Treatment did not increase yield."),
    (
        "What was Kano prevalence?",
        "Kano prevalence was 17%.",
        "| State | Prevalence |\n|---|---|\n| Sokoto | 17% |\n| Kano | 21% |",
    ),
])
def test_number_negation_and_wrong_table_row_cannot_pass_deterministic_check(
    question, answer, evidence,
):
    model = HashEncoder()
    index = semantic.build_text_semantic_index("P1", evidence, model)
    routed = semantic.route_quarantined_pair(pair(question, answer), index, model)
    assert routed["route"] == "STILL_QUARANTINED"
    assert routed["release_to_sft"] is False


def test_per_sentence_queries_rescue_multi_span_candidate_but_keep_it_quarantined():
    model = HashEncoder()
    text = (
        "The Sokoto survey measured prevalence at 17%.\n\n"
        "Unrelated agricultural background about soil and rainfall.\n\n"
        "The Kano survey measured prevalence at 21%."
    )
    index = semantic.build_text_semantic_index(
        "P1", text, model, max_tokens=14, overlap_tokens=3
    )
    routed = semantic.route_quarantined_pair(
        pair(
            "What prevalence was measured in Sokoto and Kano?",
            "Sokoto prevalence was 17%. Kano prevalence was 21%.",
        ),
        index,
        model,
        max_spans=2,
        top_k_per_query=3,
    )
    assert routed["route"] == "VECTOR_CANDIDATE_DETERMINISTIC_PASS"
    assert len(routed["bundle"]) == 2
    assert routed["release_to_sft"] is False


def test_paper_router_batches_all_pair_queries_in_one_encode_call(tmp_path):
    model = CountingEncoder()
    (tmp_path / "P1.md").write_text(
        "Sokoto prevalence was 17%.\n\nKano prevalence was 21%.", encoding="utf-8"
    )
    records = [vars(pair("What was Sokoto prevalence?", "It was 17%.", pair_id="a")),
               vars(pair("What was Kano prevalence?", "It was 21%.", pair_id="b"))]
    routed = semantic.route_quarantined_paper(records, tmp_path, model)
    assert set(routed) == {"a", "b"}
    assert model.calls == 2  # one paper-chunk batch, one all-query batch


def test_batch_ranker_uses_one_query_encode_and_rejects_cross_paper():
    model = CountingEncoder()
    index = semantic.build_text_semantic_index(
        "P1", "Sokoto prevalence was 17%.\n\nKano prevalence was 21%.", model,
    )
    before = model.calls
    records = [vars(pair("Sokoto prevalence?", "17%.", pair_id="a")),
               vars(pair("Kano prevalence?", "21%.", pair_id="b"))]
    ranked = semantic.rank_semantic_records(index, records, model)
    assert set(ranked) == {"a", "b"}
    assert model.calls == before + 1
    records[1]["paper_id"] = "P2"
    with pytest.raises(ValueError, match="Same-paper isolation"):
        semantic.rank_semantic_records(index, records, model)


def test_ranked_candidate_audit_is_process_safe_and_rejects_foreign_span():
    candidate = vars(pair("Sokoto prevalence?", "17%."))
    hits = [{
        "span": {"paper_id": "P1", "quote": "Sokoto prevalence was 17%."},
        "long_query_split_count": 0,
        "table_header_embedding_only": False,
    }]
    result = semantic.evaluate_ranked_paper(
        [candidate], "P1", {candidate["pair_id"]: hits},
    )
    assert result[candidate["pair_id"]]["release_to_sft"] is False
    hits[0]["span"]["paper_id"] = "P2"
    with pytest.raises(ValueError, match="cross-paper"):
        semantic.evaluate_ranked_candidates(candidate, "P1", hits)


def test_long_table_header_is_embedding_only_and_cannot_fake_unit_support():
    model = HashEncoder()
    header = "| District | Concentration (mg/L) |"
    rows = "\n".join(f"| Area {index} | {index} |" for index in range(40))
    text = f"{header}\n|---|---|\n{rows}"
    index = semantic.build_text_semantic_index(
        "P1", text, model, max_tokens=16, overlap_tokens=4
    )
    routed = semantic.route_quarantined_pair(
        pair("What concentration was measured in Area 39?", "It was 39 mg/L."),
        index,
        model,
        top_k_per_query=5,
    )
    assert routed["route"] == "STILL_QUARANTINED"
    assert routed["retrieval_limitations"]


def test_long_queries_are_split_before_encoding_and_never_silently_truncated():
    model = HashEncoder()
    model.max_seq_length = 16
    index = semantic.build_text_semantic_index("P1", "The prevalence was 17%.", model)
    long_question = " ".join(f"qualifier{index}" for index in range(40))
    hits = semantic.rank_semantic_chunks(
        index, long_question, "The prevalence was 17%.", model
    )
    assert hits[0]["query_truncation_count"] == 0
    assert hits[0]["long_query_split_count"] > 0
    assert semantic._query_truncation_count([
        {"token_count": 15, "max_query_tokens": 14},
        {"token_count": 14, "max_query_tokens": 14},
    ]) == 1
