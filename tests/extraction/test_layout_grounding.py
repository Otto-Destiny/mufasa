import contextlib
import hashlib
import io
import json
import math
import re
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def extraction_contract():
    notebook_path = (
        Path(__file__).resolve().parents[2]
        / "01-data-engineering"
        / "data-extraction"
        / "llm-claim-extraction.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"hashlib": hashlib, "json": json, "math": math, "re": re}

    def clean_text(value):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        value = str(value).strip()
        return "" if value.lower() in {"nan", "none", "null"} else value

    namespace["clean_text"] = clean_text
    with contextlib.redirect_stdout(io.StringIO()):
        for cell_index in (2, 4, 5):
            exec("".join(notebook["cells"][cell_index]["source"]), namespace)
    return namespace


def _context_payload(raw_model_quote):
    return {
        "task_id": "T1",
        "paper_profile": {
            "coverage_complete": True,
            "language": "en",
            "key_contribution": "test fixture",
            "is_real_science": True,
            "is_africa_relevant": True,
            "mufasa_domain": "AGR",
            "discipline": "VETERINARY_SCIENCE",
            "discipline_secondary": [],
            "missing_content": None,
        },
        "study_contexts": [
            {
                "local_id": "C1",
                "label": "test",
                "study_design": "",
                "population_text": (
                    "postpubertal bulls, postpubertal heifers, breeding bulls and cows"
                ),
                "period_text": "",
                "sample_size_text": "",
                "conditions": [],
                "entities": [
                    {
                        "source_mention_local_id": "M1",
                        "source_evidence_local_id": "E1",
                        "provenance_scope": "OWNER_EVIDENCE",
                        "role": "POPULATION",
                        "surface_text": "breeding bulls",
                        "atom_text": "breeding bulls",
                        "entity_type": "POPULATION",
                        "identity_scope": "STUDY_INSTANCE",
                        "instance_local_id": "P1",
                        "qualifiers": [],
                        "aliases": [],
                    }
                ],
                "evidence": [
                    {
                        "local_id": "E1",
                        "source_kind": "TEXT",
                        "source_label": "",
                        "page": 3,
                        "section": "",
                        "quote": raw_model_quote,
                    }
                ],
            }
        ],
    }


def test_unique_pdf_line_wrap_is_restored_before_strict_validation(
    extraction_contract,
):
    raw = (
        "All the postpubertal bulls, postpubertal heifers, breed- ing bulls "
        "and cows were sampled."
    )
    model = raw.replace("breed- ing", "breeding")
    task = {
        "task_kind": "CONTEXT",
        "task_id": "T1",
        "coverage_complete": True,
        "page_texts": {3: raw},
        "allowed_spans": {3: [(0, len(raw))]},
    }

    aligned, repairs = extraction_contract["align_payload_grounding"](
        _context_payload(model), task
    )
    context = aligned["study_contexts"][0]

    assert context["evidence"][0]["quote"] == raw
    assert "breed- ing bulls" in context["population_text"]
    assert context["entities"][0]["surface_text"] == "breed- ing bulls"
    assert context["entities"][0]["atom_text"] == "breeding bulls"
    assert len(repairs) == 3
    extraction_contract["validate_payload"](aligned, task)

    second, second_repairs = extraction_contract["align_payload_grounding"](
        aligned, task
    )
    assert second == aligned
    assert second_repairs == []


def test_alignment_refuses_ambiguity_semantic_change_case_change_and_block_crossing(
    extraction_contract,
):
    unique_layout_span = extraction_contract["unique_layout_span"]
    assert unique_layout_span("breed- ing x breed- ing", "breeding") is None
    assert unique_layout_span("breed- ing bulls", "breeding cows") is None
    assert unique_layout_span("Breed- ing bulls", "breeding bulls") is None
    assert unique_layout_span(
        "alpha XX beta", "alpha beta", [(0, 5), (9, 13)]
    ) is None


def test_soft_hyphen_is_recoverable_only_as_the_original_raw_slice(
    extraction_contract,
):
    raw = "breed\u00ading bulls"
    span = extraction_contract["unique_layout_span"](raw, "breeding bulls")
    assert span is not None
    assert raw[span[0] : span[1]] == raw
