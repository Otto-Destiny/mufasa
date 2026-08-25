"""The property axis. These rules are what abstention rests on."""

from __future__ import annotations

import pytest
from mufasa_retrieval.properties import (
    FACET_LABELS,
    claim_facets,
    primary_question_facet,
    question_facets,
)


def test_claim_facets_from_measurement_keys() -> None:
    claim = {
        "text": "The 96-hour LC50 was 39.97 mg/L.",
        "claim_type": "experimental_result",
        "measurement": {"value": 39.97, "unit": "mg/L", "ci95_low": 35.95, "exposure_time": 96},
        "entities": [{"name": "fenthion", "type": "Pesticide", "role": "chemical"}],
    }
    facets = claim_facets(claim)
    assert "toxicity_dose_response" in facets
    assert "concentration_measurement" in facets


def test_claim_facets_from_entity_type() -> None:
    claim = {
        "text": "SARIMA outperformed the alternative on RMSE.",
        "claim_type": "model_comparison",
        "measurement": {"chosen_RSS": 1.0},
        "entities": [{"name": "SARIMA", "type": "StatisticalModel", "role": "model"}],
    }
    assert "statistical_model" in claim_facets(claim)


def test_question_facets_are_ordered_most_specific_first() -> None:
    """Order is load-bearing: a question naming benzene also matches
    concentration_measurement, and the corpus is full of benzene
    concentrations."""
    q = "What was the clinically confirmed cancer incidence rate caused by benzene in Ogale?"
    facets = question_facets(q)
    assert facets[0] == "clinical_incidence"
    assert "concentration_measurement" in facets
    assert primary_question_facet(q) == "clinical_incidence"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Which bacterial species were detected in the Bosso samples?", "microbiology"),
        ("What would it cost to build an engineered landfill?", "cost_economics"),
        ("Which smartphone sensor did the flood study deploy?", "technology_device"),
        ("How accurate was the forecast five years after deployment?", "longitudinal_followup"),
        ("What 96-hour LC50 was reported for African catfish?", "toxicity_dose_response"),
    ],
)
def test_primary_facet_of_each_unanswerable_shape(question: str, expected: str) -> None:
    assert primary_question_facet(question) == expected


def test_question_with_no_property_constraint() -> None:
    """No opinion is a valid outcome; the gate then lets relevance stand alone."""
    assert primary_question_facet("Tell me about the corpus", exclude=frozenset()) is None


def test_every_facet_has_a_human_label() -> None:
    """The abstention sentence prints these, so a missing label ships as a slug."""
    from mufasa_retrieval.properties import _CLAIM_TEXT_RULES, _QUESTION_RULES

    used = {f for f, _ in _CLAIM_TEXT_RULES} | {f for f, _ in _QUESTION_RULES}
    assert used <= set(FACET_LABELS)
