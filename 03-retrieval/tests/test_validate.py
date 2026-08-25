"""Citation validator — invented tags, unsupported numbers, furniture masking."""

from __future__ import annotations

from mufasa_retrieval.bundle import EvidenceBundle, EvidenceRecord
from mufasa_retrieval.gate import GateDecision
from mufasa_retrieval.validate import mask_citation_furniture, soften, validate


def _bundle(*texts: str, answerable: bool = True) -> EvidenceBundle:
    records = []
    for i, text in enumerate(texts, start=1):
        records.append(
            EvidenceRecord(
                tag=f"E{i}",
                claim_id=f"C-{i}",
                text=text,
                quote=text,
                paper_id=f"P-{i}",
                paper_title="Synthetic",
                paper_year=2020,
                paper_journal=None,
                paper_doi=None,
                page=8,
                section="Results",
                measurement={"value": 31.2, "unit": "MPa"} if "31.2" in text else {},
                conditions={},
                limitations=[],
                study_family_id=None,
                facets=["compressive_strength"],
                licence_tier=1,
                quote_withheld=False,
            )
        )
    return EvidenceBundle(
        records=records,
        study_families=1,
        decision=GateDecision(answerable=answerable, reason="test"),
    )


def test_invented_tag_is_ungrounded() -> None:
    report = validate("The ash reached 31.2 MPa [E9].", _bundle("The ash reached 31.2 MPa."))
    assert report.verdict == "ungrounded"
    assert "E9" in report.invented_tags


def test_number_not_in_cited_evidence_is_flagged() -> None:
    report = validate(
        "The mix reached 99.9 MPa at 28 days [E1].",
        _bundle("The ash reached 31.2 MPa at 28 days."),
    )
    assert report.unsupported_claims
    assert report.verdict in ("partly_grounded", "ungrounded")


def test_supported_number_is_grounded() -> None:
    report = validate(
        "The ash reached 31.2 MPa at 28 days [E1].",
        _bundle("The ash reached 31.2 MPa at 28 days."),
    )
    assert report.verdict == "grounded"
    assert report.cited_tags == ["E1"]


def test_citation_header_digits_are_not_flagged() -> None:
    """retrieval-v1 bug: echoing 'P-G088' or 'page 8' looked like invented facts."""
    answer = "Source: P-G088 page 8 reports 31.2 MPa [E1]."
    masked = mask_citation_furniture(answer)
    assert "P-G088" not in masked
    report = validate(answer, _bundle("The ash reached 31.2 MPa."))
    assert report.verdict == "grounded"


def test_untagged_specific_number_fails() -> None:
    report = validate(
        "The ash reached 31.2 MPa at 28 days.",
        _bundle("The ash reached 31.2 MPa at 28 days."),
    )
    assert report.verdict == "ungrounded"
    assert report.sentences[0].untagged_specific


def test_soften_drops_failed_sentences() -> None:
    bundle = _bundle("The ash reached 31.2 MPa.")
    answer = "The ash reached 31.2 MPa [E1]. It also hit 99.9 MPa [E1]."
    report = validate(answer, bundle)
    repaired = soften(answer, report)
    assert "31.2" in repaired
    assert "99.9" not in repaired
