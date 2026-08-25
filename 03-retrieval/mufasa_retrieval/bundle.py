"""Turn ranked claims into the 6-10 record evidence bundle the model reads.

Short on purpose. Prompt reading dominates latency on a CPU, so ten
well-chosen records beat fifty average ones for accuracy *and* speed.

Licence tiers are enforced here, not in the model: tier 1 ships the quoted
sentence, tier 2 ships the finding with a citation and no source text, tier 3
never reaches a bundle at all.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from .gate import GateDecision
from .search import Claim, Hit, load_claims, study_family_count

MIN_RECORDS = 6
MAX_RECORDS = 10


@dataclass
class EvidenceRecord:
    tag: str
    claim_id: str
    text: str
    quote: str | None
    paper_id: str
    paper_title: str
    paper_year: int | None
    paper_journal: str | None
    paper_doi: str | None
    page: int | None
    section: str | None
    measurement: dict[str, Any]
    conditions: dict[str, Any]
    limitations: list[str]
    study_family_id: str | None
    facets: list[str]
    licence_tier: int
    quote_withheld: bool
    channels: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "claim_id": self.claim_id,
            "text": self.text,
            "quote": self.quote,
            "paper": {
                "paper_id": self.paper_id,
                "title": self.paper_title,
                "year": self.paper_year,
                "journal": self.paper_journal,
                "doi": self.paper_doi,
            },
            "page": self.page,
            "section": self.section,
            "measurement": self.measurement,
            "conditions": self.conditions,
            "limitations": self.limitations,
            "study_family_id": self.study_family_id,
            "facets": self.facets,
            "licence_tier": self.licence_tier,
            "quote_withheld": self.quote_withheld,
            "channels": self.channels,
        }


@dataclass
class EvidenceBundle:
    records: list[EvidenceRecord]
    study_families: int
    decision: GateDecision

    @property
    def tags(self) -> list[str]:
        return [r.tag for r in self.records]

    def by_tag(self, tag: str) -> EvidenceRecord | None:
        return next((r for r in self.records if r.tag == tag), None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "records": [r.as_dict() for r in self.records],
            "study_families": self.study_families,
            "decision": self.decision.as_dict(),
        }


def _record(tag: str, claim: Claim, channels: list[str]) -> EvidenceRecord:
    withheld = claim.licence_tier >= 2
    return EvidenceRecord(
        tag=tag,
        claim_id=claim.claim_id,
        text=claim.text,
        quote=None if withheld else claim.quote,
        paper_id=claim.paper_id,
        paper_title=claim.paper_title,
        paper_year=claim.paper_year,
        paper_journal=claim.paper_journal,
        paper_doi=claim.paper_doi,
        page=claim.page,
        section=claim.section,
        measurement=claim.measurement,
        conditions=claim.conditions,
        limitations=claim.limitations,
        study_family_id=claim.study_family_id,
        facets=claim.facets,
        licence_tier=claim.licence_tier,
        quote_withheld=withheld,
        channels=channels,
    )


def build_bundle(
    conn: sqlite3.Connection,
    hits: list[Hit],
    decision: GateDecision,
    *,
    max_records: int = MAX_RECORDS,
) -> EvidenceBundle:
    """Select the records that go to the model.

    When the gate says answerable, the bundle is the supporting claims in rank
    order. When it abstains, the bundle carries the nearest related evidence so
    the answer can point at it instead of stopping dead.
    """
    if decision.answerable:
        wanted = decision.supporting_claim_ids or [h.claim_id for h in hits]
    else:
        wanted = decision.nearest_claim_ids

    order = {h.claim_id: i for i, h in enumerate(hits)}
    channels = {h.claim_id: h.channels for h in hits}
    wanted = sorted(dict.fromkeys(wanted), key=lambda cid: order.get(cid, 10**6))[:max_records]

    claims = load_claims(conn, wanted)
    # Quotes stay off for tier 2+. Findings still ship: dropping them left the
    # gate saying "answerable" with an empty prompt, so every covered question
    # failed validation.
    selected = [claims[cid] for cid in wanted if cid in claims]

    records = [
        _record(f"E{i}", claim, channels.get(claim.claim_id, []))
        for i, claim in enumerate(selected, start=1)
    ]
    return EvidenceBundle(
        records=records,
        study_families=study_family_count(selected),
        decision=decision,
    )
