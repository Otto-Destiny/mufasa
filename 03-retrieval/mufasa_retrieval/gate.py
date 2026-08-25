"""The coverage gate — two axes, both required.

retrieval-v1 measured recall@10 = 25/25 and abstention 0/5, and the reason was
structural rather than a threshold that needed tuning: every unanswerable
question in the fixture is *topically adjacent* to real claims. Asked which
bacteria were in the Bosso samples, BM25 finds the corpus's physicochemical
measurements of those exact samples, scores them highly, and answers.

So the decision is a conjunction:

    entity axis    which corpus entities is the question about?
    property axis  what kind of fact is being asked for?

A claim supports an answer only if it satisfies both at once. When the entity
axis matches and the property axis does not, the system can say something much
more useful than "no results":

    The corpus has 4 physicochemical measurements for these Bosso samples,
    but no bacteriological findings.

Absence here is a statement about corpus v1, never about the literature.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from . import properties
from .search import Claim, Hit, load_claims, match_entities

#: Facets that are too generic to carry an abstention on their own. A question
#: asking only "where" or "what should be done" has not really constrained the
#: kind of fact it wants.
WEAK_FACETS = frozenset({"spatial_distribution", "recommendation_action", "method_protocol"})


@dataclass
class GateDecision:
    answerable: bool
    reason: str
    requested_facets: list[str] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    supporting_claim_ids: list[str] = field(default_factory=list)
    nearest_claim_ids: list[str] = field(default_factory=list)
    available_facets: list[str] = field(default_factory=list)
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "answerable": self.answerable,
            "reason": self.reason,
            "requested_facets": self.requested_facets,
            "matched_entities": self.matched_entities,
            "supporting_claim_ids": self.supporting_claim_ids,
            "nearest_claim_ids": self.nearest_claim_ids,
            "available_facets": self.available_facets,
            "message": self.message,
        }


def _corpus_sentence(conn: sqlite3.Connection) -> str:
    man = {r[0]: r[1] for r in conn.execute("SELECT key, value FROM manifest")}
    version = man.get("corpus_version", "corpus v1")
    papers = man.get("count_papers", "?")
    y0, y1 = man.get("year_min", ""), man.get("year_max", "")
    span = f", {y0}–{y1}" if y0 and y1 else ""
    return f"MUFASA {version} ({papers} papers{span})"


def _facet_summary(claims: list[Claim], exclude: set[str]) -> list[str]:
    seen: dict[str, int] = {}
    for c in claims:
        for f in c.facets:
            if f in exclude:
                continue
            seen[f] = seen.get(f, 0) + 1
    return [f for f, _ in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))]


def decide(
    conn: sqlite3.Connection,
    question: str,
    hits: list[Hit],
    *,
    consider: int = 10,
) -> GateDecision:
    corpus = _corpus_sentence(conn)
    requested = properties.question_facets(question)
    primary = properties.primary_question_facet(question, exclude=WEAK_FACETS)
    entities = [e["name"] for e in match_entities(conn, question)]
    candidate_ids = [h.claim_id for h in hits[:consider]]
    claims_by_id = load_claims(conn, candidate_ids)
    candidates = [claims_by_id[c] for c in candidate_ids if c in claims_by_id]

    if not candidates:
        return GateDecision(
            answerable=False,
            reason="no_candidates",
            requested_facets=requested,
            matched_entities=entities,
            message=f"No verified matching evidence in {corpus}.",
        )

    if primary is None:
        # The question did not constrain the kind of fact it wants, so the
        # property axis has no opinion and lexical relevance stands alone.
        return GateDecision(
            answerable=True,
            reason="no_property_constraint",
            requested_facets=requested,
            matched_entities=entities,
            supporting_claim_ids=candidate_ids,
        )

    supporting = [c for c in candidates if primary in c.facets]
    if supporting:
        return GateDecision(
            answerable=True,
            reason="both_axes_matched",
            requested_facets=requested,
            matched_entities=entities,
            supporting_claim_ids=[c.claim_id for c in supporting],
        )

    # Entity axis matched, property axis empty: the honest and useful case.
    available = _facet_summary(candidates, exclude=WEAK_FACETS | {primary})
    wanted = properties.label(primary)
    if available:
        have = properties.label(available[0])
        subject = f" for {entities[0]}" if entities else ""
        message = (
            f"No verified matching evidence in {corpus}. The corpus has {have}"
            f"{subject}, but no {wanted}."
        )
    else:
        message = f"No verified matching evidence in {corpus} for {wanted}."

    return GateDecision(
        answerable=False,
        reason="no_property_match",
        requested_facets=requested,
        matched_entities=entities,
        nearest_claim_ids=candidate_ids[:3],
        available_facets=available,
        message=message,
    )
