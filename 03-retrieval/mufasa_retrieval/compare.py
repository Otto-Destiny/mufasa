"""Compare Studies and the disagreement matrix.

Take the claims that share a subject entity and a facet, sort them into
supporting / conflicting / inconclusive, then name the condition that differs
between the groups:

    Four observations, three agree. The outlier differs only in ash burning
    temperature — 600 °C against 800 °C.

One query plus a grouping, and it produces the table a working researcher
actually wants. Study families rather than papers, so a thesis, a conference
paper and a journal article about one experiment count once.

`direction` is deliberately not trusted as the grouping key: only 40 of the
fixture's 112 claims carry it, and the values present are free text
(`well_highest`, `east-west`) rather than the four-value enum the drafted
architecture assumed. Agreement is derived from the measured values instead.
"""

from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass, field
from typing import Any

from .search import Claim, load_claims

#: Two values agree when they sit within this fraction of their mean.
AGREEMENT_TOLERANCE = 0.25


@dataclass
class ComparisonRow:
    claim_id: str
    paper_id: str
    study_family_id: str | None
    paper_title: str
    paper_year: int | None
    text: str
    value: float | None
    unit: str | None
    conditions: dict[str, Any]
    stance: str  # supporting | conflicting | inconclusive
    quote: str | None
    page: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "paper_id": self.paper_id,
            "study_family_id": self.study_family_id,
            "paper_title": self.paper_title,
            "paper_year": self.paper_year,
            "text": self.text,
            "value": self.value,
            "unit": self.unit,
            "conditions": self.conditions,
            "stance": self.stance,
            "quote": self.quote,
            "page": self.page,
        }


@dataclass
class DisagreementGroup:
    subject: str
    facet: str
    rows: list[ComparisonRow]
    study_families: int
    supporting: int
    conflicting: int
    inconclusive: int
    differing_conditions: list[str] = field(default_factory=list)
    narrative: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "facet": self.facet,
            "rows": [r.as_dict() for r in self.rows],
            "study_families": self.study_families,
            "counts": {
                "supporting": self.supporting,
                "conflicting": self.conflicting,
                "inconclusive": self.inconclusive,
            },
            "differing_conditions": self.differing_conditions,
            "narrative": self.narrative,
        }


def _scalar(measurement: dict[str, Any]) -> tuple[float | None, str | None]:
    if not isinstance(measurement, dict):
        return None, None
    unit = measurement.get("unit")
    for key in ("value", "average", "estimate", "coefficient", "share_pct", "mortality_pct"):
        v = measurement.get(key)
        if isinstance(v, (int, float)):
            return float(v), unit
    values = measurement.get("values")
    if isinstance(values, list) and values and all(isinstance(v, (int, float)) for v in values):
        return float(statistics.fmean(values)), unit
    lo, hi = measurement.get("min_value"), measurement.get("max_value")
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        return (float(lo) + float(hi)) / 2, unit
    return None, unit


def _stance(value: float | None, reference: float | None) -> str:
    if value is None or reference is None:
        return "inconclusive"
    scale = (abs(value) + abs(reference)) / 2 or 1.0
    return "supporting" if abs(value - reference) / scale <= AGREEMENT_TOLERANCE else "conflicting"


def _differing_conditions(rows: list[ComparisonRow]) -> list[str]:
    supporting = [r for r in rows if r.stance == "supporting"]
    conflicting = [r for r in rows if r.stance == "conflicting"]
    if not supporting or not conflicting:
        return []
    keys = {k for r in rows for k in r.conditions}
    differing = []
    for key in sorted(keys):
        a = {repr(r.conditions.get(key)) for r in supporting}
        b = {repr(r.conditions.get(key)) for r in conflicting}
        if a and b and not (a & b):
            differing.append(key)
    return differing


def _narrative(group: DisagreementGroup) -> str:
    n = len(group.rows)
    if group.conflicting == 0:
        return (
            f"{n} observation{'s' if n != 1 else ''} across "
            f"{group.study_families} study famil{'ies' if group.study_families != 1 else 'y'}; "
            f"no disagreement in the measured values."
        )
    base = (
        f"{n} observations, {group.supporting} agree. "
        f"{group.conflicting} disagree"
    )
    if group.differing_conditions:
        cond = group.differing_conditions[0]
        vals_a = {
            str(r.conditions.get(cond)) for r in group.rows if r.stance == "supporting"
        }
        vals_b = {
            str(r.conditions.get(cond)) for r in group.rows if r.stance == "conflicting"
        }
        return (
            f"{base}. The outlier{'s' if group.conflicting != 1 else ''} differ"
            f"{'' if group.conflicting != 1 else 's'} in {cond.replace('_', ' ')}: "
            f"{', '.join(sorted(vals_b))} against {', '.join(sorted(vals_a))}."
        )
    return f"{base}, and no recorded condition separates them."


def compare(
    conn: sqlite3.Connection,
    claim_ids: list[str],
    *,
    min_group: int = 2,
) -> list[DisagreementGroup]:
    """Group the given claims into comparable sets and classify agreement."""
    claims = load_claims(conn, claim_ids)
    buckets: dict[tuple[str, str], list[Claim]] = {}

    for claim in claims.values():
        subjects = [
            e["name"]
            for e in claim.entities
            if e["role"] in ("subject", "chemical", "organism", "contaminant", "resource",
                            "water_source", "model", "method", "analyte", "place")
        ] or [e["name"] for e in claim.entities[:1]]
        for facet in claim.facets or ["unclassified"]:
            for subject in subjects[:1]:
                buckets.setdefault((subject, facet), []).append(claim)

    groups: list[DisagreementGroup] = []
    for (subject, facet), members in sorted(buckets.items()):
        if len(members) < min_group:
            continue
        scalars = [(_scalar(c.measurement)) for c in members]
        numeric = [v for v, _ in scalars if v is not None]
        reference = statistics.median(numeric) if numeric else None

        rows = []
        for claim, (value, unit) in zip(members, scalars, strict=True):
            rows.append(
                ComparisonRow(
                    claim_id=claim.claim_id,
                    paper_id=claim.paper_id,
                    study_family_id=claim.study_family_id,
                    paper_title=claim.paper_title,
                    paper_year=claim.paper_year,
                    text=claim.text,
                    value=value,
                    unit=unit,
                    conditions=claim.conditions,
                    stance=_stance(value, reference),
                    quote=claim.quote if claim.licence_tier == 1 else None,
                    page=claim.page,
                )
            )
        group = DisagreementGroup(
            subject=subject,
            facet=facet,
            rows=sorted(rows, key=lambda r: (r.stance, r.claim_id)),
            study_families=len({r.study_family_id or r.paper_id for r in rows}),
            supporting=sum(1 for r in rows if r.stance == "supporting"),
            conflicting=sum(1 for r in rows if r.stance == "conflicting"),
            inconclusive=sum(1 for r in rows if r.stance == "inconclusive"),
        )
        group.differing_conditions = _differing_conditions(group.rows)
        group.narrative = _narrative(group)
        groups.append(group)

    groups.sort(key=lambda g: (-g.conflicting, -len(g.rows), g.subject))
    return groups
