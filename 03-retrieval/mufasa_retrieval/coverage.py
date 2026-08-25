"""Coverage, not novelty.

A missing edge can mean the paper was never collected, the PDF was unavailable,
extraction missed the claim, or the work sits outside the corpus. So the system
reports what corpus v1 contains and says so precisely. It never reports that
nobody has studied something.

This module backs the coverage card beside every answer, the Coverage & Sources
view, and the Statistics screen — one query set, so the three cannot disagree.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from . import properties


@dataclass
class CoverageCard:
    corpus_version: str
    papers: int
    claims: int
    year_min: int | None
    year_max: int | None
    facet_vocabulary: str
    embedder: str

    def sentence(self) -> str:
        span = f" · {self.year_min}–{self.year_max}" if self.year_min and self.year_max else ""
        return f"MUFASA {self.corpus_version} · {self.papers} papers · {self.claims} findings{span}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "corpus_version": self.corpus_version,
            "papers": self.papers,
            "claims": self.claims,
            "year_min": self.year_min,
            "year_max": self.year_max,
            "facet_vocabulary": self.facet_vocabulary,
            "embedder": self.embedder,
            "sentence": self.sentence(),
        }


@dataclass
class CorpusStatistics:
    card: CoverageCard
    facets: list[dict[str, Any]] = field(default_factory=list)
    entity_types: list[dict[str, Any]] = field(default_factory=list)
    places: list[dict[str, Any]] = field(default_factory=list)
    licence_tiers: list[dict[str, Any]] = field(default_factory=list)
    papers_by_year: list[dict[str, Any]] = field(default_factory=list)
    withheld_papers: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "card": self.card.as_dict(),
            "facets": self.facets,
            "entity_types": self.entity_types,
            "places": self.places,
            "licence_tiers": self.licence_tiers,
            "papers_by_year": self.papers_by_year,
            "withheld_papers": self.withheld_papers,
        }


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except ValueError:
        return None


def card(conn: sqlite3.Connection) -> CoverageCard:
    man = {r[0]: r[1] for r in conn.execute("SELECT key, value FROM manifest")}
    return CoverageCard(
        corpus_version=man.get("corpus_version", "corpus_v1"),
        papers=_int(man.get("count_papers")) or 0,
        claims=_int(man.get("count_claims")) or 0,
        year_min=_int(man.get("year_min")),
        year_max=_int(man.get("year_max")),
        facet_vocabulary=man.get("facet_vocabulary", ""),
        embedder=man.get("embedder", ""),
    )


def statistics(conn: sqlite3.Connection) -> CorpusStatistics:
    facets = [
        {"facet": r[0], "label": properties.label(r[0]), "claims": r[1]}
        for r in conn.execute(
            "SELECT facet, COUNT(*) FROM claim_facet GROUP BY facet ORDER BY COUNT(*) DESC, facet"
        )
    ]
    entity_types = [
        {"type": r[0], "entities": r[1]}
        for r in conn.execute(
            "SELECT type, COUNT(*) FROM entity GROUP BY type ORDER BY COUNT(*) DESC, type LIMIT 20"
        )
    ]
    places = [
        {"name": r[0], "claims": r[1]}
        for r in conn.execute(
            """SELECT e.name, COUNT(DISTINCT ce.claim_id) AS n
                 FROM entity e JOIN claim_entity ce ON ce.entity_id = e.entity_id
                WHERE e.type IN ('Place', 'UrbanArea', 'River', 'WaterBody', 'PlaceGroup')
                GROUP BY e.name ORDER BY n DESC, e.name LIMIT 20"""
        )
    ]
    tiers = [
        {"tier": r[0], "papers": r[1]}
        for r in conn.execute(
            "SELECT licence_tier, COUNT(*) FROM paper GROUP BY licence_tier ORDER BY licence_tier"
        )
    ]
    by_year = [
        {"year": r[0], "papers": r[1]}
        for r in conn.execute(
            "SELECT year, COUNT(*) FROM paper WHERE year IS NOT NULL GROUP BY year ORDER BY year"
        )
    ]
    withheld = conn.execute("SELECT COUNT(*) FROM paper WHERE licence_tier >= 3").fetchone()[0]

    return CorpusStatistics(
        card=card(conn),
        facets=facets,
        entity_types=entity_types,
        places=places,
        licence_tiers=tiers,
        papers_by_year=by_year,
        withheld_papers=withheld,
    )


def papers(conn: sqlite3.Connection, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT p.*, (SELECT COUNT(*) FROM claim c WHERE c.paper_id = p.paper_id) AS claims
             FROM paper p ORDER BY p.paper_id LIMIT ? OFFSET ?""",
        (limit, offset),
    ).fetchall()
    return [
        {
            "paper_id": r["paper_id"],
            "title": r["title"],
            "authors": json.loads(r["authors_json"]),
            "year": r["year"],
            "journal": r["journal"],
            "doi": r["doi"],
            "study_type": r["study_type"],
            "licence": r["licence"],
            "licence_tier": r["licence_tier"],
            "geographic_scope": json.loads(r["geographic_scope_json"]),
            "topics": json.loads(r["topics_json"]),
            "claims": r["claims"],
        }
        for r in rows
    ]


def total_papers(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM paper").fetchone()[0]
