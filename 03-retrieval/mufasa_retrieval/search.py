"""Runtime plane: question -> ranked claims.

Three channels, merged by reciprocal-rank fusion:

  BM25        FTS5, compiled into SQLite itself. No extension binary, no
              network, it cannot fail to load on the judging laptop.
  vector      int8 brute-force scan of claim_vec. Same id space as everything
              else, so a dense hit is already a claim row.
  one hop     claims sharing an entity with a seed hit. One hop is a join, and
              one hop is all this design needs.

Channels are independent on purpose. If vectors are unavailable the other two
still answer, and the degraded state is reported rather than hidden.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .build import vector_matrix
from .embed import Embedder, dequantize, get_embedder
from .normalize import content_tokens, fts_query, norm

RRF_K = 60

#: Channel weights for the fusion, chosen by sweeping the 30-question fixture
#: with abstention accuracy held at 5/5 (see docs/retrieval-v2.md). A matched
#: corpus entity turns out to be a stronger signal than a shared word, so the
#: entity channel leads.
#:
#: `vector` is low because the default embedder is the deterministic hashing
#: backend, which cannot bridge paraphrase — "gap-filling" and "imputation"
#: share no character n-gram. Re-weight it after switching to the ONNX encoder;
#: that is the measurement that decides whether dense retrieval earns its place.
CHANNEL_WEIGHTS = {"bm25": 1.0, "entity": 1.2, "vector": 0.5, "graph": 0.4}


@dataclass
class Hit:
    claim_id: str
    score: float
    channels: list[str] = field(default_factory=list)
    ranks: dict[str, int] = field(default_factory=dict)
    via_entities: list[str] = field(default_factory=list)


@dataclass
class Claim:
    claim_id: str
    paper_id: str
    study_family_id: str | None
    text: str
    claim_type: str | None
    measurement: dict[str, Any]
    conditions: dict[str, Any]
    limitations: list[str]
    direction: str | None
    quote: str | None
    page: int | None
    section: str | None
    facets: list[str]
    entities: list[dict[str, str]]
    paper_title: str
    paper_year: int | None
    paper_journal: str | None
    paper_doi: str | None
    licence_tier: int

    @property
    def quotable(self) -> bool:
        """Tier 1 ships the sentence; tier 2 ships a citation; tier 3 neither."""
        return self.licence_tier == 1


@dataclass
class SearchResult:
    hits: list[Hit]
    channels_used: list[str]
    channels_degraded: list[str]
    aliases_used: list[str] = field(default_factory=list)


# -- channels --------------------------------------------------------------


def bm25_search(conn: sqlite3.Connection, question: str, limit: int = 50) -> list[tuple[str, float]]:
    q = fts_query(question)
    rows = conn.execute(
        """SELECT claim_id, bm25(claim_fts, 1.0, 0.6, 0.8) AS score
             FROM claim_fts WHERE claim_fts MATCH ?
            ORDER BY score LIMIT ?""",
        (q, limit),
    ).fetchall()
    # bm25() is more negative for better matches; flip so bigger is better.
    return [(r["claim_id"], -float(r["score"])) for r in rows]


def vector_search(
    conn: sqlite3.Connection, question: str, embedder: Embedder, limit: int = 50
) -> list[tuple[str, float]]:
    ids, mat, scale = vector_matrix(conn)
    if not ids:
        return []
    qv = embedder.encode([question])[0]
    sims = dequantize(mat, scale) @ qv.astype(np.float32)
    order = np.argsort(-sims)[:limit]
    return [(ids[i], float(sims[i])) for i in order]


def expand_one_hop(
    conn: sqlite3.Connection, seed_ids: list[str], limit: int = 40
) -> list[tuple[str, float, list[str]]]:
    """Claims sharing an entity with a seed, ranked by how many they share."""
    if not seed_ids:
        return []
    marks = ",".join("?" * len(seed_ids))
    rows = conn.execute(
        f"""SELECT ce2.claim_id            AS claim_id,
                   COUNT(DISTINCT e.name)  AS shared,
                   GROUP_CONCAT(DISTINCT e.name) AS names
              FROM claim_entity ce1
              JOIN entity e        ON e.entity_id = ce1.entity_id
              JOIN claim_entity ce2 ON ce2.entity_id = ce1.entity_id
             WHERE ce1.claim_id IN ({marks})
               AND ce2.claim_id NOT IN ({marks})
             GROUP BY ce2.claim_id
             ORDER BY shared DESC, claim_id
             LIMIT ?""",
        (*seed_ids, *seed_ids, limit),
    ).fetchall()
    return [(r["claim_id"], float(r["shared"]), (r["names"] or "").split(",")) for r in rows]


#: Alias → entity containment is only safe for place-like types. Resolving a
#: common noun like "borehole" onto every `borehole water` / `deep borehole`
#: row pulls topically adjacent papers into the shortlist and breaks the
#: property-axis abstention (Q-026: Bosso samples exist, bacteria do not).
_PLACE_TYPES = frozenset({
    "Place", "PlaceGroup", "City", "Region", "Country", "Location", "Site",
    "Community", "MonitoringStation", "UrbanArea",
})


def match_entities(conn: sqlite3.Connection, question: str) -> list[dict[str, Any]]:
    """Entity axis: which corpus entities is this question actually about?

    A multi-token entity matches only when all its content tokens are present,
    which keeps "urban residents in sub-Saharan Africa" from matching anything
    that merely says "urban".

    Aliases resolve onto real entity rows. Place names often live inside a
    comma-separated PlaceGroup ("… Bosso, Keteren Gwari …") rather than as a
    lone `Bosso` row, so an unresolved alias would leave the entity channel
    empty for those questions.
    """
    hay = norm(question)
    hay_tokens = set(content_tokens(question))
    entities = list(conn.execute("SELECT entity_id, name, type, norm_name FROM entity"))
    matched: list[dict[str, Any]] = []

    for row in entities:
        ent_tokens = content_tokens(row["norm_name"])
        if not ent_tokens:
            continue
        if _mentions(hay, row["norm_name"]) or set(ent_tokens).issubset(hay_tokens):
            matched.append(
                {"entity_id": row["entity_id"], "name": row["name"], "type": row["type"],
                 "specificity": len(ent_tokens)}
            )

    for row in conn.execute("SELECT norm_alias, entity_name FROM alias"):
        alias, canonical = row["norm_alias"], row["entity_name"]
        if not alias or not _mentions(hay, alias):
            continue
        canon_key = norm(canonical)
        resolved = False
        for ent in entities:
            if ent["norm_name"] == canon_key:
                ok = True
            elif ent["type"] in _PLACE_TYPES and _mentions(ent["norm_name"], canon_key):
                ok = True
            else:
                ok = False
            if not ok:
                continue
            matched.append(
                {"entity_id": ent["entity_id"], "name": ent["name"], "type": ent["type"],
                 "specificity": max(len(content_tokens(alias)),
                                    len(content_tokens(ent["norm_name"])))}
            )
            resolved = True
        if not resolved:
            matched.append(
                {"entity_id": None, "name": canonical, "type": "alias",
                 "specificity": len(content_tokens(alias))}
            )

    matched.sort(key=lambda m: (-m["specificity"], m["name"]))
    seen, unique = set(), []
    for m in matched:
        key = (m["entity_id"], m["name"].casefold())
        if key in seen:
            continue
        seen.add(key)
        unique.append(m)
    return unique


# -- fusion ----------------------------------------------------------------


def rrf_merge(channel_rankings: dict[str, list[str]], k: int = RRF_K) -> list[Hit]:
    """Weighted reciprocal-rank fusion. Score-free, so BM25 and cosine never
    need calibrating against each other."""
    acc: dict[str, Hit] = {}
    for channel, ids in channel_rankings.items():
        weight = CHANNEL_WEIGHTS.get(channel, 1.0)
        for rank, cid in enumerate(ids, start=1):
            hit = acc.setdefault(cid, Hit(claim_id=cid, score=0.0))
            hit.score += weight / (k + rank)
            hit.channels.append(channel)
            hit.ranks[channel] = rank
    return sorted(acc.values(), key=lambda h: (-h.score, h.claim_id))


def _mentions(haystack: str, phrase: str) -> bool:
    """Whole-word containment.

    Plain substring matching is actively dangerous with acronyms: `ERI` is
    inside `Nigeria`, and `VES` is inside `harvesting`, so a question about
    rainwater harvesting in northern Nigeria resolved to electrical resistivity
    imaging and vertical electrical sounding.
    """
    if not phrase:
        return False
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", haystack) is not None


#: A paper may take this many slots before the diversity penalty applies.
#: Several fixture questions legitimately need three or four findings from one
#: study, so the first few are free; only a monopoly is discouraged.
FREE_SLOTS_PER_PAPER = 3
DIVERSITY_DECAY = 0.85


def diversify(
    conn: sqlite3.Connection,
    hits: list[Hit],
    k: int,
    *,
    decay: float = DIVERSITY_DECAY,
    free_slots: int = FREE_SLOTS_PER_PAPER,
) -> list[Hit]:
    """Greedy re-rank that stops one paper monopolising the shortlist.

    Cross-paper questions are the ones the graph exists for, and they fail in a
    specific way: a paper dense in the question's vocabulary fills all ten slots
    while the connecting evidence sits in three other papers. Each additional
    claim from an already-represented paper is discounted, so breadth wins ties
    without a hard per-paper cap — a hard cap would break the questions that
    legitimately need three findings from one study.
    """
    if not hits:
        return []
    ids = [h.claim_id for h in hits]
    marks = ",".join("?" * len(ids))
    paper_of = {
        r["claim_id"]: r["paper_id"]
        for r in conn.execute(
            f"SELECT claim_id, paper_id FROM claim WHERE claim_id IN ({marks})", ids
        )
    }
    remaining, chosen, seen = list(hits), [], {}
    while remaining and len(chosen) < k:
        best_i, best_score = 0, float("-inf")
        for i, h in enumerate(remaining):
            paper = paper_of.get(h.claim_id)
            score = h.score * (decay ** max(0, seen.get(paper, 0) - free_slots + 1))
            if score > best_score:
                best_i, best_score = i, score
        pick = remaining.pop(best_i)
        paper = paper_of.get(pick.claim_id)
        seen[paper] = seen.get(paper, 0) + 1
        chosen.append(pick)
    return chosen


def expand_query(conn: sqlite3.Connection, question: str) -> tuple[str, list[str]]:
    """Rewrite a question with the canonical names its wording implies.

    Without this an English-only, jargon-free question never reaches the claim
    that answers it: "gap-filling method" and "imputation" share no token, and
    neither do "underground scan" and "electrical resistivity imaging". The
    alias list is the cheap offline way across that gap — no model, no index.
    """
    hay = norm(question)
    hits: list[str] = []
    for row in conn.execute("SELECT norm_alias, entity_name FROM alias ORDER BY norm_alias"):
        alias, canonical = row["norm_alias"], row["entity_name"]
        if not alias:
            continue
        if _mentions(hay, alias) and not _mentions(hay, norm(canonical)):
            hits.append(canonical)
    unique = list(dict.fromkeys(hits))
    return (question + " " + " ".join(unique)).strip() if unique else question, unique


def entity_search(
    conn: sqlite3.Connection, question: str, limit: int = 50
) -> list[tuple[str, float]]:
    """Claims about the entities this question names, ranked by how many match.

    The entity axis of the gate, reused as a retrieval channel. It is what
    carries the multi-paper questions, where the connecting concept is named in
    the question but the answering sentences share little vocabulary with it.
    """
    matched = match_entities(conn, question)
    ids = [m["entity_id"] for m in matched if m["entity_id"] is not None]
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"""SELECT ce.claim_id, COUNT(DISTINCT ce.entity_id) AS matched,
                   SUM(LENGTH(e.norm_name)) AS specificity
              FROM claim_entity ce JOIN entity e ON e.entity_id = ce.entity_id
             WHERE ce.entity_id IN ({marks})
             GROUP BY ce.claim_id
             ORDER BY matched DESC, specificity DESC, ce.claim_id
             LIMIT ?""",
        (*ids, limit),
    ).fetchall()
    return [(r["claim_id"], float(r["matched"])) for r in rows]


def search(
    conn: sqlite3.Connection,
    question: str,
    *,
    k: int = 10,
    pool: int = 50,
    embedder: Embedder | None = None,
    use_vectors: bool = True,
    use_graph: bool = True,
) -> SearchResult:
    rankings: dict[str, list[str]] = {}
    used, degraded = [], []

    expanded, aliases_used = expand_query(conn, question)

    bm = bm25_search(conn, expanded, limit=pool)
    rankings["bm25"] = [cid for cid, _ in bm]
    used.append("bm25")

    ent = entity_search(conn, expanded, limit=pool)
    if ent:
        rankings["entity"] = [cid for cid, _ in ent]
        used.append("entity")

    if use_vectors:
        try:
            emb = embedder or get_embedder()
            vec = vector_search(conn, expanded, emb, limit=pool)
            rankings["vector"] = [cid for cid, _ in vec]
            used.append("vector")
        except Exception as exc:  # noqa: BLE001 - a dead channel must not kill the query
            degraded.append(f"vector: {exc}")

    via: dict[str, list[str]] = {}
    if use_graph:
        seeds = [cid for cid, _ in bm[:8]]
        hop = expand_one_hop(conn, seeds, limit=pool)
        if hop:
            rankings["graph"] = [cid for cid, _, _ in hop]
            used.append("graph")
            via = {cid: names for cid, _, names in hop}

    hits = diversify(conn, rrf_merge(rankings), k)
    for h in hits:
        h.via_entities = via.get(h.claim_id, [])
    return SearchResult(
        hits=hits,
        channels_used=used,
        channels_degraded=degraded,
        aliases_used=aliases_used,
    )


# -- payload ---------------------------------------------------------------


def load_claims(conn: sqlite3.Connection, claim_ids: list[str]) -> dict[str, Claim]:
    if not claim_ids:
        return {}
    marks = ",".join("?" * len(claim_ids))
    rows = conn.execute(
        f"""SELECT c.*, s.quote, s.page, s.section,
                   p.title AS paper_title, p.year AS paper_year, p.journal AS paper_journal,
                   p.doi AS paper_doi, p.licence_tier AS licence_tier
              FROM claim c
              JOIN paper p ON p.paper_id = c.paper_id
         LEFT JOIN evidence_span s ON s.claim_id = c.claim_id
             WHERE c.claim_id IN ({marks})""",
        claim_ids,
    ).fetchall()

    facets: dict[str, list[str]] = {}
    for r in conn.execute(
        f"SELECT claim_id, facet FROM claim_facet WHERE claim_id IN ({marks}) ORDER BY facet",
        claim_ids,
    ):
        facets.setdefault(r["claim_id"], []).append(r["facet"])

    ents: dict[str, list[dict[str, str]]] = {}
    for r in conn.execute(
        f"""SELECT ce.claim_id, e.name, e.type, ce.role
              FROM claim_entity ce JOIN entity e ON e.entity_id = ce.entity_id
             WHERE ce.claim_id IN ({marks}) ORDER BY ce.role""",
        claim_ids,
    ):
        ents.setdefault(r["claim_id"], []).append(
            {"name": r["name"], "type": r["type"], "role": r["role"]}
        )

    out: dict[str, Claim] = {}
    for r in rows:
        cid = r["claim_id"]
        out[cid] = Claim(
            claim_id=cid,
            paper_id=r["paper_id"],
            study_family_id=r["study_family_id"],
            text=r["text"],
            claim_type=r["claim_type"],
            measurement=json.loads(r["measurement_json"]),
            conditions=json.loads(r["conditions_json"]),
            limitations=json.loads(r["limitations_json"]),
            direction=r["direction"],
            quote=r["quote"],
            page=r["page"],
            section=r["section"],
            facets=facets.get(cid, []),
            entities=ents.get(cid, []),
            paper_title=r["paper_title"],
            paper_year=r["paper_year"],
            paper_journal=r["paper_journal"],
            paper_doi=r["paper_doi"],
            licence_tier=r["licence_tier"],
        )
    return out


def study_family_count(claims: list[Claim]) -> int:
    """Separate experiments, not paper count. A thesis, a conference paper and a
    journal article about one experiment are one confirmation, not three."""
    return len({c.study_family_id or c.paper_id for c in claims})
