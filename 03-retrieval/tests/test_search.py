from __future__ import annotations

import sqlite3

from mufasa_retrieval.search import (
    Hit,
    _mentions,
    bm25_search,
    diversify,
    entity_search,
    expand_query,
    load_claims,
    match_entities,
    rrf_merge,
    search,
    study_family_count,
)


def test_mentions_requires_word_boundaries() -> None:
    """The bug this replaced: `ERI` is inside `Nigeria` and `VES` is inside
    `harvesting`, so a rainwater-harvesting question in northern Nigeria
    resolved to electrical resistivity imaging."""
    assert _mentions("northern nigeria", "eri") is False
    assert _mentions("rainwater harvesting", "ves") is False
    assert _mentions("we used eri on site", "eri") is True
    assert _mentions("bosso samples", "bosso") is True


def test_alias_expansion_bridges_plain_language(conn: sqlite3.Connection) -> None:
    expanded, used = expand_query(conn, "Which gap-filling method was strongest?")
    assert "imputation" in used
    assert "imputation" in expanded.lower()


def test_alias_expansion_skips_already_present_canonical(conn: sqlite3.Connection) -> None:
    _, used = expand_query(conn, "which imputation method performed best")
    assert "imputation" not in used


def test_entity_channel_finds_claims_by_matched_entity(conn: sqlite3.Connection) -> None:
    hits = entity_search(conn, "What was measured in the Bosso borehole samples?")
    assert hits, "entity channel returned nothing for a question naming corpus entities"
    claims = load_claims(conn, [cid for cid, _ in hits[:5]])
    assert claims


def test_match_entities_prefers_specific_names(conn: sqlite3.Connection) -> None:
    matched = match_entities(conn, "borehole water in Bosso")
    assert matched
    assert matched[0]["specificity"] >= matched[-1]["specificity"]


def test_bm25_returns_higher_scores_for_better_matches(conn: sqlite3.Connection) -> None:
    rows = bm25_search(conn, "electrical conductivity rainwater borehole", limit=10)
    assert rows
    assert rows[0][1] >= rows[-1][1]


def test_rrf_merge_is_weighted_and_deterministic() -> None:
    rankings = {"bm25": ["a", "b", "c"], "entity": ["c", "a"]}
    first = rrf_merge(rankings)
    second = rrf_merge(rankings)
    assert [h.claim_id for h in first] == [h.claim_id for h in second]
    assert first[0].claim_id in ("a", "c")
    assert set(first[0].channels) <= {"bm25", "entity"}


def test_diversify_leaves_the_free_slots_alone(conn: sqlite3.Connection) -> None:
    """Several fixture questions legitimately need three findings from one
    study, so a hard per-paper cap would be wrong."""
    ids = [r[0] for r in conn.execute(
        "SELECT claim_id FROM claim WHERE paper_id = 'P-G149' ORDER BY claim_id LIMIT 6")]
    hits = [Hit(claim_id=c, score=1.0 - i * 0.01) for i, c in enumerate(ids)]
    kept = diversify(conn, hits, k=3)
    assert [h.claim_id for h in kept] == ids[:3]


def test_diversify_breaks_a_single_paper_monopoly(conn: sqlite3.Connection) -> None:
    one = [r[0] for r in conn.execute(
        "SELECT claim_id FROM claim WHERE paper_id = 'P-G149' ORDER BY claim_id LIMIT 8")]
    other = [r[0] for r in conn.execute(
        "SELECT claim_id FROM claim WHERE paper_id = 'P-G059' ORDER BY claim_id LIMIT 2")]
    hits = [Hit(claim_id=c, score=1.0) for c in one] + [Hit(claim_id=c, score=0.9) for c in other]
    kept = [h.claim_id for h in diversify(conn, hits, k=6)]
    assert any(c in kept for c in other), "one paper took every slot"


def test_search_reports_which_channels_ran(conn: sqlite3.Connection) -> None:
    result = search(conn, "What 96-hour LC50 was reported for African catfish?")
    assert "bm25" in result.channels_used
    assert not result.channels_degraded


def test_search_degrades_without_vectors(conn: sqlite3.Connection) -> None:
    """If one channel is unavailable the others must still answer."""
    result = search(conn, "borehole water quality in Bosso", use_vectors=False)
    assert result.hits
    assert "vector" not in result.channels_used


def test_load_claims_attaches_span_paper_and_facets(conn: sqlite3.Connection) -> None:
    result = search(conn, "96-hour LC50 fenthion African catfish")
    claims = load_claims(conn, [h.claim_id for h in result.hits])
    sample = next(iter(claims.values()))
    assert sample.paper_title
    assert sample.facets is not None
    assert sample.entities is not None


def test_study_family_count_counts_experiments_not_papers(conn: sqlite3.Connection) -> None:
    claims = list(load_claims(conn, [
        r[0] for r in conn.execute("SELECT claim_id FROM claim LIMIT 20")]).values())
    assert study_family_count(claims) <= len(claims)
