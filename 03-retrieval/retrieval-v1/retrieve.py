"""
Retrieval v1: BM25 (FTS5) seed search, then a one-hop expansion through
shared entities, matching the architecture doc's runtime shape
(retrieval-architecture.md: "BM25 and vector search in parallel -> ONE-hop
graph expansion -> merge, deduplicate, rank"). Vectors are not implemented
yet -- see Task 7 notes on whether they're needed at Gate 1.
"""
import os
import re
import sqlite3

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "retrieval.sqlite")

# sqlite's bm25(): lower (more negative) score = more relevant.
EXPANSION_PENALTY = 2.0


def connect():
    return sqlite3.connect(DB_PATH)


def _fts_query(question):
    tokens = re.findall(r"[A-Za-z0-9']+", question)
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


def retrieve(con, question, k=10, seed_n=15):
    """Returns (ranked_list, best_bm25_score). ranked_list is
    [(claim_id, score), ...], best score across seed hits only (used for
    abstention -- expansion hits never make a question look more answerable
    than its own text match did)."""
    match = _fts_query(question)
    if not match:
        return [], None

    cur = con.cursor()
    cur.execute(
        """
        SELECT claim_id, bm25(claim_fts) AS score
        FROM claim_fts
        WHERE claim_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (match, seed_n),
    )
    seeds = cur.fetchall()
    if not seeds:
        return [], None

    best_score = seeds[0][1]
    worst_seed_score = seeds[-1][1]
    scores = {cid: sc for cid, sc in seeds}
    seed_ids = list(scores.keys())

    placeholders = ",".join("?" * len(seed_ids))
    cur.execute(
        f"""
        SELECT DISTINCT ce2.claim_id
        FROM claim_entity ce1
        JOIN claim_entity ce2
          ON ce1.entity_name = ce2.entity_name AND ce1.entity_type = ce2.entity_type
        WHERE ce1.claim_id IN ({placeholders})
          AND ce2.claim_id NOT IN ({placeholders})
        """,
        seed_ids + seed_ids,
    )
    for (cid,) in cur.fetchall():
        scores.setdefault(cid, worst_seed_score + EXPANSION_PENALTY)

    ranked = sorted(scores.items(), key=lambda kv: kv[1])[:k]
    return ranked, best_score


def get_claim(con, claim_id):
    cur = con.cursor()
    cur.execute(
        "SELECT id, paper_id, text, quote, page, section FROM claim WHERE id = ?",
        (claim_id,),
    )
    return cur.fetchone()
