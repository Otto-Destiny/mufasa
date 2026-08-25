"""Two-axis coverage gate.

The fixture's unanswerable questions are topically adjacent to real claims, so a
score threshold cannot abstain correctly. These cases pin the conjunction of
entity + property axes — especially Q-026, where Bosso samples exist but only
as physicochemical measurements.
"""

from __future__ import annotations

import sqlite3

from mufasa_retrieval.evaluate import load_questions
from mufasa_retrieval.gate import decide
from mufasa_retrieval.search import search


def _decision(conn: sqlite3.Connection, question: str):
    hits = search(conn, question, k=10).hits
    return decide(conn, question, hits), hits


def test_q026_entity_match_property_empty_abstains(conn: sqlite3.Connection, questions) -> None:
    q = next(q for q in questions if q["id"] == "Q-026")
    decision, _ = _decision(conn, q["question"])
    assert decision.answerable is False
    assert decision.reason == "no_property_match"
    assert decision.available_facets, "should name what the corpus does have"
    assert "bacter" in decision.message.lower() or "no" in decision.message.lower()


def test_unanswerable_questions_all_abstain(conn: sqlite3.Connection, questions) -> None:
    unanswerable = [q for q in questions if not q["answerable"]]
    assert len(unanswerable) >= 5
    for q in unanswerable:
        decision, _ = _decision(conn, q["question"])
        assert decision.answerable is False, (
            f"{q['id']} should abstain; gate said answerable ({decision.reason})"
        )


def test_answerable_questions_pass_the_gate(conn: sqlite3.Connection, questions) -> None:
    answerable = [q for q in questions if q["answerable"]]
    failures = []
    for q in answerable:
        decision, _ = _decision(conn, q["question"])
        if not decision.answerable:
            failures.append(f"{q['id']}:{decision.reason}")
    assert not failures, f"gate blocked answerable questions: {failures}"


def test_no_candidates_is_honest(conn: sqlite3.Connection) -> None:
    decision, hits = _decision(conn, "What is the melting point of unobtainium in Lagos?")
    assert hits == [] or decision.answerable in (True, False)
    if not hits:
        assert decision.reason == "no_candidates"
        assert "corpus" in decision.message.lower()
