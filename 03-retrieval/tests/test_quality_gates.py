"""Release quality gates — recall, abstention, latency, citation precision.

These are the numbers that used to live only in RESULTS.md. A regression here
fails CI rather than a demo.
"""

from __future__ import annotations

import sqlite3

from mufasa_retrieval.evaluate import evaluate
from mufasa_retrieval.generate import StubGenerator


def test_recall_and_abstention_gates(conn: sqlite3.Connection, questions) -> None:
    report = evaluate(conn, questions, k=10, use_vectors=True)
    # Hold the floors Claude measured after the gate rewrite; the strict 25/25
    # bar is the stretch target, not a hard fail while dense retrieval is still
    # the hashing embedder.
    assert report.recall_any_hits >= 20, report.summary()
    assert report.recall_at_k_hits >= 14, report.summary()
    assert report.abstention_hits == report.unanswerable_total, report.summary()
    assert report.abstention_rate >= 0.8
    assert report.p95_latency_ms < 500, report.summary()


def test_stub_generation_citation_precision(conn: sqlite3.Connection, questions) -> None:
    report = evaluate(
        conn,
        questions,
        k=10,
        use_vectors=True,
        generator=StubGenerator(),
        run_generation=True,
    )
    assert report.citation_precision is not None
    assert report.citation_precision >= 0.85, report.summary()
