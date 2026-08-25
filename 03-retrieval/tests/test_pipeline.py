"""Pipeline: a weak model must not wipe a covered answer."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from mufasa_retrieval.pipeline import answer


@dataclass
class UngroundedGenerator:
    name: str = "ungrounded-test"

    def generate(self, prompt: str, *, max_tokens: int = 400, temperature: float = 0.0) -> str:
        return "Turbidity was 999 NTU across every well."


def test_weak_model_falls_back_to_cited_evidence(conn: sqlite3.Connection) -> None:
    result = answer(
        conn,
        "What was the turbidity of borehole water?",
        UngroundedGenerator(),
        k=10,
    )
    assert result.bundle.decision.answerable
    assert result.verdict == "grounded"
    assert result.cited_tags
    assert "could not produce an answer" not in result.answer.lower()
