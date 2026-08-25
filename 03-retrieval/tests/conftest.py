"""Shared fixtures.

The 10-paper / 112-claim / 30-question set in `milestone1-test-data` is the
test corpus: small, versioned, and gold-labelled, so retrieval can be scored in
a loop rather than by eye. The database is built once per session into a temp
directory — never into the repo — and torn down with it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from mufasa_retrieval import build, connect
from mufasa_retrieval.evaluate import load_questions

DATA = Path(__file__).resolve().parents[1] / "milestone1-test-data"
ALIASES = Path(__file__).resolve().parents[1] / "mufasa_retrieval" / "aliases" / "flagship.yaml"


@pytest.fixture(scope="session")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("corpus") / "mufasa.db"
    build(
        claims_path=DATA / "claims.jsonl",
        papers_path=DATA / "papers.jsonl",
        db_path=out,
        corpus_version="corpus_test",
        aliases_path=ALIASES,
    )
    return out


@pytest.fixture
def conn(db_path: Path) -> sqlite3.Connection:
    connection = connect(db_path)
    yield connection
    connection.close()


@pytest.fixture(scope="session")
def questions() -> list[dict]:
    return load_questions(DATA / "questions.jsonl")


@pytest.fixture(scope="session")
def claims_path() -> Path:
    return DATA / "claims.jsonl"


@pytest.fixture(scope="session")
def papers_path() -> Path:
    return DATA / "papers.jsonl"
