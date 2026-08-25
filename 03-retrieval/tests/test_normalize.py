from __future__ import annotations

import pytest
from mufasa_retrieval.normalize import content_tokens, fts_query, norm, tokens


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Rice Husk Ash", "rice husk ash"),
        ("  multiple   spaces ", "multiple spaces"),
        ("Bosso, Minna (Niger State)", "bosso minna niger state"),
        ("39.97 mg/L", "39.97 mg/l"),
        ("well’s", "well's"),
    ],
)
def test_norm_is_predictable(raw: str, expected: str) -> None:
    assert norm(raw) == expected


def test_norm_keeps_diacritics() -> None:
    """African place, personal and species names are not ASCII-folded: folding
    them silently merges distinct entities and mangles the display name."""
    assert "é" in norm("Ségou")
    assert "ô" in norm("Côte d'Ivoire")


def test_stopwords_dropped_from_content_tokens() -> None:
    assert "the" not in content_tokens("the quality of the borehole water")
    assert "borehole" in content_tokens("the quality of the borehole water")


def test_fts_query_quotes_every_token() -> None:
    """FTS5 operators a user types must be matched literally, not executed."""
    q = fts_query('borehole AND NOT "water" * : -depth')
    assert q.count('"') % 2 == 0
    assert " OR " in q
    assert "AND" not in q.replace('"and"', "")


def test_fts_query_never_returns_empty() -> None:
    assert fts_query("") == '""'
    assert fts_query("of the") != ""


def test_tokens_deduplicate_in_query() -> None:
    q = fts_query("water water water quality")
    assert q.count('"water"') == 1
