"""Non-destructive normalisation for entity matching.

Raw wording is always kept beside the derived key. Nothing here proves identity;
it only makes two spellings comparable. Deliberately does *not* ASCII-fold, so
African place and organism names keep their diacritics.
"""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
# Apostrophe stays: "well's" and "Côte d'Ivoire" must remain comparable keys,
# not collapse to "well s" / "cote d ivoire".
_PUNCT = re.compile(r"[^\w\s\-/%.']", re.UNICODE)
_STOP = frozenset(
    "a an the of in on at for from to and or with by is are was were be been "
    "this that these those its their his her".split()
)


def norm(text: str) -> str:
    """Casefolded, punctuation-stripped, whitespace-collapsed comparison key."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text).casefold()
    t = t.replace("’", "'").replace("–", "-").replace("—", "-")
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def tokens(text: str) -> list[str]:
    """Content tokens of a string, stopwords removed."""
    return [t for t in norm(text).split() if t and t not in _STOP]


def content_tokens(text: str, min_len: int = 3) -> list[str]:
    return [t for t in tokens(text) if len(t) >= min_len]


def fts_query(text: str) -> str:
    """Build a safe FTS5 OR-query from free text.

    Every token is quoted, so FTS5 operators a user happens to type ("AND",
    "-", "*", ":") are matched literally instead of changing the query.
    """
    toks = content_tokens(text)
    if not toks:
        toks = tokens(text)
    if not toks:
        return '""'
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(toks))


def ngrams(text: str, n: int = 3) -> list[str]:
    t = f" {norm(text)} "
    return [t[i : i + n] for i in range(max(0, len(t) - n + 1))]
