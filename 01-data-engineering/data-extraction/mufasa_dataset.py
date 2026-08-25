"""Deterministic training-data salvage and support routing for MUFASA.

The sampling notebook imports this module; the production builder must call the
same public router before it is allowed to publish. Nothing here calls an LLM,
embedding API, or vector database.

Provenance recovery, support verification, and curriculum assignment are kept
separate. A narrow quote is widened inside the same paper before a target is
rejected. Verified targets route to OPEN_AS_IS or OPEN_WIDENED; targets for
which no compact supporting bundle can be proven route to QUARANTINE_UNVERIFIED.
Closed-book selection happens only after paper verification.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


TABLES = (
    "training_pairs", "evidence_spans", "extraction_status", "paper_profiles",
    "study_contexts", "african_innovation",
)


def clean(value: Any) -> str:
    """Strip strings without converting nulls to the literal ``nan``."""
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"none", "nan", "null"} else text


SPAN_FIELDS = (
    "evidence_id", "paper_id", "owner_kind", "owner_id", "local_id",
    "source_kind", "source_label", "page", "section", "quote", "char_start",
    "char_end",
)


def _span_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    span = {key: value.get(key) for key in SPAN_FIELDS if key in value}
    span["quote"] = clean(value.get("quote"))
    return span


def _canonical_page(value: Any) -> str:
    """Normalize Parquet/JSON page scalars without merging different pages."""
    text = clean(value)
    if re.fullmatch(r"\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def _span_key(span: Mapping[str, Any]) -> tuple[str, ...]:
    # Source-table and raw-checkpoint recovery often describe the same quote
    # with slightly different metadata. One copy is enough in the prompt; the
    # same text on another page remains a distinct span.
    return (
        clean(span.get("paper_id")),
        _canonical_page(span.get("page")),
        clean(span.get("quote")),
    )


def dedupe_evidence(spans: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    output: list[dict[str, Any]] = []
    for raw in spans:
        if not isinstance(raw, Mapping) or not clean(raw.get("quote")):
            continue
        span = _span_dict(raw)
        key = _span_key(span)
        if key not in seen:
            seen.add(key)
            output.append(span)
    return output


def evidence_bundles(spans: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Preserve every TRAINING-owned evidence span, grouped by owner ID."""
    if spans.empty:
        return {}
    training = spans[spans["owner_kind"].astype(str).eq("TRAINING")]
    columns = [column for column in SPAN_FIELDS if column in training.columns]
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    # A single vector-filtered tuple pass is much faster than constructing one
    # DataFrame and list of record dictionaries per owner.
    for values in training[columns].itertuples(index=False, name=None):
        raw = dict(zip(columns, values, strict=True))
        owner = clean(raw.get("owner_id"))
        if not owner or not clean(raw.get("quote")):
            continue
        span = _span_dict(raw)
        key = _span_key(span)
        if key not in seen[owner]:
            seen[owner].add(key)
            output[owner].append(span)
    return dict(output)


def load_tables(extraction: str | Path) -> dict[str, Any]:
    extraction = Path(extraction)
    data: dict[str, Any] = {
        name: pd.read_parquet(extraction / f"{name}.parquet") for name in TABLES
    }
    data["training_evidence"] = evidence_bundles(data["evidence_spans"])
    # Compatibility only. New callers use training_evidence, never this first row.
    training = data["evidence_spans"]
    training = training[training["owner_kind"].astype(str).eq("TRAINING")]
    data["training_spans"] = training.drop_duplicates("owner_id").set_index("owner_id")
    return data


# --------------------------------------------------- provenance recovery

def _as_evidence_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        if clean(value.get("quote")):
            return [_span_dict(value)]
        found: list[dict[str, Any]] = []
        for child in value.values():
            found.extend(_as_evidence_list(child))
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for child in value:
            found.extend(_as_evidence_list(child))
        return found
    if isinstance(value, str) and value.strip():
        return [{"quote": value.strip()}]
    return []


def evidence_of_all(item: Mapping[str, Any], kind: str = "") -> tuple[list[dict[str, Any]], list[str]]:
    """Read every evidence shape observed in raw training payloads."""
    del kind
    spans: list[dict[str, Any]] = []
    shapes: list[str] = []
    direct = _as_evidence_list(item.get("evidence"))
    if direct:
        spans.extend(direct)
        value = item.get("evidence")
        shapes.append("nested dict" if isinstance(value, Mapping) else
                      "list" if isinstance(value, list) else "bare string")
    if clean(item.get("quote")):
        spans.append({key: item.get(key) for key in (
            "quote", "page", "section", "source_kind", "source_label",
        )})
        shapes.append("flattened onto the pair")
    for alternate in (
        "positive_evidence", "chosen_evidence", "source", "provenance",
        "citation", "span",
    ):
        found = _as_evidence_list(item.get(alternate))
        if found:
            spans.extend(found)
            shapes.append(alternate)
    return dedupe_evidence(spans), list(dict.fromkeys(shapes))


def evidence_of(item: Mapping[str, Any], kind: str = "") -> tuple[dict[str, Any] | None, str]:
    """Compatibility wrapper returning the first recovered span."""
    spans, shapes = evidence_of_all(item, kind)
    return (spans[0], shapes[0]) if spans else (None, "")


def recover_evidence_bundles(
    raw_dir: str | Path, papers: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return ``pair_id -> {spans, shapes}`` without overwrite/first-span loss."""
    raw_dir = Path(raw_dir)
    wanted = set(papers) if papers is not None else None
    output: dict[str, dict[str, Any]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        if wanted is not None and path.stem not in wanted:
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        payload = (record.get("tasks", {}).get("training") or {}).get("payload") or {}
        if not isinstance(payload, Mapping):
            continue
        paper_id = clean(record.get("paper_id")) or path.stem
        for kind in ("factual", "reasoning", "reranker", "preference"):
            for index, item in enumerate(payload.get(kind) or []):
                if not isinstance(item, Mapping):
                    continue
                local = clean(item.get("local_id")) or f"{kind[:1]}{index}"
                pair_id = f"{paper_id}:{kind}:{local}"
                spans, shapes = evidence_of_all(item, kind)
                if not spans:
                    continue
                for span in spans:
                    span.setdefault("paper_id", paper_id)
                slot = output.setdefault(pair_id, {"spans": [], "shapes": []})
                slot["spans"].extend(spans)
                slot["shapes"].extend(shapes)
    for value in output.values():
        value["spans"] = dedupe_evidence(value["spans"])
        value["shapes"] = list(dict.fromkeys(value["shapes"]))
    return output


def recover_evidence(raw_dir: str | Path, papers: Iterable[str] | None = None) -> dict[str, tuple[dict[str, Any], str]]:
    """Compatibility wrapper for the original one-span API."""
    recovered = recover_evidence_bundles(raw_dir, papers)
    return {
        key: (value["spans"][0], value["shapes"][0] if value["shapes"] else "")
        for key, value in recovered.items() if value["spans"]
    }


def combined_evidence(
    pair_id: str,
    table_bundles: Mapping[str, Sequence[Mapping[str, Any]]],
    recovered: Mapping[str, Any],
) -> list[dict[str, Any]]:
    spans = list(table_bundles.get(pair_id, []))
    raw = recovered.get(pair_id)
    if isinstance(raw, Mapping):
        spans.extend(raw.get("spans") or [])
    elif isinstance(raw, tuple) and raw:
        spans.append(raw[0])
    return dedupe_evidence(spans)


# --------------------------------------------------------- language/structure

WRONG_SCRIPT = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u052f]")
CORRUPT_TEXT = re.compile(
    r"\ufffd|\u00e2\u20ac|\u00c3[\u0080-\u00ff]|\u00c2(?=[\u00b0\u00b5])"
)
REPEATED_ELLIPSIS = re.compile(r"(?:\.{3}|\u2026).*(?:\.{3}|\u2026)", re.S)


def wrong_script(text: Any, threshold: float = 0.10) -> bool:
    value = clean(text)
    hits = WRONG_SCRIPT.findall(value)
    return bool(hits) and len(hits) / max(len(value), 1) >= threshold


def strip_strays(text: Any) -> str:
    return WRONG_SCRIPT.sub("", clean(text)).strip()


def assistant_turn(row: Any) -> str:
    pair_type = clean(getattr(row, "pair_type", "")).upper()
    if pair_type == "PREFERENCE":
        return clean(getattr(row, "chosen", ""))
    answer = clean(getattr(row, "answer", ""))
    reasoning = clean(getattr(row, "reasoning", ""))
    return f"{reasoning}\n\nAnswer: {answer}" if pair_type == "REASONING" and reasoning else answer


def discard_reasons(
    row: Any,
    complete_papers: set[str],
    not_real: set[str],
    not_africa: set[str],
) -> list[str]:
    """Hard structural gates only; semantic support is routed later."""
    reasons: list[str] = []
    pair_type = clean(getattr(row, "pair_type", "")).upper()
    question = clean(getattr(row, "question", ""))
    target = assistant_turn(row)
    if pair_type not in {"FACTUAL", "REASONING", "PREFERENCE", "RERANKER"}:
        reasons.append("unknown pair type")
    if not question:
        reasons.append("blank question")
    if pair_type in {"FACTUAL", "REASONING"} and not clean(getattr(row, "answer", "")):
        reasons.append("blank answer")
    if pair_type == "PREFERENCE":
        chosen = clean(getattr(row, "chosen", ""))
        rejected = clean(getattr(row, "rejected", ""))
        if not chosen:
            reasons.append("blank chosen")
        if not rejected:
            reasons.append("blank rejected")
        if chosen and rejected and skeleton(chosen) == skeleton(rejected):
            reasons.append("chosen and rejected are identical")
    if pair_type == "RERANKER" and not clean(getattr(row, "positive_quote", "")):
        reasons.append("blank positive quote")
    if pair_type == "RERANKER" and not clean(getattr(row, "hard_negative_quote", "")):
        reasons.append("blank hard negative quote")
    if pair_type == "RERANKER":
        positive = clean(getattr(row, "positive_quote", ""))
        negative = clean(getattr(row, "hard_negative_quote", ""))
        if positive and negative and skeleton(positive) == skeleton(negative):
            reasons.append("positive and hard negative are identical")
    rejected = clean(getattr(row, "rejected", "")) if pair_type == "PREFERENCE" else ""
    if wrong_script(question) or wrong_script(target) or wrong_script(rejected):
        reasons.append("wrong language")
    if any(CORRUPT_TEXT.search(value) for value in (question, target, rejected) if value):
        reasons.append("text encoding corruption")
    if REPEATED_ELLIPSIS.search(target):
        reasons.append("placeholder ellipsis in target")
    if clean(getattr(row, "paper_id", "")) not in complete_papers:
        reasons.append("task was truncated")
    if row.paper_id in not_real:
        reasons.append("judged not real science")
    if row.paper_id in not_africa:
        reasons.append("judged not africa relevant")
    return reasons


def failed_verdicts(profiles_frame: pd.DataFrame) -> tuple[set[str], set[str]]:
    frame = profiles_frame.drop_duplicates("paper_id").set_index("paper_id")

    def explicitly_false(column: str) -> set[str]:
        if column not in frame.columns:
            return set()
        values = frame[column].astype("boolean")
        return set(frame[values.notna() & (values == False)].index)  # noqa: E712

    return explicitly_false("is_real_science"), explicitly_false("is_africa_relevant")


def _row_signature(row: pd.Series) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(column), re.sub(r"\s+", " ", clean(value)))
        for column, value in row.items() if column != "pair_id"
    )


def resolve_duplicates(
    pairs: pd.DataFrame,
    complete_papers: set[str] | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse exact copies; drop whole unresolved same-ID conflicts.

    Length is never used as a proxy for correctness. The final two arguments
    remain accepted for compatibility with the old build notebook.
    """
    del complete_papers, evidence
    duplicated = pairs["pair_id"].duplicated(keep=False)
    if not duplicated.any():
        return pairs.copy(), pairs.iloc[0:0].copy()
    remove: set[Any] = set()
    for _, group in pairs[duplicated].groupby("pair_id", sort=False):
        signatures: dict[tuple[tuple[str, str], ...], list[Any]] = defaultdict(list)
        for index, row in group.iterrows():
            signatures[_row_signature(row)].append(index)
        if len(signatures) == 1:
            remove.update(list(group.index)[1:])
        else:
            remove.update(group.index)
    return (
        pairs.loc[~pairs.index.isin(remove)].copy(),
        pairs.loc[pairs.index.isin(remove)].copy(),
    )


# ----------------------------------------------------------- source indexing

_PAGE_MARKER = re.compile(r"<!--\s*MUFASA_PDF_PAGE:\s*(\d+)\s*-->")
_HEADING = re.compile(r"(?m)^#{1,6}\s+(.+?)\s*$")
_REFERENCES = re.compile(r"(?im)^(?:#{1,6}\s*)?(?:references|bibliography|works\s+cited)\s*$")
_PARAGRAPH = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.S)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def paper_text(paper_id: str, markdown_dir: str | Path, cache: dict[str, str]) -> str:
    if paper_id not in cache:
        path = Path(markdown_dir) / f"{paper_id}.md"
        cache[paper_id] = path.read_text(encoding="utf-8") if path.is_file() else ""
    return cache[paper_id]


def _latest(items: Sequence[tuple[int, Any]], position: int, default: Any) -> Any:
    value = default
    for offset, candidate in items:
        if offset > position:
            break
        value = candidate
    return value


def _split_block(text: str, start: int, max_chars: int) -> list[tuple[int, int]]:
    if len(text) <= max_chars:
        return [(start, start + len(text))]
    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        hard = min(len(text), cursor + max_chars)
        if hard < len(text):
            ends = [m.end() for m in _SENTENCE_END.finditer(text, cursor, hard)]
            end = ends[-1] if ends and ends[-1] > cursor + max_chars // 3 else hard
        else:
            end = len(text)
        left, right = cursor, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if right > left:
            spans.append((start + left, start + right))
        cursor = max(end, cursor + 1)
    return spans


def chunk_markdown(
    paper_id: str, text: str, max_chars: int = 1800, min_chars: int = 40,
) -> dict[str, Any]:
    """Build bounded exact-source chunks with page/section/raw offsets."""
    reference = _REFERENCES.search(text)
    body_end = reference.start() if reference else len(text)
    body = text[:body_end]
    pages = [(m.start(), int(m.group(1))) for m in _PAGE_MARKER.finditer(body)]
    headings = [(m.start(), clean(m.group(1))) for m in _HEADING.finditer(body)]
    chunks: list[dict[str, Any]] = []
    for paragraph in _PARAGRAPH.finditer(body):
        raw = paragraph.group(0)
        if not clean(raw):
            continue
        for start, end in _split_block(raw, paragraph.start(), max_chars):
            quote = text[start:end]
            if len(clean(quote)) < min_chars:
                continue
            chunks.append({
                "paper_id": paper_id,
                "quote": quote,
                "page": _latest(pages, start, None),
                "section": _latest(headings, start, ""),
                "source_kind": "TABLE" if quote.lstrip().startswith("|") else "TEXT",
                "source_label": "same-paper lexical recovery",
                "char_start": start,
                "char_end": end,
            })
    document_frequency: Counter[str] = Counter()
    for chunk in chunks:
        chunk["_terms"] = Counter(content_terms(chunk["quote"]))
        chunk["_numbers"] = quantitative_mentions(chunk["quote"])
        document_frequency.update(set(chunk["_terms"]))
    return {
        "paper_id": paper_id,
        "text": text,
        "body_end": body_end,
        "chunks": chunks,
        "document_frequency": document_frequency,
    }


def paper_index(
    paper_id: str, markdown_dir: str | Path, cache: dict[str, Any], max_chars: int = 1800,
) -> dict[str, Any]:
    if paper_id not in cache:
        path = Path(markdown_dir) / f"{paper_id}.md"
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        cache[paper_id] = chunk_markdown(paper_id, text, max_chars=max_chars)
    return cache[paper_id]


def _layout_projection(text: str) -> tuple[str, list[int], list[int]]:
    """Collapse layout only, retaining a map back to exact source offsets."""
    out: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\u00ad":
            index += 1
            continue
        if char == "-" and index > 0 and text[index - 1].isalpha() and index + 1 < len(text) and text[index + 1].isspace():
            look = index + 1
            while look < len(text) and text[look].isspace():
                look += 1
            if look < len(text) and text[look].isalpha():
                index = look
                continue
        if char.isspace():
            begin = index
            while index < len(text) and text[index].isspace():
                index += 1
            if out and out[-1] != " ":
                out.append(" ")
                starts.append(begin)
                ends.append(index)
            continue
        for piece in unicodedata.normalize("NFC", char).casefold():
            out.append(piece)
            starts.append(index)
            ends.append(index + 1)
        index += 1
    return "".join(out).strip(), starts, ends


def _page_ranges(text: str) -> dict[int, tuple[int, int]]:
    markers = list(_PAGE_MARKER.finditer(text))
    return {
        int(marker.group(1)): (
            marker.end(), markers[pos + 1].start() if pos + 1 < len(markers) else len(text),
        )
        for pos, marker in enumerate(markers)
    }


def align_evidence_span(span: Mapping[str, Any], index: Mapping[str, Any]) -> dict[str, Any] | None:
    """Rewrite a quote to exact paper bytes after unique layout-only alignment."""
    quote = clean(span.get("quote"))
    text = str(index.get("text") or "")
    if not quote or not text:
        return None
    try:
        page = int(float(clean(span.get("page")))) if clean(span.get("page")) else None
    except ValueError:
        page = None
    allowed_start, allowed_end = _page_ranges(text).get(
        page, (0, int(index.get("body_end") or len(text))),
    )
    allowed = text[allowed_start:allowed_end]
    exact = [m.start() for m in re.finditer(re.escape(quote), allowed)]
    if exact:
        start = allowed_start + exact[0]
        end = start + len(quote)
    else:
        source_key, starts, ends = _layout_projection(allowed)
        query_key, _, _ = _layout_projection(quote)
        found = [m.start() for m in re.finditer(re.escape(query_key), source_key)] if query_key else []
        if len(found) != 1:
            return None
        left, right = found[0], found[0] + len(query_key) - 1
        if right >= len(ends):
            return None
        start = allowed_start + starts[left]
        end = allowed_start + ends[right]
    aligned = _span_dict(span)
    aligned.update({
        "paper_id": index.get("paper_id"), "quote": text[start:end],
        "char_start": start, "char_end": end,
        "page": page if page is not None else aligned.get("page"),
        "alignment": "EXACT_SOURCE",
    })
    return aligned


def align_evidence_bundle(
    spans: Sequence[Mapping[str, Any]], index: Mapping[str, Any],
) -> list[dict[str, Any]]:
    aligned = [hit for span in spans if (hit := align_evidence_span(span, index))]
    return dedupe_evidence(aligned)


# --------------------------------------------------------- support semantics

NUMBER_TOKEN = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\w|\.\d)"
)
UNIT_AFTER = re.compile(
    r"^\s*(?P<unit>%(?!\w)|(?:percentage\s+points?|°\s*[CF]|degrees?\s+[CF]|"
    r"µg(?:/mL)?|ug(?:/mL)?|mg(?:/[A-Za-z]+)?|kg(?:/[A-Za-z]+)?|"
    r"g(?:/[A-Za-z]+)?|mL|L|km|cm|mm|m|ha|ppm|ppb|cfu(?:/mL|/g)?|"
    r"years?|months?|weeks?|days?|hours?|hrs?|minutes?|mins?|seconds?|s)\b)",
    re.I,
)
UNIT = re.compile(NUMBER_TOKEN.pattern + UNIT_AFTER.pattern[1:], re.I)
WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’-]{1,}")
CAPITAL_WORD = r"(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]{2,}|[A-Z]{2,})"
MULTIWORD_PROPER = re.compile(rf"\b{CAPITAL_WORD}(?:\s+{CAPITAL_WORD})+\b")
ACRONYM = re.compile(r"\b[A-Z]{3,}\b")
GENERIC_PROPER_PREFIXES = {
    "A", "After", "An", "Answer", "Because", "Evidence", "Figure", "However", "In",
    "Introducing", "Question", "Since", "Table", "The", "There", "These",
    "This", "Those", "Under", "When", "While", "Out",
}

STOPWORDS = {
    "a", "an", "and", "or", "is", "of", "to", "about", "after", "again",
    "against", "also", "among", "answer", "are",
    "because", "been", "before", "being", "between", "both", "could", "did",
    "does", "during", "each", "evidence", "for", "from", "had", "has", "have",
    "into", "its", "may", "more", "most", "only", "other", "paper", "reported",
    "research", "result", "results", "showed", "study", "than", "that", "the",
    "their", "there", "these", "they", "this", "those", "through", "under",
    "using", "very", "was", "were", "what", "when", "where", "which", "while",
    "with", "would", "yes", "no", "according", "found", "finding",
}
GENERIC_QUANT_TERMS = {
    "sample", "participant", "total", "enroll", "value", "rate", "prevalence",
    "mean", "median", "record", "measure", "include", "subject", "respondent",
    "estimate", "year", "level", "account", "observe", "occur", "report",
}
TERM_ALIASES = {
    "increased": "increase", "increases": "increase", "increasing": "increase",
    "rose": "increase", "risen": "increase", "higher": "increase", "elevated": "increase",
    "decreased": "decrease", "decreases": "decrease", "decreasing": "decrease",
    "declined": "decrease", "reduced": "decrease", "lower": "decrease",
    "associated": "association", "associations": "association",
    "correlated": "correlation", "correlations": "correlation",
    "caused": "cause", "causes": "cause", "causing": "cause", "resulted": "cause",
    "participants": "participant", "samples": "sample", "farmers": "farmer",
    "children": "child", "women": "woman", "men": "man",
    "included": "enroll", "include": "enroll", "enrolled": "enroll",
    "enrolling": "enroll",
}


def skeleton(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean(text).casefold())


def _term(word: str) -> str:
    value = unicodedata.normalize("NFC", word).casefold().strip("'’- ")
    value = TERM_ALIASES.get(value, value)
    if len(value) > 5 and value.endswith("ies"):
        value = value[:-3] + "y"
    elif len(value) > 5 and value.endswith("ing"):
        value = value[:-3]
    elif len(value) > 4 and value.endswith("ed"):
        value = value[:-2]
    elif len(value) > 4 and value.endswith("s") and not value.endswith("ss"):
        value = value[:-1]
    return TERM_ALIASES.get(value, value)


def content_terms(text: Any) -> list[str]:
    return [
        term for word in WORD.findall(clean(text))
        if (term := _term(word)) and len(term) >= 3 and term not in STOPWORDS
    ]


def _decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None


def _unit(raw: str) -> str:
    value = re.sub(r"\s+", "", clean(raw).casefold())
    value = value.replace("degrees", "°").replace("degree", "°")
    value = value.replace("percentagepoints", "percentagepoint").replace("ug", "µg")
    aliases = {
        "hrs": "hour", "hr": "hour", "hours": "hour", "mins": "minute",
        "minutes": "minute", "secs": "second", "seconds": "second",
    }
    return aliases.get(value, value.rstrip("s"))


def quantitative_mentions(text: Any) -> list[dict[str, Any]]:
    value = clean(text)
    output: list[dict[str, Any]] = []
    for match in NUMBER_TOKEN.finditer(value):
        decimal = _decimal(match.group(0))
        if decimal is None:
            continue
        unit_match = UNIT_AFTER.match(value[match.end():match.end() + 32])
        output.append({
            "raw": match.group(0), "value": decimal,
            "unit": _unit(unit_match.group("unit")) if unit_match else "",
            "start": match.start(),
            "end": match.end() + (unit_match.end() if unit_match else 0),
        })
    return output


STANDALONE_UNIT = re.compile(
    r"(?<![A-Za-z])(%|percentage\s+points?|°\s*[CF]|µg(?:/mL)?|ug(?:/mL)?|"
    r"mg(?:/[A-Za-z]+)?|kg(?:/[A-Za-z]+)?|g(?:/[A-Za-z]+)?|mL|km|cm|mm|"
    r"ha|ppm|ppb|cfu(?:/mL|/g)?)(?![A-Za-z])",
    re.I,
)


def units_in_text(text: Any) -> set[str]:
    return {_unit(match.group(1)) for match in STANDALONE_UNIT.finditer(clean(text))}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}|\nAnswer:\s*", text)
    return [clean(part.removeprefix("Answer:")) for part in parts if clean(part)]


def _near_terms(text: str, start: int, end: int, width: int = 100) -> set[str]:
    return set(content_terms(text[max(0, start - width):min(len(text), end + width)]))


def _source_unit_at(text: str, start: int, end: int) -> str:
    """Return the claim-bearing sentence or table header+matched row."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end < 0 else line_end
    line = text[line_start:line_end]
    if "|" in line:
        lines = text[:line_end].splitlines()
        current = len(lines) - 1
        group_start = current
        while group_start > 0 and "|" in lines[group_start - 1]:
            group_start -= 1
        header = next(
            (candidate for candidate in lines[group_start:current + 1]
             if "|" in candidate and not re.fullmatch(r"[\s|:\-]+", candidate)),
            "",
        )
        # Bind a bare table number to its own column header/unit and preceding
        # row labels, not to unrelated columns or neighbouring rows.
        column = line[:max(0, start - line_start)].count("|")
        row_cells = line.split("|")
        header_cells = header.split("|") if header else []
        if 0 < column < len(row_cells):
            row_context = " | ".join(
                clean(cell) for cell in row_cells[1:column + 1] if clean(cell)
            )
            header_context = (
                clean(header_cells[column]) if column < len(header_cells) else ""
            )
            return " | ".join(
                piece for piece in (header_context, row_context) if piece
            )
        return f"{header}\n{line}" if header and header != line else line
    left_candidates = [
        text.rfind(mark, 0, start) for mark in (". ", "! ", "? ", "\n\n")
    ]
    unit_start = max(left_candidates) + (2 if max(left_candidates) >= 0 else 0)
    right_candidates = [
        position for mark in (". ", "! ", "? ", "\n\n")
        if (position := text.find(mark, end)) >= 0
    ]
    unit_end = min(right_candidates) + 1 if right_candidates else min(len(text), end + 300)
    return text[max(0, unit_start):unit_end]


def _target_clause_at(text: str, start: int, end: int) -> str:
    left = max(text.rfind(mark, 0, start) for mark in (";", "\n", ".", "!", "?"))
    right_candidates = [
        position for mark in (";", "\n", ".", "!", "?")
        if (position := text.find(mark, end)) >= 0
    ]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1:right]


def _proper_terms(text: str) -> list[str]:
    """Only multi-word names and acronyms; sentence-initial words are ignored."""
    return list(dict.fromkeys([
        *(match.group(0) for match in MULTIWORD_PROPER.finditer(text)
          if match.group(0).split()[0] not in GENERIC_PROPER_PREFIXES),
        *(match.group(0) for match in ACRONYM.finditer(text)),
    ]))


QUESTION_WORDS = {
    "What", "Which", "Who", "Whom", "Whose", "Where", "When", "Why", "How",
    "Does", "Did", "Do", "Is", "Are", "Was", "Were", "Can", "Could", "Would",
    "In", "At", "From", "For", "Of", "The", "A", "An", "After", "Out",
    "Table", "Figure", "Equation", "Result", "Results", "January", "February",
    "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December",
}


def _question_anchors(text: str) -> list[str]:
    """Named query anchors used to bind an answer to the correct entity."""
    proper_matches = list(MULTIWORD_PROPER.finditer(text))
    anchors = [
        match.group(0) for match in proper_matches
        if match.group(0).split()[0] not in GENERIC_PROPER_PREFIXES
    ]
    for match in re.finditer(CAPITAL_WORD, text):
        token = match.group(0)
        inside_longer_name = any(
            proper.start() <= match.start() and match.end() <= proper.end()
            for proper in proper_matches
        )
        if not inside_longer_name and token not in QUESTION_WORDS and token not in anchors:
            anchors.append(token)
    return anchors


def _anchor_present(anchor: str, text: str) -> bool:
    """Boundary-safe name lookup: Niger must not match Nigeria."""
    modifier = re.fullmatch(r"(.+?)[-\u2010-\u2015](positive|negative)", clean(anchor), re.I)
    if modifier:
        base, state = modifier.groups()
        # Scientific shorthand such as HBsAg-positive is commonly rendered as
        # "positive to HBsAg". Require both bounded components in one source
        # unit, without weakening ordinary multi-word place matching.
        return any(
            _anchor_present(base, unit) and _anchor_present(state, unit)
            for unit in (_sentences(clean(text)) or [clean(text)])
        )
    words = WORD.findall(unicodedata.normalize("NFC", clean(anchor)).casefold())
    if not words:
        return False
    pattern = r"(?<!\w)" + r"(?:[\s\W_]+)".join(re.escape(word) for word in words) + r"(?!\w)"
    return re.search(pattern, unicodedata.normalize("NFC", clean(text)).casefold()) is not None


def _direction_conflicts(target: str, evidence: str) -> list[str]:
    conflicts: list[str] = []
    evidence_sentences = _sentences(evidence) or [evidence]
    for target_sentence in _sentences(target) or [target]:
        wanted = set(content_terms(target_sentence))
        if not wanted:
            continue
        shown_sentence = max(
            evidence_sentences,
            key=lambda sentence: len(wanted & set(content_terms(sentence))),
        )
        shown = set(content_terms(shown_sentence))
        if "increase" in wanted and "increase" not in shown and "decrease" in shown:
            conflicts.append("increase/decrease conflict")
        if "decrease" in wanted and "decrease" not in shown and "increase" in shown:
            conflicts.append("decrease/increase conflict")
        target_negated = bool(re.search(
            r"\b(?:no|not|never|without|didn't|did\s+not)\b", target_sentence, re.I,
        ))
        evidence_negated = bool(re.search(
            r"\b(?:no|not|never|without|didn't|did\s+not)\b", shown_sentence, re.I,
        ))
        if target_negated != evidence_negated:
            conflicts.append(
                "negation absent from evidence" if target_negated
                else "evidence negates the positive target"
            )
        target_causal = bool(re.search(
            r"\b(?:caus(?:e|ed|es|ing)|led\s+to|resulted\s+in|because)\b",
            target_sentence, re.I,
        ))
        evidence_causal = bool(re.search(
            r"\b(?:caus(?:e|ed|es|ing)|led\s+to|resulted\s+in|because)\b",
            shown_sentence, re.I,
        ))
        association_only = bool(re.search(
            r"\b(?:associat|correlat|linked\s+with)\w*\b", shown_sentence, re.I,
        ))
        if target_causal and not evidence_causal:
            conflicts.append(
                "causal claim supported only by association" if association_only
                else "causal relation absent from evidence"
            )
    return list(dict.fromkeys(conflicts))


def _arithmetic_support(
    missing: Mapping[str, Any], evidence_numbers: Sequence[Mapping[str, Any]],
    question: str, evidence_text: str,
) -> dict[str, Any] | None:
    """Permit only explicit subtraction and percent-change questions."""
    wanted = missing["value"]
    wanted_unit = missing.get("unit") or ""
    operation_words = {
        "difference", "percentage", "point", "change", "increase", "decrease",
        "baseline", "follow", "many", "more",
    }
    metrics = set(content_terms(question)) - operation_words
    if not metrics:
        return None

    def operands_bound(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        left_terms = _near_terms(evidence_text, left["start"], left["end"])
        right_terms = _near_terms(evidence_text, right["start"], right["end"])
        return bool(metrics & left_terms) and bool(metrics & right_terms)

    if re.search(r"\b(?:difference|how\s+many\s+more|percentage\s+points?|change)\b", question, re.I):
        for left, right in itertools.permutations(evidence_numbers, 2):
            if left["unit"] and right["unit"] and left["unit"] != right["unit"]:
                continue
            compatible = (
                left["unit"] == right["unit"] == "%"
                if wanted_unit == "percentagepoint"
                else not wanted_unit or wanted_unit in {left["unit"], right["unit"]}
            )
            if (
                compatible
                and operands_bound(left, right)
                and abs((right["value"] - left["value"]) - wanted) <= Decimal("0.01")
            ):
                return {"operation": "subtract", "operands": [str(left["value"]), str(right["value"])], "result": str(wanted)}
    if re.search(r"\bpercent(?:age)?\s+(?:increase|decrease|change)\b", question, re.I):
        for left, right in itertools.permutations(evidence_numbers, 2):
            if left["value"] == 0:
                continue
            result = (right["value"] - left["value"]) / abs(left["value"]) * 100
            if (
                left["unit"] == right["unit"] == "%"
                and operands_bound(left, right)
                and abs(result - wanted) <= Decimal("0.15")
            ):
                return {"operation": "percent_change", "operands": [str(left["value"]), str(right["value"])], "result": str(wanted)}
    return None


def support_report(
    assistant_text: Any,
    evidence: Sequence[Mapping[str, Any]] | str,
    question: Any = "",
    coverage_threshold: float = 0.55,
) -> dict[str, Any]:
    """Check exact quantities and sentence-level prose against a source bundle."""
    bundle = ([{"quote": evidence}] if isinstance(evidence, str) and clean(evidence)
              else dedupe_evidence(evidence) if not isinstance(evidence, str) else [])
    evidence_text = "\n\n".join(clean(span.get("quote")) for span in bundle)
    target = clean(assistant_text)
    asked = clean(question)
    report: dict[str, Any] = {
        "supported": False, "missing_numbers": [], "unit_mismatches": [],
        "unbound_numbers": [], "missing_proper_terms": [], "sentence_coverages": [],
        "missing_question_anchors": [], "unbound_question_anchors": [],
        "direction_conflicts": [], "arithmetic": [], "relevance": 0.0, "reason": "",
    }
    if not evidence_text:
        report["reason"] = "no exact source evidence"
        return report

    evidence_numbers = quantitative_mentions(evidence_text)
    question_anchors = _question_anchors(asked)
    target_numbers = quantitative_mentions(target)
    for number in target_numbers:
        matches = [item for item in evidence_numbers if item["value"] == number["value"]]
        if number["unit"]:
            unit_matches = [item for item in matches if item["unit"] == number["unit"]]
            inherited = [
                item for item in matches
                if not item["unit"]
                and number["unit"] in units_in_text(
                    _source_unit_at(evidence_text, item["start"], item["end"])
                )
            ]
            if matches and not unit_matches and not inherited:
                report["unit_mismatches"].append(f"{number['raw']} {number['unit']}")
                continue
            matches = [*unit_matches, *inherited]
        if not matches:
            derived = _arithmetic_support(number, evidence_numbers, asked, evidence_text)
            if derived:
                report["arithmetic"].append(derived)
            else:
                suffix = f" {number['unit']}" if number["unit"] else ""
                report["missing_numbers"].append(number["raw"] + suffix)
            continue
        local_target = _near_terms(target, number["start"], number["end"])
        local_target |= set(content_terms(asked))
        local_target -= STOPWORDS
        viable_contexts = [
            _source_unit_at(evidence_text, match["start"], match["end"])
            for match in matches
        ]
        if local_target and not any(
            local_target & set(content_terms(context)) for context in viable_contexts
        ):
            report["unbound_numbers"].append(number["raw"])
        local_anchors = _question_anchors(
            _target_clause_at(target, number["start"], number["end"])
        )
        needed_anchors = local_anchors or question_anchors
        if needed_anchors and not any(
            any(_anchor_present(anchor, context) for anchor in needed_anchors)
            for context in viable_contexts
        ):
            report["unbound_question_anchors"].extend(needed_anchors)
    report["unbound_question_anchors"] = sorted(set(report["unbound_question_anchors"]))

    evidence_terms = set(content_terms(evidence_text))
    evidence_units = [set(content_terms(unit)) for unit in _sentences(evidence_text)]
    if not evidence_units:
        evidence_units = [evidence_terms]
    question_terms = set(content_terms(asked))
    for sentence in _sentences(target):
        terms = content_terms(sentence)
        if not terms:
            continue
        scored_terms = (
            [term for term in terms if term not in GENERIC_QUANT_TERMS]
            if len(terms) <= 4 else terms
        ) or terms
        matched_count = max(
            (sum(term in unit for term in scored_terms) for unit in evidence_units),
            default=0,
        )
        coverage = matched_count / len(scored_terms)
        # A fully bound quantity plus one generic paraphrased noun/verb (for
        # example "sample included" versus "participants enrolled") should not
        # fail an otherwise exact quantitative label.
        if (
            quantitative_mentions(sentence)
            and not report["missing_numbers"]
            and not report["unit_mismatches"]
            and not report["unbound_numbers"]
            and any(set(terms) - unit <= GENERIC_QUANT_TERMS for unit in evidence_units)
        ):
            coverage = 1.0
        if report["arithmetic"] and set(terms) - evidence_terms <= question_terms:
            coverage = 1.0
        required = 1.0 if len(terms) <= 4 else coverage_threshold
        report["sentence_coverages"].append({
            "sentence": sentence, "coverage": round(coverage, 3),
            "required": required,
            "missing_terms": sorted(set(terms) - evidence_terms)[:12],
        })
    if question_terms:
        report["relevance"] = round(len(question_terms & evidence_terms) / len(question_terms), 3)
    else:
        report["relevance"] = 1.0

    for proper in _proper_terms(target):
        if not _anchor_present(proper, evidence_text):
            report["missing_proper_terms"].append(proper)
    for proper in _question_anchors(asked):
        if not _anchor_present(proper, evidence_text):
            report["missing_question_anchors"].append(proper)
    report["direction_conflicts"] = _direction_conflicts(target, evidence_text)
    low = [
        item for item in report["sentence_coverages"]
        if item["coverage"] < item["required"]
    ]
    relevant = bool(report["arithmetic"]) or report["relevance"] >= 0.18 or bool(
        set(_proper_terms(asked)) & set(_proper_terms(evidence_text))
    )
    report["supported"] = not any((
        report["missing_numbers"], report["unit_mismatches"], report["unbound_numbers"],
        report["missing_proper_terms"], report["missing_question_anchors"],
        report["unbound_question_anchors"], report["direction_conflicts"], low,
        not relevant,
    ))
    if report["supported"]:
        report["reason"] = "supported"
    elif report["missing_numbers"]:
        report["reason"] = "figure absent from evidence"
    elif report["unit_mismatches"]:
        report["reason"] = "unit mismatch"
    elif report["unbound_numbers"]:
        report["reason"] = "figure is not bound to the asked metric"
    elif report["unbound_question_anchors"]:
        report["reason"] = "figure is not bound to the asked entity"
    elif report["direction_conflicts"]:
        report["reason"] = report["direction_conflicts"][0]
    elif report["missing_proper_terms"]:
        report["reason"] = "named term absent from evidence"
    elif report["missing_question_anchors"]:
        report["reason"] = "asked entity is not bound to the evidence"
    elif low:
        report["reason"] = "one or more target sentences lack source support"
    else:
        report["reason"] = "evidence is not relevant to the question"
    return report


def unsupported(assistant_text: Any, quote: Any, question: Any = "") -> set[str]:
    """Compatibility view over the richer support report."""
    report = support_report(assistant_text, quote, question)
    missing = set().union(
        report["missing_numbers"], report["unit_mismatches"], report["unbound_numbers"],
        report["missing_proper_terms"], report["missing_question_anchors"],
        report["unbound_question_anchors"], report["direction_conflicts"],
    )
    for sentence in report["sentence_coverages"]:
        if sentence["coverage"] < sentence["required"]:
            missing.add("unsupported prose: " + sentence["sentence"][:80])
    if not report["supported"] and not missing:
        missing.add(report["reason"])
    return missing


# ------------------------------------------------------- retrieval and routing

def _chunk_score(
    chunk: Mapping[str, Any], query_terms: Counter[str],
    numbers: Sequence[Mapping[str, Any]], total: int, frequency: Mapping[str, int],
) -> float:
    quote = clean(chunk.get("quote"))
    terms = chunk.get("_terms") or Counter(content_terms(quote))
    score = 0.0
    for term, count in query_terms.items():
        if term in terms:
            score += min(count, terms[term]) * (math.log((total + 1) / (frequency.get(term, 0) + 1)) + 1)
    shown_numbers = chunk.get("_numbers") or quantitative_mentions(quote)
    for wanted in numbers:
        if any(wanted["value"] == got["value"] and (
            not wanted["unit"] or wanted["unit"] == got["unit"]
        ) for got in shown_numbers):
            score += 12.0
    return score / (1.0 + math.log1p(max(len(quote), 1) / 800))


def _bundle_chars(bundle: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(clean(span.get("quote"))) for span in bundle)


def retrieve_bundle(
    row: Any,
    index: Mapping[str, Any],
    initial: Sequence[Mapping[str, Any]] = (),
    target_text: str | None = None,
    max_spans: int = 3,
    max_chars: int = 8000,
    candidate_limit: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Find a compact supporting bundle within one paper, without API calls.

    The answer/reasoning is used only to rank source chunks offline. Returned
    prompts contain the retrieved source spans, never this locator query.
    """
    target = target_text if target_text is not None else assistant_turn(row)
    question = clean(getattr(row, "question", ""))
    chunks = list(index.get("chunks") or [])
    if not chunks:
        return [], support_report(target, [], question)

    # Expand each supplied short quote to the bounded source chunk containing it.
    initial_chunks: list[dict[str, Any]] = []
    for span in initial:
        start, end = span.get("char_start"), span.get("char_end")
        if isinstance(start, int) and isinstance(end, int):
            overlap = [chunk for chunk in chunks if (
                int(chunk["char_start"]) <= start < int(chunk["char_end"])
                or int(chunk["char_start"]) < end <= int(chunk["char_end"])
            )]
            initial_chunks.extend(overlap[:1])

    query_terms = Counter(content_terms(f"{question}\n{target}"))
    numbers = quantitative_mentions(target)
    document_frequency = index.get("document_frequency") or Counter()
    ranked_with_scores = sorted(
        (
            _chunk_score(chunk, query_terms, numbers, len(chunks), document_frequency),
            int(chunk["char_start"]), chunk,
        )
        for chunk in chunks
    )
    ranked_with_scores.reverse()
    ranked = [chunk for score, _, chunk in ranked_with_scores[:candidate_limit] if score > 0]
    candidates = dedupe_evidence([*initial_chunks, *ranked])

    def quality(report: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            -len(report["missing_numbers"]),
            -len(report["unit_mismatches"]),
            -len(report["unbound_numbers"]),
            -len(report["unbound_question_anchors"]),
            -len(report["direction_conflicts"]),
            sum(item["coverage"] for item in report["sentence_coverages"]),
            report["relevance"],
        )

    best_bundle: list[dict[str, Any]] = []
    best_report = support_report(target, [], question)
    singles: list[tuple[tuple[Any, ...], int]] = []

    def consider(choice: Sequence[Mapping[str, Any]]) -> bool:
        nonlocal best_bundle, best_report
        bundle = dedupe_evidence(choice)
        if not bundle or _bundle_chars(bundle) > max_chars:
            return False
        report = support_report(target, bundle, question)
        if report["supported"]:
            best_bundle, best_report = bundle, report
            return True
        if not best_bundle or quality(report) > quality(best_report):
            best_bundle, best_report = bundle, report
        return False

    # Evaluate every candidate alone, then combine a small deterministic beam
    # of the best candidates. This preserves multi-span recovery while avoiding
    # the old exhaustive 41 support checks for every pair.
    for index, candidate in enumerate(candidates):
        report = support_report(target, [candidate], question)
        if report["supported"]:
            return [candidate], report
        singles.append((quality(report), index))
        if not best_bundle or quality(report) > quality(best_report):
            best_bundle, best_report = [candidate], report
    beam = [index for _, index in sorted(singles, reverse=True)[:4]]
    initial_keys = {_span_key(span) for span in initial_chunks}
    beam.extend(
        index for index, candidate in enumerate(candidates)
        if _span_key(candidate) in initial_keys and index not in beam
    )
    beam = beam[:5]
    for size in range(2, min(max_spans, len(beam)) + 1):
        for indexes in itertools.combinations(beam, size):
            if consider([candidates[index] for index in indexes]):
                return best_bundle, best_report
    return best_bundle, best_report


def route_pair(
    row: Any,
    initial_spans: Sequence[Mapping[str, Any]],
    markdown_dir: str | Path,
    cache: dict[str, Any],
    target_text: str | None = None,
    max_spans: int = 3,
    max_chars: int = 8000,
) -> dict[str, Any]:
    """Route one target through original evidence, widening, or quarantine."""
    target = target_text if target_text is not None else assistant_turn(row)
    index = paper_index(clean(row.paper_id), markdown_dir, cache)
    aligned = align_evidence_bundle(initial_spans, index)
    initial_report = support_report(target, aligned, clean(row.question))
    if initial_report["supported"]:
        return {"route": "OPEN_AS_IS", "bundle": aligned,
                "report": initial_report, "paper_verified": True}
    widened, report = retrieve_bundle(
        row, index, aligned, target, max_spans=max_spans, max_chars=max_chars,
    )
    if report["supported"]:
        return {"route": "OPEN_WIDENED", "bundle": widened, "report": report,
                "paper_verified": True, "initial_report": initial_report}
    return {"route": "QUARANTINE_UNVERIFIED", "bundle": widened, "report": report,
            "paper_verified": False, "initial_report": initial_report}


def route_paper_records(
    records: Sequence[Mapping[str, Any]],
    initial_by_pair: Mapping[str, Sequence[Mapping[str, Any]]],
    markdown_dir: str | Path,
    max_spans: int = 3,
    max_chars: int = 8000,
) -> dict[str, dict[str, Any]]:
    """Process-safe paper worker used by notebooks with ProcessPoolExecutor."""
    cache: dict[str, Any] = {}
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        row = SimpleNamespace(**dict(record))
        output[row.pair_id] = route_pair(
            row,
            initial_by_pair.get(row.pair_id, []),
            markdown_dir,
            cache,
            max_spans=max_spans,
            max_chars=max_chars,
        )
    return output


def reground(
    answer: Any, paper_id: str, markdown_dir: str | Path,
    cache: dict[str, Any], window: int = 320,
) -> dict[str, Any] | None:
    """Compatibility numeric locator with exact boundaries and source offsets."""
    del window
    index = paper_index(paper_id, markdown_dir, cache)
    wanted = quantitative_mentions(answer)
    if not wanted:
        return None
    for chunk in index.get("chunks") or []:
        present = quantitative_mentions(chunk["quote"])
        if all(any(
            need["value"] == got["value"] and (
                not need["unit"] or need["unit"] == got["unit"]
            ) for got in present
        ) for need in wanted):
            return _span_dict(chunk)
    return None


# ----------------------------------------------------------- closed-book SFT

CASE_REPORT = re.compile(r"\bcase report\b|\ba \d+[- ]year[- ]old\b|\bthe patient\b", re.I)


def _json_value(value: Any) -> Any:
    text = clean(value)
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text


def study_context_for_pair(
    paper_id: str,
    question: Any,
    evidence: Sequence[Mapping[str, Any]],
    contexts: Mapping[str, Sequence[Mapping[str, Any]]],
    profiles: Mapping[str, Mapping[str, Any]],
    innovation: Mapping[str, Mapping[str, Any]],
    max_contexts: int = 3,
    text_limit: int = 2_000,
) -> tuple[str, dict[str, Any]]:
    """Select rich scope metadata without using the proposed answer as a query."""

    profile = dict(profiles.get(paper_id) or {})
    innovation_row = dict(innovation.get(paper_id) or {})
    locator = " ".join([
        clean(question),
        *(clean(span.get("quote")) for span in evidence),
    ])
    locator_terms = set(content_terms(locator))
    candidates = []
    fields = (
        "label", "study_design", "population_text", "sample_size_text",
        "period_text", "conditions_json",
    )
    for position, context in enumerate(contexts.get(paper_id) or ()):
        context = dict(context)
        searchable = " ".join(clean(context.get(field)) for field in fields)
        overlap = len(locator_terms & set(content_terms(searchable)))
        stable = clean(context.get("context_id") or context.get("local_id"))
        candidates.append((-overlap, stable, position, context))
    selected = [item[-1] for item in sorted(candidates)[:max_contexts]]

    title = clean(profile.get("title") or innovation_row.get("title"))
    discipline = clean(profile.get("discipline") or profile.get("domain"))
    place = clean(innovation_row.get("place"))
    lines = ["Study context (scope metadata; factual support still comes from Evidence):"]
    if title:
        lines.append(f"Study: '{title}'")
    if discipline:
        lines.append(f"Discipline: {discipline}")
    if place:
        lines.append(f"Location: {place}")
    rendered_contexts = []
    for index, context in enumerate(selected, 1):
        values = []
        for field, label in (
            ("label", "focus"), ("study_design", "design"),
            ("population_text", "population"),
            ("sample_size_text", "sample"), ("period_text", "period"),
        ):
            value = clean(context.get(field))
            if value:
                values.append(f"{label}: {value}")
        conditions = _json_value(context.get("conditions_json"))
        if conditions:
            values.append(
                "conditions: " + json.dumps(conditions, ensure_ascii=False, separators=(",", ":")),
            )
        if values:
            rendered_contexts.append(f"Context {index}: " + "; ".join(values))
    lines.extend(rendered_contexts)
    innovation_bits = []
    for field, label in (
        ("innovation_type", "type"),
        ("constraint_addressed", "constraint"),
        ("materials_or_species_json", "materials/species"),
    ):
        value = _json_value(innovation_row.get(field))
        if value:
            shown = (
                json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                if not isinstance(value, str) else value
            )
            innovation_bits.append(f"{label}: {shown}")
    if innovation_bits:
        lines.append("African innovation: " + "; ".join(innovation_bits))

    kept = []
    used = 0
    for line in lines:
        cost = len(line) + (1 if kept else 0)
        if kept and used + cost > text_limit:
            break
        kept.append(line)
        used += cost
    payload = {
        "paper_id": paper_id,
        "profile": {
            key: _json_value(profile.get(key))
            for key in (
                "title", "domain", "discipline", "discipline_secondary_json",
                "key_contribution",
            )
            if _json_value(profile.get(key)) is not None
        },
        "selected_contexts": selected,
        "african_innovation": {
            key: _json_value(innovation_row.get(key))
            for key in (
                "is_african_innovation", "innovation_type", "constraint_addressed",
                "what_is_distinctive", "why_it_matters_here", "place",
                "materials_or_species_json",
            )
            if _json_value(innovation_row.get(key)) is not None
        },
    }
    return "\n".join(kept), payload


def descriptor(
    paper_id: str,
    contexts: Mapping[str, Mapping[str, Any]],
    profiles: Mapping[str, Mapping[str, Any]],
    innovation: Mapping[str, Mapping[str, Any]],
    limit: int = 1200,
) -> str:
    """Compose a titled study description without mid-field truncation."""
    context = contexts.get(paper_id) or {}
    profile = profiles.get(paper_id) or {}
    parts: list[str] = []
    seen: set[str] = set()
    for field in ("study_design", "population_text", "sample_size_text"):
        value = clean(context.get(field))
        if not value:
            continue
        bones = skeleton(value)
        if any(bones in other or other in bones for other in seen):
            continue
        seen.add(bones)
        parts.append(value.rstrip(" .;,"))
    body = ", ".join(parts)
    place = clean((innovation.get(paper_id) or {}).get("place"))
    if place:
        body = f"{body}, in {place}" if body else f"work in {place}"
    period = clean(context.get("period_text"))
    if period:
        body = f"{body} ({period})"
    title = clean(profile.get("title"))
    if title:
        body = f"{body}, reported in '{title}'" if body else f"the study '{title}'"
    if len(body) > limit:
        body = body[:limit].rsplit(" ", 1)[0].rstrip(" ,;") + "..."
    return body


_AUTHOR_YEAR = re.compile(
    r"^(?P<author>[^\d()]{1,140}?),\s*(?P<year>(?:18|19|20)\d{2}[a-z]?)$",
    re.I,
)
_AUTHOR_PAREN_YEAR = re.compile(
    r"^(?P<author>[^\d()]{1,140}?)\s*\(\s*(?P<year>(?:18|19|20)\d{2}[a-z]?)\s*\)$",
    re.I,
)
_FORBIDDEN_CITATION_TEXT = re.compile(
    r"\b(?:doi|openalex|pmid|paper[_ -]?id)\b|https?://|10\.\d{4,9}/",
    re.I,
)


def normalize_citation_label(value: Any) -> str:
    """Normalize a supplied author-year label without inventing provenance.

    This intentionally does not derive an author or a year. Callers must build
    the label from trusted paper metadata. DOI, URL, OpenAlex and internal-ID
    strings are rejected because a compact model should not be trained to
    reproduce high-entropy identifiers from memory.
    """

    text = re.sub(r"\s+", " ", clean(value)).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    if not text or _FORBIDDEN_CITATION_TEXT.search(text):
        return ""
    match = _AUTHOR_YEAR.fullmatch(text) or _AUTHOR_PAREN_YEAR.fullmatch(text)
    if not match:
        return ""
    author = match.group("author").strip(" ,;")
    year = match.group("year")
    if not re.search(r"[^\W\d_]", author, re.UNICODE):
        return ""
    return f"{author}, {year}"


def _context_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    text = clean(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _basis_fragment(value: Any, word_limit: int = 36) -> str:
    """Return one plain metadata fragment, never an identifier or citation."""

    if isinstance(value, (list, tuple, set)):
        value = ", ".join(clean(item) for item in value if clean(item))
    text = re.sub(r"\s+", " ", clean(value)).strip(" .;,:'\"")
    if not text or _FORBIDDEN_CITATION_TEXT.search(text):
        return ""
    words = text.split()
    if len(words) > word_limit:
        text = " ".join(words[:word_limit]).rstrip(" ,;:")
    return text


def _context_text_fields(value: Any) -> dict[str, str]:
    """Recover only semantic fields from the human-readable context block."""

    output: dict[str, str] = {}
    for line in clean(value).splitlines():
        line = line.strip()
        lower = line.casefold()
        # The exact title is deliberately ignored.
        if lower.startswith("discipline:"):
            output.setdefault("discipline", line.split(":", 1)[1].strip())
        elif lower.startswith("location:"):
            output.setdefault("place", line.split(":", 1)[1].strip())
        elif lower.startswith("context ") and ":" in line:
            remainder = line.split(":", 1)[1]
            for item in remainder.split(";"):
                if ":" not in item:
                    continue
                key, item_value = item.split(":", 1)
                key = key.strip().casefold()
                if key in {"focus", "design", "population", "period"}:
                    output.setdefault(key, item_value.strip())
    return output


def semantic_study_basis(paper_context: Any, *, limit: int = 320) -> str:
    """Build a short, searchable study description from trusted metadata.

    The result is semantic provenance, not a formal citation. It uses study
    design, population, place, period, discipline and focus when available,
    while deliberately ignoring titles, paper IDs and external identifiers.
    Complete fields are preferred over mid-field truncation.
    """

    payload = _context_mapping(paper_context)
    text_fields = _context_text_fields(paper_context) if not payload else {}
    profile = payload.get("profile") if isinstance(payload.get("profile"), Mapping) else {}
    innovation = (
        payload.get("african_innovation")
        if isinstance(payload.get("african_innovation"), Mapping) else {}
    )
    selected = payload.get("selected_contexts")
    context = next(
        (item for item in selected or () if isinstance(item, Mapping)),
        {},
    )

    fields = {
        "discipline": profile.get("discipline") or profile.get("domain")
        or text_fields.get("discipline"),
        "design": context.get("study_design") or text_fields.get("design"),
        "population": context.get("population_text") or text_fields.get("population"),
        "place": innovation.get("place") or text_fields.get("place"),
        "period": context.get("period_text") or text_fields.get("period"),
        "focus": context.get("label") or text_fields.get("focus"),
    }
    labels = (
        ("discipline", "discipline"),
        ("design", "design"),
        ("population", "population"),
        ("place", "location"),
        ("period", "period"),
        ("focus", "focus"),
    )
    parts: list[str] = []
    seen: set[str] = set()
    for key, label in labels:
        fragment = _basis_fragment(fields.get(key))
        bones = skeleton(fragment)
        if not fragment or bones in seen:
            continue
        candidate = f"{label}: {fragment}"
        proposed = "; ".join([*parts, candidate])
        if len(proposed) > limit:
            continue
        seen.add(bones)
        parts.append(candidate)
    return "; ".join(parts)


def descriptor_reveals_target(row: Any, described: str) -> bool:
    """Reject closed prompts that copy the answer into the study descriptor."""
    answer = clean(getattr(row, "answer", ""))
    if not answer or not described:
        return False
    answer_key = re.sub(r"[^\w]+", " ", answer.casefold()).strip()
    descriptor_key = re.sub(r"[^\w]+", " ", described.casefold()).strip()
    if answer_key and re.search(
        rf"(?<!\w){re.escape(answer_key)}(?!\w)", descriptor_key,
    ):
        return True
    quantities = quantitative_mentions(answer)
    shown = quantitative_mentions(described)
    return bool(quantities) and all(
        any(
            wanted["value"] == got["value"]
            and (not wanted["unit"] or wanted["unit"] == got["unit"])
            for got in shown
        )
        for wanted in quantities
    )


def verified_closed_ready(row: Any, described: str, paper_verified: bool) -> tuple[bool, str]:
    """Allow verified factual and reasoning knowledge, including figures."""
    if not paper_verified:
        return False, "not verified against the paper"
    if clean(getattr(row, "pair_type", "")).upper() not in {"FACTUAL", "REASONING"}:
        return False, "not an SFT knowledge pair"
    if not described or len(described) < 45 or "'" not in described:
        return False, "no titled study descriptor"
    if CASE_REPORT.search(described) or CASE_REPORT.search(clean(getattr(row, "question", ""))):
        return False, "single case is not a stable closed-book target"
    if not clean(getattr(row, "question", "")) or not assistant_turn(row):
        return False, "question or target is blank"
    if descriptor_reveals_target(row, described):
        return False, "descriptor reveals the target"
    return True, ""


def permissive_closed_ready(row: Any, described: str) -> tuple[bool, str]:
    """Check the closed prompt contract without asserting evidence verification."""
    if clean(getattr(row, "pair_type", "")).upper() not in {"FACTUAL", "REASONING"}:
        return False, "not an SFT knowledge pair"
    if len([line for line in described.splitlines() if clean(line)]) < 2:
        return False, "no usable study context"
    if not clean(getattr(row, "question", "")) or not assistant_turn(row):
        return False, "question or target is blank"
    if descriptor_reveals_target(row, described):
        return False, "study context reveals the target"
    return True, ""


def closed_book_ready(row: Any, described: str, paper_verified: bool = True) -> tuple[bool, str]:
    return verified_closed_ready(row, described, paper_verified)


def curriculum_mode(
    pair_id: str, open_share: float = 0.45, closed_share: float = 0.45,
    dual_share: float = 0.10, seed: int = 7,
) -> str:
    """Stable OPEN/CLOSED/DUAL assignment; balance production by tokens."""
    total = open_share + closed_share + dual_share
    if total <= 0:
        raise ValueError("curriculum shares must sum to a positive value")
    point = int.from_bytes(
        hashlib.sha256(f"{seed}:{pair_id}".encode()).digest()[:8], "big",
    ) / 2**64
    if point < open_share / total:
        return "OPEN"
    if point < (open_share + closed_share) / total:
        return "CLOSED"
    return "DUAL"


# -------------------------------------------------------------- renderings

OPEN_SYSTEM = (
    "You are a research assistant for African scientific literature. Answer "
    "using only the evidence provided. If the evidence does not contain the "
    "answer, say so plainly."
)
CLOSED_SYSTEM = (
    "You have learned a broad body of African scientific research. Answer from "
    "that learned research knowledge, and distinguish reported findings from "
    "general scientific inference."
)
LEAD = re.compile(
    r"^\s*(in|from|according to)\s+(this|the)\s+"
    r"(study|paper|work|research|article|survey)\s*,?\s*", re.I,
)

PROVIDED_EVIDENCE = "PROVIDED_EVIDENCE"
LEARNED_STUDY = "LEARNED_STUDY"
UNVERIFIED_STUDY = "UNVERIFIED_STUDY"
GENERAL_INFERENCE = "GENERAL_INFERENCE"
NO_PROVENANCE = "NONE"
PROVENANCE_KINDS = {
    PROVIDED_EVIDENCE,
    LEARNED_STUDY,
    UNVERIFIED_STUDY,
    GENERAL_INFERENCE,
    NO_PROVENANCE,
}


def format_provenance_response(
    target: Any,
    *,
    provenance: str,
    citation_label: Any = "",
    study_basis: Any = "",
    evidence_labels: Sequence[str] = (),
) -> str:
    """Append the invariant, machine-parseable provenance contract.

    ``citation_label`` is accepted only as author-year metadata. This function
    never derives a citation from the answer, title, identifier or evidence.
    """

    kind = clean(provenance).upper()
    if kind not in PROVENANCE_KINDS:
        raise ValueError(f"unknown provenance kind: {provenance!r}")
    citation = normalize_citation_label(citation_label)
    basis = _basis_fragment(study_basis, word_limit=60)
    labels = list(dict.fromkeys(
        clean(label) for label in evidence_labels if clean(label)
    ))
    if kind == PROVIDED_EVIDENCE:
        provenance_line = (
            f"Provenance: {kind} — {', '.join(labels)}" if labels
            else f"Provenance: {kind}"
        )
        citation_line = f"Citation: ({citation})" if citation else (
            "Citation: Author-year metadata unavailable"
        )
        basis_line = basis or "Evidence supplied in the prompt"
    elif kind == LEARNED_STUDY:
        provenance_line = f"Provenance: {kind}"
        citation_line = f"Citation: ({citation})" if citation else (
            "Citation: Author-year metadata unavailable"
        )
        basis_line = basis or "Learned African research study"
    elif kind == UNVERIFIED_STUDY:
        provenance_line = f"Provenance: {kind}"
        citation_line = (
            f"Citation: ({citation}) [unverified]" if citation
            else "Citation: Candidate author-year metadata unavailable"
        )
        basis_line = basis or "Candidate study attribution; not verified"
    elif kind == GENERAL_INFERENCE:
        provenance_line = f"Provenance: {kind}"
        citation_line = "Citation: No specific study attributed"
        basis_line = basis or "General scientific inference"
    else:
        provenance_line = f"Provenance: {kind}"
        citation_line = "Citation: No supporting study identified"
        basis_line = basis or "No supporting study identified"
    return (
        f"{clean(target)}\n\n{provenance_line}\n{citation_line}\n"
        f"Study basis: {basis_line}"
    )


def _rendered_target(
    row: Any,
    *,
    mode: str,
    verification_tier: str | None,
    citation_label: Any,
    study_basis: Any,
    evidence_labels: Sequence[str] = (),
) -> str:
    target = assistant_turn(row)
    if verification_tier is None:
        return target
    tier = clean(verification_tier).upper()
    if tier not in {"VERIFIED", "UNVERIFIED"}:
        raise ValueError(f"unknown verification tier: {verification_tier!r}")
    provenance = (
        UNVERIFIED_STUDY
        if tier == "UNVERIFIED"
        else PROVIDED_EVIDENCE if mode == "OPEN" else LEARNED_STUDY
    )
    return format_provenance_response(
        target,
        provenance=provenance,
        citation_label=citation_label,
        study_basis=study_basis,
        evidence_labels=evidence_labels,
    )


def _coerce_bundle(evidence: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str) -> list[dict[str, Any]]:
    if isinstance(evidence, str):
        return [{"quote": clean(evidence)}] if clean(evidence) else []
    if isinstance(evidence, Mapping):
        return dedupe_evidence([evidence])
    return dedupe_evidence(evidence)


def render_open(
    row: Any, evidence: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str,
    study_context: str = "",
    *,
    citation_label: Any = "",
    paper_context: Any = None,
    study_basis: Any = "",
    verification_tier: str | None = None,
) -> dict[str, Any]:
    shown = []
    for position, span in enumerate(_coerce_bundle(evidence), 1):
        labels = []
        if clean(span.get("page")):
            labels.append(f"page {clean(span.get('page'))}")
        if clean(span.get("section")):
            labels.append(clean(span.get("section")))
        suffix = f" ({'; '.join(labels)})" if labels else ""
        shown.append(f"Evidence {position}{suffix}:\n{clean(span.get('quote'))}")
    context_block = f"\n\n{clean(study_context)}" if clean(study_context) else ""
    user = (
        f"{OPEN_SYSTEM}{context_block}\n\n" + "\n\n".join(shown)
        + f"\n\nQuestion: {clean(row.question)}"
    )
    basis = clean(study_basis) or semantic_study_basis(
        paper_context if paper_context is not None else study_context,
    )
    target = _rendered_target(
        row,
        mode="OPEN",
        verification_tier=verification_tier,
        citation_label=citation_label,
        study_basis=basis,
        evidence_labels=tuple(f"Evidence {position}" for position in range(1, len(shown) + 1)),
    )
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": target},
    ]}


def render_closed(
    row: Any,
    described: str,
    *,
    citation_label: Any = "",
    paper_context: Any = None,
    study_basis: Any = "",
    verification_tier: str | None = None,
) -> dict[str, Any]:
    question = LEAD.sub("", clean(row.question)).strip()
    if question:
        question = question[:1].upper() + question[1:]
    user = f"{CLOSED_SYSTEM}\n\n{described.rstrip('.')}\n\n{question}"
    basis = clean(study_basis) or semantic_study_basis(
        paper_context if paper_context is not None else described,
    )
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": _rendered_target(
            row,
            mode="CLOSED",
            verification_tier=verification_tier,
            citation_label=citation_label,
            study_basis=basis,
        )},
    ]}


REFUSAL = (
    "The evidence provided does not establish this. It concerns a related "
    "point but does not answer the question asked."
)


def render_refusal(
    row: Any,
    wrong_quote: str,
    *,
    provenance_footer: bool = False,
) -> dict[str, Any]:
    """Render only a separately validated hard negative (not called by sampler)."""
    user = f'{OPEN_SYSTEM}\n\nEvidence:\n"{clean(wrong_quote)}"\n\nQuestion: {clean(row.question)}'
    target = (
        format_provenance_response(REFUSAL, provenance=NO_PROVENANCE)
        if provenance_footer else REFUSAL
    )
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": target},
    ]}


# ------------------------------------------------------------- preferences

def preference_ready(
    row: Any, evidence: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str,
) -> tuple[bool, str]:
    chosen, rejected = clean(row.chosen), clean(row.rejected)
    if not chosen or not rejected:
        return False, "chosen or rejected missing"
    if skeleton(chosen) == skeleton(rejected):
        return False, "chosen and rejected are the same"
    report = support_report(chosen, _coerce_bundle(evidence), clean(row.question))
    return (True, "") if report["supported"] else (False, "chosen is not supported by the evidence")


def render_preference(
    row: Any, evidence: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str,
    study_context: str = "",
    *,
    citation_label: Any = "",
    paper_context: Any = None,
    study_basis: Any = "",
    verification_tier: str | None = None,
) -> dict[str, Any]:
    rendered = render_open(
        row,
        evidence,
        study_context,
        citation_label=citation_label,
        paper_context=paper_context,
        study_basis=study_basis,
        verification_tier=verification_tier,
    )
    prompt = rendered["messages"][0]["content"]
    basis = clean(study_basis) or semantic_study_basis(
        paper_context if paper_context is not None else study_context,
    )
    bundle = _coerce_bundle(evidence)

    def preference_target(value: Any) -> str:
        if verification_tier is None:
            return clean(value)
        tier = clean(verification_tier).upper()
        provenance = PROVIDED_EVIDENCE if tier == "VERIFIED" else UNVERIFIED_STUDY
        return format_provenance_response(
            value,
            provenance=provenance,
            citation_label=citation_label,
            study_basis=basis,
            evidence_labels=tuple(
                f"Evidence {position}" for position in range(1, len(bundle) + 1)
            ),
        )
    return {
        "prompt": prompt,
        "chosen": preference_target(row.chosen),
        "rejected": preference_target(row.rejected),
        "rejection_reason": clean(getattr(row, "rejection_reason", "")),
    }


# ------------------------------------------------------------- study splits

STOP = {
    "the", "of", "and", "in", "on", "for", "a", "an", "to", "from", "with",
    "study", "studies", "analysis", "assessment", "evaluation", "nigeria",
    "nigerian", "african", "africa", "some", "its", "their",
}


def family_key(title: Any, discipline: Any) -> str:
    words = [word for word in re.findall(r"[a-z]{4,}", clean(title).lower()) if word not in STOP]
    return f"{clean(discipline)}|{' '.join(sorted(words)[:4])}" if words else f"{clean(discipline)}|"


def study_families(profiles: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {
        paper_id: family_key(row.get("title"), row.get("discipline"))
        for paper_id, row in profiles.items()
    }


def split_by_family(
    paper_ids: Iterable[str], families: Mapping[str, str],
    eval_fraction: float = 0.02, seed: int = 7,
) -> set[str]:
    import random

    paper_ids = list(paper_ids)
    grouped: dict[str, list[str]] = defaultdict(list)
    for paper_id in paper_ids:
        grouped[families.get(paper_id, paper_id)].append(paper_id)
    keys = sorted(grouped)
    random.Random(seed).shuffle(keys)
    held: set[str] = set()
    target = max(1, int(len(paper_ids) * eval_fraction))
    for key in keys:
        if len(held) >= target:
            break
        held.update(grouped[key])
    return held
