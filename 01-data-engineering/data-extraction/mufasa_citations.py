"""Author-year citation metadata for MUFASA training examples.

This module is deliberately *audit only*.  It combines the existing OpenAlex
author cache and document manifest with strong signals from the first page of
each paper, but it never decides whether a paper or training pair survives.
Every requested ``paper_id`` receives a row and a usable citation label.

The public contract is:

``prepare_citation_metadata(...)``
    Build one canonical row per frozen paper and optionally write it atomically
    to ``citation_metadata.parquet``.

``load_citation_metadata(...)``
    Load that Parquet as a mapping keyed by the short ``W...`` paper id.

``citation_for_paper(...)``
    Return the stored row, or a non-blocking fallback when a caller encounters
    a paper that was not present while the metadata table was prepared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


CITATION_METADATA_VERSION = "mufasa-author-year-audit-v1"
AUDIT_MODE = "AUDIT_ONLY"

STATUS_VERIFIED = "VERIFIED_DOCUMENT"
STATUS_CORRECTED = "CORRECTED_DOCUMENT"
STATUS_METADATA = "METADATA_ONLY"
STATUS_CONFLICT = "CONFLICT"
STATUS_INVALID = "INVALID"

_OPENALEX_ID_RE = re.compile(r"W\d+", re.IGNORECASE)
_YEAR_RE = re.compile(r"(?<!\d)((?:18|19|20)\d{2})(?!\d)")
_PAGE_TWO_RE = re.compile(
    r"<!--\s*MUFASA_PDF_PAGE:\s*2\s*-->|^##\s+PDF page\s+2\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_MARKDOWN_RE = re.compile(r"[#*_`<>]+")
_SPACE_RE = re.compile(r"\s+")
_AUTHOR_SPLIT_RE = re.compile(r"\s*(?:;|\||\s+and\s+)\s*", re.IGNORECASE)
_INITIALS_RE = re.compile(r"^(?:(?:[A-Z]\.){1,5}|[A-Z]{1,4})$")
_AFFILIATION_MARK_RE = re.compile(
    r"(?:\s*[\[(]\s*\d+[a-z]?\s*[\])]|\s*[¹²³⁴⁵⁶⁷⁸⁹⁰*†‡]+)+\s*$"
)

_NON_NAME_WORDS = frozenset(
    {
        "abstract", "article", "articles", "author", "authors", "copyright",
        "department", "doi", "epidemiology", "introduction", "journal",
        "keywords", "original", "published", "research", "university",
        "volume", "www",
    }
)
_NAME_PARTICLES = frozenset(
    {"al", "bin", "da", "de", "del", "der", "di", "dos", "du", "el", "la", "le", "van", "von"}
)

CITATION_COLUMNS = (
    "paper_id",
    "openalex_id",
    "title",
    "raw_authors_json",
    "openalex_first_author",
    "openalex_first_author_family",
    "openalex_year",
    "openalex_label",
    "document_first_author_family",
    "document_year",
    "author_status",
    "year_status",
    "citation_status",
    "citation_label",
    "citation_parenthetical",
    "metadata_source",
    "author_evidence",
    "year_evidence",
    "document_year_candidates_json",
    "collision_key",
    "collision_count",
    "fallback_used",
    "audit_mode",
    "audit_version",
)

CITATION_SCHEMA = pa.schema(
    [
        pa.field(column, pa.int64(), nullable=False)
        if column == "collision_count"
        else pa.field(column, pa.bool_(), nullable=False)
        if column == "fallback_used"
        else pa.field(column, pa.string(), nullable=False)
        for column in CITATION_COLUMNS
    ]
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def short_openalex_id(value: Any) -> str:
    """Normalize either an OpenAlex URL or a short id to ``W123...``."""

    match = _OPENALEX_ID_RE.search(_text(value))
    return match.group(0).upper() if match else ""


def _normalise(value: Any) -> str:
    value = unicodedata.normalize("NFKD", _text(value))
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.replace("’", "'").casefold()
    return " ".join(re.findall(r"[a-z0-9]+", value))


def _clean_name(value: Any) -> str:
    value = _MARKDOWN_RE.sub(" ", _text(value))
    value = re.sub(r"\s*\([^)]*(?:affiliation|corresponding|email)[^)]*\)\s*", " ", value, flags=re.I)
    value = _AFFILIATION_MARK_RE.sub("", value)
    return _SPACE_RE.sub(" ", value).strip(" ,;:-")


def parse_authors(value: Any) -> list[str]:
    """Parse the JSON author array without treating malformed data as a gate."""

    if isinstance(value, (list, tuple)):
        candidates = value
    else:
        raw = _text(value)
        if not raw:
            return []
        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        candidates = decoded if isinstance(decoded, list) else []
    return [cleaned for item in candidates if (cleaned := _clean_name(item))]


def family_name(display_name: Any) -> str:
    """Return a conservative family-name form from an OpenAlex display name."""

    name = _clean_name(display_name)
    if not name:
        return ""
    # OpenAlex normally stores display names in given-name/family-name order.
    # Preserve an adjacent family particle (``van Dijk``, ``de Souza``).
    tokens = name.split()
    while tokens and tokens[-1].casefold().rstrip(".") in {"jr", "sr", "ii", "iii", "iv"}:
        tokens.pop()
    if not tokens:
        return ""
    # Some African journal/OpenAlex records use ``Surname A.B.`` while most
    # use ``A.B. Surname``.  A trailing initials token is therefore a strong
    # signal that the preceding token(s) are the family name.
    if len(tokens) > 1 and _INITIALS_RE.fullmatch(tokens[-1]):
        return " ".join(tokens[:-1]).strip(" ,.")
    start = len(tokens) - 1
    while start > 0 and tokens[start - 1].casefold().rstrip(".") in _NAME_PARTICLES:
        start -= 1
    return " ".join(tokens[start:]).strip(" ,.")


def publication_year(value: Any) -> str:
    """Extract a plausible publication year from the manifest date."""

    raw = _text(value)
    if not raw:
        return ""
    match = _YEAR_RE.search(raw)
    if not match:
        return ""
    year = int(match.group(1))
    upper = datetime.now(UTC).year + 1
    return str(year) if 1800 <= year <= upper else ""


def format_citation_label(authors: Sequence[str], year: str, *, title: str = "") -> tuple[str, bool]:
    """Create a conventional author-year label and a non-blocking fallback."""

    families = [family_name(author) for author in authors]
    families = [family for family in families if family]
    clean_year = publication_year(year)
    if families:
        if len(families) == 1:
            author_part = families[0]
        elif len(families) == 2:
            author_part = f"{families[0]} & {families[1]}"
        else:
            author_part = f"{families[0]} et al."
        return f"{author_part}, {clean_year or 'n.d.'}", not bool(clean_year)

    clean_title = _SPACE_RE.sub(" ", _text(title)).strip()
    if clean_title:
        if len(clean_title) > 96:
            clean_title = clean_title[:93].rstrip() + "..."
        return f'"{clean_title}", {clean_year or "n.d."}', True
    return f"Unattributed study, {clean_year or 'n.d.'}", True


def _first_page(markdown_path: Path | None, *, max_chars: int = 16_000) -> str:
    if markdown_path is None or not markdown_path.is_file():
        return ""
    text = markdown_path.read_text(encoding="utf-8", errors="replace")[: max_chars * 2]
    match = _PAGE_TWO_RE.search(text)
    if match:
        text = text[: match.start()]
    return text[:max_chars]


def _resolve_markdown_path(
    paper_id: str,
    manifest_path: Any,
    markdown_root: Path | None,
) -> Path | None:
    if markdown_root is not None:
        candidate = markdown_root / f"{paper_id}.md"
        if candidate.is_file():
            return candidate
    raw_path = _text(manifest_path)
    if raw_path:
        candidate = Path(raw_path)
        if candidate.is_file():
            return candidate
    return None


def _evidence_line(text: str, needle: str) -> str:
    normal_needle = _normalise(needle)
    if not normal_needle:
        return ""
    for raw_line in text.splitlines():
        line = _SPACE_RE.sub(" ", _MARKDOWN_RE.sub(" ", raw_line)).strip()
        if normal_needle in _normalise(line):
            return line[:500]
    return ""


def _byline_region(first_page: str) -> str:
    """Keep front matter and exclude abstract/body prose when possible."""

    # Parser YAML repeats the title and identifiers but is not document
    # evidence.  Removing it prevents a surname appearing in a title from
    # being mistaken for an author-line confirmation.
    if first_page.startswith("---"):
        closing = first_page.find("\n---", 3)
        if closing >= 0:
            first_page = first_page[closing + 4 :]
    marker = re.search(
        r"^\s*#{0,4}\s*(?:abstract|summary|introduction|background)\b",
        first_page,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return first_page[: marker.start()] if marker else first_page[:8_000]


def _looks_like_name(candidate: str, title: str) -> bool:
    candidate = _clean_name(candidate)
    if not candidate or len(candidate) > 160:
        return False
    lowered = set(_normalise(candidate).split())
    if lowered & _NON_NAME_WORDS:
        return False
    title_words = set(_normalise(title).split())
    if lowered and len(lowered & title_words) / len(lowered) >= 0.6:
        return False
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*", candidate)
    if not 2 <= len(words) <= 7:
        return False
    significant = [word for word in words if word.casefold().rstrip(".") not in _NAME_PARTICLES]
    if len(significant) < 2:
        return False
    return all(word[0].isupper() or (len(word) <= 3 and "." in word) for word in significant)


def _document_author(
    first_page: str,
    openalex_authors: Sequence[str],
    title: str,
) -> tuple[str, str, str]:
    """Return document family, evidence, and relation to OpenAlex order.

    Known OpenAlex surnames are preferred.  Correcting an author order is only
    attempted when another surname from the same OpenAlex author list appears
    first in the byline region.  An unrelated parsed name is reported as a
    conflict rather than silently replacing metadata.
    """

    if not first_page:
        return "", "", "NONE"
    region = _byline_region(first_page)
    normal_region = _normalise(region)
    families = [family_name(author) for author in openalex_authors]
    located: list[tuple[int, int, str]] = []
    for index, family in enumerate(families):
        needle = _normalise(family)
        if not needle:
            continue
        match = re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", normal_region)
        if match:
            located.append((match.start(), index, family))
    if located:
        _, index, family = min(located)
        relation = "MATCH" if index == 0 else "OPENALEX_ORDER_CORRECTED"
        return family, _evidence_line(region, family), relation

    # Conservative generic byline fallback.  Prefer Markdown headings or an
    # explicit "By" marker; do not infer names from arbitrary abstract prose.
    for raw_line in region.splitlines():
        stripped = raw_line.strip()
        explicit = bool(re.match(r"^(?:#{2,6}\s+|\*{1,2}\s*|by\s+)", stripped, re.I))
        if not explicit:
            continue
        cleaned = re.sub(r"^(?:#{1,6}\s*|\*+\s*|by\s+)", "", stripped, flags=re.I)
        cleaned = _AUTHOR_SPLIT_RE.split(cleaned, maxsplit=1)[0]
        # A comma commonly separates authors.  It may also express
        # family-name-first, so keep both sides when the tail is initials.
        if "," in cleaned:
            left, right = [part.strip() for part in cleaned.split(",", 1)]
            cleaned = f"{left} {right}" if re.fullmatch(r"(?:[A-Z]\.?\s*){1,4}", right) else left
        if _looks_like_name(cleaned, title):
            family = family_name(cleaned)
            if family:
                return family, _clean_name(raw_line)[:500], "UNRELATED_CANDIDATE"
    return "", "", "NONE"


def _document_year(first_page: str) -> tuple[str, str, list[dict[str, Any]], bool]:
    """Extract a strong first-page publication year.

    Study periods and years mentioned in abstracts are ignored.  Only lines
    carrying publication/copyright/citation or journal volume/issue signals
    participate.  Tied contradictory candidates are marked ambiguous.
    """

    if not first_page:
        return "", "", [], False
    scored: dict[str, dict[str, Any]] = {}
    for raw_line in first_page.splitlines():
        line = _SPACE_RE.sub(" ", _MARKDOWN_RE.sub(" ", raw_line)).strip()
        if not line or len(line) > 800:
            continue
        years = _YEAR_RE.findall(line)
        # ``findall`` returns the captured full year because the regex has one
        # capture group.
        if not years:
            continue
        low = line.casefold()
        score = 0
        signals: list[str] = []
        if "how to cite" in low or "cite this" in low or "citation" in low:
            score += 7
            signals.append("citation")
        if "published" in low or "publication date" in low:
            score += 6
            signals.append("published")
        if "copyright" in low or "©" in line:
            score += 6
            signals.append("copyright")
        if re.search(r"\bvol(?:ume)?\b|\bissue\b|\bno\.\s*\d+", low):
            score += 4
            signals.append("journal_header")
        if score < 4:
            continue
        for year in years:
            if not publication_year(year):
                continue
            current = scored.get(year)
            item = {"year": year, "score": score, "signals": signals, "evidence": line[:500]}
            if current is None or score > current["score"]:
                scored[year] = item
    candidates = sorted(scored.values(), key=lambda item: (-item["score"], item["year"], item["evidence"]))
    if not candidates:
        return "", "", [], False
    top_score = candidates[0]["score"]
    tied = [item for item in candidates if item["score"] == top_score]
    ambiguous = len({item["year"] for item in tied}) > 1
    if ambiguous:
        return "", " | ".join(item["evidence"] for item in tied)[:1_000], candidates, True
    top = candidates[0]
    return str(top["year"]), str(top["evidence"]), candidates, False


def _component_status(
    metadata_value: str,
    document_value: str,
    *,
    relation: str = "",
    ambiguous: bool = False,
) -> tuple[str, str]:
    """Return ``(chosen_value, audit_status)`` for one citation component."""

    if ambiguous:
        return metadata_value, STATUS_CONFLICT if metadata_value else STATUS_INVALID
    if document_value:
        if not metadata_value:
            return document_value, STATUS_CORRECTED
        if _normalise(metadata_value) == _normalise(document_value):
            return metadata_value, STATUS_VERIFIED
        if relation == "UNRELATED_CANDIDATE":
            return metadata_value, STATUS_CONFLICT
        return document_value, STATUS_CORRECTED
    if metadata_value:
        return metadata_value, STATUS_METADATA
    return "", STATUS_INVALID


def _overall_status(author_status: str, year_status: str, *, fallback_used: bool) -> str:
    statuses = {author_status, year_status}
    if STATUS_CONFLICT in statuses:
        return STATUS_CONFLICT
    if STATUS_CORRECTED in statuses:
        return STATUS_CORRECTED
    if statuses == {STATUS_VERIFIED}:
        return STATUS_VERIFIED
    if fallback_used and STATUS_INVALID in statuses:
        return STATUS_INVALID
    return STATUS_METADATA


def _source_for(author_status: str, year_status: str, fallback_used: bool) -> str:
    if fallback_used:
        return "FALLBACK"
    if STATUS_CORRECTED in {author_status, year_status}:
        return "OPENALEX+DOCUMENT_CORRECTION"
    if STATUS_VERIFIED in {author_status, year_status}:
        return "OPENALEX+DOCUMENT"
    return "OPENALEX"


def _read_frame(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    required = set(columns)
    missing = sorted(required - schema_names)
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    return pq.read_table(path, columns=list(columns)).to_pandas()


def _tracked(items: Sequence[str], description: str) -> Iterable[str]:
    """Wrap an iterable in a progress bar when tqdm is available.

    The audit reads the first page of every paper, which on a network mount is
    tens of minutes of otherwise silent work. Falling back to the bare iterable
    keeps the module usable where tqdm is not installed.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return items
    return tqdm(items, desc=description, unit="paper", smoothing=0.05)


def prepare_citation_metadata(
    *,
    split_manifest: Path | str,
    documents_path: Path | str,
    authors_cache_path: Path | str,
    markdown_root: Path | str | None,
    paper_ids: Iterable[str] | None = None,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Build citation metadata for frozen papers without filtering any row.

    ``paper_ids`` limits work for a smoke run; it does not express eligibility.
    Every requested id receives a row, including ids missing from either source.
    """

    split_manifest = Path(split_manifest)
    documents_path = Path(documents_path)
    authors_cache_path = Path(authors_cache_path)
    markdown_root = Path(markdown_root) if markdown_root is not None else None

    split = _read_frame(split_manifest, ("paper_id", "title"))
    split["paper_id"] = split["paper_id"].map(short_openalex_id)
    split = split.drop_duplicates("paper_id", keep="first")
    split = split[split["paper_id"] != ""]
    split_titles = dict(zip(split["paper_id"], split["title"], strict=False))

    if paper_ids is None:
        selected = sorted(split_titles)
    else:
        # Preserve explicitly requested ids even if the split manifest has a
        # metadata gap: audit-only means observation, never silent removal.
        selected = sorted({short_openalex_id(item) for item in paper_ids} - {""})

    documents = _read_frame(
        documents_path,
        ("paper_id", "openalex_id", "title", "publication_date", "markdown_path"),
    )
    documents["paper_id"] = documents["paper_id"].map(short_openalex_id)
    documents = documents.sort_values(["paper_id", "openalex_id"], kind="stable")
    doc_rows = {
        paper_id: group.iloc[0].to_dict()
        for paper_id, group in documents.groupby("paper_id", sort=False)
        if paper_id
    }

    author_cache = _read_frame(authors_cache_path, ("openalex_id", "authors_json"))
    author_cache["short_id"] = author_cache["openalex_id"].map(short_openalex_id)
    author_cache = author_cache.sort_values(["short_id", "openalex_id"], kind="stable")
    author_rows = {
        paper_id: group.iloc[0].to_dict()
        for paper_id, group in author_cache.groupby("short_id", sort=False)
        if paper_id
    }

    output: list[dict[str, Any]] = []
    for paper_id in _tracked(selected, f"citation audit ({len(selected):,} papers)"):
        document = doc_rows.get(paper_id, {})
        author_row = author_rows.get(paper_id, {})
        openalex_id = _text(document.get("openalex_id") or author_row.get("openalex_id"))
        if not openalex_id:
            openalex_id = f"https://openalex.org/{paper_id}"
        title = _text(document.get("title")) or _text(split_titles.get(paper_id))
        raw_authors_json = _text(author_row.get("authors_json"))
        authors = parse_authors(raw_authors_json)
        openalex_first = authors[0] if authors else ""
        openalex_family = family_name(openalex_first)
        openalex_year = publication_year(document.get("publication_date"))
        openalex_label, openalex_fallback = format_citation_label(authors, openalex_year, title=title)

        markdown_path = _resolve_markdown_path(paper_id, document.get("markdown_path"), markdown_root)
        first_page = _first_page(markdown_path)
        document_family, author_evidence, author_relation = _document_author(first_page, authors, title)
        chosen_family, author_status = _component_status(
            openalex_family,
            document_family,
            relation=author_relation,
        )
        document_year, year_evidence, year_candidates, year_ambiguous = _document_year(first_page)
        chosen_year, year_status = _component_status(
            openalex_year,
            document_year,
            ambiguous=year_ambiguous,
        )

        chosen_authors = list(authors)
        if chosen_family and (not chosen_authors or family_name(chosen_authors[0]) != chosen_family):
            matching = next(
                (author for author in authors if _normalise(family_name(author)) == _normalise(chosen_family)),
                chosen_family,
            )
            chosen_authors = [matching] + [
                author for author in authors
                if _normalise(family_name(author)) != _normalise(chosen_family)
            ]
        citation_label, fallback_used = format_citation_label(chosen_authors, chosen_year, title=title)
        fallback_used = bool(fallback_used or openalex_fallback and not chosen_authors)
        citation_status = _overall_status(author_status, year_status, fallback_used=fallback_used)
        collision_key = f"{_normalise(chosen_family)}|{chosen_year or 'n.d.'}"

        output.append(
            {
                "paper_id": paper_id,
                "openalex_id": openalex_id,
                "title": title,
                "raw_authors_json": raw_authors_json,
                "openalex_first_author": openalex_first,
                "openalex_first_author_family": openalex_family,
                "openalex_year": openalex_year,
                "openalex_label": openalex_label,
                "document_first_author_family": document_family,
                "document_year": document_year,
                "author_status": author_status,
                "year_status": year_status,
                "citation_status": citation_status,
                "citation_label": citation_label,
                "citation_parenthetical": f"({citation_label})",
                "metadata_source": _source_for(author_status, year_status, fallback_used),
                "author_evidence": author_evidence,
                "year_evidence": year_evidence,
                "document_year_candidates_json": json.dumps(
                    year_candidates, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                ),
                "collision_key": collision_key,
                "collision_count": 0,
                "fallback_used": fallback_used,
                "audit_mode": AUDIT_MODE,
                "audit_version": CITATION_METADATA_VERSION,
            }
        )

    collision_counts = Counter(row["collision_key"] for row in output)
    for row in output:
        row["collision_count"] = int(collision_counts[row["collision_key"]])

    frame = pd.DataFrame(output, columns=CITATION_COLUMNS)
    if output_path is not None:
        write_citation_metadata(frame, output_path)
    return frame


def write_citation_metadata(frame: pd.DataFrame, output_path: Path | str) -> Path:
    """Write the canonical table atomically with a strict Arrow schema."""

    output_path = Path(output_path)
    missing = sorted(set(CITATION_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"citation metadata lacks columns: {missing}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    table = pa.Table.from_pandas(frame.loc[:, CITATION_COLUMNS], schema=CITATION_SCHEMA, preserve_index=False)
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, output_path)
    return output_path


def load_citation_metadata(path: Path | str) -> dict[str, dict[str, Any]]:
    """Load a canonical citation table keyed by short paper id."""

    path = Path(path)
    frame = _read_frame(path, CITATION_COLUMNS)
    frame["paper_id"] = frame["paper_id"].map(short_openalex_id)
    duplicates = frame[frame["paper_id"].duplicated(keep=False)]["paper_id"].tolist()
    if duplicates:
        raise ValueError(f"duplicate citation paper_id values: {sorted(set(duplicates))[:20]}")
    return {
        row["paper_id"]: row
        for row in frame.to_dict(orient="records")
        if row["paper_id"]
    }


def citation_for_paper(
    paper_id: Any,
    citations: Mapping[str, Mapping[str, Any]],
    *,
    title: str = "",
) -> dict[str, Any]:
    """Return stored citation metadata or a non-blocking fallback record."""

    short_id = short_openalex_id(paper_id)
    existing = citations.get(short_id)
    if existing is not None:
        return dict(existing)
    label, _ = format_citation_label([], "", title=title)
    return {
        "paper_id": short_id or _text(paper_id),
        "openalex_id": f"https://openalex.org/{short_id}" if short_id else "",
        "title": _text(title),
        "raw_authors_json": "",
        "openalex_first_author": "",
        "openalex_first_author_family": "",
        "openalex_year": "",
        "openalex_label": label,
        "document_first_author_family": "",
        "document_year": "",
        "author_status": STATUS_INVALID,
        "year_status": STATUS_INVALID,
        "citation_status": STATUS_INVALID,
        "citation_label": label,
        "citation_parenthetical": f"({label})",
        "metadata_source": "FALLBACK",
        "author_evidence": "",
        "year_evidence": "",
        "document_year_candidates_json": "[]",
        "collision_key": "|n.d.",
        "collision_count": 1,
        "fallback_used": True,
        "audit_mode": AUDIT_MODE,
        "audit_version": CITATION_METADATA_VERSION,
    }


def audit_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Small notebook-friendly summary; it is descriptive, never a gate."""

    return {
        "rows": int(len(frame)),
        "citation_status": frame["citation_status"].value_counts(dropna=False).to_dict(),
        "author_status": frame["author_status"].value_counts(dropna=False).to_dict(),
        "year_status": frame["year_status"].value_counts(dropna=False).to_dict(),
        "fallback_rows": int(frame["fallback_used"].sum()),
        "ambiguous_author_year_rows": int((frame["collision_count"] > 1).sum()),
        "audit_mode": AUDIT_MODE,
    }


def _default_paths(data_root: Path) -> dict[str, Path]:
    return {
        "split_manifest": data_root / "corpus_splits" / "manifest.parquet",
        "documents_path": data_root / "mufasa_corpus" / "manifests" / "documents.parquet",
        "authors_cache_path": data_root / "production" / "authors_cache.parquet",
        "markdown_root": data_root / "mufasa_corpus" / "parsed" / "markdown",
        "output_path": data_root / "citation_metadata.parquet",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare audit-only MUFASA author-year citation metadata")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--paper-limit", type=int)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    paths = _default_paths(args.data_root)
    split = _read_frame(paths["split_manifest"], ("paper_id",))
    paper_ids = None
    if args.paper_limit is not None:
        if args.paper_limit < 1:
            parser.error("--paper-limit must be positive")
        available = {
            paper_id for value in split["paper_id"]
            if (paper_id := short_openalex_id(value))
        }
        paper_ids = sorted(
            available,
            key=lambda paper_id: hashlib.sha256(
                f"{args.seed}:{paper_id}".encode("utf-8"),
            ).hexdigest(),
        )[: args.paper_limit]
    frame = prepare_citation_metadata(
        split_manifest=paths["split_manifest"],
        documents_path=paths["documents_path"],
        authors_cache_path=paths["authors_cache_path"],
        markdown_root=paths["markdown_root"],
        paper_ids=paper_ids,
        output_path=args.output or paths["output_path"],
    )
    print(json.dumps(audit_summary(frame), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
