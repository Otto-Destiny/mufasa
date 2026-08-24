"""Validation before display. Nothing reaches the screen unchecked.

Three rules:

1. A tag the model invented — [E9] when eight were supplied — fails immediately.
2. Every number in a sentence must appear in the evidence that sentence cites,
   after light numeric normalisation.
3. A sentence carrying a specific number with no tag at all is an unsupported
   specific claim.

One thing this deliberately does *not* flag: digits inside citation formatting.
retrieval-v1 reported page numbers and the "088" in "P-G088" as invented facts
whenever the model echoed the prompt's own citation header back. Those are our
formatting, not the model's claims, so headers are masked before numbers are
extracted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .bundle import EvidenceBundle

TAG_RE = re.compile(r"\[E(\d+)\]")
NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w])")

# Our own citation furniture, echoed back by weaker models.
_HEADER_PATTERNS = (
    re.compile(r"\[E\d+\]"),
    re.compile(r"\bP-[A-Za-z]?\d+\b"),
    re.compile(r"\bpages?\s*\d+\b", re.I),
    re.compile(r"\bp\.\s*\d+\b", re.I),
    re.compile(r"\bsource:[^\n]*", re.I),
    re.compile(r"\bdoi\s*:?\s*\S+", re.I),
    re.compile(r"\b10\.\d{4,9}/\S+\b"),
    re.compile(r"\btable\s*\d+\b", re.I),
    re.compile(r"\bfigure\s*\d+\b", re.I),
)

_VERDICTS = ("grounded", "partly_grounded", "ungrounded", "no_matching_evidence")


def mask_citation_furniture(text: str) -> str:
    """Blank out citation formatting so its digits are not read as claims."""
    for pattern in _HEADER_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def _canon_number(raw: str) -> str:
    n = raw.replace(",", "")
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n or "0"


def _numbers(text: str) -> list[str]:
    return [_canon_number(m.group(1)) for m in NUMBER_RE.finditer(text)]


def _evidence_numbers(record: Any) -> set[str]:
    blob = " ".join(
        str(x)
        for x in (
            record.text or "",
            record.quote or "",
            json.dumps(record.measurement, ensure_ascii=False),
            json.dumps(record.conditions, ensure_ascii=False),
            record.paper_year or "",
        )
    )
    found = set(_numbers(blob))
    # A percentage stated as a share and a fraction are the same figure.
    for n in list(found):
        try:
            val = float(n)
        except ValueError:
            continue
        if 0 < val <= 1:
            found.add(_canon_number(f"{val * 100:.6f}"))
        if val > 1:
            found.add(_canon_number(f"{val / 100:.6f}"))
    return found


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]


@dataclass
class SentenceReport:
    sentence: str
    tags: list[str]
    numbers: list[str]
    unsupported_numbers: list[str]
    untagged_specific: bool

    @property
    def ok(self) -> bool:
        return not self.unsupported_numbers and not self.untagged_specific


@dataclass
class ValidationReport:
    verdict: str
    invented_tags: list[str] = field(default_factory=list)
    cited_tags: list[str] = field(default_factory=list)
    sentences: list[SentenceReport] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict in ("grounded", "no_matching_evidence")

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "invented_tags": self.invented_tags,
            "cited_tags": self.cited_tags,
            "unsupported_claims": self.unsupported_claims,
            "sentences": [
                {
                    "sentence": s.sentence,
                    "tags": s.tags,
                    "unsupported_numbers": s.unsupported_numbers,
                    "untagged_specific": s.untagged_specific,
                    "ok": s.ok,
                }
                for s in self.sentences
            ],
        }


def validate(answer: str, bundle: EvidenceBundle) -> ValidationReport:
    available = set(bundle.tags)

    if not bundle.decision.answerable:
        # The only correct answer is the abstention sentence; any tag it carries
        # must still exist, and no invented figure may ride along with it.
        invented = sorted({t for t in TAG_RE.findall(answer) if f"E{t}" not in available})
        verdict = "no_matching_evidence" if not invented else "ungrounded"
        return ValidationReport(
            verdict=verdict,
            invented_tags=[f"E{t}" for t in invented],
            cited_tags=sorted(available & {f"E{t}" for t in TAG_RE.findall(answer)}),
        )

    cited_raw = [f"E{n}" for n in TAG_RE.findall(answer)]
    invented = sorted(set(cited_raw) - available)
    cited = sorted(set(cited_raw) & available)

    reports: list[SentenceReport] = []
    for sentence in split_sentences(answer):
        tags = [f"E{n}" for n in TAG_RE.findall(sentence)]
        masked = mask_citation_furniture(sentence)
        nums = _numbers(masked)
        supported: set[str] = set()
        for tag in tags:
            rec = bundle.by_tag(tag)
            if rec is not None:
                supported |= _evidence_numbers(rec)
        unsupported = [n for n in nums if n not in supported]
        reports.append(
            SentenceReport(
                sentence=sentence,
                tags=tags,
                numbers=nums,
                unsupported_numbers=unsupported if tags else [],
                untagged_specific=bool(nums) and not tags,
            )
        )

    unsupported_claims = [r.sentence for r in reports if not r.ok]
    if invented:
        verdict = "ungrounded"
    elif not unsupported_claims and cited:
        verdict = "grounded"
    elif unsupported_claims and len(unsupported_claims) < len(reports):
        verdict = "partly_grounded"
    elif not cited:
        verdict = "ungrounded"
    else:
        verdict = "partly_grounded"

    return ValidationReport(
        verdict=verdict,
        invented_tags=invented,
        cited_tags=cited,
        sentences=reports,
        unsupported_claims=unsupported_claims,
    )


def soften(answer: str, report: ValidationReport) -> str:
    """Repair path: drop the sentences that failed, keep the ones that held."""
    if report.ok:
        return answer
    kept = [s.sentence for s in report.sentences if s.ok]
    if not kept:
        return (
            "MUFASA could not produce an answer supported by the retrieved evidence. "
            "The evidence cards below are what the corpus does contain."
        )
    return " ".join(kept)
