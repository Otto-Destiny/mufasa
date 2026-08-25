"""Prompt assembly. Roughly 1,000-1,500 tokens of evidence, never more.

Two branches only: answerable, and corpus-does-not-cover-this. The safety route
adds a third instruction rather than a third prompt.
"""

from __future__ import annotations

import json
from typing import Any

from .bundle import EvidenceBundle

SYSTEM = (
    "You are MUFASA, a scientific research assistant for African science. "
    "Answer only from the numbered evidence you are given. "
    "Every specific number, unit or finding must carry the tag of the evidence it came "
    "from, written like [E1]. General scientific background needs no tag but must be "
    "clearly general. Never invent a tag that was not supplied. "
    "Keep the answer short: state the finding, cite it, say what is uncertain, stop."
)

SAFETY = (
    "This question touches medical, structural or industrial safety. Report what the "
    "studies measured and under what conditions, state what was not tested, and refer "
    "the reader to a qualified professional and local standards. Do not give an "
    "individual prescription or a design instruction."
)

_SAFETY_TERMS = (
    "dose", "dosage", "treat", "treatment", "cure", "safe to drink", "drink", "medicine",
    "medicinal", "therapy", "patient", "structural", "load bearing", "load-bearing",
    "foundation", "beam", "column", "should i", "is it safe",
)


def needs_safety_route(question: str) -> bool:
    q = question.casefold()
    return any(term in q for term in _SAFETY_TERMS)


def _measurement_line(measurement: dict[str, Any]) -> str:
    if not measurement:
        return ""
    return json.dumps(measurement, ensure_ascii=False, sort_keys=True)


def format_evidence(bundle: EvidenceBundle) -> str:
    blocks = []
    for r in bundle.records:
        lines = [f"[{r.tag}] {r.text}"]
        m = _measurement_line(r.measurement)
        if m:
            lines.append(f"    measurement: {m}")
        if r.conditions:
            lines.append(f"    conditions: {json.dumps(r.conditions, ensure_ascii=False, sort_keys=True)}")
        if r.quote:
            lines.append(f'    quoted: "{r.quote}"')
        elif r.quote_withheld:
            lines.append("    quoted: (withheld — licence restricts reuse of the source text)")
        cite = f"    source: {r.paper_id}"
        if r.page:
            cite += f", page {r.page}"
        if r.section:
            cite += f", {r.section}"
        lines.append(cite)
        if r.limitations:
            lines.append(f"    limitations: {'; '.join(r.limitations)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_prompt(question: str, bundle: EvidenceBundle) -> str:
    parts = [SYSTEM]
    if needs_safety_route(question):
        parts.append(SAFETY)

    if bundle.decision.answerable and bundle.records:
        parts.append("EVIDENCE:\n" + format_evidence(bundle))
        parts.append(
            f"Available tags: {', '.join(bundle.tags)}. "
            f"{bundle.study_families} separate study famil"
            f"{'y' if bundle.study_families == 1 else 'ies'} contributed this evidence."
        )
        parts.append(f"QUESTION: {question}")
        parts.append("ANSWER (short, cited):")
    else:
        parts.append(
            "The corpus contains no evidence that answers this question. "
            "Say so in one sentence using exactly this wording, then, if any related "
            "evidence is listed below, name what it does cover in one further sentence. "
            "Do not answer the question from general knowledge."
        )
        parts.append(f"REQUIRED OPENING: {bundle.decision.message}")
        if bundle.records:
            parts.append("NEAREST RELATED EVIDENCE:\n" + format_evidence(bundle))
        parts.append(f"QUESTION: {question}")
        parts.append("ANSWER:")

    return "\n\n".join(parts)
