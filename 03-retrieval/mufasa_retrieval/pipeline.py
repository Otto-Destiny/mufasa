"""Question in, verified answer out.

The stage order matters and is the one place this layer overrides the drafted
architecture. application-architecture.md puts evidence cards *before*
generation; they are moved after validation here, because a card shown at
retrieval time is a *candidate*, not evidence. The validator can cut claims, so
showing eight tagged cards and then producing an answer that cites three tells
the reader they read support that was never used.

What still appears during generation is the paper-level `sources` payload —
title, journal, year — which keeps a slow laptop feeling responsive without
implying anything about grounding. No quote, no tag, no confidence badge:
grounding is a post-validation property.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .bundle import EvidenceBundle, build_bundle
from .embed import Embedder
from .gate import decide
from .generate import Generator, GenerationError
from .prompt import build_prompt, needs_safety_route
from .search import SearchResult, search
from .validate import ValidationReport, soften, validate

Stage = str
STAGES: tuple[Stage, ...] = ("searching", "gathering", "generating", "checking", "done")


class CancelledGeneration(RuntimeError):
    """The user stopped the answer. Raised at the next stage boundary."""


@dataclass
class Source:
    """Paper-level provenance, safe to show while the model is still writing."""

    paper_id: str
    title: str
    journal: str | None
    year: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "journal": self.journal,
            "year": self.year,
        }


@dataclass
class AnswerResult:
    question: str
    answer: str
    raw_answer: str
    verdict: str
    validation: ValidationReport
    bundle: EvidenceBundle
    sources: list[Source]
    cited_tags: list[str]
    search: SearchResult
    safety_route: bool
    timings_ms: dict[str, int] = field(default_factory=dict)

    @property
    def cited_records(self) -> list[Any]:
        return [r for r in self.bundle.records if r.tag in self.cited_tags]

    @property
    def other_candidates(self) -> list[Any]:
        return [r for r in self.bundle.records if r.tag not in self.cited_tags]

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "verdict": self.verdict,
            "answerable": self.bundle.decision.answerable,
            "study_families": self.bundle.study_families,
            "safety_route": self.safety_route,
            "sources": [s.as_dict() for s in self.sources],
            "evidence": [r.as_dict() for r in self.cited_records],
            "other_candidates": [r.as_dict() for r in self.other_candidates],
            "coverage": self.bundle.decision.as_dict(),
            "validation": self.validation.as_dict(),
            "channels": {
                "used": self.search.channels_used,
                "degraded": self.search.channels_degraded,
            },
            "timings_ms": self.timings_ms,
        }


def _generate(
    generator: Generator,
    prompt: str,
    bundle: EvidenceBundle,
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    from .generate import StubGenerator

    try:
        return generator.generate(prompt, max_tokens=max_tokens, temperature=temperature)
    except GenerationError:
        if bundle.decision.answerable and bundle.records:
            return StubGenerator.from_evidence(bundle)
        return bundle.decision.message or "MUFASA could not generate an answer."


def _checked(raw: str, bundle: EvidenceBundle) -> tuple[ValidationReport, str]:
    report = validate(raw, bundle)
    final = soften(raw, report) if not report.ok else raw
    if not report.ok:
        report = validate(final, bundle)
    return report, final


def _sources_of(bundle: EvidenceBundle) -> list[Source]:
    seen: dict[str, Source] = {}
    for r in bundle.records:
        seen.setdefault(
            r.paper_id,
            Source(paper_id=r.paper_id, title=r.paper_title, journal=r.paper_journal,
                   year=r.paper_year),
        )
    return list(seen.values())


def answer(
    conn: sqlite3.Connection,
    question: str,
    generator: Generator,
    *,
    k: int = 10,
    max_records: int = 10,
    max_tokens: int = 400,
    temperature: float = 0.0,
    embedder: Embedder | None = None,
    use_vectors: bool = True,
    on_stage: Callable[[Stage, dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> AnswerResult:
    timings: dict[str, int] = {}
    _emit = on_stage or (lambda *_: None)

    def emit(stage: Stage, payload: dict[str, Any]) -> None:
        if cancel_check is not None and cancel_check():
            raise CancelledGeneration(f"cancelled before {stage}")
        _emit(stage, payload)

    t0 = time.perf_counter()
    emit("searching", {})
    result = search(conn, question, k=k, embedder=embedder, use_vectors=use_vectors)
    timings["search"] = int((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    emit("gathering", {})
    decision = decide(conn, question, result.hits)
    bundle = build_bundle(conn, result.hits, decision, max_records=max_records)
    sources = _sources_of(bundle)
    timings["gather"] = int((time.perf_counter() - t1) * 1000)
    emit("gathering", {"sources": [s.as_dict() for s in sources]})

    t2 = time.perf_counter()
    emit("generating", {})
    prompt = build_prompt(question, bundle)
    raw = _generate(generator, prompt, bundle, max_tokens=max_tokens, temperature=temperature)
    timings["generate"] = int((time.perf_counter() - t2) * 1000)

    t3 = time.perf_counter()
    emit("checking", {})
    report, final = _checked(raw, bundle)
    if (
        not report.ok
        and bundle.decision.answerable
        and bundle.records
        and getattr(generator, "name", "") != "stub"
    ):
        # Small instruct models often answer without [E#] tags. The gate already
        # found covering evidence, so cite it rather than showing a blank fail.
        from .generate import StubGenerator

        raw = StubGenerator.from_evidence(bundle)
        report, final = _checked(raw, bundle)
    timings["validate"] = int((time.perf_counter() - t3) * 1000)
    timings["total"] = int((time.perf_counter() - t0) * 1000)

    res = AnswerResult(
        question=question,
        answer=final,
        raw_answer=raw,
        verdict=report.verdict,
        validation=report,
        bundle=bundle,
        sources=sources,
        cited_tags=report.cited_tags,
        search=result,
        safety_route=needs_safety_route(question),
        timings_ms=timings,
    )
    emit("done", res.as_dict())
    return res


def answer_stream(
    conn_factory: Callable[[], sqlite3.Connection],
    question: str,
    generator: Generator,
    **kwargs: Any,
) -> Iterator[tuple[Stage, dict[str, Any]]]:
    """Same pipeline, yielded stage by stage as it happens.

    Runs on a worker thread so stages reach the client while the model is still
    writing, rather than all at once at the end. `conn_factory` because SQLite
    connections belong to the thread that made them.
    """
    import queue
    import threading

    events: queue.Queue[tuple[Stage, dict[str, Any]] | None] = queue.Queue()

    def run() -> None:
        conn = conn_factory()
        try:
            answer(conn, question, generator,
                   on_stage=lambda s, p: events.put((s, p)), **kwargs)
        except CancelledGeneration:
            events.put(("cancelled", {}))
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as a stage
            events.put(("error", {"error": str(exc)}))
        finally:
            conn.close()
            events.put(None)

    worker = threading.Thread(target=run, name="mufasa-answer", daemon=True)
    worker.start()
    while True:
        item = events.get()
        if item is None:
            break
        yield item
    worker.join(timeout=1.0)
