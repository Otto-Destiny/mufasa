"""The harness. Run it after every meaningful change.

Without this you cannot tell whether adding vectors, or one more hop, helped or
hurt. Each metric below is asserted as a release gate in
``tests/test_quality_gates.py`` so a regression fails CI rather than a demo.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .bundle import build_bundle
from .embed import Embedder, get_embedder
from .gate import decide
from .generate import Generator, StubGenerator
from .pipeline import answer
from .search import search


@dataclass
class QuestionOutcome:
    question_id: str
    question: str
    answerable: bool
    expected_claim_ids: list[str]
    minimum_expected_hits: int
    retrieved: list[str]
    hits_found: int
    recall_ok: bool
    gate_answerable: bool
    gate_reason: str
    gate_correct: bool
    latency_ms: int
    verdict: str | None = None
    cited_tags: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    total: int
    answerable_total: int
    unanswerable_total: int
    recall_at_k_hits: int
    recall_at_k: float
    abstention_hits: int
    abstention_rate: float
    gate_accuracy: float
    p50_latency_ms: int
    p95_latency_ms: int
    #: retrieval-v1 scored "at least one expected claim in the top 10". The
    #: fixture actually declares `minimum_expected_hits` (up to 4), which is the
    #: stricter and correct bar. Both are reported so the two stay comparable.
    recall_any_hits: int = 0
    recall_any: float = 0.0
    citation_precision: float | None = None
    unsupported_claim_rate: float | None = None
    outcomes: list[QuestionOutcome] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"questions               {self.total}",
            f"recall@10 strict        {self.recall_at_k_hits}/{self.answerable_total} "
            f"({self.recall_at_k:.0%})   [meets minimum_expected_hits]",
            f"recall@10 any           {self.recall_any_hits}/{self.answerable_total} "
            f"({self.recall_any:.0%})   [>=1 expected claim, the v1 bar]",
            f"abstention (unanswerable) {self.abstention_hits}/{self.unanswerable_total} "
            f"({self.abstention_rate:.0%})",
            f"gate accuracy           {self.gate_accuracy:.0%}",
            f"latency p50 / p95       {self.p50_latency_ms} ms / {self.p95_latency_ms} ms",
        ]
        if self.citation_precision is not None:
            lines.append(f"citation precision      {self.citation_precision:.0%}")
        if self.unsupported_claim_rate is not None:
            lines.append(f"unsupported-claim rate  {self.unsupported_claim_rate:.0%}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["outcomes"] = [asdict(o) for o in self.outcomes]
        return d


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate(
    conn: sqlite3.Connection,
    questions: list[dict[str, Any]],
    *,
    k: int = 10,
    embedder: Embedder | None = None,
    use_vectors: bool = True,
    generator: Generator | None = None,
    run_generation: bool = False,
) -> EvalReport:
    emb = embedder or get_embedder()
    gen = generator or StubGenerator()
    outcomes: list[QuestionOutcome] = []

    for q in questions:
        t0 = time.perf_counter()
        expected = list(q.get("expected_claim_ids") or [])
        minimum = int(q.get("minimum_expected_hits") or (1 if expected else 0))
        answerable = bool(q.get("answerable"))

        if run_generation:
            res = answer(conn, q["question"], gen, k=k, embedder=emb, use_vectors=use_vectors)
            retrieved = [h.claim_id for h in res.search.hits]
            gate_answerable = res.bundle.decision.answerable
            gate_reason = res.bundle.decision.reason
            verdict, cited = res.verdict, res.cited_tags
            unsupported = res.validation.unsupported_claims
        else:
            sr = search(conn, q["question"], k=k, embedder=emb, use_vectors=use_vectors)
            retrieved = [h.claim_id for h in sr.hits]
            decision = decide(conn, q["question"], sr.hits)
            build_bundle(conn, sr.hits, decision)
            gate_answerable, gate_reason = decision.answerable, decision.reason
            verdict, cited, unsupported = None, [], []

        found = len(set(expected) & set(retrieved))
        latency = int((time.perf_counter() - t0) * 1000)

        outcomes.append(
            QuestionOutcome(
                question_id=q["id"],
                question=q["question"],
                answerable=answerable,
                expected_claim_ids=expected,
                minimum_expected_hits=minimum,
                retrieved=retrieved,
                hits_found=found,
                recall_ok=(found >= max(1, minimum)) if answerable else True,
                gate_answerable=gate_answerable,
                gate_reason=gate_reason,
                gate_correct=(gate_answerable == answerable),
                latency_ms=latency,
                verdict=verdict,
                cited_tags=cited,
                unsupported_claims=unsupported,
            )
        )

    answerables = [o for o in outcomes if o.answerable]
    unanswerables = [o for o in outcomes if not o.answerable]
    latencies = sorted(o.latency_ms for o in outcomes) or [0]

    recall_hits = sum(1 for o in answerables if o.recall_ok)
    recall_any_hits = sum(1 for o in answerables if o.hits_found >= 1)
    abstain_hits = sum(1 for o in unanswerables if not o.gate_answerable)

    citation_precision = unsupported_rate = None
    if run_generation:
        judged = [o for o in outcomes if o.verdict is not None]
        if judged:
            citation_precision = sum(1 for o in judged if o.verdict in
                                     ("grounded", "no_matching_evidence")) / len(judged)
            unsupported_rate = sum(1 for o in judged if o.unsupported_claims) / len(judged)

    return EvalReport(
        total=len(outcomes),
        answerable_total=len(answerables),
        unanswerable_total=len(unanswerables),
        recall_at_k_hits=recall_hits,
        recall_at_k=recall_hits / len(answerables) if answerables else 0.0,
        recall_any_hits=recall_any_hits,
        recall_any=recall_any_hits / len(answerables) if answerables else 0.0,
        abstention_hits=abstain_hits,
        abstention_rate=abstain_hits / len(unanswerables) if unanswerables else 0.0,
        gate_accuracy=sum(1 for o in outcomes if o.gate_correct) / len(outcomes),
        p50_latency_ms=int(statistics.median(latencies)),
        p95_latency_ms=latencies[max(0, int(len(latencies) * 0.95) - 1)],
        citation_precision=citation_precision,
        unsupported_claim_rate=unsupported_rate,
        outcomes=outcomes,
    )
