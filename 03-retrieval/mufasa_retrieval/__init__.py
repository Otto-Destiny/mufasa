"""MUFASA retrieval layer.

Build plane: :func:`build.build` turns extracted records into one SQLite
evidence store with full-text and dense indexes.

Runtime plane: :func:`pipeline.answer` takes a question to a validated, cited
answer, or to an honest statement of what corpus v1 does and does not cover.
"""

from .build import BuildStats, build, connect, manifest
from .bundle import EvidenceBundle, EvidenceRecord, build_bundle
from .evaluate import EvalReport, evaluate, load_questions
from .gate import GateDecision, decide
from .generate import Generator, LlamaServerGenerator, StubGenerator, get_generator
from .pipeline import AnswerResult, answer, answer_stream
from .search import Claim, Hit, SearchResult, load_claims, search
from .validate import ValidationReport, validate

__all__ = [
    "AnswerResult",
    "BuildStats",
    "Claim",
    "EvalReport",
    "EvidenceBundle",
    "EvidenceRecord",
    "GateDecision",
    "Generator",
    "Hit",
    "LlamaServerGenerator",
    "SearchResult",
    "StubGenerator",
    "ValidationReport",
    "answer",
    "answer_stream",
    "build",
    "build_bundle",
    "connect",
    "decide",
    "evaluate",
    "get_generator",
    "load_claims",
    "load_questions",
    "manifest",
    "search",
    "validate",
]

__version__ = "0.1.0"
