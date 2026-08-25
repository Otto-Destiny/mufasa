"""Build-plane command line. Both commands are offline and idempotent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .build import build, connect, manifest
from .embed import get_embedder
from .evaluate import evaluate, load_questions
from .generate import get_generator

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "milestone1-test-data"


def build_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mufasa-build", description="Build the evidence store.")
    ap.add_argument("--claims", default=str(DEFAULT_DATA / "claims.jsonl"))
    ap.add_argument("--papers", default=str(DEFAULT_DATA / "papers.jsonl"))
    ap.add_argument("--aliases", default=str(Path(__file__).with_name("aliases") / "flagship.yaml"))
    ap.add_argument("--out", required=True, help="path to the .db file to write")
    ap.add_argument("--corpus-version", default="corpus_v1")
    ap.add_argument("--embed-backend", default=None, choices=[None, "hashing", "onnx"])
    args = ap.parse_args(argv)

    aliases = args.aliases if Path(args.aliases).exists() else None
    stats = build(
        claims_path=args.claims,
        papers_path=args.papers,
        db_path=args.out,
        corpus_version=args.corpus_version,
        embed_backend=args.embed_backend,
        aliases_path=aliases,
    )
    conn = connect(args.out)
    man = manifest(conn)
    conn.close()
    print(f"built {args.out}")
    for key, value in stats.__dict__.items():
        print(f"  {key:<10} {value}")
    print(f"  embedder   {man.get('embedder')}")
    print(f"  facets     {man.get('facet_vocabulary')}")
    return 0


def eval_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mufasa-eval", description="Score retrieval and the gate.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--questions", default=str(DEFAULT_DATA / "questions.jsonl"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-vectors", action="store_true")
    ap.add_argument("--generate", action="store_true",
                    help="also run generation and validation (needs a model unless "
                         "MUFASA_GENERATOR=stub)")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)

    conn = connect(args.db)
    report = evaluate(
        conn,
        load_questions(args.questions),
        k=args.k,
        embedder=get_embedder(),
        use_vectors=not args.no_vectors,
        generator=get_generator() if args.generate else None,
        run_generation=args.generate,
    )
    conn.close()

    print(report.summary())
    failures = [o for o in report.outcomes if not (o.recall_ok and o.gate_correct)]
    if failures:
        print("\nfailures:")
        for o in failures:
            why = []
            if not o.recall_ok:
                why.append(f"recall {o.hits_found}/{o.minimum_expected_hits}")
            if not o.gate_correct:
                why.append(f"gate said answerable={o.gate_answerable} ({o.gate_reason})")
            print(f"  {o.question_id}  {'; '.join(why)}")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(build_main())
