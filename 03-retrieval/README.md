# Retrieval

This layer builds an offline evidence package and retrieves compact, traceable scientific evidence for each question before generation.

## Implementation (v2)

Python package: [`mufasa_retrieval/`](./mufasa_retrieval/). SQLite + FTS5, int8 vectors in-table, two-axis coverage gate, citation validator. See root [SETUP.md](../SETUP.md) and [TESTING.md](../TESTING.md).

```bash
python -m uv run mufasa-build --out packages/corpus_v1/mufasa.db
python -m uv run mufasa-eval --db packages/corpus_v1/mufasa.db
python -m uv run pytest 03-retrieval/tests
```

![MUFASA GraphRAG retrieval architecture](./images/retrieval-architecture.svg)

See [retrieval-architecture.md](./retrieval-architecture.md) for the earlier draft proposal. **Ship from the code in this folder**, not from that draft — schema and engine choices were revised against the 10-paper fixture (SQLite over Ladybug for Gate packaging risk).

See [licence-tiers.md](./licence-tiers.md) for how papers we may not quote are still represented.

See [entity-resolution-design.md](./entity-resolution-design.md) for the proposed reusable entity-resolution module. It is a draft awaiting review and is not yet approved for implementation.

Generated graph databases, vector indexes and parsed corpora are release artifacts and are not stored in Git.
