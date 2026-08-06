# Retrieval

This layer builds an offline evidence package and retrieves compact, traceable scientific evidence for each question before generation.

![MUFASA GraphRAG retrieval architecture](./images/retrieval-architecture.svg)

See [retrieval-architecture.md](./retrieval-architecture.md) for the data model, build/runtime separation, ranking, validation and evaluation design.

See [licence-tiers.md](./licence-tiers.md) for how papers we may not quote are still represented, so the graph does not imply research is absent when it is only unquotable.

Generated graph databases, vector indexes and parsed corpora are release artifacts and are not stored in Git.
