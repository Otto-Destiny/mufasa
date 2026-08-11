# Retrieval

This layer builds an offline evidence package and retrieves compact, traceable scientific evidence for each question before generation.

![MUFASA GraphRAG retrieval architecture](./images/retrieval-architecture.svg)

See [retrieval-architecture.md](./retrieval-architecture.md) for one worked proposal covering build/runtime separation, ranking, validation and evaluation. **It is a draft, not an agreed architecture** — no schema has been written, the data model and database are still open, and the document is pending overhaul. Do not implement from it.

See [licence-tiers.md](./licence-tiers.md) for how papers we may not quote are still represented, so the graph does not imply research is absent when it is only unquotable.

See [entity-resolution-design.md](./entity-resolution-design.md) for the proposed reusable entity-resolution module. It is a draft awaiting review and is not yet approved for implementation.

Generated graph databases, vector indexes and parsed corpora are release artifacts and are not stored in Git.
