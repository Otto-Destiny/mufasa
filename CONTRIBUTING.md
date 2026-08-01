# Contributing to MUFASA

MUFASA uses a small pull-request workflow so changes remain easy to review.

## Workflow

1. Start from the latest `main`.
2. Create one focused branch; never work directly on `main`.
3. Commit and push that branch.
4. Open a pull request to `main` and complete the short template.
5. Address review comments. Only `@Otto-Destiny` merges into `main`.

Draft pull requests are welcome for unfinished work. Delete the branch after it is merged.

## Branch names

Use a lowercase prefix and a short kebab-case description:

```text
feat/offline-search
fix/citation-links
docs/retrieval-design
chore/update-dependencies
```

Allowed prefixes are:

- `feat/` — a new capability
- `fix/` — a bug fix
- `docs/` — documentation only
- `chore/` — maintenance or dependencies
- `refactor/` — internal restructuring without a feature change
- `test/` — tests or evaluation fixtures
- `ci/` — automation and repository policy
- `data/` — data-pipeline or taxonomy work
- `model/` — model-training or evaluation work

The repository checks this rule automatically on pull requests.

## Before requesting review

- Keep the pull request limited to one clear purpose.
- Run the relevant tests or document the manual checks performed.
- Update documentation when behavior or architecture changes.
- Include measured model/runtime effects when a change can affect accuracy, speed, memory or thermals.
- Record source, licence and provenance changes for data work.
- Never commit credentials, `.env` files, model weights, PDFs, bulk datasets, generated indexes or run outputs.

Contributors leave merging to the repository owner. When the owner authors a change, it still goes through a pull request and all available checks before the owner merges it.
