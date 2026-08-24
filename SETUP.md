# Setup

MUFASA retrieval + laptop-local app on branch `feat/app-and-retrieval-v2`.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`python -m pip install uv` is fine)
- Optional later: Node + Rust for the Tauri desktop shell; Piper binary for voice

## One-time install

```bash
cd Mufasa
python -m uv sync
cp 04-application/backend/.env.example 04-application/backend/.env
```

## Build the fixture corpus

```bash
python -m uv run mufasa-build --out packages/corpus_v1/mufasa.db
python -m uv run mufasa-eval --db packages/corpus_v1/mufasa.db
```

Point the app at that database (already the default in `.env.example`):

```text
MUFASA_DB=../../packages/corpus_v1/mufasa.db
MUFASA_GENERATOR=llama-server
```

Place the GGUF under `04-application/backend/models/` (filename from `models.toml`) and the `llama-server` binary at `04-application/backend/bin/llama-server`. On start, the app launches llama-server when both are present. If either is missing, it answers from evidence only until you add them. No code edits.

To force evidence-only mode:

```text
MUFASA_GENERATOR=stub
```

## Run the app

```bash
cd 04-application/backend
python -m uv run mufasa-serve
```

Open http://127.0.0.1:8756 — Ask, Compare, Library, Statistics, History, Integrity, Settings. Evidence cards appear only after validation.

## Papers Drive folder

Full-text PDFs (not in git): https://drive.google.com/drive/u/0/folders/1I_zHPKlfBvH70H3hLBWMS2mB5ItUFOAX

## Windows notes

Prefer PowerShell or WSL2. Measured RSS / thermal numbers for Gate reports should come from Ubuntu 22.04, not WSL.
