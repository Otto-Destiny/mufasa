# Testing

## Run everything

```bash
python -m uv sync
python -m uv run pytest
```

## What each layer proves

| Suite | Command | Proves |
|---|---|---|
| Retrieval unit | `pytest 03-retrieval/tests` | normalisation, aliases, entity channel, RRF, diversity |
| Gate | `test_gate.py` | Q-026 and all unanswerable questions abstain |
| Validator | `test_validate.py` | invented tags, unsupported numbers, citation-furniture masking |
| Quality gates | `test_quality_gates.py` | recall floors, abstention 5/5, citation precision with stub generator |
| Build | `test_build.py` | same JSONL → identical db hash |
| Governor / config / TTS | `04-application/backend/tests` | single-flight, model registry, speakable rewriting |
| API | `test_api.py` | SSE stages, sources without quotes/tags, 409 when busy |

## Quality gate commands

```bash
python -m uv run mufasa-build --out packages/corpus_v1/mufasa.db
python -m uv run mufasa-eval --db packages/corpus_v1/mufasa.db
# with stub generation + validation:
# PowerShell:
$env:MUFASA_GENERATOR="stub"; python -m uv run mufasa-eval --db packages/corpus_v1/mufasa.db --generate
```

## Markers

- `slow` — real GGUF / Piper (opt in: `pytest -m slow`)
- `packaging` — desktop bundle checks

## Reading a failed gate

`mufasa-eval` prints each failing question id with recall and gate reason. Fix ranking or the property lexicon before loosening the assertion in `test_quality_gates.py`.
