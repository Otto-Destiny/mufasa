# Milestone 1 — Retrieval v1 Results

Per [milestone1.md](../milestone1.md) "What to send back."

## 1. Offline test

Not run on a clean, network-disabled Ubuntu box yet — that's still outstanding. Lower-risk
than the Ladybug path though: `build_graph.py` / `retrieve.py` use only `sqlite3`, `json`,
`os`, `re` from the standard library, no network calls, no extension binaries to load. Built
and run so far on WSL2 Ubuntu-22.04 (dev machine, not the judging environment).

## 2. Retrieval results (`evaluate.py`, run against `questions.jsonl`)

- **Recall@10: 25/25 (100%)** on answerable questions — every question found at least one
  expected claim in its top 10.
- **Abstention: 0/5** unanswerable questions correctly identified as "no evidence," even with
  the abstention threshold calibrated on this same 30-question set (best achievable: 25/30
  overall accuracy, which means the calibration sacrificed all 5 unanswerable questions to
  keep every answerable one — because 25 > 5, that's the accuracy-maximizing choice).

**Why abstention failed, not just "needs tuning":** `questions.jsonl` deliberately makes each
unanswerable question topically adjacent to real claims (see its `related_claim_ids` field,
absent from the toy example in milestone1.md). E.g. Q-026 asks which bacteria were found in
Bosso water samples — the corpus has physicochemical claims about those exact samples, just no
bacteriology. BM25 correctly finds the lexical overlap and answers anyway. A bag-of-words
relevance score cannot tell "same topic" apart from "actually contains the requested fact" —
that's a different kind of check, not a threshold problem.

## 3. Task 7 answers

**Is one hop enough?** Not tested yet — none of the 25 hits needed graph expansion to be
found (all 25 were direct FTS5 seed hits; the one-hop entity join never contributed a correct
answer at k=10 in this 112-claim corpus). Can't yet say whether that's "one hop is enough" or
"corpus too small to tell." Worth re-checking once the corpus is bigger.

**Do we need vectors at Gate 1?** BM25 alone got 100% recall@10 here. But this is 112 claims
with literal keyword overlap between questions and claim text — it's not testing paraphrase
or cross-lingual matching, which is what vectors are supposed to earn their place on
([retrieval-architecture.md:264](../retrieval-architecture.md#L264)). This result doesn't
answer the question either way; it just says BM25-alone is not obviously broken at this scale.

**Is the claims schema right?** No, not as drawn in retrieval-architecture.md. The fixed
`Material`/`Property`/`Application` node tables assume a materials-science domain (the RHA/
concrete example). The actual test corpus is water/health/environmental — entity types are
`WaterSample`, `Population`, `Organism`, `StatisticalModel`, `ContaminantPlume`, etc., which
don't map onto Material/Property/Application at all. `measurement` has 57 distinct key-shapes
across 112 claims (statistical model fit stats, geophysical resistivity ranges, mortality
percentages...) — nowhere near the clean `value/unit/baseline` the architecture doc assumes.
v1 here stores entities as generic `Entity{name, type}` with `role` on the edge, and
`measurement`/`conditions` as opaque JSON, rather than forcing them into fixed columns. Also:
only 40/112 claims (36%) even have a `direction` field, and the values present
(`well_highest`, `Sdec_better_than_Kalman_better_than_random`, `east-west`...) are free text,
not the `increases | decreases | no_effect | inconclusive` enum from the schema.

**Ryu vs Ladybug:** Not checked yet — the 20 minutes mentioned in milestone1.md hasn't been
spent. Worth doing before the full-corpus build starts, independent of this milestone's SQLite
detour.

## What v1 does not do yet

- No vector search (not needed to hit the recall target here; see above).
- No merge/rank logic beyond BM25 score + a flat expansion penalty — no field filtering, no
  StudyFamily-aware deduplication (architecture doc's "one experiment counted once").

## 4. Full pipeline test: retrieve -> generate -> validate

Built `generate.py` (`generate(prompt, server) -> str`, backed by `llama-server`'s
`/v1/chat/completions`), `prompt.py` (formats retrieved claims into `[E1]`/`[E2]`-tagged
evidence per the architecture doc's "What goes to the model"), and `validate.py` (checks the
model didn't invent a tag and that any number it states appears in the evidence it cited for
it — see [retrieval-architecture.md:308-315](../retrieval-architecture.md#L308-L315)).

First attempt shelled out to `llama-cli` per-question and parsed its terminal transcript.
Dropped that: `llama-cli`'s interactive chat UI truncates long echoed lines inconsistently
(confirmed in the raw output — real `(truncated)` markers and 1700+ char lines), which made
isolating "just the model's reply" from banner/echo noise unreliable — several early results
were contaminated by the model's own build-info banner leaking into the extracted "answer"
and getting flagged as unsupported numbers. Switched to `llama-server`'s HTTP API instead:
clean JSON, chat template applied properly, no transcript to parse.

**Result on 8 questions (5 unanswerable + 3 answerable sanity checks), Gemma 3 270M:**

| Question | Expected | What happened |
|---|---|---|
| Q-001, Q-026, Q-028, Q-029, Q-030 | answer / abstain | Model didn't answer at all — copied the entire evidence block back verbatim instead of synthesizing anything |
| Q-009, Q-023 | should answer | **False abstention** — said "No matching evidence" despite good evidence being present |
| Q-027 | should abstain | Correctly abstained |

**Conclusion: this doesn't validate or refute the generation-time-validation approach — it
shows the 270M stand-in is too weak to test it with.** A model that can't reliably follow
"cite your sources or say no evidence" produces failures dominated by that incapacity, not by
whether the validation logic downstream is sound. Results are also not reproducible run-to-run
as tested: `llama-server` defaults to temperature 0.8 (sampling), so the same question can get
a different answer on a second run — worth pinning to greedy decoding (temperature 0) for any
future comparison to be meaningful.

**Known bug, not yet fixed:** `validate.py` flags citation-header digits (page numbers,
paper-ID fragments like the "088" in "P-G088") as unsupported when the model echoes the
`[E1] (paper ..., page ...)` formatting verbatim — that's copying my own prompt formatting
back, not an invented fact. Noise in the verdicts above, not a real hallucination signal.

**Recommended next step:** rerun this exact pipeline (unchanged) against Qwen3-1.7B once it's
downloaded — it's the doc's "nearer to what we'll ship" stand-in and should have enough
instruction-following capability to actually exercise the validation logic instead of masking
it.
