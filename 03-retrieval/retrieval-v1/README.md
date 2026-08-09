# Retrieval v1 (Milestone 1 proof of concept)

BM25 (SQLite FTS5) + one-hop entity expansion, built against the 10-paper
milestone-1 test set. Why SQLite+FTS5 instead of Ladybug for this
milestone, and what was found while building it, is in [RESULTS.md](./RESULTS.md).
Windows/WSL command issues hit along the way, and their fixes, are in
[../wsl-windows-notes.md](../wsl-windows-notes.md) — read that first if a
command below doesn't behave the way you'd expect.

## Files

| File | What it does |
|---|---|
| `build_graph.py` | Loads `papers.jsonl` / `claims.jsonl` into `retrieval.sqlite` (FTS5 + entity graph) |
| `retrieve.py` | `retrieve(question)` — BM25 search + one-hop entity expansion |
| `evaluate.py` | Scores `retrieve()` against `questions.jsonl` (recall@10, abstention) |
| `generate.py` | `generate(prompt, server)` — calls the local Gemma stand-in via `llama-server` |
| `prompt.py` | Formats retrieved claims into a `[E1]`/`[E2]`-tagged prompt |
| `validate.py` | Checks the model's answer for invented tags / unsupported numbers |
| `run_pipeline.py` | Full chain: retrieve → prompt → generate → validate, for a chosen set of questions |

## One-time setup

You need three things in place before any of this runs. All three are
one-time — do them once, then skip straight to "Running the tests" from now on.

**1. WSL2 Ubuntu-22.04 with build tools.**
Follow [milestone1.md Task 1](../milestone1.md#1-set-up-the-machine). If a
command from a Windows terminal misbehaves (hangs, empty variables, weird
errors), check [wsl-windows-notes.md](../wsl-windows-notes.md) before
assuming something is broken — it almost certainly isn't.

**2. The Gemma 3 270M stand-in model**, downloaded to `~/mufasa/models/`:

```bash
wget -P ~/mufasa/models https://huggingface.co/ggml-org/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf
```

**3. `llama.cpp` built**, at `~/mufasa/llama.cpp/build/bin/`:

```bash
cd ~/mufasa && git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release -j$(nproc)
```

(Full detail on both, including expected download sizes and build time, is
in [milestone1.md](../milestone1.md#2-download-the-stand-in-model).)

All commands below assume you're either already inside a WSL shell, or
running from Windows with the `wsl-windows-notes.md` pattern:

```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -- python3 "/mnt/c/Users/hp/Desktop/Mufasa/03-retrieval/retrieval-v1/<script>.py"
```

The rest of this doc just shows `<script>.py`; substitute the wrapper above
if you're driving WSL from Windows rather than a native WSL terminal.

## Running the tests, step by step

### Step 1 — build the graph

```bash
cd 03-retrieval/retrieval-v1
python3 build_graph.py
```

Expect: `Loaded 10 papers, 112 claims, 196 distinct entities, 273 claim-entity links`.
This creates `retrieval.sqlite` in this folder (git-ignored — it's a
generated artifact, rebuild it any time from the two `.jsonl` files).

### Step 2 — score retrieval on its own (no model needed, fast)

```bash
python3 evaluate.py
```

This runs all 30 questions from `questions.jsonl` through `retrieve()` only
(no generation) and reports:
- **Recall@10** on the 25 answerable questions — did the right claim come back?
- An **abstention accuracy** number, calibrated on this same small question
  set — read the caveat printed with it and in RESULTS.md before trusting it
  as a real metric; it's a debugging signal at this corpus size, not a
  generalizable score.

Expect something close to the numbers in RESULTS.md — recall@10 should be
25/25, abstention should be poor (this is the actual finding, not a bug —
see RESULTS.md for why).

### Step 3 — run the full pipeline (retrieve → generate → validate)

This needs the model + `llama.cpp` from setup step 2/3 above, and takes
longer (a few seconds per question, since it starts `llama-server` and
calls it once per question):

```bash
python3 run_pipeline.py
```

With no arguments this runs a fixed sample: the 5 unanswerable questions
(`Q-026`–`Q-030`) plus 3 answerable ones (`Q-001`, `Q-009`, `Q-023`) as a
sanity check. To run specific questions instead:

```bash
python3 run_pipeline.py --ids Q-005 Q-012 Q-028
```

For each question it prints the model's answer and the validation result
(`pass`, `abstained`, `fail_invented_tag`, or `fail_unsupported_number`).

**Known limitation, not a bug to chase:** the Gemma 3 270M stand-in is too
weak to reliably follow the "cite your sources or say no evidence"
instruction — it often just echoes the evidence block back verbatim
instead of answering. That's expected (see RESULTS.md) and is a property of
the tiny stand-in model, not of `retrieve.py`/`validate.py`. Re-run against
a stronger model (e.g. Qwen3-1.7B) for a real read on whether the
validation logic itself works.

## If something doesn't work

1. `retrieval.sqlite` missing or stale → re-run `build_graph.py`. It's safe
   to re-run any time; it deletes and rebuilds the file from scratch.
2. `run_pipeline.py` hangs or errors connecting to `llama-server` → check
   nothing else is already using port 8734:
   `pkill -9 llama-server` then try again.
3. Anything involving `wsl -d Ubuntu-22.04 --` behaves strangely (empty
   variables, garbled arguments, `mkdir` failing on an empty path) → this is
   almost always the Windows/WSL quoting issue, not your command. See
   [wsl-windows-notes.md](../wsl-windows-notes.md).
