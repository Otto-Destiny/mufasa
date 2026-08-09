# Milestone 1 — Retrieval Layer

**For:** the engineer taking on Layer 3 (retrieval).

Competition: **The Laptop LLM Challenge** (ADTC 2026). Gate 1 closes **25 August 2026**.

---

## Tasks

- [ ] **1.** Set up the machine
- [ ] **2.** Download the stand-in model
- [ ] **3.** Install llama.cpp and run the model
- [ ] **4.** Load the test papers, claims and questions
- [ ] **5.** Implement the first version of the retrieval system
- [ ] **6.** Prove retrieval works end to end
- [ ] **7.** Tell me what you'd do differently

---

## How to read this

**This is a guideline, not a specification. What I care about is results.**

Everything below was reasoned out on paper — no one has built any of it yet. Treat the steps as one route that should work, not the route. If you see a faster, simpler or smarter way to get a working retrieval system, take it. Think outside the box; I'd rather you arrive somewhere good by a road I didn't imagine than follow my directions to somewhere mediocre.

Two things I'd ask:

- Tell me when you change direction, so the rest of the system stays in step.
- Read the cautions below first. They are the only part of this document that isn't negotiable, and every one of them was found the hard way while researching this.

---

## ⚠ Key cautions — read before writing code

These came out of digging into the rules and the tooling. Each one is something that looks fine on your machine and fails on the judges' laptop.

### Things that end our run

| Caution | Detail |
|---|---|
| **7 GB peak RAM ceiling** | Not a penalty — exceeding it scores **zero** and disqualifies. Measured as **maximum RSS during audit** |
| **Zero network calls at run time** | The rules say *"100% offline with zero external network dependencies"*. One stray fetch — a CDN font, a telemetry ping, an auto-update — is the same as failing |
| **llama.cpp with GGUF only** | *"All submissions must run through llama.cpp using GGUF weights."* No Colibri, no AirLLM, no disk-streaming runtimes, however impressive their demos look |

### The Ladybug traps

The graph database downloads its full-text and vector features **over the internet** on first use. That is the single most likely way this layer fails at judging time.

| Caution | Detail |
|---|---|
| **Never call `INSTALL` in shipped code** | It fetches from Ladybug's server. Use `LOAD EXTENSION '<local path>'` with binaries you bundled. Check first-run helpers and `try/except` blocks too — grep for it before handing anything over |
| **`LOAD` is session-scoped** | Not a one-time setup that bakes into the database folder. Every process start must load the extensions again, or it works once and breaks on relaunch |
| **You need Linux x86-64 binaries** | You may develop on Windows; judging is Ubuntu 22.04. Windows extension binaries are useless there |
| **Extension version ≠ package version** | Ladybug `0.18.2` ships extension version `0.18.1`. Don't infer one from the other |
| **Fetch those binaries now, not later** | Kuzu's extension server went down and was never replaced. Ladybug is a young project running its own. If it goes dark before 25 August we cannot build at all |
| **Buffer pool defaults to ~80% of system RAM** | On an 8 GB machine that's ~6.4 GB. It's a cap rather than an upfront grab, but it must be set explicitly or peak RAM becomes unpredictable |
| **Test with an empty home directory and no cache** | An extension cached in your home folder makes a broken package look like a working one. This is how you'd pass every local test and still fail |

### Measurement and hygiene

| Caution | Detail |
|---|---|
| **Measure RSS, not PSS** | The judges record *maximum RSS*. PSS divides shared memory between processes and always reads lower — tune against it and you'll discover the gap on the judging laptop. PSS is useful for finding *where* memory goes, nothing else |
| **No PDFs, models or databases in git** | `.gitignore` already blocks them, and we may not hold redistribution rights on published papers |
| **One database, not two** | A second vector store means two ID spaces to keep in sync — a bug factory on a deadline. If you think we genuinely need one, let's talk about it early rather than find out late |

Everything outside these boxes is yours to redesign.

---

## 1. Set up the machine

Judging happens on **Ubuntu 22.04, Intel i5, 8 GB RAM, integrated graphics**. Getting close to that matters more than getting it exact — a VM or a spare laptop is fine. Windows + WSL2 is fine for day-to-day work, but numbers you report should come from real Ubuntu, since RAM accounting and thermal behaviour both differ under WSL.

If you're driving WSL from a Windows terminal (rather than working inside the WSL shell directly), see [wsl-windows-notes.md](./wsl-windows-notes.md) first — command quoting breaks in non-obvious ways crossing that boundary, and it covers the fixes so you don't have to rediscover them.

```bash
sudo apt update
sudo apt install -y build-essential cmake git python3 python3-pip python3-venv \
                    lm-sensors time curl
sudo sensors-detect --auto     # enables CPU temperature reading
```

Python 3.10 or newer:

```bash
python3 --version
```

---

## 2. Download the stand-in model

This is a placeholder so you have something real to build against. Ours is still training, and this one's answer quality isn't a signal about anything.

**Nanbeige 4.2 (3B):**
https://huggingface.co/bartowski/Nanbeige_Nanbeige4.2-3B-GGUF

**Qwen3 1.7B:**
https://huggingface.co/Qwen/Qwen3-1.7B-GGUF

**Optional lightweight test model — Qwen3.5 0.8B:**
https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF

Use its `Q4_0` GGUF (about 563 MB). It runs in a current standard llama.cpp build without any custom runtime. This is useful for testing the smallest practical offline configuration; allow roughly 1–2 GB of working RAM at a 4K context.

**Optional smallest model — Gemma 3 270M Instruct:**
https://huggingface.co/ggml-org/gemma-3-270m-it-GGUF

Take the `Q8_0` GGUF (292 MB). At 270M parameters, heavier quantisation saves almost nothing — the whole file is under 300 MB either way — so `Q8_0` is the sensible choice and stays close to full precision. Note that most of those parameters are the 262k-token embedding table rather than the transformer itself, which is why the file doesn't shrink the way you'd expect.

Its answers will be noticeably worse than the others. That's fine and it's the point: it starts in about a second and leaves the machine almost entirely free, so it's the fastest way to iterate on retrieval code without waiting on generation. It's also a useful lower bound — if retrieval brings back the right evidence, a 270M model should still be able to quote it back correctly. When it can't, that usually means retrieval handed it the wrong thing.

For Nanbeige and Qwen3-1.7B, take the `Q4_K_M` build — that's the quantisation we plan to ship. For the optional Qwen3.5-0.8B test, take `Q4_0`. For the optional Gemma 3 270M test, take `Q8_0`.

| Model | Suggested GGUF file | Expect peak RAM |
|---|---|---|
| Nanbeige4.2-3B | 2.68 GB | ~3.0–3.2 GB |
| Qwen3-1.7B | ~1.1 GB | ~1.4–1.6 GB |
| Qwen3.5-0.8B (`Q4_0`) | ~563 MB | ~1–2 GB at 4K context |
| Gemma 3 270M Instruct (`Q8_0`) | 292 MB | ~0.6–1.0 GB at 4K context |

Get the two main stand-ins. Despite its name Nanbeige "3B" is really ~4B parameters and wants about twice the RAM of our eventual model — useful for feeling a tight machine. Qwen3-1.7B is nearer what we'll ship, so tune against that one. Qwen3.5-0.8B is an optional lightweight comparison. Gemma 3 270M is the optional smallest one — reach for it when you want a fast edit-run loop rather than a realistic answer.

With a 1.7B the laptop budget is roughly: 1.5 GB model + 0.5 GB retrieval + 1.5 GB for Ubuntu and the app ≈ 3.5 GB of the 7 GB. More headroom than you'd think — worth knowing before optimising for memory.

If your numbers later suggest a different quantisation trades better, that's a real finding. Send it.

Easiest by browser: open either link, click **Files**, download the `Q4_K_M` file. From the terminal instead:

```bash
pip install -U "huggingface_hub[cli]"
mkdir -p ~/mufasa/models && cd ~/mufasa/models
```

Repos differ on capitalisation (`Q4_K_M` vs `q4_k_m`) and on how they split large files, so list what's actually there before downloading:

```bash
python3 -c "from huggingface_hub import list_repo_files; \
print('\n'.join(f for f in list_repo_files('bartowski/Nanbeige_Nanbeige4.2-3B-GGUF') if f.endswith('.gguf')))"
```

Then pull the exact filename you want:

```bash
hf download bartowski/Nanbeige_Nanbeige4.2-3B-GGUF <exact-filename>.gguf --local-dir .
hf download Qwen/Qwen3-1.7B-GGUF <exact-filename>.gguf --local-dir .
hf download ggml-org/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q4_0.gguf --local-dir .
hf download ggml-org/gemma-3-270m-it-GGUF gemma-3-270m-it-Q8_0.gguf --local-dir .
```

Gemma 3 270M is small enough to just fetch directly:

```bash
wget https://huggingface.co/ggml-org/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf
```

If that repo is ever unavailable, [`unsloth/gemma-3-270m-it-GGUF`](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF) carries the same `Q8_0` quantisation under the same filename (same weights, marginally different GGUF metadata).

Confirm the file size roughly matches the table above. If a repo splits a quant across several parts, take all of them — llama.cpp loads the first and finds the rest.

---

## 3. Install llama.cpp and run the model

llama.cpp is the program that actually runs the model file. You'll come across alternatives like Colibri or AirLLM that promise big memory savings — I looked; they're out. The rules require llama.cpp, and they buy memory by spending speed, which is backwards for us: speed is 30 marks, efficiency is 20, and we aren't short of memory.

```bash
cd ~/mufasa
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j$(nproc)
```

That gives you `llama-cli`, `llama-server` and `llama-bench` in `build/bin/`.

Confirm it runs:

```bash
./build/bin/llama-cli \
  -m ~/mufasa/models/<your-model>.gguf \
  -p "Explain what rice husk ash is used for in concrete." \
  -n 128 -t 4 -c 2048 --no-warmup
```

Two flags carried over from our resource governor, both of which are guesses waiting to be measured:

- `-t 4` — thread cap. Running flat out overheats the laptop, and a thermal trip costs 10 marks.
- `-c 2048` — context cap, i.e. how much text the model holds at once. We think 2048 is enough for 6–10 evidence records; if it isn't, that changes the design upstream and I want to know.

You don't need to tune or measure the model itself — that's the model-engineering side, covered in [02-model-engineering/milestone1.md](../02-model-engineering/milestone1.md). Here it just needs to run so you can see retrieval working end to end.

---

## 4. Load the test papers, claims and questions

Download the source PDFs from Google Drive. The three JSONL files are version-controlled in this repository:

- **Source PDFs (`papers/`)** — [download the ten test PDFs from Google Drive](https://drive.google.com/drive/folders/1I_zHPKlfBvH70H3hLBWMS2mB5ItUFOAX?usp=sharing)
- **Paper catalogue** — [`03-retrieval/milestone1-test-data/papers.jsonl`](./milestone1-test-data/papers.jsonl)
- **Extracted claims** — [`03-retrieval/milestone1-test-data/claims.jsonl`](./milestone1-test-data/claims.jsonl), one claim per line
- **Test questions and gold answers** — [`03-retrieval/milestone1-test-data/questions.jsonl`](./milestone1-test-data/questions.jsonl)

All three JSONL files are together in [`03-retrieval/milestone1-test-data/`](./milestone1-test-data/).

You don't need to write the questions — they come with known answers already attached, so you can score retrieval automatically instead of eyeballing it.

Keep the PDFs local and out of commits — `.gitignore` already blocks `*.pdf`, and redistribution rights on published papers are a question we don't need to open.

A line of `claims.jsonl`:

```json
{
  "id": "OBS-0001",
  "text": "The 10% RHA mix achieved 31.2 MPa compressive strength at 28 days.",
  "material": "rice husk ash",
  "property": "compressive strength",
  "direction": "increases",
  "value": 31.2,
  "unit": "MPa",
  "baseline": "OPC control mix",
  "conditions": {"replacement_pct": 10, "curing_days": 28},
  "quote": "The 10% RHA mix achieved 31.2 MPa at 28 days.",
  "page": 8,
  "section": "Results",
  "paper_id": "P-1024"
}
```

This maps onto the `Observation`, `EvidenceSpan` and `Paper` tables in [retrieval-architecture.md](./retrieval-architecture.md). The shape exists so the extraction side and the retrieval side meet in the middle — but it was designed before anyone tried to retrieve against it. If a field is useless, or something obvious is missing, now is the cheapest possible moment to change it. Tell me and we'll change it.

A line of `questions.jsonl`:

```json
{
  "id": "Q-001",
  "question": "Can rice husk ash replace part of the cement in concrete?",
  "expected_claim_ids": ["OBS-0001", "OBS-0007"],
  "answerable": true
}
```

Some have `"answerable": false` and an empty `expected_claim_ids` — those are deliberate. The system should say it has no matching evidence rather than reach for something loosely related.

Small set on purpose. Enough to prove retrieval works; the real corpus is 6,000 papers and comes later.

---

## 5. Implement the first version of the retrieval system

This is the real build: claims into a graph, indexes on top, queries coming back with evidence attached. How you structure it is yours — this is the part of the project where good engineering judgement shows up.

One thing worth designing in from the start rather than bolting on later: it has to run on a laptop with no internet. Leaving that to the end is how this layer fails, because the failure only appears on a machine you don't own.

The Ladybug cautions above are the ones that bite here. [retrieval-architecture.md](./retrieval-architecture.md) has a section called *"Offline packaging and startup contract"* with the details worked out, so you don't have to rediscover them. A sequence that should work:

1. Pin an exact Ladybug version and record it.
2. Fetch the Linux x86-64 extension binaries (`libfts.lbug_extension`, `libvector.lbug_extension`) **today**, while the server is definitely up, and commit them.
3. Load them by explicit path:
   ```cypher
   LOAD EXTENSION '/your/package/path/extensions/libfts.lbug_extension';
   LOAD EXTENSION '/your/package/path/extensions/libvector.lbug_extension';
   ```
4. Set the buffer pool explicitly. `256 * 1024 * 1024` bytes is a starting guess, not a conclusion — it trades RAM against query latency, and you'll be the first person able to see that curve. Find the right number and tell me what it is.
5. Prove it on a clean Ubuntu box, networking off, empty home directory.

Then load the claims and build the graph and indexes.

**Ladybug is a choice, not a law.** It earned its place by holding the graph, full-text and vectors in one engine, so a vector hit *is* a graph node you walk out from. But there's a fork called **Ryu** that claims full-text and vector are compiled in with no external dependencies — if that's true, most of steps 2 to 5 simply disappear. Nobody has checked. Twenty minutes on their docs against fifty lines of packaging risk looks like a good trade, and I'd rather you spent it than assume I got this right.

---

## 6. Prove retrieval works end to end

Run `questions.jsonl` through the system. Because each question carries the claim IDs it should find, you can score this in a loop rather than by hand — and re-score it after every change.

For each question, check:

1. Did the right claim come back in the top 10?
2. Does the returned span actually support it?
3. On an unanswerable question, does the system say *"no matching evidence in this corpus"* instead of inventing something?

The third matters as much as the other two. An honest gap is a feature we're deliberately building, not a failure to hide.

How you get there is yours. The architecture doc proposes BM25 plus vectors plus a one-hop graph walk, but the weighting between channels, how results get merged and ranked, when to filter by field — none of that has been tried. If a simpler arrangement does the job, simpler wins.

### If the stand-in model gives poor answers

Expected, and not worth your time — it isn't fine-tuned and it isn't ours.

If weak generation is making it hard to tell whether *retrieval* is working, put a free API behind the generation step temporarily (Gemini's free tier is fine). Retrieval stays local. Two things to keep true:

- Keep it behind an interface — `generate(prompt) -> str` — with local llama.cpp as the default, so switching back is one line.
- It never ships. The submission runs fully offline; this is scaffolding.

---

## 7. Tell me what you'd do differently

Genuinely part of the milestone, not a courtesy. Some things I'd particularly like your read on:

- **Is one hop enough?** The design commits to a single graph hop, which also means a plain SQL join could do most of this. If one hop really is enough, is the graph database earning its place?
- **Do we need vectors at Gate 1?** BM25 plus a graph walk needs no model, no extension and no network install. Vectors add capability and add risk. Where's the line?
- **Is 256 MB the right buffer pool?** Nobody has measured the latency cost yet.
- **Is the claims schema right?** You'll be the first to query it in anger.
- **Ryu vs Ladybug** — see the end of Task 6.

And anything in [retrieval-architecture.md](./retrieval-architecture.md) that turned out to be wrong once you built it. That document is a plan written from research, not from measurement. Parts of it will be wrong and I'd rather find out from you in week one than from a judge in September.

---

## What to send back

1. Whether the offline test passed on a clean, network-disabled Ubuntu box.
2. Retrieval results across `questions.jsonl` — how many found the right claim, and how the unanswerable ones behaved.
3. Your answers to Task 7, or at least the ones you formed an opinion on.

---

Before you hand anything over, re-read the **Key cautions** at the top. They're the only lines in this document I'd ask you not to improvise around — everything else is a proposal waiting for someone to build it and find out where it's wrong.

Results are what count. Get there however works.
