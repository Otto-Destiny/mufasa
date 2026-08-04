# MUFASA — Competition & Project Context

**Living brief for EagleTeam.** Use this file anytime you need the full picture of the hackathon rules, what MUFASA is building, why African science, the domain, and product/stack decisions. Layer docs and measured numbers win if something here drifts.

| | |
|---|---|
| **Team** | EagleTeam |
| **Project** | MUFASA — Models for Understanding the Frontiers of African Scientific Advancement |
| **Competition** | Africa Deep Tech Challenge 2026 (ADTC) — The Laptop LLM Challenge |
| **Primary track** | Math & Scientific Reasoning |
| **Integration** | Offline GraphRAG over African scientific evidence |
| **Gate 1** | **25 August 2026** |

Authoritative design detail lives in [system-architecture.md](./system-architecture.md) and the four layer folders. Historical chat notes: [conversations.md](./conversations.md). Team briefing deck: `MUFASA-EagleTeam-ADTC2026-Briefing-v3.pptx` (2 August 2026).

---

## 1. What the competition is

ADTC 2026 is an **engineering-first** contest: build a useful language-model **application** that runs on the computers Africa already has — mid/low-end laptops — **without cloud inference**.

It is **not** a proposal-only contest. Gate 1 expects a working submission package.

**Official links**

- Challenge site: https://africadeeptech.org/challenge-2026/
- Devpost: https://adtc-2026.devpost.com/
- Submission template: https://github.com/Africa-Deep-Tech-Foundation/adtc-2026-submission-template
- Local profiler: https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler

**Team size:** 1–3 people.  
**Prizes (approx.):** Grand $8,000 + residency; Runner-up $4,000; Best Integration $3,000; Best Localisation $1,500; GPU credits for semis/finals. Pool often cited ~$16.5k–$20k+.

---

## 2. What you must build

1. A working **end-to-end, on-device** language-model system.
2. Choose **one** primary problem domain (see §4).
3. Demonstrate at least one **load-bearing cross-disciplinary integration** (see §5).
4. Run **100% offline** during evaluation (no outbound network once profiling starts).
5. Ship the scored model as **GGUF** running through **llama.cpp only**.

Screenshots / clips of it running and a **max 2-minute** video are part of Gate 1.

---

## 3. Hardware envelope (disqualify if broken)

| Component | ADTC Standard Laptop |
|---|---|
| CPU | Intel Core i5 10th–12th gen **or** AMD Ryzen 5 3000–5000 (x86-64) |
| RAM | 8 GB DDR4 |
| Graphics | Integrated only — **no discrete GPU** |
| Storage | 256 GB SSD |
| OS (reference) | Ubuntu 22.04 LTS |
| Price band | ~$400–$500 new / ~$150–$250 refurbished |

**Hard fails → score 0**

- Peak RAM over **7 GB** RSS, or OOM / crash
- Network / cloud calls during the evaluation window

`download_model.sh` may use the network **before** profiling. After that: offline only.

---

## 4. Problem domains (pick one)

MUFASA chose the first:

1. **Math & Scientific Reasoning** ← ours  
2. Healthcare & Medical  
3. Agriculture  
4. Creative Writing  
5. Coding Assistants  
6. Corporate / Enterprise  
7. Autonomous AI Agents  

In `metadata.json`, domain value is like `math_scientific_reasoning`.

---

## 5. Load-bearing cross-disciplinary integration

### Plain meaning

- **Cross-disciplinary:** the LLM is wired to **another real technical field**, not just “chat.”
- **Load-bearing:** that pairing actually carries weight. Remove it and the product gets clearly worse. A fake Sources tab does not count.

Official examples: offline RAG over agricultural records, edge sensing, geospatial analysis, local medical assistance.

### MUFASA’s pairing

| Piece | Role |
|---|---|
| **Primary track** | Math & Scientific Reasoning (the model) |
| **Other discipline** | Scientific evidence systems (papers → structured observations → offline graph/retrieval) |
| **Integration** | Offline GraphRAG over African science evidence |

**Without graph:** small model guessing.  
**With graph:** cited measurements, conditions, page-level quotes, honest corpus-scoped abstention.

Submission metadata expects something like:

```json
"cross_disciplinary_pairing": {
  "discipline": "...",
  "load_bearing": true,
  "description": "..."
}
```

---

## 6. Scoring — what carries the major score

### Published formula

```text
S_total = 0.50 × S_acc + 0.30 × S_perf + 0.20 × S_eff − P_thermal
```

| Component | Weight | Meaning |
|---|---|---|
| **S_acc** | 50% | Answer quality on prompts (your 2 + organizers’ 2 hidden in-domain prompts; plus broader judge/panel judgment) |
| **S_perf** | 30% | Generation speed. Profiler normalises vs `TPS_REFERENCE = 15.0` (site also discusses relative-to-fastest-team wording — use profiler as telemetry ground truth) |
| **S_eff** | 20% | `100 × (7 GB − peak_RSS_GB) / 7 GB` — unused RAM is marks |
| **P_thermal** | −10 | If CPU throttles or core temp **> 85 °C** |

**The automated profiler scores the `.gguf` alone** through llama.cpp — **no graph, no app** in that run.

### Bonuses / multipliers (panel / claims)

- **African language** meaningful support → often cited as **+15%** on panel score  
- **Budget laptop profile** claim → often cited as **+10%**  
- Separate prizes: **Best Integration**, **Best Localisation**

### So: model vs GraphRAG

| Goal | What matters most |
|---|---|
| Leaderboard formula (acc / speed / eff / thermal) | **The model (GGUF)** |
| Requirement + Gate 2–3 defense + Best Integration | **GraphRAG + app (real, not cosmetic)** |

**Priority if time is short:** strong, small, fast GGUF first; keep GraphRAG **thin but real** (retrieval actually changes answers and citations).

---

## 7. llama.cpp + GGUF only

- **GGUF:** on-device weight file format (`.gguf`), usually quantized (e.g. Q4_K_M).
- **llama.cpp:** the only accepted **runtime** for official evaluation.
- Train with whatever you want off-device; **export and package** for llama.cpp before submit.
- Do **not** commit weights to git. Provide a credential-free `download_model.sh`.

Judges may open weights with tools like LM Studio / Ollama in docs, but the **rule** for the evaluation framework is GGUF + llama.cpp.

---

## 8. Three gates (timeline)

| Gate | When | Deliverables |
|---|---|---|
| **Gate 1 — Submission** | **25 August 2026** | Public GitHub (ADTC template expectations), `REPORT.md`, screenshots/clips, 2-min video, model downloadable, bonus claims |
| **Gate 2 — Audit & Q&A** | 8–29 September 2026 | 30-min technical Q&A, reproducibility audit, clarifications |
| **Gate 3 — Final defence** | 17 October 2026 | Pitch deck (max 10 slides), live session, setup verification |

Devpost deadline text may show “Aug 24 11:45pm PDT” — treat **25 August 2026** as the Gate 1 calendar date used by the project.

---

## 9. Gate 1 submission shape (template)

Required shape from the official template:

```text
your-submission/
├── metadata.json       # team, domain, exactly 2 test_prompts, model fields
├── download_model.sh   # public, credential-free, idempotent → model/*.gguf
├── REPORT.md           # problem, design, constraints, measured benchmarks
├── model/              # downloaded; NOT committed
└── .gitignore          # ignore *.gguf and model/
```

Checklist highlights:

- Repo **public**
- Exactly **2** test prompts; organizers add **2 hidden**
- `model.runtime` = `llama.cpp`
- Quantization is a GGUF type
- `budget_laptop_claim`: true
- Local smoke test with `adtc-profiler` before submit

This MUFASA repo is the **project / architecture** home. Gate 1 packaging must still satisfy the ADTC template (either by aligning this repo or shipping a compliant submission repo that points at the same work).

---

## 10. What MUFASA is

Offline **African scientific-reasoning** system for constrained hardware:

- Compact LM (reasoning + grounding behaviour)
- Evidence-grounded **offline GraphRAG**
- Desktop app (primary) + compute-thin phone UI over local Wi-Fi
- All inference, retrieval, and persistent scientific data stay on the **laptop**

Declared pitch line:

```text
Primary track:  Math & Scientific Reasoning
Integration:    Offline GraphRAG over African materials / scientific evidence
```

---

## 11. Two planes (design rule)

```text
BUILD PLANE — your machines, hours/days, internet allowed
  papers → rights → parse → training datasets + evidence graph package

RUNTIME PLANE — judge laptop, milliseconds, no internet/cloud, 7 GB ceiling
  question → retrieve → generate → validate → cited answer
```

Anything that needs internet, a frontier teacher model, or serious memory belongs on the **build** plane.

---

## 12. Four layers

| Layer | Role | Plane |
|---|---|---|
| **01 Data** | Discover African papers, score relevance, rights, parse, extract scientific records | Build |
| **02 Model** | Datasets → SFT → DPO → evaluate → package GGUF | Build → one runtime file |
| **03 Retrieval** | Build offline evidence graph + indexes; query at runtime | Both |
| **04 Application** | Orchestrate limits, retrieval, llama.cpp, validation, UI | Runtime |

**One corpus, two products:** the same rights-cleared papers feed training data and the graph.

Detail:

- [01-data-engineering/](./01-data-engineering/)
- [02-model-engineering/](./02-model-engineering/)
- [03-retrieval/](./03-retrieval/)
- [04-application/](./04-application/)
- [system-architecture.md](./system-architecture.md)

---

## 13. Layer roles in one breath

### Model (what it does)

1. Reads the question  
2. Uses trained scientific-reasoning behaviour  
3. Optionally uses ~6–10 retrieved evidence snippets (`[E1]`…)  
4. Writes a short answer with citations / limitations  
5. **Must also work alone** — because the profiler scores the GGUF with no graph  

It does **not** own the corpus DB, graph query engine, or UI memory policy.

### GraphRAG (what it does)

- Stores measurements as **Observation nodes** (value, unit, baseline, conditions, quoted span, page) — not bare “IMPROVES” edges  
- At runtime: entity/alias → BM25 + vectors → one-hop graph → 6–10 observations → model → validate numbers/tags  
- Ships **quoted spans**, not PDFs (PDFs are huge and rights-risky)  
- Honest coverage: “not in corpus v1” ≠ “nobody studied this”

### Application (what it does)

- Owns orchestration, context/output/thread caps, **one generation at a time**, cancel, validation before display  
- Tauri desktop primary; phone is full UI but compute-thin over optional local Wi-Fi share  
- Views: Ask & Evidence, Compare Studies (disagreement matrix), Coverage & Sources  

---

## 14. EagleTeam briefing snapshot (deck v3)

Memory budget (**estimates until profiler replaces them**):

- Model ~1.2 GB  
- Retrieval < 0.4 GB  
- App ~0.5 GB  
- Headroom toward 7 GB is the efficiency score  

Model candidates discussed:

- **Nanbeige-3B** thinking-tuned (quality-per-MB favourite in deck)  
- **Bonsai 8B 1-bit** (~1.2 GB disk)  
- **Ternary Bonsai 1.58-bit** fallback  

Training stack: reasoning SFT → DPO on hard preference pairs → quantize/package with pinned chat template, stop tokens, decoding defaults.

Data targets (deck / plans):

- 150k+ Africa-relevant candidates indexed  
- **6,000** papers for Gate 1 graph (citation-first queues; include needs scored relevance, not African authorship alone)  
- Larger graph later  
- 200-paper human-reviewed gold classification benchmark  

Do-not-slip items from the deck:

1. Ladybug / graph **offline** spike early (bundle extensions; networking off)  
2. Rights-check and extract the Gate 1 corpus while training runs  
3. Package the GGUF path in week one (template, stops, reproducible llama.cpp settings)  
4. Replace every estimate with **ADTC profiler** numbers before `REPORT.md`

---

## 15. Data relevance rule (short)

A paper is Africa-relevant when an African material, organism, crop, population, environment, dataset, industrial system, constraint, method, or scientific problem is important to the work.

**African author affiliation alone is not enough.**

Scored dimensions (0–4 each): African centrality, local specificity, scientific depth, knowledge value, local applicability. Typical include: ≥ 14/20, centrality ≥ 2, depth ≥ 2, evidence in title/abstract. Uncertain → `review`, not silent include/exclude.

Taxonomy domains: `MAT`, `AGR`, `HLT`, `ENR`, `ENV`, `TEC` — see [01-data-engineering/taxonomy/african-science-categories.md](./01-data-engineering/taxonomy/african-science-categories.md).

---

## 16. Current repo status (as of context capture)

- This repository is **architecture- and planning-first**.  
- Data discovery / African-relevance classification is the active build work.  
- Model, retrieval, and application **contracts** are written; production code and measured artifacts land as each layer is built.  
- Large corpora, PDFs, weights, indexes, and runtime DBs stay **out of Git**.

---

## 17. Domain: Math & Scientific Reasoning (what it is)

**Official ADTC track** for MUFASA. Competition wording: problem solving, proof assistance, scientific question-answering, quantitative reasoning.

**In practice for MUFASA:** answer science questions (especially Africa-relevant materials/conditions), reason carefully with numbers/units/conditions, and ground answers in evidence — offline on a laptop.

**Not our primary track:** coding assistants, creative writing, pure farm-SMS advisory, enterprise drafting.

### Example prompts (track + product)

| Type | Example |
|---|---|
| Quantitative science Q&A | “At 5% sawdust ash as cement replacement, did Nigerian tests meet structural concrete strength requirements?” |
| Compare studies | “Do sesame-straw-ash and sawdust-ash studies agree on a good replacement %?” |
| Local materials + reasoning | “Cement is expensive. I have laterite and banana-leaf ash. What mixes were tested, what strengths/CBR, what risks, which papers?” |
| Units / conditions | “What durability changes were reported for RHA + cassava peel ash at 5–25% cement replacement?” |
| Honest abstain | “Has the corpus tested bamboo ash + cassava peel ash together under coastal conditions?” → say if missing |
| More “mathy” science | Reason about w/c and ash % vs control using local high-performance concrete trial measurements |

`metadata.json` domain value: `math_scientific_reasoning`.

---

## 18. What “African science” means (detail)

**Not** “different physics” and **not** “any paper with an African author.”

**Yes:** science where an **African material, place, crop, climate, disease, industry, constraint, method, or problem** is essential to the question or evidence.

| Counts | Does not automatically count |
|---|---|
| Laterite / local clay / rice husk ash under local curing | Nigerian author on generic US steel data only |
| Cassava disease, African soils, local mineral ores | Lab address in Accra but no African component |
| Solar/cookstove performance in African household conditions | Global review that mentions Africa once |

Full eligibility rules: [01-data-engineering/catalogs/materials-sources.md](./01-data-engineering/catalogs/materials-sources.md).  
Taxonomy: [01-data-engineering/taxonomy/african-science-categories.md](./01-data-engineering/taxonomy/african-science-categories.md).

### Global science vs African science (why global alone often fails)

| | Global / generic AI | African science need |
|---|---|---|
| Concrete | “Use Portland cement, follow code mix” | “Can **rice husk ash / sawdust ash** replace part of expensive cement — what did local trials measure?” |
| Roads | Generic soil stabilization | “**This laterite** + banana-leaf / plantain-peel ash — CBR / compaction results?” |
| Industry | Handbook bentonite | “Can **local Nigerian clay / sand** replace imported foundry/drilling materials?” |
| Crops | Generic maize advice | Varieties/practices for **local soils, drought, pests** |
| Health/energy | Default Western options | Local medicinal-plant evidence, cookstove/biomass under real constraints |

**Core idea:** methods are universal; **applications and measured recipes are place-based.** Global models often know the method but miss underrepresented African literature and local conditions — and may hallucinate numbers.

### “Nigeria already uses British Standards — why MUFASA?”

Standards (BS / Eurocode / ASTM) often **work and should stay**. MUFASA does **not** replace codes.

| Standards | MUFASA |
|---|---|
| Legal/safe design framework | Find and cite **local trial evidence** |
| “How must this be tested/designed?” | “What did African papers measure for this local material?” |

Codes don’t contain every local ash %, laterite CBR, or foundry-sand grade. African papers fill that gap **alongside** standards.

### Are these findings only useful in Africa?

**No — not “useless elsewhere.”** But they are **most necessary in Africa.**

- Very local results (Tarkwa ore, Akure laterite) → mainly that site/region  
- Methods (ag-waste ash as pozzolan) → also used in Asia/LatAm; **numbers** still need local tests  
- Physics → universal  

MUFASA is Africa-first because users + corpus + competition story are Africa-first — not because science only works on one continent.

---

## 19. Is the problem real? Need? Anything before?

**Problem is real.** Documented gaps: costly/unreliable connectivity, expensive cloud/API economics, Africa thin in LLM training data, generic models weak on local science context. ADTC itself exists for offline laptop LLMs.

**Need is real** for STEM students, researchers, engineers who need **local evidence + offline / 8 GB hardware**.

**Related things exist; this full package does not.**

| Exists | Gap vs MUFASA |
|---|---|
| AJOL, AfricArXiv | Libraries / repositories, not offline reasoning + GraphRAG product |
| PaperQA / sci-RAG | Strong citations; often cloud/API or bring-your-own PDFs; not Africa-curated laptop package |
| Offline RAG demos | Generic local docs |
| ILRI / farm chatbots (MkulimaGPT, Uliza-WI, …) | Narrow domain; often online/mobile advice |
| ChatGPT | Needs net + money; weak on thin African literature |

**Fair REPORT claim:** offline sci-RAG exists in pieces; a **consumer-laptop, fully offline, Africa-relevant evidence GraphRAG + scientific-reasoning model** is still an open product gap. Do **not** claim “first RAG ever.”

---

## 20. Example African papers (what they actually say)

From [01-data-engineering/catalogs/materials-african-papers.csv](./01-data-engineering/catalogs/materials-african-papers.csv) abstracts (verify in PDF before demo quotes):

| Topic | What the paper reports (abstract-level) |
|---|---|
| Sawdust ash concrete (Nigeria, 2024) | 0–10% cement replacement; **5%** met structural concrete requirements (tests to 180 days) |
| Sesame straw ash (2022) | 0–25% replacement; **10%** exceeded design strength at 28 days |
| Banana-leaf ash + laterite (Akure, 2016) | 2–10% ash; unsoaked **CBR optimum at 4%** |
| RHA + cassava peel ash (2022) | 5–25% replacement; better acid/sulfate resistance, lower water absorption |
| Plantain peel ash + laterite (2021) | Compaction moisture for flexible pavement use |
| SW Nigeria silica sands (2021) | Silica ~80.6–86.5%; grade D for foundry moulding |
| Metakaolin + black cotton soil (2015) | Up to 10% metakaolin; suggested waste-containment liner/cover |
| Ghana Tarkwa / ASGM tailings | Local ore/tailings characterization for processing |

Pattern: **local material → measured test → practical claim** (partial cement replacement, road material, foundry sand, etc.).

---

## 21. Application stack decisions

### How desktop connects to the LLM (local only)

```text
Tauri UI → HTTP http://127.0.0.1:PORT → FastAPI sidecar → llama.cpp + .gguf on disk
```

No cloud inference. Loopback by default (works with networking off).

### Phone path (optional)

Phone browser → local Wi-Fi/hotspot → same FastAPI on laptop → model/graph stay on laptop.  
Phone is compute-thin (UI only). Sharing off by default; revoke kills the session.

### Is phone required?

**No.** Competition does not require mobile. Desktop-only offline demo is enough. Phone is demo/pitch sugar.

### Is Tauri best?

- Competition does **not** mandate Tauri.  
- If time is tight: FastAPI + localhost web UI is enough to compete.  
- **If time is not the issue (EagleTeam stance): Tauri + FastAPI + llama.cpp is the recommended shell** — lighter than Electron, real offline desktop product, matches architecture docs. Avoid Electron for RAM.

### How to “submit” the application

No separate app-upload format. Ship app **in the public repo**, explain run steps in **`REPORT.md`**, prove it in **screenshots / 2-min video**. Scored artifact remains **`download_model.sh` → GGUF**.

---

## 22. REPORT.md problem blurb (shape)

Judges ask: problem, target users, why offline/consumer hardware.

**Shape for MUFASA:**

- **Problem:** Trusted scientific Q&A about African materials/conditions needs reasoning + cited local evidence; cloud AI is often unavailable, costly, or generic.  
- **Users:** African STEM students, researchers, practitioners on mid-range laptops.  
- **Why offline:** connectivity, cost, on-device privacy/control; must fit the 8 GB laptop people already own.

---

## 23. Quick FAQ (team)

**What is our domain?**  
Math & Scientific Reasoning (`math_scientific_reasoning`).

**Were we asked to build an app?**  
Yes — an on-device LM **application** / end-to-end system, not only a weight file. Stack (Tauri, etc.) is our choice.

**Does GraphRAG affect the automated score?**  
Not the RAM/speed telemetry path. It is still required for integration, Gates 2–3, and Best Integration.

**What carries the major score?**  
**The model (GGUF).** Graph carries requirement + defense + integration prize lane.

**What is “load-bearing”?**  
If turning GraphRAG off barely changes the product, it is not load-bearing.

**Why African science if BS codes work?**  
Codes stay. MUFASA surfaces **local experimental evidence** codes don’t list.

**Is phone required?**  
No.

**Is Tauri required?**  
No — but recommended when time allows.

---

## 24. Where to go next in this repo

| Need | Open |
|---|---|
| Bird’s-eye system | [system-architecture.md](./system-architecture.md) |
| Data / taxonomy / classification | [01-data-engineering/](./01-data-engineering/) |
| Example paper catalog | [01-data-engineering/catalogs/materials-african-papers.csv](./01-data-engineering/catalogs/materials-african-papers.csv) |
| Training pipeline | [02-model-engineering/model-training-pipeline.md](./02-model-engineering/model-training-pipeline.md) |
| GraphRAG design | [03-retrieval/retrieval-architecture.md](./03-retrieval/retrieval-architecture.md) |
| Desktop / mobile / orchestrator | [04-application/application-architecture.md](./04-application/application-architecture.md) |
| PR workflow | [CONTRIBUTING.md](./CONTRIBUTING.md) |
| Old decision dump | [conversations.md](./conversations.md) |
