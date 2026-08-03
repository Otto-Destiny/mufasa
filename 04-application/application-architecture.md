# The Application Layer

**Layer 04 of 04.** MUFASA runs primarily as a Tauri desktop application on the laptop. A paired phone can use the same complete research interface over local Wi-Fi, while inference, retrieval, validation and all persistent scientific data remain on the laptop.

> **All examples here are synthetic.** Paper identifiers are placeholders and no real DOI appears. Nothing goes into a report or a demo until it comes from a verified record.

Scope note: the first release uses a lightweight Tauri desktop shell, one local Python service and one responsive interface bundled with the desktop and served to the mobile browser. It has no cloud dependency, user account, remote server, CDN or externally hosted asset.

## The application is the orchestrator

This is not a web page wrapped around a model. It is the component that owns the whole answer, and — more importantly on this hardware — owns the machine's resources.

It controls:

| Responsibility | Why it belongs here |
|---|---|
| Retrieval calls | It decides what to ask Layer 3 for, and with what budget |
| Context construction | It builds the prompt, and enforces its size |
| Generation | It runs llama.cpp with fixed limits |
| Validation | Nothing reaches the screen unchecked |
| Cancellation | The user can stop a slow answer and free the memory |
| Concurrency | **One generation at a time across desktop and mobile**, always |
| Resource limits | Context cap, output cap, thread cap, graph buffer cap |
| Feedback | Written on the laptop, never sent to a cloud service or retained on the phone |

Loopback by default, single user, one active generation. An explicit **Share on local Wi-Fi** switch temporarily exposes the laptop service to one paired mobile session. Tauri remains the host and controller, and the same single-flight limit covers both clients. This keeps peak memory predictable under a ceiling where going over scores zero.

## Two clients, one laptop runtime

| Surface | What runs there | Product responsibility |
|---|---|---|
| **Desktop app** | Tauri and the operating-system webview | Primary interface, launcher and host controls; works with all radios disabled |
| **Mobile app** | Responsive web app served by the laptop | Full Ask, Compare, Coverage, progress, cancel and feedback experience over paired local Wi-Fi |
| **Laptop runtime** | FastAPI, `llama.cpp`, Ladybug and the scientific package | All inference, retrieval, validation, resource control and persistent storage |

The mobile app is feature-complete but compute-thin. It never receives model weights, graph/index files, SQLite or other database files, the corpus, or raw documents. Only the question, progress and cancellation events, the validated answer, and the selected evidence excerpts needed for display cross the local connection.

## How the two halves are judged

The official evaluation runs the `.gguf` directly through `llama.cpp` tooling for model quality, throughput, memory and thermals. This application is not present in that direct model run.

But the challenge also requires a working on-device system with a load-bearing cross-disciplinary integration, and Gates 2 and 3 — a 30-minute technical Q&A and a live pitch — judge the whole thing. There is a Best Integration award as well.

So both statements are true at once:

- **MUFASA must be strong with no retrieval at all**, because that is what the profiler measures.
- **The graph and the application must be genuinely useful**, because people will interrogate them directly.

Packaging the model correctly — chat template, stop tokens, sensible defaults and reproducible direct `llama.cpp` settings — is a task in its own right. Do it in week one, not on the last day.

## What the user sees: three views

Both the desktop and mobile apps expose these same research views. Host administration—starting or stopping the runtime, choosing a package, checking integrity and enabling local sharing—stays desktop-only.

**1. Ask & Evidence.** Question, answer, evidence cards, confidence, limitations. Claims carry `[E1]`, `[E2]`; clicking one opens its card:

```text
"The 10% RHA mix achieved 31.2 MPa at 28 days."      (synthetic example)

Paper P-1024 · page 8, table 4 · Results
2 study families report this · conditions match your question
```

This one view covers the clever questions, because they are just questions — substitutes for an import, why studies disagree, what to try next. The retrieval layer decides how to handle each. No separate screen per question type.

**2. Compare Studies.** A table: one row per study family, columns for conditions, measurements and results, every cell clickable to its span. Underneath it, the **disagreement matrix** — supporting, conflicting and inconclusive observations grouped, with the conditions that differ between them named:

> Four observations, three agree. The outlier differs only in ash burning temperature.

This is the strongest cheap feature in the product. It is a query plus a grouping, and it is what a working researcher actually wants.

**3. Coverage & Sources.** What is in the corpus: version, paper count, disciplines, countries, date range, last updated — and which material-property combinations have no matching evidence in it.

Gaps live **here**, not in their own screen, and that placement is the point. A gap is a statement about your corpus, not about science. The panel makes that framing unavoidable.

## The coverage card

Beside every answer, and especially beside every "nothing found":

```text
MUFASA corpus v1 · 6,000 papers · 20 fields
2005–2024 · Africa-relevant · built 12 Aug 2026
```

Show the count actually loaded, not the 6,000 target — rights exclusions and parse failures will put it lower — and let the panel break it down by field, since a corpus strong in materials and thin elsewhere should say so.

When there is no evidence, the answer reads:

> "No verified matching evidence in MUFASA corpus v1 (6,000 papers, 20 fields, 2005–2024). The nearest related evidence is [E3]."

Never "nobody has studied this." A 6,000-paper corpus is still a capped selection from 155,825 candidates; absence in it is not absence in the literature, and a judge will ask.

## Confidence, shown plainly

Four short lines beside every answer:

- **Grounded / partly grounded / no matching evidence in corpus**
- **3 study families** — separate experiments, not paper count
- **Conditions match** — or which ones do not
- **All numbers checked** — or which failed

Two rules stop this becoming decoration: thin evidence must *look* different at a glance, and "not in this corpus" is a complete, respectable answer rather than an error state.

## Do not stream unverified text

The answer is validated before the user reads it, so raw token streaming cannot also be true. Instead, stay responsive by showing real progress:

```text
1. Searching…              evidence cards appear here, within about a second
2. Comparing studies…      the compare table fills in
3. Generating…             a stage indicator, not raw tokens
4. Checking citations…
5. Answer
```

The user is reading real sources during step 1, which is where trust is actually built, and the model's slowness is hidden behind something useful rather than a spinner. Then they see one verified answer, never a draft claim that gets retracted.

Keep answers short by default. Speed is 30% of the automated score and a rambling model is worse to use. Answer, cite, state what is uncertain, stop.

## Safety route

Medical, structural and industrial questions return an **evidence summary with limitations**, never an individual prescription. Show what studies found, under what conditions, and what was not tested. Add a line pointing to a qualified professional and local standards.

This costs one branch in the prompt logic and removes an entire category of risk.

## What it is built from

| Part | Choice | Why |
|---|---|---|
| Model | Direct `llama.cpp` runtime | The accepted GGUF follows the same runtime path used for profiling |
| Local service | FastAPI | Owns retrieval, generation, validation and single-flight resource limits |
| Desktop app | Tauri using the operating-system webview | Lightweight primary product and host controller without an Electron runtime |
| Mobile app | Responsive web app served by the laptop | Full research interface; no on-phone model, database, graph, corpus or cloud state |
| Front end | One bundled responsive interface | Same Ask, Compare and Coverage views on Tauri and phone; no CDN |
| Local Wi-Fi | Disabled by default; short-lived pairing when enabled | A phone can use the laptop without cloud inference or internet access |
| Evidence card | A panel showing the quoted span and its citation | No PDF viewer, no PDFs. Instant |
| Storage | The Ladybug package from Layer 3. **No PDFs** | The app adds no data store of its own |

One launcher starts and stops the FastAPI sidecar and Tauri window. FastAPI serves the API and the same bundled interface. It binds to `127.0.0.1` normally. **Share on local Wi-Fi** opens a private-LAN listener, displays a QR/pairing code, exchanges it for a short-lived session token, and revokes that token and closes the listener when sharing is disabled. The listener accepts only the expected host and origin, and the mobile app stops working when the laptop service stops.

Dynamic responses use `Cache-Control: no-store`. The mobile app keeps no corpus cache in IndexedDB, SQLite, local storage or a service worker; only static interface assets may be cached. For demos, prefer a laptop-created private hotspot rather than a shared public network.

## The resource governor

Small, and it protects the only two failure modes that score zero.

```python
MAX_CONTEXT_TOKENS = 2048     # prompt reading dominates CPU latency
MAX_OUTPUT_TOKENS  = 400      # short answers, deliberately
LLAMA_THREADS      = 4        # leave headroom; full load overheats
LADYBUG_BUFFER_MB  = 256      # never accept the default
SINGLE_FLIGHT      = True     # one generation at a time, enforced
```

Plus a cancel button that actually frees the work, and a refusal to start a second generation while one is running. Out-of-memory is disqualification; thermal throttling costs 10 marks. Both are prevented here.

## Integrity manifest

Ship a file listing hashes for everything, and show the corpus version in the UI:

```text
mufasa.Q4_K_M.gguf     sha256:...
mufasa-graph/          sha256:...   corpus_v1, 6,000 papers, 20 fields
embedding-model/       sha256:...
prompts/system.txt     sha256:...
ladybug                version + wheel hash
extensions/            fts, vector — bundled, offline-verified
```

An hour of work. It makes the submission reproducible and auditable, and it answers the "how do we know this is what you tested?" question before it is asked.

## Feedback

A thumbs-down button and a text box, written to a local file on the laptop with the question, answer and evidence used. Mobile feedback is posted to that same laptop endpoint and is not retained on the phone.

**Nothing enters training automatically.** You read the queue and decide. Bad answers become training examples, missing evidence becomes papers to acquire. An hour of work, and it gives a real answer when a judge asks how the system improves.

## Build order

About one week, after retrieval works. Gate 1 closes **25 August 2026**.

| Days | Step | Cut? |
|---|---|---|
| 1 | Package the model: chat template, defaults and direct `llama.cpp` profiler run | **Never — this is what is scored** |
| 1 | Resource governor and single-flight generation, measured with the profiler | **Never — this prevents a zero** |
| 1–2 | FastAPI service, responsive interface and staged progress display | No |
| 2–3 | Ask & Evidence with clickable spans | No — this is the demo |
| 3 | Package Tauri, the sidecar and required runtime dependencies; cold-launch on clean Ubuntu 22.04 with networking disabled | No |
| 3–4 | Mobile web app, local pairing, revoke and disconnect flow | Keep simple — reuse the same interface |
| 4 | Compare Studies table | No |
| 4–5 | Disagreement matrix | Keep — the standout feature |
| 5 | Coverage & Sources view | Keep |
| 5 | Safety route | Cheap, keep |
| 6 | Integrity manifest and feedback queue | Cheap, keep |
| 6–7 | One African language, tested end to end — aliases in, full answer out | Keep if the model handles it well |
| — | Store distribution, exports, multi-user and several languages | Not this month |

Also due for Gate 1: the public repo on the ADTC template, `REPORT.md` with **measured** numbers, screenshots, and the two-minute video.

## The demo

First run the Tauri desktop with all networking disabled. Ask a hard question in your flagship domain. Evidence cards appear in about a second. Click one and read the exact sentence, its page and its study family. Open Compare and show three studies agreeing and one disagreeing, with the differing condition named. Then ask a question that crosses fields, which the 6,000-paper graph now supports. Finally ask something the corpus does not cover, and watch it say precisely that — corpus v1, 6,000 papers, this date range — instead of inventing an answer.

For the mobile path, keep the internet disconnected, enable the laptop's private hotspot and **Share on local Wi-Fi**, pair the phone, and repeat a short Ask/Compare flow. Turn sharing off and show that the session is immediately revoked. The desktop path remains fully usable throughout.

On a $400 laptop, from a folder that fits on a cheap USB stick, with the memory monitor visible in the corner.

## Diagram

![The MUFASA application layer](./images/application-architecture.svg)

```mermaid
flowchart TB
    classDef title fill:#FFFFFF,stroke:#FFFFFF,color:#0D315A,font-size:22px,font-weight:bold;
    classDef source fill:#E8F5E9,stroke:#2E7D32,color:#123B18,stroke-width:1.3px;
    classDef process fill:#E3F2FD,stroke:#1565C0,color:#0D315A,stroke-width:1.3px;
    classDef artifact fill:#FFF8E1,stroke:#F9A825,color:#5D4500,stroke-width:1.3px;
    classDef gate fill:#FCE4EC,stroke:#C2185B,color:#651036,stroke-width:1.3px;
    classDef train fill:#EDE7F6,stroke:#6A1B9A,color:#34104D,stroke-width:1.3px;
    classDef eval fill:#E0F7FA,stroke:#00838F,color:#00464D,stroke-width:1.3px;
    classDef deploy fill:#F3E5F5,stroke:#7B1FA2,color:#3C0F4F,stroke-width:1.3px;
    classDef reject fill:#F5F5F5,stroke:#757575,color:#333,stroke-dasharray:5 5;

    TITLE["MUFASA Application Layer"]:::title

    MOBILE["<b>Mobile web app - request</b><br/>Question, cancel or feedback action<br/>UI only: no model, graph, database or corpus<br/><i>Paired local Wi-Fi; no internet</i>"]:::source

    MOBILE_VIEW["<b>9B - Mobile web app</b><br/>Full Ask, Compare and Coverage views<br/>Progress, validated answer and selected excerpts<br/><i>Dynamic responses are not cached</i>"]:::deploy

    subgraph LAPTOP["TARGET LAPTOP - ALL COMPUTE AND PERSISTENT DATA"]
        direction TB

        DESKTOP["<b>1A - Tauri desktop request</b><br/>Question, cancel or host-control action<br/>Primary launcher and controller<br/>Works with every radio disabled"]:::source

        SHARE{"<b>1B - Share on local Wi-Fi</b><br/>Off by default<br/>QR/pairing code to short-lived session token<br/>Revoke token and close listener when disabled"}:::gate

        SERVICE["<b>2 - Laptop-local FastAPI service</b><br/>Serves the bundled interface and API<br/>One request path for desktop and mobile"]:::process

        DISK["<b>What sits on the laptop</b><br/>MUFASA 4-bit GGUF<br/>Ladybug graph, full-text, vectors and quoted spans<br/>Bundled extensions and interface assets<br/><b>No PDFs.</b>"]:::artifact

        GOV{"<b>3 - Resource governor</b><br/>One generation across both clients<br/>Context, output, thread and graph-buffer caps<br/><i>Cancel frees the active work</i>"}:::gate

        FIND["<b>4 - Retrieve</b> - Layer 3<br/>Full-text and vector, then one hop<br/>6 to 10 Observations with quoted spans"]:::process

        CARDS["<b>5 - Return evidence first</b><br/>Selected excerpts can appear before generation<br/><i>Only display data crosses to the phone</i>"]:::artifact

        MODEL["<b>6 - MUFASA generates</b> - Layer 2<br/>llama.cpp, short answer by default<br/>Progress stages shown, raw tokens not streamed"]:::train

        CHECK{"<b>7 - Validate before display</b><br/>Numbers, units and citations match their evidence?<br/>Safety route for medical and structural questions"}:::eval

        FIX["Retry that claim once,<br/>then soften the answer"]:::reject

        ANSWER["<b>8 - Verified response</b><br/>Answer, selected evidence excerpts and pages<br/>Confidence, condition match and corpus coverage"]:::deploy

        DESKTOP_VIEW["<b>9A - Tauri desktop app</b><br/>Ask and Evidence<br/>Compare Studies and disagreement matrix<br/>Coverage and Sources"]:::deploy

        FEED["<b>10 - Laptop feedback file</b><br/>Optional thumbs down saved with evidence used<br/><i>Read by a person - nothing trains automatically</i>"]:::artifact
    end

    JUDGES["<b>Scored separately</b><br/>The GGUF is run directly through llama.cpp tooling<br/>Pinned settings and plain chat template<br/><i>No graph, no app - so the model must stand alone</i>"]:::eval

    style LAPTOP fill:#FBFDFF,stroke:#1565C0,stroke-width:2px

    TITLE --> DESKTOP
    TITLE --> MOBILE
    DESKTOP --> SERVICE
    MOBILE -- "paired local Wi-Fi" --> SHARE --> SERVICE
    DISK --> SERVICE
    DISK --> FIND
    DISK --> MODEL
    DISK --> JUDGES
    SERVICE --> GOV --> FIND --> CARDS --> MODEL --> CHECK
    CARDS -. "early progress and evidence" .-> DESKTOP_VIEW
    CARDS -. "early progress and selected excerpts" .-> MOBILE_VIEW
    CHECK -- "Pass" --> ANSWER
    CHECK -- "Fail" --> FIX --> MODEL
    ANSWER --> DESKTOP_VIEW
    ANSWER -- "validated answer and selected excerpts" --> MOBILE_VIEW
    DESKTOP_VIEW -. "optional feedback" .-> FEED
    MOBILE_VIEW -. "optional feedback" .-> FEED
```

## How to Read the Diagram

- **Green** is a request or control action from the primary Tauri desktop or the compute-thin mobile web app.
- **Blue** is work the application does.
- **Yellow** is what sits on disk, or what gets written back to it.
- **Pink** is an explicit gate: mobile sharing or the resource governor. Sharing is closed by default, and the governor runs before expensive work.
- **Purple** is MUFASA generating.
- **Cyan** is validation, and the separate box for what the judges actually run.
- **Violet** is what appears on screen.
- **Grey dashed** is the repair path.

The blue laptop boundary is the critical ownership rule: FastAPI, `llama.cpp`, Ladybug and every persistent scientific artifact stay inside it. The mobile app crosses that boundary only through the paired local API.

Two things worth reading twice. Step 5 comes **before** step 6 — evidence cards appear while the model is still starting, which is what makes a slow laptop feel responsive. And step 7 sits between generation and the screen, which is why raw tokens are never streamed.

The box on the right stands alone on purpose: the automated scoring runs the model file with none of this attached.
