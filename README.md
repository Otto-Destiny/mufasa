# MUFASA

**Models for Understanding the Frontiers of African Scientific Advancement**

MUFASA is an offline African scientific-reasoning project being developed for constrained, affordable hardware. It combines a compact language model with evidence-grounded retrieval so that scientific answers can remain useful, inspectable and locally relevant without depending on cloud inference.

### Full context (start here anytime)

**[CONTEXT.md](./CONTEXT.md)** — single always-on brief for the team. Open it for strategy, compliance, and “why are we building this?” questions.

It covers:

- ADTC rules, hardware, scoring, gates, GGUF / llama.cpp, submission format  
- Domain (**Math & Scientific Reasoning**) + example prompts  
- What **African science** is, why global AI alone is not enough, BS-standards vs local papers  
- Whether the need is real / what already exists  
- Example African papers from the materials catalog  
- Model vs GraphRAG score weight; load-bearing integration  
- Desktop (Tauri ↔ localhost FastAPI ↔ llama.cpp), optional phone, how the app is submitted  
- EagleTeam briefing notes and FAQ  

Layer design detail still lives in [system-architecture.md](./system-architecture.md) and the `01`–`04` folders.

![MUFASA four-layer architecture](./images/system-architecture.svg)

## Four layers

- **01.** [Data engineering](./01-data-engineering/) discovers, classifies and prepares African scientific evidence.
- **02.** [Model engineering](./02-model-engineering/) builds, evaluates and packages the compact MUFASA model. — [Milestone 1](./02-model-engineering/milestone1.md)
- **03.** [Retrieval](./03-retrieval/) turns verified evidence into an offline GraphRAG package. — [Milestone 1](./03-retrieval/milestone1.md)
- **04.** [Application](./04-application/) provides the primary Tauri desktop app and a complete, compute-thin mobile web app over local Wi-Fi; inference and retrieval remain on the laptop.

The complete system design is described in [system-architecture.md](./system-architecture.md).

## Key dates

Built for **The Laptop LLM Challenge** (African Deep Tech Challenge 2026).

| Gate | Date | What it covers |
|---|---|---|
| **Gate 1** | **25 August 2026** | Submission — public repo, `REPORT.md` with measured numbers, screenshots, two-minute video, and the `.gguf` on the dashboard |
| **Gate 2** | **8–29 September 2026** | 30-minute technical Q&A and reproducibility audit |
| **Gate 3** | **17 October 2026** | Live pitch |

Judging runs on a fixed envelope: Ubuntu 22.04, Intel i5, 8 GB RAM, **7 GB peak RSS ceiling**, no internet, llama.cpp with GGUF only. Exceeding the memory ceiling or reaching the network during evaluation scores zero. Scoring is `0.50 × accuracy + 0.30 × speed + 0.20 × efficiency − thermal penalty`.

For a single always-on brief of competition + project context, use [CONTEXT.md](./CONTEXT.md). Historical discussions are in [conversations.md](./conversations.md); reviewed layer documents take precedence where they differ.

## Current status

Data discovery and African-relevance classification are active. Model training, retrieval and application documents currently define the implementation contracts and release gates; their production code will be added as each layer is built and measured.

Large corpora, PDFs, generated partitions, model weights and runtime databases are deliberately kept out of Git. Public releases will use reproducible download/build scripts, manifests and checksums.

## Contributing

Changes are proposed from a prefixed branch and reviewed through a pull request; `main` is never a working branch. See [CONTRIBUTING.md](./CONTRIBUTING.md).

## Licence

Apache License 2.0 — see [LICENSE](./LICENSE).

The licence covers this repository's code, documentation and schemas. It does not extend to the scientific papers MUFASA processes, which stay under their publishers' terms and are tracked per paper in the rights ledger, or to model weights, which carry their own licences. Neither is distributed here. See [NOTICE](./NOTICE).
