# 🦁 MUFASA

### Models for Understanding the Frontiers of African Scientific Advancement

**MUFASA** is a compact, offline scientific-reasoning model designed for African scientific and technical use cases on affordable hardware.

The project explores a practical question:

> **How much useful scientific reasoning can we retain in a model small enough to run locally on ordinary CPU hardware?**

Rather than optimizing only for raw benchmark accuracy, MUFASA is designed around the combined requirements of **reasoning quality, inference speed, latency, memory efficiency, and offline accessibility**.

---

## 🌍 Why MUFASA?

Many modern AI systems assume:

* powerful GPUs;
* large amounts of RAM;
* stable broadband;
* continuous access to cloud APIs;
* and recurring inference costs.

Those assumptions do not fit every environment.

MUFASA is being developed around a different deployment target: a scientific assistant that can operate locally, remain useful on modest hardware, and eventually retrieve supporting scientific evidence without depending on cloud inference.

The long-term architecture is:

[
\text{MUFASA}
=============

\text{Compact Reasoning Model}
+
\text{Local Scientific Retrieval}
]

---

## 🤗 Model

The current MUFASA model is a Gemma 3 1B derivative adapted through continued pretraining and supervised fine-tuning.

**GGUF model:**

[`DestinyOtto/mufasa-gemma3-1b-sft-gguf`](https://huggingface.co/DestinyOtto/mufasa-gemma3-1b-sft-gguf)

### Deployment format

* **Architecture:** Gemma 3 1B
* **Runtime:** `llama.cpp`
* **Format:** GGUF
* **Recommended quantization:** `Q4_K_M`
* **Target:** CPU-first local inference
* **Cloud inference required:** No

---

# 🧠 How MUFASA Was Built

MUFASA was developed in three main stages.

## 1. Model Selection

Before training MUFASA, we benchmarked several compact and mid-sized language models using the **Africa Deep Tech Challenge profiler**.

Candidate families included:

* Qwen
* Phi
* Gemma
* LiquidAI LFM

We measured:

* ARC-Easy accuracy
* generation throughput
* first-token latency
* peak process-tree memory
* CPU efficiency
* estimated weighted deployment score

Gemma 3 1B produced the strongest overall deployment trade-off in our development benchmark.

### Gemma 3 1B development benchmark

| Metric                   |         Result |
| ------------------------ | -------------: |
| Generation throughput    | **17.4 tok/s** |
| First-token latency      |    **~2.54 s** |
| Peak process-tree RSS    |   **~1.33 GB** |
| ARC-Easy accuracy        |       **0.50** |
| Estimated weighted score |      **0.713** |

The weighted score combined reasoning performance with deployment efficiency.

> These results were produced during **base-model selection**. They should not be interpreted as the final benchmark score of the fine-tuned MUFASA checkpoint.

---

## 2. Continued Pretraining

The actual MUFASA training pipeline begins from:

```text
unsloth/gemma-3-1b-pt
```

We continued pretraining the model on a corpus of African research literature before supervised instruction tuning.

### CPT corpus

The executed training pipeline processed:

| Statistic                  |                Value |
| -------------------------- | -------------------: |
| Training papers discovered |            **9,858** |
| Eligible cleaned papers    |            **9,852** |
| Raw text                   | **~394M characters** |
| Cleaned text               | **~320M characters** |
| Training windows           |           **27,358** |
| Training tokens            |          **~91.57M** |
| Context length             |     **4,096 tokens** |

The corpus pipeline preserves the ordering of each paper and avoids treating arbitrary 4,096-token chunk boundaries as real document endings.

EOS is introduced only at true document boundaries.

### CPT strategy

Continued pretraining used **RSLoRA** across attention and MLP components while also allowing the embedding layer and language-model head to adapt.

This stage was intended to expose the model to:

* African scientific literature;
* technical terminology;
* measurements and scientific units;
* geographic and environmental concepts;
* regional research contexts;
* and domain-specific language patterns.

---

## 3. Supervised Fine-Tuning

After continued pretraining, the merged checkpoint was adapted into a scientific assistant using **full-parameter supervised fine-tuning**.

### SFT dataset

The training pipeline loaded:

```text
353,697 conversations
```

including:

| Split        | Conversations |
| ------------ | ------------: |
| VERIFIED     |   **124,114** |
| UNVERIFIED   |   **229,583** |
| Total loaded |   **353,697** |

Five examples were removed because no valid assistant-response labels survived tokenization.

Final usable training examples:

```text
353,692
```

### SFT configuration

* **Trainable parameters:** 999,885,952
* **Parameters trained:** 100%
* **Maximum sequence length:** 4,096
* **Batch size:** 16
* **Gradient accumulation:** 4
* **Effective batch size:** 64
* **Learning rate:** `2e-5`
* **Held-out evaluation conversations:** 400
* **Loss:** assistant-response-only

The prompt portion of each conversation is masked from the loss.

This separates the two learning stages conceptually:

[
\text{Continued Pretraining}
\rightarrow
\text{Learn the scientific domain}
]

[
\text{Supervised Fine-Tuning}
\rightarrow
\text{Learn how to answer}
]

---

# ⚡ Designed for Local Inference

MUFASA is intentionally built around constrained deployment.

The final model is distributed as a quantized GGUF checkpoint compatible with `llama.cpp`.

### Why Q4_K_M?

`Q4_K_M` provides a useful compromise between:

* model quality;
* memory use;
* disk size;
* CPU throughput;
* and deployment simplicity.

A compact 1B model also leaves considerably more memory available for retrieval indexes, application logic, and document processing than larger reasoning models.

---

# 🚀 Running MUFASA Locally

Install or build `llama.cpp`, then download the Q4_K_M GGUF from the Hugging Face repository.

Example:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -B build
cmake --build build --config Release -j
```

Then run MUFASA:

```bash
./build/bin/llama-cli \
  -m /path/to/MUFASA-Gemma3-1B-SFT-Q4_K_M.gguf \
  -p "Explain how vegetation loss can reinforce drought conditions in the Sahel."
```

The model can run locally after the weights have been downloaded.

No cloud inference API is required.

---

# 📊 Benchmark Philosophy

MUFASA is not designed around the idea that the largest model always wins.

For offline systems, model quality is constrained by several objectives simultaneously:

[
\text{Deployment Quality}
=========================

f(
\text{Accuracy},
\text{Latency},
\text{Throughput},
\text{Memory},
\text{Reliability}
)
]

For example, our candidate benchmark showed that some models achieved higher ARC-Easy scores than Gemma 3 1B but required much more memory or had substantially worse latency.

Other models generated significantly faster but produced weaker reasoning accuracy.

This trade-off motivated our decision to specialize a 1B model rather than simply deploy a larger general-purpose checkpoint.

---

# 🔬 Scientific Focus

MUFASA is being developed toward scientific and technical domains with particular relevance to African applications, including:

* 🌾 agriculture and food systems
* 🌦️ climate and weather
* 🛰️ geospatial science and remote sensing
* 🌱 environmental science
* 💧 hydrology and water resources
* 🧪 scientific reasoning
* 🏥 public-health reasoning
* ⚡ energy systems
* 📊 quantitative problem solving

The goal is not to encode all scientific knowledge inside a 1B model.

Instead, the model should become an efficient reasoning layer over locally available scientific evidence.

---

# 📚 Evidence-Grounded Retrieval

The next major MUFASA component is local scientific retrieval.

A small language model has limited capacity. Rather than forcing it to memorize an entire scientific knowledge base, we want MUFASA to retrieve relevant documents and use its parameters primarily for interpretation and reasoning.

The intended workflow is:

```text
User Question
      │
      ▼
Local Scientific Corpus
      │
      ▼
Evidence Retrieval
      │
      ▼
MUFASA 1B
      │
      ▼
Grounded Scientific Response
```

This architecture has several advantages:

* knowledge can be updated without retraining the model;
* supporting evidence can be inspected;
* hallucination risk can be reduced;
* private or institutional document collections can remain local;
* and the complete system can continue operating offline.

---

# 🧪 Evaluation

MUFASA development uses held-out evaluation at multiple stages.

### Continued pretraining

Evaluation includes:

* held-out research papers;
* language-model loss;
* perplexity where appropriate;
* tokenizer-independent comparisons;
* and scientific concept probes.

### Supervised fine-tuning

The SFT pipeline maintains held-out conversations separate from the training examples.

### Deployment evaluation

We use the **Africa Deep Tech Challenge profiler** to measure:

* generation throughput;
* first-token latency;
* process-tree memory;
* CPU behavior;
* thermal characteristics;
* and standardized reasoning accuracy.

---

# 🛠️ Project Principles

MUFASA is being developed around a few core principles.

### Small enough to deploy

A useful model should not require data-center hardware.

### Offline by design

Cloud inference should be optional rather than a requirement.

### Evidence over memorization

External scientific knowledge is better retrieved explicitly than compressed entirely into model parameters.

### African scientific context

Domain adaptation should include scientific work, environments, challenges, and terminology relevant to Africa.

### Reproducible evaluation

Performance claims should come from repeatable benchmark pipelines rather than selected demonstration prompts.

### Practical efficiency

Tokens per second and memory usage matter alongside benchmark accuracy.

---

# 🗺️ Roadmap

Planned work includes:

* [ ] benchmark the final MUFASA SFT GGUF across broader reasoning datasets
* [ ] expand the verified scientific SFT dataset
* [ ] increase African research-literature coverage
* [ ] build a fully local retrieval pipeline
* [ ] add evidence citations to generated answers
* [ ] evaluate retrieval faithfulness and hallucination
* [ ] create agriculture-specific evaluation sets
* [ ] create climate and environmental-science evaluations
* [ ] create geospatial and remote-sensing evaluations
* [ ] evaluate public-health scientific reasoning
* [ ] measure real laptop energy consumption
* [ ] benchmark additional quantization levels
* [ ] explore knowledge distillation
* [ ] test deployments on affordable laptops and edge systems

---

# 🦁 What MUFASA Stands For

**MUFASA**

**M**odels for
**U**nderstanding the
**F**rontiers of
**A**frican
**S**cientific
**A**dvancement

---

# 🌍 Vision

MUFASA is ultimately about making capable scientific AI more accessible.

We believe useful AI should not automatically require:

```text
large GPUs + cloud APIs + constant connectivity
```

We are exploring another path:

```text
compact models
+ domain adaptation
+ local evidence
+ efficient CPU inference
```

The goal is a scientific AI system that is:

**small enough to run locally, specialized enough to be useful, grounded enough to inspect, and efficient enough to work on hardware people actually have.**

---

## Links

**🤗 MUFASA GGUF**

https://huggingface.co/DestinyOtto/mufasa-gemma3-1b-sft-gguf

**🏆 Africa Deep Tech Challenge 2026**

https://adtc-2026.devpost.com/

**🦁 Devpost Project**

https://devpost.com/software/mufasa
