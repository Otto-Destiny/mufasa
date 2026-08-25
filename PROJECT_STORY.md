# MUFASA

**Models for Understanding the Frontiers of African Scientific Advancement**

MUFASA is an offline African scientific-reasoning project designed for constrained and affordable hardware. It combines a compact language model with an evidence-grounded retrieval direction so that scientific answers can remain useful, inspectable, and locally relevant without depending on cloud inference.

## Inspiration

Powerful AI systems increasingly assume powerful hardware, reliable connectivity, and continuous access to cloud infrastructure. Those assumptions do not fit every environment in which scientific assistance could be useful across Africa.

We wanted to explore a different question:

**Can a small model learn enough African scientific context and reasoning behavior to remain useful while running locally on ordinary CPU hardware?**

That question shaped MUFASA from the beginning. Instead of optimizing only for raw benchmark accuracy, we treated model selection as a deployment problem:

\[
\text{Practical Utility}
=
f(\text{accuracy},\text{throughput},\text{latency},\text{memory},\text{offline accessibility})
\]

The aim is not to build the largest model. It is to build a model that can actually be used.

## What it does

MUFASA is a compact scientific-reasoning model intended for offline use.

The project currently has two complementary directions:

1. **A domain-adapted 1B language model** that can reason and respond locally.
2. **Evidence-grounded retrieval** so scientific answers can eventually be supported by locally stored research rather than relying only on information encoded in the model's parameters.

The released model is packaged as a quantized GGUF checkpoint for `llama.cpp`, allowing it to run without cloud inference after the model file has been downloaded.

Our public model repository is:

`DestinyOtto/mufasa-gemma3-1b-sft-gguf`

## How we built it

We did not start by choosing a model based on reputation alone. We first benchmarked several compact model families using the Africa Deep Tech Challenge profiler components.

The development benchmark compared Qwen, Phi, Gemma, and LiquidAI models using generation throughput, first-token latency, process memory, and ARC-Easy accuracy.

The Gemma 3 1B size class gave us the strongest overall deployment profile. In our model-selection run, Gemma 3 1B IT reached approximately:

| Metric | Result |
|---|---:|
| Generation throughput | **17.4 tokens/s** |
| First-token latency | **2.54 s** |
| Peak process-tree RSS | **1.33 GB** |
| ARC-Easy accuracy | **0.50** |
| Estimated weighted score | **0.713** |

The 0.713 weighted result was the highest in that development comparison.

But selecting the size class was only the beginning.

### Stage 1: continued pretraining on African research

The actual MUFASA training pipeline starts from the pretrained `unsloth/gemma-3-1b-pt` checkpoint.

We first performed continued pretraining on an African research corpus so that the model could become more familiar with the language, places, measurements, and scientific concepts appearing in the literature before instruction tuning.

The executed CPT notebook processed:

- **9,858 training papers**, of which **9,852** passed the cleaning rules;
- about **320 million cleaned characters**;
- **27,358** ordered training windows;
- approximately **91.57 million training tokens**;
- 4,096-token context windows.

The windowing pipeline was deliberately lossless. Artificial context-window boundaries were not treated as paper endings; EOS was added only at the real end of a paper.

We used RSLoRA during CPT, including attention and MLP projections, and explicitly trained the token embeddings and language-model head so the model could absorb out-of-distribution African scientific vocabulary rather than only modifying intermediate layers.

### Stage 2: full-parameter supervised fine-tuning

The second stage converted the domain-adapted model into a scientific assistant.

The SFT run loaded **353,697 training conversations**:

- **124,114 VERIFIED**
- **229,583 UNVERIFIED**

Five examples were automatically removed because no valid assistant-response labels survived tokenization, leaving **353,692 usable training examples**.

Unlike the CPT stage, SFT used **full-parameter fine-tuning**. All **999,885,952 parameters** were trainable.

The training configuration used:

- 4,096-token maximum sequence length;
- batch size 16;
- gradient accumulation of 4;
- effective batch size 64;
- learning rate `2e-5`;
- response-only loss;
- 400 held-out conversations for evaluation during training.

This two-stage design lets CPT teach the model **what African scientific literature looks like**, while SFT teaches it **how to answer users**.

### Stage 3: CPU deployment

The final model is distributed in GGUF form and targeted at `llama.cpp`.

We use Q4_K_M quantization because it gives a practical balance between model quality and the memory footprint required for affordable hardware.

## Challenges we ran into

The largest challenge was that model quality and deployability often pull in different directions.

Some larger models produced stronger benchmark accuracy but consumed much more memory or had significantly worse latency. Some smaller models generated very quickly but lost too much accuracy.

That is why we benchmarked multiple dimensions rather than selecting a model from one score.

We also encountered engineering challenges around:

- changing GGUF architecture support;
- `llama.cpp` and `llama-cpp-python` version mismatches;
- Hugging Face filename differences;
- large-model CPU inference times;
- memory-safe benchmarking;
- checkpoint recovery during long training;
- preserving real document boundaries during continued pretraining;
- preventing prompt tokens from contributing to SFT loss;
- and distinguishing domain learning from simple memorization.

The CPT pipeline therefore includes held-out papers, corpus hashes, a reproducible windowing policy, optimizer preflight checks, and evaluation designed to distinguish retention from generalization.

The SFT pipeline similarly uses a held-out set and response-only labels.

Another lesson was that dataset size alone is not enough. Verification tier, formatting quality, clean train/evaluation separation, and the training objective matter just as much as the raw number of rows.

## Accomplishments that we're proud of

We are proud that MUFASA progressed from benchmarking into a complete model-engineering pipeline.

We built:

- a reproducible multi-model CPU benchmark;
- a model-selection process based on accuracy, speed, latency, and RAM rather than parameter count alone;
- an African research corpus pipeline containing thousands of papers;
- a CPT run covering roughly **91.57M tokens**;
- a supervised dataset containing **353,697 training conversations**;
- a full-parameter 1B-model SFT pipeline;
- a public GGUF model for local inference;
- and the foundation for an offline retrieval-grounded scientific assistant.

The project also demonstrates why small models are interesting. The selected 1B Gemma candidate used only around **1.33 GB peak process-tree RSS** in our development profiler run while generating about **17.4 tokens per second**.

That leaves meaningful hardware headroom for the retrieval component and other application logic.

## What we learned

The strongest lesson was that **the best model is not necessarily the model with the highest raw accuracy**.

A locally deployed model has to satisfy several constraints at once.

Gemma 3 4B, for example, achieved stronger ARC-Easy accuracy in our model-selection experiment, but Gemma 3 1B was significantly faster and smaller. LiquidAI's 1.2B Thinking model generated even faster, but its limited ARC-Easy result in that run was much weaker.

The 1B Gemma size class gave us the most useful balance to build on.

We also learned that small-model specialization benefits from separating learning stages.

Continued pretraining can teach a model the statistical structure and vocabulary of a scientific domain, while supervised fine-tuning can teach the interaction contract. Trying to make one training stage do both jobs makes it harder to understand what the model actually learned.

We also learned to be careful with evaluation. Perplexity is meaningful when comparing a model with its own earlier checkpoint, but can be misleading across different tokenizers. Our CPT evaluation therefore also uses tokenizer-independent bits-per-byte for cross-family comparisons and separate held-out concept probes.

Finally, we learned that deployment engineering is part of model quality. A model that cannot be loaded reliably, exceeds the memory budget, or depends on cloud services does not satisfy the problem MUFASA is trying to solve.

## What's next for MUFASA

The next stage is to make MUFASA a complete **offline evidence-grounded scientific reasoning system**.

A 1B model should not be expected to memorize all scientific knowledge. Instead, we want the model to retrieve relevant evidence locally and focus its limited capacity on interpretation and reasoning.

Conceptually:

\[
\text{MUFASA}
=
\text{Compact Reasoning Model}
+
\text{Local Scientific Retrieval}
\]

This allows knowledge to be updated without retraining the entire model.

Next steps include:

- completing broader evaluation of the final SFT GGUF rather than relying on the base-model selection benchmark;
- improving the verification mix of the SFT dataset;
- expanding African scientific literature coverage;
- adding local evidence retrieval and citations;
- testing agriculture, climate, health, geospatial, and environmental-science tasks separately;
- evaluating hallucination and citation fidelity;
- measuring real energy use on affordable laptops;
- exploring further distillation and quantization;
- and testing MUFASA in fully offline field-style deployments.

Our long-term goal is straightforward:

**build a scientific AI system that is small enough to run locally, specialized enough to be useful, grounded enough to inspect, and efficient enough to work on hardware people actually have.**
