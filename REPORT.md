# Technical Report — MUFASA

**Team ID:** Project MUFASA  
**Domain:** `math_scientific_reasoning`  
**Model:** `MUFASA-Gemma3-1B-SFT-Q4_K_M`

---

## Problem

MUFASA — **Models for Understanding the Frontiers of African Scientific Advancement** — is an offline scientific-reasoning project for constrained, affordable hardware.

Many capable language models assume reliable broadband, cloud inference, and accelerator hardware. MUFASA is designed for African learners, researchers, extension workers, and technical practitioners who may need useful scientific assistance when those assumptions do not hold. The model therefore targets local CPU inference, low memory use, and an evidence-grounded architecture that can later pair the compact model with locally stored scientific material.

The submission focuses on the compact language-model component. Once its GGUF weight file has been downloaded, inference is fully local through `llama.cpp`.

---

## Design Decisions

### Model-selection benchmark

Before training MUFASA, we profiled several compact and mid-sized GGUF models with the ADTC profiler components. Candidate families included Qwen, Phi, Gemma, and LiquidAI LFM.

The Gemma 3 1B size class gave the strongest deployment trade-off in our development benchmark. The Gemma 3 1B IT candidate recorded:

| Development metric | Gemma 3 1B IT |
|---|---:|
| Generation throughput | **17.4 tokens/s** |
| First-token latency | **~2,540 ms** |
| Peak process-tree RSS | **~1,330 MB** |
| ARC-Easy accuracy in the limited development run | **0.50** |
| Estimated weighted development score | **0.713** |

These numbers motivated the 1B Gemma choice. They are **model-selection results, not the final SFT checkpoint's official audit measurements**.

### Actual MUFASA training path

The training notebooks show a two-stage adaptation path:

1. **Continued pretraining (CPT)** starting from `unsloth/gemma-3-1b-pt`.
2. **Full-parameter supervised fine-tuning (SFT)** starting from the merged CPT checkpoint.

This distinction is important: the Gemma 3 1B IT checkpoint was useful for deployment-size benchmarking, while the MUFASA training pipeline starts from the pretrained Gemma 3 1B weights and teaches domain knowledge before instruction behavior.

### Continued pretraining

The CPT corpus was built from African research papers.

The executed notebook records:

- **9,858** training papers discovered;
- **9,852** eligible papers retained after cleaning;
- **394M raw characters → 320M cleaned characters**;
- **27,358** lossless training windows;
- **91.57M training tokens including document-end handling**;
- 4,096-token windows;
- 40 held-out papers / 116 complete held-out windows used in the live evaluation subset.

The CPT stage used RSLoRA with rank 128 and alpha 32 across attention and MLP projections, while `embed_tokens` and `lm_head` were explicitly trainable. The training contract used a 5e-5 main learning rate and a smaller 1e-5 embedding learning rate.

The data pipeline preserves papers as ordered token windows and appends EOS only at the true paper boundary rather than treating artificial 4,096-token cuts as document endings.

### Supervised fine-tuning

The SFT notebook performs **full-parameter fine-tuning**, not LoRA, from the CPT checkpoint.

The executed dataset load records:

- **353,697 training conversations** before response-marker filtering;
- **124,114 VERIFIED** conversations;
- **229,583 UNVERIFIED** conversations;
- **400 held-out conversations** used during training;
- five rows removed because no valid response labels survived, leaving **353,692 usable training examples**;
- **999,885,952 trainable parameters**, or 100% of the model;
- 4,096-token maximum sequence length;
- batch size 16 × gradient accumulation 4 = effective batch size **64**;
- learning rate **2e-5**;
- a one-epoch training configuration.

Response-only loss is used so the model is trained on the assistant answer rather than being rewarded for reproducing the user prompt.

### Quantization

The submission uses **GGUF Q4_K_M**.

Q4_K_M provides a strong balance between model quality, disk footprint, inference speed, and memory headroom. A 1B-class Q4 model leaves substantial RAM available for the runtime and future local retrieval components while remaining practical on the 8 GB target laptop.

---

## Constraints

The submission is designed around the official laptop profile:

- **8 GB RAM target**
- CPU-first inference via `llama.cpp`
- integrated GPU only; no discrete GPU requirement
- public GGUF download before evaluation
- zero network dependency once model evaluation begins
- low enough memory use to leave room for a future local retrieval index

The training process itself used GPU hardware, but the **submitted artifact is explicitly optimized for CPU inference**.

---

## Benchmarks

The development benchmark above was used to choose the model family and size.

The final authoritative measurements for the submitted MUFASA SFT checkpoint should be produced with:

```bash
bash download_model.sh

adtc-profiler run \
  --submission . \
  --mode participant \
  --output submission.json
```

A quick structural smoke test may use `--skip-accuracy`, but the final report should retain the accuracy stage.

The resulting `submission.json` is the authoritative measurement of the submitted Q4_K_M checkpoint.

---

## Offline operation

`download_model.sh` is the only network-dependent step. It queries the public Hugging Face repository:

`DestinyOtto/mufasa-gemma3-1b-sft-gguf`

for its Q4_K_M GGUF and writes it to:

`model/MUFASA-Gemma3-1B-SFT-Q4_K_M.gguf`

After that download, inference is local through `llama.cpp` and does not require a cloud model API.

---

## Reproducibility

The repository does not commit weight files. Both `model/` and `*.gguf` are excluded by `.gitignore`.

The training notebooks document the CPT and SFT configuration, dataset counts, held-out evaluation paths, checkpointing, and GGUF export process. The submission repository contains only the lightweight reproducibility files required to download and profile the released model.
