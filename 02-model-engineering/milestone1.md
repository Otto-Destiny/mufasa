# Milestone 1 — Model Engineering

**For:** the engineer taking on Layer 2 (model).

Competition: **The Laptop LLM Challenge** (ADTC 2026). Gate 1 closes **25 August 2026**.

---

## Tasks

- [ ] **1.** Set up the machine
- [ ] **2.** Download the candidate models
- [ ] **3.** Install llama.cpp — the required runtime
- [ ] **4.** Benchmark each candidate and report four numbers
- [ ] **5.** Tell me what the numbers imply

---

## How to read this

**This is a guideline, not a specification. What I care about is results.**

The numbers below decide which base model we build MUFASA on, so getting them right matters more than following my exact steps. If you find a better way to measure, or spot something these four numbers miss, take it further — that's the whole point of the task.

---

## ⚠ Key cautions

| Caution | Detail |
|---|---|
| **7 GB peak RAM ceiling** | Exceeding it scores **zero** and disqualifies. Measured as **maximum RSS** |
| **llama.cpp with GGUF only** | *"All submissions must run through llama.cpp using GGUF weights."* No Colibri, no AirLLM, no disk-streaming runtimes — they buy memory by spending speed, which is backwards for us |
| **Measure RSS, not PSS** | The judges record *maximum RSS*. PSS divides shared memory between processes and reads lower — tune against it and you'll find the gap on the judging laptop |
| **Report the machine with every number** | Numbers without a machine attached don't compare to anything |
| **No model files in git** | `.gitignore` already blocks `*.gguf` |

---

## 1. Set up the machine

Judging happens on **Ubuntu 22.04, Intel i5, 8 GB RAM, integrated graphics**. Match it as closely as you reasonably can — a VM or spare laptop is fine. WSL2 works for day-to-day, but reported numbers must come from real Ubuntu, since RAM accounting and thermals both differ under WSL.

```bash
sudo apt update
sudo apt install -y build-essential cmake git python3 python3-pip python3-venv \
                    lm-sensors time curl
sudo sensors-detect --auto     # enables CPU temperature reading
```

---

## 2. Download the candidate models

Start with these two. They bracket the size range we're choosing between.

Four candidates: two conventional small models, and two extreme-quantisation 8B models that trade weight precision for size.

| # | Model | Quant | File size | Expect peak RAM |
|---|---|---|---|---|
| 1 | **Nanbeige4.2-3B** | `Q4_K_M` | 2.68 GB | ~3.0–3.2 GB |
| 2 | **Qwen3.5-4B** | `Q4_K_M` | 2.74 GB | ~3.1–3.3 GB |
| 3 | **Bonsai-8B** (1-bit) | `Q1_0` | 1.16 GB | ~1.6–2.0 GB |
| 4 | **Ternary-Bonsai-8B** | `Q2_0` | 2.18 GB | ~2.5–2.9 GB |

### Direct download links

Exact files — nothing to search for, nothing to guess:

```bash
mkdir -p ~/mufasa/models && cd ~/mufasa/models

# 1. Nanbeige4.2-3B  (2.68 GB)
wget https://huggingface.co/bartowski/Nanbeige_Nanbeige4.2-3B-GGUF/resolve/main/Nanbeige_Nanbeige4.2-3B-Q4_K_M.gguf

# 2. Qwen3.5-4B  (2.74 GB)
wget https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf

# 3. Bonsai-8B, 1-bit  (1.16 GB)
wget https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/Bonsai-8B-Q1_0.gguf

# 4. Ternary-Bonsai-8B  (2.18 GB)
wget https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf/resolve/main/Ternary-Bonsai-8B-Q2_0.gguf
```

Or with the Hugging Face CLI, which resumes cleanly if a download drops:

```bash
pip install -U "huggingface_hub[cli]"

hf download bartowski/Nanbeige_Nanbeige4.2-3B-GGUF Nanbeige_Nanbeige4.2-3B-Q4_K_M.gguf --local-dir .
hf download unsloth/Qwen3.5-4B-GGUF              Qwen3.5-4B-Q4_K_M.gguf              --local-dir .
hf download prism-ml/Bonsai-8B-gguf              Bonsai-8B-Q1_0.gguf                 --local-dir .
hf download prism-ml/Ternary-Bonsai-8B-gguf      Ternary-Bonsai-8B-Q2_0.gguf         --local-dir .
```

Roughly 8.8 GB in total. Check each file's size against the table before running anything — a truncated download fails in confusing ways.

### What each one is

**1 — Nanbeige4.2-3B** · [repo](https://huggingface.co/bartowski/Nanbeige_Nanbeige4.2-3B-GGUF)
Despite the name it is ~4B total parameters (3B non-embedding). Uses a Looped Transformer, reusing layers to add capacity without adding parameters. Reported to beat Qwen3.5-9B and Gemma4-12B on reasoning. **This is the one we plan to fine-tune**, so its numbers matter most.

**2 — Qwen3.5-4B** · [repo](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
The conventional baseline at the same size class. Note this is Qwen **3.5**, not 3.6 — the 3.6 series only ships a 27B dense model and a 35B MoE, both far past our 7 GB ceiling, so 3.5 is the newest Qwen at a usable size.

**3 — Bonsai-8B, 1-bit** · [repo](https://huggingface.co/prism-ml/Bonsai-8B-gguf)
Qwen3-8B architecture with weights squeezed to {−1, +1}. Reported benchmark average **70.5**. Smallest file of the four by some margin.

**4 — Ternary-Bonsai-8B** · [repo](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf)
Same 8.19B Qwen3-8B base, but weights are {−1, **0**, +1} — about 1.71 bits each once the FP16 group scales are counted. That extra zero is worth **5 points**: reported average **75.5**, ranking 2nd among all compared models at roughly a ninth of their size.

That repo also holds `Ternary-Bonsai-8B-F16.gguf` (16 GB) and two variants, `PQ2_0` and `Q2_0_g64`. Ignore all three — `Q2_0` is the one to measure, and F16 is only the re-quantisation source.

> ⚠ **Ternary will not run on stock llama.cpp.** PrismML's own documentation: *"Ternary (`Q2_0`) needs the PrismML fork (`prism` branch) or its pre-built binaries; stock builds cannot run it."* 1-bit (`Q1_0`) **is** merged upstream and runs on normal builds.
>
> This matters more than its benchmark score. The judges download our `.gguf` and run it in **LM Studio or Ollama**, which ship stock llama.cpp — so a ternary submission would fail to load and score zero. Benchmark it anyway, because the numbers tell us what we are giving up, but treat it as ruled out for shipping unless upstream support lands and reaches those tools before 25 August. See Task 3 for building the fork.

Verify all four arrived intact before benchmarking anything:

```bash
ls -la --block-size=M ~/mufasa/models/*.gguf
```

---

## 3. Install llama.cpp

llama.cpp is the program that actually runs the model file, and the rules require it.

```bash
cd ~/mufasa
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j$(nproc)
```

That gives you `llama-cli`, `llama-server` and `llama-bench` in `build/bin/`.

**This stock build runs models 1, 2 and 3.** Only Ternary-Bonsai-8B needs PrismML's fork, which adds the `Q2_0` type:

```bash
cd ~/mufasa
git clone -b prism https://github.com/PrismML-Eng/llama.cpp llama.cpp-prism
cd llama.cpp-prism
cmake -B build -DGGML_CUDA=ON      # drop the flag for a CPU-only box
cmake --build build --config Release -j$(nproc)
```

Keep the two builds in separate directories and record which binary produced which number — a stock build silently refusing to load ternary is not the same measurement as ternary being slow.

The fact that ternary needs a second build **is** the finding. If we cannot hand the judges a model their own tooling loads, its benchmark score is academic. Worth checking whether upstream has since gained `Q2_0`:

```bash
./build/bin/llama-cli --list-types 2>/dev/null | grep -i q2_0 || echo "Q2_0 not in this build"
```

Confirm it runs:

```bash
./build/bin/llama-cli \
  -m ~/mufasa/models/<your-model>.gguf \
  -p "Explain what rice husk ash is used for in concrete." \
  -n 128 -t 4 -c 2048 --no-warmup
```

- `-t 4` — thread cap. Running flat out overheats the laptop, and a thermal trip costs 10 marks.
- `-c 2048` — context cap. We expect to send 1,000–1,500 tokens of evidence per question.

Both are starting guesses. Settling them is part of Task 4.

---

## 4. Benchmark each candidate and report four numbers

**"Benchmark" here just means: run the model under fixed conditions and record how it behaves.** Same prompt, same flags, same machine each time, so the models can be compared to each other.

Four numbers, because each maps to a piece of the score:

| # | Number | Why it matters |
|---|---|---|
| 1 | **Prompt processing** (tokens/sec) | We send 1,000–1,500 tokens of evidence per question. On CPU, reading the prompt often dominates the wait |
| 2 | **Generation speed** (tokens/sec) | Speed is 30 of 100 marks |
| 3 | **Peak RAM** (MB, max RSS) | Efficiency is 20 marks, and over 7 GB ends the run |
| 4 | **Peak CPU temperature** (°C) | Over 85 °C costs 10 marks |

### Numbers 1 and 2 — speed

`llama-bench` does both at once:

```bash
./build/bin/llama-bench \
  -m ~/mufasa/models/<your-model>.gguf \
  -p 512 -n 128 -t 4
```

In the output table, `pp512` is prompt processing and `tg128` is generation, both tokens/sec.

### Number 3 — peak RAM

`Maximum resident set size` is the same thing the judges record:

```bash
/usr/bin/time -v ./build/bin/llama-cli \
  -m ~/mufasa/models/<your-model>.gguf \
  -p "Explain what rice husk ash is used for in concrete." \
  -n 128 -t 4 -c 2048 2>&1 | grep "Maximum resident"
```

It prints kilobytes — divide by 1024 for MB.

### Number 4 — temperature

In a second terminal, while it's generating:

```bash
watch -n 1 sensors
```

Record the highest core temperature you see.

### What to send me

| Model | Quant | pp (tok/s) | tg (tok/s) | Peak RAM (MB) | Peak temp (°C) |
|---|---|---|---|---|---|
| Nanbeige4.2-3B | `Q4_K_M` | | | | |
| Qwen3.5-4B | `Q4_K_M` | | | | |
| Bonsai-8B | `Q1_0` | | | | |
| Ternary-Bonsai-8B | `Q2_0` | | | | |

Note which build produced each row — stock llama.cpp or the PrismML fork.

Plus one line on the machine — CPU, RAM, OS.

### Worth exploring if you have time

There's real room inside llama.cpp: thread count, context length, batch size, quantising the KV cache (`--cache-type-k q8_0 --cache-type-v q8_0`, which roughly halves that scratch memory), and other quantisations besides `Q4_K_M`. `./build/bin/llama-cli --help` lists the rest. Any combination that moves these numbers is a direct contribution to the score.

---

## 5. Tell me what the numbers imply

The scoring formula is `0.50 × accuracy + 0.30 × speed + 0.20 × efficiency − thermal penalty`, so a smaller, faster model can beat a larger, smarter one. Your measurements are the first real data we have on any of this.

The four candidates are asking a real question: **can extreme quantisation give us 8B-class reasoning inside a 3B-class footprint?** Ternary Bonsai claims exactly that — 8.19B parameters in 2.18 GB, smaller than either 4B model. Qwen3.5-4B and Nanbeige are the control: conventional models at the size we would otherwise ship.

Two things constrain the answer, and both need your read:

1. **Ternary cannot be loaded by the judges' tooling** (see Task 2). So its score tells us the size of the prize, not whether we can collect it. If 1-bit Bonsai lands close to ternary, we get most of the benefit in a format that actually ships.
2. **We can only fine-tune Nanbeige.** The Bonsai models are published as inference artifacts with no training path — quantisation-aware training pipeline unpublished, no safetensors. Everything in Layer 2 (SFT, preference tuning) assumes a model we can train, which today means Nanbeige4.2-3B.

So the question isn't only "which is fastest". It's: **how much accuracy would we give up by training Nanbeige rather than shipping an untrainable Bonsai?** If Bonsai's lead is small, the decision is easy. If it's large, we should talk.

Also flag anything that surprised you, or any assumption in [model-training-pipeline.md](./model-training-pipeline.md) your measurements contradict.
