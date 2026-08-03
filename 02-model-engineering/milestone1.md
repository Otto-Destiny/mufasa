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

**Nanbeige 4.2 (3B):** https://huggingface.co/bartowski/Nanbeige_Nanbeige4.2-3B-GGUF
**Qwen3 1.7B:** https://huggingface.co/Qwen/Qwen3-1.7B-GGUF

| Model | `Q4_K_M` file | Expect peak RAM |
|---|---|---|
| Nanbeige4.2-3B | 2.68 GB | ~3.0–3.2 GB |
| Qwen3-1.7B | ~1.1 GB | ~1.4–1.6 GB |

Despite its name, Nanbeige "3B" is really around 4B parameters. Add more candidates if you think they're worth measuring — the shortlist isn't fixed.

Easiest by browser: open a link, click **Files**, download the `Q4_K_M` file. From the terminal:

```bash
pip install -U "huggingface_hub[cli]"
mkdir -p ~/mufasa/models && cd ~/mufasa/models
```

Repos differ on capitalisation (`Q4_K_M` vs `q4_k_m`), so list what's there first:

```bash
python3 -c "from huggingface_hub import list_repo_files; \
print('\n'.join(f for f in list_repo_files('bartowski/Nanbeige_Nanbeige4.2-3B-GGUF') if f.endswith('.gguf')))"
```

Then pull the exact filename:

```bash
hf download bartowski/Nanbeige_Nanbeige4.2-3B-GGUF <exact-filename>.gguf --local-dir .
```

If a quant is split across parts, take all of them — llama.cpp loads the first and finds the rest.

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
| Qwen3-1.7B | `Q4_K_M` | | | | |

Plus one line on the machine — CPU, RAM, OS.

### Worth exploring if you have time

There's real room inside llama.cpp: thread count, context length, batch size, quantising the KV cache (`--cache-type-k q8_0 --cache-type-v q8_0`, which roughly halves that scratch memory), and other quantisations besides `Q4_K_M`. `./build/bin/llama-cli --help` lists the rest. Any combination that moves these numbers is a direct contribution to the score.

---

## 5. Tell me what the numbers imply

The scoring formula is `0.50 × accuracy + 0.30 × speed + 0.20 × efficiency − thermal penalty`, so a smaller model that's twice as fast can beat a larger, smarter one. Our rough estimate is that a 4B would need to be about **36 accuracy points** better than a 1.7B just to break even — but that was calculated on paper, and your measurements are the first real data.

So: which size should we build on, and what do the numbers say that the estimate didn't?

Also worth flagging anything that surprised you, or any assumption in [model-training-pipeline.md](./model-training-pipeline.md) that your measurements contradict.
