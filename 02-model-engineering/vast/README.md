# MUFASA SFT on rented GPUs

Full-parameter SFT, one epoch, multi-GPU. Model in from the Hub, weights out to
the Hub. Nothing passes through your machine and nothing durable lives on the
rented box.

## Why this is a script and not the notebook

**Unsloth's free tier is single-GPU.** The notebook's own banner said
`Num GPUs used = 1`. Eight cards means standard `trl.SFTTrainer` under
`torchrun` instead — the same trainer the notebook used underneath, without
Unsloth's kernels. Expect somewhat lower per-GPU throughput than Unsloth would
give on one card, more than repaid by having eight.

## There are no demo values here

The earlier notebook run stopped after `max_steps = 100`, which at batch 4 is
**400 of 353,697 examples — 0.1% of one epoch**. That is all "smoke test"
meant. This script has no `max_steps`: the run is defined in epochs and ends
when the data has been seen.

## Step 1 — Colab: Drive to the Hub

```python
!python push_to_hub.py --user YOUR_HF_USER --token hf_xxx --run gemma3-1b-cpt-v4
```

Creates two private repos: the merged CPT model (~2 GB) and the training set
(~113 MB). It refuses to upload a model folder with no tokenizer files, because
that exact gap already cost one failed run.

## Step 2 — Vast: train

Pick an image with CUDA 12.4+ and PyTorch preinstalled. Then:

```bash
export HF_TOKEN=hf_xxx
export HF_USER=your_user
bash run.sh gemma3-1b-cpt-v4
```

`run.sh` detects the GPU count, installs the pinned stack, and launches
`torchrun`. When it finishes the weights are on the Hub and the box can be
destroyed.

## The knobs

| flag | default | note |
|---|---|---|
| `--epochs` | 1.0 | fractional is allowed, e.g. `0.5` |
| `--batch-size` | 16 (32 in run.sh) | per device; effective = batch x accum x GPUs |
| `--lr` | 2e-5 | full finetuning wants less than LoRA |
| `--max-seq-length` | 4096 | measured p99 is 1,874, max 3,309; nothing is truncated |
| `--grad-checkpointing` | **off** | a 1B model is ~10 GB on a 288 GB card; leaving it off buys 30-40% throughput. Turn on only for OOM |
| `--attn` | flash_attention_2 | `sdpa` if flash-attn will not build |
| `--tiers` | all | e.g. `VERIFIED` to train on the strict subset only |
| `--no-push` | off | train without uploading, for a one-GPU dry run |

## Time budget, honestly

Measured with Gemma's own tokenizer on 3,000 sampled training rows:
mean **876** tokens, median 824, p95 1,564, p99 1,874, max 3,309. So one epoch
is **310M tokens**, and nothing exceeds the 4,096 window.

| achieved throughput | 8 GPUs | wall time |
|---|---|---|
| 20,000 tok/s/GPU | 160k tok/s | 32 min |
| 35,000 tok/s/GPU | 280k tok/s | 18 min |
| 50,000 tok/s/GPU | 400k tok/s | 13 min |

Under 30 minutes needs about 22k tok/s per GPU. That is plausible on B300 for a
1B model in bf16 with checkpointing off and flash-attention on, but it is a
projection, not a measurement — nobody here has benchmarked a 1B full finetune
on that part. **Read `train_samples_per_second` from the first eval and
recompute before assuming the whole epoch fits your budget.**

Two settings do most of the work: `--grad-checkpointing` staying off, and
`group_by_length` (always on) which sorts similar-length sequences together.
The measured median is 824 tokens against a 4,096 window, so without that
sorting the cards would spend most of their time multiplying padding.

## What the script does that the notebook did not

- **Prompt masking via prompt/completion columns.** TRL masks the prompt
  itself, so loss is computed on the answer only. Same intent as
  `train_on_responses_only`, without depending on the chat template exposing a
  generation marker.
- **`load_best_model_at_end` on `eval_loss`.** If the run overfits before the
  epoch ends, the pushed weights are the best checkpoint, not the last one.
- **Resume.** An interrupted run picks up from the newest local checkpoint.
- **`mufasa_run.json`** written beside the weights: every argument, the world
  size, the effective batch, and the final metrics.

## Sanity checks before the long run

```bash
torchrun --standalone --nproc_per_node 1 train_sft.py \
    --base $HF_USER/mufasa-gemma3-1b-cpt-v4 \
    --data $HF_USER/mufasa-sft-mixed \
    --output unused --epochs 0.002 --no-push
```

One GPU, a fraction of a percent of the data, no upload. It proves the repos
resolve, the tokenizer loads, the masking works and the loss is finite. That is
a smoke test — and this time it is labelled as one.
