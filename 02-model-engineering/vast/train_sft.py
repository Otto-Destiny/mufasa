"""Full-parameter SFT for MUFASA, multi-GPU, everything through the Hub.

Nothing here is a demo value. There is no `max_steps`: the run is defined in
epochs, so it ends when the data has been seen, not when a counter runs out.

Launched with torchrun, one process per GPU:

    torchrun --nproc_per_node 8 train_sft.py \
        --base   USER/mufasa-gemma3-1b-cpt \
        --data   USER/mufasa-sft-mixed \
        --output USER/mufasa-gemma3-1b-sft

Model in from the Hub, weights out to the Hub. The instance disk holds only
transient checkpoints, because a rented box is not storage.

Prompt masking is done by giving TRL a prompt/completion dataset rather than
pre-rendered text: TRL then masks the prompt itself, which is the same thing
`train_on_responses_only` did in the notebook but without depending on the
chat template exposing a generation marker.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_MAIN = LOCAL_RANK == 0


def say(*parts):
    if IS_MAIN:
        print(*parts, flush=True)


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="HF repo of the merged CPT model")
    p.add_argument("--data", required=True, help="HF dataset repo holding sft_mixed")
    p.add_argument("--output", required=True, help="HF repo to push the result to")
    p.add_argument("--data-file", default="sft_mixed.parquet")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16,
                   help="per device; effective = batch x accum x GPUs")
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-seq-length", type=int, default=4096,
                   help="p99 of this corpus is ~2,300 tokens; 4096 truncates nothing")
    p.add_argument("--tiers", default="",
                   help="comma-separated verification_tier filter; empty = all")
    p.add_argument("--eval-rows", type=int, default=400)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--local-dir", default="/workspace/sft_run")
    p.add_argument("--attn", default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])
    # A 1B model in bf16 with fused Adam is ~10 GB. On a 288 GB B300 there is
    # nothing to save, and checkpointing costs 30-40% throughput by recomputing
    # activations. Turn it on only if you see OOM.
    p.add_argument("--grad-checkpointing", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--private", action="store_true", default=True)
    p.add_argument("--no-push", action="store_true",
                   help="train but do not upload; for a dry run on one GPU")
    return p.parse_args()


@dataclass
class Split:
    train: Dataset
    holdout: Dataset


def build_data(args) -> Split:
    """prompt/completion pairs, so TRL masks the prompt and trains the answer."""
    raw = load_dataset(args.data, data_files=args.data_file, split="train")
    keep = ["split", "verification_tier", "prompt", "response"]
    raw = raw.remove_columns([c for c in raw.column_names if c not in keep])

    tiers = {t.strip() for t in args.tiers.split(",") if t.strip()}
    if tiers:
        raw = raw.filter(lambda r: r["verification_tier"] in tiers,
                         num_proc=8, desc="tier filter")

    def shape(rows):
        return {"prompt": rows["prompt"], "completion": rows["response"]}

    train = raw.filter(lambda r: r["split"] == "train", num_proc=8, desc="train split")
    holdout = raw.filter(lambda r: r["split"] == "evaluate", num_proc=8, desc="held out")
    # Shuffle before selecting. Taking the first N rows draws them from a
    # handful of papers - the parquet is ordered by paper - so the curve would
    # measure those few papers rather than the held-out split. Seeded, so the
    # same rows are scored at every eval and across runs.
    holdout = holdout.shuffle(seed=args.seed)
    holdout = holdout.select(range(min(args.eval_rows, len(holdout))))

    drop = ["split", "verification_tier", "response"]
    train = train.map(shape, batched=True, num_proc=8, remove_columns=drop,
                      desc="prompt/completion")
    holdout = holdout.map(shape, batched=True, num_proc=8, remove_columns=drop,
                          desc="prompt/completion")
    return Split(train=train, holdout=holdout)


def main():
    args = parse()
    set_seed(args.seed)

    if IS_MAIN:
        card = torch.cuda.get_device_properties(0)
        say(f"GPUs           : {WORLD_SIZE} x {card.name} "
            f"({card.total_memory / 1e9:,.0f} GB each)")
        say(f"bfloat16       : {torch.cuda.is_bf16_supported()}")

    data = build_data(args)
    effective = args.batch_size * args.accum * WORLD_SIZE
    steps = int(len(data.train) * args.epochs / max(effective, 1))
    say(f"train examples : {len(data.train):,}")
    say(f"held out       : {len(data.holdout):,}")
    say(f"effective batch: {effective}  ({args.batch_size} x {args.accum} x {WORLD_SIZE})")
    say(f"optimiser steps: ~{steps:,} for {args.epochs} epoch(s)")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )
    model.config.use_cache = False          # never needed while training

    # Built as a dict and filtered against what this TRL/transformers pair
    # actually accepts. Argument names move between versions - group_by_length
    # was dropped from TrainingArguments in transformers 5.x - and a rented box
    # is the wrong place to discover that.
    wanted = dict(
        output_dir=args.local_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.001,
        max_length=args.max_seq_length,
        bf16=True,
        # Sorting similar lengths together removes most of the padding waste.
        # The corpus median is ~1,065 tokens against a 4,096 window, so without
        # this the GPUs spend a large share of their time on padding.
        group_by_length=True,
        gradient_checkpointing=args.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        # DDP: a 1B model fits on one card many times over, so there is nothing
        # to shard. Each rank holds the whole model and takes a different slice.
        ddp_find_unused_parameters=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
    )

    from dataclasses import fields as _fields
    accepted = {f.name for f in _fields(SFTConfig)}
    dropped = sorted(set(wanted) - accepted)
    if dropped:
        say(f"note: this TRL build ignores {dropped} - dropped")
    config = SFTConfig(**{k: v for k, v in wanted.items() if k in accepted})

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=data.train,
        eval_dataset=data.holdout,
        processing_class=tokenizer,
    )

    resume = any(
        name.startswith("checkpoint-")
        for name in (os.listdir(args.local_dir) if os.path.isdir(args.local_dir) else [])
    )
    say("resuming from checkpoint" if resume else "starting fresh")
    result = trainer.train(resume_from_checkpoint=resume or None)

    if IS_MAIN:
        say("\n" + "=" * 66)
        for key in ("train_runtime", "train_samples_per_second", "train_loss"):
            if key in result.metrics:
                say(f"  {key:<28} {result.metrics[key]:,.4f}")
        final = trainer.evaluate()
        say(f"  {'final eval_loss':<28} {final.get('eval_loss', float('nan')):,.4f}")

    trainer.save_model(args.local_dir)
    if IS_MAIN:
        tokenizer.save_pretrained(args.local_dir)
        (open(os.path.join(args.local_dir, "mufasa_run.json"), "w")
         .write(json.dumps({**vars(args), "world_size": WORLD_SIZE,
                            "effective_batch": effective,
                            "metrics": result.metrics}, indent=2, default=str)))

    if IS_MAIN and not args.no_push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.output, private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=args.local_dir, repo_id=args.output,
            ignore_patterns=["checkpoint-*", "*.pt", "optimizer*", "scheduler*", "rng*"],
        )
        say(f"pushed to https://huggingface.co/{args.output}")


if __name__ == "__main__":
    main()
