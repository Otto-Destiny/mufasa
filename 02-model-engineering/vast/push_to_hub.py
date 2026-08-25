"""Colab-side: move the CPT checkpoint and the training set from Drive to the Hub.

Run this in Colab, where both already live. Nothing passes through your laptop.

    !python push_to_hub.py --user YOUR_HF_USER --token hf_xxx --run gemma3-1b-cpt-v4

Both repos are created private. The Vast instance pulls from them directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/mufasa")


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--user", required=True, help="your HF username")
    p.add_argument("--token", required=True, help="HF token with write access")
    p.add_argument("--run", required=True,
                   help="checkpoint folder name, e.g. gemma3-1b-cpt-v4")
    p.add_argument("--drive", default=str(DRIVE))
    p.add_argument("--model-repo", default="", help="defaults to mufasa-<run>")
    p.add_argument("--data-repo", default="", help="defaults to mufasa-sft-mixed")
    p.add_argument("--parquet", default="",
                   help="path to sft_mixed.parquet; by default it is found via "
                        "training_set/LATEST.json")
    p.add_argument("--skip-model", action="store_true", help="upload only the data")
    p.add_argument("--skip-data", action="store_true", help="upload only the model")
    return p.parse_args()


def main():
    args = parse()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from huggingface_hub import HfApi

    drive = Path(args.drive)
    merged = drive / "checkpoints" / args.run / "merged_16bit"
    model_repo = f"{args.user}/{args.model_repo or 'mufasa-' + args.run}"
    data_repo = f"{args.user}/{args.data_repo or 'mufasa-sft-mixed'}"

    # The tokenizer has to be in the folder, or the Vast side loads weights and
    # then fails looking for a tokenizer - the exact failure we already hit once.
    api = HfApi(token=args.token)

    # ---- 1. the merged CPT model ------------------------------------------
    if not args.skip_model:
        if not merged.is_dir():
            raise SystemExit(f"no merged model at {merged}")
        present = {p.name for p in merged.iterdir()}
        if "config.json" not in present:
            raise SystemExit(f"{merged} has no config.json")
        # The tokenizer must travel with the weights. Uploading without it
        # reproduces the failure we already hit: weights load, then the run
        # dies looking for a tokenizer it cannot find.
        if not any(n.startswith("tokenizer") for n in present):
            raise SystemExit(
                f"{merged} has no tokenizer files. Copy them from "
                f"{merged.parent / 'final_adapter'} first."
            )
        size = sum(p.stat().st_size for p in merged.iterdir() if p.is_file()) / 1e9
        print(f"model: {size:,.2f} GB from {merged.name} -> {model_repo}")
        api.create_repo(model_repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(merged), repo_id=model_repo)
        print("   done")

    # ---- 2. the training set ----------------------------------------------
    if not args.skip_data:
        if args.parquet:
            parquet = Path(args.parquet)
        else:
            training_root = drive / "training_set"
            pointer = training_root / "LATEST.json"
            if not pointer.is_file():
                raise SystemExit(
                    f"no {pointer}. Pass --parquet with the full path to "
                    "sft_mixed.parquet instead."
                )
            run = json.loads(pointer.read_text(encoding="utf-8"))
            parquet = training_root / run["directory"] / "sft_mixed.parquet"
        if not parquet.is_file():
            raise SystemExit(f"no training set at {parquet}")
        print(f"data : {parquet.stat().st_size / 1e6:,.0f} MB from "
              f"{parquet.parent.name} -> {data_repo}")
        api.create_repo(data_repo, private=True, exist_ok=True, repo_type="dataset")
        api.upload_file(path_or_fileobj=str(parquet), path_in_repo="sft_mixed.parquet",
                        repo_id=data_repo, repo_type="dataset")
        print("   done")

    print()
    print("On Vast:")
    print(f"  --base   {model_repo}")
    print(f"  --data   {data_repo}")
    print(f"  --output {args.user}/mufasa-{args.run.replace('-cpt', '')}-sft")
    return 0


if __name__ == "__main__":
    sys.exit(main())
