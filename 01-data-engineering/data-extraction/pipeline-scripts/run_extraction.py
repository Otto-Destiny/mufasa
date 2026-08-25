"""Converted from run-extraction.ipynb.

Generated for readability in review; the notebook remains the
authority for anything that was actually run.
"""


# # MUFASA extraction run
#
# Three plain calls per paper - observations, study context and profile,
# training examples - written straight to Parquet in `extraction_output/`.
#
# 1. **Run** - set the batch range and watch the bar. It advances once per
#    task, so three times per paper.
# 2. **Results** - row counts and a look at the tables.


# --- knobs, then run it here so you can watch ---------------------------------
import importlib
import sys
from pathlib import Path

DATA = next(
    (folder for start in [Path.cwd(), *Path.cwd().parents]
     for folder in (start, start / "01-data-engineering" / "data-extraction")
     if (folder / "mufasa_corpus" / "parsed" / "markdown").is_dir()),
    None,
)
sys.path.insert(0, str(DATA))
import mufasa_extract
importlib.reload(mufasa_extract)          # pick up edits without restarting

written = mufasa_extract.run(
    # One word swaps the gateway. Both sets of settings live in PROVIDERS in
    # mufasa_extract.py, so the Cavoti configuration we spent so long getting
    # right is still there - put "cavoti" back here and change nothing else.
    provider="openrouter",              # or "cavoti", "tabitoken", "fireworks"

    batch_start=63,          # inclusive, 1-10
    batch_end=70,           # inclusive
    batches=70,             # how many batches the papers are split into
    papers=7000,             # total papers, taken from the top of the corpus

    # model, base_url and key_name come from the provider above. Every numbered
    # variant of its key found in .env is used, so TOKENROUTER_API_KEY picks up
    # KEY2 ... KEY10 as well, exactly as the Cavoti keys did.
    reasoning_effort=None,  # None takes the provider's own default

    # Papers run in parallel, and a paper's three tasks run one after another,
    # so this is also the number of requests in flight. Ten keys, ten workers:
    # roughly one request each. Free model, so overreaching costs rate limits
    # and retries rather than money - drop it if 429s start appearing.
    workers=75,             # papers at once, capped by keys x workers_per_key
    workers_per_key=3,      # headroom for the round-robin landing unevenly
    write_every=25,         # rebuild the Parquet tables every N papers
    attempts=3,             # per request
    max_tokens=120000,
    out="extraction_output",
    redo=False,             # True re-extracts papers already finished
)
print(written)


# --- results -----------------------------------------------------------------
from pathlib import Path

import pandas as pd

DATA = next(
    (folder for start in [Path.cwd(), *Path.cwd().parents]
     for folder in (start, start / "01-data-engineering" / "data-extraction")
     if (folder / "mufasa_corpus" / "parsed" / "markdown").is_dir()),
    None,
)

OUT = DATA / "extraction_output"

TABLES = ("extraction_status", "study_contexts", "observations",
          "entity_mentions", "evidence_spans", "paper_profiles", "training_pairs",
          "african_innovation")
loaded = {}
for name in TABLES:
    path = OUT / f"{name}.parquet"
    if path.exists():
        loaded[name] = pd.read_parquet(path)
        print(f"{name:<20} {len(loaded[name]):>7,} rows")
    else:
        print(f"{name:<20} (not written yet)")

if "extraction_status" in loaded and len(loaded["extraction_status"]):
    status = loaded["extraction_status"]
    print(f"\npapers finished: {len(status)}")
    print(f"mean seconds per paper: {status['seconds'].mean():.0f}")
    print(f"completion tokens: {status['completion_tokens'].sum():,}")

if "training_pairs" in loaded and len(loaded["training_pairs"]):
    print("\ntraining pairs by type:",
          dict(loaded["training_pairs"].pair_type.value_counts()))

if "observations" in loaded and len(loaded["observations"]):
    print("observations by kind:",
          dict(loaded["observations"].statement_kind.value_counts()))
    display(loaded["observations"].head(5))

if "african_innovation" in loaded and len(loaded["african_innovation"]):
    innovation = loaded["african_innovation"]
    print("\nafrican innovation by type:",
          dict(innovation.innovation_type.value_counts()))
    print("papers judged an African innovation:",
          int(innovation.is_african_innovation.fillna(False).astype(bool).sum()),
          "of", len(innovation))
    display(innovation[["paper_id", "innovation_type", "constraint_addressed",
                        "place", "what_is_distinctive"]].head(5))

failures = OUT / "failures.jsonl"
if failures.exists():
    lines = failures.read_text(encoding="utf-8").strip().splitlines()
    print(f"\nfailures logged: {len(lines)}")
    for line in lines[-5:]:
        print(" ", line[:150])

