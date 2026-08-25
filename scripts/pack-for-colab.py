"""Build the single archive that goes to Drive.

Uploading 10,480 loose files to Drive is painfully slow - the per-file overhead
dominates and the transfer stalls repeatedly. One zip moves in a few minutes and
unpacks on the Colab VM in seconds.

  python scripts/pack-for-colab.py            # CPT: markdown only, ~130 MB
  python scripts/pack-for-colab.py --with-raw # adds extraction JSON, for SFT

mufasa_eval.py is placed inside the archive, so the evaluation cells can import
it after unpacking without a second upload.
"""

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "01-data-engineering" / "data-extraction" / "corpus_splits"
EVAL_MODULE = ROOT / "02-model-engineering" / "cpt-notebooks" / "mufasa_eval.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-raw", action="store_true",
                        help="include extraction JSON (needed for SFT, not CPT)")
    parser.add_argument("--out", default=str(ROOT / "corpus_splits.zip"))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not SPLITS.is_dir():
        raise SystemExit(f"no splits at {SPLITS} - run split-corpus.ipynb first")

    kinds = ["markdown"] + (["raw"] if args.with_raw else [])
    out = Path(args.out)
    total, count = 0, 0
    # ZIP_DEFLATED at level 6: markdown compresses ~3x and the extra levels cost
    # minutes for a percent or two.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for split in ("train", "evaluate", "test"):
            for kind in kinds:
                folder = SPLITS / split / kind
                if not folder.is_dir():
                    continue
                files = sorted(folder.iterdir())
                for path in files:
                    archive.write(path, f"corpus_splits/{split}/{kind}/{path.name}")
                    total += path.stat().st_size
                    count += 1
                print(f"   {split}/{kind:<8} {len(files):>6,} files")
        manifest = SPLITS / "manifest.parquet"
        if manifest.is_file():
            archive.write(manifest, "corpus_splits/manifest.parquet")
            count += 1
        if EVAL_MODULE.is_file():
            archive.write(EVAL_MODULE, "corpus_splits/mufasa_eval.py")
            count += 1
            print("   + mufasa_eval.py (so the eval cells can import it)")

    packed = out.stat().st_size
    print(f"\n{out}")
    print(f"   {count:,} files   {total/1e6:,.0f} MB -> {packed/1e6:,.0f} MB "
          f"({100 * packed / max(total, 1):.0f}% of original)")
    print("\nUpload it to Google Drive at  MyDrive/mufasa/corpus_splits.zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
