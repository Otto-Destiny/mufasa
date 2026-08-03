"""Assemble agent classification results into final qwen-labeled CSV."""
import csv
import json
from pathlib import Path

base_dir = Path(__file__).parent
batches_dir = base_dir / "batches"
input_csv = base_dir / "classification_benchmark_200.csv"
output_csv = base_dir / "classification_benchmark_200_qwen_labeled.csv"

# Load all result files
all_results = {}
for i in range(1, 9):
    result_file = batches_dir / f"results_{i:02d}.json"
    if not result_file.exists():
        print(f"WARNING: Missing {result_file}")
        continue
    with open(result_file, "r", encoding="utf-8") as f:
        batch_results = json.load(f)
    for item in batch_results:
        bid = item.get("benchmark_id", "")
        all_results[bid] = item
    print(f"  Loaded {result_file.name}: {len(batch_results)} papers")

print(f"\nTotal classified: {len(all_results)} papers")

# Define qwen columns
QWEN_COLUMNS = [
    "qwen_decision",
    "qwen_hard_exclusion",
    "qwen_african_centrality",
    "qwen_local_specificity",
    "qwen_scientific_depth",
    "qwen_knowledge_value",
    "qwen_local_applicability",
    "qwen_total_score",
    "qwen_african_focus",
    "qwen_scientific_evidence",
    "qwen_african_country_codes",
    "qwen_african_relevance_tags",
    "qwen_evidence",
    "qwen_reason",
    "qwen_primary_domain",
    "qwen_category_codes",
]

# Read original CSV
with open(input_csv, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    original_fieldnames = list(reader.fieldnames)
    rows = list(reader)

print(f"Original CSV: {len(rows)} rows, {len(original_fieldnames)} columns")

# Output fieldnames: all original + qwen columns
output_fieldnames = original_fieldnames + QWEN_COLUMNS

# Merge results into rows
matched = 0
missing = 0
for row in rows:
    bid = row.get("benchmark_id", "")
    if bid in all_results:
        result = all_results[bid]
        for col in QWEN_COLUMNS:
            val = result.get(col, "")
            # Convert lists/dicts to JSON strings for CSV
            if isinstance(val, (list, dict)):
                val = json.dumps(val, ensure_ascii=False)
            row[col] = val
        matched += 1
    else:
        # Fill with empty if missing
        for col in QWEN_COLUMNS:
            row[col] = ""
        missing += 1

print(f"Matched: {matched}, Missing: {missing}")

# Write output CSV
with open(output_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=output_fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print(f"\nFinal output: {output_csv}")
print(f"Total columns: {len(output_fieldnames)} (original {len(original_fieldnames)} + {len(QWEN_COLUMNS)} qwen)")

# Summary statistics
from collections import Counter
decisions = Counter(all_results[bid].get("qwen_decision", "MISSING") for bid in all_results)
print(f"\n=== Decision Distribution ===")
for k, v in decisions.most_common():
    print(f"  {k}: {v}")

focuses = Counter(all_results[bid].get("qwen_african_focus", "MISSING") for bid in all_results)
print(f"\n=== African Focus Distribution ===")
for k, v in focuses.most_common():
    print(f"  {k}: {v}")

domains = Counter(all_results[bid].get("qwen_primary_domain", "MISSING") for bid in all_results)
print(f"\n=== Domain Distribution ===")
for k, v in domains.most_common():
    print(f"  {k}: {v}")
