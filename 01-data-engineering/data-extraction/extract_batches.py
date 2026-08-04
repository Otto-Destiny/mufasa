"""Extract paper metadata (no gold columns) into JSON batches for manual reasoning."""
import csv
import json
from pathlib import Path

input_file = Path(__file__).parent / "classification_benchmark_200.csv"
output_dir = Path(__file__).parent / "batches"
output_dir.mkdir(exist_ok=True)

# Fields we need for classification reasoning (NO gold_* columns)
KEEP_FIELDS = [
    "benchmark_id", "field_id", "field_name", "openalex_id", "doi",
    "title", "abstract", "publication_date", "publication_year",
    "work_type", "language", "cited_by_count",
    "primary_domain", "primary_field", "primary_subfield", "primary_topic",
    "keywords_json", "authors_json", "institutions_json", "countries_json",
    "journal",
]

with open(input_file, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total papers: {len(rows)}")

# Extract only needed fields
papers = []
for row in rows:
    paper = {}
    for field in KEEP_FIELDS:
        val = row.get(field, "")
        # Parse JSON fields for readability
        if field in ("keywords_json", "authors_json", "institutions_json", "countries_json"):
            try:
                val = json.loads(val) if val else []
            except (json.JSONDecodeError, TypeError):
                val = []
        paper[field] = val
    papers.append(paper)

# Split into 8 batches of 25
BATCH_SIZE = 25
batches = []
for i in range(0, len(papers), BATCH_SIZE):
    batch = papers[i:i+BATCH_SIZE]
    batches.append(batch)

print(f"Created {len(batches)} batches of ~{BATCH_SIZE} papers each")

# Write each batch
for idx, batch in enumerate(batches):
    batch_file = output_dir / f"batch_{idx+1:02d}.json"
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(batch, f, indent=2, ensure_ascii=False)
    print(f"  {batch_file.name}: {len(batch)} papers")

# Also write the full extraction for reference
full_file = output_dir / "all_papers_metadata.json"
with open(full_file, "w", encoding="utf-8") as f:
    json.dump(papers, f, indent=2, ensure_ascii=False)
print(f"\nFull metadata: {full_file}")
