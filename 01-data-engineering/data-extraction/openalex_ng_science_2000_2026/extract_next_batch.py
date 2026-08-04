"""Extract next 400 unclassified papers from partitions for continued classification."""
import json
from pathlib import Path

base = Path(__file__).parent
final_file = base / "classification_output" / "final" / "classified_african_relevance.jsonl"
partitions_dir = base / "classification_output" / "partitions"
output_dir = base / "classification_output" / "next_batches"
output_dir.mkdir(exist_ok=True)

# Load already-classified openalex_ids
classified_ids = set()
with open(final_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            classified_ids.add(rec.get("openalex_id", ""))
        except json.JSONDecodeError:
            continue

print(f"Already classified: {len(classified_ids)} papers")

# Scan partitions for unclassified papers
TARGET = 400
unclassified = []
partition_files = sorted(partitions_dir.glob("partition_*.jsonl"))
print(f"Scanning {len(partition_files)} partition files...")

for pf in partition_files:
    if len(unclassified) >= TARGET:
        break
    with open(pf, "r", encoding="utf-8") as f:
        for line in f:
            if len(unclassified) >= TARGET:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                oid = rec.get("openalex_id", "")
                if oid and oid not in classified_ids:
                    # Extract only the fields the protocol allows
                    paper = {
                        "openalex_id": oid,
                        "title": rec.get("title", ""),
                        "abstract": rec.get("abstract", ""),
                        "field_name": rec.get("field_name", ""),
                        "primary_topic": rec.get("primary_topic", ""),
                        "keywords": rec.get("keywords", [])[:5] if isinstance(rec.get("keywords"), list) else [],
                        "work_type": rec.get("work_type", ""),
                    }
                    # Only include if there's a title and abstract
                    if paper["title"] and paper["abstract"]:
                        unclassified.append(paper)
            except json.JSONDecodeError:
                continue

print(f"Found {len(unclassified)} unclassified papers with title+abstract")

# Split into batches of 100
BATCH_SIZE = 100
batches = []
for i in range(0, len(unclassified), BATCH_SIZE):
    batch = unclassified[i:i+BATCH_SIZE]
    batches.append(batch)

print(f"Created {len(batches)} batches of ~{BATCH_SIZE}")

for idx, batch in enumerate(batches):
    out_file = output_dir / f"next_batch_{idx+1:02d}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for paper in batch:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")
    print(f"  {out_file.name}: {len(batch)} papers")

print(f"\nReady for classification. Total papers to process: {len(unclassified)}")
