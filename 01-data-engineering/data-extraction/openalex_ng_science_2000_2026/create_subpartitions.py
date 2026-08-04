#!/usr/bin/env python3
"""Create sub-partitions of ~500 records each for AI agent classification."""
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARTITIONS_DIR = os.path.join(BASE_DIR, 'classification_output', 'partitions')
SUBPART_DIR = os.path.join(BASE_DIR, 'classification_output', 'sub_partitions')
RECORDS_PER_SUB = 500

os.makedirs(SUBPART_DIR, exist_ok=True)

sub_idx = 0
total_records = 0
manifest = []

for pfile in sorted(os.listdir(PARTITIONS_DIR)):
    if not pfile.endswith('.jsonl'):
        continue
    ppath = os.path.join(PARTITIONS_DIR, pfile)

    with open(ppath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i in range(0, len(lines), RECORDS_PER_SUB):
        chunk = lines[i:i + RECORDS_PER_SUB]
        sub_name = f"sub_{sub_idx:04d}.jsonl"
        sub_path = os.path.join(SUBPART_DIR, sub_name)

        with open(sub_path, 'w', encoding='utf-8') as f:
            f.writelines(chunk)

        manifest.append({
            'sub_id': sub_idx,
            'file': sub_name,
            'records': len(chunk),
            'source_partition': pfile,
        })
        total_records += len(chunk)
        sub_idx += 1

# Write manifest
with open(os.path.join(SUBPART_DIR, 'manifest.json'), 'w') as f:
    json.dump({'total_records': total_records, 'num_sub_partitions': sub_idx, 'sub_partitions': manifest}, f, indent=2)

print(f"Created {sub_idx} sub-partitions with {total_records:,} total records")
print(f"Average: {total_records // sub_idx} records per sub-partition")
print(f"Output: {SUBPART_DIR}")
