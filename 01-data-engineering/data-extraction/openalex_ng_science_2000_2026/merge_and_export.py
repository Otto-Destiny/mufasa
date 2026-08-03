#!/usr/bin/env python3
"""
Merge all AI agent classification outputs into final formats.
- Combines all JSONL result files
- Writes 10 split JSONL parts
- Writes combined CSV
- Writes combined Parquet

This script only merges and formats — classification was done by AI agents.
"""

import json
import csv
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'classification_output')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
FINAL_DIR = os.path.join(OUTPUT_DIR, 'final')

NUM_JSONL_PARTS = 10

OUTPUT_COLUMNS = [
    'openalex_id', 'title', 'decision', 'hard_exclusion',
    'hard_exclusion_reason', 'evidence_level', 'african_centrality',
    'local_specificity', 'scientific_depth', 'knowledge_value',
    'local_applicability', 'total_score', 'african_focus',
    'scientific_evidence', 'african_country_codes', 'african_relevance_tags',
    'evidence', 'inference_basis', 'reason',
    'field_name', 'primary_topic', 'work_type',
]


def main():
    os.makedirs(FINAL_DIR, exist_ok=True)

    print("=" * 60)
    print("  MUFASA Classification Merge & Export")
    print("=" * 60)

    # Collect all result files
    result_files = sorted([
        f for f in os.listdir(RESULTS_DIR)
        if f.endswith('.jsonl')
    ])
    print(f"\nFound {len(result_files)} result files in {RESULTS_DIR}")

    # Read all results
    all_results = []
    errors = 0
    for fname in result_files:
        fpath = os.path.join(RESULTS_DIR, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Validate required fields
                    if 'decision' not in record:
                        errors += 1
                        continue
                    all_results.append(record)
                except json.JSONDecodeError:
                    errors += 1

    total = len(all_results)
    print(f"Total records merged: {total:,}")
    if errors:
        print(f"Parse errors skipped: {errors}")

    if total == 0:
        print("ERROR: No records to merge!")
        sys.exit(1)

    # Statistics
    from collections import Counter
    decisions = Counter(r.get('decision', 'unknown') for r in all_results)
    evidence = Counter(r.get('evidence_level', 'unknown') for r in all_results)

    print(f"\nDecision distribution:")
    for d, c in decisions.most_common():
        print(f"  {d:>10s}: {c:>8,}  ({c/total*100:.1f}%)")

    print(f"\nEvidence levels:")
    for e, c in evidence.most_common():
        print(f"  {e:>20s}: {c:>8,}  ({c/total*100:.1f}%)")

    # ---- Write 10 JSONL parts ----
    print(f"\nWriting {NUM_JSONL_PARTS} JSONL parts...")
    part_size = total // NUM_JSONL_PARTS
    remainder = total % NUM_JSONL_PARTS

    offset = 0
    for i in range(NUM_JSONL_PARTS):
        size = part_size + (1 if i < remainder else 0)
        part_records = all_results[offset:offset + size]
        offset += size

        fname = f"classified_african_relevance_part_{i+1:02d}.jsonl"
        fpath = os.path.join(FINAL_DIR, fname)

        with open(fpath, 'w', encoding='utf-8') as f:
            for r in part_records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        sz = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  Part {i+1:2d}: {len(part_records):>7,} records ({sz:.1f} MB)")

    # ---- Write CSV ----
    print("\nWriting CSV...")
    csv_path = os.path.join(FINAL_DIR, 'classified_african_relevance.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for r in all_results:
            row = {}
            for col in OUTPUT_COLUMNS:
                val = r.get(col, '')
                if isinstance(val, bool):
                    val = str(val).upper()
                elif isinstance(val, list):
                    val = json.dumps(val)
                row[col] = val
            writer.writerow(row)
    sz = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"  CSV: {csv_path} ({sz:.1f} MB)")

    # ---- Write Parquet ----
    print("Writing Parquet...")
    parquet_path = os.path.join(FINAL_DIR, 'classified_african_relevance.parquet')

    # Build typed columns
    data = {col: [] for col in OUTPUT_COLUMNS}
    for r in all_results:
        for col in OUTPUT_COLUMNS:
            val = r.get(col, '')
            if col == 'hard_exclusion':
                val = bool(val) if val != '' else False
            elif col in ('african_centrality', 'local_specificity',
                         'scientific_depth', 'knowledge_value',
                         'local_applicability', 'total_score'):
                try:
                    val = int(val) if val != '' else 0
                except (ValueError, TypeError):
                    val = 0
            elif isinstance(val, list):
                val = json.dumps(val)
            elif val is None:
                val = ''
            data[col].append(val)

    table = pa.table(data)
    pq.write_table(table, parquet_path, compression='snappy')
    sz = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"  Parquet: {parquet_path} ({sz:.1f} MB)")

    print(f"\n{'='*60}")
    print(f"  EXPORT COMPLETE")
    print(f"  Total records: {total:,}")
    print(f"  JSONL parts: {NUM_JSONL_PARTS}")
    print(f"  Output dir: {FINAL_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
