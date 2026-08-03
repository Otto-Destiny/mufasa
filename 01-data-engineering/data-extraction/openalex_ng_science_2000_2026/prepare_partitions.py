#!/usr/bin/env python3
"""
Data preparation script for MUFASA African Relevance Classification.
This script ONLY reads, extracts, and partitions the data.
Classification reasoning is done by AI agents, NOT by this script.

Steps:
  1. Read openalex_all_fields.csv (handle encoding)
  2. Extract only protocol-relevant fields (title, abstract, field_name, primary_topic, keywords, work_type)
  3. Split into 20 partition JSONL files for parallel agent processing
  4. Each partition: ~7,800 records with only the fields the protocol allows
"""

import json
import csv
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, 'openalex_all_fields.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'classification_output')
PARTITIONS_DIR = os.path.join(OUTPUT_DIR, 'partitions')
NUM_PARTITIONS = 20

# Protocol-relevant fields ONLY
# Excluded: authors, institutions, countries, DOI, journal, citations, dates, URLs, licences
USE_FIELDS = ['openalex_id', 'title', 'abstract', 'field_name', 'primary_topic', 'work_type', 'keywords_json']


def safe_json_parse(val):
    if val is None or val == '' or val == 'nan':
        return None
    try:
        return json.loads(str(val))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def extract_keywords(kw_json, max_kw=5):
    parsed = safe_json_parse(kw_json)
    if not parsed or not isinstance(parsed, list):
        return []
    result = []
    for item in parsed[:max_kw]:
        if isinstance(item, dict):
            name = item.get('name', '')
            if name:
                result.append(name)
        elif isinstance(item, str):
            result.append(item)
    return result


def main():
    os.makedirs(PARTITIONS_DIR, exist_ok=True)

    print("=" * 60)
    print("  MUFASA Data Preparation")
    print("  Extracting protocol-relevant fields only")
    print("=" * 60)

    # Read CSV with encoding fallback
    print(f"\nReading: {INPUT_CSV}")
    for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
        try:
            with open(INPUT_CSV, 'r', encoding=encoding, errors='replace') as f:
                reader = csv.DictReader(f)
                records = []
                for row in reader:
                    rec = {}
                    for field in USE_FIELDS:
                        val = row.get(field, '')
                        if val == 'nan' or val is None:
                            val = ''
                        rec[field] = str(val) if val else ''
                    # Extract clean keywords
                    rec['keywords'] = extract_keywords(rec.get('keywords_json', ''))
                    del rec['keywords_json']  # Remove raw JSON
                    records.append(rec)
            print(f"  Successfully read with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            print(f"  Encoding {encoding} failed, trying next...")
            continue
    else:
        print("ERROR: Could not read CSV with any encoding")
        sys.exit(1)

    total = len(records)
    print(f"  Total records: {total:,}")

    # Calculate partition sizes
    base_size = total // NUM_PARTITIONS
    remainder = total % NUM_PARTITIONS

    print(f"\nPartitioning into {NUM_PARTITIONS} files...")
    print(f"  Base size: {base_size} records/partition")
    print(f"  Remainder: {remainder} extra records")

    offset = 0
    partition_info = []
    for i in range(NUM_PARTITIONS):
        size = base_size + (1 if i < remainder else 0)
        partition_records = records[offset:offset + size]
        offset += size

        filename = f"partition_{i:02d}.jsonl"
        filepath = os.path.join(PARTITIONS_DIR, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            for rec in partition_records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        partition_info.append({
            'partition_id': i,
            'filename': filename,
            'filepath': filepath,
            'record_count': len(partition_records),
            'size_mb': round(file_size_mb, 2),
        })
        print(f"  Partition {i:2d}: {len(partition_records):>6,} records ({file_size_mb:.1f} MB)")

    # Write partition manifest
    manifest_path = os.path.join(OUTPUT_DIR, 'partition_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_records': total,
            'num_partitions': NUM_PARTITIONS,
            'partitions': partition_info,
        }, f, indent=2)

    print(f"\nManifest: {manifest_path}")
    print(f"Partitions: {PARTITIONS_DIR}")
    print(f"\n{'=' * 60}")
    print(f"  Data preparation complete!")
    print(f"  {total:,} records partitioned into {NUM_PARTITIONS} files")
    print(f"  Ready for AI agent classification")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
