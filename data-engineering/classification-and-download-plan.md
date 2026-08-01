# MUFASA Classification and Download Plan

Status: agreed working plan
Current focus: classify metadata on Kaggle; do not download PDFs yet

## 1. Goal

Select exactly **6,000 Africa-relevant scientific papers** from the OpenAlex candidate pool.

Classification answers two separate questions:

1. Is the paper scientifically relevant to Africa?
2. Which MUFASA discipline and category codes describe it?

OpenAlex fields organize the queues. They do not determine the MUFASA discipline label.

## 2. Sources

The rubric and taxonomy come from:

- `MUFASA/README.md`
- `MUFASA/model-engineering/model-training-pipeline.md`
- `MUFASA/data-engineering/taxonomy/african-science-categories.md`
- `MUFASA/data-engineering/taxonomy/categories/*.md`
- `MUFASA/data-engineering/catalogs/materials-sources.md`

`download_and_classify.md` contains useful background ideas, but it is not authoritative.

## 3. Production Candidate Ordering

Build one queue for each of the 20 OpenAlex `primary_field` values.

Within every field, sort in this exact order:

1. `cited_by_count` descending
2. `publication_date` descending
3. `openalex_id` ascending as a stable tie-breaker

Therefore, citation count is the most important ordering rule. A newer paper is considered first only when citation counts are equal.

## 4. Classification Decision

Each paper receives one decision:

- `include`: sufficiently supported as Africa-relevant scientific work
- `exclude`: clearly outside the rubric
- `review`: possibly relevant, but the metadata is insufficient or ambiguous

Only `include` counts toward the 6,000-paper target. `review` is saved for later inspection and is not silently treated as either included or excluded.

### Relevance test

A paper may be included when an African material, organism, crop, population, environment, dataset, industrial system, constraint, method, or scientific problem is important to the research question, evidence, interpretation, or application.

African author affiliation alone is not enough.

### Scores

Score each dimension from 0 to 4:

- `african_centrality`
- `local_specificity`
- `scientific_depth`
- `knowledge_value`
- `local_applicability`

An `include` normally requires:

- no hard exclusion
- total score of at least 14/20
- `african_centrality >= 2`
- `scientific_depth >= 2`
- supporting evidence in the title or abstract

Uncertain cases go to `review`, protecting against false negatives.

## 5. Reaching Exactly 6,000

### First pass

Process each field's ordered queue until that field has:

- 300 `include` papers, or
- no candidates left.

Excluded and review papers do not count toward 300, so the classifier continues down that field's queue until it reaches the target or exhausts the field.

### Carry-over pass

Some fields cannot supply 300 included papers. Add all such shortages into one deficit.

Merge the unprocessed tails of fields that still have candidates. Sort that combined carry-over queue by:

1. `cited_by_count` descending
2. `publication_date` descending
3. `openalex_id` ascending

Continue classification until the overall number of `include` papers is exactly 6,000. Then stop. Do not classify the rest of the 155,825 candidates.

This keeps the procedure simple and lets the strongest remaining cited candidates fill field shortages.

### Prepared Kaggle input

`openalex_candidates_slim.parquet` is ready for upload to Kaggle:

- 155,759 usable papers
- 20 OpenAlex primary fields
- 11 essential classification and identification columns
- approximately 76 MB
- DOI and institution names retained
- abstracts capped at 1,800 characters because the complete model prompt is capped at that length
- URL, license, endpoint, full JSON, duplicate field, queue-rank, and hash columns removed
- no gold-label or MUFASA category columns
- verified citation/date/ID ordering

The 11 columns are:

```text
field_id, field_name, openalex_id, doi, title, abstract,
publication_date, cited_by_count, primary_topic, keywords,
institution_names
```

The original CSV remains the source for complete abstracts and later PDF/download
metadata. The slim file is only the Kaggle classification input.

## 6. Golden Benchmark

Use one **200-paper** benchmark:

- 10 papers from each of the 20 OpenAlex primary fields
- not limited to the ten most-cited papers
- queue ranks 1, 5, and 10 from each field
- seven additional papers evenly spaced through the first 3,000 queue positions, or through the whole field when it contains fewer than 3,000 candidates
- same records, order, prompt, and limits for every model

The runtime benchmark file is `classification_benchmark_200.parquet`. The CSV
copy is not used by the Kaggle notebook.

Citation-first ordering controls the production workflow. The benchmark deliberately covers multiple queue depths so model quality is not judged only on unusually highly cited papers.

All 200 `gold_*` decisions and reasons have now been manually reviewed and
confirmed. Every row has:

```text
gold_review_status = confirmed
gold_label_source = frontier_seed
```

Gold columns and MUFASA category codes are never passed to a model. Benchmark
prompts contain only the relevance rubric and selected paper metadata. The
current benchmark evaluates relevance decisions, not taxonomy codes.

This same set measures both classification quality and rough elapsed processing time. There is no separate 1,000-paper throughput set.

## 7. Models to Compare on Kaggle

Test:

1. Qwen3.5 2B
2. Qwen3.5 4B
3. Bonsai 27B 1-bit

Common rules:

- paper metadata payload below 600 tokens
- fixed rubric and JSON schema
- deterministic classification
- same maximum output length
- one retry for invalid structured output
- persistent failures become `review`

Choose the fastest model that is acceptably accurate on the 200 confirmed human labels. The most important safety measure is avoiding false exclusions of genuinely relevant papers.

Use 100 records as a convenient processing and checkpoint chunk. This is only an operational batch size; it does not change selection or evaluation.

## 8. Kaggle Notebook

The self-contained Kaggle notebook will:

1. Load and validate the slim input Parquet.
2. Create the citation-first queue for every OpenAlex field.
3. Run a 10-paper smoke test.
4. Benchmark each model on the same 200 papers.
5. Display a metrics table, quality bars, confusion matrix, decision counts, and processing speed.
6. Save predictions, accuracy measures, plots, elapsed time, and failures.
7. Compare all completed model benchmarks in one quality-and-speed view.
8. Stop for model selection.
9. Run the selected model using the 300-per-field and carry-over procedure.
10. Stop immediately when 6,000 papers have decision `include`.
11. Save the completed relevance outputs as Parquet and CSV.

Checkpoint after each 100 processed papers so a Kaggle interruption can resume without repeating finished records.

Visible progress bars report completion percentage, elapsed time, estimated
remaining time, current field/stage, processed count, and live decision or
selection counts.

MUFASA discipline/category classification will be a separate later pass. Its
category list is not part of the relevance-classification prompt.

## 9. Parquet Results

Keep the original metadata columns and append new result columns, including:

- `model_decision` and `model_recommended_decision`
- the five scores and their total
- African focus, country codes, relevance tags, and evidence type
- evidence excerpt and concise reason
- model, prompt, and rubric versions
- latency, token counts, retry count, and error

Save every processed result, including `exclude` and `review`. Maintain a deterministic processed-ID checkpoint and preserve raw model output separately for debugging.

Production writes:

- `results.jsonl`, `results.parquet`, and `results.csv`
- `selected_papers.parquet` and `selected_papers.csv`
- `run_summary.json` and `run_summary.csv`

## 10. Later PDF Phase

No licence filtering or PDF downloading is part of the current classification run.

When downloading begins:

- attempt the 6,000 selected papers
- validate the response and PDF
- record failures and rights information
- if a selected PDF cannot be obtained, resume the remaining citation-first carry-over queue only far enough to select a replacement

For storage, package accepted PDFs into uncompressed WebDataset `.tar` shards:

- target about 1 GB per shard
- maximum 750 PDFs per shard
- close a shard when either limit is reached

Maintain a Parquet manifest with the MUFASA ID, source, licence, shard/member name, byte size, checksum, and processing status. Keep raw PDF shards separate from derived text shards.

## 11. Immediate Next Steps

1. Upload the confirmed gold Parquet and notebook to Kaggle.
2. Prepare the slim citation-first candidate Parquet.
3. Benchmark the three models.
4. Select the winning model.
5. Enable the guarded 6,000-paper production cell.
