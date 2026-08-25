# Data quality audit

Every column of the source checked against its expected shape, plus content
checks on the classified output. Measured, not sampled.

**Snapshot:** 155,825 source rows · 74,799 classified · 27,011 includes ·
38,432 excludes. Taken 2026-08-08.

Regenerate from `production/source_cache.parquet`, `production/batches/*.parquet`,
`production/download_cache.parquet` and `production/licence_cache.parquet`.

---

## Part 1 — the full inventory

### A. Source corruption: unescaped delimiters in the CSV

**113 rows (0.073%)** where a comma inside a field pushed every later column one
position right. This is a single defect with many symptoms, which is why it kept
surfacing as separate discoveries.

| symptom | rows |
|---|---|
| `publication_year` fails | 113 |
| `is_retracted` fails | 113 |
| `publication_date` fails | 113 |
| `work_type` fails | 104 |
| `keywords_json` unparseable | 103 |
| `cited_by_count` fails | 102 |
| `institutions_json` unparseable | 82 |
| `countries_json` unparseable | 70 |

The cause is visible in the data - a gene list containing commas:

```
title            : Preliminary genomic characterisation and antimicrobial resistance...
publication_year : " aac(6')"
work_type        : ' and fosM . Virulence genes'
cited_by_count   : ' avrA'
countries_json   : 'FALSE'
```

Exactly **1** row fails 1-2 checks. This is not a spectrum: 113 broken rows,
155,711 clean.

**22 of the 113 are in the include set.**

### B. Content quality of abstracts

Measured on 26,445 includes at the time of the abstract-quality run.

| flag | papers | share |
|---|---|---|
| reused - same text on two or more papers | 186 | 0.70% |
| too short - under 250 characters | 127 | 0.48% |
| placeholder - "No Abstract", "International audience" | 69 | 0.26% |
| title mismatch | 30 | 0.11% |
| author list in the abstract field | 24 | 0.09% |
| unreadable title | 4 | 0.02% |

216 unusable, 208 suspect. Pool-wide, 2,846 abstracts (1.83%) are under 250
characters.

Largest placeholder group: `"International audience"` on **48 papers**. Worst
blurb: a journal's own description on 4 unrelated papers.

### C. Encoding

The largest category by volume, across the whole pool:

| | rows | share |
|---|---|---|
| abstracts with HTML entities | **25,487** | **16.36%** |
| abstracts ending mid-sentence | 20,098 | 12.90% |
| titles with HTML entities | 5,031 | 3.23% |
| abstracts with markup tags | 3,399 | 2.18% |
| titles with markup tags | 2,538 | 1.63% |

The mid-sentence figure is an upper bound - some abstracts simply end without
terminal punctuation rather than being truncated.

### D. Identifiers

| check | result |
|---|---|
| duplicate `openalex_id` | **0** |
| missing DOI | 1,561 (1.00%) |
| duplicate DOI | 14 rows across 7 DOIs |
| titles that are a URL | 11 |
| titles more than 30% `?` | 14 |
| titles over 15% non-ASCII | 2 |

### E. A dead rubric rule

`is_retracted` is `FALSE` on 155,712 of 155,825 rows. **Not one `TRUE`.** The
remaining 113 are the bleed cluster, holding values like `EN`, `ARTICLE`, `2024`.

Hard exclusion 5 - *"Retracted: the work type says so"* - has never fired and
cannot fire. The field is not populated for this slice; retraction status would
have to come from Crossref instead.

### F. Pipeline artefacts

Ours, not the source's.

| | count |
|---|---|
| classified | 74,799 |
| failed, parked as `review` | 0 - the retry notebook cleared them |
| includes | 27,011 |
| includes labelled `OUTSIDE_TAXONOMY` | 71 |
| reviews labelled `OUTSIDE_TAXONOMY` | 9 |
| self-report papers manually overridden to exclude | 214 |
| duplicate study pairs merged | 136 |
| pairs left for human review | 55 |
| papers scored under the pre-correction rubric | 30,000 (batches 0-29) |

---

## Part 2 — what still matters

**The reframing:** abstracts are used only to decide which papers to download.
Claims are extracted from the downloaded full text. So abstract cleanliness is
not a corpus-quality problem - it is a *decision-quality* problem, and only
where it changed a decision.

That retires most of Part 1. What remains:

### 1. Decisions made on a broken abstract — the real residue

| | papers | why it matters |
|---|---|---|
| **includes** with an unusable abstract | **201** (0.74%) | downloaded on a call the classifier could not have made properly |
| **excludes** with an unusable abstract | **571** (1.49%) | possibly good papers dropped silently, and invisibly |

The 571 are the more serious number. A paper whose abstract read
`"International audience"` was excluded on no evidence, and nothing downstream
will ever reveal it. **Re-classifying those 571 from their full text is the one
recovery action this audit argues for.**

The 201 are self-correcting: extraction from full text will show whether the
paper belongs, and a bad one can be dropped then.

### 2. Titles — these do ship

Titles are not just classification inputs. They are the dedup key, they sit on
the `Paper` node, and they appear in every citation MUFASA renders.

Within the includes:

| | papers |
|---|---|
| HTML entities | 782 (2.90%) |
| markup tags | 474 (1.75%) |
| is a URL, not a title | 2 |
| mostly `?` | 4 |

Fixable by normalisation at graph-build time. The 6 URL/unreadable titles need
the real title fetched from Crossref.

### 3. The 22 bled includes

Their `publication_year`, `journal` and `work_type` are shifted, so the `Paper`
node would carry a wrong year and journal - visible in every citation drawn from
them. Small, and worth repairing from Crossref by DOI.

### 4. Retraction — matters more now, not less

Going to full text means extracting claims from whatever is downloaded. With
`is_retracted` dead, a retracted paper enters the corpus unnoticed and its
claims are indistinguishable from sound ones. A Crossref lookup at download time
closes this.

### 5. Study families — unchanged

Nothing about the reframing touches this. Duplicate studies still inflate
apparent replication, and still leak between train and test splits. 136 merges
found, 55 pairs pending review.

### 6. Fetchability — good news

| | includes |
|---|---|
| have a `download_url` | **27,011 (100%)** |
| have `best_oa_pdf_url` | 26,486 (98.06%) |
| have an OpenAlex cached PDF endpoint | 25,733 (95.27%) |
| have **no** PDF URL at all | **0** |
| have no DOI | 99 (0.37%) |

Every included paper has at least one route to a PDF. Note this measures
presence, not that the URL resolves - link rot across 1,777 distinct hosts is a
separate question the download run will answer.

---

## What this retires

Given claims come from full text, these need no action:

- 25,487 abstracts with HTML entities
- 20,098 abstracts ending mid-sentence
- 3,399 abstracts with markup
- 186 reused abstracts, except where they changed a decision
- the abstract-truncation question entirely

They mattered while the abstract was the evidence. It no longer is.

## Action list, in order

1. **Re-classify the 571 excludes with unusable abstracts** from full text or
   from Crossref abstracts. The only silent, unrecoverable loss in the audit.
2. **Normalise titles** at graph-build time - entities and markup, 1,256 papers.
3. **Repair the 22 bled includes** from Crossref by DOI.
4. **Add a Crossref retraction check** at download time; hard exclusion 5 cannot
   do it.
5. **Fetch real titles** for the 6 URL or unreadable ones.
6. **Finish the 55 pending family-review pairs** before freezing splits.
