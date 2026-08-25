# Download failure analysis and recovery backlog

> **NOT AUTHORITATIVE — this file does not control any decision.**
> A working note kept for the measurements in it. Where it disagrees with the
> notebooks or the reviewed layer documents linked from `README.md`, those win.

**Status:** deferred. Parked deliberately to get a first trained model working on the
10,321 papers already parsed. Come back to this when more corpus is worth the time.

**Measured:** 11 August 2026, against the completed Kaggle run
(`mufasa_corpus/manifests/documents.parquet`, all 171 batches).

---

## 1. Where the corpus stands

| status | papers | share |
|---|---|---|
| **ok — parsed and usable** | **10,321** | 60.5% |
| download_failed | 6,577 | 38.6% |
| identity_review | 108 | 0.6% |
| needs_ocr | 42 | 0.2% |
| quality_review | 1 | — |
| **total queued** | **17,049** | |

Parsed corpus: 415.9 M characters of Markdown, roughly 114 M tokens.
Identity verified on every ok paper — 8,901 by DOI, 1,420 by title, none unmatched.
OCR repair touched 949 pages (Tesseract *was* present in the Kaggle image).

---

## 2. Why 6,577 downloads failed

| cause | count | share |
|---|---|---|
| **PermanentDownloadError** (server answered, not a PDF) | **3,058** | 46.5% |
| HTTP 403 | 2,776 | 42.2% |
| *(regex artifact — see note)* | 516 | 7.8% |
| HTTP 404 | 127 | 1.9% |
| ConnectionError, 405, 429, 500, timeout, 522, 503, 401 | ~100 | 1.5% |

> **Note on the 516.** These were labelled `HTTP 443` by the analysis script, which
> is not a status code — the regex matched `:443` inside URLs. Their real cause is
> unknown and worth re-deriving. `failure_class` in the notebook is unaffected: 443
> is not in `PERMANENT_HTTP`, so they classify as retryable.

### The largest bucket is not a block

`PermanentDownloadError: file does not contain a PDF header` means the server
**responded** — we received HTML rather than a PDF. That is one hop away from
success, not a refusal. It is the biggest single cause and the most recoverable.

---

## 3. Failure by host — the top 10 are 65% of all failures

| host | failed | total | rate |
|---|---|---|---|
| **ajol.info** | **1,165** | 1,626 | **72%** |
| mdpi.com | 897 | 897 | 100% |
| article.sciencepublishinggroup.com | 484 | 484 | 100% |
| downloads.hindawi.com | 457 | 457 | 100% |
| onlinelibrary.wiley.com | 361 | 361 | 100% |
| scirp.org | 297 | 593 | 50% |
| tandfonline.com | 219 | 219 | 100% |
| **pmc.ncbi.nlm.nih.gov** | **157** | 157 | 100% |
| academic.oup.com | 106 | 106 | 100% |
| doi.org | 101 | 333 | 30% |

---

## 4. What the failing servers actually return

Probed 24 URLs drawn from the `PermanentDownloadError` set, two per host:

| response | share |
|---|---|
| other HTML | 46% |
| **a working PDF, on plain retry** | **17%** |
| Cloudflare challenge | 17% |
| HTML containing a PDF link | 8% |
| CAPTCHA | 8% |
| 404 page | 4% |

```
ajol.info      PDF (works now!)   HTTP 200     158,880 B
nature.com     PDF (works now!)   HTTP 200   6,272,897 B
thelancet.com  Cloudflare         HTTP 403       5,601 B
cell.com       Cloudflare         HTTP 403       5,596 B
```

### The key inference

**Much of this is rate-limiting during the bulk run, not policy blocking.**

- 17% of failed URLs serve a good PDF on a plain retry with the same user agent
- AJOL fails at **72%, not 100%** — a publisher that blocks bots blocks all of them;
  one that is being hammered fails most of them
- We issued 1,626 requests to AJOL at 4 concurrent connections from one Kaggle IP

Bot-blocking is real but is roughly a quarter of this bucket, concentrated in
commercial publishers (Lancet, Cell, MDPI, Wiley, Taylor & Francis, OUP).

---

## 5. Recovery backlog, in priority order

### 5.1 AJOL — ~1,165 papers, best return available
African Journals Online is the largest host **and** the largest failure source, and
it is the literature this project exists to cover. Evidence points to rate-limiting.

- Retry AJOL **alone**, not mixed with other hosts
- `RETRY_PER_HOST_WORKERS = 1` (already set in `retry-download-and-parse.ipynb`)
- **Add a per-host cooldown** of 1–2 s between requests — the retry notebook does
  not have this yet, and one-at-a-time is still fast enough to trip a limiter
- Expect to recover most of them

### 5.2 PMC — 157 papers, near-certain
NCBI blocks scraping the web PDF but publishes an official OA service and an FTP
mirror. Documented endpoints, solved problem.

### 5.3 Follow the PDF link in returned HTML — ~250 papers
8% of the sample returned an HTML page containing a direct PDF link. Add a single
extra hop: if the response is HTML, parse it for a `.pdf` href and fetch that.

### 5.4 Investigate "other HTML" — ~1,400 papers, unknown value
The largest unclassified group. Sample twenty and read them. If they are repository
landing pages, 5.3 recovers them; if paywalls, they are lost. **This sample decides
whether a third of the losses are addressable** and is the cheapest way to find out.

### 5.5 Unpaywall for the commercial blocks — ~2,700 papers, partial
MDPI, Wiley, Taylor & Francis, OUP, Hindawi are hard blocks; no user-agent trick
helps (verified: a Chrome UA gets the same 403). Unpaywall's API returns every
known OA location for a DOI and would find repository copies for some fraction.

### 5.6 Not worth pursuing
Cloudflare challenges and CAPTCHAs, ~750 papers. Defeating them is neither
technically simple nor obviously appropriate. Record the DOIs as an honest account
of what the corpus does not contain.

---

## 6. Realistic ceiling

| action | papers |
|---|---|
| current | 10,321 |
| + AJOL | ~11,500 |
| + PMC | ~11,650 |
| + HTML second hop | ~11,900 |
| + a share of Unpaywall | 12,500–13,500 |

Roughly **13,000 papers** is the plausible target, against 17,049 queued.

---

## 7. Two smaller items, unrelated to downloads

**1,079 papers have `retraction_status: unknown_error`** — Crossref 429s from running
without `CROSSREF_MAILTO`. Those papers are in the ok set with **unverified retraction
status**, 6.3% of the corpus. Cheap to fix: re-query just those DOIs with a mailto
set. No re-downloading. Worth doing before the corpus is treated as final.

**108 papers sit in `identity_review`.** Their Markdown parsed fine but the title
match fell below `MIN_TITLE_TOKEN_OVERLAP = 0.70`. Repository PDFs with cover sheets
are a known cause. Worth eyeballing a handful before assuming the threshold is right —
this status is terminal and excludes a paper from extraction.

---

## 8. How to resume

1. Read §5.4 first — the "other HTML" sample is the cheapest high-information step
2. Add the per-host cooldown to `retry-download-and-parse.ipynb`
3. Run AJOL alone, measure the recovery rate, then decide about the rest
4. `retry-download-and-parse.ipynb` merges recovered papers into `documents.parquet`
   automatically and never touches batch state, so it is safe to run repeatedly
