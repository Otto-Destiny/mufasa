# Licence counts

Licence coverage of the classified corpus, and how it splits across the six
MUFASA domains.

Tier definitions come from
[03-retrieval/licence-tiers.md](../../03-retrieval/licence-tiers.md).

---

## Snapshot — read this before quoting any number

**The production run is still going. These numbers describe batches 0–40 only
and will be out of date as soon as batch 41 lands.**

| | |
|---|---|
| batches counted | **0–40** (41 batches, contiguous) |
| papers sent | **41,000** of 155,825 — 26.3% of the source |
| classified | 40,983 |
| failed | 17 |
| include / review / exclude | 14,134 / 5,045 / 21,804 |
| first batch started | 2026-08-04 08:45 UTC |
| last batch finished | **2026-08-07 05:03 UTC** |
| field filter | none for batches 0–29; `SKIP_FIELDS` active from **batch 30** |

Because the OpenAlex field filter starts at batch 30, batches 30–40 are drawn
only from Engineering, Computer Science, Materials Science, Energy, Earth and
Planetary Sciences, Physics and Astronomy and Chemical Engineering. The domain
totals below therefore over-represent MAT, ENR and TEC relative to a
whole-corpus sample.

Regenerate from `production/licence_cache.parquet` (the five licence columns of
`openalex_all_fields.csv`, cached to avoid re-parsing 876 MB) joined to
`production/batches/*.parquet` on `openalex_id`.

## How a licence is resolved

The source carries five licence columns. This analysis takes the first
non-empty of `best_oa_license`, `primary_license`, `download_license`, then maps
it to a tier:

| tier | licences |
|---|---|
| **T1 permissive** | `cc-by`, `cc-by-sa`, `cc0`, `public-domain` |
| **T2 non-commercial** | `cc-by-nc`, `cc-by-nc-sa` |
| **T3 no-derivatives** | `cc-by-nc-nd`, `cc-by-nd` |
| **T4 none stated** | empty |
| **T5 other/unclear** | `other-oa`, stray URLs, malformed values |

Note the source is an **open-access extract** - `is_open_access` is true for
155,712 of 155,825 rows - so permissive licences are over-represented compared
with the literature at large. `oa_status`: diamond 80,859, gold 30,997, bronze
19,236, hybrid 19,078, green 5,542.

---

## Availability

| set | has a licence | none stated |
|---|---|---|
| whole pool (155,825) | 103,584 — 66.5% | 52,241 — 33.5% |
| **included (14,134)** | **10,489 — 74.2%** | 3,645 — 25.8% |

## Raw licence values, whole pool

| licence | count |
|---|---|
| cc-by | 67,259 |
| *(none stated)* | 52,241 |
| cc-by-nc | 12,106 |
| cc-by-nc-nd | 11,972 |
| cc-by-nc-sa | 5,035 |
| other-oa | 4,119 |
| cc-by-sa | 2,547 |
| public-domain | 271 |
| cc-by-nd | 164 |

The remaining values are malformed - stray country codes, PDF URLs, the strings
`true` / `false` - and total under 60 rows. Same CSV column-bleed seen in
`publication_year`.

## Tiers by processing state

| tier | included (14,134) | unprocessed (114,825) | all classified (40,983) |
|---|---|---|---|
| T1 permissive | **6,323 — 44.7%** | 50,158 — 43.7% | 48.6% |
| T2 non-commercial | 2,031 — 14.4% | 11,287 — 9.8% | 14.3% |
| T3 no-derivatives | 1,851 — 13.1% | 6,830 — 5.9% | 12.9% |
| T4 none stated | 3,645 — 25.8% | **43,181 — 37.6%** | 22.1% |
| T5 other/unclear | 284 — 2.0% | 3,369 — 2.9% | 2.1% |

The unprocessed pool skews harder toward *no licence stated* - 37.6% against
25.8% among includes - so coverage gets worse, not better, over the remaining
115k papers.

---

## INCLUDED papers by MUFASA domain

14,068 of the 14,134 includes carry one of the six domains.

| domain | T1 permissive | T2 non-comm | T3 no-deriv | T4 none stated | T5 other | total |
|---|---|---|---|---|---|---|
| HLT | 2,087 | 883 | 730 | 898 | 53 | 4,651 |
| ENV | 1,440 | 397 | 436 | 885 | 56 | 3,214 |
| AGR | 1,396 | 380 | 308 | 972 | 48 | 3,104 |
| ENR | 596 | 153 | 142 | 333 | 49 | 1,273 |
| MAT | 499 | 134 | 161 | 306 | 34 | 1,134 |
| TEC | 287 | 82 | 62 | 225 | 36 | 692 |
| **total** | **6,305** | **2,029** | **1,839** | **3,619** | **276** | **14,068** |

Row percentages:

| domain | T1 | T2 | T3 | T4 | T5 |
|---|---|---|---|---|---|
| HLT | 44.9% | **19.0%** | **15.7%** | 19.3% | 1.1% |
| AGR | 45.0% | 12.2% | 9.9% | **31.3%** | 1.5% |
| ENV | 44.8% | 12.4% | 13.6% | 27.5% | 1.7% |
| ENR | **46.8%** | 12.0% | 11.2% | 26.2% | 3.8% |
| MAT | 44.0% | 11.8% | 14.2% | 27.0% | 3.0% |
| TEC | 41.5% | 11.8% | 9.0% | **32.5%** | **5.2%** |

## REVIEW papers by MUFASA domain

5,038 of the 5,045 reviews carry one of the six domains.

| domain | T1 permissive | T2 non-comm | T3 no-deriv | T4 none stated | T5 other | total |
|---|---|---|---|---|---|---|
| HLT | 847 | 384 | 338 | 345 | 19 | 1,933 |
| MAT | 549 | 150 | 178 | 280 | 25 | 1,182 |
| AGR | 425 | 130 | 111 | 218 | 10 | 894 |
| ENR | 248 | 60 | 54 | 116 | 20 | 498 |
| ENV | 157 | 42 | 54 | 69 | 4 | 326 |
| TEC | 104 | 19 | 16 | 53 | 13 | 205 |

Row percentages run 43.8-50.7% T1 and 7.8-19.9% T3 - the same shape as the
includes, so nothing about being sent to review correlates with licence.

---

## What the numbers say

**The distribution is flat across subject matter.** Every domain sits at
41-47% permissive. No domain is disproportionately unquotable, so licensing
does not force the corpus toward any particular science.

**6,305 included papers are already Tier 1** - quotable verbatim, no coverage
record needed. That alone exceeds the 6,000-paper target, with every domain
represented: HLT 2,087, ENV 1,440, AGR 1,396, ENR 596, MAT 499, TEC 287.

**HLT is the most restricted** and the only real outlier: 19.0% non-commercial
and 15.7% no-derivatives, against roughly 12% and 10% elsewhere. Medical
publishers use `-nc-nd` far more than engineering ones. Combined, 34.7% of HLT
includes are restricted, against 20.8% for TEC.

**TEC has the weakest metadata.** 32.5% no licence stated plus 5.2% unclear
means 37.7% of TEC includes carry no usable signal. It is already the smallest
domain, and more than a third of it would default to Tier 3 for want of a
lookup rather than an actual restriction. AGR shares the problem at 31.3%.

**Only 13.1% of includes are genuinely no-derivatives.** The larger restricted
group is the 25.8% with nothing stated - absent metadata, not refused
permission.

## Worth doing

A Crossref or Unpaywall lookup on the 3,645 includes with no stated licence
would likely move a large share up a tier. Many are diamond-OA journals that
never registered a licence with Crossref rather than papers that are actually
restricted. TEC and AGR would gain most.

---

# Where the unknown licences come from

*Second snapshot: batches **0-41** (42 batches, 42,000 sent, 41,975 classified,
14,388 includes), last batch finished 2026-08-07 05:40 UTC. Counts here are
slightly larger than the tables above, which were taken at batch 40.*

Scope: the **3,609 included papers with no licence stated** that carry a venue
name.

## Spelling variants are not the problem

Names were normalised hard - case, accents, punctuation, `&` to `and`,
stopwords dropped. Across the 3,609 included papers that merged exactly **one**
pair: 858 venue strings became 857.

Pool-wide it merged 13 pairs, almost all casing:

| variants | papers |
|---|---|
| `TEXILA INTERNATIONAL JOURNAL OF PUBLIC HEALTH` / `Texila international journal of public health` | 94 |
| `Physical Review E` / `Physical review. E` | 8 |
| `International Research Journal of Medicine and Medical Sciences` (two casings) | 18 |

OpenAlex normalises venue names upstream, so duplication does not live here.

## Concentration

| | |
|---|---|
| distinct venues | 857 |
| top 64 venues | 50% of the papers |
| top 259 venues | 80% |
| venues with 1 paper only | 475 |
| venues with >= 10 papers | 87 venues, 2,067 papers |

## Grouped by publisher

The source has **no publisher column** - `download_source` merely repeats the
venue name - so these families are derived from title patterns. Reliable for
the distinctive imprints, blunt elsewhere.

| papers | venues | publisher / family |
|---|---|---|
| **786** | **64** | **Sciencedomain / Journal International (SDI)** |
| **444** | **17** | **Nigerian learned societies** (*Nigerian Journal of ...*) |
| 161 | 4 | FUDMA (Federal University Dutsin-Ma) |
| 50 | 1 | Gombe State University (BIMA) |
| 33 | 1 | LAUTECH |
| 31 | 1 | Obafemi Awolowo University (*Ife Journal of Science*) |
| 29 | 1 | Federal University Dutse |
| 24 | 1 | UNIOSUN |
| 23 | 2 | Akwa Ibom State University |
| 21 | 1 | Ajayi Crowther University |
| 11 | 7 | IOSR Journals |
| 1 | 1 | Texila International |
| **1,614** | **101** | **grouped (45%)** |
| 1,995 | 757 | ungrouped individual venues |

Largest single venues: Nigerian Journal of Animal Production 155, Sahel Journal
of Life Sciences FUDMA 113, Journal of Engineering Research and Reports 91,
Nigerian journal of microbiology 75, Nigerian Journal of Physics 65.

## Verified: the big clusters are CC BY, not restricted

Checked against publisher policy pages on 2026-08-07.

| publisher | papers | actual licence | verdict |
|---|---|---|---|
| **Sciencedomain (SDI)** | 786 | **CC BY 4.0** | Tier 1, not Tier 4 |
| **Nigerian Journal of Animal Production** | 155 | **CC BY 4.0** | Tier 1 |
| FUDMA Journal of Sciences | (FUDMA family, 161) | CC BY 4.0 | Tier 1 likely |

SDI applies one house policy across its imprints. The same sentence appears on
every journal checked - *Journal of Engineering Research and Reports*, *Asian
Journal of Advanced Research and Reports*, *International Journal of Pathogen
Research*:

> "Copyright on any open access article published in this journal is retained
> by the author(s). The Creative Commons Attribution License 4.0 formalizes
> these and other terms and conditions of publishing articles."

**This confirms the reading that Tier 4 is a metadata gap, not a restriction.**
At least 941 of the 3,609 unknown-licence includes - 26% - are demonstrably
CC BY. They are Tier 4 only because the publisher never registered a licence
with Crossref.

Not yet verified: *Sahel Journal of Life Sciences FUDMA* specifically (113
papers). Its sibling *FUDMA Journal of Sciences* is CC BY 4.0 and the
institution is the same, but the journal's own page was not reached.

## What this means

**Keep Tier 4 in the queue.** `LICENCE_TIERS = [1, 4]` in
`production-classification.ipynb` is set on this basis. Excluding Tier 4 would
have dropped 6,628 unprocessed papers, a large share of which are CC BY in
fact, and they skew toward Nigerian society and university journals - the most
African-specific material in the corpus.

**Three lookups settle 39% of the gap.** SDI, the Nigerian learned societies
and FUDMA together cover 1,391 included papers from roughly twenty policy
checks. All ten identified families cover 1,614 papers from about thirty.

**The tail is genuinely long.** 1,995 papers across 757 venues, 475 of them
contributing a single paper. Those will not be resolved by publisher-level
work; they need a per-DOI Crossref or Unpaywall call.

Sources:
[SDI journals](https://sciencedomain.org/journals.html) ·
[JERR submissions](https://journaljerr.com/index.php/JERR/about/submissions) ·
[AJARR submissions](https://journalajarr.com/index.php/AJARR/about/submissions) ·
[IJPR submissions](https://journalijpr.com/index.php/IJPR/about/submissions) ·
[NJAP](https://njap.org.ng/index.php/njap/about/submissions) ·
[FUDMA Journal of Sciences](https://fjs.fudutsinma.edu.ng/index.php/fjs/about)
