# Entity canonicalisation

> **⚠ Provisional — guidance, not specification.** Use the direction and the reasoning. Treat the concrete specifics — schema, queries, database choice, exact node and field names — as illustrative only: they are one possible shape, not a decision, and they might change.

How entity mentions become shared nodes, so that papers connect.

**Status:** design, not built. Belongs to build-plane step 5, *"Normalise — units
to SI, names to canonical entities"*, in
[retrieval-architecture.md](./retrieval-architecture.md). Related:
[licence-tiers.md](./licence-tiers.md).

---

## 1. The problem, measured

Taken from the ten-paper sample in `milestone1-test-data/claims.jsonl` — 112
claims, the output of extraction with no normalisation applied.

| | |
|---|---|
| distinct predicates | **112 for 112 claims — zero reuse** |
| entity mentions | 273 |
| distinct entity names | 194 |
| **names appearing in more than one paper** | **1** (`annual rainfall`) |

One cross-paper link in the entire sample. The near-misses show why:

```
"groundwater" appears in 12 distinct entity names:
    groundwater · groundwater quality · groundwater table
    deep borehole groundwater · groundwater contamination vulnerability
    groundwater quality around Ilokun dumpsite ...

"leachate" appears in 6:
    leachate · leachate accumulation · migrated leachate
    Ilokun leachate plume · leachate seepage pathway ...
```

Twelve unconnected nodes where there should be one. As extracted, the graph has
almost no cross-paper edges — which means the eight `multi_paper_graph`
benchmark questions cannot be answered by walking entities.

**This is expected.** The sample is raw extraction. Canonicalisation is a
separate pass that has not been written.

## 2. What actually explodes: compounds, not scale

Distinct *atomic* vocabulary grows sublinearly (Heaps' law). Compound phrases do
not — they are the cross-product of their parts, and have no ceiling.

**57% of the sample's entity names are three words or more:**

| words in name | names |
|---|---|
| 1 | 27 |
| 2 | 56 |
| 3 | 53 |
| 4 | 29 |
| 5 | 15 |
| 6+ | 14 |

Real examples of the failure:

```
Warri rainfall  ·  Warri monthly rainfall  ·  Warri rainfall time series
Akwa Ibom International Airport rainfall records
groundwater quality around Ilokun dumpsite
Soje, Makera, Kpakungu, Bosso, Keteren Gwari, Farm Centre, Mandela Road,
    Shango and Chanchaga        <- nine places in one entity name
```

Any projection of vocabulary size that assumes atomic entities is answering the
wrong question while extraction emits phrases like these.

### Rule 1 — decompose at extraction

The extractor returns the atom and its modifiers in **separate fields**:

```json
"Warri rainfall time series"
  →  { "entity": "rainfall", "type": "EnvironmentalMeasure",
       "place": "Warri",
       "qualifiers": ["time series"] }
```

All three Warri variants now collapse to two nodes plus differing qualifiers.
The nine-town string becomes nine `:Place` nodes instead of one useless one.

Enforced by the validator, not by hope: **reject any entity name over three
words or containing a conjunction**, the same way `mufasa_domain` is already
rejected when it is not one of seven values.

With this in place the vocabulary is bounded, because the atomic sets are
bounded — there are only so many species, chemicals, Nigerian towns and
measurable properties. Without it, no index size helps.

## 3. Half of the entities never need canonicalising

From the sample, **52% of mentions are things in the world; 48% are descriptive
context**:

| joins across papers | never joins |
|---|---|
| `Place` 32 · `WaterSource` 15 | `RiskFactor` "inadequate waste management" |
| `Population` 14 · `Organism` 10 | `HealthOutcome` "health risk to groundwater users" |
| `Pesticide` 9 · `Chemical` 5 | `ExposureCondition` "long-term contaminant loading" |
| `Contaminant` 5 | `OperationalCondition` "high abstraction rates" |

Nobody will ask a question requiring `inadequate waste management` in one paper
to join the same phrase in another. Those belong as **properties on the
Observation**, not as nodes.

### Rule 2 — a fixed type list, split in two

Collapse the 83 types that emerged freely to roughly 25, and mark each as
`joins` or `describes`. Only `joins` types become nodes and enter the canonical
index. Decided once, in the prompt and the validator.

## 4. You do not invent canonical names — you link to authorities

The synonym sets already exist. **All ship as bulk files, not just APIs**, so
they are downloaded once at build time and never touched at runtime.

| domain | authority | bulk file | size |
|---|---|---|---|
| organisms | NCBI Taxonomy | `taxdump.tar.gz` (`names.dmp` carries synonyms and common names) | **73 MB** |
| places | GeoNames | `allCountries.zip` + `alternateNamesV2.zip` | 396 MB + ~100 MB |
| chemicals | ChEBI, PubChem | OBO / SDF | tens of MB |
| biomedical | MeSH | XML from NLM | ~30k descriptors |
| agriculture | AGROVOC (FAO) | SKOS/RDF | ~40k concepts |
| units | UCUM | one XML file | small |

Canonicalisation becomes **lookup**, not invention. `Vernonia amygdalina`
resolves against NCBI, which already knows its synonyms.

### Rule 3 — ship the subset, not the authority

What ships is a compiled index of the entities the corpus actually touched:

```
canonical_entities.parquet
  canonical_id    ENT-004182
  canonical_name  Vernonia amygdalina
  type            Organism
  authority       NCBI:taxid 199217
  synonyms        ["bitter leaf","onugbu","ewuro","Gymnanthemum amygdalinum"]
  embedding       float32[384]
  confidence      0.94
```

For ~17,000 papers: roughly 6,000-8,000 canonical entities, embeddings at 1.5 KB
each — **about 15 MB**. Against the 7 GB ceiling that is nothing, and it is a
file, so **zero network calls at runtime**.

Do not ship a wider slice "just in case". Africa-wide GeoNames alone would be
100-200 MB competing with the model, to serve a case that degrades gracefully.

## 5. Matching a mention to a canonical entity

Vector search is a **recall** device — it narrows thousands of candidates to ten.
Precision comes from the decision after it. Same shape as blocking-then-scoring
in `study-families.ipynb`.

```
mention "African catfish", type :Organism
  ↓  filter the index to :Organism only            8,000 → 900
  ↓  embed, cosine similarity, take top 10           900 → 10
  ↓  decide
```

| condition | outcome |
|---|---|
| top-1 ≥ 0.92 **and** ≥ 0.10 clear of top-2 | auto-match |
| top-1 ≥ 0.92, top-2 within 0.10 | ambiguous → string similarity, then LLM, then review queue |
| 0.75 ≤ top-1 < 0.92 | LLM adjudicates: "one of these, or new?" |
| top-1 < 0.75 | create a new canonical entity |

**The type filter does most of the work.** Because the mention is typed
`:Organism`, `rice husk ash` can never be compared with `rice` the crop. That
single constraint removes most of the false positives embeddings are feared for.

### Rule 4 — match incrementally, during extraction

Not: extract 100,000 free names, then merge. That is quadratic and brutal to
retrofit.

Instead maintain a **growing canonical index**, and match as you go. This is
O(n) — one retrieval per entity — and it gets cheaper as it runs: by paper 5,000
almost every entity already exists, so the model confirms rather than creates.

The vector index is therefore not only a query-time doorway; it is what makes
canonicalisation affordable at build time.

## 6. The vernacular alias list

**It will run to thousands, not hundreds.** Nigeria alone has 500+ languages,
the medicinal-plant literature covers thousands of species, and five or more
vernacular names per species is normal. An earlier estimate of "a few hundred
lines" in this repository was wrong.

But they are **harvested, not authored**. Measured across the 27,011 included
abstracts:

| pattern | abstracts | share |
|---|---|---|
| "locally known as" | 4 | 0.01% |
| "commonly known as" | 47 | 0.17% |
| "also known as" | 65 | 0.24% |
| "in Yoruba / Igbo / Hausa" | 40 | 0.15% |
| **binomial followed by a parenthesis** | **23,068** | **85.4%** |

Pairs the corpus already hands over:

> "*Hibiscus sabdariffa* Linn., **locally known as "zobo" in Nigeria**"
> "*Dioscorea alata* (**commonly known as Agaabidjan/Florido**)"
> "packaged sachet water, **commonly known as "pure water" in Nigeria**"

From abstracts this yields sparsely — roughly 90 explicit pairs from 27,000
papers. **Full text is where the density is**: ethnobotanical papers list
vernacular names in tables, and one Nigerian medicinal-plant survey can
contribute 50-200 pairs from a single table. Seed from PROTA (Plant Resources of
Tropical Africa) and the African Plant Database.

### Why this one does not block extraction

The canonical index must exist **before** extraction, because retrofitting
100,000 free-text names is brutal.

The alias list is the opposite — purely additive, and it **degrades
gracefully**. A missing alias means someone searching "onugbu" does not find the
*Vernonia amygdalina* papers. It does not corrupt the graph, break a join, or
invent a false replication. Add it next month and those papers become findable,
with no re-extraction.

It will never be complete. It does not need to be.

## 7. Offline: what happens when a user ingests their own paper

The planned ingestion layer must make **zero network calls**, per the
competition rules. It can:

1. **Mention matches an existing canonical entity** — the common case, since
   17,000 papers already cover the domain vocabulary. Joins the graph properly.
2. **Matches nothing** — a new canonical entity is created from the surface
   form, with `authority: null`. It still becomes a node, still joins with the
   next paper that mentions the same thing, still retrievable. It is simply not
   linked to NCBI until the index is refreshed.
3. **Ambiguous** — goes to a review queue in the UI.

Only the authority link is lost offline, and it is recoverable on the next
corpus update.

**Runtime cost:** the embedding model must ship (~90 MB for
`all-MiniLM-L6-v2`), but it is already required for the vector retrieval
channel, so canonicalisation free-rides. The "one of these, or new?" judgement
can use MUFASA itself, or be skipped entirely in favour of vector + string
similarity with a conservative threshold and a review queue — defensible, given
that over-merging invents agreement while under-merging only costs a
corroboration.

## 8. What was taken from Semantica

[`semantica-agi/semantica`](https://github.com/semantica-agi/semantica) — MIT,
Python, created 2025-06-25, 2,510 stars, 318 forks, actively developed. Reviewed
at source rather than from its README.

**Not adopted.** Its default blocking is `blocks.setdefault(name[0], ...)` —
first character, 26 buckets for the whole corpus, weaker than the rare-token
index already in `study-families.ipynb`. Its `SimilarityCalculator` defaults to
`embedding_weight=0.0`, and `calculate_embedding_similarity` takes vectors as
arguments rather than computing them. Its `incremental_detect` is a nested loop
over all existing entities despite a docstring claiming otherwise. Most
importantly, **it has no ontology linking at all** — pointed at this corpus it
would keep `Vernonia amygdalina`, `bitter leaf`, `onugbu` and `ewuro` as four
separate nodes forever.

**Three ideas worth taking**, roughly forty lines and no dependency:

1. **Merge provenance** — `EntityMerger(preserve_provenance=True)` keeps the
   record of what merged into what and why, inside the graph. Study-family dedup
   currently picks a winner and drops the losers into a CSV.
2. **Typed conflict handling** — value, type, relationship, temporal and logical
   conflicts detected explicitly, resolved by a named strategy
   (`credibility_weighted`, `most_recent`, `voting`) rather than last-write-wins,
   with source credibility accruing over time.
3. **Confidence on the resolution itself** — `NormalizedEntity(canonical=...,
   confidence=0.91)`. The schema has `extraction_confidence` on the Observation
   but nothing recording how sure we are that two mentions are the same entity.

## 9. Build order

1. **Fix the type list** — 83 down to ~25, split `joins` / `describes`.
2. **Add the decomposition rule** to the extraction schema and validator.
3. **Compile the canonical index** from the authority bulk files, scoped to the
   corpus.
4. **Wire incremental matching** into extraction.
5. **Harvest aliases** from full text as extraction runs; seed from PROTA.

Steps 1 and 2 are structural and cheap now. Step 3 depends on knowing the
flagship domain. Step 5 never finishes and does not need to.

## 10. Open questions

- **Predicates.** 112 distinct for 112 claims makes them useless for querying.
  Either a controlled list of ~40 relation types, or accept they are
  documentation rather than structure. **This cannot be settled until the data
  model is chosen** — in a triple store the predicates *are* the structure; in a
  design that reifies measurements they are largely redundant. Any earlier
  argument in this repository that dismissed them by pointing at a schema was
  circular: no schema has been written.
- **Which authority wins** when NCBI and AGROVOC disagree on a plant name.
- **Whether `WaterSample`-style types join or describe** — they were counted as
  joining above, but a per-study sample is arguably study-specific and should be
  a property.
- **Refresh policy** for the shipped index once users have created local
  entities against it.
