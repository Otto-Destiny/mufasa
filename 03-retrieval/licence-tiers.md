# Licence tiers and coverage records

**What this decides:** how MUFASA represents a paper whose licence does not allow
us to reproduce its text, without pretending the paper does not exist.

**Status:** design decision, not yet built. Affects the `Paper` node and adds one
new node type. Referenced from [retrieval-architecture.md](./retrieval-architecture.md).

> Not legal advice. The principle below is well settled, but the specific
> questions flagged under *Open questions* should be checked before the graph is
> distributed outside the team.

---

## The problem

Papers fall into three groups, and only the first is straightforward.

1. Open licences — we may quote them.
2. Training permitted, redistribution restricted — we may learn from them, but
   we may not ship their text inside our database.
3. No training, no derivatives — we may do neither.

Dropping groups 2 and 3 would be the safe move, and it would be wrong. It
creates a false-absence gap: MUFASA would imply no research exists in an area
where a great deal does, because the research it can see is only the research it
is allowed to quote. A user cannot tell the difference between *"nobody has
studied this"* and *"we are not allowed to tell you."* That gap is worse than the
licensing problem it avoids.

## The principle

**Copyright protects expression, not facts.** A scientific finding is a fact
about the world. The sentences the authors wrote to describe it are protected;
the fact is not. US law states this directly — 17 USC §102(b) excludes any
"discovery" from protection — and Berne signatories including Nigeria follow the
same idea/expression split.

This is not a loophole. It is the legal basis for abstracting and indexing
services, for every systematic review that extracts results from paywalled
papers, and for every textbook. We would be doing the same thing at a different
scale.

## The three tiers

Every `Paper` node carries a `licence_tier`. Unknown licences default to **3**.

| Tier | Licence | What the graph stores |
|---|---|---|
| **1** | CC-BY, CC0, open access | Verbatim `EvidenceSpan` — exact quote, page, section |
| **2** | Training allowed, redistribution restricted | Paraphrased claim, full citation, DOI. `EvidenceSpan` carries no source text |
| **3** | No training or no derivatives | Metadata and a coverage record only — no findings at all |

Tier 3 is the conservative default because getting it wrong in that direction
costs a user one click to the DOI. Getting it wrong the other way ships text we
had no right to ship.

## Coverage records

**A Tier 3 record states the research question, not the answer.**

"Cadmium was measured in beef from Hadejia-Nguru" describes what was done.
"Cadmium exceeded WHO limits by three times" is the finding. The first is a fact
about a study's design and scope. The second is the product of the authors' work.

Coverage nodes use the prefix **`COV-`**, not `OBS-`. Different prefix, different
node type, enforced by schema — so a finding cannot later be dropped into one by
accident.

Fields kept: `studied`, `materials`, `organism`, `agent`, `properties_measured`,
`method`, `place`, `conditions`, `paper_id`, `doi`, `year`, `institutions`.

Fields removed: `direction`, `value`, `unit`, `baseline`, `quote`, `page`,
`section` — every field that carries the answer.

### Worked examples

All four are real papers from the classified corpus.

```json
{
  "id": "COV-0001",
  "paper_id": "P-4471",
  "studied": "genotoxicity and oxidative stress in a catfish exposed to cadmium",
  "organism": "Heterobranchus bidorsalis",
  "agent": "cadmium",
  "properties_measured": ["micronucleus frequency", "oxidative stress markers"],
  "method": "laboratory exposure assay",
  "place": null,
  "doi": "10.21608/ejabf.2025.430601.6824",
  "year": 2025,
  "institutions": ["University of Ibadan",
                   "National Centre for Genetic Resources and Biotechnology"],
  "findings_available": false,
  "licence_tier": 3
}
```

```json
{
  "id": "COV-0002",
  "paper_id": "P-4512",
  "studied": "optimum conditions for removing zinc from contaminated dumpsite soil",
  "agent": "saponin-based soil washing",
  "target": "zinc",
  "properties_measured": ["removal efficiency"],
  "method": "soil washing across multiple soils",
  "place": "Onyeama dumpsite, Enugu State, Nigeria",
  "doi": "10.38035/gijes.v3i2.545",
  "findings_available": false,
  "licence_tier": 3
}
```

```json
{
  "id": "COV-0003",
  "paper_id": "P-3390",
  "studied": "strength of a soil-based concrete with two agricultural ash additions",
  "materials": ["Biu soil", "rice husk ash", "soybean hull ash"],
  "properties_measured": ["compressive strength", "other strength properties"],
  "method": "laboratory specimen testing",
  "place": "Biu, Borno State, Nigeria",
  "doi": "10.70382/hujaeed.v9i4.008",
  "findings_available": false,
  "licence_tier": 3
}
```

```json
{
  "id": "COV-0004",
  "paper_id": "P-5108",
  "studied": "ambient air quality in crude-oil producing communities",
  "properties_measured": ["ambient air quality indices"],
  "method": "field measurement",
  "place": "five communities in Ogbia, Bayelsa State, Nigeria",
  "doi": "10.1007/s44292-026-00073-x",
  "findings_available": false,
  "licence_tier": 3
}
```

### How it renders

> **Rice husk ash in concrete — 3 studies found, 1 with extractable evidence**
>
> ✔ Yusuf et al. (2024) measured 31.2 MPa at 28 days for a 10% RHA mix, against
> an OPC control. *[quote, p.8]*
>
> ○ A 2025 study tested Biu soil concrete with rice husk ash and soybean hull ash
> for strength properties. **Findings not available in MUFASA** — licence
> restricts reuse. → doi.org/10.70382/hujaeed.v9i4.008

The user learns the work exists, what it covers, and where to read it. No finding
is reproduced.

## The phrasing rule

| Safe — describes the study | Crosses the line — states the result |
|---|---|
| measured cadmium genotoxicity in *H. bidorsalis* | found cadmium caused significant DNA damage |
| tested saponin washing for zinc removal at Onyeama | achieved 78% zinc removal at pH 5 |
| assessed ambient air quality in five Ogbia communities | air quality was "Marginally Polluted" in all five |
| compared microscopy, RDT and PCR for malaria diagnosis | PCR outperformed microscopy by 22% |

**The verb is the tell.** *Measured, tested, assessed, compared, characterised,
surveyed* describe method. *Found, showed, achieved, exceeded, improved,
demonstrated* state findings. Constrain the generator to the first set.

## What has to change

| Where | Change |
|---|---|
| `Paper` node | Add `licence_tier`, default 3 |
| Graph schema | Add the `COV-` node type, distinct from `OBS-` |
| Graph build | Strip `model_evidence` and `abstract` for any paper above Tier 1 |
| Retrieval | Coverage nodes match and rank normally; only quoting is withheld |
| Display | Distinguish "no evidence retrieved" from "findings withheld by licence" |

**Add `licence_tier` before building 6,000 papers.** Retrofitting it means
re-deriving every `EvidenceSpan`.

### The classification pipeline already carries verbatim text

`model_evidence` holds a **5–20 word quote copied verbatim from the title or
abstract** — the rubric requires exactly that, and it is what makes the field
trustworthy at Tier 1. For a Tier 3 paper it is precisely the thing that must not
ship. Keep it in the working store, since it is needed to verify that `studied`
is accurate, and strip it at graph-build time. Same for `abstract`. One filter at
build time, not a decision made per paper later.

## The risk that is larger than copyright

If a model writes the Tier 2 paraphrase, it can misstate what a paper found — and
that attributes a fabricated claim to a named researcher at a named institution.
Unlike a licensing question, that has an identifiable victim.

Mitigate it structurally, not by care: generate the paraphrase from the abstract,
keep the verbatim span internally for verification even when it is not shipped,
and treat any Tier 2 claim that cannot be checked against source text as
unusable. `EvidenceSpan` already provides the hook — this adds a
`verified_against` flag, not a new concept.

## Open questions

- **The corpus already stores abstracts verbatim.** Every batch parquet has a
  full `abstract` column. Fine for internal processing; a different question once
  it ships inside a packaged database. Check the provenance and terms of the
  OpenAlex extract before distribution. This is a nearer-term exposure than the
  full-text question.
- **EU database right.** The *sui generis* right protects investment in compiling
  a database independently of whether the contents are copyrightable. Systematic
  extraction of a substantial part of one publisher's corpus is the risk shape;
  spreading across many sources is not.
- **Contract can bind where copyright does not.** A paper obtained under
  subscription or TDM terms restricting derived data is governed by those terms
  regardless of the fact/expression line.
- **Tier assignment itself.** Nothing currently determines a paper's tier.
  Crossref and Unpaywall both expose licence metadata; that lookup is unbuilt.

## Why this is worth the effort

A coverage node still participates in retrieval. It matches on material,
organism, place and property; it appears in coverage counts; it walks the graph
like any other node. The only thing it cannot do is supply a quote — and the
system is already designed to say so honestly rather than invent one.

This turns a licensing constraint into a visible feature. MUFASA distinguishes
*"nobody has studied this"* from *"three people have, and here is where to read
them."* Most retrieval systems cannot tell those apart.

---

Related: [retrieval-architecture.md](./retrieval-architecture.md) —
*The data model*, *No PDFs ship*, *Coverage, not novelty*.
