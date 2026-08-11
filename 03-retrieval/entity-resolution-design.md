# Proposed MUFASA Entity Resolution Module

**Status:** draft proposal awaiting review and approval; no implementation exists yet. Do not implement from this document until it has been reviewed.

**Location of this specification:** `03-retrieval/entity-resolution-design.md`

This document consolidates the proposed corrections to `entity-canonicalisation.md`. It does not supersede that file yet, and neither document is an approved implementation contract. Any disagreement remains open until this proposal is reviewed.

## 1. Purpose

The entity-resolution module turns each extracted mention into a stable identity that MUFASA can reuse across papers and across its other layers.

In plain terms, it must answer two different questions:

1. **What shared scientific concept is this?** For example, `groundwater`, `nitrate`, `malaria`, or `Minna`.
2. **Is this a particular thing inside one study?** For example, one borehole sample, one experimental group, or one monitoring station.

The resolver must connect genuinely identical concepts without erasing scientifically meaningful differences. When the evidence is insufficient, it must abstain and send the mention to review. A correct unresolved result is safer than a false merge that invents agreement between papers.

## 2. Proposed architectural shape

Entity resolution is a reusable Python module, not notebook logic.

```text
MUFASA/
|-- scripts/
|   |-- __init__.py
|   `-- entity_resolution/
|       |-- __init__.py
|       |-- contracts.py
|       |-- normalization.py
|       |-- authorities.py
|       |-- matching.py
|       |-- registry.py
|       |-- pipeline.py
|       |-- io.py
|       |-- evaluation.py
|       |-- adapters/
|       |   |-- __init__.py
|       |   `-- mufasa.py
|       `-- policies/
|           `-- mufasa-v1.yaml
|-- tests/
|   `-- entity_resolution/
`-- 03-retrieval/
    |-- entity-resolution-design.md
    `-- entity-resolution-evaluation.ipynb
```

The future evaluation notebook will import and call this module. It will contain controls, charts, samples, and review views, but no matching rules or duplicated resolver code. Production workflows and other MUFASA layers will call the same public module.

The package will expose only three main operations:

- resolve a batch against a frozen registry;
- explicitly commit a validated resolution run;
- evaluate a run against reviewed examples.

The core must not contain hard-coded corpus paths. MUFASA-specific vocabularies, authority choices, qualifier rules, and calibrated thresholds belong in a versioned policy profile so another part of MUFASA can reuse the resolver with a different profile.

Each file has one job:

| File | Responsibility |
|---|---|
| `contracts.py` | Immutable records, enums, schemas, and invariants |
| `normalization.py` | Non-destructive creation of typed comparison keys |
| `authorities.py` | Read-only access to pinned authority snapshots |
| `matching.py` | Candidate generation, features, conflicts, and decisions |
| `registry.py` | Concepts, aliases, instances, IDs, and lineage |
| `pipeline.py` | Deterministic batch orchestration and public operations |
| `io.py` | Generic readers/writers, hashing, manifests, and atomic publication |
| `adapters/mufasa.py` | Corpus eligibility and conversion of MUFASA Parquets into core records |
| `evaluation.py` | Gold-set scoring, calibration reports, and regression checks |

`__init__.py` exposes only the small public surface; callers do not import internal matching helpers.

## 3. Position in the pipeline

```text
manifest-approved parsed papers
        -> candidate LLM extraction
        -> extraction pilot and approval gate
        -> validated atomic observations + unresolved entity mentions
        -> entity-resolution module
        -> concepts, study instances, aliases, and audited mention mappings
        -> GraphRAG build
```

Resolution happens **after extraction**, as a separate batch pass. The extraction model never chooses or invents canonical IDs.

The current `llm-claim-extraction.ipynb` is an unrun, unapproved prototype. Its tables and vocabularies are candidate contracts, not established facts or proven outputs. They become the resolver input contract only after the extraction design is reviewed and the staged extraction pilots in Section 6 pass. Until then, statements in this document describe intended behaviour rather than completed capability.

For the first corpus build, the resolver works corpus-wide against a frozen registry so results do not depend on paper order. After registry version 1 is frozen, later papers can resolve incrementally against the same read-only snapshot. New proposals from one paper never become candidates for later papers inside that run. They are periodically reconciled in another deterministic batch and become visible only after an explicit registry commit.

This is efficient without all-pairs comparison. Type-aware indexes and candidate blocking make the work approximately linear or `n log n`, rather than quadratic.

## 4. The identity model

### 4.1 Shared concept

A shared concept receives an immutable `concept_id` and may connect papers.

Examples include:

- the chemical nitrate;
- the organism *Vernonia amygdalina*;
- the concept groundwater;
- the city Minna;
- a measurement such as turbidity;
- a method such as k-nearest-neighbour imputation.

### 4.2 Study-local instance

A particular object, sample, cohort, station, or group inside a study receives an `instance_id`. In version 1 it is scoped to `paper_id + context_id` and must never be merged across papers merely because its wording is similar. A later reviewed same-study link may connect publications, but it does not silently widen instance identity.

When supported, it links to a shared concept:

```text
PAPER-123 borehole sample  --INSTANCE_OF-->  water sample
```

`INSTANCE_OF` and `IS_A` are relationships, not identity merges. The resolver may accept `IS_A` only from a pinned authority or a reviewed MUFASA relation; it never infers an ontology from wording alone.
Only genuine equivalence may place two mentions on the same `concept_id`. A
broader, narrower, related, or commonly co-occurring concept remains a separate
node connected by an explicit relation.

### 4.3 Qualifier

A qualifier describes a concept or instance and normally does not receive its own identity.

Entity-qualifier examples include depth, protection status, life stage, chemical form, and urban/rural setting. Season, dose, duration, and result direction normally remain observation conditions or result fields rather than being reclassified as entity qualifiers. All such distinctions must be preserved because they may explain why papers disagree.

Each qualifier kind is governed by the policy as one of:

- descriptive;
- useful for disambiguation;
- identity-bearing;
- instance-defining.

An off-list qualifier kind is not valid typed output: extraction retries it, and an exhausted case preserves the raw value only in an invalid/audit record. It cannot support automatic matching until the vocabulary is deliberately revised and versioned.

### 4.4 Correct handling of compound phrases

Semantic decomposition belongs to extraction. The resolver does not blindly shorten phrases to their head word.

For example, extraction may turn `deep borehole groundwater sample` into:

- a study-local water sample;
- the concept `groundwater`;
- the infrastructure concept `borehole`;
- the qualifier `depth = deep`.

It must not assert that this physical sample is identical to every other groundwater sample.

The original surface wording is always retained. All atoms derived from one source phrase share a stable `source_mention_id`, so the decomposition can be reconstructed and audited. A suspicious compound that reaches the resolver undecomposed is flagged for review rather than guessed apart.

Atomicity alone is not enough. Extraction must also preserve every meaningful concept and identity-relevant qualifier. Returning only `groundwater` would be atomic but incomplete if the source also identified a borehole and a depth. Conversely, a genuine proper name must not be split merely to improve an atomicity score.

## 5. Corrections incorporated in this proposal

The proposed implementation rules are:

1. There is no three-word limit for a valid entity name.
2. A name is not rejected simply because it contains `and`.
3. Long official places, institutions, methods, and chemical names remain intact when they denote one referent.
4. Lists and coordinated phrases are split by semantic structure, not word count.
5. There is no blind head-noun collapse.
6. `groundwater`, `groundwater table`, `groundwater quality`, `borehole`, a well, and a water sample are not automatically the same entity.
7. Entity type, role, identity scope, and qualifiers are separate fields. A type alone does not determine identity.
8. The proposed extraction contract uses controlled type and role vocabularies instead of free-form type and role invention. There is no predicate field: atomic observations use candidate controlled statement-kind, direction, result-basis, role, and condition fields. This remains to be proven by the extraction pilots.
9. The resolver handles each mention occurrence, not only each unique string. `Niger`, `spring`, and `RF` can mean different things in different contexts.
10. Study context is inherited by observations but is not copied into every entity label.
11. The extractor or MUFASA input adapter must emit evidence-backed, context-owned `PLACE` mentions. The resolver resolves those mentions; it never mines labels or conditions to manufacture a missing place. The earlier Bosso/Minna example did contain the place in its study context, so structured upstream emission—not resolver guesswork—is the correction.
12. One observation must contain one atomic result. Multi-sample or multi-value arrays are split upstream into scalar observations with a shared comparison-group ID.
13. Blank or absent measurements remain missing. The resolver does not infer them.
14. Unit conversion is a separate normalization module. Entity resolution preserves reported measurements and units unchanged.
15. Observation direction and conditions remain observation fields. The resolver consumes emitted entity qualifiers but does not reclassify dose, duration, direction, or other conditions into qualifiers.
16. Qualifier kinds are controlled and versioned rather than free text.
17. Every atom remains linked to its original source phrase and evidence.
18. Decomposition must be both pure and complete: one concept per atom, no meaningful concept or identity-relevant qualifier omitted, and no improper splitting of genuine names.
19. The resolver never invents or canonicalizes legacy free-form predicates.

## 6. Input contract

The generic module accepts immutable mention records plus optional contextual records. Once approved, the MUFASA adapter supplies all available validated structured context and represents missing or unknown context explicitly.

### Proposed MUFASA adapter inputs

| Input | Purpose |
|---|---|
| `entity_mentions.parquet` | The mentions to resolve |
| `observations.parquet` | Scientific result and comparison context |
| `study_contexts.parquet` | Study design, population, place, and inherited context |
| `evidence_spans.parquet` | Source wording and page-level traceability |
| `extraction_status.parquet` | Confirms that extraction succeeded |
| Extraction manifest and run summary | Supplies source, schema, prompt, model, and settings hashes through one composed run descriptor |
| `mufasa_corpus/manifests/documents.parquet` | Authoritative document eligibility and provenance |

An optional, separately provenance-tracked `authority_hints.parquet` may attach an external identifier to a mention before resolution. The current extraction schema does not emit external authority IDs, so exact-ID matching is unavailable when this enrichment is absent.

These extraction Parquets are proposed outputs of the unapproved LLM extraction notebook; none has yet been produced by a completed run. The completed parsed corpus currently supplies only the document manifest and per-paper parsed artifacts. The generic resolver core receives validated mention records plus a normalized context object. Corpus eligibility, table joins, decomposition validation, and observation atomicity are responsibilities of the MUFASA adapter before it invokes the core.

The candidate extraction seam will be tested with the following required mention fields:

- `mention_id`;
- `source_mention_id`, shared by every atom derived from one source phrase;
- `source_evidence_id` plus the source phrase's character offsets within that evidence span;
- `paper_id`;
- `owner_kind` and `owner_id`;
- `role`;
- exact `surface_text`;
- semantic `atom_text`;
- controlled `entity_type`;
- `identity_scope` of `CANONICAL` or `STUDY_INSTANCE`;
- structured qualifiers;
- extraction schema and prompt versions through the run manifest.

`source_evidence_id`, exact surface text, and character offsets are verified or assigned deterministically against immutable parser-derived text; model-supplied offsets are never trusted without verification. `source_mention_id` is derived from `paper_id + source_evidence_id + verified source offsets`. `mention_id` is derived from that source group plus stable role, atom, type, scope, and qualifier keys after deterministic sorting and duplicate removal. Neither ID may depend on model list order or DataFrame row position.

The null `canonical_id` and `UNRESOLVED` fields currently written by extraction are compatibility placeholders. The resolver must not overwrite the raw extraction table.

### Candidate controlled vocabulary for version 1

The following enums are candidates taken from the untested extraction notebook. They are not frozen and must be reviewed against pilot output before freezing the production extraction and resolver contracts.

**Roles:** `SUBJECT`, `OUTCOME`, `AGENT`, `COMPARATOR`, `METHOD`,
`INTERVENTION`, `MEDIUM`, `TARGET`, `PLACE`, `POPULATION`, `CONTEXT`.

**Entity types:** `PLACE`, `ORGANISM`, `POPULATION`, `SAMPLE_SPECIMEN`,
`MATERIAL`, `CHEMICAL`, `ENVIRONMENTAL_FEATURE`, `HEALTH_CONDITION`,
`PROPERTY_METRIC`, `METHOD`, `MODEL_ALGORITHM`, `DATASET`,
`INTERVENTION_ACTION`, `INFRASTRUCTURE_DEVICE`, `ORGANIZATION`,
`EVENT_PROCESS`, `HAZARD_RISK`, `APPLICATION_USE`, `TIME_PERIOD`,
`STANDARD_POLICY`, `OTHER`.

**Identity scopes:** `CANONICAL`, `STUDY_INSTANCE`.

**Owner kinds:** `CONTEXT`, `OBSERVATION`.

An off-list enum value is not silently converted to `OTHER`: extraction retries
it, and an exhausted case is preserved as invalid for review. `OTHER` is an
explicit allowed choice only when no approved type fits. Physical Parquet fields such as
`qualifiers_json` and `conditions_json` remain preserved strings; the adapter
parses them into typed internal records. `surface_text` is copied from evidence,
whereas `atom_text` is a semantic extraction and need not be a literal span.

Qualifier kinds require a small versioned controlled vocabulary before even the
first extraction pilot. `OTHER` or `UNKNOWN`, if retained, are themselves
members of that enum. A model-emitted off-list spelling may be mapped only by a
versioned deterministic alias map; otherwise extraction retries it and an
exhausted case is preserved as audit-invalid. Free-form `qualifier.kind` values
are never valid typed output or positive matching evidence. Raw qualifier
wording is always preserved.

### Corpus eligibility

The manifest is authoritative. Files must never be discovered by scanning the corpus directories.

Eligibility is the intersection of:

- `pipeline_status == "ok"`;
- rights status permitting this processing;
- retraction status not equal to retracted;
- completed, valid extraction status.

**Dated audit note (2026-08-11, not a permanent design constant):** the current manifest has 17,049 rows and 10,321 rows with `pipeline_status == "ok"`. Review statuses remain excluded unless a signed, typed, versioned override admits them. An override can never bypass rights or retraction rules. Orphan or stale files cannot be overridden; they must first be reconciled into the authoritative manifest with matching hashes and provenance.

The resolver consumes extraction tables, not PDFs or Markdown. If source text is needed for an audit, the adapter reconstructs `{corpus_root}/parsed/structured/{paper_id}.json` rather than trusting obsolete absolute Kaggle paths. It verifies the local `paper_id`, `openalex_id`, PDF hash, and parse/identity status against the manifest and computes a structured-content hash for the run. Parser-derived page text, page numbers, offsets, parser version/configuration, and their hashes remain immutable; repaired or normalized text is stored separately. Null quality values mean unknown, not zero, and page-level structured metadata is consulted when manifest summaries are absent.

The adapter preserves raw metadata and derives typed comparison fields separately. In particular, DOI may be a full `https://doi.org/...` value or missing, while publication date and citation count require explicit parsing rather than string comparison.

### Fail-closed validation

Before matching begins, the module validates:

- source hashes and schema versions;
- unique mention IDs and unique primary IDs in the observation, context, and evidence tables; repeated mention `owner_id` foreign keys are valid;
- referential integrity between mentions, contexts, observations, evidence, and documents;
- controlled type, role, and identity-scope values;
- valid qualifier JSON;
- controlled qualifier kind and permitted value/type for the applicable entity type and identity scope;
- complete `source_mention_id` and evidence linkage for every atom;
- consistent source groups in which all atoms trace to the same exact source phrase;
- suspicious compound mentions, which are retained and routed to review using semantic policy checks rather than token count;
- duplicate atoms caused by overlapping extraction chunks;
- at most one scalar or one complete range per quantitative observation, never both; qualitative observations may have neither;
- document and extraction eligibility.

Invalid rows are recorded as `INVALID_INPUT`; they are never silently dropped, coerced into a match, or converted into legitimate evidence.

### Extraction-contract approval gate

The resolver assumes that extraction has already produced valid atoms. It cannot recover a concept or qualifier that extraction omitted, and it must not guess how to split an undecomposed phrase. The extraction contract therefore has its own approval gate before resolver calibration or full-corpus extraction.

The staged pilot is:

1. run the existing ten test papers;
2. inspect and correct the contract, prompt, and validator;
3. proceed only after that smoke test passes;
4. run a representative 50–100-paper pilot spanning all MUFASA domains, document qualities, tables, long names, and both identity scopes;
5. freeze the approved extraction contract before the full run.

Before either run, its sample, rubric, and thresholds are recorded in a versioned pilot policy. Thresholds cannot be chosen after seeing the results.

The manual decomposition audit works on **source-mention groups**, not isolated atom rows. An initial sample of about 100 stratified source groups is enough to expose major design failures, but it is not proof of near-perfect accuracy. Each group is scored for:

- **atom purity:** every emitted atom denotes one semantic concept or instance;
- **decomposition completeness:** every meaningful concept and identity-relevant qualifier in the source phrase was retained;
- **proper-name integrity:** a genuine name was not split merely because it was long or contained `and`;
- correct entity type, role, identity scope, and qualifier attachment;
- exact source-group and evidence traceability.

A separate passage-level sample is required to find mentions the model omitted completely. Reviewers exhaustively mark every relevant mention in bounded paragraphs, table regions, or other evidence passages, and mention-detection precision and recall are scored against that inventory. Source-group review alone cannot measure omissions because a missing group is invisible in model output.

The pilot also reports:

- `OTHER` share and correctness by domain and entity role;
- allowed and observed qualifier kinds, with **zero off-list kinds** in valid output;
- mentions per 1,000 source words rather than only mentions per paper;
- duplicate-mention rate from overlapping chunks;
- explicit context-place recovery where a place is stated;
- atomic handling of table rows, samples, scalar values, and ranges.

A low `OTHER` rate is not automatically good: it may mean the model forced unfamiliar entities into incorrect allowed types. `OTHER` cases are sampled for correctness, and repeated patterns trigger vocabulary review. Semantic atomicity is never judged by a word-count or conjunction rule.

Every successful output row must be structurally valid, source-linked, and free of off-list qualifier kinds. The owner-approved purity, completeness, improper-split, type/scope, and place-recovery thresholds must also pass before production extraction. Any critical loss or semantic conflation of distinct referents stops approval even if an average score passes.

## 7. Registry design

MUFASA uses its own immutable IDs even when an external authority exists. An authority link can be corrected or added later without changing the MUFASA ID.

### Core durable tables

#### `canonical_entities.parquet`

One row per shared concept:

- `concept_id`;
- preferred label;
- controlled entity type;
- lifecycle status;
- creation and update run IDs;
- registry version;
- provenance.

#### `entity_instances.parquet`

One row per study-local instance:

- `instance_id`;
- `paper_id` and `context_id`;
- local label and type;
- zero or one primary `concept_id` for its authoritative `INSTANCE_OF` target;
- identity-bearing qualifiers;
- creation and update run IDs.

That `concept_id` field is the source of truth for the primary `INSTANCE_OF` link; the graph exporter derives the edge from it. Confirmed cross-publication lineage is stored separately and can never be inferred from a shared `family_id` alone.

#### `mention_resolutions.parquet`

One row for every input mention, including unresolved mentions:

- all source identifiers and original wording;
- `source_mention_id` and its evidence pointer;
- nullable `concept_id` and `instance_id`;
- nullable `proposal_id` for a dry-run concept or instance proposal;
- decision status and reason codes;
- decision method;
- feature scores and runner-up margin;
- calibrated probability only when calibration is valid;
- policy, resolver, registry, authority, and run versions;
- reviewer decision and override provenance.

### Supporting tables

- `canonical_aliases.parquet`: one scoped alias per row, including language, region, type, source, trust level, and version.
- `canonical_authority_links.parquet`: one external authority identifier and snapshot version per concept link.
- `entity_relations.parquet`: authority-backed or reviewed concept-to-concept relations such as `IS_A`; it never duplicates identity or the primary instance link.
- `resolution_candidates.parquet`: top-K and review-bound candidates with feature traces. Exact accepted matches retain compact reason codes instead of an unbounded candidate history; the run policy defines K and retention.
- `resolution_events.parquet`: append-only creation, merge, split, reassignment, alias, review, and supersession history.
- `resolution_review.csv`: a human-friendly queue; it is not the source of truth.
- `run_manifest.json`: hashes and versions needed to reproduce the run.

The review unit is normally one ambiguous alias, candidate pair, or proposed cluster—not every repeated mention occurrence. A reviewed decision may propagate only to occurrences that pass the same type, scope, qualifier, context, and cannot-link checks. Every propagated mapping is still written separately to `mention_resolutions.parquet` with the originating review event.

IDs are persistent opaque identifiers created only at commit time, never row numbers, names, or current batch positions. For a new accepted cluster, the implementation selects the lowest stable source `mention_id` as its immutable seed and derives a namespaced UUID from that seed; once committed, later additions never change it. Proposal IDs are hashes of the frozen cluster membership and policy version. Existing registry IDs always win over newly derived IDs. Merges retain redirects; splits create new IDs and lineage events. No registry history is deleted.

## 8. Authority policy

Resolution is authority-first, not authority-only.

The authority interface supports this cascade:

| Entity | Preferred authority |
|---|---|
| Organisms | NCBI Taxonomy, with GBIF/Catalogue of Life crosswalks when needed |
| Places and administrative areas | GeoNames and Wikidata |
| Chemicals and pesticides | ChEBI, then PubChem |
| Agricultural concepts | AGROVOC |
| Diseases and health conditions | MeSH and MONDO |
| Organizations | ROR |
| Units | UCUM, handled by the separate unit-normalization module |
| Local materials, methods, infrastructure, and engineering concepts | Versioned MUFASA registry |

Authority records, aliases, crosswalks, versions, and licences are cached as pinned local snapshots. The core resolution run has no network dependency. Only the used, licence-compatible authority subset is packaged.

Exact compatible authority identifiers may be used after the extraction semantics and authority mapping are validated; they do not require a fuzzy-matching gold set. Curated aliases, crosswalks, and approximate authority-name matches are activated per type only after they improve measured resolution-gold performance without violating precision gates.

Before any authority data is downloaded, the acquisition recipe records the required geographic or scientific scope, files, version date, licence/attribution, checksum, and refresh policy. Authority downloads can proceed alongside pilot work, but downloading a large global snapshot is not itself a project milestone and does not unblock an unapproved extraction contract.

An authority hit does not override a type, rank, geographic, chemical-form, or other hard conflict. Absence of an authority hit proves neither sameness nor difference. A clean unmatched mention may still self-seed during the deterministic bootstrap, but only after all bootstrap checks pass; absence alone is never the reason.

## 9. Resolution pipeline

### Initial empty-registry bootstrap

The first corpus run does not require manual approval of every new concept. Registry version 1 is built in one deterministic bootstrap:

1. freeze the eligible mentions, approved versioned extraction contract, policy, and any pinned authority snapshots;
2. group exact compatible authority IDs when supplied;
3. group trusted curated aliases and collision-free, type-safe exact normalized names allowed by policy;
4. create separate stable local concepts for clean unmatched canonical mentions and singleton groups;
5. create paper-and-context-scoped study instances separately;
6. keep fuzzy synonym groups, conflicting labels, and ambiguous cases as review candidates;
7. validate all clusters and publish the accepted snapshot atomically as registry version 1.

This deliberately prefers temporary duplicate concepts over fabricated merges. A normal clean singleton does not require individual human approval; review is reserved for ambiguous or high-impact proposed merges and extraction defects. Fuzzy automatic matching remains disabled for each entity type until that type passes resolution-gold calibration.

### Step 1: Freeze the run

Hash the input tables, registry snapshot, authority snapshots, code revision, policy, and calibration artifacts. Matching runs against this frozen state and cannot mutate it.

### Step 2: Build comparison keys

Create separate matching keys while preserving the original text. Normalization may include Unicode normalization, case folding, whitespace repair, punctuation variants, spelling variants, acronyms, and type-specific forms.

Derived comparison keys may be lossy, but normalization is non-destructive and auditable because the untouched raw text is retained beside every key. It is not proof of identity. Curated spelling variants and acronym expansions are aliases, not generic normalization rules. Do not ASCII-fold stored African place names, personal names, scientific names, chemical symbols, or units.

### Step 3: Route by identity scope

- `CANONICAL` mentions follow the shared-concept path.
- `STUDY_INSTANCE` mentions follow the paper/context-scoped path.
- `OTHER`, malformed, or suspiciously compound mentions enter review. Weak-source flags are derived only from recorded fields such as owner review status, evidence exact-match status, parser warnings, low-text pages, and OCR metadata.

### Step 4: Apply high-certainty matches

In descending order of trust:

1. an exact compatible authority identifier, but only when an optional pre-enrichment record actually supplies that identifier;
2. an exact trusted, type-scoped, human-curated alias;
3. a collision-free normalized primary-name match permitted by that type's policy.

Any collision or hard conflict blocks automatic acceptance.

### Step 5: Generate candidates

Candidate retrieval has two separate paths:

- same-identity candidates use compatible type and scope; a study instance can search only instances inside its own `paper_id + context_id`;
- concept-link candidates let a study instance search compatible canonical concepts through an explicit instance-type-to-concept-type policy.

Candidates are then recalled through:

- authority and crosswalk indexes;
- curated alias indexes;
- normalized, character, and token indexes;
- type-specific scientific keys;
- vector similarity as a recall aid only.

Embeddings never prove identity and cannot override a hard conflict.

### Step 6: Score compatible candidates

The score uses type-appropriate evidence such as:

- exact or curated alias agreement;
- authority agreement;
- acronym expansion;
- scientific-name or chemical-form agreement;
- place hierarchy and country context;
- study, role, domain, and qualifier agreement;
- source quality and OCR/parser warnings.

Type, authority, taxonomic rank, geographic hierarchy, chemical form, version, and identity-bearing qualifier conflicts are explicit cannot-link signals.

### Step 7: Decide conservatively

Every mention receives one of these statuses:

- `MATCHED`;
- `NEW_CONCEPT_PROPOSED`;
- `NEW_INSTANCE_PROPOSED`;
- `REVIEW_REQUIRED`;
- `UNRESOLVED`;
- `INVALID_INPUT`.

There is no universal string or cosine threshold. Score-based and approximate methods receive independently calibrated acceptance and margin thresholds by entity type. Deterministic exact-ID, trusted-alias, and bootstrap rules use explicit hard policy checks rather than statistical thresholds. Low similarity alone never means “create a new entity.” A close runner-up means ambiguity and therefore review.

### Step 8: Reconcile new concepts

New concepts first exist as dry-run proposals, not live registry mutations. Bootstrap proposals satisfying the deterministic singleton or exact-group rules above are policy-approved automatically; all other within-batch mentions may group automatically only through strong equivalence evidence, such as the same compatible authority ID or a collision-free trusted exact key.

An unambiguous clean canonical mention may seed a local concept automatically only when its type policy permits self-seeding, it has no candidate above that type's review floor, it has no unknown identity-relevant qualifier, and all hard checks pass. Fuzzy grouping, authority ambiguity, risky types, and conflicting context require human approval. Unapproved proposals remain visible only in review/debug artifacts; they never enter the production registry or GraphRAG. They do not block unrelated approved changes.

Fuzzy matches never form unrestricted transitive clusters. The resolver must prevent the `A is close to B`, `B is close to C`, but `A is not the same as C` failure. Every proposed cluster must satisfy all pairwise hard constraints and have one coherent type and granularity.

### Step 9: Validate, then commit atomically

The proposal run first produces a complete dry-run artifact. An explicit commit publishes accepted mappings and approved registry mutations as a new immutable snapshot. `REVIEW_REQUIRED` and `UNRESOLVED` rows, together with their queue, may be published while still pending; they never block unrelated accepted work and never enter the live registry as matches.

Worker processes may perform normalization, candidate retrieval, and scoring in parallel. They cannot mutate the registry. Decisions are sorted by stable IDs, and registry publication has one atomic writer. This preserves speed without creating order-dependent identities.

## 10. Accuracy safeguards

### Hard invariants

- No version-1 study instance is shared across papers.
- No concept contains incompatible controlled entity types.
- No concept contains conflicting identifiers from the same authority, or cross-authority identifiers without a trusted crosswalk.
- No automatic match violates an identity-bearing qualifier.
- No unresolved or invalid mention disappears from the output.
- Every accepted decision has machine-readable reasons and provenance.
- Reordering input cannot change the logical result.
- Re-running identical inputs against the same snapshot is idempotent.
- Adding unrelated mentions cannot silently rewrite existing IDs.
- Human must-link, cannot-link, and override decisions survive every rerun.

Resolution-row lifecycles are also fixed:

- a committed `CANONICAL` mention receives `concept_id` but never `instance_id`;
- a committed `STUDY_INSTANCE` mention receives `instance_id` and may receive one primary `concept_id`;
- `REVIEW_REQUIRED`, `UNRESOLVED`, and `INVALID_INPUT` receive neither committed ID;
- a proposed row carries `proposal_id` instead of a committed ID until commit, after which it becomes a committed mapping.

Licence-tier evidence eligibility is enforced by the graph-export contract, not by identity matching. If a future coverage-only record is allowed to resolve to a concept, its upstream licence-tier flag must remain attached so the graph loader excludes it from supporting/conflicting evidence counts.

### Precision with minimum useful coverage

False merges are the most damaging error because they manufacture cross-paper support or contradiction. The automatic lane is therefore high precision, while review and unresolved lanes preserve useful text search without fabricated graph edges. Safety cannot be achieved merely by abstaining from everything.

Every release policy therefore contains non-zero gates derived from the resolver pilot on approved extraction output for:

- recall on manually confirmed cross-paper same-concept pairs;
- automatic resolution coverage among gold-labelled resolvable mentions;
- candidate recall before adjudication;
- maximum review workload and unresolved rate;
- a minimum number of automatic decisions sufficient to make precision meaningful.

A zero-match or extremely low-coverage run fails automatically; its precision is undefined or operationally meaningless. Numeric coverage and workload thresholds are frozen after the resolver calibration pilot and before its locked test is inspected. Results are reported by entity type and domain so a strong common type cannot hide failure elsewhere.

Raw cross-paper link count is a diagnostic only, never a release gate. It can be inflated by incorrect broad merges. Correct connectivity is assessed with labelled same-identity recall, cluster precision/recall, and downstream retrieval tests.

### Review capacity

Review workload is counted as distinct ambiguous alias, candidate-pair, or cluster decisions rather than raw mention occurrences. The pilot estimates cluster tasks per 1,000 mentions and reviewer time. Before production, the owner sets a fixed human-capacity ceiling.

The queue is deduplicated and prioritized by recurrence, risk of a false merge, expected graph/retrieval impact, and uncertainty. One reviewed decision propagates only to compatible occurrences and remains fully auditable. If the queue exceeds capacity, low-impact uncertain cases remain unresolved; automatic thresholds are never weakened simply to empty the queue.

An optional LLM may later summarize context for a reviewer. It must not be the sole basis for an automatic identity merge in version 1.

## 11. Reproducibility and resilience

Each run is identified by a fingerprint containing:

- input table hashes;
- extraction schema and prompt versions;
- policy and calibration versions;
- registry and authority snapshot versions;
- normalization version;
- embedding model and file hash, vector-index implementation/version, deterministic seed, and score-rounding rule when vector recall is enabled;
- resolver code revision;
- all effective controls.

Processing is batched, resumable, and idempotent. Batch outputs are written to temporary files and atomically renamed. The manifest is published only after every required output passes schema and referential-integrity checks. A failed run cannot replace the last good registry.

Parallel workers return immutable proposals. Shared mutable state, order-sensitive union operations, and direct worker writes to the registry are forbidden.

Approximate vector results close to an acceptance boundary cannot control an automatic decision unless repeated-run stability has been demonstrated. Stable candidate sorting and tie-breaking use explicit rounded scores followed by stable IDs.

## 12. Evaluation notebook

The future `03-retrieval/entity-resolution-evaluation.ipynb` will be a thin test and inspection surface.

Its top cell will expose only practical controls:

- input and registry paths;
- entity types or domains to test;
- sample size;
- batch size and worker count;
- fresh-run or resume mode;
- `COMMIT = False` by default.

It will:

1. load a frozen input and registry snapshot;
2. validate the approved extraction contract and its pilot report;
3. stop before resolution if source grouping, decomposition, qualifier, evidence, or other approved extraction gates fail;
4. call the reusable module;
5. show progress and status/type distributions;
6. score the extraction and resolution gold sets separately;
7. display false merges, false splits, decomposition errors, ambiguous candidates, and reason traces;
8. show concept clusters and study-instance links with their source evidence;
9. show the review queue as deduplicated cluster decisions with projected reviewer effort;
10. shuffle the same input and rerun to prove order invariance;
11. export the review queue, metrics, and run manifest.

The notebook must not contain a second normalization, matching, or registry implementation.

## 13. Evaluation standard

Extraction and resolution are evaluated separately. A resolver cannot compensate for a concept that extraction omitted, and one combined score would hide which layer failed.

### 13.1 Extraction gold set

Each reviewed source phrase maps to the complete expected set of atoms and qualifiers. Labels cover:

- mention detection;
- atom purity and decomposition completeness;
- proper-name integrity and improper splitting;
- entity type, role, and identity scope;
- qualifier kind, value, and attachment;
- source-group and evidence linkage;
- atomic table and measurement handling.

The ten existing papers seed this set only after they are re-extracted and reviewed under the new source-group contract; their legacy claim labels do not automatically become decomposition gold. If they pass, the set expands from the representative 50–100-paper pilot. About 100 manually inspected source groups is an initial diagnostic, not evidence of near-perfect accuracy.

### 13.2 Resolution gold set

This set begins only with extraction outputs that passed the extraction gate. Its labels distinguish:

- same concept;
- different concept;
- same study-local instance;
- instance of concept;
- insufficient evidence.

It is stratified by type and difficulty and includes exact names, trusted aliases, African vernacular and multilingual names, homonyms, acronyms, OCR/Unicode corruption, long official names, nested places, organism ranks, chemical forms, broader/narrower concepts, study-local objects, and the groundwater/borehole/well, `Niger`, `spring`, `RF`, and Bosso/Minna cases.

The locked ambiguous and high-risk test subset is independently double-annotated, disagreements are adjudicated, and inter-annotator agreement plus label/version provenance are recorded. Straightforward pilot labels may be owner-reviewed once; full double annotation is not a prerequisite for module design or the deterministic exact/bootstrap lane.

Calibration and final testing use separate locked partitions. Papers, study families, aliases, and canonical entities are grouped so the same identity cannot leak between partitions. The final test includes unseen concepts and unseen aliases. Fuzzy automatic matching is enabled separately for an entity type only after that type has sufficient reviewed calibration evidence; otherwise it remains candidate-only.

After the representative extraction pilot passes, its approved 50–100-paper output supplies a distinct **resolver pilot**. Reviewers label an initial calibration partition and a locked test partition while the deterministic exact/bootstrap lane is implemented and exercised. This pilot measures automatic coverage, candidate recall, ambiguity clusters, and projected review work. It freezes the resolver gates before the locked partition is opened; fuzzy matching remains candidate-only during this stage.

### 13.3 Resolver metrics and release gates

Metrics are reported overall and by entity type, domain, language, ambiguity class, and source quality:

- candidate recall;
- automatic-match precision, recall, and resolvable-mention coverage;
- cross-paper same-concept pair recall;
- pairwise and B-cubed cluster precision/recall;
- overmerge and undermerge rates;
- authority-link and concept-instance accuracy;
- review-cluster and abstention rates;
- estimated reviewer effort;
- canonical-ID churn;
- downstream graph-join precision and retrieval effect.

The aspirational automatic-match precision target remains at least 99.5%, with a 95% lower confidence bound of at least 99%, zero critical overmerges in the frozen set, zero cross-paper study-instance leakage, and complete audit coverage. The target is evaluated only with a predeclared confidence method and enough automatic decisions to make the bound meaningful.

The versioned policy also freezes non-zero minimums for candidate recall, confirmed same-concept recall, and automatic coverage of gold-resolvable mentions, plus maximum review workload and unresolved rate. Those numeric limits are approved from the resolver calibration pilot before its locked test is opened. A zero-match or extremely low-coverage resolver fails automatically. No raw cross-paper link-count target is used.

### 13.4 Hybrid retrieval regression

Questions Q-013 through Q-020 in [milestone1-test-data/questions.jsonl](./milestone1-test-data/questions.jsonl) test the complete BM25 + vector + graph retrieval path, not entity identity alone. For each question, the integrated retriever must return at least its declared `minimum_expected_hits` within the top ten results and record which results came from lexical search, semantic search, and graph expansion.

Their current `OBS-G...` expected-claim identifiers belong to the legacy fixture. After re-extraction, a versioned, reviewed mapping must connect each question to the approved new observation and evidence IDs before the regression is scored. The questions and gold answers remain stable; stale IDs are never treated as current evidence.

These eight questions are development regressions from the ten-paper fixture, not a sufficient independent release benchmark. They must later be supplemented with locked questions from the representative corpus. Related-but-distinct bridge concepts must never be identity-merged merely to make a question traversable.

## 14. Agreed sequence after design approval

No implementation begins from this draft before owner review. Once approved, the sequence is:

1. finalize the candidate extraction schema, including source-mention grouping, controlled qualifier kinds, and validator rules;
2. run and review the existing ten-paper extraction smoke test;
3. revise until the smoke-test gates pass;
4. pre-register and run the representative 50–100-paper extraction pilot;
5. freeze extraction contract version 1 and create the extraction gold set;
6. label initial resolution calibration and locked-test examples from approved extraction output;
7. define and pin only the authority subsets justified by the target types, including licences and hashes;
8. implement the deterministic exact/bootstrap lane while the initial resolution labels are completed;
9. run the resolver pilot, expand the resolution gold set, estimate review capacity, and freeze resolver gates;
10. run full extraction while authority indexing and per-type fuzzy calibration proceed in parallel;
11. enable fuzzy automatic matching only for types that pass their locked gates;
12. run corpus-wide resolution and the hybrid retrieval regressions.

Full-corpus resolution and final calibration require validated production extraction output. Module-interface design, extraction-contract review, deterministic normalization design, pilot fixtures, and scoped authority planning do not require all 10,321 currently eligible papers to be extracted first.

## 15. Definition of done for the later implementation

The module is ready for production only when:

- the extraction, qualifier, and resolution vocabularies are versioned;
- the extraction contract is approved and its ten-paper and representative pilots pass;
- source-mention grouping makes decomposition auditable, and the extraction pilot/gold set demonstrates measured purity and completeness without improper proper-name splitting;
- passage-level extraction gold measures omitted-mention precision and recall;
- zero off-list qualifier kinds occur in valid extraction output;
- separate extraction and resolution gold sets exist;
- every eligible mention receives a durable resolution row;
- all hard invariants pass;
- the gold and adversarial tests pass their release gates;
- the exact/bootstrap lane is reproducible before any fuzzy lane is enabled;
- the resolver calibration pilot freezes coverage and review-capacity gates before its locked test is opened;
- recall, automatic-coverage, and review-capacity gates pass alongside precision gates;
- shuffled and resumed runs reproduce the same logical results;
- merge, split, redirect, and human-override history is preserved;
- the notebook imports the module and contains no resolver logic;
- production and notebook runs produce the same decisions from the same inputs;
- graph export uses immutable IDs rather than entity names;
- Q-013 through Q-020 pass as hybrid retrieval regressions rather than resolver-only identity tests;
- all registry and authority artifacts can run offline from pinned, licence-checked snapshots.

If approved and validated, this design would give MUFASA one resolver, one durable identity registry, and one test notebook. Its intended accuracy comes from preserving distinctions, measuring useful coverage, using authoritative evidence where available, and refusing to guess when identity is genuinely uncertain.
