# MUFASA African-Relevance Classification Protocol

## Purpose and Scope

Use this protocol to decide whether a scientific paper is sufficiently relevant to Africa for the MUFASA corpus.

This pass classifies **African scientific relevance only**. Discipline/category codes are assigned later. DOI, citations, dates, licences, URLs, and download availability do not affect relevance. Do not force a target percentage of `include`, `review`, or `exclude` decisions.

## 1. Classifier Input

Give the model only:

- `title`
- `abstract`
- OpenAlex `field_name`
- OpenAlex `primary_topic`
- up to five concise OpenAlex keywords
- `work_type`, when available

Title and abstract are the primary paper evidence. Field, topic, and keywords may clarify an entity already present in the text, but they cannot independently create African relevance, move an otherwise absent case to `review`, justify `include`, or establish a country.

Keep these outside the prompt: authors, `institution_names`, affiliation countries, DOI, journal, citations, dates, URLs, licences, gold labels, and MUFASA category codes.

An institution named inside the abstract counts only when it is recognisably African **and** supplies the studied patients, samples, records, or measurements. An author affiliation never counts.

## 2. Core Relevance Test

Ask:

> Does an African subject, condition, entity, problem, dataset, or application materially affect the scientific question, evidence, interpretation, or application?

The words `Africa` or `African` are not required. African geography, crops, organisms, disease burdens, resources, environments, local methods, or industrial constraints may provide evidence.

Use exactly one evidence level:

| Evidence level | Meaning | Decision effect |
|---|---|---|
| `direct` | Title or abstract explicitly identifies an African subject, source, setting, dataset, condition, or application | May support `include`; centrality is recorded separately |
| `inherent` | A named entity is Africa-exclusive or unmistakably African even though the text does not spell out the geography | May support `include`; never invent a country or collection site |
| `latent` | A central entity is credibly important or common in Africa but is not Africa-exclusive | Must be at least `review` when in scientific scope and not contradicted |
| `affiliation_only` | The only African signal is authorship, institution, venue, or affiliation country | `exclude` |
| `absent` | No direct, inherent, or credible latent signal | `exclude` |
| `contradicted` | The studied population, samples, material source, environment, dataset, or intended application is explicitly non-African, with no meaningful African component | `exclude` |

`evidence_level` records **how** the connection was detected; `african_focus` records **how important** it is. For example, an explicit but minor African result is `evidence_level=direct` and `african_focus=incidental`.

### Inherent versus latent

- Africa-exclusive or unmistakably African entity: `inherent`.
- Native to Africa but also widespread elsewhere: `latent`, unless the paper adds direct African evidence.
- Common or especially important in Africa but globally distributed: `latent`.
- Unfamiliar organism or material name: not evidence by itself.

Possible latent signals include major African crops/biomass (such as cassava, maize, millet, plantain, or rice husk), high-burden diseases (such as malaria or schistosomiasis), locally important materials/resources, and characteristic environmental or industrial constraints. This is not a whitelist. The entity must be central; a passing keyword is insufficient.

Assign `latent` only when all are true:

1. the entity is central in the title or abstract;
2. its African salience has a widely established basis, such as major African production/use, disease burden, native distribution, characteristic resource/environment, or recognised operating constraint;
3. that basis can be stated plainly in `inference_basis`;
4. no contrary research geography is supplied.

Mere occurrence somewhere in Africa or generic transferability is insufficient. If the classifier cannot state the salience basis, use `absent` and `exclude`.

Explicit research geography overrides latent relevance. A cassava, maize, malaria, or biomass study based entirely on a non-African population, source, environment, dataset, or application is `contradicted` unless Africa is meaningfully compared or targeted. A non-African author institution or analysis laboratory alone is not contrary research geography.

## 3. Anti-Hallucination Rules

Separate internally:

1. **Observed facts** explicitly stated in the supplied title or abstract.
2. **Permitted general inference** used only to recognise an entity as inherent or Africa-salient.

Apply these rules without exception:

- Never use remembered details about this particular paper, its authors, journal, hospital, university, country, sample source, or study site.
- Never infer geography from names, affiliations, journal, DOI, language, or model familiarity.
- Never turn an unnamed "city," "hospital," "community," "dump site," or "tertiary centre" into a Nigerian or African one.
- General knowledge may establish only broad African specificity or salience; it cannot establish where this paper obtained its data or materials.
- Put every permitted inference in `inference_basis`. Any other unsupported factual claim must be removed.
- An `include` without direct African wording is allowed only for a confidently `inherent` entity. Other outside-knowledge cases are capped at `review`.
- Quote an exact title/abstract phrase in `evidence`. If neither the quote nor a valid declared inference supports the African connection, use `exclude`.
- List an African country code only when the research content identifies it. Never copy affiliation country codes.
- If title and abstract conflict, do not repair the record from memory; use `review` and state the conflict.

Example: "samples from dump sites around the city" does not permit "samples from Nigerian dump sites," even when the authors are Nigerian.

## 4. Scientific Scope and Hard Exclusions

Set `hard_exclusion=true` when the paper is primarily:

- African only through affiliation, institution, venue, or funding;
- explicitly non-African with no meaningful African comparison or application;
- economics, business, law, governance, humanities, philosophy, or education-only work;
- an awareness, attitude, perception, preference, service-utilisation, or training survey without substantial biological, clinical, environmental, agricultural, or engineering measurement;
- an editorial, announcement, advocacy statement, quotation, nomenclature proposal, or opinion without scientific analysis.

Quantitative statistics alone do not place a subject inside scope:

| In scientific scope | Outside this corpus scope |
|---|---|
| Rainfall, temperature, crop-yield, emissions, or climate-model evidence | Climate policy, emissions trading, governance, or economic adaptation without scientific measurement |
| Diagnosed disease, laboratory results, pathology, treatment response, exposure, or biological risk | Awareness, knowledge, perception, stigma, preference, counselling need, or service use alone |
| Measured livestock disease, antimicrobial-use pathways, crops, pests, soil, food composition, or production conditions | Opinions, marketing behaviour, or training needs without substantive agricultural/biological evidence |
| Tested materials, devices, infrastructure, algorithms, energy systems, or environmental conditions | Adoption intentions, consumer behaviour, business strategy, or policy recommendations alone |
| Systematic review or meta-analysis of scientific outcomes | Editorial, advocacy, call to action, quotation, or narrative opinion without scientific synthesis |

A questionnaire can still support in-scope epidemiology, field science, environmental health, livestock, or agriculture when it measures actual conditions or practices rather than knowledge and perceptions alone.

Retractions, duplicates, and confirmed wrong records should be removed upstream using explicit record evidence. A suspected or partial title-abstract mismatch goes to `review`.

## 5. Scores

Scores explain the decision; they cannot manufacture relevance.

Score every dimension honestly before applying a hard-exclusion override. Do not erase real African centrality merely because the contribution is policy, awareness-only, or otherwise outside scope. For example, an Africa-centred economic-policy paper may have high centrality but low scientific depth and still be excluded by the hard-exclusion rule.

- `african_centrality`: `4` explicit and fundamental; `3` direct/inherent and highly meaningful; `2` meaningful subset/comparison/application; `1` latent or limited African result; `0` affiliation-only, absent, contradicted, or generically useful.
- `local_specificity`: `4` named African site/source/population/condition integral to the study; `3` named country/region or African-specific entity; `2` meaningful regional subset or defined salient resource; `1` globally distributed salient entity; `0` none.
- `scientific_depth`: `4` robust experiment, measurement, large dataset, or validated model; `3` sound observational, clinical, epidemiological, computational, or field analysis; `2` limited/descriptive study or structured review; `1` case report, minimal analysis, or incomplete abstract; `0` no method.
- `knowledge_value`: `4` rich reusable findings/data; `3` substantive findings; `2` limited, confirmatory, or synthesised knowledge; `1` minimal new knowledge; `0` none.
- `local_applicability`: `4` directly solves a defined African need; `3` strong African system/resource/constraint application; `2` credible application through a salient entity; `1` generically transferable; `0` none.

Do not inflate centrality or applicability because a generic technology might someday be used in Africa. Almost any technology could.

## 6. Decision Rules

### Include

Choose `include` only when every condition holds:

1. no hard exclusion;
2. evidence is `direct` or `inherent`;
3. the African element materially affects the science;
4. `african_centrality >= 2`;
5. `scientific_depth >= 2`;
6. the five-score sum is at least `14/20`;
7. the evidence quote and any declared inference support the decision.

### Review

Choose `review` when no hard exclusion applies and any condition holds:

- a credibly Africa-salient entity is central, the work is scientifically in scope, and no contrary research geography exists;
- direct or inherent African evidence exists but any include gate fails;
- the African title is supported by a missing, truncated, boilerplate, or mismatched abstract;
- a global study contains a substantive African subset or African-specific result, but Africa is not central;
- positive African evidence is incomplete or conflicts with other supplied evidence.

Latent evidence has a ceiling of `review`. It becomes eligible for `include` only when the text adds direct evidence or the entity is confidently Africa-specific and therefore `inherent`.

### Exclude

Choose `exclude` for a hard exclusion, `affiliation_only`, `absent`, or `contradicted` evidence; generic work that merely could be useful in Africa; an incidental African mention with no African analysis; or content with no meaningful scientific contribution.

Set `african_focus` separately:

- `essential`: direct/inherent African evidence is fundamental;
- `meaningful`: direct/inherent evidence materially contributes;
- `potential`: evidence is latent;
- `incidental`: Africa is a minor subset or passing content mention;
- `absent`: affiliation-only, absent, or contradicted.

## 7. Edge-Case Calibration

| Case | Decision | Reason |
|---|---|---|
| Nigerian patients/materials/site are central to measured scientific evidence | `include` if score/depth gates pass | Direct African science |
| Cassava, maize, laterite, rice husk, or similar salient material is central but no geography is supplied | `review` | Relevant to Africa but not Africa-exclusive; do not invent origin |
| Same salient entity explicitly sourced and applied only in Thailand, Peru, China, etc. | `exclude` | Contrary research geography overrides latent relevance |
| Generic substantive malaria/antimalarial study with no location | `review` | Africa-salient disease, but no direct African evidence |
| Reliably Africa-endemic organism is the central experimental subject | `include` may be allowed | `inherent`; leave country codes empty unless stated |
| Pantropical organism with possible African importance but no source | `review` only if salience is credibly known; otherwise `exclude` | Never infer Nigerian collection from authors |
| Global study merely includes African countries | `exclude` | African participation is incidental |
| Global study reports a distinct substantive African result/subset | `review` | Meaningful but not central |
| African data are a major component or shape conclusions | `include` may be allowed | Africa materially affects evidence/interpretation |
| "Climate change and Africa" discusses only policy, markets, or emissions trading | `exclude` | Direct Africa, but outside scientific scope |
| Snakebite advocacy or quotation without data/synthesis | `exclude` | Salient disease cannot overcome absent science |
| Nigerian autism knowledge, HIV awareness, or healthcare-utilisation survey | `exclude` | Knowledge/perception/service outcome only |
| Nigerian clinical or epidemiological study with diagnoses, assays, pathology, or measured outcomes | `include` may be allowed | Direct population plus scientific evidence |
| "Patients at Lagos State University Teaching Hospital" supply the records | Direct evidence | Hospital is explicit and is the data source |
| "Patients at a tertiary referral centre" with Nigerian authors | Not direct evidence | The location is unnamed; do not hallucinate it |
| African title but missing/boilerplate abstract | `review` | Relevance is visible; scientific depth is unverified |
| Generic global science by African authors | `exclude` | Affiliation only |
| African-American or diaspora population outside Africa | `exclude` unless continental African comparison/data/application is substantive | Ancestry is not continental research geography |

## 8. Explanation Standard

The explanation must retain Qwen's useful paper-specific clarity while eliminating unsupported details. Never use a generic template as the complete reason.

Write one or two concise sentences, normally 25-60 words. Every reason must contain:

1. **What the paper actually does**: name the concrete subject and method or evidence type.
2. **What the African connection is**: distinguish direct text, inherent identity, latent salience, incidental content, or affiliation only.
3. **Why that produces this decision**: name the satisfied gate, missing evidence, contrary geography, or hard exclusion.

For `review`, state exactly what is missing: African source/location, sufficiently complete abstract, scientific depth, or centrality. For `exclude`, state the precise failure rather than saying only "generic, global, or outside Africa." For `include`, identify how the African element materially affects the science.

| Weak or unsafe reason | Required clearer reason |
|---|---|
| "The work is generic or outside Africa." | "The paper synthesises Schiff-base corrosion inhibitors and tests them in standard HCl, but names no African material, source, operating condition, dataset, or application; African affiliation alone cannot establish relevance." |
| "Cassava is African, so include." | "The experiment converts cassava peel into biopolymer. Cassava is highly Africa-salient but globally distributed, and no African source or application is supplied; classify as review rather than inventing a location." |
| "This is Nigerian clinical research." | "The abstract explicitly analyses 90 jaw-lesion records from Lagos State University and reports clinicopathological distributions; the Nigerian patient dataset directly supplies the scientific evidence, so include if the score gates pass." |
| "Africa is central, so include." | "The paper discusses African private-sector adaptation and emissions trading but provides no scientific measurements or modelling; Africa is central, yet the contribution is economics/policy and is hard-excluded." |

The reason may mention a general African-salience inference only when the same statement appears in `inference_basis`. Never add a paper-specific country, institution, population, sample source, or result that is absent from the quoted evidence.

## 9. Grounded Output

Return one valid JSON object:

```json
{
  "decision": "include | review | exclude",
  "hard_exclusion": false,
  "hard_exclusion_reason": "",
  "evidence_level": "direct | inherent | latent | affiliation_only | absent | contradicted",
  "african_centrality": 0,
  "local_specificity": 0,
  "scientific_depth": 0,
  "knowledge_value": 0,
  "local_applicability": 0,
  "african_focus": "essential | meaningful | potential | incidental | absent",
  "scientific_evidence": "experimental | observational | computational | systematic_review | theoretical | case_report | weak_or_none",
  "african_country_codes": [],
  "african_relevance_tags": [],
  "evidence": "exact title/abstract excerpt, at most 25 words",
  "inference_basis": "brief permitted inference, or empty",
  "reason": "paper-specific grounded explanation, normally 25-60 words"
}
```

The pipeline computes the five-score total; the model does not output it. Do not output discipline/category codes.

When `hard_exclusion=true`, use one reason: `affiliation_only`, `explicit_non_african_scope`, `outside_scientific_scope`, `awareness_perception_or_service_only`, or `editorial_advocacy_or_opinion`. When false, leave `hard_exclusion_reason` empty. A generic paper with no signal is a normal exclusion: `hard_exclusion=false`, `evidence_level="absent"`.

Useful relevance tags are: `african_location`, `african_population`, `african_dataset`, `african_material`, `african_crop`, `african_organism`, `african_environment`, `african_resource`, `african_industrial_problem`, `local_method_or_knowledge`, `africa_specific_application`, and `latent_africa_relevance`.

Reserve tags beginning `african_` for direct or inherent evidence; they must not imply an unstated African source. A location-unknown cassava, maize, malaria, material, or organism paper receives only `latent_africa_relevance`, while its exact salience is explained in `inference_basis`. Use no relevance tags for affiliation-only, absent, or contradicted cases without meaningful African content.

## 10. Final Check

Before returning:

1. Identify the actual scientific contribution and primary outcome.
2. Separate observed paper facts from declared general inference.
3. Apply the latent-review floor to genuinely salient central entities.
4. Check contrary research geography and scientific-scope exclusions.
5. Remove every invented country, institution, sample source, population, or paper-specific fact.
6. Ensure an `include` passes every gate and quotes exact evidence.
7. Ensure the reason names the actual study, states what is observed versus inferred, and gives the precise inclusion, review, or exclusion basis.
