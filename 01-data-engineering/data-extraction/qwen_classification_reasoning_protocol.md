# MUFASA Independent Classification — Full Reasoning Protocol

This document captures the complete reasoning methodology used to independently classify
the 200-paper benchmark. It is written as a self-contained instruction set that another
AI classifier can follow to reproduce these results with high fidelity.

---

## 1. The Core Question

For every paper, answer ONE question first:

> **Is this paper scientifically ABOUT Africa, or merely BY Africans?**

- "About Africa" = the research question, evidence, materials, organisms, populations,
  conditions, or application are fundamentally African.
- "By Africans" = the authors happen to work at African institutions, but the science
  itself could have been done anywhere and is not shaped by African context.

This single distinction drives ~70% of all decisions.

---

## 2. Input Data (What You Read)

For each paper, you receive:
- `title`
- `abstract` (the primary evidence source)
- `keywords_json` (OpenAlex-assigned keywords)
- `topics_json` (OpenAlex topic hierarchy)
- `institutions_json` (author affiliations)
- `countries_json` (country codes of affiliations)
- `field_name`, `primary_topic`, `primary_subfield`
- `journal`, `work_type`, `cited_by_count`

**You do NOT read or use any `gold_*` columns.** Classification is derived purely from
the paper's own metadata.

---

## 3. The Five Scoring Dimensions (0–4 each)

### 3.1 African Centrality (Is Africa central to the research question?)

| Score | Criterion | Examples |
|-------|-----------|----------|
| **4** | Africa/African country appears in the title AND the research question is fundamentally about an African subject | "Phytochemical constituents of some **Nigerian** medicinal plants"; "Biodiversity of nematodes of sugarcane in **Bacita, Nigeria**" |
| **3** | African materials/organisms/conditions are the study subject but Africa isn't in the title, OR Africa is in the title but the framing is somewhat broader | "Tridax procumbens leaves screened at University of Port Harcourt" (plant is Nigerian but title doesn't say Nigeria); "Cassava peel waste for bioplastics" (African crop, study done in UK) |
| **2** | Some African connection in the content but it's not the main focus; OR a multi-country study where Africa is a significant subset | "Global AMR metagenomics with 10+ African countries sampled"; "Okra production survey in Ebonyi State" (African but descriptive) |
| **1** | Only African affiliation or a single incidental mention of Africa in a global study | "Global Burden of Disease Study covering 195 countries" (mentions sub-Saharan Africa once); generic review by Nigerian authors |
| **0** | No African connection whatsoever in the content | "Antibacterial activity of Northern **Peruvian** medicinal plants" (NG in countries but zero African content) |

**Key nuance**: A paper can score centrality=4 but still be EXCLUDED if it fails the
STEM requirement (e.g., "Climate change and Africa" is an economics/policy paper →
centrality=4 but hard-excluded as outside STEM).

### 3.2 Local Specificity (Does it involve specific African materials/resources/conditions?)

| Score | Criterion | Examples |
|-------|-----------|----------|
| **4** | Names specific African locations, materials, species, or conditions that are INTEGRAL to the study design | "Laterite from three sites in southeastern Nigeria"; "Fulani pastoral communities in North-central Nigeria"; "Soot from illegal refineries in Rivers State" |
| **3** | References African resources/conditions meaningfully but with less geographic precision | "Cassava waste, abundant in developing countries"; "Hibiscus asper, an African medicinal plant" |
| **2** | Some African-specific elements present but not deeply specified | "African agricultural waste" without naming the specific waste or location |
| **1** | Vague African connection (e.g., "tropical conditions" without naming Africa) | "Suitable for tropical climates" (could be any tropical country) |
| **0** | No African-specific content at all | Generic global review; standard benchmark problems; international datasets |

**Key nuance**: The presence of a Nigerian institution in `institutions_json` does NOT
contribute to local_specificity. Only the CONTENT of the research (what was studied,
where samples came from, what conditions were tested) counts.

### 3.3 Scientific Depth (Does it contain methods, experiments, measurements, results?)

| Score | Criterion | Examples |
|-------|-----------|----------|
| **4** | Full experimental/analytical study: clear methods, quantitative data, statistical analysis, results | Phytochemical screening with GC-MS identifying 39 alkaloids; field survey identifying 12 nematode species with counts; metagenomics with diversity indices |
| **3** | Substantial scientific content: has data and analysis but may be observational/survey-based, or a focused computational study | Cross-sectional study of 1,013 children with statistical comparisons; CFD simulation with validated results; questionnaire survey with chi-square analysis |
| **2** | Moderate scientific content: a review that synthesizes findings, or a study with limited data | Systematic review with structured comparison; descriptive statistics only; theoretical framework with some validation |
| **1** | Minimal scientific content: perspective pieces, brief communications, or papers with almost no data | Policy advocacy piece; conference abstract with one paragraph; naming convention proposal |
| **0** | No scientific content: editorials, announcements, letters, conference schedules | Kofi Annan quote about snakebite; editorial calling for action |

**Key nuance**: A well-conducted survey/questionnaire study gets depth=3, not depth=4.
Depth=4 requires laboratory experiments, field measurements, computational modeling with
validation, or large-scale data analysis with statistical rigor.

**Key nuance**: Systematic reviews get depth=2 (they synthesize but don't generate new
data). However, a meta-analysis with statistical pooling gets depth=3.

### 3.4 Knowledge Value (Could the findings teach useful scientific knowledge?)

| Score | Criterion | Examples |
|-------|-----------|----------|
| **4** | Rich findings that would teach substantial domain knowledge to a model | Full phytochemical profile identifying 23+ compounds; comprehensive nematode biodiversity data; detailed AMR gene abundance patterns across continents |
| **3** | Good scientific content with useful, citable findings | Specific compound identification; growth performance data; correlation between variables; identified species at a location |
| **2** | Some useful knowledge but limited novelty or depth | Confirmatory study; known methods applied to new location; review summarizing existing knowledge |
| **1** | Limited knowledge value; mostly confirms what's known or provides minimal new information | Brief communication; pilot study with tiny sample; opinion piece |
| **0** | No extractable scientific knowledge | Editorial; announcement; retracted paper |

**Key nuance**: A study of African medicinal plants that identifies specific bioactive
compounds gets knowledge=4 because it teaches the model which plants contain which
compounds and at what concentrations. A generic review of "drug delivery systems" gets
knowledge=2 because it summarizes existing knowledge without generating new African-specific findings.

### 3.5 Local Applicability (Could the work inform locally appropriate solutions?)

| Score | Criterion | Examples |
|-------|-----------|----------|
| **4** | Directly applicable to African conditions/needs; addresses a specifically African problem | Nigerian medicinal plants for malaria therapy; periwinkle shell as construction aggregate in Niger Delta; antimicrobial usage patterns in Nigerian pastoral communities |
| **3** | Likely applicable to African context; findings transferable to African settings | Cassava waste for bioplastics (applicable to African biorefineries); solar panel soiling in tropical coastal environments; brucellosis prevalence data for Nigerian livestock policy |
| **2** | Potentially applicable but not specifically designed for African conditions | Generic water treatment method that could work anywhere; standard engineering approach tested with one African material |
| **1** | Marginal applicability; would require significant adaptation | Generic algorithm that could be applied anywhere; fundamental physics with no clear application path |
| **0** | No local applicability whatsoever | Pure mathematics; research about non-African regions; theoretical physics |

---

## 4. Decision Logic

### 4.1 Hard Exclusions (Check FIRST — override everything)

If ANY of these apply, the decision is **exclude** regardless of scores:

| Hard Exclusion | How to Detect | Example |
|----------------|---------------|---------|
| **African affiliation only** | The abstract/title/keywords contain NO African materials, organisms, locations, conditions, or research questions. The ONLY African signal is in `institutions_json` or `countries_json`. | "Generic global review of polymer composites" by a Nigerian author |
| **Non-African research region** | The study is explicitly about another region (Peru, China, Korea, Europe, US) and Africa appears only as a co-author affiliation | "Antibacterial activity of Northern Peruvian medicinal plants" with NG in countries |
| **Outside STEM** | The paper is purely economics, policy, law, business, education pedagogy, sociology, philosophy, or humanities with no scientific/experimental content | "Climate change and Africa" published in Oxford Review of Economic Policy; keywords are "economics, business, government, market economy" |
| **Editorial/advocacy** | The abstract is a call to action, a quote, a policy statement, or a conference announcement with no original data | Abstract is a Kofi Annan quote about snakebite |
| **Retracted** | `work_type` indicates retraction | — |

**Critical nuance on "Outside STEM"**: A paper about climate change impacts on African
agriculture IS STEM (it involves crop yields, temperature data, precipitation modeling).
A paper about climate change POLICY, emissions trading frameworks, and economic adaptation
strategies is NOT STEM. The distinction is: does it contain scientific data, measurements,
models, or experiments? Or is it purely about governance, economics, and policy recommendations?

**Critical nuance on "African affiliation only"**: Many papers in this benchmark are
generic global reviews (nanoparticle toxicity, drug delivery systems, robotics in
manufacturing, optimization algorithms) written by researchers at Nigerian universities.
These are EXCLUDED. The test is: if you removed the author affiliations, would you know
this paper has anything to do with Africa? If no → exclude.

### 4.2 Scoring Thresholds

After checking hard exclusions:

```
IF total_score >= 14 AND african_centrality >= 2 AND scientific_depth >= 2:
    decision = "include"
ELIF total_score >= 10:
    decision = "review"
ELSE:
    decision = "exclude"
```

### 4.3 Override Rules (Applied AFTER threshold logic)

1. **High centrality but non-STEM**: If centrality >= 3 but the paper is clearly
   economics/policy/law/education → override to **exclude** with hard_exclusion note.

2. **High scientific value, low centrality**: If total >= 14 but centrality = 1
   (e.g., landmark global genomics study that includes African populations as a subset)
   → downgrade to **review**, not include. The paper must be ABOUT Africa, not just
   include Africa in its data.

3. **Borderline include/review**: If total = 13-14 and the paper has genuine African
   content but limited scientific depth (e.g., descriptive survey with no statistics)
   → prefer **review** over include.

4. **Borderline review/exclude**: If total = 9-10 and the paper has some African
   connection (centrality >= 2) → prefer **review** over exclude. False negatives
   (excluding a relevant paper) are worse than false positives.

---

## 5. African Focus Level

Determined AFTER scoring:

| Focus | Condition |
|-------|-----------|
| **essential** | centrality >= 4, OR (centrality >= 3 AND specificity >= 3) |
| **meaningful** | centrality >= 2 AND specificity >= 2, OR centrality >= 3 |
| **incidental** | centrality >= 1, OR African countries present but content is global |
| **absent** | centrality = 0, OR hard exclusion for non-African content |

---

## 6. Scientific Evidence Type

Classify the METHODOLOGY, not the topic:

| Type | Indicators |
|------|-----------|
| **experimental** | Lab work, measurements, assays, synthesis, characterization, field experiments, clinical measurements, phytochemical screening, GC-MS, PCR, animal trials |
| **observational** | Surveys, field observations, monitoring, epidemiological studies, cross-sectional studies, questionnaires with statistical analysis, biodiversity surveys |
| **computational** | Simulations, algorithms, machine learning, numerical modeling, CFD, finite element analysis, bioinformatics pipelines |
| **systematic_review** | Literature reviews, meta-analyses, systematic reviews, "review of...", "advances in...", "challenges and potentials" |
| **theoretical** | Mathematical derivations, theoretical frameworks, policy analysis, conceptual papers |
| **weak_or_none** | Editorials, letters, announcements, advocacy pieces, papers with no discernible methodology |

---

## 7. Domain and Category Assignment

### 7.1 Primary Domain

Assign based on the RESEARCH CONTENT, not the OpenAlex field:

| Domain | Content Signals |
|--------|----------------|
| **MAT** | Materials characterization, construction, concrete, ceramics, corrosion, manufacturing, machining, composites, building materials, roads, minerals processing |
| **AGR** | Crops, soil, farming, livestock, veterinary, fisheries, food processing, forestry, biodiversity, ecology, pest management, aquaculture |
| **HLT** | Medicinal plants, pharmacology, disease, infection, diagnostics, vaccines, genomics, nutrition, biomedical, public health, epidemiology, clinical studies |
| **ENR** | Petroleum, gas, solar, wind, hydro, geothermal, biofuels, batteries, hydrogen, cooking fuels, power systems, mining, reservoirs |
| **ENV** | Water treatment, groundwater, hydrology, climate, geology, pollution, remote sensing, coastal, wetlands, erosion, atmospheric science, toxicology |
| **TEC** | Computing, AI/ML, electronics, sensors, IoT, telecommunications, robotics, drones, GIS, embedded systems, instrumentation |
| **OUTSIDE_TAXONOMY** | Economics, policy, education, law, business, pure mathematics, pure physics (no application), social sciences, humanities |

**Key nuance**: A paper about "antimicrobial resistance in Nigerian dairy cows" is AGR
(veterinary/livestock), not HLT, because the primary subject is animal husbandry. A paper
about "antimicrobial resistance in Nigerian hospital patients" is HLT.

**Key nuance**: A paper about "cassava waste for bioplastic production" is MAT
(manufacturing/materials), not AGR, because the research question is about material
production, not about growing cassava.

**Key nuance**: A paper about "soot impact on solar panels in Rivers State, Nigeria"
is ENR (solar energy), not ENV (pollution), because the research question is about
energy system performance.

### 7.2 Category Codes

Assign 1-3 codes from the taxonomy. Primary code first. Only assign codes that are
directly supported by the paper's content. If excluded as OUTSIDE_TAXONOMY, use `[]`.

---

## 8. African Country Codes

Extract from BOTH:
1. `countries_json` — but ONLY include African country codes that are also reflected
   in the CONTENT (title/abstract/keywords). A "NG" in countries_json that comes solely
   from author affiliation should NOT be listed if the content has zero Nigerian reference.
2. Content mentions — any African country/region named in the title or abstract.

**Key nuance**: If a paper says "conducted at University of Lagos" but studies a generic
chemical reaction with no Nigerian materials or context, do NOT list "NG" in country codes.
The country codes should reflect where the RESEARCH SUBJECT is from, not where the authors sit.

Exception: If the study is explicitly about a Nigerian population (e.g., "1,013 Nigerian
children"), then NG is listed because the population IS the research subject.

---

## 9. Relevance Tags

Assign from this controlled vocabulary based on what makes the paper African-relevant:

| Tag | When to Use |
|-----|-------------|
| `african_material` | Studies African minerals, clays, plants, biomass, wastes, or physical materials |
| `african_organism` | Studies African plants, animals, pathogens, or microorganisms |
| `african_environment` | Studies African climate, geology, ecosystems, or environmental conditions |
| `african_dataset` | Uses data collected from African populations, sites, or systems |
| `african_population` | Studies African human or animal populations directly |
| `african_industrial_problem` | Addresses African manufacturing, infrastructure, or industrial challenges |
| `african_experimental_conditions` | Experiments conducted under African conditions (climate, resources, constraints) |
| `locally_developed_method` | Uses or validates indigenous/local techniques or knowledge |
| `local_resource` | Exploits locally available resources for a solution |

Use 0-4 tags per paper. Empty array `[]` for excluded papers with no African content.

---

## 10. Evidence Excerpt

Extract a SHORT phrase (10-30 words) from the title or abstract that demonstrates:
- For includes: the specific African element that makes it relevant
- For excludes: the generic/non-African nature of the content
- For reviews: the ambiguous element that makes it borderline

This should be a DIRECT QUOTE or close paraphrase from the metadata, not your interpretation.

---

## 11. Reason Statement

Write 1-2 sentences explaining WHY you made the decision. Structure:

- **For include**: "[What was studied] + [where/what African element] + [why it matters]"
  - Example: "Experimental phytochemical screening of 10 Nigerian medicinal plants with detailed constituent analysis directly relevant to Nigerian ethnomedicine."

- **For exclude**: "[What the paper actually is] + [why it fails]"
  - Example: "Generic global review on composting techniques with no African-specific materials, conditions, or data. African connection is solely through author affiliations."

- **For review**: "[What's African about it] + [what's insufficient/ambiguous]"
  - Example: "Survey-based study of Okra farming in Ebonyi State, Nigeria. African-centric but limited scientific depth (descriptive statistics only). Borderline case meriting review."

---

## 12. Worked Examples (Calibration)

### Example A: Clear Include (Score 20)
**Paper**: "Phytochemical constituents of some Nigerian medicinal plants"
- Title says "Nigerian" → centrality anchor
- Studies 10 specific plants used in Nigerian ethnomedicine → specificity=4
- Alkaloids, tannins, saponins, flavonoids assayed → depth=4 (experimental)
- Teaches which plants contain which compounds → knowledge=4
- Directly validates traditional medicine use → applicability=4
- **Decision**: include, HLT, [HLT-NAT-001]

### Example B: Clear Exclude (Affiliation Only)
**Paper**: "Waste Management through Composting: Challenges and Potentials"
- Generic global review of composting methods
- No African materials, conditions, or data in the abstract
- Authors at North-West University (SA) and Obafemi Awolowo University (NG)
- If you removed affiliations, you'd never know this involves Africa
- **Decision**: exclude, hard_exclusion="African ONLY because of author affiliation"

### Example C: Clear Exclude (Non-African Region)
**Paper**: "Proving that Traditional Knowledge Works: antibacterial activity of Northern Peruvian medicinal plants"
- Studies plants in Trujillo, PERU
- Funded by US institutions
- NG appears in countries_json (one co-author has Nigerian affiliation)
- Zero African content in title, abstract, or keywords
- **Decision**: exclude, hard_exclusion="Research about non-African regions (Peru)"

### Example D: Exclude Despite High Centrality (Non-STEM)
**Paper**: "Climate change and Africa" (Oxford Review of Economic Policy)
- Africa IS central (centrality=4)
- But: discusses emissions trading, private-sector adaptation, business environments
- Keywords: economics, business, government, market economy, commodity
- No experiments, no data, no scientific methodology
- **Decision**: exclude, hard_exclusion="Outside STEM domains (economics/policy)"

### Example E: Include with Lower Scores (Score 14)
**Paper**: "Cassava peel waste for bioplastic (PHA) production"
- Cassava is a major African crop (centrality=2, not in title as "African")
- Study conducted in UK, but cassava waste is the central material (specificity=2)
- Full experimental: fermentation, extraction, characterization (depth=4)
- Teaches PHA production from African biomass (knowledge=3)
- Directly applicable to African biorefinery development (applicability=3)
- **Decision**: include, MAT, [MAT-MAN-001, MAT-RES-001]

### Example F: Review (Borderline)
**Paper**: "Okra production, processing, marketing in Ebonyi State, Nigeria"
- Clearly Nigerian (centrality=3)
- Specific location (specificity=3)
- But: descriptive survey with only percentages and frequencies (depth=2)
- Some useful agricultural data (knowledge=2)
- Applicable to Nigerian agriculture (applicability=3)
- Total=13 → **review** (meets centrality but depth is borderline)

### Example G: Review (Global Study with African Subset)
**Paper**: "A global reference for human genetic variation" (1000 Genomes)
- Landmark study, high scientific value (depth=4, knowledge=4)
- Includes some African populations as subset of 26 global populations
- But Africa is NOT the focus; it's one part of a global survey
- No African institutions involved
- Centrality=1, total=10 → **review** (high value but not Africa-focused)

### Example H: Exclude (Pure Physics, Affiliation Only)
**Paper**: "Schrödinger equation solutions for Hulthén-Yukawa potential"
- Pure theoretical quantum mechanics
- No African content whatsoever
- Nigerian author at a Nigerian university
- **Decision**: exclude, hard_exclusion="African ONLY because of author affiliation"

### Example I: Include (African Infrastructure/Environment)
**Paper**: "Soot impact on solar panel performance in Rivers State, Nigeria"
- Specific Nigerian location (Rivers State)
- Studies illegal refinery soot (African industrial problem)
- Experimental: measures solar panel degradation (depth=4)
- Directly relevant to Nigerian energy infrastructure (applicability=4)
- **Decision**: include, ENR, [ENR-REN-002]

### Example J: Include (Veterinary/Agricultural)
**Paper**: "Brucellosis in cattle in northern Nigeria"
- Specific Nigerian region
- Studies disease prevalence in Nigerian livestock
- Serological testing with statistical analysis
- Directly applicable to Nigerian veterinary policy
- **Decision**: include, AGR, [AGR-LIV-002]

---

## 13. Common Patterns and Traps

### Pattern 1: The "Nigerian Author, Global Topic" Trap
~40% of papers in this benchmark are generic global reviews or standard methodology
papers written by researchers at Nigerian/South African universities. These are ALWAYS
excluded unless the content itself references African materials, conditions, or data.

**Detection**: If the abstract reads like it could have been written by anyone anywhere
(no African locations, materials, organisms, or conditions mentioned), it's affiliation-only.

### Pattern 2: The "African Crop Studied Abroad" Case
A paper studying cassava, plantain, or other African crops at a UK/European university
CAN be included if the crop is central and the findings are applicable to Africa.
The key is whether the RESEARCH SUBJECT is African, not where the lab is located.

### Pattern 3: The "Global Study with African Data" Spectrum
- 195-country study mentioning Africa once → exclude (incidental)
- 60-country study with 10+ African countries and Africa-specific findings → include
- 26-population study with African populations as subset → review
- Study specifically comparing African vs. non-African patterns → include

### Pattern 4: The "African Disease, Generic Method" Case
A paper using standard PCR to detect malaria parasites in Nigerian patients IS included
(the African population and disease are the subject). A paper developing a new PCR method
tested on generic lab strains with a Nigerian co-author is NOT.

### Pattern 5: The "Economics/Policy Paper About Africa" Case
Papers about African climate POLICY, African economic DEVELOPMENT, African governance,
or African education SYSTEMS are excluded as non-STEM, even if centrality=4.
Papers about African climate SCIENCE (temperature data, crop yield models, rainfall
patterns) ARE included.

### Pattern 6: The "Pure Science by African Authors" Case
Papers on quantum mechanics, pure mathematics, theoretical physics, or generic algorithm
design by African authors are excluded. There is no African-specific scientific content.
Exception: if the paper uses African data (e.g., "ionospheric studies using Nigerian
GPS stations") → review or include depending on depth.

---

## 14. Output Schema

For each paper, produce exactly:

```json
{
  "benchmark_id": "GOLD-XXX",
  "qwen_decision": "include | exclude | review",
  "qwen_hard_exclusion": "" | "specific reason",
  "qwen_african_centrality": 0-4,
  "qwen_local_specificity": 0-4,
  "qwen_scientific_depth": 0-4,
  "qwen_knowledge_value": 0-4,
  "qwen_local_applicability": 0-4,
  "qwen_total_score": <sum of above 5>,
  "qwen_african_focus": "essential | meaningful | incidental | absent",
  "qwen_scientific_evidence": "experimental | observational | computational | systematic_review | theoretical | weak_or_none",
  "qwen_african_country_codes": ["XX", ...],
  "qwen_african_relevance_tags": ["tag1", ...],
  "qwen_evidence": "short excerpt from paper metadata",
  "qwen_reason": "1-2 sentence justification",
  "qwen_primary_domain": "MAT | AGR | HLT | ENR | ENV | TEC | OUTSIDE_TAXONOMY",
  "qwen_category_codes": ["CODE-1", "CODE-2"]
}
```

---

## 15. Processing Order

For each paper, follow this exact sequence:

1. **Read** the title, abstract, keywords, and topics carefully.
2. **Identify** what the paper is actually about (the research question and subject).
3. **Check hard exclusions** first:
   - Is it non-STEM? → exclude
   - Is it about a non-African region? → exclude
   - Is it editorial/advocacy with no data? → exclude
   - Is the ONLY African signal the author affiliation? → exclude
4. **If not hard-excluded**, score all 5 dimensions.
5. **Apply threshold logic** (include/review/exclude).
6. **Apply override rules** (non-STEM with high centrality, etc.)
7. **Assign** focus level, evidence type, country codes, tags.
8. **Determine** domain and category codes from content.
9. **Write** evidence excerpt and reason.

---

## 16. Calibration Targets

When classifying the full 200-paper benchmark, expect approximately:
- **Include**: 70-80 papers (35-40%)
- **Exclude**: 90-100 papers (45-50%)
- **Review**: 25-35 papers (12-17%)

If your include rate is below 30%, you're being too strict on African relevance.
If your exclude rate is below 40%, you're letting too many affiliation-only papers through.

The most common exclusion reason (~50% of excludes) is "African ONLY because of author
affiliation" — generic global reviews and standard methodology papers by African authors.
