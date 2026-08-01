# MUFASA MAT Research Sources

## 1. Scope: What Counts as African Research

MUFASA is interested in **African research**, not merely research performed in Africa or written by African authors. An African component must be essential to the research question, experimental design, evidence, interpretation or intended application.

A record is eligible when at least one of the following is a critical consideration:

- African indigenous or locally available materials, minerals, ores, soils, clays, biomass, wastes, organisms or products
- African climatic, geological, ecological, disease, agricultural or environmental conditions
- African infrastructure, manufacturing, processing, maintenance or operating conditions
- African resource availability, supply-chain limitations, affordability constraints or import-substitution needs
- Indigenous African techniques, production methods, knowledge or locally developed technologies
- Performance, validation or adaptation specifically under conditions found in an identified African country or region
- A scientific problem whose conclusions or recommendations materially depend on African evidence

The paper should normally identify the relevant African country, region, material, condition or community. African authorship, institutional affiliation, study location or conference venue **alone is insufficient**.

Exclude:

- Generic research conducted at an African institution with no material African component
- Studies that use African participants, samples or locations merely as convenient data sources without examining an African-specific scientific condition
- Global reviews that mention Africa only incidentally
- Imported technologies tested without meaningful adaptation to African materials, conditions or constraints
- Social-science, business, management, legal, humanities and education-only research

For multidisciplinary journals and proceedings, screen individual papers against these rules; never include an entire venue automatically.

## 2. Extraction Schema

### 2.1 `paper_id`

A permanent internal identifier assigned by MUFASA. Use the format `MUFASA-000001`, increasing the number for each new paper. Every paper must have exactly one unique ID. Never reuse or change an ID after assignment.

### 2.2 `title`

The exact published title of the paper. Minor whitespace and encoding corrections are allowed. Do not shorten, translate, summarize or rewrite the title. If sources disagree, use the title from the official paper or repository record.

### 2.3 `abstract`

The paper's original abstract. Store it exactly as published, with only basic formatting cleanup. Use `null` if no abstract is available. Do not generate, summarize or infer an abstract using AI.

### 2.4 `authors`

An ordered list of the paper's authors.

```json
["Author One", "Author Two"]
```

Preserve the published author order. Do not combine all authors into one string, add affiliations to their names or invent missing names.

### 2.5 `publication`

Contains the publication year, document type and publication venue.

```json
{
  "year": 2024,
  "document_type": "journal_article",
  "venue": "African Journal of Engineering"
}
```

Allowed document types are:

```text
journal_article
conference_paper
thesis
dissertation
preprint
technical_report
government_report
book_chapter
dataset
patent
standard
manual
```

Use the official year and venue. Use `null` when information is unavailable instead of estimating it. A repository hosting the paper must not be recorded as the journal or publisher.

### 2.6 `persistent_id`

The best permanent external identifier for the paper. Prefer the DOI when available.

```text
doi:10.xxxx/example
openalex:W123456789
core:123456
handle:1234/5678
```

Do not use search-result links, temporary URLs or invented identifiers. Different copies of the same paper should normally share one record instead of receiving separate paper IDs.

### 2.7 `category_codes`

The MUFASA scientific categories assigned to the paper.

```json
["MAT-BLD-001", "MAT-RES-001"]
```

Put the primary category first and secondary categories afterwards. Use only codes defined in the MUFASA category files. One to five codes is usually sufficient. Do not assign unsupported categories or create new codes during paper collection.

### 2.8 `african_relevance`

Explains how the research is scientifically relevant to Africa.

```json
["NG", "african_material", "african_experimental_conditions"]
```

Use ISO country codes such as `NG`, `GH`, `KE` and `ZA`. Allowed relevance tags are:

```text
african_material
african_organism
african_environment
african_dataset
african_population
african_industrial_problem
african_experimental_conditions
locally_developed_method
local_resource
```

An African author or university affiliation alone is not enough. The research itself must examine an African resource, condition, environment, population, method or scientific problem.

### 2.9 `source_url`

The canonical landing page where the paper's record can be inspected. Use the publisher, journal, conference or institutional-repository page. Do not use search-engine result pages, temporary redirects or direct PDF links. Direct files belong in `download_urls`.

### 2.10 `download_urls`

A list of direct, lawful full-text download links.

```json
[
  "https://repository.example.edu/paper.pdf"
]
```

Publisher and authorized repository links are allowed. Multiple links may be stored when different legitimate copies exist. Do not include piracy sites, guessed URLs, inaccessible pages or links that only lead to an abstract.

### 2.11 `license_status`

Records the paper's licence and whether it can be used for model training. Use the format `LICENCE | PERMISSION`.

```text
CC-BY-4.0 | training_allowed
CC0-1.0 | training_allowed
unknown | review_required
all-rights-reserved | not_allowed
```

Allowed permission values are:

```text
training_allowed
review_required
not_allowed
```

Do not assume that free access means training is allowed. If the licence is missing or unclear, use `review_required`.

### 2.12 `ranking_metrics`

Contains the values used to rank and filter papers.

```json
{
  "citation_count": 42,
  "citation_percentile": 0.87,
  "relevance_score": 0.95,
  "source_quality": "peer_reviewed",
  "is_retracted": false,
  "metrics_date": "2026-07-12"
}
```

`citation_count` must be a non-negative integer obtained from a scholarly index.

`citation_percentile` must be between `0.0` and `1.0`. It is more useful than raw citations when comparing papers of different ages or scientific fields.

`relevance_score` must be between `0.0` and `1.0` and must follow one consistent MUFASA relevance rubric.

`source_quality` must use one of these controlled values:

```text
peer_reviewed
thesis
conference
technical_report
government_report
preprint
unknown
```

`is_retracted` must be `true` or `false`. Retracted papers must not enter the training corpus.

`metrics_date` must use the `YYYY-MM-DD` format because citation metrics change over time.

## 3. MAT Sources

These journals, events and organizations are discovery sources for African materials, manufacturing and infrastructure research. Inclusion still depends on paper-level eligibility under the scope above.

### 3.1 African Journals

1. Nigerian Journal of Materials Science and Engineering
2. Journal of Construction and Materials Technology
3. Nigerian Journal of Technology
4. Nigerian Journal of Technological Development
5. Arid Zone Journal of Engineering, Technology and Environment
6. FUOYE Journal of Engineering and Technology
7. FUTA Journal of Engineering and Engineering Technology
8. LAUTECH Journal of Engineering and Technology
9. Ghana Mining Journal
10. Journal of the Southern African Institute of Mining and Metallurgy
11. Journal of the South African Institution of Civil Engineering
12. Rwanda Journal of Engineering, Science, Technology and Environment
13. Botswana Journal of Technology
14. Ethiopian Journal of Science and Technology
15. African Journal of Science, Technology, Innovation and Development
16. African Journal of Environmental Science and Technology
17. African Journal of Pure and Applied Chemistry
18. Bulletin of the Chemical Society of Ethiopia

### 3.2 Conferences, Proceedings and Professional Bodies

Include professional or scientific societies headquartered in Africa, together with active African national councils, sections and chapters of international societies. An international parent organization's headquarters outside Africa does not disqualify its African body or African conference.

An African council, section or chapter qualifies when it conducts substantive scientific activity in Africa and produces discoverable African research through conferences, proceedings, journals or technical publications. Local membership or an African mailing address alone is insufficient.

1. Society of Petroleum Engineers Nigeria Council and its Nigerian sections — Nigeria Annual International Conference and Exhibition (NAICE)
2. American Chemical Society Nigeria International Chemical Sciences Chapter — Nigeria Annual Symposium
3. IEEE Nigeria Section — Nigeria International Conference on Electro-Computing Technologies (NIGERCON)
4. American Society of Civil Engineers Nigeria Section and its Nigerian branches — Sustainable Infrastructure Conference
5. Materials Science and Technology Society of Nigeria — Nigerian International Materials Congress (NIMACON)
6. Nigerian Building and Road Research Institute conferences
7. Nigerian Society of Engineers — International Engineering Conference
8. Nigerian Institution of Civil Engineers conferences
9. Nigerian Institution of Highway and Transportation Engineers conferences
10. Nigerian Institution of Structural Engineers conferences
11. Nigerian Institution of Geotechnical Engineers conferences
12. Nigerian Mining and Geosciences Society — Annual International Conference
13. Nigerian Society of Mining Engineers — International Conference
14. Nigerian Society of Chemical Engineers — Annual Conference
15. Nigerian Institution of Mechanical Engineers conferences
16. Nigerian Institution of Production Engineers conferences
17. Nigerian Corrosion Association — Nigerian corrosion conferences
18. Raw Materials Research and Development Council technical conferences and Techno-Expo
19. African Materials Research Society — International Conference
20. Southern African Institute of Mining and Metallurgy conferences
21. African Corrosion Congress
22. Ceramics and Geomaterials in Central Africa
23. Manufacturing Indaba
24. Mining Indaba
25. International Conference on Infrastructure Development in Africa
26. Nigerian Institution of Metallurgical, Mining and Materials Engineers
27. Nigerian Society of Economic Geologists
28. Nigerian Institution of Environmental Engineers
29. South African Institution of Civil Engineering
30. South African Institute of Materials Engineering
31. Corrosion Institute of Southern Africa
32. Composites and Advanced Ceramics Society
33. African Association of Automotive Manufacturers

Regulators, government institutes and standards bodies such as COMEG, COREN, NBRRI, RMRDC and the Standards Organisation of Nigeria are valuable technical sources, but they should not be labelled professional societies.

Do not automatically include all outputs from these bodies or events. Multidisciplinary proceedings must be screened paper by paper against the African-research scope test.

## 4. Discovery and Collection Platforms

1. OpenAlex
2. Crossref
3. CORE
4. Institutional and national research portals
5. Repository APIs and OAI-PMH feeds
6. DOI and citation trails
7. Approved repository crawling
8. Direct author, university and research-institute partnerships

Use discovery platforms to locate records, then collect full text only from an openly licensed or explicitly permitted source.
