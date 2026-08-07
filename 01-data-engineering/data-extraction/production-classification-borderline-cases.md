# Production Classification Borderline Cases

**Reviewed through:** `batch_00017.parquet`

This is a small, cumulative register of cases where manual analysis found the **current decision itself genuinely arguable or inconsistent**. It is not a list of papers whose decision is `review`.

## Maintenance rule

After analysing each new batch, add a paper only when the analysis independently identifies a real borderline decision. Do not add ordinary review cases, clear decisions or API failures.

| Batch | Paper | Current output | Why it is genuinely borderline |
|---:|---|---|---|
| 0 | [W7161769642](https://openalex.org/W7161769642) — Malaria KAP and prevalence in Ondo State | Exclude · HLT · 20 | It was manually excluded as self-report-only, but the recorded reasoning says malaria prevalence was measured with rapid diagnostic tests. That physical measurement may make exclusion too strict. |
| 14 | [W4408242987](https://openalex.org/W4408242987) — Global Diet Quality Score among Nigerian adults | Include · HLT · 18 | Nigerian nutritional evidence is central, but intake was collected through a Food Frequency Questionnaire. It sits directly on the boundary between nutritional epidemiology and self-report-only research. |
| 16 | [W4415428117](https://openalex.org/W4415428117) — Response rates in a Nigerian COVID-19 seroepidemiological survey | Include · HLT · 18 | The paper analyses participation and response-rate patterns rather than the biological serology results, so its inclusion may conflict with the self-report/social-research exclusion. |
| 16 | [W4406245591](https://openalex.org/W4406245591) — Excreta disposal and Lagos Lagoon | Include · ENV · 17 | The lagoon is a concrete Nigerian environmental subject, but the stated evidence relies on resident and waste-handler reports and self-reported illness rather than water-quality measurements. |
| 16 | [W4409464049](https://openalex.org/W4409464049) — Insecticide-treated-net utilisation in The Gambia | Include · HLT · 18 | Malaria relevance and Gambian geography are strong, but the outcome is utilisation reported through DHS survey data rather than a physical or biological measurement. |
| 16 | [W4409621812](https://openalex.org/W4409621812) — Acute gastrointestinal illness in four African countries | Include · HLT · 19 | The African epidemiological question is strong, but the illness incidence appears to come from population self-report rather than clinical confirmation. |
| 17 | [W4405723816](https://openalex.org/W4405723816) — Quantitative ethnobotany of the Afenmai people | Include · AGR · 19 | It documents 36 crop species, but its central evidence concerns people’s reported crop uses and preservation practices. This makes the boundary between agricultural science and self-report cultural research genuinely unclear. |

