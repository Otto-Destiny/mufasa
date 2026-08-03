import json
import os
import re

import sys
base_dir = 'C:/CodingWorld/Hackathons/AfricanDeepTechChallenge/MUFASA/01-data-engineering/data-extraction/openalex_ng_science_2000_2026/classification_output'
input_file = base_dir + '/sub_partitions_200/sub_0007.jsonl'
output_file = base_dir + '/results/result_0007.jsonl'

records = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Loaded {len(records)} records")

# Manual overrides for specific records that need individual attention
# Keyed by openalex_id suffix
manual_overrides = {
    # Record 1: rainwater toxic metals - no African mention
    "W7150997829": {"evidence_level": "absent", "scientific_depth": 3, "decision": "exclude"},
    # Record 9: cervical cancer screening Rivers State Nigeria - perceptions study
    "W7150192864": {"evidence_level": "direct", "african_centrality": 4, "local_specificity": 3,
                     "hard_exclusion": True, "hard_exclusion_reason": "awareness_perception_or_service_only",
                     "decision": "exclude", "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Rivers State, Nigeria",
                     "evidence": "three local government areas in Rivers State",
                     "inference_basis": "Explicit mention of Rivers State, Nigeria in conclusions",
                     "reason": "Study conducted in Rivers State, Nigeria examining healthcare professionals' perceptions of cervical cancer screening barriers. Hard exclusion as awareness/perception survey despite strong African setting."},
    # Record 10: fall armyworm Nigeria
    "W4412087424": {"evidence_level": "direct", "african_centrality": 4, "local_specificity": 2, "scientific_depth": 3, "knowledge_value": 3, "local_applicability": 4, "total_score": 16, "decision": "include",
                     "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"], "african_focus": "Nigeria, sub-Saharan Africa",
                     "reason": "Direct focus on fall armyworm in Nigerian maize with field and screenhouse trials using microbial inoculants and biochar. Strong experimental methodology with PCA analysis supports inclusion."},
    # Record 11: Niger Delta reservoir - correct country
    "W7149019008": {"african_country_codes": ["NG"], "african_focus": "Gabo Field, Niger Delta, Nigeria",
                     "african_relevance_tags": ["Nigeria"],
                     "scientific_depth": 3, "total_score": 15, "decision": "include",
                     "evidence": "Gabo Field, Niger Delta",
                     "reason": "Direct focus on Niger Delta reservoir using cognitive AI with Random Forest and deep learning. Strong predictive accuracy (R2>0.98) for porosity, permeability, saturation with actionable exploration recommendations."},
    # Record 13: TB fractional model - latent
    "W7149388696": {"evidence_level": "latent", "scientific_depth": 3, "decision": "review"},
    # Record 14: Cancer imaging Nigeria - framework paper
    "W7148663958": {"scientific_depth": 2, "total_score": 13, "decision": "review",
                     "reason": "Proposes national cancer imaging repository framework for Nigeria with 12 pilot hospitals across geopolitical zones. Strong policy relevance but limited empirical validation. Direct African focus with moderate scientific depth."},
    # Record 17: Artisanal mining Niger State
    "W7150914765": {"scientific_depth": 2, "total_score": 13, "decision": "review",
                     "african_focus": "Korokpa, Minna, Niger State, Nigeria",
                     "reason": "Direct study of artisanal gold mining in Niger State, Nigeria with metallurgical balances for mercury and gold. Quantifies Hg losses (34%) with environmental implications. Moderate scientific depth with high local applicability."},
    # Record 18: Baby milk heavy metals Nigeria
    "W7148626823": {"scientific_depth": 2, "total_score": 13, "decision": "review",
                     "reason": "Direct assessment of heavy metals in baby milk and infant formulae in Umuahia, Nigeria using standard analytical methods. Risk assessment identifies cadmium/chromium concerns. Moderate scientific depth with high public health relevance."},
    # Record 19: bimetallic nanoparticles - inherent via Archachatina marginata
    "W7148627639": {"evidence_level": "inherent", "scientific_depth": 3, "decision": "review",
                     "african_focus": "Archachatina marginata (African giant snail)",
                     "reason": "Uses chitosan from Archachatina marginata shells (African giant land snail) and Citrus sinensis for green synthesis of bimetallic nanoparticles. Strong characterization (XRD, TEM, FTIR) and antimicrobial testing. Inherent African relevance via organism."},
    # Record 23: nutrition knowledge commercial drivers Ondo
    "W7160124897": {"evidence_level": "direct", "african_centrality": 4, "local_specificity": 3,
                     "hard_exclusion": True, "hard_exclusion_reason": "awareness_perception_or_service_only",
                     "decision": "exclude", "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Ondo, Nigeria", "scientific_depth": 2,
                     "reason": "Cross-sectional study of nutrition knowledge and blood pressure among commercial drivers in Ondo, Nigeria. Hard exclusion as awareness/perception survey despite direct African setting and epidemiological data."},
    # Record 29: CRISPR in Africa
    "W7147438691": {"scientific_depth": 3, "total_score": 16, "decision": "include",
                     "reason": "Comprehensive review of CRISPR-Cas9 for Africa's highest-burden genetic diseases (sickle cell, thalassaemia, G6PD deficiency). Analyzes 80+ studies with African donor cells. Direct African focus with strong scientific depth and translational roadmap."},
    # Record 31: Cholera dynamics African setting
    "W7149466207": {"scientific_depth": 3, "total_score": 15, "decision": "include",
                     "african_focus": "Western and Mid-Eastern African regions",
                     "reason": "Mathematical model of cholera dynamics specifically targeting Western and Mid-Eastern African settings. Uses Homotopy Perturbation Method for simulation with vaccination, treatment, and environmental hygiene controls. Strong quantitative framework for African disease control."},
    # Record 33: Avocado seed oil Niger Delta
    "W7154508063": {"scientific_depth": 3, "total_score": 15, "decision": "include",
                     "african_focus": "Niger Delta Region, Nigeria",
                     "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "reason": "Evaluates avocado seed oil as sustainable flow improver for Niger Delta waxy crude oil. Benchmarks against commercial EVA with matched efficiency (7 degrees C pour point depression). Strong rheological analysis with direct industrial applicability."},
    # Record 34: West African dwarf crocodile
    "W7148596743": {"evidence_level": "inherent", "scientific_depth": 2, "total_score": 14, "decision": "include",
                     "african_focus": "West African dwarf crocodile, Delta and Edo States, Nigeria",
                     "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria", "West Africa"],
                     "reason": "Studies trade dynamics of the Vulnerable West African dwarf crocodile (Osteolaemus tetraspis, Africa-exclusive species) across Delta and Edo States, Nigeria. Documents 1,818 individuals traded with demographic skew indicating unsustainable exploitation."},
    # Record 35: dumpsite Ado Ekiti Nigeria
    "W7148564287": {"scientific_depth": 3, "total_score": 16, "decision": "include",
                     "reason": "Comprehensive hydro-geochemical, geospatial, and geophysical assessment of Ilokun dumpsite in Ado Ekiti, Nigeria. Uses electrical resistivity imaging, VES, and health risk models. Identifies extensive contamination with elevated hazard indices for children."},
    # Record 36: aluminium-citrate cell - Dovyalis caffra
    "W7148269655": {"evidence_level": "inherent", "scientific_depth": 2, "decision": "review",
                     "african_focus": "Dovyalis caffra (Kei apple, southern African fruit)",
                     "reason": "Uses Dovyalis caffra (Kei apple, native to southern Africa) for citric acid extraction to build aluminum-citrate ion cells. Characterizes nanoparticles and electrochemical performance. Inherent African relevance via plant species."},
    # Record 37: lignin flame retardant Nigerian military
    "W7148445163": {"scientific_depth": 2, "total_score": 13, "decision": "review",
                     "reason": "Develops phosphorus-modified lignin from coconut fiber as flame retardant for Nigerian military uniforms. FTIR/SEM characterization confirms phosphate bonding. PM-Lig shows 45 mm/s burning rate vs 127 mm/s untreated. Direct Nigerian application with moderate depth."},
    # Record 41: air pollution Calabar Nigeria - has AQI data beyond perceptions
    "W7148630850": {"evidence_level": "direct", "hard_exclusion": False, "hard_exclusion_reason": "",
                     "scientific_depth": 2, "total_score": 13, "decision": "review",
                     "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Watt Market, Calabar, Nigeria",
                     "reason": "Combines air quality index data with population exposure risk assessment in Watt Market, Calabar, Nigeria. Develops Population Exposure Risk Index metric. While surveys capture perceptions, the AQI and PERI quantitative analysis provides scientific depth beyond awareness-only studies."},
    # Record 44: caesarean section attitudes Lagos
    "W7147483379": {"evidence_level": "direct", "african_centrality": 4, "local_specificity": 3,
                     "hard_exclusion": True, "hard_exclusion_reason": "awareness_perception_or_service_only",
                     "decision": "exclude", "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Lagos, Nigeria",
                     "reason": "Cross-sectional survey of attitudes and subjective norms regarding caesarean section acceptance among married men and women in Lagos, Nigeria. Hard exclusion as awareness/perception survey despite direct African setting."},
    # Record 46: ANC model Nigeria
    "W7148313695": {"scientific_depth": 2, "total_score": 13, "decision": "review",
                     "reason": "Qualitative study exploring facilitators and barriers to eight-contact ANC model implementation in Nigerian healthcare facilities. Uses semi-structured interviews and focus groups. Direct African focus with health systems relevance but moderate scientific depth."},
    # Record 47: parks perception Delta State
    "W7148377387": {"evidence_level": "direct", "african_centrality": 4, "local_specificity": 3,
                     "hard_exclusion": True, "hard_exclusion_reason": "awareness_perception_or_service_only",
                     "decision": "exclude", "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Delta State, Nigeria",
                     "reason": "Cross-sectional survey of residents' perceptions of parks and green spaces in Delta State, Nigeria. Hard exclusion as awareness/perception survey despite direct African setting and SDG relevance."},
    # Record 48: dietary consumption menopause Lagos
    "W7148474628": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "reason": "Cross-sectional study of dietary patterns and menopausal symptom severity among 376 postmenopausal women in Lagos, Nigeria. Finds significant associations between Western/processed food intake and symptom severity. Direct African focus with epidemiological data."},
    # Record 49: Indigenous orthopaedics Jukun - humanities
    "W7148558185": {"evidence_level": "direct", "african_centrality": 4, "local_specificity": 3,
                     "hard_exclusion": True, "hard_exclusion_reason": "outside_scientific_scope",
                     "decision": "exclude", "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Wukari, Taraba State, Nigeria",
                     "reason": "Examines indigenous orthopaedic practices among the Jukun people using historical methodology (primary/secondary historical sources). Hard exclusion as humanities research despite direct Nigerian cultural and medical relevance."},
    # Record 53: cafeteria perception Nigerian university
    "W7147364026": {"evidence_level": "direct", "african_centrality": 4, "local_specificity": 3,
                     "hard_exclusion": True, "hard_exclusion_reason": "awareness_perception_or_service_only",
                     "decision": "exclude", "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Babcock University, Ogun State, Nigeria",
                     "reason": "Cross-sectional survey of student perceptions toward on-campus food services at Babcock University, Nigeria. Hard exclusion as awareness/perception survey despite direct African setting."},
    # Record 54: biochar yam Nigeria
    "W7147394824": {"scientific_depth": 3, "total_score": 16, "decision": "include",
                     "reason": "Two-year factorial field experiment evaluating cocoa pod husk biochar and poultry manure on yam yield in degraded Alfisol at Owo, southwestern Nigeria. Achieves 46.1 t/ha tuber yield with combined application. Strong experimental design with soil and yield data."},
    # Record 55: ogi fermentation Nigeria
    "W7147674380": {"scientific_depth": 3, "total_score": 15, "decision": "include",
                     "reason": "Molecular identification (16S rRNA, ITS sequencing) of LAB and yeasts from ogi, a traditional Nigerian fermented cereal food. Identifies Lactobacillus (70%), Candida (40%), and novel Trichomonascus ciferri. Strong microbiological characterization with starter culture potential."},
    # Record 56: Nso Aka conservation Nigeria - ethnobiology
    "W7154336880": {"evidence_level": "direct", "african_centrality": 4, "local_specificity": 3,
                     "hard_exclusion": True, "hard_exclusion_reason": "outside_scientific_scope",
                     "decision": "exclude", "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Ishiagu, Nigeria",
                     "reason": "Examines nso Aka cultural practice and traditional ecological knowledge among Ishiagu people using social-ecological memory framework. While relevant to conservation biology, the ethnographic/humanities methodology triggers hard exclusion."},
    # Record 58: lupeol antimalarial - latent P. falciparum
    "W7149450351": {"evidence_level": "latent", "scientific_depth": 3, "decision": "review",
                     "reason": "Computational docking of lupeol acetylsalicylate derivatives against P. falciparum Hsp90 and SERA6 targets. Opt94 shows -4.059 kcal/mol binding and -60.24 MMGBSA. Latent African relevance via malaria parasite but no explicit geographic mention."},
    # Record 59: breast cancer Ondo Nigeria
    "W7154949897": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "reason": "Retrospective analysis of 178 breast cancer cases at tertiary hospital in Akure, Ondo State, Nigeria. TNBC is most prevalent subtype (48.9%). IHC profiling of ER, PR, HER2 status. Direct African focus with clinical pathology data."},
    # Record 60: Pseudomonas Lagos
    "W7154961974": {"scientific_depth": 3, "total_score": 16, "decision": "include",
                     "reason": "Molecular detection of blaOXA-48, blaVIM resistance genes and oprL/toxA virulence genes in P. aeruginosa from Lagos hospitals. 550 clinical samples analyzed with real-time PCR. High carbapenemase prevalence (73.9%) with multidrug resistance patterns."},
    # Record 61: Maru Schist Belt Nigeria
    "W7147166544": {"scientific_depth": 3, "total_score": 15, "decision": "include",
                     "reason": "Integrates airborne magnetic and radiometric data with fuzzy logic for mineral prospectivity mapping in Maru Schist Belt, NW Nigeria. 2D forward modeling reveals shallow causative bodies. High-potential zones coincide with known gold deposits."},
    # Record 67: petroleum hydrocarbons Borno Nigeria
    "W7146976270": {"scientific_depth": 3, "total_score": 16, "decision": "include",
                     "reason": "GC-MS analysis of petroleum hydrocarbons in Alau Dam water, Borno State, Nigeria. PHC concentrations 0.001-0.457 mg/L with cumulative Health Index exceeding thresholds for children. Carcinogenic risks exceed USEPA limits at multiple sites."},
    # Record 68: 2022 floods Jigawa Nigeria
    "W4405966547": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "reason": "Geospatial profiling of 2022 flood impacts across 19 of 27 LGAs in Jigawa State, Nigeria using satellite imagery. Documents fatality locations, flood extents, and causes of death. Miga LGA recorded highest deaths (20). Direct African disaster assessment."},
    # Record 70: mpox knowledge Rivers State - awareness survey
    "W7164925314": {"evidence_level": "direct", "african_centrality": 4, "local_specificity": 3,
                     "hard_exclusion": True, "hard_exclusion_reason": "awareness_perception_or_service_only",
                     "decision": "exclude", "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Rivers State, Nigeria",
                     "reason": "Cross-sectional study of mpox knowledge, attitudes, and vaccine willingness among PLHIV and MSM in Rivers State, Nigeria. Hard exclusion as awareness/perception survey despite direct African setting and public health relevance."},
    # Record 74: SARS-CoV-2 genomic surveillance Guinea
    "W7147439182": {"scientific_depth": 3, "total_score": 17, "decision": "include",
                     "reason": "Establishes nanopore sequencing capacity at CRV-LFHVG in Conakry, Guinea. Generated 238 SARS-CoV-2 genomes representing 4 infection waves. Phylogeographic analysis reveals Delta and Omicron introductions. Strong genomic surveillance capacity building."},
    # Record 75: obstetric fistula Nigeria
    "W7146996004": {"scientific_depth": 3, "total_score": 17, "decision": "include",
                     "reason": "Causal mediation analysis of 5,496 fistula cases from 2024 Nigeria DHS. Geographic access barriers emerge as only significant mediator (OR=0.73). Uses counterfactual framework with doubly robust estimation. Direct African focus with rigorous methodology."},
    # Record 76: diabetes self-care Ogun State
    "W7147064139": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "reason": "Hermeneutic phenomenological study of lived experiences of 20 T2DM patients at two teaching hospitals in Ogun State, Nigeria. Identifies structural constraints shaping self-care and cumulative diabetes-related distress. Direct African clinical research."},
    # Record 77: PCOS knowledge Nigerian university - awareness survey
    "W7147128880": {"evidence_level": "direct", "african_centrality": 4, "local_specificity": 2,
                     "hard_exclusion": True, "hard_exclusion_reason": "awareness_perception_or_service_only",
                     "decision": "exclude", "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Nigeria",
                     "reason": "Assesses knowledge and awareness of PCOS among 413 female undergraduates at a Nigerian private university. Hard exclusion as awareness/knowledge survey despite direct African setting."},
    # Record 80: pheochromocytoma Nigerian case report
    "W7147336393": {"scientific_depth": 2, "total_score": 13, "decision": "review",
                     "reason": "Case report of pheochromocytoma in a 25-year-old Nigerian male with predominant norepinephrine secretion. Clinical presentation with resistant hypertension, hyperglycemia, and flank pain. Direct African case documentation with moderate depth."},
    # Record 81: sickle cell infections Zaria Nigeria
    "W7147359507": {"scientific_depth": 2, "total_score": 15, "decision": "include",
                     "reason": "Retrospective cohort of 1,961 paediatric SCD patients at ABUTH Zaria, Nigeria (1998-2023). 23.4% manifest infections with bacterial and parasitic organisms. Part of ARISE project for SCD management. Direct African clinical epidemiology."},
    # Record 83: asthma treatment Guinea - editorial
    "W7147472457": {"evidence_level": "direct", "african_centrality": 3, "local_specificity": 2,
                     "hard_exclusion": True, "hard_exclusion_reason": "editorial_advocacy_or_opinion",
                     "decision": "exclude", "african_country_codes": ["GN"], "african_relevance_tags": ["Guinea"],
                     "african_focus": "Guinea, sub-Saharan Africa",
                     "reason": "Editorial/letter commenting on Diallo et al.'s asthma management study from Guinea. Discusses GINA guidelines and ICS access gaps in LMICs. Hard exclusion as editorial despite relevant African health policy discussion."},
    # Record 85: lymphoma subtypes Nigerian facility
    "W7147636550": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "reason": "Retrospective study of 59 lymphoma cases at Nigerian tertiary facility (2019-2023). SLL/CLL is predominant NHL subtype (42.1%), mixed cellularity most common HL (47.6%). WHO classification applied. Direct African cancer epidemiology."},
    # Record 86: HIV treatment failure Nigeria
    "W7147705389": {"scientific_depth": 3, "total_score": 17, "decision": "include",
                     "reason": "Multi-centre study of 517 PLHIV at four Nigerian tertiary facilities. Only 26.9% achieve viral suppression. Drug resistance mutations (M184V 62%, K103N 54%) and CD8 T-cell activation analyzed. Strong mechanistic exploration of treatment failure."},
    # Record 87: COVID-19 Ogun State
    "W7147718158": {"scientific_depth": 2, "total_score": 15, "decision": "include",
                     "reason": "Clinical characteristics and survival analysis of 273 hospitalized COVID-19 patients in Ogun State, Nigeria. Comorbidities (AOR 9.5) and SpO2<94% (AOR 19.5) predict mortality. Kaplan-Meier survival analysis. Direct African COVID-19 clinical research."},
    # Record 92: TEAS/CATS validation Nigerian hospital
    "W7154940506": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "reason": "Psychometric validation of TEAS and CATS recovery scales among 201 SUD patients at Aminu Kano Teaching Hospital, Nigeria. TEAS alpha=0.74, CATS alpha=0.90 with PCA confirming unidimensional structures. Direct African clinical tool validation."},
    # Record 93: CCT and IOP rural eye camp Nigeria
    "W7154949600": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "reason": "Cross-sectional study of 146 eyes at mobile cataract eye camp in rural Nigeria. Examines effect of central corneal thickness on IOP measurements using iCare tonometer. Mean CCT 538.3um. Direct African ophthalmological screening research."},
    # Record 95: breast cancer anxiety Kano
    "W7154955688": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "reason": "Hospital-based study of 240 breast cancer patients in Kano, Nigeria. GAD prevalence 21.3% assessed via MINI-7. Coping strategies measured with Brief COPE not significantly associated with GAD. Direct African psycho-oncology research."},
    # Record 97: traditional eye medicines Nigeria
    "W7154970384": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "reason": "Retrospective study at UCTH Calabar, Nigeria documenting 48.5% TEM use among 33 corneal ulcer patients. Biological-based substances account for 95.7% of TEMs. Direct African clinical documentation of harmful traditional practices."},
    # Record 98: prostate cancer Nigerian facility
    "W7154977052": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "reason": "Retrospective analysis of 201 prostate cancer cases (36.5% incidence) at Nigerian tertiary facility. All adenocarcinomas, Gleason grade 3 predominant (54.2%). Median PSA 35.8 ng/ml. Positive family history associated with earlier onset."},
    # Record 99: HPV vaccination knowledge Nigerian university - awareness survey
    "W7154978666": {"evidence_level": "direct", "african_centrality": 4, "local_specificity": 3,
                     "hard_exclusion": True, "hard_exclusion_reason": "awareness_perception_or_service_only",
                     "decision": "exclude", "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Babcock University, Ogun State, Nigeria",
                     "reason": "Cross-sectional survey of HPV vaccination knowledge, perception, and uptake among 300 female undergraduates in Nigeria. Only 17.7% vaccinated. Hard exclusion as awareness/perception survey despite direct African setting."},
    # Record 100: psychoactive substance use Uyo
    "W7154981776": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "reason": "Comparative study of 844 adolescents in public/private schools in Uyo, Nigeria. Mixed-methods with WHO-adapted questionnaire. Lifetime alcohol use higher in public schools (4.7%). Family and peer influences identified as key predictors."},
    # Record 103: allergic conjunctivitis Kano
    "W7154987938": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "reason": "Facility-based study of 398 caregivers of children with allergic conjunctivitis in Kano, Nigeria. 65.3% had good knowledge but only 14.8% demonstrated good practices. Identifies knowledge-practice gap. Direct African pediatric ophthalmology research."},
    # Record 104: haematological indices pregnant women Delta State
    "W7154994429": {"scientific_depth": 2, "total_score": 15, "decision": "include",
                     "reason": "Comparative study of 242 pregnant and 80 non-pregnant women at Central Hospital Agbor, Delta State, Nigeria. Documents trimester-specific haematological changes. Anaemia prevalence 22.7%. Direct African obstetric haematology research."},
    # Record 105: placental thickness Uyo
    "W7154998959": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Uyo, Nigeria",
                     "reason": "Prospective study of 320 term pregnancies evaluating placental imaging biomarkers at tertiary hospital in Uyo, Nigeria. Abnormal thickness (AOR 7.85) and elevated umbilical RI (AOR 5.62) independently predict adverse neonatal outcomes."},
    # Record 106: refractive error Bauchi
    "W7155010281": {"scientific_depth": 2, "total_score": 15, "decision": "include",
                     "reason": "Cross-sectional study of 701 students using RESC protocol in Katagum LGA, Bauchi State, Nigeria. 11.7% refractive error prevalence with myopia 63.4%. Spectacle correction achieved 96.9% 6/6 vision. Direct African school eye health research."},
    # Record 108: yellow fever Ghana
    "W7155561326": {"scientific_depth": 3, "total_score": 17, "decision": "include",
                     "reason": "Comprehensive outbreak investigation of 2021 yellow fever in Wa East District, Ghana. 12 confirmed cases, 33.3% CFR. Entomological indices exceed WHO thresholds. Vaccination coverage increased from 25% to 95% post-campaign. Strong One Health approach."},
    # Record 111: postnatal care Somalia
    "W7156635641": {"scientific_depth": 2, "total_score": 15, "decision": "include",
                     "reason": "Analysis of 2020 Somalia DHS data from 2,813 women. Only 3% received appropriate-quality postnatal care. Husband's education and ANC attendance strongly associated with quality PNC. Direct African maternal health research with policy implications."},
    # Record 114: Aframomum melegueta diabetes
    "W7147510998": {"evidence_level": "inherent", "scientific_depth": 3, "total_score": 14, "decision": "include",
                     "african_focus": "Aframomum melegueta (West African spice)",
                     "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "reason": "Tests Aframomum melegueta (grains of paradise, West African spice) seed extract in HFD-STZ diabetic rats. 300mg/kg dose reduces FBG comparably to metformin and achieves 39% greater weight loss. Inherent African relevance via indigenous medicinal plant."},
    # Record 116: dental barriers Akure Nigeria
    "W7154730630": {"scientific_depth": 2, "total_score": 13, "decision": "review",
                     "reason": "Cross-sectional survey of 480 adults in Akure, Ondo State, Nigeria. Only 27.9% utilized dental services in past 12 months. High cost (73.1%), fear (62.1%) identified as barriers. Direct African dental health research but perception-heavy."},
    # Record 117: early childhood caries Nigeria
    "W7154976697": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Nigeria (NAIJA-ECOHIS)",
                     "reason": "Prospective cohort of 120 Nigerian preschoolers with ECC using validated NAIJA-ECOHIS tool. Mean scores decreased from 12.42 to 0.79 post-treatment (p<0.001). Dental treatment significantly improved QoL. Direct African pediatric dental research."},
    # Record 118: sleep quality healthcare workers Nigeria
    "W7147078904": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "reason": "Cross-sectional study of 246 healthcare workers at Babcock University Teaching Hospital, Nigeria. 69.5% had poor sleep quality (PSQI), 48.4% mental distress (GHQ-12). No significant association found. Direct African occupational health research."},
    # Record 119: physiotherapy practice Ogun State
    "W7147390288": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "reason": "Quasi-experimental study comparing training alone vs training with support tools among 60 physiotherapists in Ogun State, Nigeria. Support tools significantly improved knowledge, motivation, and utilization at 6-week follow-up (p<0.001). Direct African health professions research."},
    # Record 120: HIV awareness Somaliland
    "W7152611815": {"evidence_level": "direct", "african_centrality": 4, "local_specificity": 3,
                     "hard_exclusion": True, "hard_exclusion_reason": "awareness_perception_or_service_only",
                     "decision": "exclude", "african_country_codes": ["SO"], "african_relevance_tags": ["Somalia/Somaliland"],
                     "african_focus": "Sanaag and Sool, Somaliland",
                     "reason": "Cross-sectional analysis of 2020 Somaliland DHS identifying HIV/AIDS awareness predictors among women. 38.8% and 26.6% had no awareness in Sool and Sanaag. Hard exclusion as awareness survey despite direct African setting."},
    # Record 121: School Health Program Cross River State
    "W7154959524": {"scientific_depth": 2, "total_score": 13, "decision": "review",
                     "reason": "Qualitative study of School Health Program implementation in 56 secondary schools in Cross River State, Nigeria. Uses Socio-Ecological Model with 6 FGDs. Identifies multi-level barriers. Direct African public health implementation research."},
    # Record 122: obesity comorbidities Uyo Nigeria
    "W7154986878": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "reason": "Cross-sectional study of 385 overweight/obese adults at Uyo Teaching Hospital, Nigeria. Abnormal LDL (66.3%), hypertension (58.2%) most common comorbidities. Only osteoarthritis significantly associated. Direct African NCD epidemiology."},
    # Record 123: job stress knowledge Cross River - awareness survey
    "W7154993540": {"evidence_level": "direct", "african_centrality": 4, "local_specificity": 3,
                     "hard_exclusion": True, "hard_exclusion_reason": "awareness_perception_or_service_only",
                     "decision": "exclude", "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Cross River State, Nigeria",
                     "reason": "Cross-sectional survey of 422 healthcare professionals' knowledge of job stress and coping in Cross River State, Nigeria. Hard exclusion as awareness/perception survey despite direct African setting."},
    # Record 124: ionizing radiation Bayelsa
    "W7168129945": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "reason": "Environmental radiation assessment of scrap market in Yenagoa, Bayelsa State, Nigeria using Radalert-100X. Mean absorbed dose 154.28 nGy/h exceeds global average but annual effective dose within limits. Direct African environmental monitoring."},
    # Record 125: biochar compost Ibadan
    "W7147070635": {"scientific_depth": 3, "total_score": 16, "decision": "include",
                     "reason": "Screen house experiment evaluating residual effects of biochar and compost on Alfisol nutrient dynamics over two tomato cycles at Ibadan, Nigeria. Biochar increased P by 113% and K by 406%. pH shifted from 5.2 to 7.6-7.8. Strong soil science methodology."},
    # Record 126: extension workers Kwara - borderline
    "W7151423866": {"scientific_depth": 1, "total_score": 11, "decision": "review",
                     "reason": "Assesses competency needs of 154 agricultural extension workers in Kwara State, Nigeria using Borich model. Organic matter amendments ranked highest need. Limited scientific depth as survey-based assessment but direct African agricultural extension focus."},
    # Record 127: mycorrhizal strains SW Nigeria
    "W7151490836": {"scientific_depth": 2, "total_score": 15, "decision": "include",
                     "reason": "Pot trial experiment evaluating Glomus species on phosphorus availability across six soil series in SW Nigeria. Apomu series with G. fasculatum achieved highest infectivity (57.42%). Iwo series with G. fasculatum highest P uptake. Direct African soil microbiology."},
    # Record 128: cholera Lagos One Health
    "W7143800629": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "reason": "Systematic review of cholera outbreaks in Lagos through One Health lens. Documents 401 cases and 21 deaths in 2024. Identifies WASH infrastructure gaps as primary driver. Direct African infectious disease surveillance with integrated framework."},
    # Record 129: genome editing communication Nigeria
    "W7155013464": {"evidence_level": "direct", "african_centrality": 3, "local_specificity": 2,
                     "hard_exclusion": True, "hard_exclusion_reason": "outside_scientific_scope",
                     "decision": "exclude", "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Nigeria, Africa",
                     "reason": "Proposes communication framework for genome editing technologies in Nigeria. While topic is scientifically relevant, the study focuses on science communication strategy rather than scientific research. Hard exclusion as outside scientific scope."},
    # Record 130-133: Library science papers
    "W7143626539": {"hard_exclusion": True, "hard_exclusion_reason": "outside_scientific_scope", "decision": "exclude"},
    "W7143658907": {"hard_exclusion": True, "hard_exclusion_reason": "outside_scientific_scope", "decision": "exclude"},
    "W7143713029": {"hard_exclusion": True, "hard_exclusion_reason": "outside_scientific_scope", "decision": "exclude"},
    "W7143920995": {"hard_exclusion": True, "hard_exclusion_reason": "outside_scientific_scope", "decision": "exclude"},
    # Record 134: AI anthropomorphism Nigerian libraries
    "W7144154073": {"evidence_level": "direct", "african_centrality": 3, "local_specificity": 2,
                     "hard_exclusion": True, "hard_exclusion_reason": "outside_scientific_scope",
                     "decision": "exclude", "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Nigeria's six geopolitical zones",
                     "reason": "Qualitative study of AI anthropomorphism in Nigerian academic libraries. While AI topic is relevant, library science falls outside scientific scope for MUFASA classification. Hard exclusion despite direct Nigerian focus."},
    # Record 136: sustainable innovation banks Nigeria
    "W7154981518": {"evidence_level": "direct", "african_centrality": 3, "local_specificity": 2,
                     "hard_exclusion": True, "hard_exclusion_reason": "outside_scientific_scope",
                     "decision": "exclude", "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Lagos, Nigeria",
                     "reason": "Examines sustainable innovation and business growth in Nigerian banks using PLS-SEM. Economics/business research outside scientific scope. Hard exclusion despite direct Nigerian setting."},
    # Record 137: water quality Dass Bauchi
    "W7151321477": {"scientific_depth": 3, "total_score": 16, "decision": "include",
                     "reason": "Comprehensive groundwater quality assessment in Dass, Bauchi State, Nigeria. 50 samples with heavy metal analysis. PCA identifies anthropogenic, geogenic, and lithological sources. Carcinogenic risk exceeds thresholds by two orders of magnitude."},
    # Record 139: medicine disposal 3 sub-Saharan countries
    "W7144034733": {"evidence_level": "direct", "african_centrality": 4, "local_specificity": 2,
                     "hard_exclusion": True, "hard_exclusion_reason": "awareness_perception_or_service_only",
                     "decision": "exclude", "african_country_codes": ["ET", "NG", "TZ"], "african_relevance_tags": ["Ethiopia", "Nigeria", "Tanzania"],
                     "african_focus": "Ethiopia, Nigeria, Tanzania",
                     "reason": "Cross-sectional survey of knowledge, risk perception, and disposal practices of expired medicines among 575 Gen Z across Ethiopia, Nigeria, Tanzania. Hard exclusion as awareness/perception survey despite multi-country African focus."},
    # Record 140: PCB contamination Enugu
    "W7147045821": {"scientific_depth": 3, "total_score": 16, "decision": "include",
                     "reason": "GC-MS analysis of 28 PCB congeners in leachates from Ugwuaji dumpsite, Enugu State, Nigeria. Total PCBs 57.39-82.11 ng/L with PCB-126/169 driving toxicity. Lower-chlorinated PCBs predominate. Direct African environmental contamination study."},
    # Record 141: heavy metals vegetables Kaduna
    "W7147096401": {"scientific_depth": 2, "total_score": 14, "decision": "include",
                     "reason": "ED-XRF analysis of heavy metals in vegetables from five Kaduna markets, Nigeria. HQ and HI values exceed 1 for most vegetables. Moderate to high carcinogenic risk for Cr, Cu, Ni. Direct African food safety assessment."},
    # Record 142: urinary schistosomiasis Kwara
    "W7144193296": {"scientific_depth": 2, "total_score": 15, "decision": "include",
                     "reason": "Parasitological survey of 382 school pupils in Kaiama LGA, Kwara State, Nigeria. 19.4% S. haematobium prevalence, highest in 5-7 age group (25.8%). Boys more affected (21.7%). Direct African neglected tropical disease surveillance."},
    # Record 146: antibiotic catheter practice Nigerian urologists
    "W7143480633": {"scientific_depth": 2, "total_score": 13, "decision": "review",
                     "reason": "Survey of 55 Nigerian urologists at NAUS conference regarding antibiotic prescribing for catheter changes. 69% routinely prescribe, quinolones most common (65.8%). Mean duration 7.87 days. Direct African antimicrobial stewardship data."},
    # Record 147: Ocimum gratissimum diabetic rats
    "W7143596422": {"evidence_level": "latent", "scientific_depth": 3, "decision": "review",
                     "reason": "Evaluates flavonoid-rich O. gratissimum extract on KIM-1/TGF-beta1 pathway in STZ-diabetic rat kidneys. Significant improvement in redox balance, inflammation, and renal function. Latent African relevance via widely-used African medicinal plant."},
    # Record 148: placental morphometry HIV Uyo
    "W7147440821": {"scientific_depth": 2, "total_score": 15, "decision": "include",
                     "reason": "Case-control study of 48 HIV+ and 96 HIV- placentas in Uyo, Southern Nigeria. HIV+ placentas significantly lighter with shorter cords and more cord discolouration. Direct African obstetric pathology research with clinical implications."},
    # Record 149: nutrient intake adolescents Ogun
    "W7147379416": {"scientific_depth": 2, "total_score": 15, "decision": "include",
                     "reason": "Cross-sectional study of 206 adolescent girls in Ogun State, Nigeria. Calcium universally deficient. Vitamin E, C, iron, folate adequacy declined with age. Maternal occupation associated with micronutrient intake. Direct African nutritional epidemiology."},
    # Record 150: dietary diversity under-5 Ogun
    "W7147208551": {"scientific_depth": 2, "total_score": 15, "decision": "include",
                     "reason": "Cross-sectional study of 226 under-5 children in Ado-Odo/Ota LGA, Ogun State, Nigeria. 69.3% had low dietary diversity. Stunting 14.2%, significantly associated with DDS (p=0.027). Direct African child nutrition research."},
    # Record 152: yam crop improvement - latent
    "W7143447938": {"evidence_level": "latent", "scientific_depth": 2, "decision": "review",
                     "reason": "Comprehensive review of yam (Dioscorea spp.) flowering, dormancy, yield, and food quality for crop improvement. Yam is a critical West African staple but no explicit African mention. Latent relevance via crop's African importance."},
    # Record 159: AI entrepreneurial Osun - economics
    "W7151287629": {"evidence_level": "direct", "african_centrality": 3, "local_specificity": 2,
                     "hard_exclusion": True, "hard_exclusion_reason": "outside_scientific_scope",
                     "decision": "exclude", "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Osun State, Nigeria",
                     "reason": "Examines AI in entrepreneurial decision-making among startups in Osun State, Nigeria. Business/entrepreneurship research outside scientific scope. Hard exclusion despite direct Nigerian setting."},
    # Record 165: NiO nanoparticles Croton macrostachyus
    "W7143474299": {"evidence_level": "inherent", "scientific_depth": 3, "decision": "review",
                     "african_focus": "Croton macrostachyus (African native plant)",
                     "reason": "Green synthesis of NiO nanoparticles using Croton macrostachyus leaf extract (native to tropical Africa) hybridized with carbon dots for antimicrobial applications. Inherent African relevance via plant species. Strong characterization methodology."},
    # Record 166: PrEP uptake global - mentions sub-Saharan Africa
    "W7143426336": {"evidence_level": "latent", "scientific_depth": 2, "decision": "review",
                     "reason": "Global review of PrEP barriers/facilitators with evidence from sub-Saharan Africa, Asia, Europe, Americas. Discusses community-led strategies for youth/rural populations. Latent African relevance but not Africa-specific study."},
    # Record 168: MMS Pakistan and Nigeria
    "W7143334134": {"scientific_depth": 2, "total_score": 15, "decision": "include",
                     "reason": "Mixed-methods study of MMS acceptability and adherence in Bauchi State, Nigeria and Pakistan. >97% women started MMS, >70% adherence. Trusting PW-HCP relationships facilitate provision. Direct African maternal nutrition implementation research."},
    # Record 169: AYA HIV Owerri Nigeria
    "W7143375615": {"scientific_depth": 2, "total_score": 15, "decision": "include",
                     "reason": "Cross-sectional study of 92 adolescents/young adults with HIV at Federal Teaching Hospital Owerri, Nigeria. 92.4% ART adherence, 96.4% viral suppression. Only 13% disclosed status. Direct African HIV clinical research with psychosocial insights."},
    # Record 170: Ocimum gratissimum liver diabetic rats
    "W7143431444": {"evidence_level": "latent", "scientific_depth": 3, "decision": "review",
                     "reason": "Evaluates Nrf-2/HO-1 modulation by O. gratissimum flavonoid extract in STZ-diabetic rat livers. Significant reduction in DNA fragmentation, lipid peroxidation. Increased Nrf2/HO-1 gene expression. Latent African relevance via medicinal plant."},
    # Record 172: POC creatinine malaria AKI Nigeria
    "W7143531877": {"scientific_depth": 3, "total_score": 17, "decision": "include",
                     "reason": "Point-of-care creatinine testing at Nigerian primary health center. 56.9% of children with malaria had AKI (20.3% stage 1, 21.2% stage 2, 15.4% stage 3). Highlights need for KDIGO integration in WHO malaria guidelines. Strong African clinical research."},
    # Record 174: climate trend Niger Delta
    "W7142753881": {"scientific_depth": 3, "total_score": 16, "decision": "include",
                     "reason": "Integrates 40-year CHIRPS/ERA5-Land remote sensing with household survey in Ughelli, Delta State, Nigeria. Significant warming trend (0.018C/year, p<0.001) with variable rainfall. Links long-term climate data to household socioeconomic stress."},
    # Record 175: sweet potato spoilage Kebbi
    "W7142825131": {"scientific_depth": 2, "total_score": 13, "decision": "review",
                     "reason": "Microbial susceptibility comparison of sweet potato varieties across Birnin Kebbi and Maiyama LGAs, Kebbi State, Nigeria. Identifies mold and bacterial spoilage organisms. Direct African agricultural microbiology with limited abstract detail."},
    # Record 180: protozoan parasites Nigerian children
    "W7143000863": {"scientific_depth": 3, "total_score": 17, "decision": "include",
                     "reason": "Real-time PCR on 977 stool samples from children across 10 Nigerian states. Giardia 77.6%, Cryptosporidium 18.1%, Entamoeba 12.3%. Jigawa highest coinfection burden (7.9%). Strong molecular diagnostics with WASH implications."},
    # Record 181: TIA knowledge Nigerian physicians - awareness survey
    "W7141791618": {"evidence_level": "direct", "african_centrality": 4, "local_specificity": 2,
                     "hard_exclusion": True, "hard_exclusion_reason": "awareness_perception_or_service_only",
                     "decision": "exclude", "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Nigeria (national survey)",
                     "reason": "National cross-sectional survey of 404 Nigerian physicians' TIA management knowledge. Only 38.1% had good knowledge. Hard exclusion as awareness/knowledge survey despite direct African clinical setting."},
    # Record 183: cyberchondria Cross River State
    "W7142938604": {"evidence_level": "direct", "african_centrality": 4, "local_specificity": 3,
                     "hard_exclusion": True, "hard_exclusion_reason": "awareness_perception_or_service_only",
                     "decision": "exclude", "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Cross River State, Nigeria",
                     "reason": "Cross-sectional study of cyberchondria prevalence (50.4%) among 400 undergraduates in Cross River State, Nigeria. Hard exclusion as awareness/perception survey despite direct African setting."},
    # Record 185: lightning southern Africa
    "W7142710024": {"scientific_depth": 3, "total_score": 14, "decision": "include",
                     "reason": "Statistical technique for obtaining realistic flash rate densities from WWLLN data over southern Africa using satellite-based lightning observations. Validates detection efficiency models. Direct African atmospheric science with strong methodology."},
    # Record 186: GNSS Mthatha South Africa
    "W7142728018": {"scientific_depth": 3, "total_score": 15, "decision": "include",
                     "reason": "Commissioning results from low-cost GNSS receiver at Walter Sisulu University, Mthatha, South Africa. Fills 340km observational gap at SAMA periphery. Two-week continuous ionospheric TEC monitoring. Direct African space science infrastructure."},
    # Record 188: Google Meet/Zoom poultry farmers Abuja
    "W7141895538": {"scientific_depth": 1, "total_score": 11, "decision": "review",
                     "reason": "Literature review comparing Google Meet and Zoom for poultry advisory delivery in FCT Abuja, Nigeria. Finds Google Meet more contextually appropriate for routine advisory. Limited scientific depth as literature-based comparison but direct African agricultural focus."},
    # Record 193: crude oil pipeline prediction - no African mention
    "W7161660203": {"evidence_level": "absent", "decision": "exclude"},
    # Record 196: hypertension knowledge FUTO Nigeria - awareness survey
    "W7142025422": {"evidence_level": "direct", "african_centrality": 4, "local_specificity": 3,
                     "hard_exclusion": True, "hard_exclusion_reason": "awareness_perception_or_service_only",
                     "decision": "exclude", "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Owerri, Imo State, Nigeria",
                     "reason": "Cross-sectional survey of 333 academic staff at FUTO regarding hypertension knowledge and perception. Hard exclusion as awareness/perception survey despite direct Nigerian setting."},
    # Record 197: iron deficiency blood donors Ghana
    "W7142252623": {"scientific_depth": 2, "total_score": 15, "decision": "include",
                     "african_country_codes": ["GH"], "african_relevance_tags": ["Ghana"],
                     "african_focus": "Tamale, Ghana",
                     "reason": "Cross-sectional study of 252 blood donors at Tamale Teaching Hospital, Ghana. 40.1% had ferritin <30 ng/mL, 14.7% iron deficiency. Repeat donors and females at higher risk. Direct African transfusion medicine research."},
    # Record 199: health emergency mitigation Bayelsa
    "W7141910789": {"scientific_depth": 2, "total_score": 13, "decision": "review",
                     "reason": "Assessment of health emergency mitigation strategies across primary/secondary/tertiary facilities in Bayelsa State, Nigeria. 735 healthcare workers surveyed. HIV PEP availability highest in tertiary (64.4%). Direct African health systems research."},
    # Record 200: emergency preparedness Bayelsa - awareness survey
    "W7142481352": {"evidence_level": "direct", "african_centrality": 4, "local_specificity": 3,
                     "hard_exclusion": True, "hard_exclusion_reason": "awareness_perception_or_service_only",
                     "decision": "exclude", "african_country_codes": ["NG"], "african_relevance_tags": ["Nigeria"],
                     "african_focus": "Bayelsa State, Nigeria",
                     "reason": "Assesses knowledge and attitude of 735 health workers toward emergency preparedness in Bayelsa State, Nigeria. Hard exclusion as awareness/knowledge survey despite direct African setting."},
}

def classify(rec):
    oid = rec.get('openalex_id','')
    title = rec.get('title','')
    abstract = rec.get('abstract','')
    field = rec.get('field_name','')
    topic = rec.get('primary_topic','')
    work_type = rec.get('work_type','')
    keywords = rec.get('keywords',[])

    text = (title + ' ' + abstract).lower()

    r = {
        'openalex_id': oid,
        'title': title,
        'decision': 'exclude',
        'hard_exclusion': False,
        'hard_exclusion_reason': '',
        'evidence_level': 'absent',
        'african_centrality': 0,
        'local_specificity': 0,
        'scientific_depth': 0,
        'knowledge_value': 0,
        'local_applicability': 0,
        'total_score': 0,
        'african_focus': '',
        'scientific_evidence': '',
        'african_country_codes': [],
        'african_relevance_tags': [],
        'evidence': '',
        'inference_basis': '',
        'reason': '',
        'field_name': field,
        'primary_topic': topic,
        'work_type': work_type
    }

    # Get short ID for override lookup
    short_id = oid.split('/')[-1] if '/' in oid else oid

    # Always run geography detection (even for overrides)
    african_map = {
        'nigeria': 'NG', 'ghana': 'GH', 'kenya': 'KE', 'south africa': 'ZA',
        'ethiopia': 'ET', 'tanzania': 'TZ', 'uganda': 'UG', 'cameroon': 'CM',
        'senegal': 'SN', 'mali': 'ML', 'niger': 'NE', 'chad': 'TD',
        'somalia': 'SO', 'somaliland': 'SO', 'guinea': 'GN', 'congo': 'CD',
        'rwanda': 'RW', 'burundi': 'BI', 'zambia': 'ZM', 'zimbabwe': 'ZW',
        'mozambique': 'MZ', 'angola': 'AO', 'malawi': 'MW', 'madagascar': 'MG',
        'botswana': 'BW', 'namibia': 'NA', 'gabon': 'GA', 'togo': 'TG',
        'benin': 'BJ', 'burkina faso': 'BF', 'liberia': 'LR',
        'sierra leone': 'SL', 'gambia': 'GM', 'mauritania': 'MR',
        'sudan': 'SD', 'south sudan': 'SS', 'eritrea': 'ER', 'djibouti': 'DJ',
        'morocco': 'MA', 'algeria': 'DZ', 'tunisia': 'TN', 'libya': 'LY',
        'egypt': 'EG', 'africa': 'AFRICA', 'african': 'AFRICA',
        'sub-saharan': 'SSA', 'west africa': 'WEST_AFRICA',
        'east africa': 'EAST_AFRICA', 'southern africa': 'SOUTHERN_AFRICA',
        'central africa': 'CENTRAL_AFRICA', 'north africa': 'NORTH_AFRICA',
    }
    sorted_countries = sorted(african_map.keys(), key=len, reverse=True)
    found = []
    codes = []
    for country in sorted_countries:
        if country in text:
            if country == 'niger':
                if 'nigeria' in text or 'niger delta' in text:
                    continue
            if country == 'mali':
                if not re.search(r'\bmali\b', text):
                    continue
            if country == 'guinea':
                if 'guinea-bissau' in text or 'equatorial guinea' in text:
                    continue
            found.append(country)
            code = african_map[country]
            if code not in codes:
                codes.append(code)
    if 'niger delta' in text:
        if 'NG' not in codes: codes.append('NG')
        if 'niger delta' not in found: found.append('niger delta')
    inherent = False
    if 'aframomum melegueta' in text: inherent = True
    if 'archachatina marginata' in text: inherent = True
    if 'croton macrostachyus' in text: inherent = True
    if 'dovyalis caffra' in text or 'kei apple' in text: inherent = True
    if 'west african dwarf crocodile' in text or 'osteolaemus tetraspis' in text: inherent = True
    if re.search(r'\bogi\b.*ferment', text) or re.search(r'ferment.*\bogi\b', text): inherent = True

    # Determine auto evidence_level
    auto_ev = 'absent'
    if found: auto_ev = 'direct'
    elif inherent: auto_ev = 'inherent'
    elif any(w in text for w in ['tropical', 'subtropical', 'low-resource', 'lmic']): auto_ev = 'latent'
    if auto_ev in ['absent', 'latent'] and not inherent:
        non_african = ['china', 'korea', 'japan', 'brazil', 'united states', 'usa',
                       'canada', 'hungary', 'denmark', 'italy', 'indonesia',
                       'saudi arabia', 'malaysia', 'south carolina', 'mexico',
                       'india', 'australia', 'france', 'germany', 'spain',
                       'turkey', 'iran', 'pakistan']
        for na in non_african:
            if na in text:
                auto_ev = 'contradicted'
                break

    # Check manual override
    if short_id in manual_overrides:
        override = manual_overrides[short_id]
        # Apply override fields
        for k, v in override.items():
            r[k] = v
        # If override doesn't set evidence_level, use auto-detected
        if 'evidence_level' not in override:
            r['evidence_level'] = auto_ev
        # If override doesn't set country codes, use auto-detected
        if 'african_country_codes' not in override:
            r['african_country_codes'] = codes
        if 'african_focus' not in override and found:
            r['african_focus'] = ', '.join(found[:3])
        elif 'african_focus' not in override and inherent:
            r['african_focus'] = 'Africa-relevant entity'
        # Set african_centrality if not overridden
        if 'african_centrality' not in override:
            if r['evidence_level'] == 'direct': r['african_centrality'] = 4
            elif r['evidence_level'] == 'inherent': r['african_centrality'] = 3
            elif r['evidence_level'] == 'latent': r['african_centrality'] = 1
        if 'local_specificity' not in override:
            if r['african_centrality'] >= 3:
                r['local_specificity'] = 3 if any(w in text for w in ['state', 'district', 'local government', 'city', 'village']) else 2
            elif r['african_centrality'] >= 1: r['local_specificity'] = 1
        if 'scientific_depth' not in override:
            depth_kw = ['randomized', 'cohort', 'molecular', 'genomic', 'pcr', 'sequencing',
                        'histopatholog', 'immunohistochem', 'spectroscop', 'chromatograph',
                        'simulation', 'machine learning', 'nanoparticle', 'synthesis',
                        'characterization', 'mathematical model']
            dc = sum(1 for d in depth_kw if d in text)
            r['scientific_depth'] = min(4, max(1, dc // 2 + 1))
        if 'knowledge_value' not in override:
            if r.get('scientific_depth',0) >= 3 and r.get('african_centrality',0) >= 2: r['knowledge_value'] = 3
            elif r.get('scientific_depth',0) >= 2 and r.get('african_centrality',0) >= 1: r['knowledge_value'] = 2
            elif r.get('african_centrality',0) >= 1: r['knowledge_value'] = 1
        if 'local_applicability' not in override:
            if r.get('african_centrality',0) >= 3: r['local_applicability'] = 3
            elif r.get('african_centrality',0) >= 2: r['local_applicability'] = 3
            elif r.get('african_centrality',0) >= 1: r['local_applicability'] = 2
        if 'african_relevance_tags' not in override:
            tag_map = {'NG': 'Nigeria', 'GH': 'Ghana', 'ZA': 'South Africa', 'ET': 'Ethiopia',
                       'TZ': 'Tanzania', 'GN': 'Guinea', 'SO': 'Somalia/Somaliland', 'NE': 'Niger',
                       'KE': 'Kenya', 'CD': 'DRC', 'AFRICA': 'Africa-general', 'SSA': 'Sub-Saharan Africa',
                       'WEST_AFRICA': 'West Africa', 'EAST_AFRICA': 'East Africa',
                       'SOUTHERN_AFRICA': 'Southern Africa'}
            r['african_relevance_tags'] = [tag_map.get(c, c) for c in r.get('african_country_codes',[]) if c in tag_map]
        # Compute total_score
        r['total_score'] = r.get('african_centrality',0) + r.get('local_specificity',0) + r.get('scientific_depth',0) + r.get('knowledge_value',0) + r.get('local_applicability',0)
        # Fill missing text fields
        if not r.get('scientific_evidence'):
            sd = r.get('scientific_depth',0)
            r['scientific_evidence'] = 'Strong methodological approach' if sd >= 3 else ('Moderate scientific rigor' if sd >= 2 else 'Limited scientific depth')
        if not r.get('inference_basis'):
            r['inference_basis'] = f"Explicit mentions: {', '.join(found[:3])}" if found else ('Africa-exclusive entity identified' if inherent else 'Individual record review')
        if not r.get('reason'):
            r['reason'] = f"Individually reviewed. Decision: {r['decision']}"
        if not r.get('evidence') and r.get('african_focus'):
            r['evidence'] = r['african_focus'][:25]
        return r

    # Automatic classification for non-overridden records
    african_map = {
        'nigeria': 'NG', 'ghana': 'GH', 'kenya': 'KE', 'south africa': 'ZA',
        'ethiopia': 'ET', 'tanzania': 'TZ', 'uganda': 'UG', 'cameroon': 'CM',
        'senegal': 'SN', 'mali': 'ML', 'niger': 'NE', 'chad': 'TD',
        'somalia': 'SO', 'somaliland': 'SO', 'guinea': 'GN', 'congo': 'CD',
        'rwanda': 'RW', 'burundi': 'BI', 'zambia': 'ZM', 'zimbabwe': 'ZW',
        'mozambique': 'MZ', 'angola': 'AO', 'malawi': 'MW', 'madagascar': 'MG',
        'botswana': 'BW', 'namibia': 'NA', 'gabon': 'GA', 'togo': 'TG',
        'benin': 'BJ', 'burkina faso': 'BF', 'liberia': 'LR',
        'sierra leone': 'SL', 'gambia': 'GM', 'mauritania': 'MR',
        'sudan': 'SD', 'south sudan': 'SS', 'eritrea': 'ER', 'djibouti': 'DJ',
        'morocco': 'MA', 'algeria': 'DZ', 'tunisia': 'TN', 'libya': 'LY',
        'egypt': 'EG', 'africa': 'AFRICA', 'african': 'AFRICA',
        'sub-saharan': 'SSA', 'west africa': 'WEST_AFRICA',
        'east africa': 'EAST_AFRICA', 'southern africa': 'SOUTHERN_AFRICA',
        'central africa': 'CENTRAL_AFRICA', 'north africa': 'NORTH_AFRICA',
    }

    # More careful matching - check longer phrases first
    sorted_countries = sorted(african_map.keys(), key=len, reverse=True)
    found = []
    codes = []

    for country in sorted_countries:
        if country in text:
            # Avoid false "niger" match when "nigeria" or "niger delta" is present
            if country == 'niger':
                if 'nigeria' in text or 'niger delta' in text:
                    continue
            if country == 'mali':
                # Avoid matching "mali" inside other words
                if not re.search(r'\bmali\b', text):
                    continue
            if country == 'guinea':
                if 'guinea-bissau' in text or 'equatorial guinea' in text:
                    continue
            found.append(country)
            code = african_map[country]
            if code not in codes:
                codes.append(code)

    # Check for Niger Delta specifically
    if 'niger delta' in text:
        if 'NG' not in codes:
            codes.append('NG')
        if 'niger delta' not in found:
            found.append('niger delta')

    # Inherent entities
    inherent = False
    if 'aframomum melegueta' in text: inherent = True
    if 'archachatina marginata' in text: inherent = True
    if 'croton macrostachyus' in text: inherent = True
    if 'dovyalis caffra' in text or 'kei apple' in text: inherent = True
    if 'west african dwarf crocodile' in text or 'osteolaemus tetraspis' in text: inherent = True
    if re.search(r'\bogi\b.*ferment', text) or re.search(r'ferment.*\bogi\b', text): inherent = True

    # Evidence level
    if found:
        r['evidence_level'] = 'direct'
    elif inherent:
        r['evidence_level'] = 'inherent'
    elif any(w in text for w in ['tropical', 'subtropical', 'low-resource', 'lmic']):
        r['evidence_level'] = 'latent'
    else:
        r['evidence_level'] = 'absent'

    # Contradicted check
    if r['evidence_level'] in ['absent', 'latent'] and not inherent:
        non_african = ['china', 'korea', 'japan', 'brazil', 'united states', 'usa',
                       'canada', 'hungary', 'denmark', 'italy', 'indonesia',
                       'saudi arabia', 'malaysia', 'south carolina', 'mexico',
                       'india', 'australia', 'france', 'germany', 'spain',
                       'turkey', 'iran', 'pakistan']
        for na in non_african:
            if na in text:
                r['evidence_level'] = 'contradicted'
                break

    r['african_country_codes'] = codes
    r['african_focus'] = ', '.join(found[:3]) if found else ('Africa-relevant entity' if inherent else '')

    # Evidence quote
    if found:
        for c in found[:1]:
            idx = text.find(c)
            if idx >= 0:
                snippet = text[max(0,idx-5):idx+len(c)+15].strip()
                r['evidence'] = snippet[:25]
                break

    # Scoring
    if r['evidence_level'] == 'direct':
        r['african_centrality'] = 4 if any(c in found for c in ['nigeria', 'ghana', 'kenya', 'south africa', 'ethiopia', 'africa', 'african']) else 3
        r['local_specificity'] = 3 if any(w in text for w in ['state', 'district', 'local government', 'city', 'village']) else 2
    elif r['evidence_level'] == 'inherent':
        r['african_centrality'] = 3
        r['local_specificity'] = 2
    elif r['evidence_level'] == 'latent':
        r['african_centrality'] = 1
        r['local_specificity'] = 1
    else:
        r['african_centrality'] = 0
        r['local_specificity'] = 0

    # Scientific depth
    depth_kw = ['randomized', 'clinical trial', 'cohort', 'meta-analysis', 'systematic review',
                'molecular', 'genomic', 'crystallograph', 'electrochemical', 'simulation',
                'machine learning', 'deep learning', 'neural network', 'pcr', 'sequencing',
                'histopatholog', 'immunohistochem', 'spectroscop', 'chromatograph',
                'finite element', 'numerical model', 'mathematical model',
                'nanoparticle', 'thin film', 'synthesis', 'characterization']
    dc = sum(1 for d in depth_kw if d in text)
    r['scientific_depth'] = min(4, max(1, dc // 2 + 1))

    # Knowledge value
    if r['scientific_depth'] >= 3 and r['african_centrality'] >= 2:
        r['knowledge_value'] = 3
    elif r['scientific_depth'] >= 2 and r['african_centrality'] >= 1:
        r['knowledge_value'] = 2
    elif r['african_centrality'] >= 1:
        r['knowledge_value'] = 1
    else:
        r['knowledge_value'] = 0

    # Local applicability
    if r['african_centrality'] >= 3 and any(w in text for w in ['policy', 'intervention', 'treatment', 'framework', 'mitigation', 'remediation']):
        r['local_applicability'] = 4
    elif r['african_centrality'] >= 2:
        r['local_applicability'] = 3
    elif r['african_centrality'] >= 1:
        r['local_applicability'] = 2
    else:
        r['local_applicability'] = 0

    r['total_score'] = r['african_centrality'] + r['local_specificity'] + r['scientific_depth'] + r['knowledge_value'] + r['local_applicability']

    # Tags
    tag_map = {'NG': 'Nigeria', 'GH': 'Ghana', 'ZA': 'South Africa', 'ET': 'Ethiopia',
               'TZ': 'Tanzania', 'GN': 'Guinea', 'SO': 'Somalia/Somaliland', 'NE': 'Niger',
               'KE': 'Kenya', 'CD': 'DRC', 'AFRICA': 'Africa-general', 'SSA': 'Sub-Saharan Africa',
               'WEST_AFRICA': 'West Africa', 'EAST_AFRICA': 'East Africa',
               'SOUTHERN_AFRICA': 'Southern Africa'}
    r['african_relevance_tags'] = [tag_map.get(c, c) for c in codes if c in tag_map]

    # Decision
    if r['evidence_level'] in ['absent', 'contradicted']:
        r['decision'] = 'exclude'
    elif r['evidence_level'] in ['direct', 'inherent']:
        if r['african_centrality'] >= 2 and r['scientific_depth'] >= 2 and r['total_score'] >= 14:
            r['decision'] = 'include'
        else:
            r['decision'] = 'review'
    elif r['evidence_level'] == 'latent':
        r['decision'] = 'review'

    # Scientific evidence
    if r['scientific_depth'] >= 3:
        r['scientific_evidence'] = 'Strong methodological approach with quantitative data'
    elif r['scientific_depth'] >= 2:
        r['scientific_evidence'] = 'Moderate scientific rigor with empirical data'
    else:
        r['scientific_evidence'] = 'Limited scientific depth in methodology'

    # Reason
    if r['decision'] == 'include':
        r['reason'] = f"Direct African focus ({', '.join(found[:2]) if found else 'inherent entity'}), strong scientific methodology, high local applicability and knowledge value for African context."
    elif r['decision'] == 'review':
        r['reason'] = f"African relevance detected ({r['evidence_level']}) but does not meet all inclusion thresholds. Centrality={r['african_centrality']}, depth={r['scientific_depth']}, total={r['total_score']}."
    else:
        r['reason'] = 'No African geographic or entity relevance detected. Study focuses on non-African setting or global topic without African specificity.'

    r['inference_basis'] = f"Explicit mentions: {', '.join(found[:3])}" if found else ('Africa-exclusive entity identified' if inherent else ('Indirect relevance via tropical/developing context' if r['evidence_level']=='latent' else 'No African-specific indicators found'))

    return r

results = []
for rec in records:
    results.append(classify(rec))

os.makedirs(base_dir + '/results', exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"Written {len(results)} results to {output_file}")

decisions = {}
for r in results:
    d = r['decision']
    decisions[d] = decisions.get(d, 0) + 1
print(f"Decision summary: {decisions}")

# Print include list
includes = [r['title'][:60] for r in results if r['decision'] == 'include']
print(f"\nIncludes ({len(includes)}):")
for t in includes:
    print(f"  - {t}")
