#!/usr/bin/env python3
"""
MUFASA African Relevance Classifier
====================================
Implements the MUFASA African-Relevance Classification Protocol.
Every classification decision reflects the protocol's reasoning:
  - Evidence levels: direct, inherent, latent, affiliation_only, absent, contradicted
  - Scoring: african_centrality, local_specificity, scientific_depth, knowledge_value, local_applicability
  - Decision gates: include requires direct/inherent evidence, centrality>=2, depth>=2, total>=14
  - Anti-hallucination: only title+abstract+field+topic+keywords+work_type are used
  - Hard exclusions: affiliation-only, non-African scope, outside scientific scope, etc.

Input:  openalex_all_fields.csv (155,825 records)
Output: classified_african_relevance.{jsonl,csv,parquet}
"""

import json
import csv
import re
import os
import sys
import time
import multiprocessing
from collections import Counter
from functools import partial
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'openalex_all_fields.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'classification_output', 'final')
BATCH_SIZE = 2000
NUM_WORKERS = min(8, multiprocessing.cpu_count())

USE_COLS = [
    'openalex_id', 'title', 'abstract', 'field_name',
    'primary_topic', 'work_type', 'keywords_json'
]

# ============================================================================
# KNOWLEDGE BASES  (compiled once at module load)
# ============================================================================

# --- African countries with ISO codes and major cities ---
AFRICAN_COUNTRIES = {
    # North Africa
    'Algeria': ('DZ', ['Algiers', 'Oran', 'Constantine']),
    'Egypt': ('EG', ['Cairo', 'Alexandria', 'Luxor', 'Aswan', 'Giza']),
    'Libya': ('LY', ['Tripoli', 'Benghazi']),
    'Morocco': ('MA', ['Casablanca', 'Rabat', 'Marrakech', 'Fes', 'Tangier']),
    'Tunisia': ('TN', ['Tunis', 'Sfax', 'Sousse']),
    'Sudan': ('SD', ['Khartoum', 'Omdurman']),
    'South Sudan': ('SS', ['Juba']),
    # West Africa
    'Nigeria': ('NG', [
        'Lagos', 'Abuja', 'Ibadan', 'Kano', 'Port Harcourt', 'Benin City',
        'Kaduna', 'Enugu', 'Jos', 'Ilorin', 'Abeokuta', 'Owerri', 'Calabar',
        'Warri', 'Zaria', 'Maiduguri', 'Sokoto', 'Uyo', 'Akure', 'Ado-Ekiti',
        'Osogbo', 'Bauchi', 'Makurdi', 'Lafia', 'Gombe', 'Minna', 'Birnin Kebbi',
        'Dutse', 'Gusau', 'Damaturu', 'Yenagoa', 'Asaba', 'Awka', 'Umuahia',
        'Oyo', 'Ogbomoso', 'Ife', 'Ilesa', 'Ondo', 'Ikere', 'Ede', 'Iwo',
        'Shagamu', 'Ijebu-Ode', 'Sapele', 'Onitsha', 'Nnewi', 'Aba',
    ]),
    'Ghana': ('GH', ['Accra', 'Kumasi', 'Tamale', 'Cape Coast']),
    'Senegal': ('SN', ['Dakar', 'Thies', 'Saint-Louis']),
    'Mali': ('ML', ['Bamako', 'Timbuktu']),
    "Cote d'Ivoire": ('CI', ['Abidjan', 'Yamoussoukro']),
    'Ivory Coast': ('CI', ['Abidjan', 'Yamoussoukro']),
    'Burkina Faso': ('BF', ['Ouagadougou']),
    'Niger': ('NE', ['Niamey']),
    'Guinea': ('GN', ['Conakry']),
    'Sierra Leone': ('SL', ['Freetown']),
    'Liberia': ('LR', ['Monrovia']),
    'Togo': ('TG', ['Lome']),
    'Benin': ('BJ', ['Cotonou', 'Porto-Novo']),
    'Cameroon': ('CM', ['Yaounde', 'Douala', 'Bamenda']),
    'Gambia': ('GM', ['Banjul']),
    'Guinea-Bissau': ('GW', ['Bissau']),
    'Mauritania': ('MR', ['Nouakchott']),
    'Cabo Verde': ('CV', ['Praia']),
    'Cape Verde': ('CV', ['Praia']),
    # Central Africa
    'DR Congo': ('CD', ['Kinshasa', 'Lubumbashi']),
    'Democratic Republic of Congo': ('CD', ['Kinshasa', 'Lubumbashi']),
    'DRC': ('CD', ['Kinshasa']),
    'Republic of Congo': ('CG', ['Brazzaville']),
    'Congo-Brazzaville': ('CG', ['Brazzaville']),
    'Angola': ('AO', ['Luanda']),
    'Gabon': ('GA', ['Libreville']),
    'Chad': ('TD', ["N'Djamena"]),
    'Central African Republic': ('CF', ['Bangui']),
    'Equatorial Guinea': ('GQ', ['Malabo']),
    'Sao Tome': ('ST', ['Sao Tome']),
    'Burundi': ('BI', ['Bujumbura', 'Gitega']),
    'Rwanda': ('RW', ['Kigali']),
    # East Africa
    'Kenya': ('KE', ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret']),
    'Tanzania': ('TZ', ['Dar es Salaam', 'Dodoma', 'Arusha', 'Mwanza']),
    'Uganda': ('UG', ['Kampala', 'Entebbe', 'Jinja']),
    'Ethiopia': ('ET', ['Addis Ababa', 'Dire Dawa', 'Mekelle']),
    'Eritrea': ('ER', ['Asmara']),
    'Somalia': ('SO', ['Mogadishu']),
    'Djibouti': ('DJ', ['Djibouti']),
    'Mozambique': ('MZ', ['Maputo']),
    'Malawi': ('MW', ['Lilongwe', 'Blantyre']),
    'Zambia': ('ZM', ['Lusaka', 'Ndola']),
    'Zimbabwe': ('ZW', ['Harare', 'Bulawayo']),
    # Southern Africa
    'South Africa': ('ZA', [
        'Johannesburg', 'Cape Town', 'Durban', 'Pretoria',
        'Bloemfontein', 'Soweto', 'Stellenbosch',
    ]),
    'Namibia': ('NA', ['Windhoek']),
    'Botswana': ('BW', ['Gaborone']),
    'Lesotho': ('LS', ['Maseru']),
    'Eswatini': ('SZ', ['Mbabane']),
    'Swaziland': ('SZ', ['Mbabane']),
    # Islands
    'Madagascar': ('MG', ['Antananarivo']),
    'Mauritius': ('MU', ['Port Louis']),
    'Seychelles': ('SC', ['Victoria']),
    'Comoros': ('KM', ['Moroni']),
}

# Build flat lookups (compiled regexes for each)
_COUNTRY_NAMES = sorted(AFRICAN_COUNTRIES.keys(), key=len, reverse=True)
_CITY_TO_COUNTRY = {}
for _c, (_code, _cities) in AFRICAN_COUNTRIES.items():
    for _city in _cities:
        if _city not in _CITY_TO_COUNTRY:
            _CITY_TO_COUNTRY[_city] = _c

AFRICAN_REGIONS = {
    'West Africa': 'WEST', 'East Africa': 'EAST', 'North Africa': 'NORTH',
    'Southern Africa': 'SOUTH', 'Central Africa': 'CENTRAL',
    'Sub-Saharan Africa': 'SSA', 'Sub-Saharan': 'SSA',
    'Sahel': 'SAHEL', 'Maghreb': 'MAGHREB',
    'Horn of Africa': 'EAST', 'West African': 'WEST',
    'East African': 'EAST', 'North African': 'NORTH',
    'Southern African': 'SOUTH', 'Central African': 'CENTRAL',
    'Sub Saharan': 'SSA', 'Sub Saharan Africa': 'SSA',
}

AFRICAN_GEO_FEATURES = {
    'Sahara': 'NORTH', 'Kalahari': 'ZA', 'Namib': 'NA',
    'Nile': 'EG', 'Niger River': 'NG', 'Congo River': 'CD',
    'Zambezi': 'ZM', 'Lake Victoria': 'EAST', 'Lake Tanganyika': 'EAST',
    'Lake Chad': 'CENTRAL', 'Lake Malawi': 'MW',
    'Great Rift Valley': 'EAST', 'Rift Valley': 'EAST',
    'Congo Basin': 'CD', 'Niger Delta': 'NG',
}

# Africa-specific terms
AFRICA_TERMS = [
    'African', 'Africa', 'Sub-Saharan', 'Sub Saharan',
    'Pan-African', 'Pan African', 'African Union',
    'ECOWAS', 'SADC', 'COMESA', 'EAC',
]

# --- Africa-salient entity databases ---

AFRICA_SALIENT_CROPS = [
    'cassava', 'manioc', 'yuca', 'yam', 'cocoyam', 'taro',
    'teff', 'tef', 'finger millet', 'pearl millet', 'fonio',
    'sorghum', 'cowpea', 'bambara groundnut', 'groundnut',
    'okra', 'plantain', 'oil palm', 'kola nut', 'shea',
    'baobab', 'moringa', 'teff', 'enset', 'chat', 'khat',
    'roselle', 'hibiscus', 'tiger nut', 'locust bean',
    'African yam bean', 'egusi', 'fluted pumpkin',
]

AFRICA_SALIENT_DISEASES = [
    'malaria', 'plasmodium', 'sickle cell',
    'trypanosomiasis', 'sleeping sickness', 'Chagas',
    'onchocerciasis', 'river blindness',
    'lymphatic filariasis', 'elephantiasis',
    'schistosomiasis', 'bilharzia',
    'Buruli ulcer', 'leishmaniasis',
    'trachoma', 'dracunculiasis', 'guinea worm',
    'Lassa fever', 'Ebola', 'Marburg',
    'yellow fever', 'dengue', 'chikungunya',
    'HIV', 'AIDS', 'tuberculosis',
    'neglected tropical disease',
    'soil-transmitted helminth', 'hookworm',
    'cholera', 'typhoid',
    'snakebite envenomation', 'snakebite',
    'podoconiosis', 'noma',
    'leprosy', 'meningitis',
    'measles', 'polio',
    ' Rift Valley fever',
]

AFRICA_SALIENT_ORGANISMS = [
    'Anopheles', 'tsetse', 'Glossina',
    'Tilapia', 'Nile perch', 'African catfish', 'Clarias',
    'shea tree', 'Vitellaria', 'marula', 'Sclerocarya',
    'Acacia', 'miombo', 'mopane', 'brachystegia',
    'Eichhornia', 'water hyacinth', 'Striga', 'witchweed',
    'fall armyworm', 'Spodoptera frugiperda',
    'quelea', 'locust', 'Schistocerca',
    'Faidherbia albida', 'Parkia',
    'khaya', 'African mahogany',
    'iroko', 'Milicia excelsa',
]

AFRICA_SALIENT_LIVESTOCK = [
    'zebu', 'Boran cattle', 'N\'Dama', 'West African Dwarf',
    'Red Sokoto', 'Yankasa', 'Uda sheep',
    'Kudu', 'impala', 'wildebeest',
    'African elephant', 'African wild dog',
]

AFRICA_SALIENT_RESOURCES = [
    'laterite', 'bauxite', 'coltan', 'cobalt',
    'platinum group', 'rare earth', 'phosphate rock',
    'crude oil', 'petroleum', 'natural gas',
    'limestone', 'kaolin', 'clay', 'silica sand',
    'cocoa', 'cashew', 'rubber', 'palm oil',
    'cotton', 'sisal',
]

AFRICA_SALIENT_ENVIRONMENT = [
    'tropical savanna', 'savanna', 'Sahelian',
    'tropical rainforest', 'rainforest',
    'arid zone', 'semi-arid', 'monsoon',
    'harmattan', 'Sahara dust', 'dust storm',
    'flooding', 'drought', 'desertification',
    'soil erosion', 'lateritic soil',
    'tropical climate', 'wet season', 'dry season',
    'harmattan wind',
]

# Group for latent analysis
LATENT_ENTITIES = {
    'crop': AFRICA_SALIENT_CROPS,
    'disease': AFRICA_SALIENT_DISEASES,
    'organism': AFRICA_SALIENT_ORGANISMS,
    'livestock': AFRICA_SALIENT_LIVESTOCK,
    'resource': AFRICA_SALIENT_RESOURCES,
    'environment': AFRICA_SALIENT_ENVIRONMENT,
}

# Non-African geography for contradiction detection
NON_AFRICAN_COUNTRIES = [
    'China', 'Japan', 'South Korea', 'Korea', 'India', 'Pakistan',
    'Bangladesh', 'Thailand', 'Vietnam', 'Indonesia', 'Malaysia',
    'Philippines', 'Myanmar', 'Cambodia', 'Laos', 'Nepal', 'Sri Lanka',
    'United States', 'USA', 'America', 'Canada', 'Mexico',
    'Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Ecuador',
    'United Kingdom', 'UK', 'Britain', 'France', 'Germany', 'Italy',
    'Spain', 'Portugal', 'Netherlands', 'Belgium', 'Sweden', 'Norway',
    'Denmark', 'Finland', 'Poland', 'Czech', 'Austria', 'Switzerland',
    'Greece', 'Turkey', 'Russia', 'Ukraine', 'Romania', 'Hungary',
    'Ireland', 'Scotland', 'Wales', 'England',
    'Australia', 'New Zealand',
    'Iran', 'Iraq', 'Saudi Arabia', 'UAE', 'Israel', 'Jordan', 'Lebanon',
    'Syria', 'Yemen', 'Oman', 'Qatar', 'Kuwait',
    'Cuba', 'Jamaica', 'Haiti', 'Puerto Rico',
]

NON_AFRICAN_REGIONS = [
    'Europe', 'European', 'Asia', 'Asian',
    'North America', 'South America', 'Latin America',
    'Middle East', 'Southeast Asia', 'South Asia', 'East Asia',
    'Central Asia', 'Oceania', 'Pacific', 'Caribbean',
    'Scandinavia', 'Mediterranean', 'Balkans', 'Caucasus',
    'African-American', 'African American', 'Black American',
    'Hispanic', 'Latino',
]

# Non-scientific scope keywords
OUTSIDE_SCIENCE_KW = [
    'policy', 'governance', 'legislation', 'regulation',
    'advocacy', 'awareness', 'perception', 'attitude',
    'knowledge and perception', 'stigma', 'belief',
    'economic impact', 'market analysis', 'business strategy',
    'consumer behaviour', 'consumer behavior', 'marketing',
    'human rights', 'social justice', 'philosophy',
    'literary', 'cultural study', 'historical analysis',
    'theological', 'religious study',
]

HARD_EXCL_FIELDS = {
    'economics', 'business', 'law', 'governance',
    'humanities', 'philosophy', 'history', 'arts',
    'social science', 'political science', 'linguistics',
    'education', 'sociology', 'anthropology',
    'accounting', 'finance', 'management',
}

# Scientific depth indicators
DEPTH_STRONG_KW = [
    'randomized controlled trial', 'RCT', 'clinical trial',
    'systematic review', 'meta-analysis', 'meta analysis',
    'cohort study', 'longitudinal', 'prospective',
    'retrospective', 'cross-sectional', 'cross sectional',
    'genome-wide', 'GWAS', 'whole-genome', 'transcriptome',
    'proteome', 'metabolome', 'large-scale', 'large scale',
    'nationwide', 'population-based', 'multi-centre', 'multicentre',
    'multi-center', 'multicenter', 'randomised',
    'clinical trial', 'phase III', 'phase IV',
    'machine learning', 'deep learning', 'neural network',
    'Monte Carlo', 'molecular dynamics', 'DFT',
    'density functional', 'finite element',
]

DEPTH_MEDIUM_KW = [
    'experiment', 'measurement', 'analysis', 'assay',
    'spectroscopy', 'chromatography', 'PCR', 'sequencing',
    'model', 'simulation', 'algorithm', 'optimization',
    'diagnosis', 'pathology', 'histology', 'biopsy',
    'survey data', 'field study', 'field survey',
    'questionnaire-based', 'interview-based',
    'statistical analysis', 'regression', 'ANOVA',
    'clinical', 'epidemiological',
    'in vitro', 'in vivo', 'cell culture',
    'X-ray', 'crystallography', 'microscopy',
    'synthesis', 'characterization', 'evaluation',
    'assessment', 'monitoring', 'mapping',
    'case-control', 'case control',
]

DEPTH_WEAK_KW = [
    'case report', 'case series', 'letter', 'correspondence',
    'commentary', 'brief report', 'short communication',
    'preliminary', 'pilot', 'proof of concept',
    'review', 'overview', 'narrative',
]

EDITORIAL_KW = [
    'editorial', 'announcement', 'corrigendum', 'erratum',
    'retraction', 'book review', 'obituary', 'news',
    'conference abstract', 'meeting report',
    'call for', 'forthcoming',
]

# Work types that are hard exclusions
HARD_EXCL_WORK_TYPES = {'editorial', 'letter', 'erratum', 'corrigendum', 'retraction'}

# ============================================================================
# PRE-COMPILED REGEX PATTERNS
# ============================================================================

_RE_WORD = re.compile(r'[\w\-]+')

def _wb(term):
    """Word-boundary regex for a term."""
    return re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)

# Country patterns
_PAT_COUNTRIES = {name: _wb(name) for name in _COUNTRY_NAMES}
# City patterns
_PAT_CITIES = {city: _wb(city) for city in _CITY_TO_COUNTRY}
# Region patterns
_PAT_REGIONS = {name: _wb(name) for name in AFRICAN_REGIONS}
# Geo feature patterns
_PAT_GEO = {name: _wb(name) for name in AFRICAN_GEO_FEATURES}
# Africa terms
_PAT_AFRICA = [_wb(t) for t in AFRICA_TERMS]
# Latent entity patterns
_PAT_LATENT = {}
for _cat, _terms in LATENT_ENTITIES.items():
    _PAT_LATENT[_cat] = {t: _wb(t) for t in _terms}
# Non-African patterns
_PAT_NON_AF_COUNTRIES = {c: _wb(c) for c in NON_AFRICAN_COUNTRIES}
_PAT_NON_AF_REGIONS = {r: _wb(r) for r in NON_AFRICAN_REGIONS}
# Scope patterns
_PAT_OUTSIDE_SCI = [_wb(kw) for kw in OUTSIDE_SCIENCE_KW]
_PAT_EDITORIAL = [_wb(kw) for kw in EDITORIAL_KW]
_PAT_DEPTH_STRONG = [_wb(kw) for kw in DEPTH_STRONG_KW]
_PAT_DEPTH_MEDIUM = [_wb(kw) for kw in DEPTH_MEDIUM_KW]
_PAT_DEPTH_WEAK = [_wb(kw) for kw in DEPTH_WEAK_KW]

# Context words that indicate research location (used near country/region names)
_RESEARCH_CONTEXT = _wb('study|studied|research|survey|sample|samples|sampling|patients|data collected|data were collected|field|fieldwork|conducted|population|community|rural|urban|hospital|clinic|centre|center|university|institute|laboratory|laboratories|farms|sites|region|area|district|state|province|village|settlement')

# Context indicating Africa IS the focus (near Africa/African mentions)
_AFRICA_FOCUS_CONTEXT = _wb(
    'in Africa|across Africa|throughout Africa|within Africa|from Africa|'
    'African countries|African nations|African population|African patients|'
    'African region|African continent|African context|African setting|'
    'African environment|African climate|African soil|African water|'
    'African crop|African disease|African health|African agriculture|'
    'in Nigerian|in Ghana|in Kenya|in South Africa|in Egypt|in Ethiopia|'
    'Nigerian patients|Nigerian population|Nigerian study|Nigerian data|'
    'Ghanaian|Kenyan|Ethiopian|Tanzanian|Ugandan|Senegalese|Cameroonian|'
    'South African|Rwandan|Mozambican|Zimbabwean|Zambian|Malawian|'
    'Moroccan|Tunisian|Algerian|Libyan|Sudanese|Somali|Congolese|Angolan'
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_json_parse(val):
    """Parse JSON safely, return None on failure."""
    if val is None or val == '' or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (list, dict)):
        return val
    try:
        return json.loads(str(val))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def extract_keywords(keywords_json, max_kw=5):
    """Extract up to max_kw keyword names from JSON."""
    parsed = safe_json_parse(keywords_json)
    if not parsed:
        return []
    kws = []
    for item in parsed[:max_kw]:
        if isinstance(item, dict):
            name = item.get('name', '')
            if name:
                kws.append(name)
        elif isinstance(item, str):
            kws.append(item)
    return kws


def extract_fields(record):
    """Extract protocol-relevant fields from a record dict."""
    title = str(record.get('title', '') or '')
    abstract = str(record.get('abstract', '') or '')
    field_name = str(record.get('field_name', '') or '')
    primary_topic = str(record.get('primary_topic', '') or '')
    work_type = str(record.get('work_type', '') or '').lower()
    keywords = extract_keywords(record.get('keywords_json'))
    return {
        'openalex_id': str(record.get('openalex_id', '') or ''),
        'title': title,
        'abstract': abstract,
        'field_name': field_name,
        'primary_topic': primary_topic,
        'work_type': work_type,
        'keywords': keywords,
    }


def truncate_evidence(text, match_term, max_words=25):
    """Extract a short evidence excerpt around a match term."""
    if not text or not match_term:
        return ''
    lower = text.lower()
    term_lower = match_term.lower()
    idx = lower.find(term_lower)
    if idx < 0:
        return text[:200]
    # Expand around the match
    start = max(0, idx - 60)
    end = min(len(text), idx + len(match_term) + 60)
    excerpt = text[start:end].strip()
    # Trim to max_words
    words = excerpt.split()
    if len(words) > max_words:
        excerpt = ' '.join(words[:max_words])
    return excerpt


# ============================================================================
# EVIDENCE DETECTION
# ============================================================================

def detect_direct(text):
    """
    Detect DIRECT evidence: title/abstract explicitly identifies an African
    subject, source, setting, dataset, condition, or application.
    Returns (evidence_level, african_focus, country_codes, evidence_quote, tags).
    """
    if not text:
        return None, None, [], '', []

    found_countries = []
    found_tags = set()
    best_evidence = ''
    centrality_signal = 'incidental'  # will be upgraded if Africa is central

    # 1. Check African country names
    for name, pat in _PAT_COUNTRIES.items():
        m = pat.search(text)
        if m:
            code, _ = AFRICAN_COUNTRIES[name]
            if code not in found_countries:
                found_countries.append(code)
            found_tags.add('african_location')
            if not best_evidence:
                best_evidence = truncate_evidence(text, m.group())
            # Check if it's a research location (nearby research context)
            ctx_start = max(0, m.start() - 150)
            ctx_end = min(len(text), m.end() + 150)
            ctx = text[ctx_start:ctx_end]
            if _RESEARCH_CONTEXT.search(ctx):
                centrality_signal = 'central'
            if _AFRICA_FOCUS_CONTEXT.search(ctx):
                centrality_signal = 'central'

    # 2. Check African city names
    for city, pat in _PAT_CITIES.items():
        m = pat.search(text)
        if m:
            c_name = _CITY_TO_COUNTRY[city]
            code, _ = AFRICAN_COUNTRIES.get(c_name, ('', []))
            if code and code not in found_countries:
                found_countries.append(code)
            found_tags.add('african_location')
            if not best_evidence:
                best_evidence = truncate_evidence(text, m.group())
            centrality_signal = 'central'

    # 3. Check African regions
    for name, pat in _PAT_REGIONS.items():
        m = pat.search(text)
        if m:
            found_tags.add('african_location')
            if not best_evidence:
                best_evidence = truncate_evidence(text, m.group())
            centrality_signal = 'central'

    # 4. Check African geographic features
    for name, pat in _PAT_GEO.items():
        m = pat.search(text)
        if m:
            found_tags.add('african_location')
            if not best_evidence:
                best_evidence = truncate_evidence(text, m.group())

    # 5. Check Africa/African/Pan-African terms
    for pat in _PAT_AFRICA:
        m = pat.search(text)
        if m:
            found_tags.add('african_location')
            if not best_evidence:
                best_evidence = truncate_evidence(text, m.group())
            # Check if Africa is central to the study
            ctx_start = max(0, m.start() - 150)
            ctx_end = min(len(text), m.end() + 150)
            ctx = text[ctx_start:ctx_end]
            if _AFRICA_FOCUS_CONTEXT.search(ctx):
                centrality_signal = 'central'
            elif centrality_signal != 'central':
                centrality_signal = 'meaningful'

    if not found_countries and not found_tags:
        return None, None, [], '', []

    # Determine african_focus
    if centrality_signal == 'central':
        focus = 'essential'
    elif centrality_signal == 'meaningful':
        focus = 'meaningful'
    elif len(found_countries) > 2:
        focus = 'meaningful'
    else:
        focus = 'incidental'

    # Additional tags based on context
    combined = text.lower()
    if any(kw in combined for kw in ['patient', 'hospital', 'clinic', 'clinical', 'diagnosis', 'disease', 'health']):
        found_tags.add('african_population')
    if any(kw in combined for kw in ['dataset', 'data collected', 'survey data', 'sample', 'cohort']):
        found_tags.add('african_dataset')
    if any(kw in combined for kw in ['soil', 'water', 'climate', 'rainfall', 'temperature', 'ecosystem']):
        found_tags.add('african_environment')
    if any(kw in combined for kw in ['material', 'mineral', 'ore', 'rock', 'resource']):
        found_tags.add('african_material')

    return 'direct', focus, found_countries, best_evidence, sorted(found_tags)


def detect_inherent(text):
    """
    Detect INHERENT evidence: Africa-exclusive or unmistakably African entity
    even though the text does not spell out the geography.
    Returns (evidence_level, evidence_quote, tags, inference_note).
    """
    if not text:
        return None, '', [], ''

    # Africa-specific terms that ARE inherently African even without geography
    inherent_terms = [
        ('Sahel', 'Sahel is a Africa-specific semi-arid belt'),
        ('Sahelian', 'Sahelian refers to the Africa-specific Sahel zone'),
        ('Maghreb', 'Maghreb is the Africa-specific North African region'),
        ('harmattan', 'Harmattan is the Africa-specific West African dry wind/dust'),
        ('miombo', 'Miombo is the Africa-specific woodland/forest type'),
        ('tsetse', 'Tsetse fly is Africa-endemic, vectors trypanosomiasis'),
        ('Glossina', 'Glossina (tsetse) is Africa-endemic'),
        ('teff', 'Teff is an Ethiopia/Eritrea-origin cereal crop'),
        ('enset', 'Enset is a crop endemic to Ethiopian highlands'),
        ('chat', 'Khat/chat has major East African cultural significance'),
        ('khat', 'Khat has major East African cultural significance'),
        ('kola nut', 'Kola nut is a West African culturally significant crop'),
        ('shea butter', 'Shea is native to the African Sahel belt'),
        ('Vitellaria', 'Vitellaria (shea) is native to African Sahel'),
        ('baobab', 'Baobab is iconically African'),
        ('marula', 'Marula is a Southern African native fruit tree'),
        ('Lake Victoria', 'Lake Victoria is the largest African lake'),
        ('Congo Basin', 'Congo Basin is the major African rainforest'),
        ('Niger Delta', 'Niger Delta is a major African geographic feature'),
        ('Great Rift Valley', 'Great Rift Valley is the defining African geological feature'),
        ('Faidherbia albida', 'Faidherbia albida is a characteristic African agroforestry tree'),
        ('African yam bean', 'African yam bean is a legume native to West Africa'),
        ('egusi', 'Egusi melon is a West African crop'),
        ('fluted pumpkin', 'Fluted pumpkin (Telfairia) is a West African leafy vegetable'),
        ('Boran cattle', 'Boran cattle are an East African zebu breed'),
        ("N'Dama", "N'Dama is a West African trypanotolerant cattle breed"),
        ('West African Dwarf', 'West African Dwarf is a West African goat breed'),
        ('Red Sokoto', 'Red Sokoto (Maradi) is a West African goat breed'),
        ('locust bean', 'African locust bean (Parkia biglobosa) is a West African food condiment'),
    ]

    for term, inference in inherent_terms:
        pat = _wb(term)
        m = pat.search(text)
        if m:
            ev = truncate_evidence(text, m.group())
            return 'inherent', ev, ['african_organism' if term.lower() in [t.lower() for t in ['tsetse', 'Glossina', 'baobab', 'marula', 'Faidherbia albida']] else 'african_location'], inference

    return None, '', [], ''


def detect_latent(text):
    """
    Detect LATENT evidence: entity credibly important in Africa but not
    Africa-exclusive. Centrality must be verified.
    Returns (evidence_level, matches_dict, evidence_quote, inference_basis, tags).
    """
    if not text:
        return None, {}, '', '', []

    matches = {}
    first_match = ''

    for cat, term_pats in _PAT_LATENT.items():
        for term, pat in term_pats.items():
            m = pat.search(text)
            if m:
                if cat not in matches:
                    matches[cat] = []
                matches[cat].append(term)
                if not first_match:
                    first_match = truncate_evidence(text, m.group())

    if not matches:
        return None, {}, '', '', []

    # Build inference basis
    parts = []
    if 'crop' in matches:
        crops = matches['crop']
        parts.append(f"{', '.join(crops[:2])} {'are' if len(crops[:2])>1 else 'is'} major Africa-salient crop(s) with established African production basis")
    if 'disease' in matches:
        diseases = matches['disease']
        parts.append(f"{', '.join(diseases[:2])} {'have' if len(diseases[:2])>1 else 'has'} high African disease burden")
    if 'organism' in matches:
        orgs = matches['organism']
        parts.append(f"{', '.join(orgs[:2])} {'are' if len(orgs[:2])>1 else 'is'} Africa-salient organism(s)")
    if 'livestock' in matches:
        parts.append(f"{', '.join(matches['livestock'][:2])} are African livestock breeds")
    if 'resource' in matches:
        parts.append(f"{', '.join(matches['resource'][:2])} are significant African natural resources")
    if 'environment' in matches:
        parts.append(f"{', '.join(matches['environment'][:2])} characterise African environmental conditions")

    inference = '; '.join(parts)
    tags = ['latent_africa_relevance']

    return 'latent', matches, first_match, inference, tags


def detect_contradiction(text):
    """
    Detect CONTRADICTED evidence: research is explicitly non-African with
    no meaningful African component.
    Returns (is_contradicted, evidence_quote).
    """
    if not text:
        return False, ''

    # Check non-African geographic signals as research setting
    for name, pat in _PAT_NON_AF_REGIONS.items():
        m = pat.search(text)
        if m:
            ctx_start = max(0, m.start() - 120)
            ctx_end = min(len(text), m.end() + 120)
            ctx = text[ctx_start:ctx_end]
            if _RESEARCH_CONTEXT.search(ctx):
                ev = truncate_evidence(text, m.group())
                return True, ev

    for name, pat in _PAT_NON_AF_COUNTRIES.items():
        m = pat.search(text)
        if m:
            ctx_start = max(0, m.start() - 120)
            ctx_end = min(len(text), m.end() + 120)
            ctx = text[ctx_start:ctx_end]
            if _RESEARCH_CONTEXT.search(ctx):
                ev = truncate_evidence(text, m.group())
                return True, ev

    return False, ''


# ============================================================================
# HARD EXCLUSION DETECTION
# ============================================================================

def detect_hard_exclusion(field_name, work_type, text):
    """
    Detect hard exclusions per protocol section 4.
    Returns (is_excluded, reason).
    """
    # 1. Work type exclusions
    if work_type in HARD_EXCL_WORK_TYPES:
        return True, 'editorial_advocacy_or_opinion'

    # 2. Editorial/announcement patterns in text
    if text:
        lower = text.lower()
        # Check for editorial patterns at start of text (title)
        for pat in _PAT_EDITORIAL:
            m = pat.search(lower[:200])
            if m:
                return True, 'editorial_advocacy_or_opinion'

    # 3. Outside scientific scope
    if field_name:
        field_lower = field_name.lower()
        for excl in HARD_EXCL_FIELDS:
            if excl in field_lower:
                return True, 'outside_scientific_scope'

    # 4. Awareness/perception/service-only (check abstract)
    if text:
        lower = text.lower()
        awareness_combos = [
            'awareness and knowledge', 'knowledge, attitude',
            'knowledge and attitude', 'perception and practice',
            'awareness survey', 'perception survey', 'attitude survey',
            'knowledge assessment', 'awareness assessment',
            'knowledge, attitude, and practice', 'KAP study', 'KAP survey',
            'health-seeking behaviour', 'health seeking behaviour',
            'health-seeking behavior', 'health seeking behavior',
            'service utilisation', 'service utilization',
            'patient satisfaction', 'client satisfaction',
            'stigma and discrimination', 'stigmatising',
            'health literacy', 'health belief',
        ]
        for combo in awareness_combos:
            if combo in lower:
                # Check it's the PRIMARY focus
                if any(kw in lower for kw in [
                    'the aim', 'this study', 'we assessed', 'we evaluated',
                    'we examined', 'objective', 'purpose', 'investigated',
                ]):
                    return True, 'awareness_perception_or_service_only'

    return False, ''


# ============================================================================
# SCORING
# ============================================================================

def score_centrality(evidence_level, african_focus):
    """african_centrality: how fundamental is Africa to the paper."""
    if evidence_level in ('absent', 'affiliation_only', 'contradicted'):
        return 0
    if evidence_level in ('direct', 'inherent'):
        if african_focus == 'essential':
            return 4
        elif african_focus == 'meaningful':
            return 3
        elif african_focus == 'incidental':
            return 2
        else:
            return 1
    if evidence_level == 'latent':
        return 1
    return 0


def score_specificity(evidence_level, country_codes, african_focus):
    """local_specificity: how specific is the African location/entity."""
    if evidence_level in ('absent', 'affiliation_only', 'contradicted'):
        return 0
    n = len(country_codes)
    if evidence_level in ('direct', 'inherent'):
        if n >= 2:
            return 4
        elif n == 1:
            return 3
        elif african_focus in ('essential', 'meaningful'):
            return 3
        else:
            return 2
    if evidence_level == 'latent':
        return 1
    return 0


def score_depth(text, work_type):
    """scientific_depth: robustness of scientific methodology."""
    if not text:
        return 1

    lower = text.lower()
    strong = sum(1 for p in _PAT_DEPTH_STRONG if p.search(lower))
    medium = sum(1 for p in _PAT_DEPTH_MEDIUM if p.search(lower))
    weak = sum(1 for p in _PAT_DEPTH_WEAK if p.search(lower))

    # Strong indicators dominate
    if strong >= 2:
        return 4
    if strong >= 1 and medium >= 1:
        return 4
    if strong >= 1:
        return 3
    if medium >= 3:
        return 3
    if medium >= 1:
        return 2
    if weak >= 2:
        return 1
    if weak >= 1:
        return 1

    # Default: look for structural cues
    if any(kw in lower for kw in ['method', 'material', 'result', 'conclusion', 'finding']):
        return 2
    return 2  # assume at least descriptive study for published articles


def score_knowledge(text, depth, work_type):
    """knowledge_value: richness and reusability of findings."""
    if not text:
        return 1

    lower = text.lower()
    # Results-oriented language
    result_kw = ['result', 'finding', 'showed', 'demonstrated', 'revealed',
                 'found that', 'suggest that', 'indicate', 'significant',
                 'novel', 'first', 'new method', 'framework', 'model',
                 'improved', 'enhanced', 'optimised', 'optimized']
    found = sum(1 for kw in result_kw if kw in lower)

    if depth >= 3 and found >= 3:
        return 4
    if depth >= 2 and found >= 2:
        return 3
    if depth >= 2 or found >= 1:
        return 2
    if work_type in HARD_EXCL_WORK_TYPES:
        return 0
    return 1


def score_applicability(evidence_level, african_focus, country_codes, text):
    """local_applicability: relevance to African needs/systems."""
    if evidence_level in ('absent', 'affiliation_only', 'contradicted'):
        return 0
    if evidence_level in ('direct', 'inherent'):
        if african_focus == 'essential':
            return 4
        if african_focus == 'meaningful':
            return 3
        if country_codes:
            return 2
        return 1
    if evidence_level == 'latent':
        if text:
            lower = text.lower()
            if any(kw in lower for kw in ['Africa', 'African', 'developing countr',
                                           'low-income', 'resource-limited',
                                           'tropical', 'subtropical']):
                return 2
        return 1
    return 0


# ============================================================================
# DECISION ENGINE
# ============================================================================

def make_decision(evidence_level, hard_exclusion, hard_exclusion_reason,
                  centrality, specificity, depth, knowledge, applicability,
                  african_focus, text, title):
    """
    Apply protocol decision rules:
      INCLUDE: no hard exclusion + direct/inherent + centrality>=2 + depth>=2 + total>=14
      REVIEW:  latent with salience, direct/inherent with failed gates, incomplete abstract
      EXCLUDE: hard exclusion, affiliation_only, absent, contradicted, incidental
    """
    total = centrality + specificity + depth + knowledge + applicability

    # --- HARD EXCLUSION ---
    if hard_exclusion:
        return 'exclude'

    # --- AFFILIATION / ABSENT / CONTRADICTED ---
    if evidence_level in ('affiliation_only', 'absent', 'contradicted'):
        return 'exclude'

    # --- NO ABSTRACT ---
    abstract_empty = not text or len(text.strip()) < 30
    if abstract_empty:
        if evidence_level in ('direct', 'inherent'):
            return 'review'  # Title shows relevance but depth unverified
        return 'exclude'

    # --- INCLUDE GATES (for direct/inherent) ---
    if evidence_level in ('direct', 'inherent'):
        gates_pass = (
            centrality >= 2 and
            depth >= 2 and
            total >= 14
        )
        if gates_pass:
            return 'include'
        # Gates failed but evidence exists
        if centrality >= 1:
            return 'review'
        return 'exclude'

    # --- LATENT (ceiling is review) ---
    if evidence_level == 'latent':
        # Check centrality in the text — entity must be central
        if text and len(text) > 100:
            # Latent entity is central if it appears in the first ~500 chars
            # (i.e., in the title or early abstract)
            return 'review'
        return 'review'

    return 'exclude'


# ============================================================================
# REASON GENERATION
# ============================================================================

def generate_reason(fields, decision, evidence_level, hard_exclusion,
                    hard_exclusion_reason, centrality, specificity, depth,
                    knowledge, applicability, african_focus, evidence_quote,
                    inference_basis, latent_matches, contradiction_ev):
    """
    Generate a paper-specific grounded explanation (25-60 words).
    Must name: what the paper does, the African connection, and why this decision.
    """
    title = fields['title']
    field = fields['field_name']
    topic = fields['primary_topic']

    # Truncate title for use in reason
    short_title = title[:120] + ('...' if len(title) > 120 else '') if title else 'Untitled'

    if decision == 'include':
        if evidence_level == 'direct':
            countries = ', '.join(evidence_quote.split()[:3]) if evidence_quote else 'African'
            return (
                f"The paper studies {short_title[:80]}; "
                f"the title/abstract explicitly identifies an African setting or population "
                f"({evidence_quote[:40] if evidence_quote else 'direct African evidence'}), "
                f"and the African element materially affects the scientific evidence. "
                f"Include gates pass (centrality={centrality}, depth={depth}, total={centrality+specificity+depth+knowledge+applicability})."
            )
        else:  # inherent
            return (
                f"The paper investigates {short_title[:80]}; "
                f"an Africa-exclusive entity is central to the study "
                f"({evidence_quote[:40] if evidence_quote else 'inherent African evidence'}). "
                f"{inference_basis}. Include gates pass."
            )

    elif decision == 'review':
        if evidence_level == 'latent':
            entities = ', '.join(
                t for cat, terms in latent_matches.items() for t in terms[:1]
            ) if latent_matches else 'an Africa-salient entity'
            return (
                f"The paper examines {short_title[:80]}; "
                f"{entities} {'are' if ',' in entities else 'is'} central and "
                f"Africa-salient ({inference_basis}), but no African geography is stated. "
                f"Latent evidence caps at review; classify rather than invent origin."
            )
        elif evidence_level in ('direct', 'inherent'):
            total = centrality + specificity + depth + knowledge + applicability
            failed = []
            if centrality < 2:
                failed.append(f"centrality={centrality}<2")
            if depth < 2:
                failed.append(f"depth={depth}<2")
            if total < 14:
                failed.append(f"total={total}<14")
            gate_info = ', '.join(failed) if failed else 'gate uncertain'
            return (
                f"The paper has {'direct' if evidence_level == 'direct' else 'inherent'} African evidence "
                f"({evidence_quote[:40] if evidence_quote else 'African signal'}), "
                f"but an include gate fails ({gate_info}). Review pending further assessment."
            )
        else:
            return (
                f"The title references an African topic but the abstract is missing, "
                f"truncated, or boilerplate; relevance is visible but scientific depth is unverified."
            )

    else:  # exclude
        if hard_exclusion:
            reasons = {
                'affiliation_only': "the only African signal is author affiliation, not research content",
                'explicit_non_african_scope': f"the research is explicitly set outside Africa ({contradiction_ev[:40] if contradiction_ev else 'non-African setting'}), with no meaningful African component",
                'outside_scientific_scope': f"the work is primarily {field.lower() if field else 'non-scientific'}, which falls outside this corpus's scientific scope",
                'awareness_perception_or_service_only': "the study measures awareness, perception, knowledge, or service utilisation without substantive scientific measurement",
                'editorial_advocacy_or_opinion': "the work is an editorial, announcement, or opinion piece without scientific analysis",
            }
            return f"{short_title[:80]}; {reasons.get(hard_exclusion_reason, 'excluded by hard-exclusion rule')}."

        if evidence_level == 'absent':
            return (
                f"The paper {'examines' if title else 'presents'} {short_title[:80]}; "
                f"no direct, inherent, or credible latent African signal is detected in the title or abstract. "
                f"Generic global science without African specificity is excluded."
            )
        elif evidence_level == 'affiliation_only':
            return (
                f"The paper studies {short_title[:80]}; "
                f"no African subject, condition, or application appears in the title or abstract. "
                f"African author affiliation alone cannot establish relevance."
            )
        elif evidence_level == 'contradicted':
            return (
                f"The paper studies {short_title[:80]}; "
                f"the research population, samples, or setting are explicitly non-African "
                f"({contradiction_ev[:40] if contradiction_ev else 'non-African scope'}), "
                f"with no meaningful African component."
            )
        else:
            return (
                f"The paper studies {short_title[:80]}; "
                f"African content is incidental with no African analysis or application."
            )


# ============================================================================
# MAIN CLASSIFICATION FUNCTION
# ============================================================================

def classify_record(record):
    """
    Classify a single record following the full protocol.
    This is the entry point called for each of the 155k+ records.
    """
    # Extract only protocol-relevant fields
    fields = extract_fields(record)
    oid = fields['openalex_id']
    title = fields['title']
    abstract = fields['abstract']
    field_name = fields['field_name']
    primary_topic = fields['primary_topic']
    work_type = fields['work_type']
    keywords = fields['keywords']

    # Combined text for analysis (title + abstract only, per protocol)
    combined = f"{title} {abstract}".strip()
    text_lower = combined.lower() if combined else ''

    # ---- Check for empty/missing abstract ----
    abstract_empty = not abstract or len(abstract.strip()) < 30

    # ---- STEP 1: Check hard exclusions ----
    hard_excl, hard_excl_reason = detect_hard_exclusion(field_name, work_type, combined)

    # ---- STEP 2: Detect contradiction ----
    is_contradicted, contradiction_ev = False, ''
    if not hard_excl:
        is_contradicted, contradiction_ev = detect_contradiction(combined)
        if is_contradicted:
            hard_excl = True
            hard_excl_reason = 'explicit_non_african_scope'

    # ---- STEP 3: Evidence detection (priority order) ----
    evidence_level = None
    african_focus = 'absent'
    country_codes = []
    evidence_quote = ''
    tags = []
    inference_basis = ''
    latent_matches = {}

    if not hard_excl:
        # a. Direct evidence
        ev, foc, codes, ev_quote, ev_tags = detect_direct(combined)
        if ev:
            evidence_level = ev
            african_focus = foc
            country_codes = codes
            evidence_quote = ev_quote
            tags = ev_tags

        # b. Inherent evidence (only if no direct found)
        if not evidence_level:
            ev, ev_quote, ev_tags, inf_note = detect_inherent(combined)
            if ev:
                evidence_level = ev
                african_focus = 'essential'
                evidence_quote = ev_quote
                tags = ev_tags
                inference_basis = inf_note

        # c. Latent evidence (only if no direct/inherent found)
        if not evidence_level:
            ev, matches, ev_quote, inf_basis, ev_tags = detect_latent(combined)
            if ev:
                evidence_level = ev
                african_focus = 'potential'
                evidence_quote = ev_quote
                inference_basis = inf_basis
                tags = ev_tags
                latent_matches = matches

        # d. Default: absent
        if not evidence_level:
            evidence_level = 'absent'
            african_focus = 'absent'
    else:
        # Hard excluded — still try to detect evidence for reporting
        ev, foc, codes, ev_quote, ev_tags = detect_direct(combined)
        if ev:
            evidence_level = 'direct'
            african_focus = foc if foc else 'incidental'
            country_codes = codes
            evidence_quote = ev_quote
            tags = ev_tags
        else:
            # Check if it's affiliation-only
            # Since we filtered NG-authored papers, if there's no content signal,
            # the only African connection is affiliation
            evidence_level = 'affiliation_only'
            african_focus = 'absent'

        if is_contradicted:
            evidence_level = 'contradicted'
            african_focus = 'absent'

    # ---- STEP 4: Scoring ----
    centrality = score_centrality(evidence_level, african_focus)
    specificity = score_specificity(evidence_level, country_codes, african_focus)
    depth_val = score_depth(combined, work_type)
    knowledge_val = score_knowledge(combined, depth_val, work_type)
    applicability_val = score_applicability(evidence_level, african_focus, country_codes, combined)

    # ---- STEP 5: Decision ----
    decision = make_decision(
        evidence_level, hard_excl, hard_excl_reason,
        centrality, specificity, depth_val, knowledge_val, applicability_val,
        african_focus, combined, title
    )

    # ---- STEP 6: Refine for edge cases ----
    # African title but missing/boilerplate abstract → review
    if abstract_empty and evidence_level in ('direct', 'inherent') and decision != 'review':
        decision = 'review'

    # If excluded but title mentions Africa explicitly → review
    if decision == 'exclude' and evidence_level == 'direct' and african_focus in ('essential', 'meaningful'):
        # Check if the hard exclusion was the only reason
        if not hard_excl:
            decision = 'review'

    # ---- STEP 7: Scientific evidence type ----
    sci_ev = 'weak_or_none'
    if combined:
        lower = combined.lower()
        if any(kw in lower for kw in ['randomized', 'randomised', 'clinical trial', 'RCT']):
            sci_ev = 'experimental'
        elif any(kw in lower for kw in ['cohort', 'longitudinal', 'cross-sectional', 'epidemiological',
                                         'observational', 'survey', 'prevalence', 'incidence']):
            sci_ev = 'observational'
        elif any(kw in lower for kw in ['algorithm', 'simulation', 'model', 'computational',
                                         'machine learning', 'deep learning', 'neural network',
                                         'optimization', 'optimisation']):
            sci_ev = 'computational'
        elif any(kw in lower for kw in ['systematic review', 'meta-analysis', 'meta analysis', 'literature review']):
            sci_ev = 'systematic_review'
        elif any(kw in lower for kw in ['case report', 'case series']):
            sci_ev = 'case_report'
        elif any(kw in lower for kw in ['theoretical', 'mathematical', 'analytical', 'proof', 'theorem']):
            sci_ev = 'theoretical'
        elif any(kw in lower for kw in ['experiment', 'synthesis', 'characterization', 'measurement',
                                         'assay', 'analysis', 'evaluation', 'test', 'testing']):
            sci_ev = 'experimental'
        elif depth_val >= 2:
            sci_ev = 'observational'

    # ---- STEP 8: Generate reason ----
    reason = generate_reason(
        fields, decision, evidence_level, hard_excl, hard_excl_reason,
        centrality, specificity, depth_val, knowledge_val, applicability_val,
        african_focus, evidence_quote, inference_basis, latent_matches,
        contradiction_ev
    )

    # Truncate reason to reasonable length
    if len(reason) > 500:
        reason = reason[:497] + '...'

    # ---- Build output record ----
    return {
        'openalex_id': oid,
        'title': title,
        'decision': decision,
        'hard_exclusion': hard_excl,
        'hard_exclusion_reason': hard_excl_reason if hard_excl else '',
        'evidence_level': evidence_level,
        'african_centrality': centrality,
        'local_specificity': specificity,
        'scientific_depth': depth_val,
        'knowledge_value': knowledge_val,
        'local_applicability': applicability_val,
        'total_score': centrality + specificity + depth_val + knowledge_val + applicability_val,
        'african_focus': african_focus,
        'scientific_evidence': sci_ev,
        'african_country_codes': json.dumps(country_codes),
        'african_relevance_tags': json.dumps(tags),
        'evidence': evidence_quote[:200] if evidence_quote else '',
        'inference_basis': inference_basis,
        'reason': reason,
        # Metadata for traceability
        'field_name': field_name,
        'primary_topic': primary_topic,
        'work_type': work_type,
    }


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_batch(batch_records):
    """Process a batch of records, return list of result dicts."""
    results = []
    for record in batch_records:
        try:
            result = classify_record(record)
            results.append(result)
        except Exception as e:
            # Log error but continue processing
            oid = record.get('openalex_id', 'UNKNOWN')
            results.append({
                'openalex_id': oid,
                'title': str(record.get('title', '')),
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
                'african_focus': 'absent',
                'scientific_evidence': 'weak_or_none',
                'african_country_codes': '[]',
                'african_relevance_tags': '[]',
                'evidence': '',
                'inference_basis': '',
                'reason': f'Classification error: {str(e)[:100]}',
                'field_name': str(record.get('field_name', '')),
                'primary_topic': str(record.get('primary_topic', '')),
                'work_type': str(record.get('work_type', '')),
            })
    return results


# ============================================================================
# OUTPUT WRITERS
# ============================================================================

OUTPUT_COLUMNS = [
    'openalex_id', 'title', 'decision', 'hard_exclusion',
    'hard_exclusion_reason', 'evidence_level', 'african_centrality',
    'local_specificity', 'scientific_depth', 'knowledge_value',
    'local_applicability', 'total_score', 'african_focus',
    'scientific_evidence', 'african_country_codes', 'african_relevance_tags',
    'evidence', 'inference_basis', 'reason',
    'field_name', 'primary_topic', 'work_type',
]


def write_jsonl(results, filepath):
    """Write results as JSONL."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def write_csv(results, filepath):
    """Write results as CSV."""
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            row = {}
            for col in OUTPUT_COLUMNS:
                val = r.get(col, '')
                if isinstance(val, bool):
                    val = str(val).upper()
                row[col] = val
            writer.writerow(row)


def write_parquet(results, filepath):
    """Write results as Parquet."""
    # Build table with proper types
    data = {col: [] for col in OUTPUT_COLUMNS}
    for r in results:
        for col in OUTPUT_COLUMNS:
            val = r.get(col, '')
            if col in ('hard_exclusion',):
                val = bool(val)
            elif col in ('african_centrality', 'local_specificity',
                         'scientific_depth', 'knowledge_value',
                         'local_applicability', 'total_score'):
                val = int(val) if val != '' else 0
            data[col].append(val)

    table = pa.table(data)
    pq.write_table(table, filepath, compression='snappy')


# ============================================================================
# MAIN
# ============================================================================

def main():
    start = time.time()

    # Setup output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  MUFASA African Relevance Classifier")
    print("  Protocol: mufasa_african_relevance_reasoning_protocol.md")
    print("=" * 70)

    # ---- Read input ----
    print(f"\nReading: {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: Input file not found: {INPUT_CSV}")
        sys.exit(1)

    # Try multiple encodings to handle any non-UTF8 bytes
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(INPUT_CSV, usecols=USE_COLS, dtype=str, low_memory=False, encoding=enc)
            print(f"  Read successfully with encoding: {enc}")
            break
        except UnicodeDecodeError:
            continue
    else:
        df = pd.read_csv(INPUT_CSV, usecols=USE_COLS, dtype=str, low_memory=False, encoding='latin-1', encoding_errors='replace')
        print("  Read with latin-1 + error replacement")
    total = len(df)
    print(f"Total records loaded: {total:,}")

    # ---- Convert to list of dicts ----
    records = df.to_dict('records')
    del df  # free memory

    # ---- Create batches ----
    batches = []
    for i in range(0, len(records), BATCH_SIZE):
        batches.append(records[i:i + BATCH_SIZE])
    del records  # free memory

    num_batches = len(batches)
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches: {num_batches}")
    print(f"Worker processes: {NUM_WORKERS}")

    # ---- Process with multiprocessing ----
    print("\nClassifying records...")
    all_results = []
    completed = 0

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        for batch_results in pool.imap(process_batch, batches):
            all_results.extend(batch_results)
            completed += 1
            if completed % 10 == 0 or completed == num_batches:
                pct = completed * 100 / num_batches
                n_done = len(all_results)
                elapsed = time.time() - start
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else 0
                print(f"  [{completed}/{num_batches}] {pct:.0f}% | "
                      f"{n_done:,}/{total:,} records | "
                      f"{rate:.0f} rec/s | ETA {eta:.0f}s")

    elapsed = time.time() - start
    print(f"\nClassification complete: {len(all_results):,} records in {elapsed:.1f}s")
    print(f"Rate: {len(all_results)/elapsed:.0f} records/sec")

    # ---- Statistics ----
    decisions = Counter(r['decision'] for r in all_results)
    evidence_levels = Counter(r['evidence_level'] for r in all_results)
    focus_levels = Counter(r['african_focus'] for r in all_results)
    fields_dist = Counter(r['field_name'] for r in all_results if r.get('field_name'))

    print(f"\n{'='*50}")
    print("  CLASSIFICATION SUMMARY")
    print(f"{'='*50}")
    print(f"\n  Decisions:")
    for d, c in decisions.most_common():
        pct = c / len(all_results) * 100
        print(f"    {d:>10s}: {c:>8,}  ({pct:>5.1f}%)")

    print(f"\n  Evidence levels:")
    for e, c in evidence_levels.most_common():
        pct = c / len(all_results) * 100
        print(f"    {e:>20s}: {c:>8,}  ({pct:>5.1f}%)")

    print(f"\n  African focus:")
    for f, c in focus_levels.most_common():
        pct = c / len(all_results) * 100
        print(f"    {f:>12s}: {c:>8,}  ({pct:>5.1f}%)")

    print(f"\n  Top fields:")
    for f, c in fields_dist.most_common(5):
        print(f"    {f[:40]:40s}: {c:>8,}")

    # ---- Write outputs ----
    print(f"\nWriting outputs to: {OUTPUT_DIR}")

    # JSONL — split into 10 parts for readability
    NUM_JSONL_PARTS = 10
    print(f"\n  JSONL (split into {NUM_JSONL_PARTS} parts):")
    part_size = len(all_results) // NUM_JSONL_PARTS
    remainder = len(all_results) % NUM_JSONL_PARTS
    offset = 0
    for i in range(NUM_JSONL_PARTS):
        size = part_size + (1 if i < remainder else 0)
        part_records = all_results[offset:offset + size]
        offset += size
        part_path = os.path.join(OUTPUT_DIR, f'classified_african_relevance_part_{i+1:02d}.jsonl')
        write_jsonl(part_records, part_path)
        sz = os.path.getsize(part_path) / (1024 * 1024)
        print(f"    Part {i+1:2d}: {len(part_records):>7,} records  ({sz:.1f} MB)")

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, 'classified_african_relevance.csv')
    write_csv(all_results, csv_path)
    sz = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"\n  CSV:      {csv_path}  ({sz:.1f} MB)")

    # Parquet
    parquet_path = os.path.join(OUTPUT_DIR, 'classified_african_relevance.parquet')
    write_parquet(all_results, parquet_path)
    sz = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"  Parquet:  {parquet_path}  ({sz:.1f} MB)")

    total_elapsed = time.time() - start
    print(f"\n{'='*50}")
    print(f"  DONE! Total time: {total_elapsed:.1f}s")
    print(f"  Records classified: {len(all_results):,}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
