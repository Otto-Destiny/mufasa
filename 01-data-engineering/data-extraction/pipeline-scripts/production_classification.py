"""Converted from production-classification.ipynb.

Generated for readability in review; the notebook remains the
authority for anything that was actually run.
"""


# # MUFASA production classification
#
# The benchmark notebook, pointed at the full pool and made resumable.
#
# **The rubric, prompt, validation and decision rules are copied verbatim from
# `benchmark-200-kimi.ipynb`** - the run that scored 86.5% accuracy with 100%
# include retention. Nothing about how a paper is judged has changed. What is
# new is batching, resume and per-batch output.
#
# **Run it, close the laptop, come back, run it again.** It continues at the
# next unfinished batch. Interrupt it whenever you like.
#
# ### Output
#
# ```
# production/
#   manifest.json                 one file, appended after every batch
#   failures.jsonl                one file, appended - every failed paper
#   source_cache.parquet          the CSV columns we need, parsed once
#   batches/batch_00000.parquet   one per batch
# ```
#
# Nothing here touches `runs/` or the benchmark files.
#
# ### Source
#
# `openalex_ng_science_2000_2026/openalex_all_fields.csv` - 836 MB, **155,825
# papers**, 54 columns. It carries every field the benchmark prompt used, so the
# prompt needs no adaptation at all.
#
# The CSV is not UTF-8 and takes about 30s to parse, so the 18 columns we need
# are written once to `production/source_cache.parquet` and loaded from there
# on every later run.
#
# ### Size
#
# The benchmark measured **6.1s per paper** at 8 workers across 4 keys.
#
# | Scope | Papers | Time |
# |---|---|---|
# | One batch | 1,000 | ~1.7 h |
# | Whole pool | 155,825 | **~264 h (11 days)** |
#
# Wall-clock at 4 keys; concurrency scales with the number of keys. Use
# `MAX_BATCHES_THIS_RUN` to bound a single sitting.


# ## 1. Configuration


from pathlib import Path
from datetime import datetime, timezone

SOURCE   = Path("openalex_ng_science_2000_2026/openalex_all_fields.csv")
OUTDIR   = Path("production")
CACHE    = OUTDIR / "source_cache.parquet"
LICENCES = OUTDIR / "licence_cache.parquet"
BATCHES  = OUTDIR / "batches"
MANIFEST = OUTDIR / "manifest.json"
FAILURES = OUTDIR / "failures.jsonl"
for folder in (OUTDIR, BATCHES):
    folder.mkdir(parents=True, exist_ok=True)

# --- what to run ----------------------------------------------------------
BATCH_SIZE = 1000

# False resumes at the next unfinished batch - the normal case, and the default.
# True forgets all recorded progress and starts again at batch 0.
FRESH_START = False

# Stop after this many batches in one sitting. None runs to the end of the pool.
MAX_BATCHES_THIS_RUN = None

# --- which OpenAlex fields to queue ---------------------------------------
# The MUFASA domain is only known after a paper has been classified, so the
# only thing available for steering the queue is the OpenAlex field.
#
#   SKIP_FIELDS  drop these fields
#   ONLY_FIELDS  restrict to these fields
#
# Set both to None (or []) for the original behaviour - every paper, in source
# order. Names must match field_name exactly; section 5 checks them and stops
# if one does not match, so a typo cannot silently empty the queue.
# The twenty OpenAlex fields split into the two groups we alternate between.
# Whichever group is NOT being run goes into SKIP_FIELDS - so switching the run
# from one to the other is a one-word edit on the SKIP_FIELDS line.
LIFE_AND_HEALTH = [
    "Medicine",
    "Agricultural and Biological Sciences",
    "Environmental Science",
    "Biochemistry, Genetics and Molecular Biology",
    "Nursing",
    "Dentistry",
    "Neuroscience",
    "Pharmacology, Toxicology and Pharmaceutics",
    "Health Professions",
    "Immunology and Microbiology",
    "Veterinary",
    "Mathematics",
    "Chemistry",
]
PHYSICAL_AND_ENGINEERING = [
    "Engineering",
    "Computer Science",
    "Materials Science",
    "Earth and Planetary Sciences",
    "Energy",
    "Physics and Astronomy",
    "Chemical Engineering",
]

# Running LIFE_AND_HEALTH, so the physical group is the one being skipped.
# Put LIFE_AND_HEALTH here instead to go back to the engineering run.
SKIP_FIELDS = PHYSICAL_AND_ENGINEERING
ONLY_FIELDS = None

# --- which licence tiers to queue -----------------------------------------
# Tiers are defined in 03-retrieval/licence-tiers.md and resolved from the
# source's own licence columns:
#
#   1  permissive       cc-by, cc-by-sa, cc0, public-domain   - quotable
#   2  non-commercial   cc-by-nc, cc-by-nc-sa
#   3  no-derivatives   cc-by-nc-nd, cc-by-nd
#   4  none stated      the source carries no licence at all
#   5  other / unclear  other-oa and malformed values
#
# A list of the tiers to keep, e.g. [1] or [1, 2]. None switches the licence
# filter off entirely - the licence file is not even read, so a run without it
# behaves exactly as before.
#
# [1, 4] is the working set: tier 1 is quotable today, and tier 4 is not
# restricted - it is simply unrecorded, mostly diamond-OA journals that never
# registered a licence with Crossref. Classifying them now keeps the option
# open; a later Crossref or Unpaywall lookup can promote the ones that are
# genuinely permissive.
LICENCE_TIERS = [1]

# --- how each batch is put together ---------------------------------------
# BATCH_SHAPING is the master switch. False ignores everything below and fills
# batches straight from source order, exactly as the notebook did before any of
# this existed.
BATCH_SHAPING = True

# Queue the most-cited papers first. Sorting happens inside each field, so it
# composes with STRATIFY instead of fighting it.
ORDER_BY_CITATIONS = True

# Draw a set number of papers from every field in each batch. Independent of
# ORDER_BY_CITATIONS - either can be on without the other.
STRATIFY = True

# Papers per field per batch. When this is set, THE BATCH SIZE IS THE SUM OF
# THESE NUMBERS - BATCH_SIZE is not consulted at all. So the batch is exactly
# as big as the contributions written here:
#
#   FIELD_QUOTAS = {"Engineering": 400, "Computer Science": 250,
#                   "Materials Science": 150, "Earth and Planetary Sciences": 100,
#                   "Energy": 50, "Physics and Astronomy": 40,
#                   "Chemical Engineering": 10}          # -> batches of 1000
#
# A field left out of the dict contributes nothing. None instead splits
# BATCH_SIZE evenly across whatever fields survive SKIP_FIELDS / ONLY_FIELDS.
FIELD_QUOTAS = None

# What to do when a field runs out and cannot fill its quota.
#   True  - its unused places pass to the fields that still have papers, so
#           every batch stays the full size the quotas ask for. Section 5 still
#           prints which field ran out and which fields absorbed its places.
#   False - the batch is simply left short, which is useful while tuning
#           because the gap is impossible to miss.
TOP_UP_SHORTFALL = True

# --- model: identical to the benchmark ------------------------------------
BASE_URL = "https://api.tokenrouter.com/v1"
MODEL    = "moonshotai/kimi-k3-free"

WORKERS_PER_KEY  = 2
TIMEOUT          = 30
RETRIES          = 4
MAX_OUTPUT       = 1024
REVIEW_THRESHOLD = 8
STRICT_REASON    = True

print(f"source : {SOURCE}")
print(f"output : {OUTDIR}/")
print(f"batch  : {BATCH_SIZE} papers")
print(f"mode   : {'FRESH START - begins at batch 0' if FRESH_START else 'resume where it stopped'}")
if SKIP_FIELDS or ONLY_FIELDS:
    print(f"filter : {'only ' + str(len(ONLY_FIELDS)) if ONLY_FIELDS else ''}"
          f"{' / ' if ONLY_FIELDS and SKIP_FIELDS else ''}"
          f"{'skip ' + str(len(SKIP_FIELDS)) if SKIP_FIELDS else ''} OpenAlex field(s)")
else:
    print("filter : none - every field")
print(f"licence: {'tier ' + ', '.join(map(str, LICENCE_TIERS)) if LICENCE_TIERS else 'not filtered'}")
if BATCH_SHAPING:
    print(f"shaping: {'most-cited first' if ORDER_BY_CITATIONS else 'source order'}"
          f"{', per-field quotas' if STRATIFY else ', no quotas'}"
          f"{' (hand-set)' if STRATIFY and FIELD_QUOTAS else ''}"
          f"{', top up shortfall' if STRATIFY and TOP_UP_SHORTFALL else ''}")
else:
    print("shaping: off - plain source order")


# ## 2. API keys
#
# Verbatim from the benchmark. Every `TOKENROUTER_API_KEY*` in `.env` is used,
# and concurrency is `WORKERS_PER_KEY x keys`, because limits are per account.


import os
from pathlib import Path

ENV_FILE = Path(".env")

def read_env(path):
    """Minimal .env parser - no dependency, ignores comments and blank lines."""
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values

# Collect every TOKENROUTER_API_KEY* found. Shell wins over the file.
# Any name starting with the prefix counts, so adding a 5th key needs no code
# change - just another line in .env.
merged = {**read_env(ENV_FILE),
          **{k: v for k, v in os.environ.items() if k.startswith("TOKENROUTER_API_KEY")}}
API_KEYS = [(name, merged[name]) for name in sorted(merged)
            if name.startswith("TOKENROUTER_API_KEY") and merged[name].strip()]

if not API_KEYS:
    raise SystemExit(
        f"No TOKENROUTER_API_KEY* found.\n"
        f"Add to {ENV_FILE.resolve()}\n"
        f"    TOKENROUTER_API_KEY=sk-...\n"
        f"    TOKENROUTER_API_KEY2=sk-...")

# Distinct accounts mean distinct rate limits. Duplicates would not add any.
unique = {value for _, value in API_KEYS}
if len(unique) != len(API_KEYS):
    raise SystemExit(f"{len(API_KEYS)} keys but only {len(unique)} are distinct - "
                     "duplicates share a rate limit and add nothing.")

WORKERS = WORKERS_PER_KEY * len(API_KEYS)

for name, value in API_KEYS:
    print(f"  {name:<24} {value[:6]}...{value[-4:]}")
print(f"\n{len(API_KEYS)} keys x {WORKERS_PER_KEY} workers = {WORKERS} concurrent requests")


# ## 3. Rubric
#
# **Verbatim from the benchmark.** Editing this cell means production no longer
# matches a measured result - re-run the benchmark first if you must change it.


RUBRIC_CORE = """You screen scientific papers for MUFASA.
The one question: is this paper scientifically ABOUT Africa, or merely BY Africans?
"About Africa" means the research question, evidence, materials, organisms, populations, conditions or application are fundamentally African. "By Africans" means the authors sit at African institutions but the science could have been done anywhere.

HARD EXCLUSIONS - check these FIRST. Any one forces exclude regardless of scores:
1. Affiliation only - title, abstract and keywords contain no African material, organism, location, condition or research question, and the sole African signal is an author institution or country. Test: remove the affiliations; would you still know this concerns Africa? If no, exclude.
2. Non-African research region - the study is explicitly about Peru, China, Korea, Europe, the US or elsewhere, and Africa appears only as a co-author affiliation.
3. Outside STEM - purely economics, policy, law, business, education, sociology or humanities with no measurement, experiment, model or data. This also covers any study whose only data is what people reported about themselves: knowledge-attitudes-practices, acceptability, willingness, perception, awareness, satisfaction, self-reported behaviour or lifestyle. Asking people what they think is social science, not measurement. It is NOT excluded when something physical or biological was measured on the participants - anthropometry, blood pressure, biomarkers, weighed food intake, clinical tests, specimens - even if a questionnaire was used as well. Climate impact on crop yields is STEM; climate policy and emissions trading is not. Two worked cases: "awareness and acceptance of donor human milk among 303 mothers" is excluded - the whole finding is what mothers said. "Nutrient intakes of 345 toddlers in Ibadan creches, weighed 24-hour recall against EAR" is NOT excluded - food was weighed, so something was measured even though carers were also questioned.
4. Editorial or advocacy - a call to action, quote, policy statement or announcement carrying no original data.
5. Retracted - the work type says so.

DO NOT EXCLUDE ONLY FOR ABSENCE OF DIRECT EVIDENCE. This narrow rule never overrides the five hard exclusions above; those still force exclude. It applies in one situation: the material, organism, crop, mineral, waste or population under study is one known to occur in or originate from Africa, but no African place or context is stated. There, treat the subject itself as African content, score it, and return review rather than exclude - including when the work was done abroad. A paper wrongly excluded is lost for good; a paper wrongly sent to review costs a human a minute.

Author institutions and affiliation countries are supplied ONLY for hard exclusion 1. They never raise african_centrality or local_specificity - only research content counts.

Score 0-4 each: african_centrality, local_specificity, scientific_depth, knowledge_value, local_applicability.
  centrality  4 African country in the title and the question is African | 3 African materials or organisms are the subject but Africa is not in the title | 2 some African connection, not the main focus | 1 affiliation or one incidental mention only | 0 nothing African
  specificity 4 names African places, materials or species integral to the design | 3 African resources meaningfully referenced, less geographic precision | 2 present but thinly specified | 1 vague, e.g. "tropical" | 0 none
  depth       4 laboratory experiment, field measurement, instrument or assay reading | 3 measured survey of the physical or biological world (prevalence with laboratory confirmation, biodiversity counts, seismic or geophysical survey, remote sensing) or a validated computational study | 2 review, limited data, or data that is only self-reported | 1 perspective or brief note | 0 editorial
  knowledge   4 rich extractable findings | 3 useful citable results | 2 confirmatory or summarising | 1 minimal | 0 none
  applicability 4 addresses a specifically African problem | 3 transferable to African settings | 2 generic but usable | 1 marginal | 0 none

Return one minified JSON object and nothing else. ALL TEN keys are required:
{"decision":"include|review|exclude","hard_exclusion":false,"african_centrality":0,"local_specificity":0,"scientific_depth":0,"knowledge_value":0,"local_applicability":0,"mufasa_domain":"","evidence":"","reason":""}

"hard_exclusion" must be present, true or false. Never omit it.
"mufasa_domain" is exactly one of MAT, AGR, HLT, ENR, ENV, TEC, OUTSIDE_TAXONOMY, assigned from the research CONTENT rather than the OpenAlex field:
  MAT materials, manufacturing, infrastructure - building materials, soil stabilisation, construction, roads, minerals, ores, metallurgy, ceramics, composites, machining, biomass and agricultural-waste utilisation, recycling, corrosion
  AGR agriculture, food, biological sciences - crops, breeding, plant pathology, pest control, soils, food processing, post-harvest storage, livestock, veterinary, fisheries, agricultural biotechnology, agroforestry, biodiversity, conservation
  HLT health, medicine, biotechnology - medicinal plants, natural products, pharmacognosy, venoms, disease vectors, diagnostics, vaccines, therapeutics, antimicrobial resistance, parasitology, pathogen genomics, medical devices, nutritional biochemistry
  ENR energy, petroleum, mining - reservoirs, drilling, gas processing, solar, wind, hydro, geothermal, biofuels, batteries, hydrogen, clean cooking, mine safety and tailings
  ENV water, earth, environment - water purification, wastewater, groundwater, hydrology, drought, geology, seismology, erosion, climate modelling, atmospheric pollution, coastal and marine, wetlands, toxicology, remote sensing
  TEC computing, electronics, applied engineering - precision-agriculture sensors, geospatial computing, machine learning on African datasets, embedded and low-power electronics, microgrid control, rural networks, IoT, cold chain, drones, robotics, instrumentation
  OUTSIDE_TAXONOMY economics, policy, education, law, business, pure mathematics, pure physics with no application, social sciences, humanities
  Disambiguation: antimicrobial resistance in livestock is AGR, in hospital patients HLT. Cassava waste made into bioplastic is MAT, not AGR. Soot degrading solar panels is ENR, not ENV.
"evidence" must be 5 to 20 words copied verbatim from the title or abstract. Never empty, never paraphrased.
"""

REASON_STRICT = """"reason" must be at most 45 words and specific to THIS paper:
- Name the actual material, organism, place, population, measurement or condition it studies.
- Say what makes the African element load-bearing for the science rather than incidental. If the same work could have been done anywhere with any inputs, say so.
- If the decision was close, name the single thing that tipped it.
- For exclude, name what is missing, not what is present.
Never use these phrases: "locally sourced", "valorizes", "addresses local needs", "directly applicable", "sustainable development", "local relevance", "African context".
A reason that could be copied onto a different paper without editing is wrong. Write the sentence only this paper's abstract could produce.
"""

REASON_PLAIN = '"reason" is at most 30 words in your own words.\n'

RUBRIC = RUBRIC_CORE + (REASON_STRICT if STRICT_REASON else REASON_PLAIN) + """
Use only the supplied metadata. Prefer review over an unsupported exclusion."""

REPAIR = ("\nYour previous reply was rejected. Return the JSON object again with "
          "every one of the nine keys present, including hard_exclusion and a "
          "non-empty verbatim evidence quote.")

import html, json, math, re

def clean(v, limit=None):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    t = re.sub(r"\s+", " ", html.unescape(str(v))).strip()
    return t[:limit] if limit else t

def render_list(value, maximum=6, limit=200):
    """Flatten an OpenAlex JSON column into a short readable list."""
    text = clean(value)
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except Exception:
        return clean(text, limit)
    names = []
    for item in payload if isinstance(payload, list) else []:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = (item.get("display_name") or item.get("name")
                    or item.get("keyword") or item.get("country_code") or "")
        else:
            name = ""
        name = clean(name, 60)
        if name and name not in names:
            names.append(name)
        if len(names) >= maximum:
            break
    return clean("; ".join(names), limit)

def build_prompt(row, repair=False):
    return (
        RUBRIC + (REPAIR if repair else "")
        + "\n\nPAPER"
        + "\nTitle: " + clean(row.get("title"), 300)
        + "\nOpenAlex field: " + clean(row.get("field_name") or row.get("primary_field"), 100)
        + "\nSubfield: " + clean(row.get("primary_subfield"), 100)
        + "\nTopic: " + clean(row.get("primary_topic"), 140)
        + "\nKeywords: " + render_list(row.get("keywords_json"), 8, 220)
        + "\nJournal: " + clean(row.get("journal"), 120)
        + "\nWork type: " + clean(row.get("work_type"), 40)
        + "\nAuthor institutions (affiliation only): " + render_list(row.get("institutions_json"), 6, 240)
        + "\nAffiliation countries (affiliation only): " + render_list(row.get("countries_json"), 10, 120)
        + "\nAbstract: " + clean(row.get("abstract"), 5000)
    )

print(f"rubric is {len(RUBRIC)} characters")


# ## 4. Validation and decision rules
#
# **Verbatim from the benchmark**, including the rule that honours the model
# when it says review rather than overriding it to include.


REQUIRED = {"decision", "hard_exclusion", "african_centrality", "local_specificity",
            "scientific_depth", "knowledge_value", "local_applicability",
            "mufasa_domain", "evidence", "reason"}
SCORES = ("african_centrality", "local_specificity", "scientific_depth",
          "knowledge_value", "local_applicability")
DOMAINS = {"MAT", "AGR", "HLT", "ENR", "ENV", "TEC", "OUTSIDE_TAXONOMY"}

def parse(text):
    """Return the validated payload, or raise with the reason it failed."""
    body = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    body = body.replace("```json", "").replace("```", "").strip()
    a, b = body.find("{"), body.rfind("}")
    if a < 0 or b < a:
        raise ValueError("no complete JSON object")
    payload = json.loads(body[a:b + 1])

    missing = REQUIRED - set(payload)
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    if payload["decision"] not in {"include", "review", "exclude"}:
        raise ValueError(f"bad decision: {payload['decision']!r}")
    if clean(payload["mufasa_domain"]).upper() not in DOMAINS:
        raise ValueError(f"bad mufasa_domain: {payload['mufasa_domain']!r}")
    if not clean(payload["evidence"]):
        raise ValueError("evidence is empty")

    scores = {}
    for name in SCORES:
        v = payload[name]
        if isinstance(v, bool):
            raise ValueError(f"{name} is a bool")
        v = float(v)
        if not v.is_integer() or not 0 <= v <= 4:
            raise ValueError(f"{name}={v} outside 0-4")
        scores[name] = int(v)
    return payload, scores

def final_decision(said, hard, s):
    """Thresholds from the Qwen reasoning protocol, section 4.2.

    The 10-13 band is what stops a merely low-scoring paper from being thrown
    away: it goes to review, where a human can still rescue it. Only a hard
    exclusion or a genuinely empty score drops a paper for good.
    """
    if hard:
        return "exclude"
    # Honour the model when it hedges. On the 200-paper run the rule overrode
    # the model 24 times and every one was "said review, forced include" -
    # discarding the judgement we asked for. Respecting it moved accuracy from
    # 80.5% to 83.0% and put the include share on 34%, exactly gold's.
    if said == "review":
        return "review"
    total = sum(s.values())
    if total >= 14 and s["african_centrality"] >= 2 and s["scientific_depth"] >= 2:
        # Protocol section 4.3 rule 2: high total but no African focus is a
        # review, not an include - the paper must be ABOUT Africa.
        return "review" if s["african_centrality"] <= 1 else "include"
    if total >= REVIEW_THRESHOLD:
        return "review"
    return "exclude"

import json
print("required keys:", len(REQUIRED))


# ## 5. Load the pool and find where to resume
#
# A paper's batch is fixed by its row position, so the same paper always lands
# in the same batch. The manifest stores a fingerprint of the source; if the
# file ever changes, the next run refuses rather than silently mixing two
# different orderings.


import hashlib
import pandas as pd

# 18 of the CSV's 54 columns - everything the prompt and the output need.
KEEP = ["openalex_id", "doi", "field_id", "field_name", "title", "abstract",
        "primary_field", "primary_subfield", "primary_topic", "keywords_json",
        "journal", "work_type", "is_retracted", "institutions_json",
        "countries_json", "publication_date", "publication_year", "cited_by_count"]

if CACHE.exists():
    pool = pd.read_parquet(CACHE)
    print(f"loaded {CACHE.name}")
else:
    print(f"first run - parsing {SOURCE.name} (836 MB, about 30s)...")
    pool = pd.read_csv(SOURCE, usecols=KEEP, encoding="cp1252",
                       encoding_errors="replace", low_memory=False)
    pool.to_parquet(CACHE, index=False)
    print(f"cached {len(pool):,} rows - later runs load in about a second")

assert pool["openalex_id"].is_unique, "openalex_id must be unique"
TOTAL_BATCHES = (len(pool) + BATCH_SIZE - 1) // BATCH_SIZE

fingerprint = hashlib.sha256(
    f"{len(pool)}|{pool.openalex_id.iloc[0]}|{pool.openalex_id.iloc[-1]}".encode()
).hexdigest()[:16]

def load_manifest():
    if MANIFEST.exists() and not FRESH_START:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if data.get("fingerprint") != fingerprint:
            raise SystemExit(
                "The source has changed since the last run, so batch numbers would "
                "no longer line up.\n"
                f"  manifest: {data.get('fingerprint')}\n"
                f"  current : {fingerprint}\n"
                "Restore the original source, or set FRESH_START = True.")
        return data
    return {"source": SOURCE.name, "source_rows": int(len(pool)),
            "fingerprint": fingerprint, "batch_size": BATCH_SIZE,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "batches": {}}

manifest = load_manifest()

def save_manifest():
    """Write through a temp file so an interrupt cannot leave it half-written."""
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    tmp.replace(MANIFEST)

# A batch counts as done only if the manifest says so AND its file is present.
done = {int(k) for k, v in manifest["batches"].items()
        if v.get("complete") and (BATCHES / f"batch_{int(k):05d}.parquet").exists()}

# Which papers have already been sent. Read from the batch files rather than
# from row positions, so the queue can be filtered without disturbing any
# batch already on disk. A paper in a finished batch is never queued again.
sent = set()
for batch_no in sorted(done):
    sent |= set(pd.read_parquet(BATCHES / f"batch_{batch_no:05d}.parquet",
                                columns=["openalex_id"])["openalex_id"])

# Fail loudly on a misspelled field rather than quietly queueing nothing.
present = set(pool["field_name"].dropna().unique())
unknown = [f for f in (SKIP_FIELDS or []) + (ONLY_FIELDS or []) if f not in present]
if unknown:
    raise SystemExit("These field names do not appear in the pool:\n  "
                     + "\n  ".join(unknown)
                     + "\n\nAvailable:\n  " + "\n  ".join(sorted(present)))

eligible = pool
if ONLY_FIELDS:
    eligible = eligible[eligible["field_name"].isin(ONLY_FIELDS)]
if SKIP_FIELDS:
    eligible = eligible[~eligible["field_name"].isin(SKIP_FIELDS)]

# The licence lives in the source CSV but not in the 18 cached columns, so it
# is read once into its own parquet and joined by openalex_id. Skipped
# entirely when LICENCE_TIERS is None.
if LICENCE_TIERS:
    unknown_tiers = [t for t in LICENCE_TIERS if t not in (1, 2, 3, 4, 5)]
    if unknown_tiers:
        raise SystemExit(f"LICENCE_TIERS accepts 1-5 or None, got {unknown_tiers}")

    if LICENCES.exists():
        licences = pd.read_parquet(LICENCES)
    else:
        print(f"first run with a licence filter - reading {SOURCE.name}...")
        licences = pd.read_csv(
            SOURCE, encoding="cp1252", encoding_errors="replace", low_memory=False,
            usecols=["openalex_id", "best_oa_license", "primary_license",
                     "download_license"])
        licences.to_parquet(LICENCES, index=False)
        print(f"cached {LICENCES.name}")

    def resolved_licence(frame):
        """First non-empty of the three licence columns, normalised."""
        out = pd.Series("", index=frame.index)
        for column in ("best_oa_license", "primary_license", "download_license"):
            if column in frame:
                value = (frame[column].fillna("").astype(str).str.strip()
                         .str.lower().replace({"nan": "", "none": ""}))
                out = out.where(out != "", value)
        return out

    NAMED_TIERS = {1: {"cc-by", "cc-by-sa", "cc0", "public-domain"},
                   2: {"cc-by-nc", "cc-by-nc-sa"},
                   3: {"cc-by-nc-nd", "cc-by-nd"}}

    def to_tier(licence):
        for tier, names in NAMED_TIERS.items():
            if licence in names:
                return tier
        return 4 if licence == "" else 5      # 4 is absent, 5 is present but odd

    licences["tier"] = resolved_licence(licences).map(to_tier)
    wanted = set(licences.loc[licences["tier"].isin(LICENCE_TIERS), "openalex_id"])
    before = len(eligible)
    eligible = eligible[eligible["openalex_id"].isin(wanted)]
    print(f"licence    : tier {LICENCE_TIERS} keeps {len(eligible):,} of {before:,}")

# Source order is preserved, so with no filter this is the same sequence the
# old positional slicing produced.
queue_frame = eligible[~eligible["openalex_id"].isin(sent)]

# New batches take the next free numbers, so they can never collide with a
# batch already recorded in the manifest.
next_batch_no = (max(done) + 1) if done else 0

QUOTAS = {}          # field -> papers per batch, filled in by build_plan
SHORTFALL = {}       # field -> places it could not fill, first batch
ABSORBED = {}        # field -> extra places it took on, first batch


def slice_evenly(frame, start_no):
    """Consecutive batches, in whatever order the frame already has."""
    return [(start_no + i, frame.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
            for i in range((len(frame) + BATCH_SIZE - 1) // BATCH_SIZE)]


def build_plan(frame, start_no):
    """Decide which papers go in which batch.

    Off  -> straight slices of the frame as it stands, the original behaviour.
    On   -> optionally sort by citations, then optionally draw a quota from
            every field. A field that runs out is recorded in SHORTFALL and,
            unless TOP_UP_SHORTFALL is set, the batch is simply left short so
            the gap is visible rather than hidden.
    """
    QUOTAS.clear()
    SHORTFALL.clear()
    ABSORBED.clear()

    if not BATCH_SHAPING:
        return slice_evenly(frame, start_no)

    ordered = frame
    if ORDER_BY_CITATIONS:
        ordered = frame.assign(
            _cites=pd.to_numeric(frame["cited_by_count"], errors="coerce").fillna(-1)
        ).sort_values("_cites", ascending=False, kind="stable").drop(columns="_cites")

    if not STRATIFY:
        return slice_evenly(ordered, start_no)

    # One queue per field, each already in the order we mean to spend it.
    queues = {name: list(group.index)
              for name, group in ordered.groupby("field_name", sort=False)}
    fields = sorted(f for f in queues if queues[f])
    if not fields:
        return []

    if FIELD_QUOTAS:
        unknown = [f for f in FIELD_QUOTAS if f not in queues]
        if unknown:
            raise SystemExit("FIELD_QUOTAS names fields that are not in the queue:\n  "
                             + "\n  ".join(unknown)
                             + "\n\nIn the queue:\n  " + "\n  ".join(fields))
        QUOTAS.update({f: int(FIELD_QUOTAS.get(f, 0)) for f in fields})
    else:
        base, extra = divmod(BATCH_SIZE, len(fields))
        QUOTAS.update({f: base + (1 if i < extra else 0) for i, f in enumerate(fields)})

    # The batch is as big as the quotas say, not as big as BATCH_SIZE says.
    target = sum(QUOTAS.values())

    plan, number, first = [], start_no, True
    while any(queues.values()):
        picked = []
        for field in fields:
            want = QUOTAS[field]
            take = min(want, len(queues[field]))
            picked += [queues[field].pop(0) for _ in range(take)]
            if first and take < want:
                SHORTFALL[field] = want - take
        if TOP_UP_SHORTFALL:
            while len(picked) < target:
                spare = sorted((f for f in fields if queues[f]),
                               key=lambda f: -len(queues[f]))
                if not spare:
                    break
                for field in spare:
                    if len(picked) >= target:
                        break
                    picked.append(queues[field].pop(0))
                    if first:
                        ABSORBED[field] = ABSORBED.get(field, 0) + 1
        if not picked:
            break
        plan.append((number, ordered.loc[picked]))
        number += 1
        first = False
    return plan


PLAN = build_plan(queue_frame, next_batch_no)

print(f"\npool       : {len(pool):,} papers")
print(f"completed  : {len(done)} batches ({len(sent):,} papers)")
if SKIP_FIELDS or ONLY_FIELDS:
    print(f"eligible   : {len(eligible):,} papers after the field filter")
print(f"queued     : {len(queue_frame):,} papers -> {len(PLAN)} batches")
if not PLAN:
    print("\nnothing left to do")
else:
    first = PLAN[0][1]
    target = sum(QUOTAS.values()) if QUOTAS else BATCH_SIZE
    short = target - len(first)
    print(f"\nnext batch : {PLAN[0][0]}  ({len(first):,} papers"
          + (f", {short} SHORT of the {target} the quotas ask for" if short > 0 else "")
          + ")")

    if BATCH_SHAPING and STRATIFY:
        got = first["field_name"].value_counts()
        have = queue_frame["field_name"].value_counts()
        print(f"\nbatch size = sum of the quotas = {target}")
        print(f"\n{'field':<38}{'quota':>7}{'in queue':>10}{'taken':>7}{'short':>7}")
        for field in sorted(QUOTAS):
            gap = QUOTAS[field] - int(got.get(field, 0))
            print(f"{field[:38]:<38}{QUOTAS[field]:>7}{int(have.get(field, 0)):>10}"
                  f"{int(got.get(field, 0)):>7}{(gap if gap > 0 else ''):>7}")
        if SHORTFALL:
            print(f"\n{len(SHORTFALL)} field(s) could not fill their quota:")
            for field, missing in sorted(SHORTFALL.items(), key=lambda kv: -kv[1]):
                print(f"  {field:<40} short {missing:>4}   "
                      f"({int(have.get(field, 0))} in queue, quota {QUOTAS[field]})")
            if TOP_UP_SHORTFALL and ABSORBED:
                print("\nThose places were passed to:")
                for field, extra in sorted(ABSORBED.items(), key=lambda kv: -kv[1]):
                    print(f"  {field:<40}  +{extra:<4}  -> {QUOTAS[field] + extra} this batch")
                print(f"\nBatch stays full at {len(first)}. Set TOP_UP_SHORTFALL = False "
                      "to leave it short instead, or set FIELD_QUOTAS by hand.")
            else:
                print(f"\nBatch is {short} short. Set TOP_UP_SHORTFALL = True to pass "
                      "those places to fields that still have papers, or edit FIELD_QUOTAS.")
    else:
        print(first["field_name"].value_counts().to_string())

    if BATCH_SHAPING and ORDER_BY_CITATIONS:
        cites = pd.to_numeric(first["cited_by_count"], errors="coerce")
        rest = pd.to_numeric(queue_frame["cited_by_count"], errors="coerce")
        print(f"\ncitations  : batch median {cites.median():.0f}, mean {cites.mean():.1f}, "
              f"top {cites.max():.0f}   (whole queue median {rest.median():.0f}, "
              f"mean {rest.mean():.1f})")


# ## 6. Run
#
# Each batch is classified, written to its own parquet, and only then recorded
# in the manifest - so an interrupted batch is redone whole and never
# half-counted.
#
# Rate limiting is handled twice over. Per request: exponential backoff that
# triples on a 429 and rotates to a different key, exactly as the benchmark
# does. Per batch: an adaptive pause that grows when a batch sees many 429s and
# decays when it does not, so a busy period slows the run instead of failing it.


import concurrent.futures, random, time
from openai import OpenAI
from tqdm.auto import tqdm

CLIENTS = [OpenAI(base_url=BASE_URL, api_key=value) for _, value in API_KEYS]
KEY_NAMES = [name for name, _ in API_KEYS]

throttle = {"delay": 0.0, "rate_limited": 0}

def classify(item):
    """The benchmark's classify, plus openalex_id on the row and the adaptive
    delay so a busy pool slows us down rather than failing us."""
    index, row = item
    started = time.perf_counter()
    last = ""
    for attempt in range(RETRIES):
        slot = (index + attempt) % len(CLIENTS)
        if throttle["delay"]:
            time.sleep(throttle["delay"] * random.uniform(0.5, 1.5))
        try:
            r = CLIENTS[slot].chat.completions.create(
                model=MODEL,
                messages=[{"role": "user",
                           "content": build_prompt(row, repair=attempt > 0)}],
                temperature=0,
                max_tokens=MAX_OUTPUT,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                timeout=TIMEOUT,
            )
            payload, scores = parse(r.choices[0].message.content)
            said = payload["decision"]
            hard = str(payload["hard_exclusion"]).lower() in {"true", "1", "yes"}
            out = {
                "openalex_id": row["openalex_id"],
                "model_recommended_decision": said,
                "model_decision": final_decision(said, hard, scores),
                "model_hard_exclusion": hard,
                "model_total_score": sum(scores.values()),
                "model_mufasa_domain": clean(payload["mufasa_domain"]).upper(),
                "model_evidence": clean(payload["evidence"], 300),
                "model_reason": clean(payload["reason"], 500),
                "model_valid_json": True,
                "model_error": "",
                "model_attempts": attempt + 1,
                "model_key": KEY_NAMES[slot],
                "model_latency_seconds": round(time.perf_counter() - started, 2),
                "model_prompt_tokens": r.usage.prompt_tokens,
                "model_output_tokens": r.usage.completion_tokens,
            }
            out.update({f"model_{k}": v for k, v in scores.items()})
            return out
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"[:200]
            if "429" in last or "RateLimit" in last:
                throttle["rate_limited"] += 1
            if attempt < RETRIES - 1:
                wait = min(2 ** attempt + random.random(), 20)
                if "429" in last or "RateLimit" in last:
                    wait = min(wait * 3, 45)
                time.sleep(wait)

    failed = {
        "openalex_id": row["openalex_id"],
        "model_recommended_decision": "review", "model_decision": "review",
        "model_hard_exclusion": False, "model_total_score": None,
        "model_mufasa_domain": "", "model_evidence": "", "model_reason": "",
        "model_valid_json": False, "model_error": last, "model_attempts": RETRIES,
        "model_key": "all keys failed",
        "model_latency_seconds": round(time.perf_counter() - started, 2),
        "model_prompt_tokens": None, "model_output_tokens": None,
    }
    failed.update({f"model_{k}": None for k in SCORES})
    return failed


def run_batch(batch_no, frame):
    throttle["rate_limited"] = 0
    started_at = datetime.now(timezone.utc)
    clock = time.perf_counter()

    items = list(enumerate(frame.to_dict("records")))
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as workers:
        results = list(tqdm(workers.map(classify, items), total=len(items),
                            desc=f"batch {batch_no:05d}", leave=False))
    elapsed = time.perf_counter() - clock

    out = frame.reset_index(drop=True).merge(
        pd.DataFrame(results), on="openalex_id", how="left")

    path = BATCHES / f"batch_{batch_no:05d}.parquet"
    out.to_parquet(path, index=False)                       # data first

    bad = out[~out["model_valid_json"].astype(bool)]
    if len(bad):
        with FAILURES.open("a", encoding="utf-8") as handle:
            for _, row in bad.iterrows():
                handle.write(json.dumps({
                    "batch": batch_no, "openalex_id": row["openalex_id"],
                    "error": row["model_error"],
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }, ensure_ascii=False) + "\n")

    manifest["batches"][str(batch_no)] = {                  # manifest last
        "file": path.name,
        "papers": int(len(out)),
        "ok": int(out["model_valid_json"].sum()),
        "failed": int(len(bad)),
        "rate_limited": int(throttle["rate_limited"]),
        "seconds": round(elapsed, 1),
        "decisions": out["model_decision"].value_counts().to_dict(),
        "domains": out["model_mufasa_domain"].replace("", "(failed)").value_counts().to_dict(),
        "fields": sorted(frame["field_name"].dropna().unique().tolist()),
        "licence_tiers": LICENCE_TIERS,
        "shaping": ({"cited_first": ORDER_BY_CITATIONS, "stratified": STRATIFY,
                     "quotas": FIELD_QUOTAS} if BATCH_SHAPING else None),
        "started": started_at.isoformat(timespec="seconds"),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "complete": True,
    }
    save_manifest()

    share = throttle["rate_limited"] / max(len(out), 1)
    if share > 0.05:
        throttle["delay"] = min(throttle["delay"] + 0.25, 3.0)
    elif share == 0:
        throttle["delay"] = max(throttle["delay"] - 0.10, 0.0)
    return manifest["batches"][str(batch_no)]


plan = PLAN[:MAX_BATCHES_THIS_RUN] if MAX_BATCHES_THIS_RUN else PLAN
print(f"{len(plan)} batches queued this session\n")

for batch_no, frame in plan:
    stats = run_batch(batch_no, frame)
    counts = stats["decisions"]
    print(f"batch {batch_no:05d}  ok {stats['ok']:>4}/{stats['papers']}  "
          f"failed {stats['failed']:>3}  429s {stats['rate_limited']:>3}  "
          f"{stats['seconds']/60:5.1f} min  "
          f"inc {counts.get('include', 0):>4} rev {counts.get('review', 0):>4} "
          f"exc {counts.get('exclude', 0):>4}"
          + (f"  delay {throttle['delay']:.2f}s" if throttle["delay"] else ""))

print("\nsession finished")


# ## 7. Progress
#
# There is no accuracy here. Production has no gold labels, and the decision
# split is whatever the corpus turns out to be - we are not steering it toward
# an expected shape. These are throughput and health numbers.


rows = []
for key, value in sorted(manifest["batches"].items(), key=lambda kv: int(kv[0])):
    rows.append({"batch": int(key), "papers": value["papers"], "ok": value["ok"],
                 "failed": value["failed"], "429s": value.get("rate_limited", 0),
                 "min": round(value["seconds"] / 60, 1),
                 **{k: value["decisions"].get(k, 0)
                    for k in ("include", "review", "exclude")}})

if not rows:
    print("no batches completed yet")
else:
    progress = pd.DataFrame(rows)
    display(progress.tail(15))

    papers = int(progress["papers"].sum())
    failed = int(progress["failed"].sum())
    minutes = float(progress["min"].sum())
    per_paper = minutes * 60 / max(papers, 1)

    print(f"\nbatches     : {len(progress)} of {TOTAL_BATCHES}  "
          f"({len(progress)/TOTAL_BATCHES:.1%})")
    print(f"papers      : {papers:,} of {len(pool):,}")
    print(f"failed      : {failed:,} ({failed/max(papers,1):.2%})")
    print(f"rate limits : {int(progress['429s'].sum()):,}")
    print(f"elapsed     : {minutes/60:.1f} h at {per_paper:.1f}s per paper")

    left = len(pool) - papers
    if left > 0:
        print(f"remaining   : {left:,} papers, roughly {left*per_paper/3600:.0f} h at this rate")

    split = progress[["include", "review", "exclude"]].sum()
    print("\ndecisions so far:")
    for name, count in split.items():
        print(f"  {name:<8} {count:>8,}  ({count/max(split.sum(),1):5.1%})")

    domains = {}
    for value in manifest["batches"].values():
        for name, count in value.get("domains", {}).items():
            domains[name] = domains.get(name, 0) + count
    if domains:
        print("\ndomains so far:")
        for name, count in sorted(domains.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<18} {count:>8,}")


# ## 8. Retry the failures
#
# Reads `failures.jsonl`, re-classifies everything still unresolved, and patches
# the batch files in place. Safe to run repeatedly - each pass shrinks the list
# and skips anything an earlier pass already recovered.


if not FAILURES.exists():
    print("no failures file - nothing has failed yet")
else:
    records = [json.loads(line) for line in
               FAILURES.read_text(encoding="utf-8").splitlines() if line.strip()]
    logged = {r["openalex_id"]: r["batch"] for r in records}

    outstanding = {}
    for paper_id, batch_no in logged.items():
        path = BATCHES / f"batch_{batch_no:05d}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        row = frame[frame["openalex_id"] == paper_id]
        if len(row) and not bool(row.iloc[0]["model_valid_json"]):
            outstanding.setdefault(batch_no, []).append(paper_id)

    total = sum(len(v) for v in outstanding.values())
    print(f"{len(records)} failures logged; {total} still unresolved across "
          f"{len(outstanding)} batches")

    recovered = 0
    for batch_no, ids in sorted(outstanding.items()):
        path = BATCHES / f"batch_{batch_no:05d}.parquet"
        frame = pd.read_parquet(path)
        subset = frame[frame["openalex_id"].isin(ids)]
        items = list(enumerate(subset.to_dict("records")))
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as workers:
            results = list(tqdm(workers.map(classify, items), total=len(items),
                                desc=f"retry {batch_no:05d}", leave=False))

        model_columns = [c for c in frame.columns if c.startswith("model_")]
        fixed = 0
        for result in results:
            if not result["model_valid_json"]:
                continue
            mask = frame["openalex_id"] == result["openalex_id"]
            for column in model_columns:
                if column in result:
                    frame.loc[mask, column] = result[column]
            fixed += 1

        if fixed:
            frame.to_parquet(path, index=False)
            entry = manifest["batches"][str(batch_no)]
            entry["ok"] = int(frame["model_valid_json"].sum())
            entry["failed"] = int((~frame["model_valid_json"].astype(bool)).sum())
            entry["decisions"] = frame["model_decision"].value_counts().to_dict()
            entry["retried_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            save_manifest()
        recovered += fixed
        print(f"  batch {batch_no:05d}: recovered {fixed} of {len(ids)}")

    print(f"\nrecovered {recovered} of {total}")


# ## 9. Combine
#
# Run any time you want a single file. Reads every completed batch and writes
# the whole classified set plus the selected papers.


files = sorted(BATCHES.glob("batch_*.parquet"))
if not files:
    print("no batches yet")
else:
    everything = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    everything.to_parquet(OUTDIR / "all_classified.parquet", index=False)

    selected = everything[everything["model_decision"] == "include"]
    selected.to_parquet(OUTDIR / "selected_papers.parquet", index=False)
    selected.to_csv(OUTDIR / "selected_papers.csv", index=False, encoding="utf-8-sig")

    print(f"{len(files)} batches -> {len(everything):,} papers")
    print("  all_classified.parquet")
    print(f"  selected_papers.parquet / .csv   ({len(selected):,} includes)")
    print(f"\ndecisions: {everything['model_decision'].value_counts().to_dict()}")
    print(f"failed   : {int((~everything['model_valid_json'].astype(bool)).sum()):,}")

