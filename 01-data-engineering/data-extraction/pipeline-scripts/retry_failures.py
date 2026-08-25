"""Converted from retry-failures.ipynb.

Generated for readability in review; the notebook remains the
authority for anything that was actually run.
"""


# # MUFASA - retry the failed papers
#
# A **separate** notebook for re-classifying the papers that failed during
# production. It exists so failures can be recovered while the main run keeps
# going.
#
# ## Safety contract
#
# This notebook is built to run **alongside** `production-classification.ipynb`
# without corrupting it.
#
# | File | Production | This notebook |
# |---|---|---|
# | `manifest.json` | read + write | **read only, never written** |
# | `failures.jsonl` | appends | **never touched** |
# | `batches/batch_*.parquet` | writes new batches | patches finished ones, atomically |
# | `retry_log.jsonl` | - | appends (its own file) |
# | `retry_manifest.json` | - | writes (its own file) |
#
# **Why `manifest.json` is never written.** The production notebook holds the
# manifest in memory and rewrites the whole file after every batch. Anything a
# second process wrote would be silently overwritten at the next batch boundary.
# So this notebook writes its own record instead and leaves that file alone.
#
# **Consequence to know about.** After a patch, the `ok`, `failed` and
# `decisions` fields in `manifest.json` are stale for that batch - they still
# describe the original run. The parquet files are correct. Section 7 prints the
# true numbers, and `retry_manifest.json` records exactly what changed.
#
# **A batch is only touched when it is finished.** Eligible batches must be
# marked `complete` in the manifest, exist on disk, and have been untouched for
# `SETTLE_SECONDS`. The batch currently being classified is never eligible.
#
# > Do not run section 8 of the production notebook while using this one - both
# > would retry the same rows.


# ## 1. Configuration


from pathlib import Path
from datetime import datetime, timezone

OUTDIR   = Path("production")
BATCHES  = OUTDIR / "batches"
MANIFEST = OUTDIR / "manifest.json"          # read only, never written here

RETRY_LOG      = OUTDIR / "retry_log.jsonl"      # this notebook's own files
RETRY_MANIFEST = OUTDIR / "retry_manifest.json"

# --- sharing the rate limit with the running production notebook ----------
# Same name and meaning as production, because section 2 is copied verbatim and
# computes WORKERS = WORKERS_PER_KEY x keys. Limits are per account, so this is
# the number that matters.
#
# Production uses 2. Whatever this notebook adds runs on top of that against the
# same six accounts, so it starts at 1. Raise it once production has stopped.
WORKERS_PER_KEY = 1

# Failed rows are worth waiting for - throughput does not matter here, so the
# timeout is generous. Production uses 30s; observed latency has reached 168s.
TIMEOUT    = 120
RETRIES    = 5
MAX_OUTPUT = 1024

# A batch must have been untouched this long to count as settled.
SETTLE_SECONDS = 300

# Bound a single sitting. None retries everything outstanding.
MAX_ROWS_THIS_RUN = None

# Must match production, or decisions would not be comparable.
REVIEW_THRESHOLD = 8
STRICT_REASON    = True

BASE_URL = "https://api.tokenrouter.com/v1"
MODEL    = "moonshotai/kimi-k3-free"

print(f"batches   : {BATCHES}")
print(f"per key   : {WORKERS_PER_KEY}  (on top of whatever production is using)")
print(f"timeout   : {TIMEOUT}s, {RETRIES} attempts")
print(f"manifest  : read only - this notebook never writes it")


# ## 2. API keys
#
# Verbatim from the production notebook - the same `.env`, the same discovery. Only the worker count differs, and that is set in section 1.


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
# **Verbatim from the production notebook.** A retried paper must be judged by exactly the same rules as one that succeeded first time, or the corpus would carry two standards.


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
# **Verbatim from the production notebook.**


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


# ## 5. Find the failed rows
#
# Scans the batch files themselves rather than `failures.jsonl`, so the list is always accurate and the log the production run is appending to is left alone. Only settled batches are considered.


import json, os, time
import pandas as pd

def settled_batches():
    """Batches safe to touch: complete in the manifest, present, and quiet.

    Production writes the parquet first and records the manifest entry second,
    so 'complete' already implies the file is whole. The mtime check is a
    second belt: it keeps us away from anything written moments ago.
    """
    if not MANIFEST.exists():
        return []
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    now = time.time()
    out = []
    for key, entry in manifest.get("batches", {}).items():
        if not entry.get("complete"):
            continue
        path = BATCHES / f"batch_{int(key):05d}.parquet"
        if not path.exists():
            continue
        age = now - path.stat().st_mtime
        if age < SETTLE_SECONDS:
            print(f"  skipping batch {int(key):05d} - written {age:.0f}s ago, still settling")
            continue
        out.append(int(key))
    return sorted(out)


eligible = settled_batches()

outstanding = {}
for batch_no in eligible:
    path = BATCHES / f"batch_{batch_no:05d}.parquet"
    frame = pd.read_parquet(path, columns=["openalex_id", "model_valid_json"])
    bad = frame[~frame["model_valid_json"].astype(bool)]
    if len(bad):
        outstanding[batch_no] = list(bad["openalex_id"])

total = sum(len(v) for v in outstanding.values())
print(f"\n{len(eligible)} settled batches, {len(outstanding)} of them hold failures")
print(f"{total:,} papers to retry\n")
for batch_no, ids in sorted(outstanding.items()):
    print(f"  batch {batch_no:05d}  {len(ids):>4}")
if not outstanding:
    print("  nothing outstanding")


# ## 6. Retry and patch
#
# Each batch is re-classified and written back through a temp file. Safe to run repeatedly - every pass shrinks the list, and anything already recovered is skipped by the scan above.
#
# Re-run section 5 before section 6 if the production run has finished more batches since.


import concurrent.futures, random, time
from openai import OpenAI
from tqdm.auto import tqdm

CLIENTS = [OpenAI(base_url=BASE_URL, api_key=value) for _, value in API_KEYS]
KEY_NAMES = [name for name, _ in API_KEYS]


def classify(item):
    """Production's classify, with a longer timeout and no shared throttle.

    Backoff is per request and triples on a 429, so a busy pool slows this
    notebook down rather than failing the paper again.
    """
    index, row = item
    started = time.perf_counter()
    last = ""
    for attempt in range(RETRIES):
        slot = (index + attempt) % len(CLIENTS)
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
            if attempt < RETRIES - 1:
                wait = min(2 ** attempt + random.random(), 30)
                if "429" in last or "RateLimit" in last:
                    wait = min(wait * 3, 60)
                time.sleep(wait)
    return {"openalex_id": row["openalex_id"], "model_valid_json": False,
            "model_error": last}


def patch(batch_no, ids):
    """Re-classify these papers and write them back into their batch file.

    The parquet is replaced through a temp file, so an interrupt leaves the
    original intact rather than a half-written one.
    """
    path = BATCHES / f"batch_{batch_no:05d}.parquet"
    frame = pd.read_parquet(path)
    subset = frame[frame["openalex_id"].isin(ids)]
    items = list(enumerate(subset.to_dict("records")))

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(tqdm(pool.map(classify, items), total=len(items),
                            desc=f"batch {batch_no:05d}", leave=False))

    # Only columns the batch already has, and only ones the result supplies.
    # This is what protects model_decision_before_override and
    # manual_override_reason in batches 0 and 1 - they are never in a result.
    columns = [c for c in frame.columns if c.startswith("model_")]
    fixed = 0
    for result in results:
        if not result.get("model_valid_json"):
            continue
        mask = frame["openalex_id"] == result["openalex_id"]
        for column in columns:
            if column in result:
                frame.loc[mask, column] = result[column]
        fixed += 1

    if fixed:
        tmp = path.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)

    with RETRY_LOG.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps({
                "batch": batch_no,
                "openalex_id": result["openalex_id"],
                "recovered": bool(result.get("model_valid_json")),
                "decision": result.get("model_decision", ""),
                "error": result.get("model_error", ""),
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, ensure_ascii=False) + "\n")

    return fixed, len(items)


budget = MAX_ROWS_THIS_RUN
session = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "workers": WORKERS, "batches": {}}
recovered = attempted = 0

for batch_no, ids in sorted(outstanding.items()):
    if budget is not None:
        if budget <= 0:
            break
        ids = ids[:budget]
    fixed, tried = patch(batch_no, ids)
    recovered += fixed
    attempted += tried
    if budget is not None:
        budget -= tried
    session["batches"][str(batch_no)] = {"attempted": tried, "recovered": fixed}
    print(f"  batch {batch_no:05d}: recovered {fixed} of {tried}")

session["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
session["attempted"] = attempted
session["recovered"] = recovered

history = []
if RETRY_MANIFEST.exists():
    history = json.loads(RETRY_MANIFEST.read_text(encoding="utf-8")).get("sessions", [])
history.append(session)
tmp = RETRY_MANIFEST.with_suffix(".json.tmp")
tmp.write_text(json.dumps({"sessions": history}, indent=1), encoding="utf-8")
tmp.replace(RETRY_MANIFEST) 

print(f"\nrecovered {recovered:,} of {attempted:,} attempted")


# ## 7. What changed
#
# Counts read straight from the parquet files, which are the source of truth once a patch has run.


files = sorted(BATCHES.glob("batch_*.parquet"))
if not files:
    print("no batches yet")
else:
    everything = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ok = everything["model_valid_json"].astype(bool)
    valid = everything[ok]

    print(f"batches on disk : {len(files)}")
    print(f"papers          : {len(everything):,}")
    print(f"classified      : {len(valid):,}")
    print(f"still failed    : {int((~ok).sum()):,}  "
          f"({(~ok).sum()/max(len(everything),1):.2%})\n")

    counts = valid["model_decision"].value_counts()
    for name in ("include", "review", "exclude"):
        n = int(counts.get(name, 0))
        print(f"  {name:<8} {n:>7,}   {n/max(len(valid),1):5.1%}")

    print("\nThese are the true numbers, read from the parquet files.")
    print("manifest.json still shows the original per-batch counts for any")
    print("batch patched here - that is expected and harmless.")

