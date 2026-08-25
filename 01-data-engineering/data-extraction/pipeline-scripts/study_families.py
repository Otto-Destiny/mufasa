"""Converted from study-families.ipynb.

Generated for readability in review; the notebook remains the
authority for anything that was actually run.
"""


# # Study families
#
# One experiment can be published four times - a thesis, a preprint, a conference
# paper and a journal article. Left unlinked they look like four independent
# confirmations, and worse, one version can sit in the training split while
# another sits in the test split, handing the model the answer.
#
# This notebook groups them. Every paper gets a `family_id`; papers that describe
# the same study share one.
#
# ## Accuracy comes first
#
# Merging two different studies makes MUFASA claim a finding was replicated when
# it was measured once. Failing to merge only costs one corroboration. The two
# errors are not equally bad, so this notebook is deliberately biased toward
# **not** merging:
#
# - a pair is merged only when **two independent signals** agree,
# - the abstract is the confirmer - titles alone never decide a merge,
# - anything close but unproven goes to a **review queue** rather than being
#   merged quietly,
# - a numeric difference between two otherwise-identical titles (Part 1 / Part 2,
#   2019 / 2020, 5% / 10%) blocks an automatic merge.
#
# That last rule matters here more than in most corpora. Nigerian papers are
# often published as near-identical series that differ only by site or year -
# "...in Lagos" and "...in Kano" score almost identically on title similarity and
# are genuinely different studies.
#
# ## Speed
#
# All-vs-all on ~23,000 papers is 263 million comparisons. Blocking on rare title
# words cuts that to roughly half a million candidates without being able to miss
# a true duplicate: two versions of one paper always share at least one uncommon
# word. Abstracts are only compared for the few thousand pairs that survive the
# title filter, so the whole run is well under a minute.
#
# ## Output
#
# ```
# production/
#   authors_cache.parquet     the authors column, parsed once
#   study_families.parquet    openalex_id -> family_id, plus family size
#   family_review.csv         pairs the notebook refused to decide
#   family_merges.csv         every merge it did make, with the evidence
# ```


# ## 1. Configuration


from pathlib import Path
import json, re, unicodedata, itertools, collections, time

SOURCE   = Path("openalex_ng_science_2000_2026/openalex_all_fields.csv")
OUTDIR   = Path("production")
CACHE    = OUTDIR / "source_cache.parquet"
AUTHORS  = OUTDIR / "authors_cache.parquet"
BATCHES  = OUTDIR / "batches"

FAMILIES = OUTDIR / "study_families.parquet"
REVIEW   = OUTDIR / "family_review.csv"
MERGES   = OUTDIR / "family_merges.csv"

# --- which papers to group -------------------------------------------------
# "include" groups only the selected corpus, which is what the splits need.
# "all" groups everything classified, which is slower and rarely useful.
SCOPE = "include"

# --- blocking --------------------------------------------------------------
# A title word appearing in more than this many papers is too common to block
# on. Lower is faster and slightly riskier; 40 leaves a wide margin.
RARE_MAX = 40

# A word shared by a huge number of papers would generate a quadratic blow-up
# in one bucket, so buckets larger than this are skipped - the pair will still
# be found through any other rare word the two titles share.
BUCKET_MAX = 60

# --- decision thresholds ---------------------------------------------------
# Cheap pre-filter. Pairs below this on title never have their abstracts read.
TITLE_FLOOR = 0.55

# A pair is merged when any of these hold. Each needs two signals to agree.
ABSTRACT_ALONE   = 0.85   # near-identical abstract is decisive on its own
TITLE_STRONG     = 0.90   # ...with a moderately similar abstract
ABSTRACT_WITH_T  = 0.60
TITLE_GOOD       = 0.80   # ...with authors and a weaker abstract
AUTHOR_WITH_T    = 0.60
ABSTRACT_WITH_TA = 0.50

# Close but unproven - goes to the review queue instead of being merged.
REVIEW_FLOOR = 0.72

# Two papers more than this many years apart are never merged automatically.
MAX_YEAR_GAP = 6

print(f"scope      : {SCOPE}")
print(f"blocking   : words in <= {RARE_MAX} papers, buckets <= {BUCKET_MAX}")
print(f"merge when : abstract >= {ABSTRACT_ALONE}")
print(f"             or title >= {TITLE_STRONG} and abstract >= {ABSTRACT_WITH_T}")
print(f"             or title >= {TITLE_GOOD} and authors >= {AUTHOR_WITH_T} "
      f"and abstract >= {ABSTRACT_WITH_TA}")
print(f"review     : title >= {REVIEW_FLOOR} but not merged")


# ## 2. Load the corpus
#
# The author column is pulled from the source CSV on first run and cached, the same way the licence and download columns were.


import pandas as pd

pool = pd.read_parquet(CACHE)

# The authors column is in the source CSV but not in the 18 cached ones, and
# author overlap is the signal that catches a preprint retitled before
# publication. Parsed once, then cached like the others.
if AUTHORS.exists():
    authors = pd.read_parquet(AUTHORS)
    print(f"loaded {AUTHORS.name}")
else:
    print(f"first run - reading authors from {SOURCE.name}...")
    authors = pd.read_csv(SOURCE, usecols=["openalex_id", "authors_json"],
                          encoding="cp1252", encoding_errors="replace",
                          low_memory=False, on_bad_lines="skip")
    authors.to_parquet(AUTHORS, index=False)
    print(f"cached {AUTHORS.name} - {len(authors):,} rows")

if SCOPE == "include":
    files = sorted(BATCHES.glob("batch_*.parquet"))
    verdicts = pd.concat(
        [pd.read_parquet(f, columns=["openalex_id", "model_decision",
                                     "model_valid_json"]) for f in files],
        ignore_index=True)
    keep = verdicts[verdicts["model_valid_json"].astype(bool)
                    & (verdicts["model_decision"] == "include")]["openalex_id"]
    papers = pool[pool["openalex_id"].isin(set(keep))].copy()
else:
    papers = pool.copy()

papers = papers.merge(authors, on="openalex_id", how="left")
papers = papers.reset_index(drop=True)
print(f"\ngrouping {len(papers):,} papers")


# ## 3. Normalise
#
# Titles and abstracts are stripped of markup, HTML entities and accents before comparison, because the same paper is often indexed once in title case and once in capitals, or with `&lt;i&gt;` around a species name. Authors reduce to surnames, which survive initials being dropped or reordered.


STOPWORDS = set("""a an the of for in on and or to with using from by at as its
their this that study analysis assessment evaluation effect effects
investigation determination influence impact case based towards toward some
selected different various among between during within into over under""".split())


def strip_accents(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def title_tokens(title):
    """Words that carry meaning, with markup and punctuation removed."""
    t = re.sub(r"<[^>]+>", " ", str(title))          # <i>, <sub> and friends
    t = re.sub(r"&[a-z]+;|&#\d+;", " ", t)           # &amp; &lt; &#39;
    t = strip_accents(t).lower()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return {w for w in t.split() if w not in STOPWORDS and len(w) > 2}


def abstract_tokens(abstract):
    """Same treatment, but keep everything - abstracts need the detail."""
    t = re.sub(r"<[^>]+>", " ", str(abstract))
    t = re.sub(r"&[a-z]+;|&#\d+;", " ", t)
    t = strip_accents(t).lower()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return {w for w in t.split() if len(w) > 2}


def surnames(authors_json):
    """Last name of each author, lowercased. Robust to reordering.

    The source CSV has column-bleed damage, so a value here can be a number or
    a fragment of someone else's abstract rather than a JSON list. Anything
    that is not a list of names yields no surnames instead of raising.
    """
    try:
        names = json.loads(authors_json) if isinstance(authors_json, str) else []
    except Exception:
        return set()
    if not isinstance(names, list):
        return set()
    out = set()
    for name in names:
        if not isinstance(name, str):
            continue
        parts = strip_accents(str(name)).lower().replace(",", " ").split()
        if parts:
            out.add(parts[-1])
    return out


def numbers_in(title):
    """Digits in a title usually separate a series: Part 2, 2019, 10%."""
    return set(re.findall(r"\d+", re.sub(r"<[^>]+>", " ", str(title))))


clock = time.perf_counter()
papers["t_tok"] = papers["title"].map(title_tokens)
papers["a_tok"] = papers["abstract"].map(abstract_tokens)
papers["surnames"] = papers["authors_json"].map(surnames)
papers["nums"] = papers["title"].map(numbers_in)
papers["year"] = pd.to_numeric(papers["publication_year"], errors="coerce")
papers["doi_n"] = (papers["doi"].fillna("").astype(str).str.lower()
                   .str.replace("https://doi.org/", "", regex=False).str.strip())

print(f"normalised in {time.perf_counter() - clock:.1f}s")
print(f"  median title tokens    : {papers['t_tok'].map(len).median():.0f}")
print(f"  median abstract tokens : {papers['a_tok'].map(len).median():.0f}")
print(f"  papers with authors    : {(papers['surnames'].map(len) > 0).sum():,}")
print(f"  papers with abstracts  : {(papers['a_tok'].map(len) > 20).sum():,}")


# ## 4. Block
#
# Only papers sharing an uncommon title word are compared. This cannot miss a true duplicate - two versions of one paper always share at least one uncommon word - and it removes almost all of the work.


clock = time.perf_counter()

# How many papers each title word appears in.
frequency = collections.Counter()
for tokens in papers["t_tok"]:
    frequency.update(tokens)

rare = {word for word, n in frequency.items() if n <= RARE_MAX}

index = collections.defaultdict(list)
for position, tokens in enumerate(papers["t_tok"]):
    for word in tokens & rare:
        index[word].append(position)

candidates = set()
skipped = 0
for word, positions in index.items():
    if len(positions) < 2:
        continue
    if len(positions) > BUCKET_MAX:
        skipped += 1                      # still reachable via another word
        continue
    candidates.update(itertools.combinations(sorted(positions), 2))

rare_only = len(candidates)

# A title built entirely from common words - "Seasonal Metal Distribution in
# Ondo Coastal Sediment, Nigeria" - has no rare word to block on, so the loop
# above can never propose it even against an identical twin. Matching on the
# whole normalised title costs nothing and closes that hole.
exact = collections.defaultdict(list)
for position, tokens in enumerate(papers["t_tok"]):
    if tokens:
        exact[" ".join(sorted(tokens))].append(position)
for positions in exact.values():
    if 2 <= len(positions) <= BUCKET_MAX:
        candidates.update(itertools.combinations(sorted(positions), 2))

total_possible = len(papers) * (len(papers) - 1) // 2
print(f"blocked in {time.perf_counter() - clock:.1f}s")
print(f"  rare words        : {len(rare):,} of {len(frequency):,}")
print(f"  buckets skipped   : {skipped:,} (over {BUCKET_MAX} papers)")
print(f"  from rare words   : {rare_only:,} pairs")
print(f"  from exact titles : {len(candidates) - rare_only:,} pairs the rare-word "
      f"index could not reach")
print(f"  candidate pairs   : {len(candidates):,}")
print(f"  all-vs-all would be {total_possible:,}  "
      f"({total_possible / max(len(candidates), 1):.0f}x more)")


# ## 5. Score
#
# Four signals per pair. Abstracts are only compared for pairs that already look alike by title, which keeps the expensive comparison off the vast majority of candidates.


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


clock = time.perf_counter()

T = papers["t_tok"].tolist()
A = papers["a_tok"].tolist()
S = papers["surnames"].tolist()
N = papers["nums"].tolist()
Y = papers["year"].tolist()
D = papers["doi_n"].tolist()

scored = []
for i, j in candidates:
    title = jaccard(T[i], T[j])
    if title < TITLE_FLOOR:                 # cheap filter before touching abstracts
        continue
    abstract = jaccard(A[i], A[j])
    author = jaccard(S[i], S[j])
    gap = abs(Y[i] - Y[j]) if pd.notna(Y[i]) and pd.notna(Y[j]) else 99
    scored.append((i, j, title, abstract, author, gap, N[i] != N[j]))

pairs = pd.DataFrame(scored, columns=["i", "j", "title", "abstract",
                                      "author", "year_gap", "nums_differ"])
print(f"scored in {time.perf_counter() - clock:.1f}s")
print(f"  pairs above the title floor : {len(pairs):,}")
if len(pairs):
    print(f"  title    median {pairs.title.median():.2f}  max {pairs.title.max():.2f}")
    print(f"  abstract median {pairs.abstract.median():.2f}  max {pairs.abstract.max():.2f}")
    print(f"  same DOI on {sum(D[r.i] == D[r.j] != '' for r in pairs.itertuples()):,} pairs")


# ## 6. Decide and group
#
# Every merge needs two signals to agree. Anything close but unconfirmed goes to review rather than being merged, because a wrong merge invents a replication and a missed merge only costs a corroboration.


def verdict(row):
    """merge, review or separate - and the reason, so a merge can be audited."""
    same_doi = D[row.i] == D[row.j] != ""
    if same_doi:
        return "merge", "identical DOI"

    if row.year_gap > MAX_YEAR_GAP:
        return ("review", "years too far apart") if row.title >= REVIEW_FLOOR else ("separate", "")

    # A digit that differs between two otherwise-alike titles usually marks a
    # series - Part 1 vs Part 2, one year vs the next. Only an almost identical
    # abstract overrides it.
    if row.nums_differ and row.abstract < ABSTRACT_ALONE:
        return ("review", "titles differ by a number") if row.title >= REVIEW_FLOOR else ("separate", "")

    if row.abstract >= ABSTRACT_ALONE:
        return "merge", f"abstract {row.abstract:.2f}"
    if row.title >= TITLE_STRONG and row.abstract >= ABSTRACT_WITH_T:
        return "merge", f"title {row.title:.2f} + abstract {row.abstract:.2f}"
    if (row.title >= TITLE_GOOD and row.author >= AUTHOR_WITH_T
            and row.abstract >= ABSTRACT_WITH_TA):
        return "merge", (f"title {row.title:.2f} + authors {row.author:.2f} "
                         f"+ abstract {row.abstract:.2f}")
    if row.title >= REVIEW_FLOOR:
        return "review", "close, not confirmed"
    return "separate", ""


decisions = [verdict(r) for r in pairs.itertuples()]
pairs["decision"] = [d for d, _ in decisions]
pairs["why"] = [w for _, w in decisions]

print(pairs["decision"].value_counts().to_string())


class Union:
    """Union-find. Merging is transitive: if A=B and B=C then A, B and C are
    one family, which is what turns pairs into groups."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


union = Union(len(papers))
for r in pairs[pairs.decision == "merge"].itertuples():
    union.join(r.i, r.j)

roots = [union.find(k) for k in range(len(papers))]
papers["family_id"] = ["SF-%06d" % r for r in roots]
sizes = papers["family_id"].value_counts()
papers["family_size"] = papers["family_id"].map(sizes)

grouped = papers[papers.family_size > 1]
print(f"\nfamilies      : {papers.family_id.nunique():,} for {len(papers):,} papers")
print(f"multi-paper   : {sizes[sizes > 1].shape[0]:,} families covering {len(grouped):,} papers")
print(f"largest       : {sizes.max()} papers")
print(f"size spread   : {sizes[sizes > 1].value_counts().sort_index().to_dict()}")


# ## 7. Check it
#
# Recall against pairs you already know are duplicates, then a sample of what was merged and what was refused. Read the refusals - they are where the thresholds are visible.


# Recall check. Put pairs you already know are the same study here; the
# notebook reports whether it caught each one. Seeded with duplicates found by
# hand in the corpus - add to it whenever you spot another.
KNOWN_SAME = [
    ("10.5281/zenodo.7714534", "10.5121/ijnlc.2023.12101"),   # preprint + journal
    ("10.54058/1ns94w23", "10.54058/w2taxw56"),               # two DOIs, one publisher
]

position = {d: k for k, d in enumerate(D) if d}
print("=== recall on known duplicates ===")
for left, right in KNOWN_SAME:
    if left not in position or right not in position:
        print(f"  ? {left} / {right}  - not both in scope")
        continue
    a, b = position[left], position[right]
    caught = papers.family_id.iloc[a] == papers.family_id.iloc[b]
    print(f"  {'PASS' if caught else 'MISS'}  {left}  <->  {right}")

print("\n=== 12 merges, with the evidence ===")
merged = pairs[pairs.decision == "merge"].sort_values("title", ascending=False)
for r in merged.head(12).itertuples():
    print(f"\n  [{r.why}]")
    print(f"    {str(papers.title.iloc[r.i])[:88]}")
    print(f"    {str(papers.title.iloc[r.j])[:88]}")

print("\n\n=== 8 pairs sent to review - these were NOT merged ===")
for r in pairs[pairs.decision == "review"].sort_values("title", ascending=False).head(8).itertuples():
    print(f"\n  [{r.why}] title {r.title:.2f} abstract {r.abstract:.2f} authors {r.author:.2f}")
    print(f"    {str(papers.title.iloc[r.i])[:88]}")
    print(f"    {str(papers.title.iloc[r.j])[:88]}")


# ## 8. Write
#
# `study_families.parquet` is the input to freezing splits. `family_review.csv` is the queue for the pairs this notebook would not decide on its own.


out = papers[["openalex_id", "doi", "title", "publication_year",
              "family_id", "family_size"]].copy()
out.to_parquet(FAMILIES, index=False)

def readable(text):
    """Abstract with markup and entities removed, case preserved for reading."""
    t = re.sub(r"<[^>]+>", " ", str(text))
    t = re.sub(r"&[a-z]+;|&#\d+;", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def side(frame, which, field, fn=None):
    return [fn(papers[field].iloc[k]) if fn else papers[field].iloc[k]
            for k in frame[which]]


for name, path in (("review", REVIEW), ("merge", MERGES)):
    subset = pairs[pairs.decision == name].reset_index(drop=True)
    if not len(subset):
        print(f"no {name} pairs")
        continue

    # Scores first so the decision is visible, then the two titles next to each
    # other, then the abstracts last - they are long, and putting them earlier
    # pushes everything else off the screen in a spreadsheet.
    report = pd.DataFrame({
        "why": subset["why"],
        "title_sim": subset["title"].round(3),
        "abstract_sim": subset["abstract"].round(3),
        "author_sim": subset["author"].round(3),
        "year_gap": subset["year_gap"],
        "i_title": side(subset, "i", "title"),
        "j_title": side(subset, "j", "title"),
        "i_year": side(subset, "i", "publication_year"),
        "j_year": side(subset, "j", "publication_year"),
        "i_journal": side(subset, "i", "journal"),
        "j_journal": side(subset, "j", "journal"),
        "i_doi": [D[k] for k in subset["i"]],
        "j_doi": [D[k] for k in subset["j"]],
        "i_abstract": side(subset, "i", "abstract", readable),
        "j_abstract": side(subset, "j", "abstract", readable),
    })
    report.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{path.name:<24} {len(report):,} pairs, {len(report.columns)} columns")

print(f"{FAMILIES.name:<24} {len(out):,} papers in {out.family_id.nunique():,} families")
print("\nUse family_id as the unit when freezing train/dev/test splits - assign")
print("whole families, never individual papers, or one version of a study can")
print("sit in train while another sits in test.")

