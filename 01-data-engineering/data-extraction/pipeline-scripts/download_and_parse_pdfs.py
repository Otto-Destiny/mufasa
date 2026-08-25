"""Converted from download-and-parse-pdfs.ipynb.

Generated for readability in review; the notebook remains the
authority for anything that was actually run.
"""


# # MUFASA corpus download and PDF-to-Markdown pipeline
#
# This notebook takes the selected Tier-1 training candidates, downloads the
# permitted PDF copy, validates it, and parses it into **one Markdown file per
# paper**. It is designed to run unattended in resumable batches.
#
# The normal workflow is simply **Run all**. Adjust only the controls in the next
# cell. A batch is recorded only after its output Parquet is safely written.
# Technical failures remain technical failures; they are never turned into valid
# documents.


# ===================== ENVIRONMENT =====================
# One switch decides paths, worker counts and whether packages are installed.
# "auto" detects Kaggle and Colab and otherwise assumes a local machine.
ENVIRONMENT = "local"          # "local" | "kaggle" | "colab" | "auto"

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def detect_environment():
    if Path("/kaggle/input").exists() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if "google.colab" in sys.modules or Path("/content").is_dir():
        return "colab"
    return "local"


ENV = detect_environment() if ENVIRONMENT == "auto" else ENVIRONMENT
if ENV not in {"local", "kaggle", "colab"}:
    raise ValueError(f"ENVIRONMENT must be local, kaggle, colab or auto; got {ENVIRONMENT!r}")
IN_KAGGLE, IN_COLAB, IS_LOCAL = ENV == "kaggle", ENV == "colab", ENV == "local"

# Hosted runtimes start empty, so they install. A local machine is assumed to be
# a deliberately prepared environment and is never modified without asking.
INSTALL_PACKAGES = not IS_LOCAL

REQUIRED_PACKAGES = {
    "fitz": "pymupdf>=1.24,<2",
    "pdf_inspector": "pdf-inspector>=0.2,<1",
    "pymupdf4llm": "pymupdf4llm>=0.0.19,<1",
    "pyarrow": "pyarrow>=15,<22",
    "requests": "requests>=2.31,<3",
    "tqdm": "tqdm>=4.66,<5",
}
missing = [package for module, package in REQUIRED_PACKAGES.items()
           if importlib.util.find_spec(module) is None]
if missing and INSTALL_PACKAGES:
    # Install only what is absent. Never broadly upgrade a hosted runtime's
    # pandas/matplotlib stack: that has previously broken Kaggle sessions.
    print("Installing:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
elif missing:
    raise RuntimeError(
        "Missing packages in the local environment:\n  " + "\n  ".join(missing) +
        "\nInstall them yourself, or set INSTALL_PACKAGES=True to let this notebook do it."
    )
else:
    print("All required packages are available.")

# Tesseract is a system binary, not a Python package, so pip cannot supply it.
# Without it OCR repair is skipped and image-only pages stay empty — which is a
# silent loss, so this block installs it, proves it actually runs, and stops the
# notebook rather than letting a long run produce quietly degraded output.
REQUIRE_OCR = True                # False only if you accept empty image pages

import shutil


def tesseract_works():
    """A path is not proof. Run it, and confirm it has English language data."""
    binary = shutil.which("tesseract")
    if not binary:
        return None, ""
    try:
        version = subprocess.run([binary, "--version"], capture_output=True,
                                 text=True, timeout=60)
        langs = subprocess.run([binary, "--list-langs"], capture_output=True,
                               text=True, timeout=60)
    except Exception as exc:
        return None, f"tesseract found at {binary} but failed to run: {exc}"
    if version.returncode != 0:
        return None, f"tesseract --version exited {version.returncode}"
    available = (langs.stdout or "").split()
    if "eng" not in available:
        return None, (f"tesseract runs but has no English data. Installed: "
                      f"{available or 'none'}. Install tesseract-ocr-eng.")
    return binary, (version.stdout or "").splitlines()[0] if version.stdout else binary


def install_tesseract():
    """Debian images (Kaggle, Colab) can apt-get it. Try with and without sudo."""
    commands = [
        ["apt-get", "-qq", "update"],
        ["apt-get", "-qq", "install", "-y", "tesseract-ocr", "tesseract-ocr-eng"],
    ]
    for command in commands:
        for attempt in (command, ["sudo"] + command):
            try:
                done = subprocess.run(attempt, capture_output=True, text=True, timeout=600)
                if done.returncode == 0:
                    break
            except Exception:
                continue
        else:
            print(f"  '{' '.join(command)}' did not succeed")
    shutil.which.cache_clear() if hasattr(shutil.which, "cache_clear") else None


TESSERACT_PATH, TESSERACT_INFO = tesseract_works()
if TESSERACT_PATH is None and not IS_LOCAL:
    print("tesseract missing - installing tesseract-ocr ...")
    install_tesseract()
    TESSERACT_PATH, TESSERACT_INFO = tesseract_works()

if TESSERACT_PATH:
    print(f"tesseract: OK  {TESSERACT_INFO}")
else:
    hint = {
        "kaggle": "apt-get install -y tesseract-ocr tesseract-ocr-eng  "
                  "(needs Internet enabled in the notebook settings)",
        "colab":  "!apt-get install -y tesseract-ocr tesseract-ocr-eng",
        "local":  "Windows: winget install UB-Mannheim.TesseractOCR   |   "
                  "macOS: brew install tesseract   |   Linux: sudo apt-get install tesseract-ocr",
    }[ENV]
    message = (f"Tesseract is not usable in this environment ({ENV}).\n"
               f"  {TESSERACT_INFO or 'not found on PATH'}\n"
               f"  Install it with: {hint}\n"
               f"  Image-only pages would silently stay empty and their papers "
               f"would be dropped as needs_ocr.\n"
               f"  Set REQUIRE_OCR = False to proceed anyway and accept that loss.")
    if REQUIRE_OCR:
        raise RuntimeError(message)
    print("WARNING: " + message)

print(f"environment: {ENV}")


# ========================= CONTROL PANEL =========================
# Paths. Leave as None to use the profile for the environment selected above.
DATA_ROOT = None                  # folder holding training_candidates.parquet
OUTPUT_ROOT = None                # where PDFs, Markdown and manifests are written
KAGGLE_INPUT_DATASET = None       # mounted folder name under /kaggle/input
COLAB_DATA_DIR = "/content/mufasa_data"

ENV_PROFILES = {
    "local":  {"output_root": None,                                 # beside the data
               "download_workers": min(32, max(8, (os.cpu_count() or 4) * 4)),
               "parse_workers": max(2, (os.cpu_count() or 4) - 1),
               "min_free_gb": 10},
    "kaggle": {"output_root": Path("/kaggle/working/mufasa_ingestion"),
               "download_workers": 48, "parse_workers": 4, "min_free_gb": 6},
    "colab":  {"output_root": Path("/content/mufasa_ingestion"),
               "download_workers": 48, "parse_workers": 2, "min_free_gb": 6},
}
PROFILE = ENV_PROFILES[ENV]

# ---------------------------------------------------------------------------
# RUN MODE.  Leave on "smoke" to prove the pipeline end to end on a handful of
# papers, then set "production" to process the whole corpus. Smoke output goes
# to its own folder, so it can never contaminate production state.
RUN_MODE = "production"                # "smoke" | "production"
SMOKE_PAPERS = 10
# ---------------------------------------------------------------------------

# Work scheduling.
# Batch size decides how often progress becomes durable: the batch Parquet and
# the manifest are written only when a whole batch finishes. At 1,000 papers
# that was ~25 minutes of work with nothing recorded, so any interruption lost
# the entire batch record. 100 lands state every couple of minutes.
BATCH_SIZE = 100
MAX_BATCH_ROUNDS = 3              # stop reworking a batch after this many passes
STALE_PART_MINUTES = 30           # sweep .part files abandoned by a killed run
FRESH_START = False               # reset batch state; valid PDFs are still reused
MAX_BATCHES_THIS_RUN = None       # None means continue until every batch is done
RETRY_FAILURES = True
RETRY_ONLY_TRANSIENT = True       # never re-request a 403/404; only timeouts and 5xx
DOWNLOAD_WORKERS = PROFILE["download_workers"]
PARSE_WORKERS = PROFILE["parse_workers"]
PER_HOST_WORKERS = 4              # be fast without hammering one repository

# Network resilience.
MAX_URL_ATTEMPTS = 3
CONNECT_TIMEOUT_SECONDS = 20
READ_TIMEOUT_SECONDS = 180
MAX_BACKOFF_SECONDS = 60
MAX_PDF_MB = 250
MAX_DOWNLOAD_SECONDS = 600        # hard wall-clock limit for one URL attempt
MIN_FREE_GB = PROFILE["min_free_gb"]
USER_AGENT = "MUFASA-research-corpus/1.0 (contact: set CROSSREF_MAILTO)"

# Parsing, OCR repair and quality gates.
# OCR is applied ONLY to pages that yielded no usable embedded text. Embedded
# text is exact; OCR is a guess from pixels, and re-reading a good page would
# corrupt digits, units and subscripts. Targeted repair beats both extremes.
OCR_REPAIR = True                 # needs the Tesseract binary on PATH
OCR_DPI = 200
MIN_TEXT_CHARS_PER_PAGE = 80      # a page below this is a candidate for OCR

# A paper is usable when its body extracted, not when every page has text.
# Trailing plates, figure pages and scanned appendices are normal here and must
# not disqualify a paper whose text is intact.
MIN_DOCUMENT_CHARS = 3000
MIN_GOOD_PAGES = 3
MIN_TITLE_TOKEN_OVERLAP = 0.70
SMOKE_PAPERS_OVERRIDE = None      # set an int to change the smoke sample size
FORCE_REPARSE = False

# Delete each PDF once it has parsed cleanly. The Markdown, the structured JSON
# and the sidecar (SHA-256 + source URL) all survive, so provenance is intact and
# the file can be re-fetched from its recorded URL if it is ever needed again.
# This turns ~20 GB of PDFs into ~1.8 GB of text, which is what makes the corpus
# fit inside a Kaggle working directory.
DELETE_PDF_AFTER_PARSE = True
USE_OPENALEX_CONTENT_FALLBACK = False  # may consume a paid/limited allowance

# Retraction verification is cached in the batch output. Set False only when
# speed is more important than checking Crossref during this run.
CHECK_CROSSREF_RETRACTIONS = True
CROSSREF_WORKERS = min(16, DOWNLOAD_WORKERS)
RUN_PREFLIGHT = True
PREFLIGHT_CANDIDATES = 5

if RUN_MODE not in {"smoke", "production"}:
    raise ValueError(f'RUN_MODE must be "smoke" or "production"; got {RUN_MODE!r}')
if RUN_MODE == "smoke":
    BATCH_SIZE = SMOKE_PAPERS
    MAX_BATCHES_THIS_RUN = 1

print(f"mode   : {RUN_MODE.upper()}"
      + (f" - {SMOKE_PAPERS} papers only" if RUN_MODE == "smoke" else " - full corpus"))
print(f"workers: {DOWNLOAD_WORKERS} download, {PARSE_WORKERS} parse, "
      f"{MIN_FREE_GB} GB disk floor")


import hashlib
import json
import math
import re
import tempfile
from datetime import datetime, timezone

import pandas as pd

# licence_cache is deliberately not read: training_candidates.parquet is already
# filtered to licence_tier 1, and carries its own `licence` column.
FILENAMES = {
    "training": "training_candidates.parquet",
    "download": "download_cache.parquet",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def search_roots():
    """Where to look for the input Parquet files, in priority order."""
    if DATA_ROOT:
        return [Path(DATA_ROOT).expanduser()]
    if IN_KAGGLE:
        base = Path("/kaggle/input")
        return [base / KAGGLE_INPUT_DATASET] if KAGGLE_INPUT_DATASET else [base]
    if IN_COLAB:
        return [Path(COLAB_DATA_DIR), Path("/content/drive/MyDrive"), Path("/content")]
    # Local: walk up from the working directory so the notebook runs from the
    # repository root, from its own folder, or from anywhere in between.
    roots, here = [], Path.cwd().resolve()
    for folder in (here, *here.parents):
        roots.append(folder / "01-data-engineering" / "data-extraction" / "production")
        roots.append(folder / "production")
        if (folder / ".git").is_dir():
            break
    return roots


def find_exact_file(filename, explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    matches = []
    for root in search_roots():
        if not root.exists():
            continue
        direct = root / filename
        matches.extend([direct.resolve()] if direct.is_file()
                       else [p.resolve() for p in root.rglob(filename)])
        if matches:
            break                      # first root that has it wins

    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        detail = "\n".join(f"  {p}" for p in matches) or "  (none found)"
        raise FileNotFoundError(
            f"Expected exactly one {filename!r}, found {len(matches)}:\n{detail}\n"
            "Set the matching *_PATH control explicitly."
        )
    return matches[0]


TRAINING = find_exact_file(FILENAMES["training"])
DOWNLOADS = find_exact_file(FILENAMES["download"])

if OUTPUT_ROOT is None:
    OUTPUT_ROOT = PROFILE["output_root"] or (TRAINING.parent / "corpus_v1")
if RUN_MODE == "smoke":
    OUTPUT_ROOT = Path(str(OUTPUT_ROOT) + "_smoke")
OUTPUT_ROOT = Path(OUTPUT_ROOT)
PDF_DIR = OUTPUT_ROOT / "raw" / "pdfs"
MARKDOWN_DIR = OUTPUT_ROOT / "parsed" / "markdown"
STRUCTURED_DIR = OUTPUT_ROOT / "parsed" / "structured"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
BATCH_DIR = MANIFEST_DIR / "batches"
MANIFEST_PATH = MANIFEST_DIR / "manifest.json"
FAILURE_DIR = MANIFEST_DIR / "failures"

for folder in (PDF_DIR, MARKDOWN_DIR, STRUCTURED_DIR, BATCH_DIR, FAILURE_DIR):
    folder.mkdir(parents=True, exist_ok=True)


def read_env(path):
    values = {}
    if not path or not Path(path).is_file():
        return values
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def find_dotenv():
    """Nearest .env walking up from the data folder, then from the notebook."""
    for start in (TRAINING.parent, Path.cwd().resolve()):
        for folder in (start, *start.parents):
            candidate = folder / ".env"
            if candidate.is_file():
                return candidate
            if (folder / ".git").is_dir():
                break
    return None


def optional_secret(name):
    if IN_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient
            value = UserSecretsClient().get_secret(name)
            if value:
                return value.strip()
        except Exception:
            pass
    if IN_COLAB:
        try:
            from google.colab import userdata
            value = userdata.get(name)
            if value:
                return value.strip()
        except Exception:
            pass
    if os.environ.get(name):
        return os.environ[name].strip()
    return read_env(find_dotenv()).get(name, "").strip()


OPENALEX_API_KEY = optional_secret("OPENALEX_API_KEY")
CROSSREF_MAILTO = optional_secret("CROSSREF_MAILTO")
if CHECK_CROSSREF_RETRACTIONS and not CROSSREF_MAILTO:
    print("Note: set CROSSREF_MAILTO for Crossref's polite API pool.")

print("inputs:")
print(" ", TRAINING)
print(" ", DOWNLOADS)
print("output:", OUTPUT_ROOT)


# Build the immutable work queue. Citation count is the primary ordering key;
# publication date is the tie-breaker, followed by OpenAlex ID.
PERMISSIVE = {"cc-by", "cc-by-sa", "cc0", "public-domain"}
KNOWN_CORRUPT_IDS = {"https://openalex.org/W4377565026"}


def clean_text(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"nan", "none", "null"} else value


def licence_name(value):
    value = clean_text(value).lower().replace("_", "-")
    cc_url = value.startswith(("https://creativecommons.org/licenses/",
                              "http://creativecommons.org/licenses/"))
    value = value.removeprefix("https://creativecommons.org/licenses/")
    value = value.removeprefix("http://creativecommons.org/licenses/")
    value = value.strip("/")
    value = re.sub(r"/\d+(?:\.\d+)?$", "", value)
    if cc_url and not value.startswith("cc-"):
        value = "cc-" + value.replace("/", "-")
    return value


def short_openalex_id(value):
    match = re.search(r"(W\d+)$", clean_text(value))
    if not match:
        raise ValueError(f"bad OpenAlex work ID: {value!r}")
    return match.group(1)


def doi_key(value):
    value = clean_text(value).lower()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    return value.strip()


selected = pd.read_parquet(TRAINING)
download_cache = pd.read_parquet(DOWNLOADS)

for name, frame in (("training", selected), ("download", download_cache)):
    if "openalex_id" not in frame or not frame["openalex_id"].is_unique:
        raise ValueError(f"{name} must contain unique openalex_id values")

pool = selected.merge(download_cache, on="openalex_id", how="left", validate="one_to_one")

# Every selected paper is already licence_tier 1, so the paper's own licence
# governs all of its copies. PERMISSIVE below stays as a cheap safety net.
if "licence_tier" in pool and not (pool["licence_tier"] == 1).all():
    raise AssertionError("training_candidates contains rows outside licence tier 1")
pool["paper_id"] = pool["openalex_id"].map(short_openalex_id)
pool["doi_key"] = pool["doi"].map(doi_key)
pool["citation_sort"] = pd.to_numeric(pool["cited_by_count"], errors="coerce").fillna(-1)
pool["date_sort"] = pd.to_datetime(pool["publication_date"], errors="coerce")
pool["metadata_corrupt"] = pool["openalex_id"].isin(KNOWN_CORRUPT_IDS)
pool = pool.sort_values(
    ["citation_sort", "date_sort", "openalex_id"],
    ascending=[False, False, True], kind="stable"
).reset_index(drop=True)


def permitted_urls(row):
    licence = licence_name(row.get("licence"))
    sources = ["download_url", "best_oa_pdf_url", "primary_pdf_url",
               "download_landing_page_url"]
    if USE_OPENALEX_CONTENT_FALLBACK:
        sources.append("openalex_content_pdf_endpoint")
    pairs = [(field.replace("_url", "").replace("_pdf", ""),
              clean_text(row.get(field)), licence) for field in sources]
    answer, seen = [], set()
    for source, url, licence in pairs:
        if not url or url in seen or licence not in PERMISSIVE:
            continue
        seen.add(url)
        answer.append({"source": source, "url": url, "licence": licence})
    return answer


pool["permitted_urls"] = [permitted_urls(row) for row in pool.to_dict("records")]
pool["rights_status"] = pool["permitted_urls"].map(
    lambda items: "permitted" if items else "rights_review"
)

if len(pool) != len(selected):
    raise AssertionError("join changed the number of selected papers")
if not pool["paper_id"].is_unique:
    raise AssertionError("paper_id must be unique")

if RUN_MODE == "smoke":
    # Truncate before the fingerprint so smoke state is self-consistent and can
    # never be mistaken for a partially complete production run.
    limit = SMOKE_PAPERS_OVERRIDE or SMOKE_PAPERS
    pool = pool.head(limit).reset_index(drop=True)
    print(f"SMOKE TEST: queue limited to the {len(pool)} most-cited papers")

fingerprint_hasher = hashlib.sha256()
FINGERPRINT_COLUMNS = [
    "openalex_id", "doi_key", "title", "family_id", "field_name",
    "model_mufasa_domain", "publication_date", "cited_by_count", "permitted_urls",
]
for row in pool[FINGERPRINT_COLUMNS].to_dict("records"):
    fingerprint_hasher.update(json.dumps(row, sort_keys=True).encode("utf-8"))
SOURCE_FINGERPRINT = fingerprint_hasher.hexdigest()
TOTAL_BATCHES = (len(pool) + BATCH_SIZE - 1) // BATCH_SIZE

print(f"papers: {len(pool):,} in {TOTAL_BATCHES} batch(es)  [{RUN_MODE}]")
print(pool["rights_status"].value_counts().to_string())


# Concurrent, host-aware downloader with validation, retry and exponential
# backoff. Every worker writes to its own .part file and renames only a valid PDF.
import concurrent.futures
import email.utils
import random
import threading
import time
from collections import defaultdict
from urllib.parse import quote, urlparse

# PyMuPDF is not the parser - pdf_inspector is. It is kept for three jobs it
# does better: validating a downloaded file really is a PDF, counting pages, and
# OCR repair on image-only pages. The module was renamed from fitz to pymupdf;
# importing the old name still works but warns, so import the new one.
try:
    import pymupdf as fitz
except ImportError:
    import fitz
import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm

_thread_local = threading.local()
_host_lock = threading.Lock()
_host_semaphores = {}
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def get_session():
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=DOWNLOAD_WORKERS,
                              pool_maxsize=DOWNLOAD_WORKERS, max_retries=0)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*;q=0.5"})
        _thread_local.session = session
    return _thread_local.session


def host_semaphore(url):
    host = (urlparse(url).hostname or "unknown").lower()
    with _host_lock:
        return _host_semaphores.setdefault(host, threading.BoundedSemaphore(PER_HOST_WORKERS))


def retry_after_seconds(response):
    value = response.headers.get("Retry-After", "").strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
            return max(0.0, when.timestamp() - time.time())
        except Exception:
            return None


def backoff(attempt, response=None):
    server_wait = retry_after_seconds(response) if response is not None else None
    wait = server_wait if server_wait is not None else min(2 ** attempt, MAX_BACKOFF_SECONDS)
    time.sleep(min(wait, MAX_BACKOFF_SECONDS) + random.uniform(0, 0.75))


def sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pdf(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 32:
        raise ValueError("missing or empty PDF")
    with path.open("rb") as handle:
        if b"%PDF-" not in handle.read(1024):
            raise ValueError("file does not contain a PDF header")
    with fitz.open(path) as document:
        if document.page_count < 1:
            raise ValueError("PDF has no pages")
        page_count = document.page_count
    return {"pdf_bytes": path.stat().st_size, "pdf_sha256": sha256_file(path),
            "pdf_pages": page_count}


class PermanentDownloadError(ValueError):
    pass


def pdf_sidecar_path(pdf_path):
    return Path(pdf_path).with_suffix(".pdf.meta.json")


def paper_source_fingerprint(row):
    payload = {"paper_id": row["paper_id"], "permitted_urls": row["permitted_urls"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def write_pdf_sidecar(pdf_path, payload):
    path = pdf_sidecar_path(pdf_path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def request_pdf(candidate, destination, source_fingerprint):
    url = candidate["url"]
    params = None
    if candidate["source"] == "openalex_content" and OPENALEX_API_KEY:
        params = {"api_key": OPENALEX_API_KEY}  # never stored in the manifest
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error = ""
    attempts_used = 0
    for attempt in range(1, MAX_URL_ATTEMPTS + 1):
        attempts_used = attempt
        response = None
        try:
            transfer_started = time.perf_counter()
            with host_semaphore(url), get_session().get(
                    url, params=params, stream=True, allow_redirects=True,
                    timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
                ) as response:
                if response.status_code in RETRYABLE_STATUS:
                    raise requests.HTTPError(f"retryable HTTP {response.status_code}", response=response)
                response.raise_for_status()
                content_length = int(response.headers.get("Content-Length", 0) or 0)
                if content_length > MAX_PDF_MB * 1024 * 1024:
                    raise PermanentDownloadError(f"PDF exceeds {MAX_PDF_MB} MB limit")
                written = 0
                with temporary.open("wb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        if not block:
                            continue
                        written += len(block)
                        if written > MAX_PDF_MB * 1024 * 1024:
                            raise PermanentDownloadError(f"PDF exceeds {MAX_PDF_MB} MB limit")
                        if time.perf_counter() - transfer_started > MAX_DOWNLOAD_SECONDS:
                            raise requests.Timeout(f"download exceeded {MAX_DOWNLOAD_SECONDS}s")
                        handle.write(block)
            try:
                details = validate_pdf(temporary)
            except ValueError as exc:
                raise PermanentDownloadError(str(exc)) from exc
            temporary.replace(destination)
            result = {
                "download_status": "ok", "download_error": "",
                "download_attempts": attempt, "download_source": candidate["source"],
                "download_licence": candidate["licence"], "download_url_used": url,
                "download_final_url": response.url.split("?", 1)[0],
                "http_content_type": response.headers.get("Content-Type", ""),
                **details,
            }
            write_pdf_sidecar(destination, {
                "schema_version": "1.0", "paper_id": destination.stem,
                "pdf_sha256": details["pdf_sha256"], "pdf_bytes": details["pdf_bytes"],
                "source": candidate["source"], "url": url, "licence": candidate["licence"],
                "source_fingerprint": source_fingerprint, "downloaded_at": utc_now(),
            })
            return result
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            last_error = f"{type(exc).__name__}: {exc}"[:500]
            status = getattr(getattr(exc, "response", None), "status_code", None)
            transient = (status in RETRYABLE_STATUS or
                         isinstance(exc, (requests.Timeout, requests.ConnectionError,
                                          requests.exceptions.ChunkedEncodingError)))
            if attempt < MAX_URL_ATTEMPTS and transient and not isinstance(exc, PermanentDownloadError):
                backoff(attempt, getattr(exc, "response", None))
                continue
            break
    return {"download_status": "failed", "download_error": last_error,
            "download_attempts": attempts_used, "download_source": candidate["source"],
            "download_licence": candidate["licence"], "download_url_used": url,
            "download_final_url": "", "http_content_type": "",
            "pdf_bytes": None, "pdf_sha256": "", "pdf_pages": None}


def download_one(row):
    started = time.perf_counter()
    destination = PDF_DIR / f"{row['paper_id']}.pdf"
    source_fingerprint = paper_source_fingerprint(row)
    if destination.exists():
        try:
            details = validate_pdf(destination)
            sidecar = json.loads(pdf_sidecar_path(destination).read_text(encoding="utf-8"))
            current_pairs = {(item["source"], item["url"], item["licence"])
                             for item in row["permitted_urls"]}
            provenance = (sidecar.get("source"), sidecar.get("url"), sidecar.get("licence"))
            if (sidecar.get("pdf_sha256") == details["pdf_sha256"] and
                    sidecar.get("source_fingerprint") == source_fingerprint and
                    provenance in current_pairs):
                return {"paper_id": row["paper_id"], "pdf_path": str(destination),
                        "download_status": "ok", "download_error": "",
                        "download_attempts": 0, "download_source": sidecar["source"],
                        "download_licence": sidecar["licence"],
                        "download_url_used": sidecar["url"], "download_final_url": "",
                        "http_content_type": "", "download_seconds":
                        round(time.perf_counter() - started, 3), **details}
        except Exception:
            # An invalid file is removed. A valid file without trusted provenance
            # is left in place until a verified replacement downloads successfully.
            try:
                validate_pdf(destination)
            except Exception:
                destination.unlink(missing_ok=True)
                pdf_sidecar_path(destination).unlink(missing_ok=True)

    last = None
    total_attempts = 0
    for candidate in row["permitted_urls"]:
        result = request_pdf(candidate, destination, source_fingerprint)
        total_attempts += int(result.get("download_attempts", 0))
        result["download_attempts"] = total_attempts
        last = result
        if result["download_status"] == "ok":
            result.update({"paper_id": row["paper_id"], "pdf_path": str(destination),
                           "download_seconds": round(time.perf_counter() - started, 3)})
            return result
    last = last or {"download_status": "failed", "download_error": "no permitted URL",
                    "download_attempts": 0, "download_source": "", "download_licence": "",
                    "download_url_used": "", "download_final_url": "", "http_content_type": "",
                    "pdf_bytes": None, "pdf_sha256": "", "pdf_pages": None}
    last.update({"paper_id": row["paper_id"], "pdf_path": str(destination),
                 "download_seconds": round(time.perf_counter() - started, 3)})
    return last


# Crossref retraction check. Success with no retraction is "clear"; missing DOI
# or any API failure remains explicitly unknown.
def check_retraction(row):
    doi = row["doi_key"]
    if not CHECK_CROSSREF_RETRACTIONS:
        return {"paper_id": row["paper_id"], "retraction_status": "not_checked",
                "retraction_error": "", "retraction_checked_at": ""}
    if not doi:
        return {"paper_id": row["paper_id"], "retraction_status": "unknown_no_doi",
                "retraction_error": "paper has no DOI", "retraction_checked_at": utc_now()}

    url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
    params = {"mailto": CROSSREF_MAILTO} if CROSSREF_MAILTO else None
    last = ""
    for attempt in range(1, MAX_URL_ATTEMPTS + 1):
        response = None
        try:
            response = get_session().get(url, params=params,
                                         timeout=(CONNECT_TIMEOUT_SECONDS, 60))
            if response.status_code in RETRYABLE_STATUS:
                raise requests.HTTPError(f"retryable HTTP {response.status_code}", response=response)
            if response.status_code == 404:
                return {"paper_id": row["paper_id"], "retraction_status": "unknown_not_found",
                        "retraction_error": "DOI not found by Crossref",
                        "retraction_checked_at": utc_now()}
            response.raise_for_status()
            message = response.json().get("message", {})
            updates = [item for key in ("update-to", "updated-by")
                       for item in (message.get(key, []) or []) if isinstance(item, dict)]
            retractions = [item for item in updates
                           if clean_text(item.get("type") or item.get("update-type")).lower()
                           == "retraction"]
            status = "retracted" if retractions else "clear"
            return {"paper_id": row["paper_id"], "retraction_status": status,
                    "retraction_notice_doi": clean_text(retractions[0].get("DOI")) if retractions else "",
                    "retraction_error": "", "retraction_checked_at": utc_now()}
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"[:500]
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if attempt < MAX_URL_ATTEMPTS and (status in RETRYABLE_STATUS or status is None):
                backoff(attempt, getattr(exc, "response", None))
                continue
            break
    return {"paper_id": row["paper_id"], "retraction_status": "unknown_error",
            "retraction_error": last, "retraction_checked_at": utc_now()}


# PDF parser. The Markdown is the LLM-facing form; structured page JSON is kept
# beside it so evidence can always be traced back to a PDF page.
import unicodedata
from difflib import SequenceMatcher

import pymupdf4llm

import shutil

TESSERACT = shutil.which("tesseract")
if OCR_REPAIR and not TESSERACT:
    print("OCR_REPAIR is on but no tesseract binary is on PATH. Image-only pages "
          "will stay empty and their papers will be flagged needs_ocr.")
elif OCR_REPAIR:
    print(f"OCR repair enabled at {OCR_DPI} dpi using {TESSERACT}")

PARSER_NAME = "pdf_inspector"
import pdf_inspector

try:                       # the wheel carries a version even when the module does not
    from importlib.metadata import version as _dist_version
    _parser_version = _dist_version("pdf-inspector")
except Exception:
    _parser_version = getattr(pdf_inspector, "__version__",
                              getattr(pdf_inspector, "version", "unknown"))
PARSER_VERSION = str(_parser_version() if callable(_parser_version) else _parser_version)
PARSER_CONFIG_HASH = hashlib.sha256(json.dumps({
    "parser": PARSER_NAME, "version": PARSER_VERSION,
    "ocr_repair": bool(OCR_REPAIR and TESSERACT), "ocr_dpi": OCR_DPI,
    "min_document_chars": MIN_DOCUMENT_CHARS, "min_good_pages": MIN_GOOD_PAGES,
    "primary_parser": "pdf_inspector",
    "min_text_chars_per_page": MIN_TEXT_CHARS_PER_PAGE,
    "min_title_token_overlap": MIN_TITLE_TOKEN_OVERLAP,
}, sort_keys=True).encode()).hexdigest()
PIPELINE_CONFIG_HASH = hashlib.sha256(json.dumps({
    "parser_config_hash": PARSER_CONFIG_HASH,
    "check_crossref_retractions": CHECK_CROSSREF_RETRACTIONS,
    "max_pdf_mb": MAX_PDF_MB, "max_download_seconds": MAX_DOWNLOAD_SECONDS,
    "permissive_licences": sorted(PERMISSIVE),
    "use_openalex_content_fallback": USE_OPENALEX_CONTENT_FALLBACK,
}, sort_keys=True).encode()).hexdigest()


def yaml_string(value):
    return json.dumps(clean_text(value), ensure_ascii=False)


def atomic_write_text(path, text):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_write_json(path, payload):
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def plain_page_chunks(pdf_path):
    chunks = []
    with fitz.open(pdf_path) as document:
        for number, page in enumerate(document, start=1):
            chunks.append({"metadata": {"page": number},
                           "text": page.get_text("text", sort=True)})
    return chunks


IDENTITY_STOPWORDS = {"the", "a", "an", "of", "in", "on", "and", "or", "for", "to",
                      "with", "from", "by", "using", "study", "analysis"}


def identity_check(row, pages):
    opening_text = "\n".join(page["text"] for page in pages[:3])
    expected_doi = doi_key(row.get("doi", ""))
    if expected_doi and expected_doi in opening_text.casefold():
        return "matched_doi", 1.0
    title = unicodedata.normalize("NFKD", clean_text(row.get("title"))).casefold()
    opening = unicodedata.normalize("NFKD", opening_text).casefold()
    title_tokens = [token for token in re.findall(r"[a-z0-9]+", title)
                    if len(token) >= 3 and token not in IDENTITY_STOPWORDS]
    opening_tokens = [token for token in re.findall(r"[a-z0-9]+", opening)
                      if len(token) >= 3 and token not in IDENTITY_STOPWORDS]
    unordered = len(set(title_tokens) & set(opening_tokens)) / max(len(set(title_tokens)), 1)
    ordered = SequenceMatcher(None, title_tokens, opening_tokens, autojunk=False).find_longest_match().size
    ordered /= max(len(title_tokens), 1)
    score = 0.6 * ordered + 0.4 * unordered
    return ("matched_title", score) if score >= MIN_TITLE_TOKEN_OVERLAP else ("identity_review", score)


def ocr_page_text(pdf_path, page_number):
    """Recover text from a page that carries no extractable text layer.
    page_number is 1-based."""
    with fitz.open(pdf_path) as document:
        page = document[page_number - 1]
        textpage = page.get_textpage_ocr(dpi=OCR_DPI, full=True)
        return page.get_text(textpage=textpage) or ""


def dense(text):
    return len(re.sub(r"\s+", "", text or ""))


def parse_one(item):
    row, downloaded = item
    started = time.perf_counter()
    pdf_path = Path(downloaded["pdf_path"])
    markdown_path = MARKDOWN_DIR / f"{row['paper_id']}.md"
    structured_path = STRUCTURED_DIR / f"{row['paper_id']}.json"

    if not FORCE_REPARSE and markdown_path.is_file() and structured_path.is_file():
        try:
            structured = json.loads(structured_path.read_text(encoding="utf-8"))
            if (structured.get("pdf_sha256") == downloaded.get("pdf_sha256") and
                    structured.get("parser_config_hash") == PARSER_CONFIG_HASH and
                    structured.get("parse_status") == "ok"):
                text = markdown_path.read_text(encoding="utf-8")
                return {"paper_id": row["paper_id"], "parse_status": structured["parse_status"],
                        "parse_error": "", "parser_name": structured.get("parser_name", ""),
                        "parser_version": structured.get("parser_version", ""),
                        "parser_warning": structured.get("parser_warning", ""),
                        "identity_status": structured.get("identity_status", ""),
                        "identity_score": structured.get("identity_score"),
                        "markdown_path": str(markdown_path),
                        "structured_path": str(structured_path),
                        "markdown_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "markdown_chars": len(text), "text_chars": structured.get("text_chars"),
                        "low_text_pages": structured.get("low_text_pages"),
                        "parse_seconds": round(time.perf_counter() - started, 3),
                        "parse_reused": True}
        except Exception:
            pass

    parser_used = PARSER_NAME
    parser_warning = ""
    try:
        # pdf_inspector returns one record per PDF page, which is what keeps the
        # page number on an EvidenceSpan honest. Its page index is 0-based.
        extracted = pdf_inspector.extract_pages_markdown(str(pdf_path))
        chunks = [{"metadata": {"page": int(page.page) + 1,
                                "needs_ocr": bool(page.needs_ocr)},
                   "text": page.markdown}
                  for page in extracted.pages]
        if not chunks:
            raise ValueError("pdf_inspector returned no pages")
    except Exception as primary_error:
        parser_warning = (f"pdf_inspector failed: {type(primary_error).__name__}: "
                          f"{primary_error}")[:1000]
        try:
            chunks = pymupdf4llm.to_markdown(
                str(pdf_path), page_chunks=True,
                write_images=False, embed_images=False, show_progress=False,
                table_strategy="lines_strict",
            )
            if isinstance(chunks, str):
                chunks = [{"metadata": {"page": 1}, "text": chunks}]
            parser_used = "pymupdf4llm_fallback"
        except Exception as second_error:
            parser_used = "pymupdf_plain_fallback"
            parser_warning += (f"; pymupdf4llm fallback failed: {type(second_error).__name__}: "
                               f"{second_error}")[:1000]
            try:
                chunks = plain_page_chunks(pdf_path)
            except Exception as fallback_error:
                return {"paper_id": row["paper_id"], "parse_status": "parse_failed",
                        "parse_error": f"{parser_warning}; plain fallback={fallback_error}"[:1000],
                        "parser_name": parser_used, "parser_version": getattr(fitz, "__version__", getattr(fitz, "VersionBind", "unknown")),
                        "parser_warning": parser_warning, "identity_status": "not_checked",
                        "identity_score": None, "markdown_path": "", "structured_path": "",
                        "markdown_sha256": "", "markdown_chars": None, "text_chars": None,
                        "low_text_pages": None,
                        "parse_seconds": round(time.perf_counter() - started, 3),
                        "parse_reused": False}

    # OCR repair: only pages that produced no usable embedded text.
    ocr_pages = []
    if OCR_REPAIR and TESSERACT:
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            number = int(metadata.get("page") or 0)
            if not number or dense(chunk.get("text")) >= MIN_TEXT_CHARS_PER_PAGE:
                continue
            try:
                recovered = ocr_page_text(pdf_path, number)
            except Exception as ocr_error:
                parser_warning = (parser_warning +
                                  f"; OCR failed on page {number}: {ocr_error}")[:1000]
                continue
            if dense(recovered) > dense(chunk.get("text")):
                chunk["text"] = recovered
                metadata["ocr_applied"] = True
                chunk["metadata"] = metadata
                ocr_pages.append(number)

    pages = []
    page_sections = []
    low_text_pages = 0
    for index, chunk in enumerate(chunks, start=1):
        text = unicodedata.normalize("NFC", clean_text(chunk.get("text", "")))
        if len(re.sub(r"\s+", "", text)) < MIN_TEXT_CHARS_PER_PAGE:
            low_text_pages += 1
        metadata = chunk.get("metadata") or {}
        page_number = metadata.get("page") or metadata.get("page_number") or index
        page_number = int(page_number)
        pages.append({"page": page_number, "metadata": metadata, "text": text})
        page_sections.append(
            f"<!-- MUFASA_PDF_PAGE: {page_number} -->\n\n## PDF page {page_number}\n\n{text.strip()}\n")

    body = "\n".join(page_sections).strip() + "\n"
    text_chars = sum(len(page["text"]) for page in pages)
    good_pages = sum(1 for page in pages
                     if dense(page["text"]) >= MIN_TEXT_CHARS_PER_PAGE)
    # Judge the body, not the page ratio. A paper with six solid pages and five
    # trailing plates is a usable paper; the old ratio rule discarded it.
    if not pages or text_chars < MIN_DOCUMENT_CHARS or good_pages < MIN_GOOD_PAGES:
        status = "needs_ocr"
        error = (f"only {good_pages} usable page(s) and {text_chars} characters "
                 f"after parsing and OCR")
    elif parser_used == "pymupdf_plain_fallback":
        status = "quality_review"
        error = "plain-text fallback may have lost tables or reading order"
    else:
        status, error = "ok", ""

    identity_status, identity_score = identity_check(row, pages)
    if status == "ok" and identity_status == "identity_review":
        status = "identity_review"
        error = "downloaded PDF did not match the expected DOI or title strongly enough"

    front_matter = "\n".join([
        "---",
        f"openalex_id: {yaml_string(row['openalex_id'])}",
        f"paper_id: {yaml_string(row['paper_id'])}",
        f"doi: {yaml_string(row.get('doi', ''))}",
        f"title: {yaml_string(row.get('title', ''))}",
        f"family_id: {yaml_string(row.get('family_id', ''))}",
        f"mufasa_domain: {yaml_string(row.get('model_mufasa_domain', ''))}",
        f"pdf_sha256: {yaml_string(downloaded.get('pdf_sha256', ''))}",
        f"licence: {yaml_string(downloaded.get('download_licence', ''))}",
        f"parser: {yaml_string(parser_used)}",
        "---",
        "",
        f"# {clean_text(row.get('title')) or row['paper_id']}",
        "",
    ])
    markdown = front_matter + body
    structured = {
        "schema_version": "1.0", "paper_id": row["paper_id"],
        "openalex_id": row["openalex_id"], "doi": clean_text(row.get("doi")),
        "title": clean_text(row.get("title")), "pdf_path": str(pdf_path),
        "pdf_sha256": downloaded.get("pdf_sha256", ""),
        "parser_name": parser_used,
        "parser_version": PARSER_VERSION if parser_used == PARSER_NAME else getattr(fitz, "__version__", getattr(fitz, "VersionBind", "unknown")),
        "parser_config_hash": PARSER_CONFIG_HASH, "parser_warning": parser_warning,
        "parse_status": status, "identity_status": identity_status,
        "identity_score": round(float(identity_score), 4),
        "text_chars": text_chars, "low_text_pages": low_text_pages,
        "good_pages": good_pages, "ocr_pages": ocr_pages, "pages": pages,
    }
    atomic_write_text(markdown_path, markdown)
    atomic_write_json(structured_path, structured)
    return {"paper_id": row["paper_id"], "parse_status": status,
            "parse_error": error, "parser_name": structured["parser_name"],
            "parser_version": structured["parser_version"],
            "parser_warning": parser_warning, "identity_status": identity_status,
            "identity_score": round(float(identity_score), 4),
            "markdown_path": str(markdown_path), "structured_path": str(structured_path),
            "markdown_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
            "markdown_chars": len(markdown), "text_chars": text_chars,
            "low_text_pages": low_text_pages, "good_pages": good_pages,
            "ocr_pages_count": len(ocr_pages),
            "parse_seconds": round(time.perf_counter() - started, 3),
            "parse_reused": False}


# Resumable batch state and execution. Each phase uses as_completed so one slow
# host cannot make the progress bar look frozen.
import shutil

BASE_COLUMNS = [
    "paper_id", "openalex_id", "doi", "title", "family_id", "field_name",
    "model_mufasa_domain", "publication_date", "cited_by_count",
    "rights_status", "metadata_corrupt",
]
PDF_DELETED_DEFAULT = False


def atomic_write_parquet(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_manifest():
    if FRESH_START or not MANIFEST_PATH.exists():
        return {
            "schema_version": "1.0", "source_fingerprint": SOURCE_FINGERPRINT,
            "pipeline_config_hash": PIPELINE_CONFIG_HASH,
            "source_rows": len(pool), "batch_size": BATCH_SIZE,
            "created_at": utc_now(), "batches": {},
        }
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if data.get("source_fingerprint") != SOURCE_FINGERPRINT:
        raise RuntimeError("Input queue changed. Set FRESH_START=True to create new batch state.")
    if data.get("pipeline_config_hash") != PIPELINE_CONFIG_HASH:
        raise RuntimeError("Pipeline settings changed. Set FRESH_START=True to rebuild state.")
    if int(data.get("batch_size", -1)) != BATCH_SIZE:
        raise RuntimeError("BATCH_SIZE changed. Restore it or set FRESH_START=True.")
    return data


manifest = load_manifest()
if FRESH_START:
    for path in BATCH_DIR.glob("batch_*.parquet"):
        path.unlink()
    for path in FAILURE_DIR.glob("batch_*.jsonl"):
        path.unlink()
    for filename in ("documents.parquet", "rights_review.csv", "quality_review.csv",
                     "run_summary.csv", "run_summary.json"):
        (MANIFEST_DIR / filename).unlink(missing_ok=True)

    # Remove only artifacts that no longer belong to a currently permitted
    # paper. Valid permitted PDFs remain reusable through their verified sidecars.
    permitted_ids = set(pool.loc[pool["rights_status"] == "permitted", "paper_id"])
    for path in PDF_DIR.glob("*.pdf"):
        if path.stem not in permitted_ids:
            path.unlink(missing_ok=True)
            pdf_sidecar_path(path).unlink(missing_ok=True)
    for path in MARKDOWN_DIR.glob("*.md"):
        if path.stem not in permitted_ids:
            path.unlink(missing_ok=True)
    for path in STRUCTURED_DIR.glob("*.json"):
        if path.stem not in permitted_ids:
            path.unlink(missing_ok=True)


def save_manifest():
    atomic_write_json(MANIFEST_PATH, manifest)


if FRESH_START:
    save_manifest()


def run_parallel(items, function, workers, description):
    answers = {}
    if not items:
        return answers
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(function, item): item["paper_id"] if isinstance(item, dict)
                   else item[0]["paper_id"] for item in items}
        with tqdm(total=len(futures), desc=description, leave=False) as progress:
            for future in concurrent.futures.as_completed(futures):
                paper_id = futures[future]
                try:
                    answers[paper_id] = future.result()
                except Exception as exc:
                    answers[paper_id] = {"paper_id": paper_id, "worker_error":
                                         f"{type(exc).__name__}: {exc}"[:1000]}
                progress.update(1)
    return answers


def completed_record_is_valid(record):
    try:
        markdown_path = Path(clean_text(record.get("markdown_path")))
        structured_path = Path(clean_text(record.get("structured_path")))
        pdf_path = Path(clean_text(record.get("pdf_path")))
        if not (markdown_path.is_file() and structured_path.is_file()):
            return False
        if hashlib.sha256(markdown_path.read_bytes()).hexdigest() != clean_text(record.get("markdown_sha256")):
            return False

        # A PDF we deleted on purpose is not a missing file. Its recorded
        # SHA-256 still ties the Markdown to the bytes it was parsed from, so
        # the record stays verifiable without the PDF being present.
        deleted = bool(record.get("pdf_deleted")) and not pdf_path.is_file()
        if deleted:
            recorded = clean_text(record.get("pdf_sha256"))
            if not recorded:
                return False
            pdf_details = {"pdf_sha256": recorded}
        else:
            if not pdf_path.is_file():
                return False
            pdf_details = validate_pdf(pdf_path)
            if pdf_details["pdf_sha256"] != clean_text(record.get("pdf_sha256")):
                return False
        structured = json.loads(structured_path.read_text(encoding="utf-8"))
        if (structured.get("parse_status") != "ok" or
                structured.get("parser_config_hash") != PARSER_CONFIG_HASH or
                structured.get("pdf_sha256") != pdf_details["pdf_sha256"]):
            return False
        sidecar = json.loads(pdf_sidecar_path(pdf_path).read_text(encoding="utf-8"))
        return (sidecar.get("pdf_sha256") == pdf_details["pdf_sha256"] and
                sidecar.get("licence") == clean_text(record.get("download_licence")) and
                sidecar.get("url") == clean_text(record.get("download_url_used")))
    except Exception:
        return False


def previous_is_terminal(record):
    if not record:
        return False
    if record.get("rights_status") == "rights_review":
        return True
    if record.get("retraction_status") == "retracted":
        return True
    if (CHECK_CROSSREF_RETRACTIONS and
            record.get("retraction_status") == "unknown_error"):
        return False
    if record.get("parse_status") == "ok" and not FORCE_REPARSE:
        return completed_record_is_valid(record)
    if record.get("parse_status") in {"quality_review", "identity_review"}:
        return True
    # A link that is simply wrong - 403, 404, gone, or no permitted URL at all -
    # will fail identically however often it is re-requested. Re-attempting it
    # wastes the run and hammers a host that has already refused us. Only
    # genuinely transient failures (timeouts, connection resets, 5xx) come back.
    if RETRY_ONLY_TRANSIENT and failure_class(record) == "permanent":
        return True
    return not RETRY_FAILURES and clean_text(record.get("parse_status")) != ""


def base_record(row):
    return {column: row.get(column, "") for column in BASE_COLUMNS}


# A 403, a 404 or a missing permitted URL is a decision, not a hiccup. Counting
# them as retryable meant a batch with any such paper could never be marked done,
# so the scheduler restarted batch 0 for ever and never reached batch 1.
PERMANENT_HTTP = {400, 401, 402, 403, 404, 405, 410, 451}


def failure_class(record):
    status = clean_text(record.get("pipeline_status"))
    if status == "needs_ocr":
        return "permanent"        # OCR repair already ran; retrying changes nothing
    blob = " ".join([clean_text(record.get("download_error")),
                     clean_text(record.get("parse_error")),
                     clean_text(record.get("retraction_error"))])
    if "no permitted URL" in blob or "PermanentDownloadError" in blob:
        return "permanent"
    for code in re.findall(r"\b([45]\d\d)\b", blob):
        if int(code) in PERMANENT_HTTP:
            return "permanent"
    return "retryable"


def sweep_stale_parts():
    """Remove .part files left behind by an interrupted run. Anything recent is
    left alone in case another process is still writing it."""
    cutoff = time.time() - STALE_PART_MINUTES * 60
    removed = 0
    for path in PDF_DIR.glob("*.part"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                removed += 1
        except OSError:
            pass
    if removed:
        print(f"swept {removed} stale .part file(s)")
    return removed


def run_batch(batch_no, frame):
    clock = time.perf_counter()
    previous_rounds = manifest["batches"].get(str(batch_no), {}).get("rounds", 0)
    batch_path = BATCH_DIR / f"batch_{batch_no:05d}.parquet"
    previous = {}
    if batch_path.exists() and not FRESH_START:
        previous = {r["paper_id"]: r for r in pd.read_parquet(batch_path).to_dict("records")}

    records = {}
    active = []
    for row in frame.to_dict("records"):
        old = previous.get(row["paper_id"])
        if previous_is_terminal(old):
            records[row["paper_id"]] = old
            continue
        record = base_record(row)
        if row["rights_status"] == "rights_review":
            record.update({"retraction_status": "not_checked", "download_status": "not_attempted",
                           "parse_status": "not_attempted", "pipeline_status": "rights_review"})
            records[row["paper_id"]] = record
        else:
            active.append(row)

    # Phase 1: retraction checks.
    retraction_items = []
    retractions = {}
    for row in active:
        old = previous.get(row["paper_id"], {})
        old_status = clean_text(old.get("retraction_status"))
        if old_status in {"clear", "retracted", "unknown_no_doi",
                          "unknown_not_found", "not_checked"}:
            retractions[row["paper_id"]] = {
                "paper_id": row["paper_id"], "retraction_status": old_status,
                "retraction_error": clean_text(old.get("retraction_error")),
                "retraction_checked_at": clean_text(old.get("retraction_checked_at")),
            }
        else:
            retraction_items.append(row)
    retractions.update(run_parallel(retraction_items, check_retraction, CROSSREF_WORKERS,
                                    f"batch {batch_no:05d} retractions"))

    # Phase 2: download every non-retracted paper concurrently.
    download_items = [row for row in active
                      if retractions[row["paper_id"]].get("retraction_status") != "retracted"]
    downloads = run_parallel(download_items, download_one, DOWNLOAD_WORKERS,
                             f"batch {batch_no:05d} downloads")

    # Phase 3: parse successful downloads concurrently.
    parse_items = [(row, downloads[row["paper_id"]]) for row in download_items
                   if downloads[row["paper_id"]].get("download_status") == "ok"]
    parses = run_parallel(parse_items, parse_one, PARSE_WORKERS,
                          f"batch {batch_no:05d} parsing")

    for row in active:
        paper_id = row["paper_id"]
        record = base_record(row)
        record.update(retractions[paper_id])
        if record.get("retraction_status") == "retracted":
            record.update({"download_status": "not_attempted", "parse_status": "not_attempted",
                           "pipeline_status": "retracted"})
        else:
            downloaded = downloads.get(paper_id, {})
            record.update(downloaded)
            if downloaded.get("download_status") == "ok":
                parsed = parses.get(paper_id, {})
                record.update(parsed)
                record["pipeline_status"] = parsed.get("parse_status", "parse_failed")
            else:
                record.update({"parse_status": "not_attempted", "parse_error": "download failed"})
                record["pipeline_status"] = "download_failed"
        records[paper_id] = record

    # Reclaim disk once the text is safely written. Only papers that parsed
    # cleanly qualify: anything under review may still need re-parsing, and a
    # failed download has no file to remove.
    if DELETE_PDF_AFTER_PARSE:
        freed = 0
        for paper_id, record in records.items():
            if record.get("parse_status") != "ok" or record.get("pdf_deleted"):
                continue
            pdf_path = Path(clean_text(record.get("pdf_path")))
            markdown_path = Path(clean_text(record.get("markdown_path")))
            structured_path = Path(clean_text(record.get("structured_path")))
            if not (markdown_path.is_file() and structured_path.is_file()):
                continue                      # never delete a source without its output
            try:
                if pdf_path.is_file():
                    freed += pdf_path.stat().st_size
                    pdf_path.unlink()
                record["pdf_deleted"] = True   # the sidecar is deliberately kept
            except OSError:
                record["pdf_deleted"] = False
        if freed:
            print(f"  reclaimed {freed / 1e9:.2f} GB by deleting parsed PDFs")

    output = pd.DataFrame([records[row["paper_id"]] for row in frame.to_dict("records")])
    atomic_write_parquet(output, batch_path)  # data first

    technical_failures = output[output["pipeline_status"].isin(
        ["download_failed", "parse_failed", "needs_ocr"]
    )]
    retraction_failures = output[output.get("retraction_status", "") == "unknown_error"]
    failures = pd.concat([technical_failures, retraction_failures]).drop_duplicates("paper_id")
    failures = failures.assign(failure_class=[failure_class(r)
                                              for r in failures.to_dict("records")])
    retryable = failures[failures["failure_class"] == "retryable"]
    failure_path = FAILURE_DIR / f"batch_{batch_no:05d}.jsonl"
    atomic_write_text(failure_path, "".join(
        json.dumps({"batch": batch_no, "paper_id": row.get("paper_id"),
                    "status": ("retraction_unknown" if row.get("retraction_status") == "unknown_error"
                               else row.get("pipeline_status")),
                    "error": (clean_text(row.get("retraction_error")) or
                              clean_text(row.get("download_error")) or clean_text(row.get("parse_error"))),
                    "failure_class": row.get("failure_class", ""),
                    "at": utc_now()}, ensure_ascii=False) + "\n"
        for row in failures.to_dict("records")
    ))

    counts = output["pipeline_status"].value_counts().to_dict()
    manifest["batches"][str(batch_no)] = {
        "file": batch_path.name, "complete": True, "papers": len(output),
        "counts": {str(k): int(v) for k, v in counts.items()},
        "failures_total": int(len(failures)),
        "failures_permanent": int((failures["failure_class"] == "permanent").sum())
                              if len(failures) else 0,
        "retryable_failures": int(len(retryable)),
        "rounds": int(previous_rounds) + 1,
        "seconds": round(time.perf_counter() - clock, 1), "finished_at": utc_now(),
    }
    save_manifest()  # manifest last
    print(f"batch {batch_no:05d}: {counts} in {manifest['batches'][str(batch_no)]['seconds']:.1f}s")
    return output


# Run unfinished batches until the queue is exhausted or MAX_BATCHES_THIS_RUN
# is reached. Failed documents are retried on the next run when RETRY_FAILURES=True.
if RUN_PREFLIGHT:
    preflight_ok = False
    preflight_errors = []
    candidates = pool[pool["rights_status"] == "permitted"].head(PREFLIGHT_CANDIDATES)
    for row in candidates.to_dict("records"):
        retraction = check_retraction(row)
        if retraction["retraction_status"] == "retracted":
            continue
        downloaded = download_one(row)
        if downloaded.get("download_status") != "ok":
            preflight_errors.append(f"{row['paper_id']}: {downloaded.get('download_error')}")
            continue
        parsed = parse_one((row, downloaded))
        if parsed.get("parse_status") == "ok":
            print(f"preflight passed: {row['paper_id']} ({parsed['parser_name']})")
            preflight_ok = True
            break
        preflight_errors.append(f"{row['paper_id']}: {parsed.get('parse_status')} - {parsed.get('parse_error')}")
    if not preflight_ok:
        raise RuntimeError("End-to-end preflight failed; bulk run stopped:\n  " +
                           "\n  ".join(preflight_errors))
else:
    print("preflight disabled")

sweep_stale_parts()

ran = 0
STOP_REASON = ""
for batch_no in range(TOTAL_BATCHES):
    entry = manifest["batches"].get(str(batch_no), {})
    batch_path = BATCH_DIR / f"batch_{batch_no:05d}.parquet"
    fully_done = (entry.get("complete") and batch_path.exists() and
                  (not RETRY_FAILURES
                   or int(entry.get("retryable_failures", 0)) == 0
                   or int(entry.get("rounds", 0)) >= MAX_BATCH_ROUNDS))
    if fully_done:
        continue
    if MAX_BATCHES_THIS_RUN is not None and ran >= MAX_BATCHES_THIS_RUN:
        break
    free_gb = shutil.disk_usage(OUTPUT_ROOT).free / (1024 ** 3)
    start = batch_no * BATCH_SIZE
    next_frame = pool.iloc[start:start + BATCH_SIZE]
    completed_files = sorted(BATCH_DIR.glob("batch_*.parquet"))
    estimated_next_gb = 0.0
    if completed_files:
        samples = pd.concat([pd.read_parquet(path, columns=["pdf_bytes", "markdown_chars"])
                             for path in completed_files], ignore_index=True)
        per_paper = (pd.to_numeric(samples["pdf_bytes"], errors="coerce").fillna(0) +
                     3 * pd.to_numeric(samples["markdown_chars"], errors="coerce").fillna(0))
        estimated_next_gb = float(per_paper.mean() * len(next_frame) * 1.25 / (1024 ** 3))
    if free_gb < MIN_FREE_GB + estimated_next_gb:
        STOP_REASON = (f"stopped before batch {batch_no:05d}: {free_gb:.1f} GB free, "
                       f"about {estimated_next_gb:.1f} GB expected for the batch")
        manifest["stopped_reason"] = STOP_REASON
        manifest["stopped_at"] = utc_now()
        save_manifest()
        print(STOP_REASON)
        break
    run_batch(batch_no, next_frame)
    ran += 1

print(f"completed {ran} batch(es) in this run")


# Rebuild the authoritative document manifest and concise human-readable outputs.
batch_files = sorted(BATCH_DIR.glob("batch_*.parquet"))
if not batch_files:
    print("No completed batch files yet.")
else:
    documents = pd.concat([pd.read_parquet(path) for path in batch_files], ignore_index=True)
    documents = documents.drop_duplicates("paper_id", keep="last")
    atomic_write_parquet(documents, MANIFEST_DIR / "documents.parquet")

    rights_review = documents[documents["pipeline_status"] == "rights_review"]
    rights_review.to_csv(MANIFEST_DIR / "rights_review.csv", index=False, encoding="utf-8-sig")
    quality_review = documents[documents["pipeline_status"].isin(
        ["quality_review", "identity_review", "needs_ocr"]
    )]
    quality_review.to_csv(MANIFEST_DIR / "quality_review.csv", index=False, encoding="utf-8-sig")

    status_counts = documents["pipeline_status"].value_counts().to_dict()
    summary = {
        "generated_at": utc_now(), "source_papers": len(pool),
        "papers_recorded": len(documents), "batches_recorded": len(batch_files),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "total_pdf_bytes": int(pd.to_numeric(documents.get("pdf_bytes"), errors="coerce").fillna(0).sum()),
        "output_root": str(OUTPUT_ROOT), "stopped_reason": manifest.get("stopped_reason", ""),
    }
    atomic_write_json(MANIFEST_DIR / "run_summary.json", summary)
    pd.DataFrame([{"metric": "source_papers", "value": len(pool)},
                  {"metric": "papers_recorded", "value": len(documents)},
                  *({"metric": f"status_{key}", "value": value}
                    for key, value in status_counts.items())]).to_csv(
        MANIFEST_DIR / "run_summary.csv", index=False, encoding="utf-8-sig"
    )

    display(pd.DataFrame([{"status": key, "papers": value}
                          for key, value in status_counts.items()]))
    print("\nSaved:")
    print(" ", MANIFEST_DIR / "documents.parquet")
    print(" ", MANIFEST_DIR / "rights_review.csv")
    print(" ", MANIFEST_DIR / "quality_review.csv")
    print(" ", MANIFEST_DIR / "run_summary.csv")
    print(" ", MANIFEST_DIR / "run_summary.json")

    if RUN_MODE == "smoke":
        good = int((documents["pipeline_status"] == "ok").sum())
        print("\n" + "=" * 68)
        if good:
            print(f"  SMOKE TEST PASSED - {good}/{len(documents)} papers parsed cleanly.")
            print('  To process the whole corpus: set RUN_MODE = "production"')
            print("  in the control panel and run all cells again.")
        else:
            print("  SMOKE TEST FAILED - no paper completed. Check quality_review.csv")
            print("  and the failures folder before starting production.")
        print("=" * 68)


# ## Kaggle persistence
#
# Everything created by this notebook is under `/kaggle/working/mufasa_ingestion`.
# When the run finishes, use **Save Version → Save & Run All** so Kaggle retains
# the output. If the extraction notebook runs in a later session, attach this
# saved notebook output as an input dataset; do not copy files into
# `/kaggle/input`, because that directory is read-only.

