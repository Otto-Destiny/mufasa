"""Converted from openalex.ipynb.

Generated for readability in review; the notebook remains the
authority for anything that was actually run.
"""


import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv(".env")

OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")


import requests

API_KEY = OPENALEX_API_KEY 

SCIENCE_FIELD_IDS = [
    11, 13, 15, 16, 17, 19, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 34, 35, 36,
]

filter_string = ",".join([
    "authorships.institutions.country_code:NG",
    f"primary_topic.field.id:{'|'.join(map(str, SCIENCE_FIELD_IDS))}",
    "type:article",
    "has_abstract:true",
    "is_retracted:false",
    "publication_year:2010-2026",
])

response = requests.get(
    "https://api.openalex.org/works",
    params={
        "api_key": API_KEY,
        "filter": filter_string,
        "select": "id",
        "per_page": 1,
    },
    timeout=30,
)

response.raise_for_status()
data = response.json()

print("Matching papers:", data["meta"]["count"])


import math
import requests
import pandas as pd

API_KEY = OPENALEX_API_KEY
TARGET_PER_FIELD = 3_000

START_DATE = "2000-01-01"
END_DATE = "2026-07-18"
LANGUAGE = "en"  # Change to None to include every language.

SCIENCE_FIELDS = {
    11: "Agricultural and Biological Sciences",
    13: "Biochemistry, Genetics and Molecular Biology",
    15: "Chemical Engineering",
    16: "Chemistry",
    17: "Computer Science",
    19: "Earth and Planetary Sciences",
    21: "Energy",
    22: "Engineering",
    23: "Environmental Science",
    24: "Immunology and Microbiology",
    25: "Materials Science",
    26: "Mathematics",
    27: "Medicine",
    28: "Neuroscience",
    29: "Nursing",
    30: "Pharmacology, Toxicology and Pharmaceutics",
    31: "Physics and Astronomy",
    34: "Veterinary",
    35: "Dentistry",
    36: "Health Professions",
}

session = requests.Session()
session.headers.update({
    "User-Agent": "MUFASA-OpenAlex-Count/1.0"
})

def get_count(filters):
    response = session.get(
        "https://api.openalex.org/works",
        params={
            "api_key": API_KEY,
            "filter": ",".join(filters),
            "select": "id",
            "per_page": 1,
        },
        timeout=60,
    )
    response.raise_for_status()
    meta = response.json()["meta"]
    return int(meta["count"]), float(meta.get("cost_usd", 0))

rows = []
count_query_cost = 0.0

for field_id, field_name in SCIENCE_FIELDS.items():
    base_filters = [
        "authorships.institutions.country_code:NG",
        f"primary_topic.field.id:{field_id}",
        "type:article",
        "has_abstract:true",
        "is_retracted:false",
        f"from_publication_date:{START_DATE}",
        f"to_publication_date:{END_DATE}",
    ]

    if LANGUAGE:
        base_filters.append(f"language:{LANGUAGE}")

    # Count before requiring open-access PDF availability.
    eligible_count, cost_1 = get_count(base_filters)

    # Final realistic pool for free paper downloading.
    downloadable_filters = base_filters + [
        "open_access.is_oa:true",
        "has_pdf_url:true",
    ]

    downloadable_count, cost_2 = get_count(downloadable_filters)
    count_query_cost += cost_1 + cost_2

    target = min(TARGET_PER_FIELD, downloadable_count)
    pages_needed = math.ceil(target / 100)
    estimated_extraction_cost = pages_needed * 0.0001

    rows.append({
        "field_id": field_id,
        "field_name": field_name,
        "eligible_articles": eligible_count,
        "open_access_with_pdf": downloadable_count,
        "oa_pdf_percentage": round(
            100 * downloadable_count / eligible_count, 1
        ) if eligible_count else 0,
        "requested": TARGET_PER_FIELD,
        "extractable": target,
        "shortfall": max(0, TARGET_PER_FIELD - downloadable_count),
        "pages_needed": pages_needed,
        "estimated_extraction_cost_usd": estimated_extraction_cost,
    })

df = pd.DataFrame(rows)

# Save the planning table.
df.to_csv("openalex_field_counts.csv", index=False)

display(df)

print(f"\nTotal eligible articles: {df['eligible_articles'].sum():,}")
print(f"Total OA papers with PDF URLs: {df['open_access_with_pdf'].sum():,}")
print(f"Planned papers to extract: {df['extractable'].sum():,}")
print(f"Count-query cost: ${count_query_cost:.4f}")
print(
    "Estimated metadata-extraction cost: "
    f"${df['estimated_extraction_cost_usd'].sum():.4f}"
)
print("\nSaved as: openalex_field_counts.csv")


# ## Full Extraction


from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv(".env")
API_KEY = os.getenv("OPENALEX_API_KEY")
if not API_KEY:
    raise ValueError("OPENALEX_API_KEY not found in .env")

WORKS_URL = "https://api.openalex.org/works"
RATE_URL = "https://api.openalex.org/rate-limit"
CONTENT_BASE_URL = "https://content.openalex.org/works"

COUNTRY = "NG"
START_DATE = "2000-01-01"
END_DATE = "2026-07-18"
LANGUAGE = "en"  # Set to None to include all languages.

MAX_WORKERS = 4
REQUESTS_PER_SECOND = 8  # Conservative relative to OpenAlex's published ceiling.
PER_PAGE = 100
MAX_RETRIES = 8
ALLOW_PAID_OVERAGE = False

# The live API will choose the first supported descending-sort syntax.
SORT_CANDIDATES = ("publication_date:desc", "-publication_date")
SORT_VALUE: str | None = None

OUT = Path("openalex_ng_science_2000_2026")
PARTS = OUT / "parts"
STATES = OUT / "states"
PARTS.mkdir(parents=True, exist_ok=True)
STATES.mkdir(parents=True, exist_ok=True)

MASTER_JSONL = OUT / "openalex_all_fields.jsonl"
MASTER_CSV = OUT / "openalex_all_fields.csv"
COUNTS_CSV = OUT / "field_counts.csv"
SUMMARY_CSV = OUT / "summary.csv"

FIELDS = {
    11: "Agricultural and Biological Sciences",
    13: "Biochemistry, Genetics and Molecular Biology",
    15: "Chemical Engineering",
    16: "Chemistry",
    17: "Computer Science",
    19: "Earth and Planetary Sciences",
    21: "Energy",
    22: "Engineering",
    23: "Environmental Science",
    24: "Immunology and Microbiology",
    25: "Materials Science",
    26: "Mathematics",
    27: "Medicine",
    28: "Neuroscience",
    29: "Nursing",
    30: "Pharmacology, Toxicology and Pharmaceutics",
    31: "Physics and Astronomy",
    34: "Veterinary",
    35: "Dentistry",
    36: "Health Professions",
}

# Audited against the live valid-select list returned by OpenAlex.
# Deliberately excludes content_url/content_urls because the docs and live schema
# have differed. has_content is stable; cached endpoints are derived from work ID.
SELECT_FIELDS = [
    "id",
    "doi",
    "ids",
    "title",
    "publication_year",
    "publication_date",
    "type",
    "language",
    "cited_by_count",
    "is_retracted",
    "is_paratext",
    "abstract_inverted_index",
    "authorships",
    "primary_topic",
    "topics",
    "keywords",
    "primary_location",
    "best_oa_location",
    "locations",
    "open_access",
    "indexed_in",
    "has_content",
    "created_date",
    "updated_date",
]

CSV_FIELDS = [
    "field_id",
    "field_name",
    "openalex_id",
    "doi",
    "title",
    "abstract",
    "publication_date",
    "publication_year",
    "work_type",
    "language",
    "cited_by_count",
    "is_retracted",
    "is_paratext",
    "primary_domain",
    "primary_field",
    "primary_subfield",
    "primary_topic",
    "topics_json",
    "keywords_json",
    "authors_json",
    "institutions_json",
    "countries_json",
    "journal",
    "is_open_access",
    "oa_status",
    "oa_url",
    "repository_has_fulltext",
    "download_url",
    "download_landing_page_url",
    "download_license",
    "download_license_id",
    "download_version",
    "download_source",
    "download_source_type",
    "download_url_missing",
    "best_oa_pdf_url",
    "best_oa_landing_page_url",
    "best_oa_license",
    "best_oa_license_id",
    "primary_pdf_url",
    "primary_landing_page_url",
    "primary_license",
    "all_download_candidates_json",
    "all_pdf_urls_json",
    "all_oa_pdf_urls_json",
    "openalex_cached_pdf_available",
    "openalex_grobid_xml_available",
    "openalex_content_pdf_endpoint",
    "openalex_content_grobid_xml_endpoint",
    "indexed_in_json",
    "ids_json",
    "created_date",
    "updated_date",
    "fetched_at",
]

# ============================================================
# HTTP, ERRORS, AND RATE LIMITING
# ============================================================

_local = threading.local()


class OpenAlexBadRequest(RuntimeError):
    def __init__(self, detail: Any):
        self.detail = detail
        super().__init__(f"OpenAlex rejected the request (400): {detail}")


class RateLimiter:
    def __init__(self, limit: int):
        self.limit = limit
        self.times: list[float] = []
        self.lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.times = [t for t in self.times if now - t < 1]
                if len(self.times) < self.limit:
                    self.times.append(now)
                    return
                delay = 1 - (now - self.times[0]) + 0.01
            time.sleep(max(delay, 0.01))


limiter = RateLimiter(REQUESTS_PER_SECOND)


def session() -> requests.Session:
    if not hasattr(_local, "session"):
        s = requests.Session()
        s.headers.update(
            {
                "User-Agent": "MUFASA-OpenAlex-Harvester/4.0",
                "Accept": "application/json",
            }
        )
        _local.session = s
    return _local.session


def response_detail(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text[:1500]


def api_get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    request_params = {**params, "api_key": API_KEY}
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        limiter.wait()
        try:
            response = session().get(
                url,
                params=request_params,
                timeout=(15, 90),
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code in {429, 500, 502, 503, 504}:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(2**attempt, 60)
                time.sleep(delay)
                continue

            detail = response_detail(response)

            if response.status_code == 400:
                raise OpenAlexBadRequest(detail)

            if response.status_code == 403:
                raise RuntimeError(
                    "OpenAlex rejected the API key or the daily budget is exhausted."
                )

            # Avoid requests' default exception because it includes the full URL,
            # which would expose the API key in notebook output.
            raise RuntimeError(
                f"OpenAlex request failed with HTTP {response.status_code}: {detail}"
            )

        except (OpenAlexBadRequest, RuntimeError):
            raise
        except requests.RequestException as error:
            last_error = error
            if attempt == MAX_RETRIES - 1:
                break
            time.sleep(min(2**attempt, 60))

    raise RuntimeError(
        f"OpenAlex request failed after retries: {type(last_error).__name__}: {last_error}"
    )


# ============================================================
# QUERY CONFIGURATION AND PREFLIGHT
# ============================================================


def filter_for(field_id: int) -> str:
    filters = [
        f"authorships.institutions.country_code:{COUNTRY}",
        f"primary_topic.field.id:{field_id}",
        "type:article",
        "has_abstract:true",
        "open_access.is_oa:true",
        "has_pdf_url:true",
        "is_retracted:false",
        f"from_publication_date:{START_DATE}",
        f"to_publication_date:{END_DATE}",
    ]
    if LANGUAGE:
        filters.append(f"language:{LANGUAGE}")
    return ",".join(filters)


def resolve_sort_syntax() -> str:
    """Probe the live endpoint and use the descending syntax it accepts."""
    field_id = next(iter(FIELDS))
    errors: list[str] = []

    for candidate in SORT_CANDIDATES:
        try:
            api_get(
                WORKS_URL,
                {
                    "filter": filter_for(field_id),
                    "select": "id,publication_date",
                    "sort": candidate,
                    "per_page": 1,
                    "cursor": "*",
                },
            )
            print(f"OpenAlex sort syntax selected: {candidate}")
            return candidate
        except OpenAlexBadRequest as error:
            errors.append(f"{candidate}: {error.detail}")

    raise RuntimeError(
        "OpenAlex rejected every supported descending-sort syntax:\n"
        + "\n".join(errors)
    )


def validate_full_query(sort_value: str) -> None:
    """Validate filters, select fields, sorting, and cursor pagination once."""
    field_id = next(iter(FIELDS))
    data = api_get(
        WORKS_URL,
        {
            "filter": filter_for(field_id),
            "select": ",".join(SELECT_FIELDS),
            "sort": sort_value,
            "per_page": 1,
            "cursor": "*",
        },
    )

    results = data.get("results") or []
    if not results:
        raise RuntimeError(
            "OpenAlex accepted the query but returned no preflight record. "
            "Check the filters before starting the full extraction."
        )

    record = results[0]
    missing = [field for field in SELECT_FIELDS if field not in record]
    if missing:
        # Some valid fields can be absent/null in individual records. This is only
        # informational; the API already accepted the select list.
        print("Preflight note: selected fields absent from test record:", missing)

    print("OpenAlex full-query preflight passed.")


# ============================================================
# DATA HELPERS
# ============================================================


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:70]


def paths(field_id: int, name: str) -> tuple[Path, Path]:
    stem = f"{field_id}_{slug(name)}"
    return PARTS / f"{stem}.jsonl", STATES / f"{stem}.json"


def signature(field_id: int, sort_value: str) -> str:
    raw = json.dumps(
        {
            "filter": filter_for(field_id),
            "select": SELECT_FIELDS,
            "sort": sort_value,
            "per_page": PER_PAGE,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def abstract_text(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    words = sorted(
        (position, word)
        for word, positions in index.items()
        for position in positions
    )
    return " ".join(word for _, word in words)


def valid_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except ValueError:
        return False


def uniq(items: list[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for item in items:
        if item in (None, "", []):
            continue
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def short_work_id(work: dict[str, Any]) -> str | None:
    work_id = work.get("id")
    if not isinstance(work_id, str):
        return None
    value = work_id.rstrip("/").rsplit("/", 1)[-1]
    return value if value.startswith("W") else None


def openalex_content_endpoints(work: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return keyless content endpoints; append ?api_key=... only when downloading."""
    wid = short_work_id(work)
    has_content = work.get("has_content") or {}
    if not wid:
        return None, None

    pdf_endpoint = (
        f"{CONTENT_BASE_URL}/{wid}.pdf"
        if has_content.get("pdf")
        else None
    )
    xml_endpoint = (
        f"{CONTENT_BASE_URL}/{wid}.grobid-xml"
        if has_content.get("grobid_xml")
        else None
    )
    return pdf_endpoint, xml_endpoint


def download_candidates(work: dict[str, Any]) -> list[dict[str, Any]]:
    ordered: list[tuple[str, dict[str, Any]]] = []
    best = work.get("best_oa_location") or {}
    primary = work.get("primary_location") or {}
    locations = work.get("locations") or []

    if best:
        ordered.append(("best_oa_location", best))

    ordered.extend(
        (f"oa_location_{index}", location)
        for index, location in enumerate(locations)
        if location and location.get("is_oa")
    )

    if primary and primary.get("is_oa"):
        ordered.append(("primary_oa_location", primary))

    ordered.extend(
        (f"location_{index}", location)
        for index, location in enumerate(locations)
        if location
    )

    if primary:
        ordered.append(("primary_location", primary))

    output: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for label, location in ordered:
        pdf_url = location.get("pdf_url")
        if not valid_url(pdf_url) or pdf_url in seen_urls:
            continue

        seen_urls.add(pdf_url)
        source = location.get("source") or {}
        version = location.get("version")
        if not version:
            if location.get("is_published"):
                version = "publishedVersion"
            elif location.get("is_accepted"):
                version = "acceptedVersion"

        output.append(
            {
                "label": label,
                "pdf_url": pdf_url,
                "landing_page_url": location.get("landing_page_url"),
                "is_oa": bool(location.get("is_oa")),
                "license": location.get("license"),
                "license_id": location.get("license_id"),
                "version": version,
                "source": source.get("display_name"),
                "source_type": source.get("type"),
            }
        )

    return output


def flatten(
    work: dict[str, Any],
    field_id: int,
    field_name: str,
    fetched_at: str,
) -> dict[str, Any]:
    topic = work.get("primary_topic") or {}
    open_access = work.get("open_access") or {}
    best = work.get("best_oa_location") or {}
    primary = work.get("primary_location") or {}
    candidates = download_candidates(work)

    chosen = next(
        (candidate for candidate in candidates if candidate["is_oa"]),
        candidates[0] if candidates else {},
    )

    authors: list[str] = []
    institutions: list[str] = []
    countries: list[str] = []

    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        if author.get("display_name"):
            authors.append(author["display_name"])

        countries.extend(authorship.get("countries") or [])

        for institution in authorship.get("institutions") or []:
            if institution.get("display_name"):
                institutions.append(institution["display_name"])
            if institution.get("country_code"):
                countries.append(institution["country_code"])

    topics = [
        {
            "id": item.get("id"),
            "name": item.get("display_name"),
            "score": item.get("score"),
            "domain": (item.get("domain") or {}).get("display_name"),
            "field": (item.get("field") or {}).get("display_name"),
            "subfield": (item.get("subfield") or {}).get("display_name"),
        }
        for item in work.get("topics") or []
    ]

    keywords = [
        {
            "id": item.get("id"),
            "name": item.get("display_name"),
            "score": item.get("score"),
        }
        for item in work.get("keywords") or []
    ]

    has_content = work.get("has_content") or {}
    content_pdf_endpoint, content_xml_endpoint = openalex_content_endpoints(work)
    primary_source = primary.get("source") or {}

    return {
        "field_id": field_id,
        "field_name": field_name,
        "openalex_id": work.get("id"),
        "doi": work.get("doi"),
        "title": work.get("title"),
        "abstract": abstract_text(work.get("abstract_inverted_index")),
        "publication_date": work.get("publication_date"),
        "publication_year": work.get("publication_year"),
        "work_type": work.get("type"),
        "language": work.get("language"),
        "cited_by_count": work.get("cited_by_count"),
        "is_retracted": work.get("is_retracted"),
        "is_paratext": work.get("is_paratext"),
        "primary_domain": (topic.get("domain") or {}).get("display_name"),
        "primary_field": (topic.get("field") or {}).get("display_name"),
        "primary_subfield": (topic.get("subfield") or {}).get("display_name"),
        "primary_topic": topic.get("display_name"),
        "topics_json": json.dumps(topics, ensure_ascii=False),
        "keywords_json": json.dumps(keywords, ensure_ascii=False),
        "authors_json": json.dumps(uniq(authors), ensure_ascii=False),
        "institutions_json": json.dumps(uniq(institutions), ensure_ascii=False),
        "countries_json": json.dumps(uniq(countries), ensure_ascii=False),
        "journal": primary_source.get("display_name"),
        "is_open_access": open_access.get("is_oa"),
        "oa_status": open_access.get("oa_status"),
        "oa_url": open_access.get("oa_url"),
        "repository_has_fulltext": open_access.get("any_repository_has_fulltext"),
        "download_url": chosen.get("pdf_url"),
        "download_landing_page_url": chosen.get("landing_page_url"),
        "download_license": chosen.get("license"),
        "download_license_id": chosen.get("license_id"),
        "download_version": chosen.get("version"),
        "download_source": chosen.get("source"),
        "download_source_type": chosen.get("source_type"),
        "download_url_missing": not bool(chosen.get("pdf_url")),
        "best_oa_pdf_url": best.get("pdf_url"),
        "best_oa_landing_page_url": best.get("landing_page_url"),
        "best_oa_license": best.get("license"),
        "best_oa_license_id": best.get("license_id"),
        "primary_pdf_url": primary.get("pdf_url"),
        "primary_landing_page_url": primary.get("landing_page_url"),
        "primary_license": primary.get("license"),
        "all_download_candidates_json": json.dumps(candidates, ensure_ascii=False),
        "all_pdf_urls_json": json.dumps(
            [item["pdf_url"] for item in candidates], ensure_ascii=False
        ),
        "all_oa_pdf_urls_json": json.dumps(
            [item["pdf_url"] for item in candidates if item["is_oa"]],
            ensure_ascii=False,
        ),
        "openalex_cached_pdf_available": has_content.get("pdf"),
        "openalex_grobid_xml_available": has_content.get("grobid_xml"),
        "openalex_content_pdf_endpoint": content_pdf_endpoint,
        "openalex_content_grobid_xml_endpoint": content_xml_endpoint,
        "indexed_in_json": json.dumps(work.get("indexed_in") or [], ensure_ascii=False),
        "ids_json": json.dumps(work.get("ids") or {}, ensure_ascii=False),
        "created_date": work.get("created_date"),
        "updated_date": work.get("updated_date"),
        "fetched_at": fetched_at,
    }


# ============================================================
# SAFE RESUME I/O
# ============================================================


def atomic_write(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def repair_tail(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return

    with path.open("rb+") as file:
        file.seek(-1, os.SEEK_END)
        if file.read(1) == b"\n":
            return

        position = file.tell() - 1
        while position > 0:
            position -= 1
            file.seek(position)
            if file.read(1) == b"\n":
                file.truncate(position + 1)
                return
        file.truncate(0)


def seen_ids(path: Path) -> set[str]:
    repair_tail(path)
    output: set[str] = set()

    if not path.exists():
        return output

    with path.open(encoding="utf-8") as file:
        for line in file:
            try:
                work_id = (json.loads(line).get("openalex") or {}).get("id")
                if work_id:
                    output.add(work_id)
            except json.JSONDecodeError:
                continue

    return output


def append_records(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return

    with path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


# ============================================================
# COUNTS
# ============================================================


def count_one(field_id: int, name: str) -> tuple[int, str, int, float]:
    data = api_get(
        WORKS_URL,
        {
            "filter": filter_for(field_id),
            "select": "id",
            "per_page": 1,
        },
    )
    meta = data.get("meta") or {}
    return (
        field_id,
        name,
        int(meta.get("count") or 0),
        float(meta.get("cost_usd") or 0),
    )


def get_counts() -> dict[int, int]:
    rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [
            pool.submit(count_one, field_id, name)
            for field_id, name in FIELDS.items()
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Counting",
            unit="field",
        ):
            field_id, name, count, cost = future.result()
            rows.append(
                {
                    "field_id": field_id,
                    "field_name": name,
                    "count": count,
                    "query_cost_usd": cost,
                }
            )

    rows.sort(key=lambda row: row["field_id"])

    with COUNTS_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return {row["field_id"]: row["count"] for row in rows}


# ============================================================
# EXTRACTION
# ============================================================


def load_or_create_state(
    field_id: int,
    name: str,
    part: Path,
    state_path: Path,
    sort_value: str,
) -> dict[str, Any]:
    current_signature = signature(field_id, sort_value)
    saved_ids = seen_ids(part)

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature") != current_signature:
            # Reset obsolete state safely only when no records were written.
            if not saved_ids:
                state_path.unlink(missing_ok=True)
            else:
                raise RuntimeError(
                    f"Configuration changed for {name}, but its part file contains "
                    "records. Use a new output folder or restore the old configuration."
                )

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "signature": current_signature,
            "field_id": field_id,
            "field_name": name,
            "cursor": "*",
            "saved": 0,
            "pages": 0,
            "cost_usd": 0.0,
            "complete": False,
            "newest_date": None,
            "oldest_date": None,
            "missing_download_urls": 0,
        }

    state["saved"] = len(saved_ids)
    return state


def extract_field(
    field_id: int,
    name: str,
    total: int,
    sort_value: str,
    bar: tqdm,
    bar_lock: threading.Lock,
) -> dict[str, Any]:
    part, state_path = paths(field_id, name)
    seen = seen_ids(part)
    state = load_or_create_state(field_id, name, part, state_path, sort_value)
    state["total"] = total

    if state.get("complete"):
        return state

    cursor = state.get("cursor") or "*"

    while cursor:
        data = api_get(
            WORKS_URL,
            {
                "filter": filter_for(field_id),
                "select": ",".join(SELECT_FIELDS),
                "sort": sort_value,
                "per_page": PER_PAGE,
                "cursor": cursor,
            },
        )

        meta = data.get("meta") or {}
        works = data.get("results") or []
        next_cursor = meta.get("next_cursor")

        if not works:
            state.update(
                {
                    "complete": True,
                    "cursor": None,
                    "reason": "no_more_results",
                }
            )
            atomic_write(state_path, state)
            break

        fetched_at = datetime.now(timezone.utc).isoformat()
        records: list[dict[str, Any]] = []
        missing_urls = 0

        for work in works:
            work_id = work.get("id")
            if not work_id or work_id in seen:
                continue

            flat = flatten(work, field_id, name, fetched_at)
            if flat["download_url_missing"]:
                missing_urls += 1

            records.append(
                {
                    "harvest": {
                        "field_id": field_id,
                        "field_name": name,
                        "filter": filter_for(field_id),
                        "sort": sort_value,
                        "fetched_at": fetched_at,
                    },
                    "flat": flat,
                    "openalex": work,
                }
            )
            seen.add(work_id)

        append_records(part, records)

        dates = [
            work.get("publication_date")
            for work in works
            if work.get("publication_date")
        ]

        previous_newest = state.get("newest_date")
        previous_oldest = state.get("oldest_date")
        page_newest = max(dates) if dates else None
        page_oldest = min(dates) if dates else None

        state.update(
            {
                "cursor": next_cursor,
                "saved": len(seen),
                "pages": int(state.get("pages") or 0) + 1,
                "cost_usd": float(state.get("cost_usd") or 0)
                + float(meta.get("cost_usd") or 0),
                "newest_date": max(
                    [value for value in (previous_newest, page_newest) if value],
                    default=None,
                ),
                "oldest_date": min(
                    [value for value in (previous_oldest, page_oldest) if value],
                    default=None,
                ),
                "missing_download_urls": int(
                    state.get("missing_download_urls") or 0
                )
                + missing_urls,
                "updated_at": fetched_at,
            }
        )

        if not next_cursor:
            state.update(
                {
                    "complete": True,
                    "reason": "cursor_exhausted",
                }
            )

        atomic_write(state_path, state)

        with bar_lock:
            bar.update(len(records))
            bar.set_postfix_str(
                f"{name[:20]} {state['saved']:,}/{total:,} | "
                f"oldest {state.get('oldest_date') or '?'}"
            )

        if state.get("complete"):
            break

        cursor = next_cursor

    return state


# ============================================================
# FINAL OUTPUTS
# ============================================================


def iter_part(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_master_files() -> None:
    part_files = [
        paths(field_id, name)[0]
        for field_id, name in FIELDS.items()
        if paths(field_id, name)[0].exists()
    ]

    streams = [iter_part(path) for path in part_files]
    merged = heapq.merge(
        *streams,
        key=lambda record: (record.get("flat") or {}).get("publication_date")
        or "0000-00-00",
        reverse=True,
    )

    json_temporary = MASTER_JSONL.with_suffix(".jsonl.tmp")
    csv_temporary = MASTER_CSV.with_suffix(".csv.tmp")

    with json_temporary.open("w", encoding="utf-8") as json_output, csv_temporary.open(
        "w", newline="", encoding="utf-8"
    ) as csv_output:
        writer = csv.DictWriter(
            csv_output,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()

        for record in merged:
            json_output.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.writerow(record.get("flat") or {})

        json_output.flush()
        os.fsync(json_output.fileno())
        csv_output.flush()
        os.fsync(csv_output.fileno())

    json_temporary.replace(MASTER_JSONL)
    csv_temporary.replace(MASTER_CSV)


def write_summary(
    counts: dict[int, int],
    states: list[dict[str, Any]],
) -> None:
    by_id = {
        int(state["field_id"]): state
        for state in states
        if state.get("field_id") is not None
    }

    rows: list[dict[str, Any]] = []
    for field_id, name in FIELDS.items():
        state = by_id.get(field_id, {})
        rows.append(
            {
                "field_id": field_id,
                "field_name": name,
                "expected": counts[field_id],
                "saved": state.get("saved", 0),
                "complete": state.get("complete", False),
                "pages": state.get("pages", 0),
                "api_cost_usd": state.get("cost_usd", 0),
                "newest_date": state.get("newest_date"),
                "oldest_date": state.get("oldest_date"),
                "missing_download_urls": state.get("missing_download_urls", 0),
                "error": state.get("error"),
            }
        )

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# RUN
# ============================================================

SORT_VALUE = resolve_sort_syntax()
validate_full_query(SORT_VALUE)
counts = get_counts()

existing = {
    field_id: len(seen_ids(paths(field_id, name)[0]))
    for field_id, name in FIELDS.items()
}

total = sum(counts.values())
already = sum(existing.values())
remaining_pages = sum(
    math.ceil(max(0, counts[field_id] - existing[field_id]) / PER_PAGE)
    for field_id in FIELDS
)

rate_payload = api_get(RATE_URL, {})
rate = rate_payload.get("rate_limit") or {}
list_cost = float((rate.get("endpoint_costs_usd") or {}).get("list") or 0.0001)
estimated_cost = remaining_pages * list_cost
remaining_budget = rate.get("daily_remaining_usd")

print(f"\nFiltered papers ({START_DATE} to {END_DATE}): {total:,}")
print(f"Already saved: {already:,}")
print(f"Estimated remaining requests: {remaining_pages:,}")
print(f"Estimated remaining API cost: ${estimated_cost:.4f}")

if remaining_budget is not None:
    print(f"Free allowance remaining today: ${float(remaining_budget):.4f}")
    if not ALLOW_PAID_OVERAGE and float(remaining_budget) < estimated_cost:
        raise RuntimeError(
            "Not enough free allowance today. Rerun after the daily reset or "
            "set ALLOW_PAID_OVERAGE=True only if you intend to use prepaid credit."
        )

states: list[dict[str, Any]] = []
bar_lock = threading.Lock()

with tqdm(
    total=total,
    initial=min(already, total),
    desc="Extracting",
    unit="paper",
    dynamic_ncols=True,
    smoothing=0.1,
) as bar:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(
                extract_field,
                field_id,
                name,
                counts[field_id],
                SORT_VALUE,
                bar,
                bar_lock,
            ): (field_id, name)
            for field_id, name in FIELDS.items()
        }

        for future in as_completed(futures):
            field_id, name = futures[future]
            try:
                state = future.result()
                states.append(state)
                tqdm.write(
                    f"Completed {name}: {state.get('saved', 0):,}/{counts[field_id]:,}"
                )
            except Exception as error:
                states.append(
                    {
                        "field_id": field_id,
                        "field_name": name,
                        "error": repr(error),
                        "complete": False,
                    }
                )
                tqdm.write(f"FAILED {name}: {error}")

write_summary(counts, states)

if len(states) == len(FIELDS) and all(state.get("complete") for state in states):
    print("\nAll fields complete. Building globally date-sorted master files...")
    build_master_files()
    print("JSONL:", MASTER_JSONL)
    print("CSV:  ", MASTER_CSV)
else:
    print("\nSome fields are incomplete. Rerun this script to resume safely.")

print("Counts:", COUNTS_CSV)
print("Summary:", SUMMARY_CSV)

