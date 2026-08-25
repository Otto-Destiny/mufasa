"""Converted from quarantine-minilm-retrieval.ipynb.

Generated for readability in review; the notebook remains the
authority for anything that was actually run.
"""


# # Refine the latest MUFASA support quarantine with local BGE
#
# This notebook is a resumable post-processing stage for the **latest immutable
# training-set generation on Google Drive**. It does not reconstruct a random
# sample from extraction tables.
#
# It selects only FACTUAL/REASONING pairs that are both:
#
# - `SUPPORT / QUARANTINE_UNVERIFIED` in the builder's `quarantine.parquet`; and
# - `UNVERIFIED` in that same generation's permissive `sft_mixed.parquet`.
#
# The default is 10 affected papers. Set `PAPER_LIMIT = None` for complete
# coverage. The pinned 768-dimensional `BAAI/bge-base-en-v1.5` encoder retrieves
# exact passages only from each pair's own paper. Cosine similarity is a locator,
# not an entailment verdict, so every refined row remains explicitly
# `UNVERIFIED`.
#
# Each completed paper is checkpointed atomically on Drive without saving model
# vectors. A disconnected Colab run resumes from those checkpoints. Ten-paper
# smoke results are persisted as immutable `semantic_runs/` and never advance the production
# semantic pointer. Only a complete (`PAPER_LIMIT = None`) run may publish the
# derived `sft_mixed.parquet` generation and switch its `LATEST.json`.


# =================== Drive, modules and source generation ===================
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import textwrap
import time
import uuid
import warnings
from collections import Counter
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from google.colab import drive
except ImportError:
    drive = None
    IN_COLAB = False
else:
    drive.mount("/content/drive")
    IN_COLAB = True

if IN_COLAB and importlib.util.find_spec("sentence_transformers") is None:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "sentence-transformers==5.5.1",
    ])

ALLOW_MODEL_DOWNLOAD = os.environ.get(
    "MUFASA_ALLOW_MODEL_DOWNLOAD", "1" if IN_COLAB else "0",
) == "1"
for flag in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    if ALLOW_MODEL_DOWNLOAD:
        os.environ.pop(flag, None)
    else:
        os.environ[flag] = "1"

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

DRIVE_DATA = Path("/content/drive/MyDrive/MUFASA/01-data-engineering/data-extraction")
override = os.environ.get("MUFASA_DATA_DIR", "").strip()
candidates = ([Path(override)] if override else []) + ([DRIVE_DATA] if IN_COLAB else [])
candidates += [
    folder for start in [Path.cwd(), *Path.cwd().parents]
    for folder in (start, start / "01-data-engineering" / "data-extraction")
]
DATA = next((
    folder.resolve() for folder in candidates
    if (folder / "training_set" / "LATEST.json").is_file()
    and (folder / "mufasa_corpus" / "parsed" / "markdown").is_dir()
), None)
if DATA is None:
    raise FileNotFoundError(
        "Run build-training-set.ipynb first, or set MUFASA_DATA_DIR to the "
        "Drive data-extraction folder containing training_set/LATEST.json"
    )

MODULE_NAMES = (
    "mufasa_citations.py", "mufasa_dataset.py", "mufasa_semantic.py",
    "mufasa_training_builder.py",
    "mufasa_semantic_refiner.py",
)
for module_name in MODULE_NAMES:
    if not (DATA / module_name).is_file():
        raise FileNotFoundError(f"Place {module_name} beside this notebook at {DATA}")
sys.path.insert(0, str(DATA))
import mufasa_citations as citations
import mufasa_dataset as funnel
import mufasa_semantic as semantic
import mufasa_training_builder as builder
import mufasa_semantic_refiner as refiner
importlib.reload(citations)
importlib.reload(funnel)
importlib.reload(semantic)
importlib.reload(builder)
importlib.reload(refiner)
for loaded, filename in (
    (citations, "mufasa_citations.py"),
    (funnel, "mufasa_dataset.py"),
    (semantic, "mufasa_semantic.py"),
    (builder, "mufasa_training_builder.py"),
    (refiner, "mufasa_semantic_refiner.py"),
):
    actual = Path(loaded.__file__).resolve()
    expected = (DATA / filename).resolve()
    if actual != expected:
        raise RuntimeError(f"Imported wrong {filename}: {actual}; expected {expected}")

# Ten unique affected papers by default. None means full semantic refinement.
PAPER_LIMIT = 10
SEED = 7
SHOW = 5
MAX_TOKENS = 480
OVERLAP_TOKENS = 48
BGE_BATCH_SIZE = int(os.environ.get("MUFASA_BGE_BATCH_SIZE", "512"))
TOP_K_PER_QUERY = 8
CANDIDATE_LIMIT = 20
NEIGHBOR_RADIUS = 1
CANDIDATE_POOL_TOKENS = 2_400
BUNDLE_BEAM = 10
MAX_SPANS = 3

TRAINING_ROOT = DATA / "training_set"
MARKDOWN = DATA / "mufasa_corpus" / "parsed" / "markdown"

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

latest = json.loads((TRAINING_ROOT / "LATEST.json").read_text(encoding="utf-8"))
SOURCE_RUN = (TRAINING_ROOT / latest["directory"]).resolve()
if SOURCE_RUN.parent != (TRAINING_ROOT / "runs").resolve():
    raise RuntimeError("training_set/LATEST.json escapes training_set/runs")
success_path = SOURCE_RUN / "_SUCCESS.json"
if not success_path.is_file():
    raise FileNotFoundError(f"Incomplete source generation: {success_path}")
SOURCE_MARKER = json.loads(success_path.read_text(encoding="utf-8"))
if SOURCE_MARKER.get("run_id") != latest.get("run_id"):
    raise RuntimeError("LATEST.json and source _SUCCESS.json disagree on run_id")
source_provenance_mode = str(
    (SOURCE_MARKER.get("config") or {}).get("provenance_mode", "OFF")
).upper()
if source_provenance_mode != "TRAINING":
    raise RuntimeError(
        "Production is paused: semantic refinement requires a builder run "
        "published explicitly with provenance_mode='TRAINING'. Review the "
        "AUDIT_ONLY citation samples in build-training-set.ipynb first."
    )
if SOURCE_MARKER.get("identity_sha256") != latest.get("identity_sha256"):
    raise RuntimeError("LATEST.json and source _SUCCESS.json disagree on identity")

for filename in ("quarantine.parquet", "sft_mixed.parquet"):
    path = SOURCE_RUN / filename
    metadata = (SOURCE_MARKER.get("files") or {}).get(filename)
    if not path.is_file() or not isinstance(metadata, dict):
        raise FileNotFoundError(
            f"{filename} is absent. Upload/run the migrated build-training-set pipeline first."
        )
    if metadata.get("sha256") != sha256_file(path):
        raise RuntimeError(f"Source hash mismatch for {filename}")

# The refiner independently verifies the complete builder file manifest and
# owns the exact eligibility contract used again during publication.
SOURCE = refiner.load_source_run(TRAINING_ROOT)
if SOURCE.run_dir != SOURCE_RUN:
    raise RuntimeError("Refiner and notebook resolved different builder generations")
if SOURCE.marker["identity_sha256"] != SOURCE_MARKER["identity_sha256"]:
    raise RuntimeError("Refiner and notebook disagree on the builder identity")
if PAPER_LIMIT is None and not SOURCE.source_complete:
    raise RuntimeError(
        "Full semantic publication requires a full builder generation; rerun "
        "build-training-set.ipynb with PAPER_LIMIT=None first"
    )

quarantine = pd.read_parquet(SOURCE_RUN / "quarantine.parquet")
sft_mixed_base = pd.read_parquet(SOURCE_RUN / "sft_mixed.parquet")
support_quarantine = quarantine[
    quarantine["stage"].eq("SUPPORT")
    & quarantine["reason_code"].eq("QUARANTINE_UNVERIFIED")
    & quarantine["pair_type"].isin(["FACTUAL", "REASONING"])
].copy()
unverified_ids = set(sft_mixed_base.loc[
    sft_mixed_base["verification_tier"].eq("UNVERIFIED")
    & sft_mixed_base["pair_type"].isin(["FACTUAL", "REASONING"]),
    "pair_id",
])
support_quarantine = support_quarantine[
    support_quarantine["pair_id"].isin(unverified_ids)
].copy()
if set(support_quarantine["pair_id"]) != set(SOURCE.eligible_pair_ids):
    raise RuntimeError("Notebook selection disagrees with refiner eligibility contract")
if support_quarantine["pair_id"].duplicated().any():
    repeated = support_quarantine.loc[
        support_quarantine["pair_id"].duplicated(False), "pair_id",
    ].head(20).tolist()
    raise RuntimeError(f"Duplicate SUPPORT quarantine records: {repeated}")
if support_quarantine.empty:
    raise RuntimeError("The latest builder generation has no support-quarantined F/R pairs")

paper_order = sorted(
    support_quarantine["paper_id"].unique(),
    key=lambda paper: hashlib.sha256(f"{SEED}:{paper}".encode()).hexdigest(),
)
selected_papers = paper_order if PAPER_LIMIT is None else paper_order[:PAPER_LIMIT]
SUBSET = set(selected_papers)
selected_quarantine = support_quarantine[
    support_quarantine["paper_id"].isin(SUBSET)
].sort_values("pair_id", kind="stable").reset_index(drop=True)

records = []
for row in selected_quarantine.itertuples(index=False):
    record = json.loads(row.source_row_json)
    if record.get("pair_id") != row.pair_id or record.get("paper_id") != row.paper_id:
        raise RuntimeError(f"source_row_json identity mismatch for {row.pair_id}")
    record["pair_type"] = funnel.clean(record.get("pair_type")).upper()
    records.append(record)
pairs = pd.DataFrame(records)
if set(pairs["pair_id"]) != set(selected_quarantine["pair_id"]):
    raise RuntimeError("Selected source rows do not match quarantine identities")
ROW = {row.pair_id: row for row in pairs.itertuples(index=False)}
QUARANTINE_BY_ID = selected_quarantine.set_index("pair_id").to_dict("index")
BASE_ROUTES = {
    pair_id: {
        "route": row["support_route"],
        "bundle": json.loads(row["evidence_json"] or "[]"),
        "report": json.loads(row["support_report_json"] or "{}"),
        "paper_verified": False,
    }
    for pair_id, row in QUARANTINE_BY_ID.items()
}
for pair_id in BASE_ROUTES:
    BASE_ROUTES[pair_id]["bundle"] = SOURCE.initial_evidence[pair_id]

RETRIEVAL_CONFIG = {
    "max_tokens": MAX_TOKENS,
    "overlap_tokens": OVERLAP_TOKENS,
    "bge_batch_size": BGE_BATCH_SIZE,
    "top_k_per_query": TOP_K_PER_QUERY,
    "candidate_limit": CANDIDATE_LIMIT,
    "neighbor_radius": NEIGHBOR_RADIUS,
    "candidate_pool_tokens": CANDIDATE_POOL_TOKENS,
    "bundle_beam": BUNDLE_BEAM,
    "max_spans": MAX_SPANS,
    "model_id": semantic.MODEL_ID,
    "model_revision": semantic.PINNED_BGE_REVISION,
}

def wrap(label, value, width=94):
    body = textwrap.fill(str(value), width=width,
                         initial_indent="      ", subsequent_indent="      ")
    print(f"   {label}\n{body}")

print("data root                :", DATA)
print("source builder run       :", SOURCE_MARKER["run_id"])
print("source builder identity  :", SOURCE_MARKER["identity_sha256"])
print("source builder complete  :", SOURCE.source_complete)
print("eligible quarantine      :", f"{len(support_quarantine):,} pairs / "
      f"{support_quarantine.paper_id.nunique():,} papers")
print("selected                 :", f"{len(pairs):,} pairs / {len(SUBSET):,} papers")
print("paper limit              :", PAPER_LIMIT if PAPER_LIMIT is not None else "FULL")
print("full run may publish     :", PAPER_LIMIT is None and SOURCE.source_complete)
print("CPU                      :", platform.processor() or platform.machine())


# ======================= pinned 768-d BGE encoder =========================
model_started = time.perf_counter()
SNAPSHOT = os.environ.get("MUFASA_BGE_SNAPSHOT")
MODEL_CACHE = Path(os.environ.get("MUFASA_BGE_CACHE", str(DATA / ".hf-cache")))
MODEL_CACHE.mkdir(parents=True, exist_ok=True)
if IN_COLAB and not torch.cuda.is_available():
    raise RuntimeError("Enable a GPU runtime before running semantic refinement")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = semantic.load_bge(
    SNAPSHOT,
    cache_dir=MODEL_CACHE,
    device=DEVICE,
    local_files_only=not ALLOW_MODEL_DOWNLOAD,
)
EMBEDDING_DIM = int(MODEL.get_sentence_embedding_dimension())
if EMBEDDING_DIM != 768:
    raise RuntimeError(f"Expected a 768-dimensional BGE model; loaded {EMBEDDING_DIM}")
MODEL_LIMIT = int(MODEL.max_seq_length)
SPECIAL_TOKENS = int(MODEL.tokenizer.num_special_tokens_to_add(pair=False))
if MAX_TOKENS + SPECIAL_TOKENS > MODEL_LIMIT:
    raise RuntimeError("Semantic chunk size exceeds the encoder sequence limit")

print("model ID          :", semantic.MODEL_ID)
print("pinned revision   :", semantic.PINNED_BGE_REVISION)
print("embedding dim     :", EMBEDDING_DIM)
print("device            :", DEVICE)
if torch.cuda.is_available():
    print("GPU               :", torch.cuda.get_device_name(0))
print("model cache       :", MODEL_CACHE)
print("BGE batch size    :", BGE_BATCH_SIZE)
print(f"model load seconds: {time.perf_counter() - model_started:.1f}")


# =========== resumable same-paper retrieval and deterministic audit =========
def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return jsonable(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return jsonable(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value

def canonical_json(value):
    return json.dumps(
        jsonable(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )

retrieval_identity = {
    "format": "mufasa-semantic-paper-checkpoint-v1",
    "source_run_id": SOURCE_MARKER["run_id"],
    "source_identity_sha256": SOURCE_MARKER["identity_sha256"],
    "retrieval_config": RETRIEVAL_CONFIG,
    "dataset_module_sha256": sha256_file(DATA / "mufasa_dataset.py"),
    "semantic_module_sha256": sha256_file(DATA / "mufasa_semantic.py"),
}
retrieval_identity_sha = hashlib.sha256(
    canonical_json(retrieval_identity).encode("utf-8")
).hexdigest()
CHECKPOINT_DIR = (
    TRAINING_ROOT / "semantic-checkpoints" / SOURCE_MARKER["run_id"]
    / retrieval_identity_sha[:24]
)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

RAW_SEMANTIC_HITS = {}
SEMANTIC_HITS = {}
VECTOR_AUDIT = {}
CHUNK_STATS = []
PAYLOAD_HASHES = {}
checkpoint_hits = 0
semantic_started = time.perf_counter()
paper_group_count = int(pairs["paper_id"].nunique())
groups = pairs.groupby("paper_id", sort=True)
try:
    from tqdm.auto import tqdm
    iterator = tqdm(groups, total=paper_group_count, unit="paper", desc="local BGE refinement")
except ImportError:
    iterator = groups

for paper_id, group in iterator:
    paper_records = group.sort_values("pair_id", kind="stable").to_dict("records")
    paper_path = MARKDOWN / f"{paper_id}.md"
    if not paper_path.is_file():
        raise FileNotFoundError(f"Missing source paper: {paper_path}")
    paper_input = {
        **retrieval_identity,
        "paper_id": paper_id,
        "paper_markdown_sha256": sha256_file(paper_path),
        "pairs": [
            {
                "pair_id": record["pair_id"],
                "source_row_sha256": hashlib.sha256(
                    canonical_json(record).encode("utf-8")
                ).hexdigest(),
                "source_evidence_sha256": hashlib.sha256(
                    canonical_json(BASE_ROUTES[record["pair_id"]]["bundle"]).encode("utf-8")
                ).hexdigest(),
            }
            for record in paper_records
        ],
    }
    payload_sha = hashlib.sha256(
        canonical_json(paper_input).encode("utf-8")
    ).hexdigest()
    PAYLOAD_HASHES[paper_id] = payload_sha
    checkpoint = CHECKPOINT_DIR / (
        hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:24] + ".json"
    )
    saved = None
    if checkpoint.is_file():
        try:
            candidate = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (
                candidate.get("payload_sha256") == payload_sha
                and set((candidate.get("vector_audit") or {}))
                == {record["pair_id"] for record in paper_records}
            ):
                saved = candidate
        except (OSError, UnicodeError, json.JSONDecodeError):
            saved = None
    if saved is not None:
        checkpoint_hits += 1
        raw_ranked = saved["raw_semantic_hits"]
        expanded = saved["semantic_hits"]
        audited = saved["vector_audit"]
        chunk_stats = saved["chunk_stats"]
    else:
        index = semantic.build_paper_semantic_index(
            paper_id, MARKDOWN, MODEL,
            max_tokens=MAX_TOKENS,
            overlap_tokens=OVERLAP_TOKENS,
            batch_size=BGE_BATCH_SIZE,
        )
        if any(chunk["paper_id"] != paper_id for chunk in index.chunks):
            raise AssertionError(f"Cross-paper semantic chunk for {paper_id}")
        if any(chunk["token_count"] > MAX_TOKENS for chunk in index.chunks):
            raise AssertionError(f"Semantic chunk truncation risk for {paper_id}")
        raw_ranked = semantic.rank_semantic_records(
            index, paper_records, MODEL,
            top_k_per_query=TOP_K_PER_QUERY,
            candidate_limit=CANDIDATE_LIMIT,
            batch_size=BGE_BATCH_SIZE,
        )
        expanded = {
            record["pair_id"]: semantic.expand_ranked_hits(
                index,
                raw_ranked[record["pair_id"]],
                MODEL.tokenizer,
                supplied_spans=BASE_ROUTES[record["pair_id"]]["bundle"],
                neighbor_radius=NEIGHBOR_RADIUS,
                max_pool_tokens=CANDIDATE_POOL_TOKENS,
            )
            for record in paper_records
        }
        audited = semantic.evaluate_ranked_paper(
            paper_records,
            paper_id,
            expanded,
            max_spans=MAX_SPANS,
            bundle_beam=BUNDLE_BEAM,
        )
        chunk_stats = {
            "paper_id": paper_id,
            "chunks": len(index.chunks),
            "table_chunks": sum(
                chunk["source_kind"] == "TABLE" for chunk in index.chunks
            ),
            "max_tokens": max(
                (chunk["token_count"] for chunk in index.chunks), default=0
            ),
        }
        saved = jsonable({
            "format": "mufasa-semantic-paper-checkpoint-v1",
            "payload_sha256": payload_sha,
            "paper_id": paper_id,
            "raw_semantic_hits": raw_ranked,
            "semantic_hits": expanded,
            "vector_audit": audited,
            "chunk_stats": chunk_stats,
        })
        temporary = checkpoint.with_name(f".{checkpoint.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(canonical_json(saved) + "\n", encoding="utf-8")
        os.replace(temporary, checkpoint)

    expected_ids = {record["pair_id"] for record in paper_records}
    if set(raw_ranked) != expected_ids or set(expanded) != expected_ids or set(audited) != expected_ids:
        raise RuntimeError(f"Incomplete semantic checkpoint for {paper_id}")
    RAW_SEMANTIC_HITS.update(raw_ranked)
    SEMANTIC_HITS.update(expanded)
    VECTOR_AUDIT.update(audited)
    CHUNK_STATS.append(chunk_stats)

expected_ids = set(pairs["pair_id"])
if set(VECTOR_AUDIT) != expected_ids:
    raise RuntimeError("Semantic audit does not cover every selected pair")
if any(result.get("paper_verified") for result in VECTOR_AUDIT.values()):
    raise RuntimeError("Local semantic retrieval must not mark a pair VERIFIED")
if any(result.get("release_to_sft") for result in VECTOR_AUDIT.values()):
    raise RuntimeError("Local semantic retrieval must not release strict SFT rows")
if any(len(result.get("bundle") or []) > MAX_SPANS for result in VECTOR_AUDIT.values()):
    raise RuntimeError("Semantic result exceeds the bundle-size contract")
if any(
    hit["span"].get("paper_id") != ROW[pair_id].paper_id
    for pair_id, hits in SEMANTIC_HITS.items() for hit in hits
):
    raise RuntimeError("Cross-paper semantic evidence detected")

sample_fingerprint = hashlib.sha256(canonical_json({
    "source_identity_sha256": SOURCE_MARKER["identity_sha256"],
    "retrieval_identity_sha256": retrieval_identity_sha,
    "paper_payloads": PAYLOAD_HASHES,
}).encode("utf-8")).hexdigest()
route_counts = Counter(result["route"] for result in VECTOR_AUDIT.values())
nonempty_bundles = sum(bool(result.get("bundle")) for result in VECTOR_AUDIT.values())
chunk_frame = pd.DataFrame(CHUNK_STATS)

print("checkpoint directory     :", CHECKPOINT_DIR)
print("resumed paper checkpoints:", f"{checkpoint_hits:,}/{paper_group_count:,}")
print("semantic route counts    :", dict(route_counts))
print("nonempty best bundles    :", f"{nonempty_bundles:,}/{len(VECTOR_AUDIT):,}")
print("semantic chunks          :", f"{int(chunk_frame.chunks.sum()):,}")
print("sample fingerprint       :", sample_fingerprint)
print(f"retrieval/audit seconds  : {time.perf_counter() - semantic_started:.1f}")


# ============= persist audit and derive a refined permissive SFT ============
REFINE_CONFIG = refiner.RefineConfig(
    training_root=TRAINING_ROOT,
    markdown_dir=MARKDOWN,
    preview=PAPER_LIMIT is not None,
)
REFINE_OUTCOME = refiner.refine_sft_mixed(
    REFINE_CONFIG,
    VECTOR_AUDIT,
    retrieval_config={
        **RETRIEVAL_CONFIG,
        "retrieval_identity_sha256": retrieval_identity_sha,
        "sample_fingerprint": sample_fingerprint,
        "paper_limit": PAPER_LIMIT,
    },
    source_run=SOURCE,
    publish=True,
)
RESULT_DIR = REFINE_OUTCOME.run_dir
if RESULT_DIR is None:
    raise RuntimeError("Semantic refinement was not persisted")
if PAPER_LIMIT is not None and REFINE_OUTCOME.latest_advanced:
    raise RuntimeError("A partial smoke run advanced the production semantic pointer")
if (
    PAPER_LIMIT is None
    and REFINE_OUTCOME.training_ready
    and not REFINE_OUTCOME.latest_advanced
):
    raise RuntimeError("A complete training-ready run did not advance the semantic pointer")

print("result directory          :", RESULT_DIR)
print("complete semantic coverage:", REFINE_OUTCOME.complete_coverage)
print("training ready            :", REFINE_OUTCOME.training_ready)
print("production pointer updated:", REFINE_OUTCOME.latest_advanced)
EFFECTIVE_SFT = refiner.resolve_effective_sft_mixed(TRAINING_ROOT)
print("effective downstream SFT   :", EFFECTIVE_SFT.path)
print("effective SFT source       :", EFFECTIVE_SFT.source)
for name, frame in REFINE_OUTCOME.frames.items():
    print(f"  {name:<24} {len(frame):>10,}")


# ============ bounded, deterministic inspection of every output table ======
from IPython.display import Markdown, display

TABLE_ROWS = min(10, max(5, int(SHOW)))
LONG_TEXT_COLUMNS = {
    "question": 180, "prompt": 240, "response": 220,
    "evidence_json": 240, "candidate_bundle_json": 240,
    "support_report_json": 200, "reason_detail": 180,
    "paper_context": 200, "evidence_before": 240,
    "evidence_after": 240, "assistant_target": 220,
}

def compact_text(value, limit=180):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True)
    else:
        try:
            if bool(pd.isna(value)):
                return ""
        except (TypeError, ValueError):
            pass
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."

def stable_pair_ids(values, limit=TABLE_ROWS):
    unique = {funnel.clean(value) for value in values if funnel.clean(value)}
    return sorted(
        unique,
        key=lambda pair_id: hashlib.sha256(
            f"{SEED}:{pair_id}".encode("utf-8")
        ).hexdigest(),
    )[:limit]

def compact_frame(frame):
    view = frame.copy()
    for column, limit in LONG_TEXT_COLUMNS.items():
        if column in view.columns:
            view[column] = view[column].map(lambda value: compact_text(value, limit))
    return view.reset_index(drop=True)

def show_table(title, frame, *, total_rows, path=None, note=""):
    location = f"  \nParquet: `{path}`" if path is not None else ""
    detail = f"  \n{note}" if note else ""
    display(Markdown(
        f"### {title}\n\nRows in full table: **{total_rows:,}**.{location}{detail}"
    ))
    if frame.empty:
        display(Markdown("_No rows in this bounded selection._"))
    else:
        with pd.option_context(
            "display.max_colwidth", 250, "display.max_columns", None,
            "display.width", 220,
        ):
            display(compact_frame(frame))

refined_sft = REFINE_OUTCOME.frames["sft_mixed"]
semantic_routes = REFINE_OUTCOME.frames["semantic_routes"]
semantic_candidates = REFINE_OUTCOME.frames["semantic_candidates"]
changed_routes = semantic_routes[
    semantic_routes["refinement_action"].str.startswith("REFINED_", na=False)
].copy()
focus_routes = changed_routes if not changed_routes.empty else semantic_routes
focus_ids = stable_pair_ids(focus_routes["pair_id"])
focus_order = {pair_id: rank for rank, pair_id in enumerate(focus_ids)}

artifact_inventory = pd.DataFrame([
    {"table": name, "rows": len(frame),
     "parquet": str(RESULT_DIR / f"{name}.parquet")}
    for name, frame in REFINE_OUTCOME.frames.items()
])
display(Markdown("## Semantic refinement outputs"))
display(artifact_inventory)
display(Markdown(
    f"**Effective production SFT path:** `{EFFECTIVE_SFT.path}`  \n"
    f"Resolved source: `{EFFECTIVE_SFT.source}`"
))
if not REFINE_OUTCOME.latest_advanced:
    display(Markdown(
        f"**Preview-only refined SFT:** `{RESULT_DIR / 'sft_mixed.parquet'}`  \n"
        "This smoke-run file is inspectable, but it is not the production pointer."
    ))


# ====================== refined sft_mixed table =============================
sft_stats = pd.DataFrame([
    ("rows", len(refined_sft)),
    ("papers", refined_sft["paper_id"].nunique()),
    ("verified rows", refined_sft["verification_tier"].eq("VERIFIED").sum()),
    ("unverified rows", refined_sft["verification_tier"].eq("UNVERIFIED").sum()),
    ("open-book rows", refined_sft["mode"].eq("OPEN").sum()),
    ("closed-book rows", refined_sft["mode"].eq("CLOSED").sum()),
    ("semantically refined rows", refined_sft["inclusion_source"].eq("SEMANTIC_REFINED").sum()),
], columns=["measure", "count"])
display(Markdown("## Refined `sft_mixed.parquet` -- statistics"))
display(sft_stats)
sft_sample = refined_sft[refined_sft["pair_id"].isin(focus_ids)].copy()
if not sft_sample.empty:
    sft_sample["_order"] = sft_sample["pair_id"].map(focus_order)
    sft_sample = sft_sample.sort_values("_order", kind="stable").drop(columns="_order")
sft_columns = [
    "example_id", "paper_id", "split", "pair_type", "mode",
    "verification_tier", "support_route", "inclusion_source",
    "question", "prompt", "response", "evidence_json",
]
show_table(
    "Refined `sft_mixed` sample", sft_sample.loc[:, sft_columns],
    total_rows=len(refined_sft), path=RESULT_DIR / "sft_mixed.parquet",
    note=("Deterministic sample of changed rows." if not changed_routes.empty
          else "No row changed; deterministic sample of audited rows."),
)


# ======================== semantic_routes table =============================
route_stats = (
    semantic_routes.groupby(
        ["semantic_route", "refinement_action"], dropna=False, sort=True,
    ).size().reset_index(name="rows")
)
display(Markdown("## `semantic_routes.parquet` -- route/action statistics"))
display(route_stats)
route_sample = semantic_routes[semantic_routes["pair_id"].isin(focus_ids)].copy()
if not route_sample.empty:
    route_sample["_order"] = route_sample["pair_id"].map(focus_order)
    route_sample = route_sample.sort_values("_order", kind="stable").drop(columns="_order")
route_columns = [
    "pair_id", "paper_id", "pair_type", "source_support_route",
    "semantic_route", "refined_support_route", "refinement_action",
    "candidate_bundle_json", "support_report_json",
]
show_table(
    "`semantic_routes` sample", route_sample.loc[:, route_columns],
    total_rows=len(semantic_routes), path=RESULT_DIR / "semantic_routes.parquet",
    note="One audit row per semantically processed pair.",
)


# ====================== semantic_candidates table ===========================
candidate_scores = pd.to_numeric(semantic_candidates["score"], errors="coerce")
candidate_stats = pd.DataFrame([
    ("candidate rows", len(semantic_candidates)),
    ("pairs with candidates", semantic_candidates["pair_id"].nunique()),
    ("selected bundle spans", semantic_candidates["selected_in_bundle"].sum()),
    ("mean cosine score", candidate_scores.mean()),
    ("maximum cosine score", candidate_scores.max()),
], columns=["measure", "value"])
display(Markdown("## `semantic_candidates.parquet` -- statistics"))
display(candidate_stats)
if semantic_candidates.empty:
    candidate_sample = semantic_candidates.copy()
else:
    candidate_sample = semantic_candidates.copy()
    candidate_sample["_pair_key"] = candidate_sample["pair_id"].map(
        lambda pair_id: hashlib.sha256(f"{SEED}:{pair_id}".encode()).hexdigest()
    )
    candidate_sample["_selected_order"] = (~candidate_sample["selected_in_bundle"]).astype(int)
    candidate_sample = candidate_sample.sort_values(
        ["_selected_order", "_pair_key", "rank"], kind="stable",
    ).head(TABLE_ROWS).drop(columns=["_pair_key", "_selected_order"])
candidate_columns = [
    "pair_id", "paper_id", "rank", "score", "candidate_origin",
    "selected_in_bundle", "evidence_token_count", "page", "section",
    "source_kind", "char_start", "char_end", "quote_sha256",
]
show_table(
    "`semantic_candidates` sample", candidate_sample.loc[:, candidate_columns],
    total_rows=len(semantic_candidates), path=RESULT_DIR / "semantic_candidates.parquet",
    note="Selected bundle spans are shown first; remaining order is deterministic.",
)


# ====================== before/after refined rows ===========================
changed_pair_set = set(changed_routes["pair_id"])
base_changed = sft_mixed_base[sft_mixed_base["pair_id"].isin(changed_pair_set)]
refined_changed = refined_sft[refined_sft["pair_id"].isin(changed_pair_set)]
if base_changed["pair_id"].duplicated().any() or refined_changed["pair_id"].duplicated().any():
    raise RuntimeError("A semantically refined pair has more than one mixed-SFT row")
base_by_pair = base_changed.set_index("pair_id", drop=False)
refined_by_pair = refined_changed.set_index("pair_id", drop=False)
changed_ids = stable_pair_ids(changed_routes["pair_id"])
action_by_pair = semantic_routes.set_index("pair_id")["refinement_action"].to_dict()

def evidence_preview(raw):
    try:
        spans = json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError):
        return compact_text(raw, 240)
    quotes = [funnel.clean(span.get("quote")) for span in spans if isinstance(span, dict)]
    return compact_text(" | ".join(quote for quote in quotes if quote), 240)

comparison_rows = []
for pair_id in changed_ids:
    before = base_by_pair.loc[pair_id]
    after = refined_by_pair.loc[pair_id]
    comparison_rows.append({
        "pair_id": pair_id,
        "action": action_by_pair[pair_id],
        "mode_before": before["mode"],
        "mode_after": after["mode"],
        "route_before": before["support_route"],
        "route_after": after["support_route"],
        "evidence_before": evidence_preview(before["evidence_json"]),
        "evidence_after": evidence_preview(after["evidence_json"]),
        "assistant_target": after["response"],
        "target_unchanged": before["response"] == after["response"],
    })
comparison = pd.DataFrame(comparison_rows, columns=[
    "pair_id", "action", "mode_before", "mode_after",
    "route_before", "route_after", "evidence_before", "evidence_after",
    "assistant_target", "target_unchanged",
])
comparison_stats = pd.DataFrame([
    ("rows actually refined", len(changed_routes)),
    ("shown below", len(comparison)),
    ("assistant targets unchanged in sample", int(comparison["target_unchanged"].sum())),
], columns=["measure", "count"])
display(Markdown("## Before/after refinement -- statistics"))
display(comparison_stats)
show_table(
    "Before/after rows actually refined", comparison,
    total_rows=len(changed_routes),
    note="Assistant targets must remain identical; only evidence/context routing may change.",
)


# ================== human-readable source evidence =========================
def show_candidate(pair_id):
    row = ROW[pair_id]
    before = BASE_ROUTES[pair_id]
    after = VECTOR_AUDIT[pair_id]
    print(f"   {pair_id} [{row.pair_type}] -> {after['route']}")
    wrap("question:", row.question)
    wrap("proposed target:", funnel.assistant_turn(row)[:650])
    print("   builder reason :", before["report"].get("reason", ""))
    print("   semantic reason:", after["report"].get("reason", ""))
    for rank, hit in enumerate((after.get("hits") or [])[:3], 1):
        span = hit["span"]
        score = hit.get("score")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        print(f"      {rank}. cosine={score_text}; origin={hit.get('candidate_origin')}; "
              f"page={span.get('page')}; kind={span.get('source_kind')}")
        wrap("passage:", funnel.clean(span.get("quote"))[:420])
    if after.get("bundle"):
        print("   best exact same-paper bundle:")
        for number, span in enumerate(after["bundle"], 1):
            wrap(f"bundle {number}:", funnel.clean(span.get("quote"))[:500])

display(Markdown("### Human-readable evidence excerpts (bounded)"))
detail_ids = focus_ids[:min(3, len(focus_ids))]
for number, pair_id in enumerate(detail_ids, 1):
    print(f"\n{'-' * 100}\n[{number}]")
    show_candidate(pair_id)


# ================================= final audit ================================
summary = pd.DataFrame([
    ("source builder papers in quarantine", support_quarantine.paper_id.nunique()),
    ("selected papers", len(SUBSET)),
    ("selected factual/reasoning pairs", len(pairs)),
    ("resumed paper checkpoints", checkpoint_hits),
    ("semantic chunks embedded/indexed", int(chunk_frame.chunks.sum())),
    ("deterministic-pass semantic bundles", sum(
        result["route"] == "VECTOR_CANDIDATE_DETERMINISTIC_PASS"
        for result in VECTOR_AUDIT.values()
    )),
    ("best-unverified nonempty bundles", sum(
        result["route"] != "VECTOR_CANDIDATE_DETERMINISTIC_PASS"
        and bool(result.get("bundle"))
        for result in VECTOR_AUDIT.values()
    )),
    ("no semantic bundle", sum(
        not result.get("bundle") for result in VECTOR_AUDIT.values()
    )),
    ("cross-paper violations", 0),
    ("strict rows modified", 0),
    ("production pointer updated", int(REFINE_OUTCOME.latest_advanced)),
], columns=["measure", "count"])
display(summary)

print("\nInterpretation")
print("- Inputs came from the latest immutable builder generation, not extraction reconstruction.")
print("- BGE is a same-paper passage locator; semantic scores never assert entailment.")
print("- Exact bundles that pass deterministic checks and best failing bundles remain UNVERIFIED.")
print("- VERIFIED strict rows are preserved byte-for-byte in the derived permissive SFT.")
print("- Per-paper Drive checkpoints contain bounded hits and routes, never embedding vectors.")
if REFINE_OUTCOME.latest_advanced:
    print("- Full coverage completed; the semantic-refined production pointer was advanced atomically.")
else:
    print("- This is a smoke run; artifacts were saved, but no production training pointer was changed.")

