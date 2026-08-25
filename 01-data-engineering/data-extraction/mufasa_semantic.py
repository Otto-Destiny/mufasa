"""Pinned BGE candidate retrieval for quarantined MUFASA training pairs.

This module is deliberately a retriever, not a semantic judge.  Encoder cosine
scores only order exact spans from the pair's own paper.  The existing
deterministic support checker is run over candidate bundles, but even a passing
bundle remains quarantined until a later semantic entailment stage can bind
population, time, intervention arm, and other scientific qualifiers.

The BGE loader is commit-pinned and can use either a Colab/Drive snapshot or the
Hugging Face cache.  The legacy MiniLM loader remains available for callers
that have provisioned its local snapshot.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from mufasa_dataset import assistant_turn, clean, support_report


MODEL_ID = "BAAI/bge-base-en-v1.5"
MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
PINNED_BGE_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
# Kept as the historical default: 220 is also safely below BGE's 512-token
# context.  BGE callers may opt into the larger safe cap explicitly.
PINNED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
PINNED_MODEL_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
DEFAULT_MAX_TOKENS = 220
DEFAULT_BGE_MAX_TOKENS = 480
DEFAULT_OVERLAP_TOKENS = 40
DEFAULT_BATCH_SIZE = 256
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_REQUIRED_SNAPSHOT_FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "1_Pooling/config.json",
)
_REFERENCES = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\|[ \t]*)?"
    r"(?:(?:\*\*|__)[ \t]*)?"
    r"(?:references|bibliography|works[ \t]+cited)[ \t]*:?"
    r"(?:[ \t]*(?:\*\*|__))?(?:[ \t]*\|)?[ \t]*$"
)
_PAGE_MARKER = re.compile(r"<!--\s*MUFASA_PDF_PAGE:\s*(\d+)\s*-->")
_HEADING = re.compile(r"(?m)^#{1,6}\s+(.+?)\s*$")
_BLOCK = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.S)
_TABLE_SEPARATOR = re.compile(r"^[\s|:\-]+$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}|\nAnswer:\s*")


class LocalEncoder(Protocol):
    """Small interface shared by SentenceTransformer and deterministic tests."""

    tokenizer: Any

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class PaperSemanticIndex:
    """One paper's exact chunks and normalized embeddings."""

    paper_id: str
    text: str
    body_end: int
    chunks: tuple[dict[str, Any], ...]
    embeddings: np.ndarray


def _validate_snapshot(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name != PINNED_REVISION:
        raise RuntimeError(
            f"MiniLM snapshot must be pinned to {PINNED_REVISION}; got {path.name!r}"
        )
    missing = [name for name in _REQUIRED_SNAPSHOT_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete offline MiniLM snapshot at {path}: missing {', '.join(missing)}"
        )
    digest = hashlib.sha256()
    try:
        with (path / "model.safetensors").open("rb") as model_file:
            for block in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RuntimeError(f"Could not verify MiniLM weights at {path}") from exc
    if digest.hexdigest() != PINNED_MODEL_SHA256:
        raise RuntimeError(
            "Pinned MiniLM model.safetensors failed SHA256 verification"
        )
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        modules = json.loads((path / "modules.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid MiniLM snapshot metadata at {path}") from exc
    architecture_ok = (
        config.get("model_type") == "bert"
        and config.get("hidden_size") == 384
        and config.get("num_hidden_layers") == 6
    )
    module_types = {item.get("type") for item in modules if isinstance(item, Mapping)}
    if not architecture_ok or not {
        "sentence_transformers.models.Transformer",
        "sentence_transformers.models.Pooling",
        "sentence_transformers.models.Normalize",
    }.issubset(module_types):
        raise RuntimeError(f"Snapshot at {path} is not all-MiniLM-L6-v2")
    return path


def _validate_bge_snapshot(path: Path) -> Path:
    """Fail closed if a configured BGE directory is not the pinned base model."""

    path = path.expanduser().resolve()
    if path.name != PINNED_BGE_REVISION:
        raise RuntimeError(
            f"BGE snapshot must be pinned to {PINNED_BGE_REVISION}; got {path.name!r}",
        )
    missing = [name for name in _REQUIRED_SNAPSHOT_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete pinned BGE snapshot at {path}: missing {', '.join(missing)}",
        )
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid BGE snapshot metadata at {path}") from exc
    if not (
        config.get("model_type") == "bert"
        and config.get("hidden_size") == 768
        and config.get("num_hidden_layers") == 12
    ):
        raise RuntimeError(f"Snapshot at {path} is not BAAI/bge-base-en-v1.5")
    return path


def resolve_minilm_snapshot(
    snapshot_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Resolve the pinned model locally and fail rather than access the network."""
    configured = snapshot_path or os.environ.get("MUFASA_MINILM_SNAPSHOT")
    if configured:
        return _validate_snapshot(Path(configured))

    from huggingface_hub import snapshot_download

    try:
        resolved = snapshot_download(
            repo_id=MINILM_MODEL_ID,
            revision=PINNED_REVISION,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=True,
        )
    except Exception as exc:
        raise FileNotFoundError(
            "The pinned all-MiniLM-L6-v2 snapshot is not available locally. "
            "Provision it before running; this resolver will not download it."
        ) from exc
    return _validate_snapshot(Path(resolved))


def load_local_minilm(
    snapshot_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    device: str | None = None,
) -> LocalEncoder:
    """Load the pinned SentenceTransformer with network and remote code disabled."""
    from sentence_transformers import SentenceTransformer

    snapshot = resolve_minilm_snapshot(snapshot_path, cache_dir)
    model = SentenceTransformer(
        str(snapshot),
        device=device,
        local_files_only=True,
        trust_remote_code=False,
    )
    model.max_seq_length = 256
    if not getattr(model.tokenizer, "is_fast", False):
        raise RuntimeError("Exact-source chunking requires a fast tokenizer")
    return model


def load_bge(
    snapshot_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    device: str | None = None,
    local_files_only: bool | None = None,
) -> LocalEncoder:
    """Load the pinned BGE encoder, using a local Colab/Drive snapshot when set.

    ``local_files_only`` defaults to true when ``snapshot_path`` is supplied and
    false otherwise, allowing first-run Colab provisioning while keeping Drive
    runs deterministic and offline.
    """
    from sentence_transformers import SentenceTransformer

    configured = snapshot_path or os.environ.get("MUFASA_BGE_SNAPSHOT")
    offline = bool(configured) if local_files_only is None else local_files_only
    source = configured
    if source is None:
        from huggingface_hub import snapshot_download

        source = snapshot_download(
            repo_id=MODEL_ID,
            revision=PINNED_BGE_REVISION,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=offline,
        )
    try:
        import torch
        dtype = torch.bfloat16 if device != "cpu" and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    except ImportError:
        dtype = None
    kwargs: dict[str, Any] = {
        "device": device,
        "local_files_only": offline,
        "trust_remote_code": False,
    }
    if dtype is not None:
        kwargs["model_kwargs"] = {"torch_dtype": dtype}
    source = _validate_bge_snapshot(Path(source))
    model = SentenceTransformer(str(source), **kwargs)
    model.max_seq_length = 512
    model.mufasa_query_prefix = True
    if model.get_sentence_embedding_dimension() != 768:
        raise RuntimeError("Pinned BGE encoder did not expose 768-dimensional embeddings")
    if not getattr(model.tokenizer, "is_fast", False):
        raise RuntimeError("Exact-source chunking requires a fast tokenizer")
    return model


def _latest(items: Sequence[tuple[int, Any]], position: int, default: Any) -> Any:
    value = default
    for offset, candidate in items:
        if offset > position:
            break
        value = candidate
    return value


def _token_offsets(tokenizer: Any, text: str) -> list[tuple[int, int]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise RuntimeError("Tokenizer must provide offset_mapping")
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list):
        offsets = offsets[0]
    output = [(int(start), int(end)) for start, end in offsets if int(end) > int(start)]
    return output


def _table_header(block: str) -> tuple[str, int]:
    cursor = 0
    for line in block.splitlines(keepends=True):
        raw = line.rstrip("\r\n")
        if "|" in raw and not _TABLE_SEPARATOR.fullmatch(raw):
            return raw, cursor + len(raw)
        cursor += len(line)
    return "", 0


def _inside_word(text: str, position: int) -> bool:
    """Whether a character boundary falls inside one alphanumeric word."""
    return (
        0 < position < len(text)
        and text[position - 1].isalnum()
        and text[position].isalnum()
    )


def _whole_word_start(
    text: str, offsets: Sequence[tuple[int, int]], token_start: int,
) -> int:
    """Advance a WordPiece overlap boundary to the next whole word."""
    original = token_start
    while token_start < len(offsets) and _inside_word(text, offsets[token_start][0]):
        token_start += 1
    return token_start if token_start < len(offsets) else original


def _bounded_end(
    text: str,
    offsets: Sequence[tuple[int, int]],
    token_start: int,
    hard_end: int,
    overlap_tokens: int,
) -> int:
    """Prefer a whole-word sentence/line end without crossing the token cap."""
    token_end = hard_end
    while (
        token_end > token_start + 1
        and token_end < len(offsets)
        and _inside_word(text, offsets[token_end - 1][1])
    ):
        token_end -= 1
    if token_end == len(offsets):
        return token_end

    minimum = token_start + max(overlap_tokens + 1, (hard_end - token_start) // 2)
    for candidate in range(token_end, minimum - 1, -1):
        boundary = offsets[candidate][0]
        before = text[:boundary].rstrip()
        gap = text[offsets[candidate - 1][1]:boundary]
        if "\n" in gap or before.endswith((".", "!", "?")):
            return candidate
    return token_end


def chunk_markdown_tokens(
    paper_id: str,
    text: str,
    tokenizer: Any,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> dict[str, Any]:
    """Create overlapping, tokenizer-bounded chunks with exact source offsets."""
    if max_tokens < 2 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("Require max_tokens >= 2 and 0 <= overlap_tokens < max_tokens")
    reference = _REFERENCES.search(text)
    body_end = reference.start() if reference else len(text)
    body = text[:body_end]
    pages = [(match.start(), int(match.group(1))) for match in _PAGE_MARKER.finditer(body)]
    headings = [(match.start(), clean(match.group(1))) for match in _HEADING.finditer(body)]
    chunks: list[dict[str, Any]] = []

    for block_match in _BLOCK.finditer(body):
        block = block_match.group(0)
        offsets = _token_offsets(tokenizer, block)
        if not offsets:
            continue
        is_table = sum("|" in line for line in block.splitlines()) >= 2
        header, header_end = _table_header(block) if is_table else ("", 0)
        header_tokens = len(_token_offsets(tokenizer, header)) if header else 0
        token_start = 0
        while token_start < len(offsets):
            token_start = _whole_word_start(block, offsets, token_start)
            begins_after_header = bool(header and offsets[token_start][0] >= header_end)
            prefix = header if begins_after_header else ""
            budget = max_tokens - header_tokens if prefix else max_tokens
            if budget <= overlap_tokens:
                prefix, budget = "", max_tokens
            hard_end = min(len(offsets), token_start + budget)
            token_end = _bounded_end(
                block, offsets, token_start, hard_end, overlap_tokens
            )
            local_start = offsets[token_start][0]
            local_end = (
                len(block) if token_end == len(offsets) else offsets[token_end][0]
            )
            start = block_match.start() + local_start
            end = block_match.start() + local_end
            quote = text[start:end]
            embedding_text = f"{prefix}\n{quote}" if prefix else quote
            chunks.append({
                "paper_id": paper_id,
                "quote": quote,
                "embedding_text": embedding_text,
                "token_count": token_end - token_start + (header_tokens if prefix else 0),
                "page": _latest(pages, start, None),
                "section": _latest(headings, start, ""),
                "source_kind": "TABLE" if is_table else "TEXT",
                "source_label": "same-paper semantic candidate",
                "char_start": start,
                "char_end": end,
            })
            if token_end == len(offsets):
                break
            token_start = _whole_word_start(
                block, offsets, max(token_start + 1, token_end - overlap_tokens)
            )

    return {"paper_id": paper_id, "text": text, "body_end": body_end, "chunks": chunks}


def _normalize(matrix: Any) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Encoder returned invalid embeddings")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def _encode(
    model: LocalEncoder,
    texts: Sequence[str],
    batch_size: int,
    is_query: bool = False,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    prepared = list(texts)
    if is_query and getattr(model, "mufasa_query_prefix", False):
        prepared = [BGE_QUERY_PREFIX + text for text in prepared]
    return _normalize(model.encode(
        prepared,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ))


def build_text_semantic_index(
    paper_id: str,
    text: str,
    model: LocalEncoder,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> PaperSemanticIndex:
    """Chunk and embed one in-memory paper exactly once."""
    source = chunk_markdown_tokens(
        paper_id, text, model.tokenizer, max_tokens, overlap_tokens
    )
    chunks = tuple(source["chunks"])
    embeddings = _encode(model, [chunk["embedding_text"] for chunk in chunks], batch_size)
    return PaperSemanticIndex(paper_id, text, source["body_end"], chunks, embeddings)


def build_paper_semantic_index(
    paper_id: str,
    markdown_dir: str | Path,
    model: LocalEncoder,
    **kwargs: Any,
) -> PaperSemanticIndex:
    """Read, chunk, and embed ``<markdown_dir>/<paper_id>.md`` locally."""
    path = Path(markdown_dir) / f"{paper_id}.md"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return build_text_semantic_index(paper_id, text, model, **kwargs)


def _sentences(text: str) -> list[str]:
    return [clean(part.removeprefix("Answer:")) for part in _SENTENCE_SPLIT.split(text) if clean(part)]


def _query_limit(model: LocalEncoder) -> int:
    model_limit = int(getattr(model, "max_seq_length", 256) or 256)
    special = 2
    special_counter = getattr(model.tokenizer, "num_special_tokens_to_add", None)
    if callable(special_counter):
        special = int(special_counter(pair=False))
    prefix = BGE_QUERY_PREFIX if getattr(model, "mufasa_query_prefix", False) else ""
    prefix_tokens = len(_token_offsets(model.tokenizer, prefix)) if prefix else 0
    return max(1, model_limit - special - prefix_tokens)


def query_variants(
    question: Any,
    target: Any,
    tokenizer: Any | None = None,
    max_query_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Return distinct bounded question, target, and target-sentence queries."""
    asked, answer = clean(question), clean(target)
    variants = [
        ("question", asked),
        ("full_target", answer),
        ("question_target", f"{asked}\n\n{answer}" if asked and answer else ""),
    ]
    variants.extend((f"target_sentence_{index}", sentence) for index, sentence in enumerate(_sentences(clean(target))))
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for kind, text in variants:
        key = re.sub(r"\s+", " ", text).casefold()
        if text and key not in seen:
            seen.add(key)
            offsets = _token_offsets(tokenizer, text) if tokenizer is not None else []
            limit = max_query_tokens or len(offsets) or 1
            if offsets and len(offsets) > limit:
                # Split before encoding; SentenceTransformer must never silently
                # truncate a retrieval query. A small overlap preserves context.
                overlap = min(20, max(0, limit // 5))
                start = 0
                part = 0
                while start < len(offsets):
                    end = min(len(offsets), start + limit)
                    fragment = text[offsets[start][0]:offsets[end - 1][1]]
                    output.append({
                        "kind": f"{kind}_part_{part}",
                        "text": fragment,
                        "token_count": end - start,
                        "max_query_tokens": limit,
                        "split_from_long_query": True,
                    })
                    if end == len(offsets):
                        break
                    start = max(start + 1, end - overlap)
                    part += 1
            else:
                output.append({
                    "kind": kind,
                    "text": text,
                    "token_count": len(offsets) if offsets else None,
                    "max_query_tokens": limit if offsets else None,
                    "split_from_long_query": False,
                })
    return output


def _query_truncation_count(variants: Sequence[Mapping[str, Any]]) -> int:
    """Count prepared queries that would exceed their actual tokenizer limit."""
    return sum(
        1 for item in variants
        if item.get("token_count") is not None
        and item.get("max_query_tokens") is not None
        and int(item["token_count"]) > int(item["max_query_tokens"])
    )


def _rank_with_embeddings(
    index: PaperSemanticIndex,
    variants: Sequence[Mapping[str, Any]],
    queries: np.ndarray,
    top_k_per_query: int = 5,
    candidate_limit: int = 12,
) -> list[dict[str, Any]]:
    truncation_count = _query_truncation_count(variants)
    if truncation_count:
        raise RuntimeError(
            f"Refusing to encode {truncation_count} over-limit retrieval queries"
        )
    similarities = queries @ index.embeddings.T
    chunk_ids = np.arange(len(index.chunks))
    selected: set[int] = set()
    for row in similarities:
        order = np.lexsort((chunk_ids, -row))
        selected.update(int(value) for value in order[:max(1, top_k_per_query)])
    maxima = similarities.max(axis=0)
    ordered = sorted(selected, key=lambda value: (-float(maxima[value]), value))
    hits = []
    for chunk_id in ordered[:candidate_limit]:
        hits.append({
            "chunk_index": chunk_id,
            "score": float(maxima[chunk_id]),
            "query_scores": {
                variants[pos]["kind"]: float(similarities[pos, chunk_id])
                for pos in range(len(variants))
            },
            "query_truncation_count": truncation_count,
            "long_query_split_count": sum(
                bool(item["split_from_long_query"]) for item in variants
            ),
            "table_header_embedding_only": bool(
                index.chunks[chunk_id]["embedding_text"]
                != index.chunks[chunk_id]["quote"]
            ),
            "span": _public_span(index.chunks[chunk_id]),
        })
    return hits


def rank_semantic_chunks(
    index: PaperSemanticIndex,
    question: Any,
    target: Any,
    model: LocalEncoder,
    top_k_per_query: int = 5,
    candidate_limit: int = 12,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Rank same-paper chunks by normalized NumPy cosine with stable ties."""
    if not index.chunks or candidate_limit <= 0:
        return []
    variants = query_variants(
        question, target, model.tokenizer, _query_limit(model)
    )
    if not variants:
        return []
    queries = _encode(model, [item["text"] for item in variants], batch_size, is_query=True)
    return _rank_with_embeddings(
        index, variants, queries, top_k_per_query, candidate_limit
    )


def rank_semantic_records(
    index: PaperSemanticIndex,
    records: Sequence[Mapping[str, Any]],
    model: LocalEncoder,
    top_k_per_query: int = 5,
    candidate_limit: int = 12,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, list[dict[str, Any]]]:
    """Batch-rank many records against exactly one paper index."""
    prepared: list[tuple[str, list[dict[str, Any]]]] = []
    flattened: list[str] = []
    seen: set[str] = set()
    for record in records:
        pair_id = clean(record.get("pair_id"))
        paper_id = clean(record.get("paper_id"))
        if paper_id != index.paper_id:
            raise ValueError(
                f"Same-paper isolation violation: pair {paper_id!r}, index {index.paper_id!r}"
            )
        if not pair_id or pair_id in seen:
            raise ValueError(f"Pair IDs must be nonblank and unique within a paper: {pair_id!r}")
        seen.add(pair_id)
        row = SimpleNamespace(**dict(record))
        variants = query_variants(
            record.get("question"), assistant_turn(row), model.tokenizer,
            _query_limit(model),
        )
        prepared.append((pair_id, variants))
        flattened.extend(item["text"] for item in variants)

    all_queries = _encode(model, flattened, batch_size, is_query=True)
    output: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for pair_id, variants in prepared:
        count = len(variants)
        output[pair_id] = _rank_with_embeddings(
            index,
            variants,
            all_queries[offset:offset + count],
            top_k_per_query,
            candidate_limit,
        ) if count and index.chunks else []
        offset += count
    return output


def _public_span(chunk: Mapping[str, Any]) -> dict[str, Any]:
    return {key: chunk.get(key) for key in (
        "paper_id", "quote", "page", "section", "source_kind", "source_label",
        "char_start", "char_end",
    )}


def _require_exact_span(
    index: PaperSemanticIndex, span: Mapping[str, Any], label: str,
) -> dict[str, Any]:
    if clean(span.get("paper_id")) != index.paper_id:
        raise ValueError(f"{label} contains a cross-paper span")
    start, end = span.get("char_start"), span.get("char_end")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not (0 <= start < end <= index.body_end)
        or clean(span.get("quote")) != index.text[start:end].strip()
    ):
        raise ValueError(f"{label} is not an exact in-body source span")
    exact = dict(span)
    exact["quote"] = index.text[start:end]
    return exact


def expand_ranked_hits(
    index: PaperSemanticIndex,
    hits: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    supplied_spans: Sequence[Mapping[str, Any]] = (),
    neighbor_radius: int = 1,
    max_pool_tokens: int = 1100,
) -> list[dict[str, Any]]:
    """Union exact supplied/neighbor spans under a bounded candidate-pool budget.

    Neighbours remain retrieval candidates only. Table headers are useful only
    when they occur in an exact returned span; no header text is synthesized.
    """
    if neighbor_radius < 0 or max_pool_tokens < 1:
        raise ValueError("Require neighbor_radius >= 0 and max_pool_tokens >= 1")
    output: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    used_tokens = 0

    def add(candidate: Mapping[str, Any], origin: str) -> None:
        nonlocal used_tokens
        span = _require_exact_span(index, candidate["span"], origin)
        key = (int(span["char_start"]), int(span["char_end"]))
        if key in seen:
            return
        cost = len(_token_offsets(tokenizer, span["quote"]))
        if not cost or used_tokens + cost > max_pool_tokens:
            return
        item = dict(candidate)
        item["span"] = span
        item["candidate_origin"] = origin
        item["evidence_token_count"] = cost
        output.append(item)
        seen.add(key)
        used_tokens += cost

    for span in supplied_spans:
        add({
            "span": span,
            "score": None,
            "query_scores": {},
            "query_truncation_count": 0,
            "long_query_split_count": 0,
            "table_header_embedding_only": False,
        }, "supplied_exact")

    for hit in hits:
        add(hit, "semantic_rank")
        chunk_index = hit.get("chunk_index")
        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool):
            continue
        for distance in range(1, neighbor_radius + 1):
            for candidate_index in (chunk_index - distance, chunk_index + distance):
                if not 0 <= candidate_index < len(index.chunks):
                    continue
                chunk = index.chunks[candidate_index]
                neighbor = {
                    **dict(hit),
                    "chunk_index": candidate_index,
                    "span": _public_span(chunk),
                    "table_header_embedding_only": bool(
                        chunk["embedding_text"] != chunk["quote"]
                    ),
                    "neighbor_of": chunk_index,
                }
                add(neighbor, "immediate_neighbor")
    return output


def _get(row: Any, key: str, default: Any = "") -> Any:
    return row.get(key, default) if isinstance(row, Mapping) else getattr(row, key, default)


def evaluate_ranked_candidates(
    row: Any,
    paper_id: str,
    hits: Sequence[Mapping[str, Any]],
    max_spans: int = 3,
    bundle_beam: int = 5,
) -> dict[str, Any]:
    """Audit ranked same-paper hits without loading or calling an encoder."""
    row_paper_id = clean(_get(row, "paper_id"))
    if row_paper_id != paper_id:
        raise ValueError(
            f"Same-paper isolation violation: pair {row_paper_id!r}, index {paper_id!r}"
        )
    if any(clean(hit.get("span", {}).get("paper_id")) != paper_id for hit in hits):
        raise ValueError("Ranked candidates contain a cross-paper span")
    question = clean(_get(row, "question"))
    target = assistant_turn(SimpleNamespace(**dict(row))) if isinstance(row, Mapping) else assistant_turn(row)
    query_truncation_count = max(
        (int(hit.get("query_truncation_count", 0)) for hit in hits), default=0
    )
    long_query_split_count = hits[0]["long_query_split_count"] if hits else 0
    table_header_embedding_only = any(
        bool(hit.get("table_header_embedding_only")) for hit in hits
    )
    candidates = [hit["span"] for hit in hits[:bundle_beam]]
    empty_report = support_report(target, [], question)
    best_bundle: list[dict[str, Any]] = []
    best_report = empty_report

    def quality(report: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            -len(report["missing_numbers"]),
            -len(report["unit_mismatches"]),
            -len(report["unbound_numbers"]),
            -len(report["unbound_question_anchors"]),
            -len(report["direction_conflicts"]),
            sum(item["coverage"] for item in report["sentence_coverages"]),
            report["relevance"],
        )

    for size in range(1, min(max_spans, len(candidates)) + 1):
        for choice in itertools.combinations(candidates, size):
            report = support_report(target, choice, question)
            if report["supported"]:
                return {
                    "route": "VECTOR_CANDIDATE_DETERMINISTIC_PASS",
                    "bundle": list(choice),
                    "report": report,
                    "hits": hits,
                    "paper_verified": False,
                    "release_to_sft": False,
                    "query_truncation_count": query_truncation_count,
                    "long_query_split_count": long_query_split_count,
                    "retrieval_limitations": (
                        ["table header influenced embedding rank but was not added to evidence"]
                        if table_header_embedding_only else []
                    ),
                }
            if not best_bundle or quality(report) > quality(best_report):
                best_bundle, best_report = list(choice), report
    return {
        "route": "STILL_QUARANTINED",
        "bundle": best_bundle,
        "report": best_report,
        "hits": hits,
        "paper_verified": False,
        "release_to_sft": False,
        "query_truncation_count": query_truncation_count,
        "long_query_split_count": long_query_split_count,
        "retrieval_limitations": (
            ["table header influenced embedding rank but was not added to evidence"]
            if table_header_embedding_only else []
        ),
    }


def evaluate_ranked_paper(
    records: Sequence[Mapping[str, Any]],
    paper_id: str,
    ranked: Mapping[str, Sequence[Mapping[str, Any]]],
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Process-safe audit of all ranked candidates from one paper."""
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        pair_id = clean(record.get("pair_id"))
        if pair_id not in ranked:
            raise KeyError(f"Missing semantic ranks for {pair_id}")
        output[pair_id] = evaluate_ranked_candidates(
            record, paper_id, ranked[pair_id], **kwargs,
        )
    return output


def route_quarantined_pair(
    row: Any,
    index: PaperSemanticIndex,
    model: LocalEncoder,
    max_spans: int = 3,
    bundle_beam: int = 5,
    precomputed_hits: Sequence[Mapping[str, Any]] | None = None,
    supplied_spans: Sequence[Mapping[str, Any]] = (),
    neighbor_radius: int = 0,
    max_pool_tokens: int = 1100,
    **rank_kwargs: Any,
) -> dict[str, Any]:
    """Retrieve candidates, then deterministically check—but never verify—them."""
    paper_id = clean(_get(row, "paper_id"))
    if paper_id != index.paper_id:
        raise ValueError(
            f"Same-paper isolation violation: pair {paper_id!r}, index {index.paper_id!r}"
        )
    hits = list(precomputed_hits) if precomputed_hits is not None else rank_semantic_chunks(
        index, clean(_get(row, "question")),
        assistant_turn(SimpleNamespace(**dict(row))) if isinstance(row, Mapping) else assistant_turn(row),
        model, **rank_kwargs,
    )
    if supplied_spans or neighbor_radius:
        hits = expand_ranked_hits(
            index,
            hits,
            model.tokenizer,
            supplied_spans=supplied_spans,
            neighbor_radius=neighbor_radius,
            max_pool_tokens=max_pool_tokens,
        )
    return evaluate_ranked_candidates(
        row, index.paper_id, hits, max_spans=max_spans, bundle_beam=bundle_beam,
    )


def route_quarantined_paper(
    records: Sequence[Mapping[str, Any]],
    markdown_dir: str | Path,
    model: LocalEncoder,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Embed one paper once and route all of its quarantined records."""
    if not records:
        return {}
    paper_ids = {clean(record.get("paper_id")) for record in records}
    if len(paper_ids) != 1:
        raise ValueError("route_quarantined_paper accepts records from exactly one paper")
    paper_id = next(iter(paper_ids))
    index_keys = {"max_tokens", "overlap_tokens", "batch_size"}
    index_kwargs = {key: kwargs.pop(key) for key in tuple(kwargs) if key in index_keys}
    index = build_paper_semantic_index(paper_id, markdown_dir, model, **index_kwargs)
    top_k = int(kwargs.pop("top_k_per_query", 5))
    candidate_limit = int(kwargs.pop("candidate_limit", 12))
    query_batch_size = int(index_kwargs.get("batch_size", DEFAULT_BATCH_SIZE))
    ranked = rank_semantic_records(
        index, records, model, top_k, candidate_limit, query_batch_size,
    )
    output: dict[str, dict[str, Any]] = {}
    route_kwargs = {
        key: value for key, value in kwargs.items()
        if key != "batch_size"
    }
    for record in records:
        pair_id = clean(record.get("pair_id"))
        output[pair_id] = route_quarantined_pair(
            record, index, model, precomputed_hits=ranked[pair_id], **route_kwargs
        )
    return output
