"""Production assembly for MUFASA SFT and DPO training data.

The builder deliberately delegates scientific support decisions to
``mufasa_dataset`` -- the same router exercised by ``sample-training-set``.
It adds production orchestration: frozen split inheritance, deterministic
curriculum assembly, leakage suppression, Parquet contracts, and atomic
publication.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import mufasa_citations as citations
import mufasa_dataset as funnel


BUILDER_VERSION = "mufasa-training-builder-v1.0-candidate.5"
OUTPUT_FORMAT = "mufasa-training-parquet-generation-v3"
TOKEN_ESTIMATOR = "unicode-word-punctuation-v1"
VALID_SPLITS = frozenset({"train", "evaluate", "test"})
SUPPORTED_ROUTES = frozenset({"OPEN_AS_IS", "OPEN_WIDENED"})
PROVENANCE_MODES = frozenset({"OFF", "AUDIT_ONLY", "TRAINING"})
CITATION_TRACE_COLUMNS = (
    "citation_label", "citation_raw_label", "citation_status",
    "citation_metadata_source", "citation_metadata_json",
)

SFT_COLUMNS = (
    "example_id", "pair_id", "paper_id", "family_id", "split", "pair_type",
    "mode", "assignment", "support_route", "question", "question_key",
    "prompt", "response", "messages", "messages_json", "descriptor", "evidence_json",
    "support_report_json", "pair_kind", "tags_json", "extraction_model",
    "paper_context", "paper_context_json", "verification_tier", "inclusion_source",
    "reason_code", "reason_detail", "source_row_json", *CITATION_TRACE_COLUMNS,
    "token_estimate",
)
DPO_COLUMNS = (
    "pair_id", "paper_id", "family_id", "split", "pair_type",
    "support_route", "question", "question_key", "prompt", "chosen",
    "rejected", "rejection_reason", "evidence_json", "support_report_json",
    "pair_kind", "tags_json", "extraction_model", "paper_context", "paper_context_json",
    "verification_tier", "inclusion_source", "reason_code", "reason_detail",
    "source_row_json", *CITATION_TRACE_COLUMNS, "token_estimate",
)
RERANKER_COLUMNS = (
    "pair_id", "paper_id", "family_id", "split", "pair_type", "question",
    "question_key", "positive_quote", "hard_negative_quote", "negative_reason",
    "paper_context", "paper_context_json", "verification_tier", "inclusion_source",
    "reason_code", "reason_detail", "source_row_json", "extraction_model",
    "token_estimate",
)
QUARANTINE_COLUMNS = (
    "record_id", "pair_id", "paper_id", "family_id", "split", "pair_type",
    "stage", "reason_code", "reason_detail", "support_route", "question",
    "question_key", "target", "evidence_json", "support_report_json",
    "source_row_json", "extraction_model",
)
DISCARDED_COLUMNS = (
    "record_id", "pair_id", "paper_id", "family_id", "split", "pair_type",
    "stage", "reason_code", "reason_detail", "question", "target",
    "source_row_json", "extraction_model",
)

MESSAGE_TYPE = pa.list_(pa.struct([
    pa.field("role", pa.string(), nullable=False),
    pa.field("content", pa.string(), nullable=False),
]))


def _string_schema(columns: Sequence[str], *, messages: bool = False) -> pa.Schema:
    fields = []
    for column in columns:
        if column == "token_estimate":
            fields.append(pa.field(column, pa.int64(), nullable=False))
        elif column == "messages" and messages:
            fields.append(pa.field(column, MESSAGE_TYPE, nullable=False))
        else:
            fields.append(pa.field(column, pa.string(), nullable=False))
    return pa.schema(fields)


OUTPUT_SCHEMAS = {
    "sft_examples": _string_schema(SFT_COLUMNS, messages=True),
    "sft_mixed": _string_schema(SFT_COLUMNS, messages=True),
    "dpo_pairs": _string_schema(DPO_COLUMNS),
    "preference_mixed": _string_schema(DPO_COLUMNS),
    "reranker_mixed": _string_schema(RERANKER_COLUMNS),
    "quarantine": _string_schema(QUARANTINE_COLUMNS),
    "discarded": _string_schema(DISCARDED_COLUMNS),
}


@dataclass(frozen=True)
class BuildConfig:
    extraction_root: Path
    markdown_dir: Path
    split_manifest: Path
    output_root: Path
    seed: int = 7
    router_workers: int = 1
    max_evidence_spans: int = 3
    max_evidence_chars: int = 8_000
    open_share: float = 0.45
    closed_share: float = 0.45
    dual_share: float = 0.10
    paper_limit: int | None = None
    progress_every: int = 100
    citation_metadata_path: Path | None = None
    provenance_mode: str = "OFF"

    def validate(self) -> None:
        if self.router_workers < 1:
            raise ValueError("router_workers must be at least 1")
        if self.max_evidence_spans < 1 or self.max_evidence_chars < 1:
            raise ValueError("evidence bounds must be positive")
        if self.paper_limit is not None and self.paper_limit < 1:
            raise ValueError("paper_limit must be positive or None")
        if self.provenance_mode.upper() not in PROVENANCE_MODES:
            raise ValueError(
                f"provenance_mode must be one of {sorted(PROVENANCE_MODES)}",
            )
        if self.provenance_mode.upper() != "OFF" and self.citation_metadata_path is None:
            raise ValueError(
                "citation_metadata_path is required when provenance rendering is enabled",
            )
        shares = (self.open_share, self.closed_share, self.dual_share)
        if any(value < 0 for value in shares) or not math.isclose(sum(shares), 1.0):
            raise ValueError("OPEN/CLOSED/DUAL shares must be nonnegative and sum to 1")


@dataclass
class BuildOutcome:
    run_id: str
    run_dir: Path | None
    frames: dict[str, pd.DataFrame]
    manifest: dict[str, Any]
    stages: list[dict[str, Any]]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        converted = value.tolist()
        if converted is not value:
            return _jsonable(converted)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aggregate_hash(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _input_fingerprints(
    config: BuildConfig, table_dir: Path, selected: Sequence[str],
) -> dict[str, str]:
    markdown_paths = [config.markdown_dir / f"{paper}.md" for paper in selected]
    raw_root = config.extraction_root / "raw"
    raw_paths = [raw_root / f"{paper}.json" for paper in selected]
    raw_paths = [path for path in raw_paths if path.is_file()]
    return {
        "split_manifest": _file_hash(config.split_manifest),
        "router_module": _file_hash(Path(funnel.__file__)),
        "builder_module": _file_hash(Path(__file__)),
        "citation_module": _file_hash(Path(citations.__file__)),
        **{
            f"table:{name}": _file_hash(table_dir / f"{name}.parquet")
            for name in funnel.TABLES
        },
        "markdown_selected": _aggregate_hash(markdown_paths, config.markdown_dir),
        "raw_selected": _aggregate_hash(raw_paths, raw_root),
        **(
            {"citation_metadata": _file_hash(config.citation_metadata_path)}
            if config.provenance_mode.upper() != "OFF"
            and config.citation_metadata_path is not None
            else {}
        ),
    }


def _safe_child(root: Path, child: Path) -> Path:
    root = root.resolve()
    child = child.resolve()
    if child.parent != root:
        raise ValueError(f"path escapes expected root: {child}")
    return child


def resolve_extraction_tables(root: Path) -> tuple[Path, dict[str, Any]]:
    """Resolve an immutable extraction generation, with a legacy-flat bridge."""

    root = root.resolve()
    required = {f"{name}.parquet" for name in funnel.TABLES}
    pointer_path = root / "current-generation.json"
    if not pointer_path.is_file():
        missing = sorted(name for name in required if not (root / name).is_file())
        if missing:
            raise FileNotFoundError(f"extraction output is missing {missing}")
        return root, {"source_format": "legacy-flat", "generation_id": "legacy-flat"}

    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    directory = pointer.get("directory")
    if not isinstance(directory, str) or not directory:
        raise ValueError("current-generation.json has no valid directory")
    generations = (root / "published-generations").resolve()
    generation = (root / directory).resolve()
    if generation.parent != generations:
        raise ValueError("extraction generation escapes published-generations")
    immutable = json.loads((generation / "generation.json").read_text(encoding="utf-8"))
    if immutable != pointer:
        raise ValueError("extraction pointer disagrees with immutable generation")
    table_meta = pointer.get("tables")
    if not isinstance(table_meta, Mapping) or not required.issubset(table_meta):
        raise ValueError("extraction generation lacks required training tables")
    for filename in sorted(required):
        path = generation / filename
        metadata = table_meta[filename]
        if not path.is_file() or _file_hash(path) != metadata.get("sha256"):
            raise ValueError(f"extraction table is missing or corrupt: {filename}")
    return generation, {
        "source_format": pointer.get("format", "published-generation"),
        "generation_id": pointer.get("generation_id", ""),
        "settings_hash": pointer.get("settings_hash", ""),
        "source_fingerprint": pointer.get("source_fingerprint", ""),
    }


def load_frozen_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_parquet(path)
    required = {"paper_id", "family_id", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"split manifest is missing {sorted(missing)}")
    manifest = manifest.copy()
    for column in required:
        manifest[column] = manifest[column].map(funnel.clean)
        if manifest[column].eq("").any():
            raise ValueError(f"split manifest contains blank {column}")
    if manifest.paper_id.duplicated().any():
        raise ValueError("split manifest paper_id is not unique")
    invalid = sorted(set(manifest.split) - VALID_SPLITS)
    if invalid:
        raise ValueError(f"split manifest has invalid partitions {invalid}")
    straddling = manifest.groupby("family_id").split.nunique()
    if (straddling > 1).any():
        raise ValueError("split manifest has study families crossing partitions")
    return manifest


def estimate_tokens(*texts: Any) -> int:
    pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    return max(1, sum(len(pattern.findall(funnel.clean(text))) for text in texts) + 8)


def _row_dict(record: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in record.items()}


def _metadata(
    record: Mapping[str, Any], split_rows: Mapping[str, Mapping[str, str]],
    models: Mapping[str, str],
) -> dict[str, str]:
    paper_id = funnel.clean(record.get("paper_id"))
    assignment = split_rows[paper_id]
    return {
        "paper_id": paper_id,
        "family_id": assignment["family_id"],
        "split": assignment["split"],
        "pair_id": funnel.clean(record.get("pair_id")),
        "pair_type": funnel.clean(record.get("pair_type")).upper(),
        "question": funnel.clean(record.get("question")),
        "question_key": funnel.skeleton(record.get("question")),
        "extraction_model": models.get(paper_id, ""),
    }


def _citation_trace_fields(value: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Return stable citation trace columns without making a support decision."""

    row = dict(value or {})
    raw_label = funnel.clean(
        row.get("citation_raw_label") or row.get("raw_citation_label")
        or row.get("openalex_label")
        or row.get("citation_label"),
    )
    label = funnel.normalize_citation_label(row.get("citation_label") or raw_label)
    status = funnel.clean(row.get("citation_status") or row.get("status"))
    source = funnel.clean(
        row.get("citation_metadata_source") or row.get("metadata_source")
        or row.get("source"),
    )
    payload = row.get("citation_metadata_json")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = {"raw": payload}
    if not isinstance(payload, Mapping):
        payload = {
            key: item for key, item in row.items()
            if key not in CITATION_TRACE_COLUMNS
        }
    return {
        "citation_label": label,
        "citation_raw_label": raw_label,
        "citation_status": status,
        "citation_metadata_source": source,
        "citation_metadata_json": canonical_json(payload),
    }


def _quarantine_row(
    record: Mapping[str, Any], split_rows: Mapping[str, Mapping[str, str]],
    models: Mapping[str, str], *, stage: str, reason_code: str,
    reason_detail: str = "", support_route: str = "", evidence: Any = (),
    report: Any = None, record_suffix: str = "",
) -> dict[str, Any]:
    meta = _metadata(record, split_rows, models)
    pair_id = meta["pair_id"]
    return {
        "record_id": f"{pair_id}:{stage}{record_suffix}",
        **meta,
        "stage": stage,
        "reason_code": reason_code,
        "reason_detail": reason_detail,
        "support_route": support_route,
        "target": funnel.assistant_turn(SimpleNamespace(**dict(record))),
        "evidence_json": canonical_json(evidence),
        "support_report_json": canonical_json(report or {}),
        "source_row_json": canonical_json(_row_dict(record)),
    }


def _discarded_row(
    record: Mapping[str, Any], split_rows: Mapping[str, Mapping[str, str]],
    models: Mapping[str, str], *, stage: str, reason_code: str,
    reason_detail: str = "", record_suffix: str = "",
) -> dict[str, Any]:
    meta = _metadata(record, split_rows, models)
    pair_id = meta["pair_id"]
    return {
        "record_id": f"{pair_id}:{stage}{record_suffix}",
        **meta,
        "stage": stage,
        "reason_code": reason_code,
        "reason_detail": reason_detail,
        "target": funnel.assistant_turn(SimpleNamespace(**dict(record))),
        "source_row_json": canonical_json(_row_dict(record)),
    }


def _route_task(
    payload: tuple[Any, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    records, initial, markdown, max_spans, max_chars, cache_path, payload_hash = payload
    cache_path = Path(cache_path)
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("payload_sha256") == payload_hash and isinstance(cached.get("routes"), dict):
                return records, cached["routes"], True
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    routes = funnel.route_paper_records(records, initial, markdown, max_spans, max_chars)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        canonical_json({"payload_sha256": payload_hash, "routes": routes}) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, cache_path)
    return records, routes, False


def _task_stream(
    frame: pd.DataFrame, table_evidence: Mapping[str, Sequence[Mapping[str, Any]]],
    recovered: Mapping[str, Any], markdown_dir: Path, max_spans: int, max_chars: int,
    cache_dir: Path, router_identity: str,
) -> Iterable[tuple[Any, ...]]:
    for paper_id, group in frame.groupby("paper_id", sort=True):
        records = group.to_dict("records")
        initial = {
            record["pair_id"]: funnel.combined_evidence(
                record["pair_id"], table_evidence, recovered,
            )
            for record in records
        }
        markdown_path = markdown_dir / f"{paper_id}.md"
        payload_identity = {
            "router_identity": router_identity,
            "paper_id": paper_id,
            "records": records,
            "initial": initial,
            "markdown_sha256": _file_hash(markdown_path),
        }
        payload_hash = hashlib.sha256(canonical_json(payload_identity).encode()).hexdigest()
        cache_name = hashlib.sha256(funnel.clean(paper_id).encode()).hexdigest()[:24] + ".json"
        yield (
            records, initial, str(markdown_dir), max_spans, max_chars,
            str(cache_dir / cache_name), payload_hash,
        )


def _bounded_parallel_routes(
    executor: ProcessPoolExecutor, tasks: Iterable[tuple[Any, ...]], limit: int,
) -> Iterable[tuple[list[dict[str, Any]], dict[str, Any], bool]]:
    """Submit a small bounded paper queue so large payloads cannot pile up."""

    stream = iter(tasks)
    pending = {}
    for _ in range(limit):
        try:
            task = next(stream)
        except StopIteration:
            break
        pending[executor.submit(_route_task, task)] = None
    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            del pending[future]
            yield future.result()
            try:
                task = next(stream)
            except StopIteration:
                continue
            pending[executor.submit(_route_task, task)] = None


def _messages(rendered: Mapping[str, Any]) -> tuple[str, str, str]:
    messages = rendered["messages"]
    if len(messages) != 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        raise ValueError("rendered SFT example does not contain one user/assistant exchange")
    prompt = funnel.clean(messages[0]["content"])
    response = funnel.clean(messages[1]["content"])
    return prompt, response, canonical_json(messages)


def _pair_context(
    record: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]],
    contexts: Mapping[str, Sequence[Mapping[str, Any]]],
    profiles: Mapping[str, Mapping[str, Any]],
    innovation: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    text, payload = funnel.study_context_for_pair(
        funnel.clean(record.get("paper_id")), record.get("question"), evidence,
        contexts, profiles, innovation,
    )
    return text, canonical_json(payload)


def _mixed_closed_fallback(record: Mapping[str, Any], context_text: str) -> str:
    """Keep a valid mixed-SFT target without copying its answer into context."""

    row = SimpleNamespace(**dict(record))
    header = "Study context (minimal identifier; support remains unverified):"
    kept = [header]
    for line in context_text.splitlines()[1:]:
        line = funnel.clean(line)
        if not line:
            continue
        candidate = "\n".join([*kept, line])
        if not funnel.descriptor_reveals_target(row, candidate):
            kept.append(line)
    paper_id = funnel.clean(record.get("paper_id"))
    identifier = f"Paper ID: {paper_id}"
    if identifier not in kept:
        kept.append(identifier)
    return "\n".join(kept)


def _sft_candidate_from_route(
    record: Mapping[str, Any], route: Mapping[str, Any], context_text: str,
    context_json: str, split_rows: Mapping[str, Mapping[str, str]],
    models: Mapping[str, str], *, citation_trace: Mapping[str, Any] | None = None,
    provenance_enabled: bool = False,
) -> dict[str, Any]:
    row = SimpleNamespace(**dict(record))
    meta = _metadata(record, split_rows, models)
    bundle = list(route.get("bundle") or [])
    citation = _citation_trace_fields(citation_trace)
    verified = bool(route.get("paper_verified"))
    verification_tier = "VERIFIED" if verified else "UNVERIFIED"
    render_options = {
        "citation_label": citation["citation_label"],
        "paper_context": context_json,
        "verification_tier": verification_tier if provenance_enabled else None,
    }
    open_prompt = open_response = open_messages = ""
    open_tokens = 0
    if bundle:
        open_prompt, open_response, open_messages = _messages(
            funnel.render_open(row, bundle, context_text, **render_options),
        )
        open_tokens = estimate_tokens(open_prompt, open_response)
    if verified:
        closed_ready, closed_reason = funnel.verified_closed_ready(
            row, context_text, True,
        )
    else:
        closed_ready, closed_reason = funnel.permissive_closed_ready(row, context_text)
        if not bundle and not closed_ready:
            context_text = _mixed_closed_fallback(record, context_text)
            closed_ready = True
            closed_reason = "minimal non-leaking mixed-SFT fallback"
    closed_prompt = closed_response = closed_messages = ""
    closed_tokens = 0
    if closed_ready:
        closed_prompt, closed_response, closed_messages = _messages(
            funnel.render_closed(row, context_text, **render_options),
        )
        closed_tokens = estimate_tokens(closed_prompt, closed_response)
    reason = funnel.clean((route.get("report") or {}).get("reason"))
    return {
        **meta,
        "pair_kind": funnel.clean(record.get("pair_kind")),
        "tags_json": funnel.clean(record.get("tags_json")) or "[]",
        "support_route": funnel.clean(route.get("route")),
        "evidence_json": canonical_json(bundle),
        "support_report_json": canonical_json(route.get("report") or {}),
        "paper_context": context_text,
        "paper_context_json": context_json,
        "verification_tier": "VERIFIED" if verified else "UNVERIFIED",
        "inclusion_source": "STRICT_SFT" if verified else "SUPPORT_QUARANTINE",
        "reason_code": "" if verified else "QUARANTINE_UNVERIFIED",
        "reason_detail": "" if verified else reason,
        "source_row_json": canonical_json(_row_dict(record)),
        **citation,
        "paper_verified": verified,
        "descriptor": context_text if closed_ready else "",
        "closed_ready": closed_ready,
        "closed_reason": closed_reason,
        "open_prompt": open_prompt,
        "open_response": open_response,
        "open_messages_json": open_messages,
        "open_tokens": open_tokens,
        "closed_prompt": closed_prompt,
        "closed_response": closed_response,
        "closed_messages_json": closed_messages,
        "closed_tokens": closed_tokens,
    }


def _preference_candidate_from_route(
    record: Mapping[str, Any], route: Mapping[str, Any], context_text: str,
    context_json: str, split_rows: Mapping[str, Mapping[str, str]],
    models: Mapping[str, str], *, citation_trace: Mapping[str, Any] | None = None,
    provenance_enabled: bool = False,
) -> dict[str, Any]:
    row = SimpleNamespace(**dict(record))
    bundle = list(route.get("bundle") or [])
    ready, reason = funnel.preference_ready(row, bundle)
    verified = bool(route.get("paper_verified")) and ready
    verification_tier = "VERIFIED" if verified else "UNVERIFIED"
    citation = _citation_trace_fields(citation_trace)
    render_options = {
        "citation_label": citation["citation_label"],
        "paper_context": context_json,
        "verification_tier": verification_tier if provenance_enabled else None,
    }
    if bundle:
        rendered = funnel.render_preference(
            row, bundle, context_text, **render_options,
        )
    else:
        chosen = funnel.clean(record.get("chosen"))
        rejected = funnel.clean(record.get("rejected"))
        if provenance_enabled:
            basis = funnel.semantic_study_basis(context_json)
            chosen = funnel.format_provenance_response(
                chosen,
                provenance=funnel.UNVERIFIED_STUDY,
                citation_label=citation["citation_label"],
                study_basis=basis,
            )
            rejected = funnel.format_provenance_response(
                rejected,
                provenance=funnel.UNVERIFIED_STUDY,
                citation_label=citation["citation_label"],
                study_basis=basis,
            )
        rendered = {
            "prompt": funnel.render_closed(
                row, context_text, **render_options,
            )["messages"][0]["content"],
            "chosen": chosen,
            "rejected": rejected,
            "rejection_reason": funnel.clean(record.get("rejection_reason")),
        }
    detail = "" if verified else (reason or funnel.clean((route.get("report") or {}).get("reason")))
    meta = _metadata(record, split_rows, models)
    return {
        **meta,
        "pair_kind": funnel.clean(record.get("pair_kind")),
        "tags_json": funnel.clean(record.get("tags_json")) or "[]",
        "support_route": funnel.clean(route.get("route")),
        "prompt": rendered["prompt"],
        "chosen": rendered["chosen"],
        "rejected": rendered["rejected"],
        "rejection_reason": rendered["rejection_reason"],
        "evidence_json": canonical_json(bundle),
        "support_report_json": canonical_json(route.get("report") or {}),
        "paper_context": context_text,
        "paper_context_json": context_json,
        "verification_tier": "VERIFIED" if verified else "UNVERIFIED",
        "inclusion_source": "STRICT_DPO" if verified else "SUPPORT_QUARANTINE",
        "reason_code": "" if verified else "PREFERENCE_UNVERIFIED",
        "reason_detail": detail,
        "strict_ready": verified,
        "source_row_json": canonical_json(_row_dict(record)),
        **citation,
        "token_estimate": estimate_tokens(
            rendered["prompt"], rendered["chosen"], rendered["rejected"],
        ),
    }


def _reranker_candidate(
    record: Mapping[str, Any], context_text: str, context_json: str,
    split_rows: Mapping[str, Mapping[str, str]], models: Mapping[str, str],
) -> dict[str, Any] | None:
    positive = funnel.clean(record.get("positive_quote"))
    negative = funnel.clean(record.get("hard_negative_quote"))
    if not positive or not negative:
        return None
    meta = _metadata(record, split_rows, models)
    return {
        **meta,
        "positive_quote": positive,
        "hard_negative_quote": negative,
        "negative_reason": funnel.clean(record.get("negative_reason")),
        "paper_context": context_text,
        "paper_context_json": context_json,
        "verification_tier": "UNVERIFIED",
        "inclusion_source": "STRUCTURALLY_VALID_RERANKER",
        "reason_code": "HARD_NEGATIVE_NOT_VALIDATED",
        "reason_detail": "The supplied hard negative has not been proven to be a non-answer.",
        "source_row_json": canonical_json(_row_dict(record)),
        "token_estimate": estimate_tokens(
            record.get("question"), positive, negative,
        ),
    }


def suppress_cross_split_lanes(
    lanes: Mapping[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], list[tuple[str, dict[str, Any], str]]]:
    """Protect locked held-out questions across every training objective."""

    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for lane, rows in lanes.items():
        for row in rows:
            grouped[row["question_key"]].append((lane, row))
    remove: set[tuple[str, str]] = set()
    suppressed: list[tuple[str, dict[str, Any], str]] = []
    for key, items in grouped.items():
        if not key:
            continue
        splits = {row["split"] for _, row in items}
        if len(splits) < 2:
            continue
        keep = "test" if "test" in splits else "evaluate"
        for lane, row in items:
            if row["split"] != keep:
                remove.add((lane, row["pair_id"]))
                suppressed.append((lane, row, keep))
    return {
        lane: [row for row in rows if (lane, row["pair_id"]) not in remove]
        for lane, rows in lanes.items()
    }, suppressed


def suppress_cross_split_questions(
    sft_candidates: list[dict[str, Any]], dpo_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[str, dict[str, Any], str]]]:
    """Compatibility wrapper for the original two-lane contract."""

    kept, suppressed = suppress_cross_split_lanes({
        "SFT": sft_candidates, "DPO": dpo_rows,
    })
    return kept["SFT"], kept["DPO"], suppressed


def assign_curriculum(
    candidates: list[dict[str, Any]], config: BuildConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deterministically approximate OPEN/CLOSED/DUAL token-mass targets."""

    targets = {
        "OPEN": config.open_share,
        "CLOSED": config.closed_share,
        "DUAL": config.dual_share,
    }
    masses = {name: 0 for name in targets}

    def order_key(row: Mapping[str, Any]) -> tuple[int, str]:
        size = max(int(row["open_tokens"]), int(row["closed_tokens"]))
        tie = hashlib.sha256(f"{config.seed}:{row['pair_id']}".encode()).hexdigest()
        return -size, tie

    for row in sorted(candidates, key=order_key):
        choices = ["OPEN"] if not row["closed_ready"] else ["OPEN", "CLOSED", "DUAL"]
        costs = {
            "OPEN": int(row["open_tokens"]),
            "CLOSED": int(row["closed_tokens"]),
            "DUAL": int(row["open_tokens"]) + int(row["closed_tokens"]),
        }
        scored = []
        for choice in choices:
            trial = dict(masses)
            trial[choice] += costs[choice]
            total = max(sum(trial.values()), 1)
            score = sum((trial[name] / total - targets[name]) ** 2 for name in targets)
            tie = hashlib.sha256(
                f"{config.seed}:{row['pair_id']}:{choice}".encode(),
            ).hexdigest()
            scored.append((score, tie, choice))
        choice = min(scored)[2]
        row["assignment"] = choice
        masses[choice] += costs[choice]
    return candidates, masses


def assign_curriculum_by_split(
    candidates: list[dict[str, Any]], config: BuildConfig,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Assign each frozen partition independently to prevent held-out influence."""

    assigned: list[dict[str, Any]] = []
    masses: dict[str, dict[str, int]] = {}
    for split in ("train", "evaluate", "test"):
        partition = [dict(row) for row in candidates if row["split"] == split]
        partition, split_masses = assign_curriculum(partition, config)
        assigned.extend(partition)
        masses[split] = split_masses
    return assigned, masses


def render_sft(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        assignment = candidate["assignment"]
        modes = ("OPEN", "CLOSED") if assignment == "DUAL" else (assignment,)
        for mode in modes:
            lower = mode.lower()
            messages_json = candidate[f"{lower}_messages_json"]
            rows.append({
                "example_id": f"{candidate['pair_id']}:{lower}",
                **{key: (
                    funnel.clean(candidate.get(key))
                    if key in CITATION_TRACE_COLUMNS else candidate[key]
                ) for key in (
                    "pair_id", "paper_id", "family_id", "split", "pair_type",
                    "support_route", "question", "question_key", "pair_kind",
                    "tags_json", "extraction_model", "paper_context_json",
                    "paper_context",
                    "verification_tier", "inclusion_source", "reason_code",
                    "reason_detail", *CITATION_TRACE_COLUMNS,
                )},
                "mode": mode,
                "assignment": assignment,
                "prompt": candidate[f"{lower}_prompt"],
                "response": candidate[f"{lower}_response"],
                "messages": json.loads(messages_json),
                "messages_json": messages_json,
                "descriptor": candidate["descriptor"] if mode == "CLOSED" else "",
                "evidence_json": candidate["evidence_json"],
                "support_report_json": candidate["support_report_json"],
                "source_row_json": candidate["source_row_json"],
                "token_estimate": int(candidate[f"{lower}_tokens"]),
            })
    return rows


def render_sft_mixed(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Render verified curriculum rows plus one usable row per unverified pair."""

    prepared = []
    for candidate in candidates:
        row = dict(candidate)
        if row.get("paper_verified"):
            if "assignment" not in row:
                raise ValueError("verified mixed candidate lacks curriculum assignment")
        elif int(row.get("open_tokens", 0)) > 0:
            row["assignment"] = "OPEN"
        elif row.get("closed_ready") and int(row.get("closed_tokens", 0)) > 0:
            row["assignment"] = "CLOSED"
        else:
            raise ValueError(
                f"mixed SFT pair {row.get('pair_id')} has neither evidence nor usable context",
            )
        prepared.append(row)
    return render_sft(prepared)


def _frame(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
    for column in columns:
        if column not in frame:
            frame[column] = pd.Series(dtype="object")
    frame = frame.loc[:, list(columns)]
    for column in columns:
        if column == "token_estimate":
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
        elif column == "messages":
            frame[column] = frame[column].map(
                lambda value: value if isinstance(value, list) else [],
            )
        else:
            frame[column] = frame[column].map(funnel.clean)
    return frame


def _count_dict(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, int]:
    if frame.empty:
        return {}
    counts = frame.groupby(list(columns), dropna=False).size()
    return {" | ".join(map(str, key if isinstance(key, tuple) else (key,))): int(value)
            for key, value in counts.items()}


def validate_frames(frames: Mapping[str, pd.DataFrame], manifest: pd.DataFrame) -> None:
    expected = {
        "sft_examples": SFT_COLUMNS,
        "sft_mixed": SFT_COLUMNS,
        "dpo_pairs": DPO_COLUMNS,
        "preference_mixed": DPO_COLUMNS,
        "reranker_mixed": RERANKER_COLUMNS,
        "quarantine": QUARANTINE_COLUMNS,
        "discarded": DISCARDED_COLUMNS,
    }
    for name, columns in expected.items():
        if list(frames[name].columns) != list(columns):
            raise ValueError(f"{name} has an invalid column contract")
    sft, dpo = frames["sft_examples"], frames["dpo_pairs"]
    sft_mixed = frames["sft_mixed"]
    preference_mixed = frames["preference_mixed"]
    reranker_mixed = frames["reranker_mixed"]
    if (
        sft.example_id.duplicated().any()
        or sft_mixed.example_id.duplicated().any()
        or dpo.pair_id.duplicated().any()
        or preference_mixed.pair_id.duplicated().any()
        or reranker_mixed.pair_id.duplicated().any()
    ):
        raise ValueError("published training IDs are not unique")
    if not set(pd.concat([sft.pair_type, sft_mixed.pair_type])).issubset(
        {"FACTUAL", "REASONING"},
    ):
        raise ValueError("non-SFT pair type leaked into SFT")
    if not set(pd.concat([dpo.pair_type, preference_mixed.pair_type])).issubset(
        {"PREFERENCE"},
    ):
        raise ValueError("non-preference pair leaked into DPO")
    if not set(reranker_mixed.pair_type).issubset({"RERANKER"}):
        raise ValueError("non-reranker pair leaked into reranker training")
    if not set(sft.support_route).issubset(SUPPORTED_ROUTES):
        raise ValueError("unverified route leaked into SFT")
    if not set(dpo.support_route).issubset(SUPPORTED_ROUTES):
        raise ValueError("unverified route leaked into DPO")
    if not set(sft.verification_tier).issubset({"VERIFIED"}):
        raise ValueError("strict SFT contains an unverified row")
    if not set(dpo.verification_tier).issubset({"VERIFIED"}):
        raise ValueError("strict DPO contains an unverified row")
    if not set(sft.example_id).issubset(set(sft_mixed.example_id)):
        raise ValueError("strict SFT is not preserved in sft_mixed")
    if not set(dpo.pair_id).issubset(set(preference_mixed.pair_id)):
        raise ValueError("strict DPO is not preserved in preference_mixed")
    comparable_sft = [column for column in SFT_COLUMNS if column != "messages"]
    mixed_sft_by_id = sft_mixed.set_index("example_id", drop=False)
    for record in sft[comparable_sft].to_dict("records"):
        other = mixed_sft_by_id.loc[record["example_id"], comparable_sft].to_dict()
        if canonical_json(record) != canonical_json(other):
            raise ValueError("strict SFT row differs from its sft_mixed copy")
    mixed_preference_by_id = preference_mixed.set_index("pair_id", drop=False)
    for record in dpo[list(DPO_COLUMNS)].to_dict("records"):
        other = mixed_preference_by_id.loc[record["pair_id"], list(DPO_COLUMNS)].to_dict()
        if canonical_json(record) != canonical_json(other):
            raise ValueError("strict DPO row differs from its preference_mixed copy")
    valid_tiers = {"VERIFIED", "UNVERIFIED"}
    if not set(sft_mixed.verification_tier).issubset(valid_tiers):
        raise ValueError("sft_mixed has an invalid verification tier")
    if not set(preference_mixed.verification_tier).issubset(valid_tiers):
        raise ValueError("preference_mixed has an invalid verification tier")
    if set(reranker_mixed.verification_tier) - {"UNVERIFIED"}:
        raise ValueError("reranker_mixed must remain explicitly unverified")
    for name, frame in (
        ("strict SFT", sft), ("mixed SFT", sft_mixed),
    ):
        for row in frame.to_dict("records"):
            messages = row["messages"]
            if len(messages) != 2 or messages[1].get("content") != row["response"]:
                raise ValueError(f"{name} response disagrees with nested messages")
            traced = bool(row["citation_status"] or row["citation_metadata_source"])
            footer = "\n\nProvenance:" in row["response"]
            if traced != footer:
                raise ValueError(f"{name} citation trace/footer contract is inconsistent")
            label = row["citation_label"]
            if label and funnel.normalize_citation_label(label) != label:
                raise ValueError(f"{name} has a malformed author-year citation label")
    for name, frame in (
        ("strict DPO", dpo), ("mixed preference", preference_mixed),
    ):
        for row in frame.to_dict("records"):
            chosen_footer = "\n\nProvenance:" in row["chosen"]
            rejected_footer = "\n\nProvenance:" in row["rejected"]
            if chosen_footer != rejected_footer:
                raise ValueError(f"{name} exposes provenance as a preference shortcut")
            traced = bool(row["citation_status"] or row["citation_metadata_source"])
            if traced != chosen_footer:
                raise ValueError(f"{name} citation trace/footer contract is inconsistent")
    if (sft.loc[sft["mode"].eq("OPEN"), "evidence_json"] == "[]").any():
        raise ValueError("OPEN SFT example has no evidence")
    if sft.loc[sft["mode"].eq("CLOSED"), "descriptor"].eq("").any():
        raise ValueError("CLOSED SFT example has no descriptor")
    for name, frame in (("strict", sft), ("mixed", sft_mixed)):
        if (frame.loc[frame["mode"].eq("OPEN"), "evidence_json"] == "[]").any():
            raise ValueError(f"OPEN {name} SFT example has no evidence")
        if frame.loc[frame["mode"].eq("CLOSED"), "descriptor"].eq("").any():
            raise ValueError(f"CLOSED {name} SFT example has no descriptor")
        dual = frame[frame.assignment.eq("DUAL")].groupby("pair_id")["mode"].agg(list)
        if any(sorted(value) != ["CLOSED", "OPEN"] for value in dual):
            raise ValueError(f"{name} DUAL assignment is not exactly OPEN plus CLOSED")
    if reranker_mixed[["positive_quote", "hard_negative_quote"]].eq("").any().any():
        raise ValueError("reranker_mixed contains a blank positive or negative")
    split_map = manifest.set_index("paper_id")[["family_id", "split"]].to_dict("index")
    for name, frame in frames.items():
        if frame.empty:
            continue
        for row in frame[["paper_id", "family_id", "split"]].itertuples(index=False):
            expected_row = split_map[row.paper_id]
            if row.family_id != expected_row["family_id"] or row.split != expected_row["split"]:
                raise ValueError(f"{name} disagrees with frozen split manifest")
    combined = pd.concat([
        sft_mixed[["question_key", "split"]],
        preference_mixed[["question_key", "split"]],
        reranker_mixed[["question_key", "split"]],
    ], ignore_index=True)
    if not combined.empty and (combined.groupby("question_key").split.nunique() > 1).any():
        raise ValueError("normalized questions still cross published splits")


def _schema_hash(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _verify_run_files(run_dir: Path, marker: Mapping[str, Any]) -> None:
    expected_names = {f"{name}.parquet" for name in OUTPUT_SCHEMAS}
    files = marker.get("files")
    if not isinstance(files, Mapping) or set(files) != expected_names:
        raise ValueError("published run has an incomplete file manifest")
    for filename in sorted(expected_names):
        path = run_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"published run is missing {filename}")
        metadata = files[filename]
        name = filename.removesuffix(".parquet")
        schema = pq.read_schema(path)
        expected_schema = OUTPUT_SCHEMAS[name]
        if not schema.equals(expected_schema, check_metadata=False):
            raise ValueError(f"published {filename} has the wrong Arrow schema")
        parquet = pq.ParquetFile(path)
        if int(metadata.get("rows", -1)) != parquet.metadata.num_rows:
            raise ValueError(f"published {filename} row count is corrupt")
        if metadata.get("columns") != list(expected_schema.names):
            raise ValueError(f"published {filename} column manifest is corrupt")
        if metadata.get("schema_sha256") != _schema_hash(expected_schema):
            raise ValueError(f"published {filename} schema hash is corrupt")
        if metadata.get("sha256") != _file_hash(path):
            raise ValueError(f"published {filename} content hash is corrupt")


def _publish(
    frames: Mapping[str, pd.DataFrame], output_root: Path, manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    output_root = output_root.resolve()
    runs = output_root / "runs"
    staging_root = output_root / ".staging"
    runs.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    run_id = manifest["run_id"]
    final = _safe_child(runs, runs / run_id)
    if final.exists():
        marker_path = final / "_SUCCESS.json"
        if not marker_path.is_file():
            raise ValueError("existing run is incomplete: _SUCCESS.json is absent")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("identity_sha256") != manifest["identity_sha256"]:
            raise ValueError("existing run ID has a different identity")
        _verify_run_files(final, marker)
        _write_latest(output_root, marker)
        return final, marker
    stage = _safe_child(staging_root, staging_root / f"{run_id}-{uuid.uuid4().hex}")
    stage.mkdir()
    try:
        files: dict[str, Any] = {}
        for name, frame in frames.items():
            path = stage / f"{name}.parquet"
            schema = OUTPUT_SCHEMAS[name]
            table = pa.Table.from_pylist(frame.to_dict("records"), schema=schema)
            pq.write_table(table, path, compression="zstd")
            check = pq.read_table(path)
            if check.num_rows != len(frame) or not check.schema.equals(
                schema, check_metadata=False,
            ):
                raise ValueError(f"Parquet round-trip failed for {name}")
            files[path.name] = {
                "sha256": _file_hash(path), "rows": len(frame),
                "columns": list(frame.columns),
                "schema_sha256": _schema_hash(schema),
            }
        marker = {**manifest, "files": files}
        (stage / "_SUCCESS.json").write_text(
            json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, final)
        _verify_run_files(final, marker)
        _write_latest(output_root, marker)
        return final, marker
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _write_latest(output_root: Path, marker: Mapping[str, Any]) -> None:
    pointer = {
        "format": OUTPUT_FORMAT,
        "run_id": marker["run_id"],
        "directory": f"runs/{marker['run_id']}",
        "identity_sha256": marker["identity_sha256"],
        "files": marker.get("files", {}),
    }
    temporary = output_root / f".LATEST-{uuid.uuid4().hex}.tmp"
    temporary.write_text(
        json.dumps(pointer, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_root / "LATEST.json")


def build_training_set(config: BuildConfig, *, publish: bool = True) -> BuildOutcome:
    config.validate()
    provenance_mode = config.provenance_mode.upper()
    if publish and provenance_mode == "AUDIT_ONLY":
        raise ValueError(
            "AUDIT_ONLY provenance renders preview targets but cannot publish; "
            "use publish=False or explicitly switch to TRAINING after review",
        )
    started = time.monotonic()
    stages: list[dict[str, Any]] = []

    def note(stage: str, kept: int, removed: int = 0, detail: str = "") -> None:
        row = {"stage": stage, "kept": int(kept), "removed": int(removed), "detail": detail}
        stages.append(row)
        print(f"{stage:<34} kept {kept:>9,}  removed {removed:>8,}  {detail}", flush=True)

    table_dir, extraction_identity = resolve_extraction_tables(config.extraction_root)
    manifest = load_frozen_manifest(config.split_manifest)
    tables = funnel.load_tables(table_dir)
    pairs = tables["training_pairs"].copy()
    pairs["paper_id"] = pairs["paper_id"].map(funnel.clean)
    pairs["pair_id"] = pairs["pair_id"].map(funnel.clean)
    pairs["pair_type"] = pairs["pair_type"].map(
        lambda value: funnel.clean(value).upper(),
    )
    pair_papers = set(pairs.paper_id)
    manifest_papers = set(manifest.paper_id)
    missing = sorted(pair_papers - manifest_papers)
    if missing:
        raise ValueError(f"training pairs are absent from frozen manifest: {missing[:20]}")
    selected = sorted(pair_papers, key=lambda paper: hashlib.sha256(
        f"{config.seed}:{paper}".encode(),
    ).hexdigest())
    if config.paper_limit is not None:
        selected = selected[:config.paper_limit]
    selected_set = set(selected)
    pairs = pairs[pairs.paper_id.isin(selected_set)].copy()
    note("0 input pairs", len(pairs), detail=f"{len(selected_set):,} papers")

    missing_markdown = [paper for paper in selected if not (config.markdown_dir / f"{paper}.md").is_file()]
    if missing_markdown:
        raise FileNotFoundError(f"missing Markdown papers: {missing_markdown[:20]}")
    citation_by_paper: dict[str, dict[str, Any]] = {}
    if provenance_mode != "OFF":
        citation_index = citations.load_citation_metadata(config.citation_metadata_path)
        title_by_paper = (
            manifest.set_index("paper_id")["title"].map(funnel.clean).to_dict()
            if "title" in manifest.columns else {}
        )
        citation_by_paper = {
            paper_id: citations.citation_for_paper(
                paper_id, citation_index, title=title_by_paper.get(paper_id, ""),
            )
            for paper_id in selected
        }
        citation_statuses = Counter(
            funnel.clean(row.get("citation_status")) or "UNSPECIFIED"
            for row in citation_by_paper.values()
        )
        note(
            "0b citation provenance", len(citation_by_paper), 0,
            f"mode={provenance_mode}; audit only, zero pair filters; "
            f"status={dict(citation_statuses)}",
        )
    input_files = _input_fingerprints(config, table_dir, selected)
    split_rows = manifest.set_index("paper_id")[["family_id", "split"]].to_dict("index")
    status = tables["extraction_status"]
    complete = set(status[status.complete.astype("boolean").fillna(False)].paper_id)
    models = status.set_index("paper_id").model.map(funnel.clean).to_dict()
    not_real, not_africa = funnel.failed_verdicts(tables["paper_profiles"])
    profiles = tables["paper_profiles"].drop_duplicates("paper_id").set_index("paper_id").to_dict("index")
    contexts_frame = tables["study_contexts"]
    if "source_task" in contexts_frame:
        contexts_frame = contexts_frame.sort_values("source_task")
    contexts = {
        paper_id: group.to_dict("records")
        for paper_id, group in contexts_frame.groupby("paper_id", sort=False)
    }
    innovation = tables["african_innovation"].drop_duplicates("paper_id").set_index("paper_id").to_dict("index")

    recovered = funnel.recover_evidence_bundles(
        config.extraction_root / "raw", papers=selected_set,
    )
    table_evidence = tables["training_evidence"]
    recovered_count = sum(
        bool(funnel.combined_evidence(pair_id, table_evidence, recovered))
        for pair_id in pairs.pair_id
    )
    note("1 evidence bundles", recovered_count, len(pairs) - recovered_count)

    discarded_rows: list[dict[str, Any]] = []
    structural_keep: list[Any] = []
    structural_reasons: Counter[str] = Counter()
    for index, row in pairs.iterrows():
        record = row.to_dict()
        reasons = funnel.discard_reasons(
            SimpleNamespace(**record), complete, not_real, not_africa,
        )
        if reasons:
            for reason in reasons:
                structural_reasons[reason] += 1
            discarded_rows.append(_discarded_row(
                record, split_rows, models, stage="STRUCTURAL",
                reason_code=";".join(sorted(reasons)),
                reason_detail=canonical_json(reasons), record_suffix=f":{index}",
            ))
        else:
            structural_keep.append(index)
    alive = pairs.loc[structural_keep].copy()
    note("2 structural gates", len(alive), len(pairs) - len(alive), str(dict(structural_reasons)))

    deduplicated, duplicate_drops = funnel.resolve_duplicates(alive)
    kept_ids = set(deduplicated.pair_id)
    for index, row in duplicate_drops.iterrows():
        record = row.to_dict()
        reason = "EXACT_DUPLICATE" if record["pair_id"] in kept_ids else "CONFLICTING_PAIR_ID"
        discarded_rows.append(_discarded_row(
            record, split_rows, models, stage="DUPLICATE", reason_code=reason,
            record_suffix=f":{index}",
        ))
    alive = deduplicated
    note("3 duplicate safety", len(alive), len(duplicate_drops))

    quarantine_rows: list[dict[str, Any]] = []
    reranker_mixed_rows: list[dict[str, Any]] = []
    for record in alive[alive.pair_type.eq("RERANKER")].to_dict("records"):
        evidence = funnel.combined_evidence(record["pair_id"], table_evidence, recovered)
        context_text, context_json = _pair_context(
            record, evidence, contexts, profiles, innovation,
        )
        mixed = _reranker_candidate(
            record, context_text, context_json, split_rows, models,
        )
        if mixed is None:
            raise RuntimeError("structural gate admitted a non-renderable RERANKER row")
        reranker_mixed_rows.append(mixed)
        quarantine_rows.append(_quarantine_row(
            record, split_rows, models, stage="RERANKER",
            reason_code="HARD_NEGATIVE_NOT_VALIDATED",
            reason_detail="RERANKER hard negatives are not proven non-answers",
            evidence=evidence,
        ))

    support = alive[alive.pair_type.isin(["FACTUAL", "REASONING", "PREFERENCE"])]
    router_identity = hashlib.sha256(canonical_json({
        "router_sha256": _file_hash(Path(funnel.__file__)),
        "builder_version": BUILDER_VERSION,
        "max_evidence_spans": config.max_evidence_spans,
        "max_evidence_chars": config.max_evidence_chars,
    }).encode()).hexdigest()[:24]
    route_cache = config.output_root.resolve() / "router-cache" / router_identity
    tasks = _task_stream(
        support, table_evidence, recovered, config.markdown_dir,
        config.max_evidence_spans, config.max_evidence_chars,
        route_cache, router_identity,
    )
    paper_total = support.paper_id.nunique()
    sft_mixed_candidates: list[dict[str, Any]] = []
    preference_mixed_rows: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()
    cache_hits = 0
    routed = 0
    iterator: Iterable[tuple[list[dict[str, Any]], dict[str, Any], bool]]
    if config.router_workers == 1:
        iterator = map(_route_task, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=config.router_workers)
        iterator = _bounded_parallel_routes(
            executor, tasks, limit=max(2, 2 * config.router_workers),
        )
    try:
        for paper_number, (records, routes, cache_hit) in enumerate(iterator, 1):
            cache_hits += int(cache_hit)
            for record in records:
                route = routes[record["pair_id"]]
                route_counts[route["route"]] += 1
                routed += 1
                pair_type = funnel.clean(record["pair_type"]).upper()
                context_text, context_json = _pair_context(
                    record, route.get("bundle") or (), contexts, profiles, innovation,
                )
                if pair_type in {"FACTUAL", "REASONING"}:
                    candidate = _sft_candidate_from_route(
                        record, route, context_text, context_json, split_rows, models,
                        citation_trace=citation_by_paper.get(record["paper_id"]),
                        provenance_enabled=provenance_mode != "OFF",
                    )
                    sft_mixed_candidates.append(candidate)
                    if not route["paper_verified"]:
                        quarantine_rows.append(_quarantine_row(
                            record, split_rows, models, stage="SUPPORT",
                            reason_code="QUARANTINE_UNVERIFIED",
                            reason_detail=funnel.clean(route["report"].get("reason")),
                            support_route=route["route"], evidence=route["bundle"],
                            report=route["report"],
                        ))
                else:
                    preference = _preference_candidate_from_route(
                        record, route, context_text, context_json, split_rows, models,
                        citation_trace=citation_by_paper.get(record["paper_id"]),
                        provenance_enabled=provenance_mode != "OFF",
                    )
                    preference_mixed_rows.append(preference)
                    if not route["paper_verified"]:
                        quarantine_rows.append(_quarantine_row(
                            record, split_rows, models, stage="SUPPORT",
                            reason_code="QUARANTINE_UNVERIFIED",
                            reason_detail=funnel.clean(route["report"].get("reason")),
                            support_route=route["route"], evidence=route["bundle"],
                            report=route["report"],
                        ))
                    elif not preference["strict_ready"]:
                        quarantine_rows.append(_quarantine_row(
                            record, split_rows, models, stage="DPO_CONTRACT",
                            reason_code="PREFERENCE_NOT_READY",
                            reason_detail=preference["reason_detail"],
                            support_route=route["route"], evidence=route["bundle"],
                            report=route["report"],
                        ))
            if config.progress_every and (
                paper_number % config.progress_every == 0 or paper_number == paper_total
            ):
                elapsed = time.monotonic() - started
                print(f"   routed {paper_number:,}/{paper_total:,} papers in {elapsed/60:.1f} min", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    note(
        "4 support router", routed, route_counts["QUARANTINE_UNVERIFIED"],
        f"{dict(route_counts)}; resumed {cache_hits:,}/{paper_total:,} paper checkpoints",
    )

    mixed_lanes, suppressed = suppress_cross_split_lanes({
        "SFT": sft_mixed_candidates,
        "PREFERENCE": preference_mixed_rows,
        "RERANKER": reranker_mixed_rows,
    })
    sft_mixed_candidates = mixed_lanes["SFT"]
    preference_mixed_rows = mixed_lanes["PREFERENCE"]
    reranker_mixed_rows = mixed_lanes["RERANKER"]
    for lane, row, kept_split in suppressed:
        source = json.loads(row["source_row_json"])
        quarantine_rows.append(_quarantine_row(
            source, split_rows, models, stage="SPLIT_LEAKAGE",
            reason_code="NORMALIZED_QUESTION_CROSSES_SPLITS",
            reason_detail=f"{lane} row withheld; the same normalized question is retained in {kept_split}",
            support_route=funnel.clean(row.get("support_route")),
            evidence=json.loads(row.get("evidence_json") or "[]"),
            report=json.loads(row.get("support_report_json") or "{}"),
            record_suffix=f":{lane}",
        ))
    note(
        "5 split leakage audit",
        len(sft_mixed_candidates) + len(preference_mixed_rows) + len(reranker_mixed_rows),
        len(suppressed),
    )

    candidates = [row for row in sft_mixed_candidates if row["paper_verified"]]
    dpo_rows = [row for row in preference_mixed_rows if row["strict_ready"]]
    candidates, token_masses = assign_curriculum_by_split(candidates, config)
    assigned_by_pair = {row["pair_id"]: row for row in candidates}
    sft_mixed_candidates = [
        assigned_by_pair.get(row["pair_id"], row) for row in sft_mixed_candidates
    ]
    renderable_mixed: list[dict[str, Any]] = []
    for row in sft_mixed_candidates:
        if (
            row.get("paper_verified")
            or int(row.get("open_tokens", 0)) > 0
            or (row.get("closed_ready") and int(row.get("closed_tokens", 0)) > 0)
        ):
            renderable_mixed.append(row)
            continue
        raise RuntimeError(
            f"structurally valid mixed-SFT pair {row['pair_id']} was not rendered",
        )
    sft_rows = render_sft(candidates)
    sft_mixed_rows = render_sft_mixed(renderable_mixed)
    note(
        "6 mixed SFT renderings", len(sft_rows),
        detail=f"assignments={dict(Counter(row['assignment'] for row in candidates))}",
    )
    note("7 grounded DPO", len(dpo_rows))
    note(
        "8 mixed training lanes",
        len(sft_mixed_rows) + len(preference_mixed_rows) + len(reranker_mixed_rows),
        0,
        detail=(
            f"SFT={len(sft_mixed_rows):,}; preference={len(preference_mixed_rows):,}; "
            f"reranker={len(reranker_mixed_rows):,}"
        ),
    )
    note("9 quarantine", len(quarantine_rows), detail="strict-quality audit queue")

    frames = {
        "sft_examples": _frame(sft_rows, SFT_COLUMNS),
        "sft_mixed": _frame(sft_mixed_rows, SFT_COLUMNS),
        "dpo_pairs": _frame(dpo_rows, DPO_COLUMNS),
        "preference_mixed": _frame(preference_mixed_rows, DPO_COLUMNS),
        "reranker_mixed": _frame(reranker_mixed_rows, RERANKER_COLUMNS),
        "quarantine": _frame(quarantine_rows, QUARANTINE_COLUMNS),
        "discarded": _frame(discarded_rows, DISCARDED_COLUMNS),
    }
    sort_keys = {
        "sft_examples": ["example_id"],
        "sft_mixed": ["example_id"],
        "dpo_pairs": ["pair_id"],
        "preference_mixed": ["pair_id"],
        "reranker_mixed": ["pair_id"],
        "quarantine": ["record_id"],
        "discarded": ["record_id"],
    }
    frames = {
        name: frame.sort_values(sort_keys[name], kind="stable").reset_index(drop=True)
        for name, frame in frames.items()
    }
    validate_frames(frames, manifest)

    final_input_files = _input_fingerprints(config, table_dir, selected)
    if final_input_files != input_files:
        raise RuntimeError("source data changed while the training build was running")
    identity = {
        "format": OUTPUT_FORMAT,
        "builder_version": BUILDER_VERSION,
        "config": {
            key: value for key, value in asdict(config).items()
            if key not in {"output_root", "router_workers", "progress_every"}
        },
        "extraction": extraction_identity,
        "inputs": input_files,
    }
    identity["config"] = _jsonable(identity["config"])
    identity_sha = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    run_id = identity_sha[:24]
    output_counts = {name: len(frame) for name, frame in frames.items()}
    run_manifest = {
        **identity,
        "identity_sha256": identity_sha,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "token_estimator": TOKEN_ESTIMATOR,
        "selected_papers": len(selected),
        "stage_counts": stages,
        "output_counts": output_counts,
        "sft_by_split_mode": _count_dict(frames["sft_examples"], ["split", "mode"]),
        "sft_by_route": _count_dict(frames["sft_examples"], ["support_route"]),
        "sft_mixed_by_split_tier_mode": _count_dict(
            frames["sft_mixed"], ["split", "verification_tier", "mode"],
        ),
        "dpo_by_split": _count_dict(frames["dpo_pairs"], ["split"]),
        "preference_mixed_by_split_tier": _count_dict(
            frames["preference_mixed"], ["split", "verification_tier"],
        ),
        "reranker_mixed_by_split": _count_dict(
            frames["reranker_mixed"], ["split"],
        ),
        "provenance_contract": {
            "mode": provenance_mode,
            "citation_metadata": (
                "audit trace attached; never used as a pair filter"
                if provenance_mode != "OFF" else "disabled"
            ),
            "assistant_turn": "unchanged scientific core used by support routing",
            "preference": "chosen and rejected carry the same provenance format",
            "publication": (
                "explicitly blocked" if provenance_mode == "AUDIT_ONLY"
                else "allowed by mode"
            ),
        },
        "citation_by_status": _count_dict(
            frames["sft_mixed"], ["citation_status"],
        ),
        "training_lane_contract": {
            "sft_examples": "verified FACTUAL/REASONING only",
            "sft_mixed": "all renderable, structurally valid FACTUAL/REASONING; verification tier retained",
            "dpo_pairs": "verified PREFERENCE only",
            "preference_mixed": "all structurally valid PREFERENCE; verification tier retained",
            "reranker_mixed": "all structurally valid RERANKER; hard-negative validity remains unverified",
            "hard_rejects": "structural failures and unresolved pair_id conflicts only",
        },
        "quarantine_by_stage_reason": _count_dict(
            frames["quarantine"], ["stage", "reason_code"],
        ),
        "discarded_by_stage_reason": _count_dict(
            frames["discarded"], ["stage", "reason_code"],
        ),
        "curriculum_assignment_token_mass": token_masses,
        "elapsed_seconds_before_publication": round(time.monotonic() - started, 3),
    }
    run_dir = None
    if publish:
        run_dir, run_manifest = _publish(frames, config.output_root, run_manifest)
        print(f"published immutable training generation: {run_dir}", flush=True)
    return BuildOutcome(run_id, run_dir, frames, run_manifest, stages)
