"""Apply local semantic retrieval results to an immutable MUFASA mixed SFT run.

The lexical builder generation is never modified.  This module verifies the
builder ``LATEST.json`` and its complete file manifest, accepts same-paper
exact spans for support-quarantined FACTUAL/REASONING pairs, and publishes a
derived immutable generation under ``semantic_runs``.  A partial/smoke run is
useful as an audit artifact but can never advance ``SEMANTIC_LATEST.json``.

Semantic retrieval is deliberately not treated as verification.  Every
refined row remains ``UNVERIFIED``; a deterministic-pass route and a best
failing semantic bundle are labelled separately.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import mufasa_dataset as funnel
import mufasa_semantic as semantic
import mufasa_training_builder as builder


REFINER_VERSION = "mufasa-semantic-refiner-v1.1"
OUTPUT_FORMAT = "mufasa-semantic-refined-sft-v2"
INPUT_ROUTES = frozenset({
    "VECTOR_CANDIDATE_DETERMINISTIC_PASS",
    "STILL_QUARANTINED",
})

SEMANTIC_ROUTE_COLUMNS = (
    "pair_id", "paper_id", "family_id", "split", "pair_type", "example_id",
    "source_support_route", "semantic_route", "refined_support_route",
    "refinement_action", "original_evidence_json", "candidate_bundle_json",
    "original_support_report_json", "support_report_json",
    "retrieval_metadata_json", "source_row_json",
)
SEMANTIC_CANDIDATE_COLUMNS = (
    "pair_id", "paper_id", "rank", "score", "candidate_origin",
    "evidence_token_count", "chunk_index", "char_start", "char_end", "page",
    "section", "source_kind", "source_label", "quote_sha256",
    "selected_in_bundle",
)

SEMANTIC_ROUTE_SCHEMA = pa.schema([
    pa.field(column, pa.string(), nullable=False)
    for column in SEMANTIC_ROUTE_COLUMNS
])
SEMANTIC_CANDIDATE_SCHEMA = pa.schema([
    pa.field("pair_id", pa.string(), nullable=False),
    pa.field("paper_id", pa.string(), nullable=False),
    pa.field("rank", pa.int64(), nullable=False),
    pa.field("score", pa.string(), nullable=False),
    pa.field("candidate_origin", pa.string(), nullable=False),
    pa.field("evidence_token_count", pa.int64(), nullable=False),
    pa.field("chunk_index", pa.int64(), nullable=False),
    pa.field("char_start", pa.int64(), nullable=False),
    pa.field("char_end", pa.int64(), nullable=False),
    pa.field("page", pa.string(), nullable=False),
    pa.field("section", pa.string(), nullable=False),
    pa.field("source_kind", pa.string(), nullable=False),
    pa.field("source_label", pa.string(), nullable=False),
    pa.field("quote_sha256", pa.string(), nullable=False),
    pa.field("selected_in_bundle", pa.bool_(), nullable=False),
])
OUTPUT_SCHEMAS = {
    "sft_mixed": builder.OUTPUT_SCHEMAS["sft_mixed"],
    "semantic_routes": SEMANTIC_ROUTE_SCHEMA,
    "semantic_candidates": SEMANTIC_CANDIDATE_SCHEMA,
}


@dataclass(frozen=True)
class RefineConfig:
    """Paths and safety policy for one semantic refinement generation."""

    training_root: Path
    markdown_dir: Path
    output_root: Path | None = None
    preview: bool = False

    def resolved_output_root(self) -> Path:
        return (self.output_root or self.training_root).resolve()

    def validate(self) -> None:
        if not self.training_root.is_dir():
            raise FileNotFoundError(f"training root does not exist: {self.training_root}")
        if not self.markdown_dir.is_dir():
            raise FileNotFoundError(f"Markdown directory does not exist: {self.markdown_dir}")


@dataclass
class SourceRun:
    """Verified builder generation plus its semantic-refinement inputs."""

    training_root: Path
    run_dir: Path
    pointer: dict[str, Any]
    marker: dict[str, Any]
    pointer_sha256: str
    sft_mixed: pd.DataFrame
    sft_examples: pd.DataFrame
    quarantine: pd.DataFrame
    eligible_records: tuple[dict[str, Any], ...]
    initial_evidence: dict[str, list[dict[str, Any]]]
    source_complete: bool

    @property
    def eligible_pair_ids(self) -> tuple[str, ...]:
        return tuple(record["pair_id"] for record in self.eligible_records)


@dataclass
class RefineOutcome:
    run_id: str
    run_dir: Path | None
    source_builder_run_id: str
    source_builder_identity_sha256: str
    frames: dict[str, pd.DataFrame]
    manifest: dict[str, Any]
    complete_coverage: bool
    training_ready: bool
    latest_advanced: bool


@dataclass(frozen=True)
class EffectiveSFT:
    """The safe mixed-SFT path selected for downstream training."""

    path: Path
    source: str
    source_builder_run_id: str
    source_builder_identity_sha256: str
    semantic_run_id: str | None = None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _parse_json(value: Any, label: str, expected: type) -> Any:
    if isinstance(value, expected):
        return value
    try:
        parsed = json.loads(funnel.clean(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, expected):
        raise ValueError(f"{label} must decode to {expected.__name__}")
    return parsed


def _messages_as_lists(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "messages" in frame:
        frame["messages"] = frame["messages"].map(
            lambda value: value.tolist() if hasattr(value, "tolist") else value,
        )
    return frame


def _safe_run_dir(root: Path, directory: Any, run_id: Any) -> Path:
    text = funnel.clean(directory)
    expected = f"runs/{funnel.clean(run_id)}"
    if text.replace("\\", "/") != expected:
        raise ValueError("builder LATEST directory does not match its run_id")
    run_dir = (root / text).resolve()
    if run_dir.parent != (root / "runs").resolve():
        raise ValueError("builder LATEST directory escapes training_root/runs")
    return run_dir


def load_source_run(training_root: str | Path) -> SourceRun:
    """Resolve and cryptographically verify the builder's current generation."""

    root = Path(training_root).resolve()
    pointer_path = root / "LATEST.json"
    if not pointer_path.is_file():
        raise FileNotFoundError(f"builder pointer is absent: {pointer_path}")
    pointer = _read_json(pointer_path)
    if pointer.get("format") != builder.OUTPUT_FORMAT:
        raise ValueError("LATEST.json is not a compatible training-builder generation")
    run_dir = _safe_run_dir(root, pointer.get("directory"), pointer.get("run_id"))
    marker_path = run_dir / "_SUCCESS.json"
    if not marker_path.is_file():
        raise FileNotFoundError("builder generation has no _SUCCESS.json")
    marker = _read_json(marker_path)
    for key in ("run_id", "identity_sha256", "files"):
        if builder.canonical_json(pointer.get(key)) != builder.canonical_json(marker.get(key)):
            raise ValueError(f"builder pointer disagrees with _SUCCESS.json for {key}")
    builder._verify_run_files(run_dir, marker)
    inputs = marker.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("builder generation does not record its module fingerprints")
    module_expectations = {
        "router_module": Path(funnel.__file__),
        "builder_module": Path(builder.__file__),
    }
    for key, path in module_expectations.items():
        if inputs.get(key) != builder._file_hash(path):
            raise RuntimeError(
                f"current {path.name} differs from the module pinned by the builder run",
            )

    sft_mixed = _messages_as_lists(pd.read_parquet(run_dir / "sft_mixed.parquet"))
    sft_examples = _messages_as_lists(pd.read_parquet(run_dir / "sft_examples.parquet"))
    quarantine = pd.read_parquet(run_dir / "quarantine.parquet")
    if list(sft_mixed.columns) != list(builder.SFT_COLUMNS):
        raise ValueError("source sft_mixed has an incompatible schema")
    if sft_mixed.example_id.duplicated().any():
        raise ValueError("source sft_mixed example_id is not unique")

    eligible_quarantine = quarantine[
        quarantine.stage.eq("SUPPORT")
        & quarantine.reason_code.eq("QUARANTINE_UNVERIFIED")
        & quarantine.pair_type.isin(["FACTUAL", "REASONING"])
    ]
    eligible_mixed = sft_mixed[
        sft_mixed.verification_tier.eq("UNVERIFIED")
        & sft_mixed.inclusion_source.eq("SUPPORT_QUARANTINE")
        & sft_mixed.pair_type.isin(["FACTUAL", "REASONING"])
        & sft_mixed["mode"].isin(["OPEN", "CLOSED"])
    ]
    eligible_ids = sorted(
        set(eligible_quarantine.pair_id) & set(eligible_mixed.pair_id),
    )
    records: list[dict[str, Any]] = []
    initial: dict[str, list[dict[str, Any]]] = {}
    for pair_id in eligible_ids:
        mixed_rows = eligible_mixed[eligible_mixed.pair_id.eq(pair_id)]
        quarantine_rows = eligible_quarantine[
            eligible_quarantine.pair_id.eq(pair_id)
        ]
        if len(mixed_rows) != 1 or len(quarantine_rows) != 1:
            raise ValueError(f"eligible pair {pair_id} is not one-to-one across source tables")
        mixed = mixed_rows.iloc[0]
        record = _parse_json(
            quarantine_rows.iloc[0].source_row_json,
            f"source_row_json for {pair_id}", dict,
        )
        if (
            funnel.clean(record.get("pair_id")) != pair_id
            or funnel.clean(record.get("paper_id")) != mixed.paper_id
            or funnel.clean(record.get("pair_type")).upper() != mixed.pair_type
        ):
            raise ValueError(f"source-row provenance mismatch for {pair_id}")
        records.append(record)
        initial[pair_id] = _parse_json(
            mixed.evidence_json, f"evidence_json for {pair_id}", list,
        )

    identity_config = marker.get("config")
    source_complete = (
        isinstance(identity_config, Mapping)
        and "paper_limit" in identity_config
        and identity_config.get("paper_limit") is None
    )
    return SourceRun(
        training_root=root,
        run_dir=run_dir,
        pointer=pointer,
        marker=marker,
        pointer_sha256=builder._file_hash(pointer_path),
        sft_mixed=sft_mixed,
        sft_examples=sft_examples,
        quarantine=quarantine,
        eligible_records=tuple(records),
        initial_evidence=initial,
        source_complete=source_complete,
    )


def resolve_effective_sft_mixed(
    training_root: str | Path,
) -> EffectiveSFT:
    """Resolve semantic SFT only when it pins the current verified builder run.

    Absence of ``SEMANTIC_LATEST.json`` cleanly falls back to the builder's
    mixed SFT.  A present but stale/corrupt semantic pointer fails loudly.
    """

    source = load_source_run(training_root)
    pointer_path = source.training_root / "SEMANTIC_LATEST.json"
    if not pointer_path.is_file():
        return EffectiveSFT(
            path=source.run_dir / "sft_mixed.parquet",
            source="BUILDER",
            source_builder_run_id=source.marker["run_id"],
            source_builder_identity_sha256=source.marker["identity_sha256"],
        )
    pointer = _read_json(pointer_path)
    if pointer.get("format") != OUTPUT_FORMAT or pointer.get("training_ready") is not True:
        raise ValueError("SEMANTIC_LATEST is not a training-ready semantic generation")
    if (
        pointer.get("source_builder_run_id") != source.marker["run_id"]
        or pointer.get("source_builder_identity_sha256")
        != source.marker["identity_sha256"]
    ):
        raise RuntimeError("SEMANTIC_LATEST is stale against the current builder LATEST")
    run_id = funnel.clean(pointer.get("run_id"))
    expected_directory = f"semantic_runs/{run_id}"
    if funnel.clean(pointer.get("directory")).replace("\\", "/") != expected_directory:
        raise ValueError("SEMANTIC_LATEST directory does not match its run_id")
    run_dir = (source.training_root / expected_directory).resolve()
    if run_dir.parent != (source.training_root / "semantic_runs").resolve():
        raise ValueError("SEMANTIC_LATEST escapes semantic_runs")
    marker = _read_json(run_dir / "_SUCCESS.json")
    for key in (
        "run_id", "identity_sha256", "source_builder_run_id",
        "source_builder_identity_sha256", "files", "training_ready",
    ):
        if builder.canonical_json(pointer.get(key)) != builder.canonical_json(marker.get(key)):
            raise ValueError(f"SEMANTIC_LATEST disagrees with _SUCCESS.json for {key}")
    _verify_derived_run(run_dir, marker)
    return EffectiveSFT(
        path=run_dir / "sft_mixed.parquet",
        source="SEMANTIC_REFINED",
        source_builder_run_id=source.marker["run_id"],
        source_builder_identity_sha256=source.marker["identity_sha256"],
        semantic_run_id=run_id,
    )


def _route_mapping(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, Mapping):
        output = {}
        for pair_id, result in value.items():
            if not isinstance(result, Mapping):
                raise ValueError(f"semantic result for {pair_id} is not an object")
            output[funnel.clean(pair_id)] = dict(result)
        return output
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.suffix.casefold() == ".parquet":
            value = pd.read_parquet(path)
        else:
            return _route_mapping(_read_json(path))
    if isinstance(value, pd.DataFrame):
        required = {
            "pair_id", "semantic_route", "candidate_bundle_json",
            "support_report_json",
        }
        missing = required - set(value.columns)
        if missing:
            raise ValueError(f"semantic route table is missing {sorted(missing)}")
        output = {}
        for row in value.to_dict("records"):
            pair_id = funnel.clean(row["pair_id"])
            output[pair_id] = {
                "route": row["semantic_route"],
                "bundle": _parse_json(
                    row["candidate_bundle_json"], "candidate_bundle_json", list,
                ),
                "report": _parse_json(
                    row["support_report_json"], "support_report_json", dict,
                ),
                "hits": [],
                **_parse_json(
                    row.get("retrieval_metadata_json", "{}"),
                    "retrieval_metadata_json", dict,
                ),
            }
        return output
    raise TypeError("semantic_results must be a route mapping, JSON, or Parquet table")


def _exact_span(
    span: Mapping[str, Any], paper_id: str, text: str, label: str,
) -> dict[str, Any]:
    if funnel.clean(span.get("paper_id")) != paper_id:
        raise ValueError(f"{label} contains a cross-paper span")
    start, end = span.get("char_start"), span.get("char_end")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not (0 <= start < end <= len(text))
        or funnel.clean(span.get("quote")) != funnel.clean(text[start:end])
    ):
        raise ValueError(f"{label} is not an exact source span")
    exact = dict(span)
    exact["paper_id"] = paper_id
    exact["quote"] = text[start:end]
    exact["char_start"] = start
    exact["char_end"] = end
    return exact


def _normalize_results(
    routes: Mapping[str, Mapping[str, Any]], source: SourceRun,
    markdown_dir: Path,
) -> dict[str, dict[str, Any]]:
    eligible = set(source.eligible_pair_ids)
    supplied = set(routes)
    if "" in supplied:
        raise ValueError("semantic results contain a blank pair_id")
    unexpected = sorted(supplied - eligible)
    if unexpected:
        raise ValueError(f"semantic results contain ineligible pairs: {unexpected[:20]}")
    if len(supplied) != len(routes):
        raise ValueError("semantic result pair_id is not unique")
    records = {record["pair_id"]: record for record in source.eligible_records}
    texts: dict[str, str] = {}
    output: dict[str, dict[str, Any]] = {}
    for pair_id in sorted(routes):
        result = dict(routes[pair_id])
        route = funnel.clean(result.get("route"))
        if route not in INPUT_ROUTES:
            raise ValueError(f"semantic result {pair_id} has invalid route {route!r}")
        paper_id = funnel.clean(records[pair_id].get("paper_id"))
        if paper_id not in texts:
            path = markdown_dir / f"{paper_id}.md"
            if not path.is_file():
                raise FileNotFoundError(f"missing Markdown source for {paper_id}")
            texts[paper_id] = path.read_text(encoding="utf-8")
        text = texts[paper_id]
        bundle_value = result.get("bundle") or []
        hits_value = result.get("hits") or []
        if not isinstance(bundle_value, Sequence) or isinstance(bundle_value, (str, bytes)):
            raise ValueError(f"semantic result {pair_id} bundle is not a list")
        if not isinstance(hits_value, Sequence) or isinstance(hits_value, (str, bytes)):
            raise ValueError(f"semantic result {pair_id} hits is not a list")
        bundle = [
            _exact_span(span, paper_id, text, f"semantic bundle for {pair_id}")
            for span in bundle_value
        ]
        hits = []
        for rank, hit_value in enumerate(hits_value, 1):
            if not isinstance(hit_value, Mapping):
                raise ValueError(f"semantic hit {rank} for {pair_id} is not an object")
            hit = dict(hit_value)
            span = hit.get("span")
            if not isinstance(span, Mapping):
                raise ValueError(f"semantic hit {rank} for {pair_id} has no span")
            hit["span"] = _exact_span(
                span, paper_id, text, f"semantic hit {rank} for {pair_id}",
            )
            hits.append(hit)
        report = result.get("report") or {}
        if not isinstance(report, Mapping):
            raise ValueError(f"semantic result {pair_id} report is not an object")
        limitations = result.get("retrieval_limitations") or []
        if not isinstance(limitations, Sequence) or isinstance(limitations, (str, bytes)):
            raise ValueError(f"semantic result {pair_id} limitations are not a list")
        output[pair_id] = {
            "route": route,
            "bundle": bundle,
            "report": dict(report),
            "hits": hits,
            "query_truncation_count": int(result.get("query_truncation_count", 0)),
            "long_query_split_count": int(result.get("long_query_split_count", 0)),
            "retrieval_limitations": [funnel.clean(item) for item in limitations],
        }
    return output


def _span_key(span: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        funnel.clean(span.get("paper_id")),
        int(span.get("char_start", -1)),
        int(span.get("char_end", -1)),
    )


def _compact_candidate_rows(
    pair_id: str, paper_id: str, result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected = {_span_key(span) for span in result["bundle"]}
    rows = []
    for rank, hit in enumerate(result["hits"], 1):
        span = hit["span"]
        score = hit.get("score")
        rows.append({
            "pair_id": pair_id,
            "paper_id": paper_id,
            "rank": rank,
            "score": "" if score is None else format(float(score), ".17g"),
            "candidate_origin": funnel.clean(hit.get("candidate_origin")),
            "evidence_token_count": int(hit.get("evidence_token_count", 0)),
            "chunk_index": int(hit.get("chunk_index", -1)),
            "char_start": int(span["char_start"]),
            "char_end": int(span["char_end"]),
            "page": funnel.clean(span.get("page")),
            "section": funnel.clean(span.get("section")),
            "source_kind": funnel.clean(span.get("source_kind")),
            "source_label": funnel.clean(span.get("source_label")),
            "quote_sha256": hashlib.sha256(span["quote"].encode("utf-8")).hexdigest(),
            "selected_in_bundle": _span_key(span) in selected,
        })
    return rows


def _semantic_route_name(result: Mapping[str, Any]) -> str:
    if result["route"] == "VECTOR_CANDIDATE_DETERMINISTIC_PASS":
        return "VECTOR_CANDIDATE_DETERMINISTIC_PASS"
    return "SEMANTIC_BEST_UNVERIFIED"


def _refine_frames(
    source: SourceRun,
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
    refined = source.sft_mixed.copy(deep=True)
    by_pair = {
        pair_id: int(index)
        for index, pair_id in zip(refined.index, refined.pair_id, strict=True)
        if pair_id in results
    }
    route_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for pair_id in sorted(results):
        result = results[pair_id]
        index = by_pair[pair_id]
        original = refined.loc[index].to_dict()
        paper_id = original["paper_id"]
        candidate_rows.extend(_compact_candidate_rows(pair_id, paper_id, result))
        original_bundle = _parse_json(
            original["evidence_json"], f"source evidence for {pair_id}", list,
        )
        candidate_bundle = list(result["bundle"])
        has_new_bundle = (
            bool(candidate_bundle)
            and builder.canonical_json(candidate_bundle)
            != builder.canonical_json(original_bundle)
        )
        refined_route = _semantic_route_name(result)
        action = "NO_BUNDLE"
        if candidate_bundle and not has_new_bundle:
            action = "UNCHANGED_IDENTICAL_BUNDLE"
        if has_new_bundle:
            source_row = _parse_json(
                original["source_row_json"], f"source row for {pair_id}", dict,
            )
            source_object = SimpleNamespace(**source_row)
            core_response = funnel.assistant_turn(source_object)
            provenance_enabled = bool(
                original.get("citation_status")
                or original.get("citation_metadata_source")
            )
            rendered = funnel.render_open(
                source_object, candidate_bundle,
                original["paper_context"],
                citation_label=original.get("citation_label", ""),
                paper_context=original.get("paper_context_json", ""),
                verification_tier=(
                    "UNVERIFIED" if provenance_enabled else None
                ),
            )
            messages = rendered["messages"]
            response = funnel.clean(messages[1]["content"])
            if response != core_response and not response.startswith(
                f"{core_response}\n\nProvenance:",
            ):
                raise ValueError(
                    f"semantic refinement changed the core target for {pair_id}",
                )
            report_reason = funnel.clean(result["report"].get("reason"))
            replacement = {
                "example_id": f"{pair_id}:open",
                "mode": "OPEN",
                "assignment": "OPEN",
                "support_route": refined_route,
                "prompt": funnel.clean(messages[0]["content"]),
                "response": response,
                "messages": messages,
                "messages_json": builder.canonical_json(messages),
                "descriptor": "",
                "evidence_json": builder.canonical_json(candidate_bundle),
                "support_report_json": builder.canonical_json(result["report"]),
                "verification_tier": "UNVERIFIED",
                "inclusion_source": "SEMANTIC_REFINED",
                "reason_code": "QUARANTINE_UNVERIFIED",
                "reason_detail": report_reason or "Semantic candidate remains unverified.",
                "token_estimate": builder.estimate_tokens(
                    messages[0]["content"], response,
                ),
            }
            for column, value in replacement.items():
                refined.at[index, column] = value
            action = (
                "REFINED_DETERMINISTIC_PASS"
                if result["route"] == "VECTOR_CANDIDATE_DETERMINISTIC_PASS"
                else "REFINED_BEST_UNVERIFIED"
            )

        retrieval_metadata = {
            "query_truncation_count": result["query_truncation_count"],
            "long_query_split_count": result["long_query_split_count"],
            "retrieval_limitations": result["retrieval_limitations"],
            "candidate_hit_count": len(result["hits"]),
        }
        route_rows.append({
            "pair_id": pair_id,
            "paper_id": paper_id,
            "family_id": original["family_id"],
            "split": original["split"],
            "pair_type": original["pair_type"],
            "example_id": funnel.clean(refined.at[index, "example_id"]),
            "source_support_route": original["support_route"],
            "semantic_route": result["route"],
            "refined_support_route": refined_route if has_new_bundle else original["support_route"],
            "refinement_action": action,
            "original_evidence_json": builder.canonical_json(original_bundle),
            "candidate_bundle_json": builder.canonical_json(candidate_bundle),
            "original_support_report_json": original["support_report_json"],
            "support_report_json": builder.canonical_json(result["report"]),
            "retrieval_metadata_json": builder.canonical_json(retrieval_metadata),
            "source_row_json": original["source_row_json"],
        })

    route_frame = pd.DataFrame(route_rows, columns=SEMANTIC_ROUTE_COLUMNS)
    for column in SEMANTIC_ROUTE_COLUMNS:
        route_frame[column] = route_frame[column].map(funnel.clean)
    candidate_frame = pd.DataFrame(
        candidate_rows, columns=SEMANTIC_CANDIDATE_COLUMNS,
    )
    if candidate_frame.empty:
        candidate_frame = pd.DataFrame({
            field.name: pd.Series(dtype=(
                "bool" if pa.types.is_boolean(field.type)
                else "int64" if pa.types.is_integer(field.type)
                else "object"
            ))
            for field in SEMANTIC_CANDIDATE_SCHEMA
        })
    refined["token_estimate"] = pd.to_numeric(
        refined["token_estimate"], errors="raise",
    ).astype("int64")
    return {
        "sft_mixed": refined.loc[:, list(builder.SFT_COLUMNS)],
        "semantic_routes": route_frame,
        "semantic_candidates": candidate_frame.loc[:, list(SEMANTIC_CANDIDATE_COLUMNS)],
    }


def _validate_refinement(
    source: SourceRun, frames: Mapping[str, pd.DataFrame],
    result_ids: set[str],
) -> None:
    refined = frames["sft_mixed"]
    base = source.sft_mixed
    if list(refined.columns) != list(builder.SFT_COLUMNS):
        raise ValueError("refined sft_mixed changed its column contract")
    if len(refined) != len(base) or refined.pair_id.tolist() != base.pair_id.tolist():
        raise ValueError("semantic refinement changed row count or pair ordering")
    if refined.example_id.duplicated().any():
        raise ValueError("semantic refinement produced a duplicate example_id")
    strict_ids = set(source.sft_examples.example_id)
    base_by_id = base.set_index("example_id", drop=False)
    refined_by_id = refined.set_index("example_id", drop=False)
    if not strict_ids.issubset(set(refined_by_id.index)):
        raise ValueError("semantic refinement removed a strict SFT example")
    for example_id in strict_ids:
        if builder.canonical_json(base_by_id.loc[example_id].to_dict()) != builder.canonical_json(
            refined_by_id.loc[example_id].to_dict(),
        ):
            raise ValueError("semantic refinement modified a strict SFT row")
    eligible = set(source.eligible_pair_ids)
    for index in base.index:
        before = base.loc[index].to_dict()
        after = refined.loc[index].to_dict()
        if before["pair_id"] not in result_ids and builder.canonical_json(before) != builder.canonical_json(after):
            raise ValueError("semantic refinement modified a pair without a route result")
        if before["pair_id"] in eligible and after["verification_tier"] != "UNVERIFIED":
            raise ValueError("semantic refinement promoted an unverified row")
        if before["pair_id"] in result_ids:
            source_row = _parse_json(
                after["source_row_json"],
                f"refined source row for {after['pair_id']}",
                dict,
            )
            core = funnel.assistant_turn(SimpleNamespace(**source_row))
            response = funnel.clean(after["response"])
            if response != core and not response.startswith(f"{core}\n\nProvenance:"):
                raise ValueError("semantic refinement changed an assistant core target")
    routes = frames["semantic_routes"]
    if routes.pair_id.duplicated().any() or set(routes.pair_id) != result_ids:
        raise ValueError("semantic route audit is not one-to-one with input results")


def _schema_hash(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _verify_derived_run(run_dir: Path, marker: Mapping[str, Any]) -> None:
    expected = {f"{name}.parquet" for name in OUTPUT_SCHEMAS}
    files = marker.get("files")
    if not isinstance(files, Mapping) or set(files) != expected:
        raise ValueError("semantic generation has an incomplete file manifest")
    for filename in sorted(expected):
        name = filename.removesuffix(".parquet")
        path = run_dir / filename
        metadata = files[filename]
        if not path.is_file() or builder._file_hash(path) != metadata.get("sha256"):
            raise ValueError(f"semantic generation file is missing or corrupt: {filename}")
        schema = pq.read_schema(path)
        expected_schema = OUTPUT_SCHEMAS[name]
        if not schema.equals(expected_schema, check_metadata=False):
            raise ValueError(f"semantic generation has wrong schema: {filename}")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != int(metadata.get("rows", -1)):
            raise ValueError(f"semantic generation row count is corrupt: {filename}")
        if metadata.get("columns") != expected_schema.names:
            raise ValueError(f"semantic generation columns are corrupt: {filename}")
        if metadata.get("schema_sha256") != _schema_hash(expected_schema):
            raise ValueError(f"semantic generation schema hash is corrupt: {filename}")


def _source_is_still_latest(source: SourceRun) -> bool:
    pointer_path = source.training_root / "LATEST.json"
    return (
        pointer_path.is_file()
        and builder._file_hash(pointer_path) == source.pointer_sha256
    )


def _write_semantic_latest(root: Path, marker: Mapping[str, Any]) -> None:
    pointer = {
        "format": OUTPUT_FORMAT,
        "run_id": marker["run_id"],
        "directory": f"semantic_runs/{marker['run_id']}",
        "identity_sha256": marker["identity_sha256"],
        "source_builder_run_id": marker["source_builder_run_id"],
        "source_builder_identity_sha256": marker[
            "source_builder_identity_sha256"
        ],
        "files": marker["files"],
        "training_ready": True,
    }
    temporary = root / f".SEMANTIC_LATEST-{uuid.uuid4().hex}.tmp"
    temporary.write_text(
        json.dumps(pointer, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, root / "SEMANTIC_LATEST.json")


def _publish(
    frames: Mapping[str, pd.DataFrame], root: Path, manifest: dict[str, Any],
    *, advance_latest: bool,
) -> tuple[Path, dict[str, Any], bool]:
    root = root.resolve()
    runs = root / "semantic_runs"
    staging = root / ".semantic-staging"
    runs.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    final = (runs / manifest["run_id"]).resolve()
    if final.parent != runs.resolve():
        raise ValueError("semantic run path escapes semantic_runs")
    if final.exists():
        marker = _read_json(final / "_SUCCESS.json")
        if marker.get("identity_sha256") != manifest["identity_sha256"]:
            raise ValueError("semantic run ID collision")
        _verify_derived_run(final, marker)
        if advance_latest:
            _write_semantic_latest(root, marker)
        return final, marker, advance_latest

    stage = (staging / f"{manifest['run_id']}-{uuid.uuid4().hex}").resolve()
    if stage.parent != staging.resolve():
        raise ValueError("semantic staging path escapes staging root")
    stage.mkdir()
    try:
        files = {}
        for name, schema in OUTPUT_SCHEMAS.items():
            path = stage / f"{name}.parquet"
            table = pa.Table.from_pylist(
                frames[name].to_dict("records"), schema=schema,
            )
            pq.write_table(table, path, compression="zstd")
            files[path.name] = {
                "sha256": builder._file_hash(path),
                "rows": len(frames[name]),
                "columns": schema.names,
                "schema_sha256": _schema_hash(schema),
            }
        marker = {**manifest, "files": files}
        (stage / "_SUCCESS.json").write_text(
            json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, final)
        _verify_derived_run(final, marker)
        if advance_latest:
            _write_semantic_latest(root, marker)
        return final, marker, advance_latest
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def refine_sft_mixed(
    config: RefineConfig,
    semantic_results: Mapping[str, Mapping[str, Any]] | pd.DataFrame | str | Path,
    *,
    retrieval_config: Mapping[str, Any],
    source_run: SourceRun | None = None,
    publish: bool = True,
    advance_latest: bool | None = None,
) -> RefineOutcome:
    """Refine mixed SFT evidence and optionally publish an immutable generation.

    ``semantic_results`` is normally the ``VECTOR_AUDIT`` mapping emitted by
    the quarantine notebook.  Publishing a partial run is safe: it creates an
    immutable audit generation but does not update ``SEMANTIC_LATEST.json``.
    """

    config.validate()
    source = source_run or load_source_run(config.training_root)
    if source.training_root != config.training_root.resolve():
        raise ValueError("source_run belongs to a different training root")
    if not isinstance(retrieval_config, Mapping):
        raise TypeError("retrieval_config must be a mapping")
    for key in ("model_id", "model_revision"):
        if not funnel.clean(retrieval_config.get(key)):
            raise ValueError(f"retrieval_config must record {key}")

    raw_routes = _route_mapping(semantic_results)
    results = _normalize_results(raw_routes, source, config.markdown_dir.resolve())
    result_ids = set(results)
    eligible_ids = set(source.eligible_pair_ids)
    complete_coverage = result_ids == eligible_ids
    semantic_limited = retrieval_config.get("paper_limit") is not None
    training_ready = (
        source.source_complete
        and complete_coverage
        and not config.preview
        and not semantic_limited
    )
    if advance_latest is None:
        advance_latest = training_ready
    if advance_latest and not training_ready:
        raise ValueError(
            "SEMANTIC_LATEST cannot advance from a preview, partial semantic "
            "coverage, or paper-limited builder generation",
        )
    if advance_latest and not _source_is_still_latest(source):
        raise RuntimeError("builder LATEST changed after the source run was loaded")

    frames = _refine_frames(source, results)
    _validate_refinement(source, frames, result_ids)
    route_hash_payload = {
        pair_id: {
            "route": result["route"],
            "bundle": result["bundle"],
            "report": result["report"],
            "hits": [
                {
                    "score": hit.get("score"),
                    "candidate_origin": hit.get("candidate_origin"),
                    "evidence_token_count": hit.get("evidence_token_count"),
                    "chunk_index": hit.get("chunk_index"),
                    "span": hit["span"],
                }
                for hit in result["hits"]
            ],
            "query_truncation_count": result["query_truncation_count"],
            "long_query_split_count": result["long_query_split_count"],
            "retrieval_limitations": result["retrieval_limitations"],
        }
        for pair_id, result in sorted(results.items())
    }
    identity = {
        "format": OUTPUT_FORMAT,
        "refiner_version": REFINER_VERSION,
        "source_builder_run_id": source.marker["run_id"],
        "source_builder_identity_sha256": source.marker["identity_sha256"],
        "source_sft_mixed_sha256": source.marker["files"]["sft_mixed.parquet"]["sha256"],
        "source_quarantine_sha256": source.marker["files"]["quarantine.parquet"]["sha256"],
        "semantic_results_sha256": hashlib.sha256(
            builder.canonical_json(route_hash_payload).encode("utf-8"),
        ).hexdigest(),
        "retrieval_config": builder._jsonable(dict(retrieval_config)),
        "module_sha256": {
            "refiner": builder._file_hash(Path(__file__)),
            "semantic": builder._file_hash(Path(semantic.__file__)),
            "dataset": builder._file_hash(Path(funnel.__file__)),
            "builder": builder._file_hash(Path(builder.__file__)),
        },
        "preview": bool(config.preview or semantic_limited),
    }
    identity_sha = hashlib.sha256(
        builder.canonical_json(identity).encode("utf-8"),
    ).hexdigest()
    run_id = identity_sha[:24]
    actions = frames["semantic_routes"].refinement_action.value_counts().to_dict()
    manifest = {
        **identity,
        "identity_sha256": identity_sha,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "eligible_pairs": len(eligible_ids),
        "semantic_result_pairs": len(result_ids),
        "complete_coverage": complete_coverage,
        "source_builder_complete": source.source_complete,
        "training_ready": training_ready,
        "latest_advanced": bool(advance_latest),
        "refinement_actions": {str(key): int(value) for key, value in actions.items()},
        "output_counts": {name: len(frame) for name, frame in frames.items()},
        "contract": {
            "verification": "all refined examples remain UNVERIFIED",
            "strict_rows": "preserved value-identically",
            "source_mutation": "none; this is a derived immutable generation",
            "partial_runs": "audit-only and forbidden from SEMANTIC_LATEST",
        },
    }
    run_dir = None
    latest_advanced = False
    if publish:
        run_dir, manifest, latest_advanced = _publish(
            frames, config.resolved_output_root(), manifest,
            advance_latest=bool(advance_latest),
        )
    return RefineOutcome(
        run_id=run_id,
        run_dir=run_dir,
        source_builder_run_id=source.marker["run_id"],
        source_builder_identity_sha256=source.marker["identity_sha256"],
        frames=frames,
        manifest=manifest,
        complete_coverage=complete_coverage,
        training_ready=training_ready,
        latest_advanced=latest_advanced,
    )
