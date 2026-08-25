"""Command-line interface for validation, dry runs, commits, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .adapters import load_mufasa_inputs
from .audit import run_summary
from .authorities import load_authority_snapshot
from .evaluation import evaluate_run
from .io import (
    atomic_write_json,
    canonical_json_hash,
    load_registry_snapshot,
    load_resolution_run,
    sha256_file,
    write_registry_snapshot,
    write_resolution_run,
)
from .matching import PrecomputedEmbeddingRecall
from .pipeline import commit_resolution_run, preflight_capabilities, resolve_batch
from .policy import load_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mufasa-entity-resolution")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate inputs and active capabilities")
    _common_inputs(validate)
    resolve = subparsers.add_parser("resolve", help="write a mutation-free resolution dry run")
    _common_inputs(resolve)
    resolve.add_argument("--output", required=True)
    resolve.add_argument("--workers", type=int, default=1)
    commit = subparsers.add_parser("commit", help="reproduce a dry run and commit accepted proposals")
    _common_inputs(commit)
    commit.add_argument("--output", required=True, help="committed resolution-run output directory")
    commit.add_argument("--registry-output", required=True, help="new immutable registry snapshot directory")
    commit.add_argument("--approve", action="append", default=[], help="additional reviewed proposal ID")
    commit.add_argument("--workers", type=int, default=1)
    evaluate = subparsers.add_parser("evaluate", help="evaluate a saved run against reviewed gold")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--mention-gold")
    evaluate.add_argument("--pair-gold")
    evaluate.add_argument("--output", required=True)
    return parser


def _common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--extraction-dir", required=True)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--policy")
    parser.add_argument("--registry")
    parser.add_argument("--authorities")
    parser.add_argument("--authority-hints")
    parser.add_argument("--embedding-manifest")
    parser.add_argument("--mention-vectors")
    parser.add_argument("--target-vectors")


def _load_common(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_policy(args.policy)
    authority = load_authority_snapshot(args.authorities)
    registry = load_registry_snapshot(args.registry, authority_snapshot=authority if authority.records else None)
    inputs = load_mufasa_inputs(
        args.extraction_dir,
        args.documents,
        policy,
        authority_hints_path=args.authority_hints,
    )
    embedding = _load_embedding(args)
    return {"policy": policy, "authority": authority, "registry": registry, "inputs": inputs, "embedding": embedding}


def _load_embedding(args: argparse.Namespace) -> PrecomputedEmbeddingRecall | None:
    supplied = [args.embedding_manifest, args.mention_vectors, args.target_vectors]
    if not any(supplied):
        return None
    if not all(supplied):
        raise SystemExit("embedding requires --embedding-manifest, --mention-vectors, and --target-vectors together")
    manifest = json.loads(Path(args.embedding_manifest).read_text(encoding="utf-8"))
    required_manifest = {
        "model_id", "model_hash", "mention_vectors_sha256", "target_vectors_sha256",
        "vector_set_fingerprint",
    }
    missing_manifest = required_manifest - set(manifest)
    if missing_manifest:
        raise SystemExit(f"embedding manifest is missing {sorted(missing_manifest)}")
    mention_hash = sha256_file(args.mention_vectors)
    target_hash = sha256_file(args.target_vectors)
    if mention_hash != str(manifest["mention_vectors_sha256"]):
        raise SystemExit("mention vector Parquet SHA-256 does not match embedding manifest")
    if target_hash != str(manifest["target_vectors_sha256"]):
        raise SystemExit("target vector Parquet SHA-256 does not match embedding manifest")
    descriptor = {
        "model_id": str(manifest["model_id"]),
        "model_hash": str(manifest["model_hash"]),
        "mention_vectors_sha256": mention_hash,
        "target_vectors_sha256": target_hash,
    }
    fingerprint = canonical_json_hash(descriptor)
    if fingerprint != str(manifest["vector_set_fingerprint"]):
        raise SystemExit("embedding vector_set_fingerprint does not match pinned artifacts/model")
    mention_frame = pd.read_parquet(args.mention_vectors)
    target_frame = pd.read_parquet(args.target_vectors)
    if not {"mention_id", "vector"} <= set(mention_frame.columns):
        raise SystemExit("mention vectors require mention_id and vector columns")
    if not {"target_id", "vector"} <= set(target_frame.columns):
        raise SystemExit("target vectors require target_id and vector columns")
    if mention_frame["mention_id"].isna().any() or mention_frame["mention_id"].astype(str).duplicated().any():
        raise SystemExit("mention vector IDs must be non-null and unique")
    if target_frame["target_id"].isna().any() or target_frame["target_id"].astype(str).duplicated().any():
        raise SystemExit("target vector IDs must be non-null and unique")
    return PrecomputedEmbeddingRecall(
        dict(zip(mention_frame["mention_id"].astype(str), mention_frame["vector"])),
        dict(zip(target_frame["target_id"].astype(str), target_frame["vector"])),
        model_id=str(manifest["model_id"]),
        model_hash=str(manifest["model_hash"]),
        mention_vectors_hash=mention_hash,
        target_vectors_hash=target_hash,
        vector_set_fingerprint=fingerprint,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        run = load_resolution_run(args.run)
        report = evaluate_run(
            run,
            mention_gold=_gold(args.mention_gold),
            pair_gold=_gold(args.pair_gold),
        )
        atomic_write_json(report, args.output)
        print(json.dumps(report.metrics, indent=2, sort_keys=True))
        return 0
    common = _load_common(args)
    inputs = common["inputs"]
    if args.command == "validate":
        capabilities = preflight_capabilities(
            common["policy"], common["registry"], common["authority"], inputs.authority_hints,
            common["embedding"],
        )
        capabilities = _with_extraction_provenance(capabilities, inputs)
        result = {
            "valid_mentions": len(inputs.mentions),
            "adapter_invalid_mentions": len(inputs.invalid_mentions),
            "input_fingerprint": inputs.input_fingerprint,
            "capabilities": capabilities,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    execution = resolve_batch(
        inputs.mentions,
        common["registry"],
        common["policy"],
        authority_snapshot=common["authority"],
        authority_hints=inputs.authority_hints,
        invalid_mentions=inputs.invalid_mentions,
        input_fingerprint=inputs.input_fingerprint,
        embedding=common["embedding"],
        workers=args.workers,
    )
    capabilities = _with_extraction_provenance(execution.capability_manifest, inputs)
    if args.command == "resolve":
        write_resolution_run(
            execution.run,
            args.output,
            conflicts=execution.conflicts,
            capability_manifest=capabilities,
        )
        print(json.dumps(run_summary(execution.run), indent=2, sort_keys=True))
        return 0
    committed = commit_resolution_run(
        execution,
        common["registry"],
        common["policy"],
        authority_snapshot=common["authority"],
        approved_proposal_ids=args.approve,
    )
    write_resolution_run(
        committed.run,
        args.output,
        conflicts=execution.conflicts,
        capability_manifest=capabilities,
        registry_diff_value=committed.diff,
    )
    write_registry_snapshot(
        committed.registry,
        args.registry_output,
        authority_snapshot=common["authority"] if common["authority"].records else None,
        manifest_extra={
            "policy_version": common["policy"].version,
            "policy_hash": common["policy"].content_hash,
            "authority_manifest_hash": common["authority"].manifest_hash,
            "source_run_id": execution.run.run_id,
        },
    )
    print(json.dumps({**run_summary(committed.run), "registry_version": committed.registry.version}, indent=2, sort_keys=True))
    return 0


def _with_extraction_provenance(
    capabilities: Mapping[str, Any], inputs: Any
) -> dict[str, Any]:
    value = dict(capabilities)
    value["extraction_input"] = {
        "generation_id": inputs.extraction_generation_id,
        "source_fingerprint": inputs.source_fingerprint,
        "settings_hash": inputs.settings_hash,
        "schema_version": inputs.extraction_schema_version,
        "prompt_version": inputs.prompt_version,
    }
    return value


def _gold(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    source = Path(path)
    return pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
