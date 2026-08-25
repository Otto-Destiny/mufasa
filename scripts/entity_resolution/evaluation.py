"""Separate, auditable entity-resolution evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from .contracts import DecisionStatus, ResolutionRun


AUTOMATIC_METHODS = frozenset(
    {
        "EXACT_AUTHORITY_ID",
        "TRUSTED_ALIAS",
        "NORMALIZED_PRIMARY",
        "EXACT_BOOTSTRAP_GROUP",
        "CLEAN_SINGLETON_BOOTSTRAP",
        "EXACT_INSTANCE_GROUP",
        "LEXICAL_CALIBRATED",
        "EMBEDDING_CALIBRATED",
    }
)


@dataclass(frozen=True)
class EvaluationReport:
    metrics: Mapping[str, float | int | None]
    by_type: tuple[Mapping[str, Any], ...]
    errors: tuple[Mapping[str, Any], ...]
    release_warnings: tuple[str, ...]


def evaluate_run(
    run: ResolutionRun,
    *,
    mention_gold: pd.DataFrame | None = None,
    pair_gold: pd.DataFrame | None = None,
) -> EvaluationReport:
    """Evaluate only labels supplied by reviewed resolution gold artifacts."""

    decisions = {item.mention_id: item for item in run.decisions}
    mentions = {item.mention_id: item for item in run.mentions}
    metrics: dict[str, float | int | None] = {
        "input_mentions": len(run.decisions),
        "automatic_decisions": sum(
            item.status == DecisionStatus.MATCHED and item.method.value in AUTOMATIC_METHODS
            for item in run.decisions
        ),
        "human_override_decisions": sum(
            item.status == DecisionStatus.MATCHED and item.method.value == "HUMAN_OVERRIDE"
            for item in run.decisions
        ),
        "uncommitted_proposal_decisions": sum(
            item.status in {DecisionStatus.NEW_CONCEPT_PROPOSED, DecisionStatus.NEW_INSTANCE_PROPOSED}
            for item in run.decisions
        ),
        "review_required": sum(item.status == DecisionStatus.REVIEW_REQUIRED for item in run.decisions),
        "unresolved": sum(item.status == DecisionStatus.UNRESOLVED for item in run.decisions),
        "invalid_input": sum(item.status == DecisionStatus.INVALID_INPUT for item in run.decisions),
        "candidates_generated": run.generated_candidate_count,
        "candidate_rows_retained": len(run.candidates),
    }
    warnings: list[str] = []
    errors: list[dict[str, Any]] = []
    by_type: list[dict[str, Any]] = []

    if metrics["automatic_decisions"] == 0:
        warnings.append("ZERO_AUTOMATIC_DECISIONS: precision is undefined and the resolver fails useful-coverage review")
    if metrics["uncommitted_proposal_decisions"]:
        warnings.append(
            "DRY_RUN_PROPOSALS_PRESENT: precision and resolved-coverage metrics exclude proposed IDs until an in-memory preview or explicit commit"
        )

    if mention_gold is not None:
        required = {"mention_id", "gold_status", "gold_concept_id", "gold_instance_id", "gold_resolvable"}
        missing = required - set(mention_gold.columns)
        if missing:
            raise ValueError(f"mention gold is missing columns {sorted(missing)}")
        rows = []
        for gold in mention_gold.to_dict("records"):
            mention_id = str(gold["mention_id"])
            decision = decisions.get(mention_id)
            if decision is None:
                errors.append({"mention_id": mention_id, "error": "MISSING_DECISION"})
                continue
            expected_concept = _nullable(gold["gold_concept_id"])
            expected_instance = _nullable(gold["gold_instance_id"])
            expected_status = str(gold["gold_status"])
            correct = (
                decision.status.value == expected_status
                and decision.concept_id == expected_concept
                and decision.instance_id == expected_instance
            )
            automatic = (
                decision.status == DecisionStatus.MATCHED
                and decision.method.value in AUTOMATIC_METHODS
            )
            rows.append(
                {
                    "mention_id": mention_id,
                    "entity_type": mentions[mention_id].entity_type.value if mention_id in mentions else "INVALID",
                    "correct": correct,
                    "automatic": automatic,
                    "resolvable": bool(gold["gold_resolvable"]),
                    "resolved": decision.status == DecisionStatus.MATCHED,
                }
            )
            if not correct:
                errors.append(
                    {
                        "mention_id": mention_id,
                        "error": "MENTION_MISMATCH",
                        "expected_status": expected_status,
                        "actual_status": decision.status.value,
                        "expected_concept_id": expected_concept,
                        "actual_concept_id": decision.concept_id,
                        "expected_instance_id": expected_instance,
                        "actual_instance_id": decision.instance_id,
                    }
                )
        scored = pd.DataFrame(rows)
        if not scored.empty:
            automatic = scored[scored["automatic"]]
            metrics["automatic_precision"] = float(automatic["correct"].mean()) if len(automatic) else None
            if len(automatic):
                low, high = wilson_interval(int(automatic["correct"].sum()), len(automatic))
                metrics["automatic_precision_wilson_95_low"] = low
                metrics["automatic_precision_wilson_95_high"] = high
            else:
                metrics["automatic_precision_wilson_95_low"] = None
                metrics["automatic_precision_wilson_95_high"] = None
            resolvable = scored[scored["resolvable"]]
            metrics["resolvable_mention_coverage"] = float(resolvable["resolved"].mean()) if len(resolvable) else None
            for entity_type, group in scored.groupby("entity_type", sort=True):
                auto = group[group["automatic"]]
                resolvable_group = group[group["resolvable"]]
                by_type.append(
                    {
                        "entity_type": entity_type,
                        "gold_mentions": len(group),
                        "automatic_precision": float(auto["correct"].mean()) if len(auto) else None,
                        "resolvable_coverage": float(resolvable_group["resolved"].mean()) if len(resolvable_group) else None,
                    }
                )

    if pair_gold is not None:
        required = {"mention_id_a", "mention_id_b", "gold_relation"}
        missing = required - set(pair_gold.columns)
        if missing:
            raise ValueError(f"pair gold is missing columns {sorted(missing)}")
        pair_total = 0
        pair_correct = 0
        same_total = 0
        same_found = 0
        false_merges = 0
        for row in pair_gold.to_dict("records"):
            left_id, right_id = str(row["mention_id_a"]), str(row["mention_id_b"])
            left, right = decisions.get(left_id), decisions.get(right_id)
            if left is None or right is None:
                errors.append({"mention_id_a": left_id, "mention_id_b": right_id, "error": "PAIR_DECISION_MISSING"})
                continue
            relation = str(row["gold_relation"])
            predicted = _pair_relation(left, right)
            pair_total += 1
            pair_correct += predicted == relation
            if relation == "SAME_CONCEPT":
                same_total += 1
                same_found += predicted == relation
            if relation == "DIFFERENT_CONCEPT" and predicted == "SAME_CONCEPT":
                false_merges += 1
                errors.append({"mention_id_a": left_id, "mention_id_b": right_id, "error": "FALSE_CONCEPT_MERGE"})
        metrics["pairwise_accuracy"] = pair_correct / pair_total if pair_total else None
        metrics["confirmed_same_concept_recall"] = same_found / same_total if same_total else None
        metrics["critical_false_merges"] = false_merges

    # Candidate recall is measured only where a gold target ID exists.
    if mention_gold is not None:
        candidates = defaultdict_set(run)
        expected_targets = []
        for row in mention_gold.to_dict("records"):
            target = _nullable(row["gold_instance_id"]) or _nullable(row["gold_concept_id"])
            if target:
                expected_targets.append((str(row["mention_id"]), target))
        metrics["candidate_recall"] = (
            sum(target in candidates.get(mention_id, set()) for mention_id, target in expected_targets)
            / len(expected_targets)
            if expected_targets
            else None
        )

    return EvaluationReport(
        metrics=metrics,
        by_type=tuple(by_type),
        errors=tuple(errors),
        release_warnings=tuple(warnings),
    )


def defaultdict_set(run: ResolutionRun) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for item in run.candidates:
        values.setdefault(item.mention_id, set()).add(item.target_id)
    for decision in run.decisions:
        if decision.concept_id:
            values.setdefault(decision.mention_id, set()).add(decision.concept_id)
        if decision.instance_id:
            values.setdefault(decision.mention_id, set()).add(decision.instance_id)
    return values


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires total > 0")
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _pair_relation(left: Any, right: Any) -> str:
    if left.instance_id and left.instance_id == right.instance_id:
        return "SAME_INSTANCE"
    if left.instance_id and left.concept_id and left.concept_id == right.concept_id and not right.instance_id:
        return "INSTANCE_OF"
    if right.instance_id and right.concept_id and right.concept_id == left.concept_id and not left.instance_id:
        return "INSTANCE_OF"
    if left.concept_id and left.concept_id == right.concept_id:
        return "SAME_CONCEPT"
    if left.status in {DecisionStatus.REVIEW_REQUIRED, DecisionStatus.UNRESOLVED, DecisionStatus.INVALID_INPUT}:
        return "INSUFFICIENT_EVIDENCE"
    if right.status in {DecisionStatus.REVIEW_REQUIRED, DecisionStatus.UNRESOLVED, DecisionStatus.INVALID_INPUT}:
        return "INSUFFICIENT_EVIDENCE"
    return "DIFFERENT_CONCEPT"


def _nullable(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None
