import pandas as pd

from conftest import concept, mention, snapshot
from scripts.entity_resolution.contracts import ConstraintType, EntityType
from scripts.entity_resolution.evaluation import evaluate_run
from scripts.entity_resolution.pipeline import resolve_batch
from scripts.entity_resolution.registry import record_human_constraint


def _gold(mention_id: str, concept_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mention_id": mention_id,
                "gold_status": "MATCHED",
                "gold_concept_id": concept_id,
                "gold_instance_id": None,
                "gold_resolvable": True,
            }
        ]
    )


def test_human_override_does_not_inflate_automatic_precision(policy):
    registry = snapshot(concept("CON-N", "nitrate", EntityType.CHEMICAL))
    registry = record_human_constraint(
        registry,
        "M1",
        "CON-N",
        ConstraintType.MUST_LINK,
        "RUN-R",
        "reviewer",
        "gold adjudication",
    ).snapshot
    run = resolve_batch([mention("M1", "local nitrate", EntityType.CHEMICAL)], registry, policy).run
    report = evaluate_run(run, mention_gold=_gold("M1", "CON-N"))
    assert report.metrics["human_override_decisions"] == 1
    assert report.metrics["automatic_decisions"] == 0
    assert report.metrics["automatic_precision"] is None


def test_candidate_recall_includes_compact_exact_winner(policy):
    registry = snapshot(concept("CON-N", "nitrate", EntityType.CHEMICAL))
    run = resolve_batch([mention("M1", "nitrate", EntityType.CHEMICAL)], registry, policy).run
    assert run.candidates == ()
    report = evaluate_run(run, mention_gold=_gold("M1", "CON-N"))
    assert report.metrics["candidate_recall"] == 1.0

