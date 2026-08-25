import json
from pathlib import Path

from conftest import concept, mention, snapshot
from scripts.entity_resolution.contracts import DecisionStatus, EntityType
from scripts.entity_resolution.pipeline import resolve_batch


def test_water_identity_golden_cases_do_not_head_noun_collapse(policy):
    cases = json.loads((Path(__file__).parent / "golden" / "water_identity_cases.json").read_text())
    for index, case in enumerate(cases):
        registry = snapshot(concept("CON-LEFT", case["left"], EntityType.ENVIRONMENTAL_FEATURE))
        value = mention(f"M{index}", case["right"], EntityType.ENVIRONMENTAL_FEATURE)
        result = resolve_batch([value], registry, policy).run.decisions[0]
        if case["same_identity"]:
            # Homonym-prone environmental terms require a reviewed/exact policy;
            # they must at least point to the candidate, never to a different node.
            assert result.status in {DecisionStatus.MATCHED, DecisionStatus.REVIEW_REQUIRED}
        else:
            assert result.concept_id is None

