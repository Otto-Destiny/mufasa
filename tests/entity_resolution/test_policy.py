from pathlib import Path

import pytest
import yaml

from scripts.entity_resolution.contracts import ContractError
from scripts.entity_resolution.policy import load_policy


def _policy_data():
    path = Path(__file__).resolve().parents[2] / "scripts/entity_resolution/policies/mufasa-v1.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_policy_rejects_string_boolean(workspace_tmp):
    data = _policy_data()
    data["matching"]["type_overrides"]["CHEMICAL"]["self_seed"] = "false"
    path = workspace_tmp / "bad-policy.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ContractError, match="YAML boolean"):
        load_policy(path)


def test_policy_rejects_missing_type_override_and_bad_qualifier_alias(workspace_tmp):
    data = _policy_data()
    del data["matching"]["type_overrides"]["PLACE"]
    path = workspace_tmp / "missing-type.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ContractError, match="explicitly contain every entity type"):
        load_policy(path)

    data = _policy_data()
    data["qualifiers"]["aliases"]["TYPO"] = "NOT_A_KIND"
    path = workspace_tmp / "bad-alias.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ContractError, match="target unknown kinds"):
        load_policy(path)

