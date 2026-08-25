import pytest

from scripts.entity_resolution.authorities import AuthorityRecord, AuthoritySnapshot
from scripts.entity_resolution.contracts import ContractError, EntityType


def test_authority_indexes_are_cached_for_large_snapshot():
    records = tuple(
        AuthorityRecord("NCBI", str(index), f"Taxon {index}", EntityType.ORGANISM, ())
        for index in range(1_000)
    )
    snapshot = AuthoritySnapshot("v1", "a" * 64, records, (), (("NCBI", "public"),))
    assert snapshot.by_identifier is snapshot.by_identifier
    assert snapshot.get("NCBI", "999").preferred_label == "Taxon 999"


def test_authority_licences_are_complete_and_unique():
    record = AuthorityRecord("NCBI", "1", "Taxon one", EntityType.ORGANISM, ())
    with pytest.raises(ContractError, match="licence authorities"):
        AuthoritySnapshot("v1", "a" * 64, (record,), (), ())
    with pytest.raises(ContractError, match="exactly one licence"):
        AuthoritySnapshot(
            "v1", "a" * 64, (record,), (), (("NCBI", "public"), ("NCBI", "other"))
        )


def test_crosswalk_endpoints_must_exist_match_type_and_be_unique():
    organism = AuthorityRecord("NCBI", "1", "Taxon one", EntityType.ORGANISM, ())
    chemical = AuthorityRecord("CHEBI", "2", "Compound", EntityType.CHEMICAL, ())
    licences = (("NCBI", "public"), ("CHEBI", "public"))
    with pytest.raises(ContractError, match="incompatible types"):
        AuthoritySnapshot(
            "v1", "a" * 64, (organism, chemical),
            (("NCBI", "1", "CHEBI", "2"),), licences,
        )
    other = AuthorityRecord("GBIF", "9", "Taxon one", EntityType.ORGANISM, ())
    with pytest.raises(ContractError, match="duplicate/reversed"):
        AuthoritySnapshot(
            "v1", "a" * 64, (organism, other),
            (("NCBI", "1", "GBIF", "9"), ("GBIF", "9", "NCBI", "1")),
            (("NCBI", "public"), ("GBIF", "public")),
        )
