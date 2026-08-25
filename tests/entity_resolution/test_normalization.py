from scripts.entity_resolution.contracts import EntityType, Qualifier
from scripts.entity_resolution.normalization import primary_key, normalized_text, qualifier_signature


def test_unicode_is_normalized_without_ascii_folding():
    assert normalized_text("  Ìbàdàn\u00a0 ") == "ìbàdàn"
    assert normalized_text("Ìbàdàn") != normalized_text("Ibadan")


def test_chemical_formula_case_is_identity_relevant():
    assert primary_key("CO", EntityType.CHEMICAL) != primary_key("Co", EntityType.CHEMICAL)
    assert primary_key("Nitrate", EntityType.CHEMICAL) == primary_key("nitrate", EntityType.CHEMICAL)


def test_country_aliases_use_pinned_iso_comparison_keys(policy):
    assert qualifier_signature((Qualifier("COUNTRY", "Nigeria"),), policy) == qualifier_signature(
        (Qualifier("COUNTRY", "Nigerian"),), policy
    ) == qualifier_signature((Qualifier("COUNTRY", "Federal Republic of Nigeria"),), policy) == qualifier_signature(
        (Qualifier("COUNTRY", "NG"),), policy
    )
    assert qualifier_signature((Qualifier("COUNTRY", "Ghanaian"),), policy) == (
        ("COUNTRY", "GH"),
    )
    assert qualifier_signature((Qualifier("COUNTRY", "Kenyan"),), policy) == (
        ("COUNTRY", "KE"),
    )
    assert qualifier_signature((Qualifier("COUNTRY", "Nigerien"),), policy) == (
        ("COUNTRY", "NE"),
    )
    assert qualifier_signature((Qualifier("COUNTRY", "Nigerian"),), policy) != qualifier_signature(
        (Qualifier("COUNTRY", "Nigerien"),), policy
    )


def test_feature_class_plural_and_conservative_synonyms(policy):
    assert qualifier_signature((Qualifier("FEATURE_CLASS", "rivers"),), policy) == (
        ("FEATURE_CLASS", "river"),
    )
    assert qualifier_signature((Qualifier("FEATURE_CLASS", "watershed"),), policy) == (
        ("FEATURE_CLASS", "basin"),
    )
    assert qualifier_signature((Qualifier("FEATURE_CLASS", "river"),), policy) != qualifier_signature(
        (Qualifier("FEATURE_CLASS", "basin"),), policy
    )
