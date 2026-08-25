"""Non-destructive, type-aware comparison keys.

The functions here create candidate-retrieval keys.  They never rewrite the
stored label and never, by themselves, prove that two entities are identical.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .contracts import EntityType, Mention, Qualifier
from .policy import ResolverPolicy


NORMALIZATION_IMPLEMENTATION = "mufasa-normalization-1.0.0"
QUALIFIER_NORMALIZATION_VERSION = "mufasa-qualifier-normalization-v1.0.0"
_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-"})
_APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u02bc": "'"})
_SPACE_RE = re.compile(r"\s+")
_LOOSE_PUNCT_RE = re.compile(r"[^\w\s-]", re.UNICODE)
_TOKEN_RE = re.compile(r"[\w]+(?:[-'][\w]+)*", re.UNICODE)
_FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d*){1,12}(?:[+-]\d*)?$")
_ORGANISM_AUTHORSHIP_RE = re.compile(
    r"\s+(?:\([A-Z][^)]{0,80}\)\s*)?(?:[A-Z][A-Za-z.-]+(?:\s+(?:ex|&|and)\s+[A-Z][A-Za-z.-]+)*)(?:,?\s*\d{4})?$"
)

# Pinned offline snapshot: common English country names for all African states
# plus a small set of high-value official/adjectival/code forms. Stored values
# are ISO 3166-1 alpha-2 comparison keys; original extraction text is unchanged.
_AFRICAN_COUNTRIES = {
    name: code
    for name, code in (
        ("algeria", "DZ"), ("angola", "AO"), ("benin", "BJ"), ("botswana", "BW"),
        ("burkina faso", "BF"), ("burundi", "BI"), ("cabo verde", "CV"),
        ("cameroon", "CM"), ("central african republic", "CF"), ("chad", "TD"),
        ("comoros", "KM"), ("democratic republic of the congo", "CD"),
        ("republic of the congo", "CG"), ("cote d ivoire", "CI"), ("djibouti", "DJ"),
        ("egypt", "EG"), ("equatorial guinea", "GQ"), ("eritrea", "ER"),
        ("eswatini", "SZ"), ("ethiopia", "ET"), ("gabon", "GA"), ("gambia", "GM"),
        ("ghana", "GH"), ("guinea", "GN"), ("guinea bissau", "GW"), ("kenya", "KE"),
        ("lesotho", "LS"), ("liberia", "LR"), ("libya", "LY"), ("madagascar", "MG"),
        ("malawi", "MW"), ("mali", "ML"), ("mauritania", "MR"), ("mauritius", "MU"),
        ("morocco", "MA"), ("mozambique", "MZ"), ("namibia", "NA"), ("niger", "NE"),
        ("nigeria", "NG"), ("rwanda", "RW"), ("sao tome and principe", "ST"),
        ("senegal", "SN"), ("seychelles", "SC"), ("sierra leone", "SL"),
        ("somalia", "SO"), ("south africa", "ZA"), ("south sudan", "SS"),
        ("sudan", "SD"), ("tanzania", "TZ"), ("togo", "TG"), ("tunisia", "TN"),
        ("uganda", "UG"), ("zambia", "ZM"), ("zimbabwe", "ZW"),
    )
}
_COUNTRY_ALIASES = {
    **_AFRICAN_COUNTRIES,
    **{code.casefold(): code for code in _AFRICAN_COUNTRIES.values()},
    "federal republic of nigeria": "NG",
    "nigerian": "NG",
    "ivory coast": "CI",
    "cote divoire": "CI",
    "côte d ivoire": "CI",
    "cape verde": "CV",
    "the gambia": "GM",
    "democratic republic of congo": "CD",
    "republic of congo": "CG",
    "são tomé and príncipe": "ST",
    "congo kinshasa": "CD",
    "dr congo": "CD",
    "drc": "CD",
    "congo brazzaville": "CG",
    "swaziland": "SZ",
    "united republic of tanzania": "TZ",
    "algerian": "DZ", "angolan": "AO", "beninese": "BJ", "botswanan": "BW",
    "motswana": "BW", "burkinabe": "BF", "burundian": "BI",
    "cabo verdean": "CV", "cape verdean": "CV", "cameroonian": "CM",
    "central african": "CF", "chadian": "TD", "comorian": "KM",
    "ivorian": "CI", "djiboutian": "DJ", "egyptian": "EG",
    "equatorial guinean": "GQ", "eritrean": "ER", "swazi": "SZ",
    "ethiopian": "ET", "gabonese": "GA", "gambian": "GM", "ghanaian": "GH",
    "bissau guinean": "GW", "kenyan": "KE", "basotho": "LS", "lesothan": "LS",
    "liberian": "LR", "libyan": "LY", "malagasy": "MG", "malawian": "MW",
    "malian": "ML", "mauritanian": "MR", "mauritian": "MU", "moroccan": "MA",
    "mozambican": "MZ", "namibian": "NA", "nigerien": "NE", "rwandan": "RW",
    "sao tomean": "ST", "senegalese": "SN", "seychellois": "SC",
    "sierra leonean": "SL", "somali": "SO", "south african": "ZA",
    "south sudanese": "SS", "sudanese": "SD", "tanzanian": "TZ", "togolese": "TG",
    "tunisian": "TN", "ugandan": "UG", "zambian": "ZM", "zimbabwean": "ZW",
}
_FEATURE_CLASS_ALIASES = {
    "river": "river",
    "rivers": "river",
    "lake": "lake",
    "lakes": "lake",
    "basin": "basin",
    "basins": "basin",
    "watershed": "basin",
    "watersheds": "basin",
}


@dataclass(frozen=True)
class ComparisonKeys:
    primary: str
    loose: str
    tokens: tuple[str, ...]
    acronym: str | None
    scientific: str | None
    qualifier_signature: tuple[tuple[str, str], ...]


def normalized_text(text: str, *, preserve_case: bool = False) -> str:
    value = unicodedata.normalize("NFC", text).translate(_DASHES).translate(_APOSTROPHES)
    value = _SPACE_RE.sub(" ", value.strip())
    return value if preserve_case else value.casefold()


def _is_chemical_formula(text: str) -> bool:
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFC", text))
    return bool(_FORMULA_RE.fullmatch(compact))


def primary_key(text: str, entity_type: EntityType) -> str:
    # Formula case is identity-relevant: CO (carbon monoxide) is not Co (cobalt).
    preserve_case = entity_type == EntityType.CHEMICAL and _is_chemical_formula(text)
    return normalized_text(text, preserve_case=preserve_case)


def loose_key(text: str, entity_type: EntityType) -> str:
    base = primary_key(text, entity_type)
    if entity_type == EntityType.CHEMICAL and _is_chemical_formula(text):
        return base
    base = _LOOSE_PUNCT_RE.sub(" ", base.replace("_", " "))
    return _SPACE_RE.sub(" ", base).strip()


def token_key(text: str, entity_type: EntityType) -> tuple[str, ...]:
    return tuple(sorted(set(token.casefold() for token in _TOKEN_RE.findall(loose_key(text, entity_type)))))


def character_ngrams(text: str, entity_type: EntityType, size: int) -> tuple[str, ...]:
    """Return deterministic boundary-aware n-grams for candidate blocking."""

    value = loose_key(text, entity_type)
    if not value:
        return ()
    padded = f"^{value}$"
    if len(padded) <= size:
        return (padded,)
    return tuple(sorted({padded[index : index + size] for index in range(len(padded) - size + 1)}))


def acronym_key(text: str) -> str | None:
    raw = unicodedata.normalize("NFC", text).strip()
    compact = re.sub(r"[^A-Za-z0-9]", "", raw)
    if 2 <= len(compact) <= 12 and compact.upper() == compact and any(ch.isalpha() for ch in compact):
        return compact
    words = _TOKEN_RE.findall(raw)
    if 2 <= len(words) <= 10:
        return "".join(word[0] for word in words).upper()
    return None


def scientific_candidate_key(text: str, entity_type: EntityType) -> str | None:
    """Return a candidate-only type-specific key, never an identity assertion."""

    if entity_type != EntityType.ORGANISM:
        return None
    value = normalized_text(text, preserve_case=True)
    without_authorship = _ORGANISM_AUTHORSHIP_RE.sub("", value).strip()
    return without_authorship.casefold() if without_authorship and without_authorship != value else None


def qualifier_signature(
    qualifiers: tuple[Qualifier, ...], policy: ResolverPolicy, semantics: set[str] | None = None
) -> tuple[tuple[str, str], ...]:
    selected = semantics or {"IDENTITY_BEARING", "INSTANCE_DEFINING"}
    pairs: list[tuple[str, str]] = []
    for qualifier in qualifiers:
        rule = policy.qualifier_rule(qualifier.kind)
        if rule and rule.semantic in selected:
            pairs.append((rule.kind, qualifier_comparison_value(rule.kind, qualifier.value_text)))
    return tuple(sorted(set(pairs)))


def qualifier_comparison_value(kind: str, value_text: str) -> str:
    value = normalized_text(value_text)
    lookup = _LOOSE_PUNCT_RE.sub(" ", value.replace("_", " "))
    lookup = _SPACE_RE.sub(" ", lookup).strip()
    if kind.upper() == "COUNTRY":
        return _COUNTRY_ALIASES.get(lookup, value)
    if kind.upper() == "FEATURE_CLASS":
        return _FEATURE_CLASS_ALIASES.get(lookup, value)
    return value


def build_keys(mention: Mention, policy: ResolverPolicy) -> ComparisonKeys:
    return ComparisonKeys(
        primary=primary_key(mention.atom_text, mention.entity_type),
        loose=loose_key(mention.atom_text, mention.entity_type),
        tokens=token_key(mention.atom_text, mention.entity_type),
        acronym=acronym_key(mention.atom_text),
        scientific=scientific_candidate_key(mention.atom_text, mention.entity_type),
        qualifier_signature=qualifier_signature(mention.qualifiers, policy),
    )
