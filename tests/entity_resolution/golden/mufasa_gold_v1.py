"""MUFASA entity-resolution gold set, version 1.

Surface forms are taken from real extraction output where possible: the
milestone1 fixture (112 claims, 273 mentions, 194 distinct names) and the
parsed corpus. The remainder are the alias, homonym and study-instance cases
that corpus sample does not happen to contain.

Labels follow the design's five categories:
    SAME_CONCEPT | DIFFERENT_CONCEPT | SAME_INSTANCE | INSTANCE_OF | INSUFFICIENT

The existing golden file (water_identity_cases.json) contains six cases, five
of them negative and the one positive being a string matched against itself.
A resolver that never matches scores 6/6 on it. This set is deliberately
two-directional so that failure to connect is as visible as false merging.
"""

from __future__ import annotations

# (key, surface_text, atom_text, entity_type, identity_scope, paper, role, qualifiers)
MENTIONS: list[tuple] = [
    # ---- groundwater family: the measured failure, 12 surface forms ---------
    ("gw_plain",       "groundwater", "groundwater", "ENVIRONMENTAL_FEATURE", "CANONICAL", "P-G142", "SUBJECT", {}),
    ("gw_upper",       "Groundwater", "groundwater", "ENVIRONMENTAL_FEATURE", "CANONICAL", "P-G063", "SUBJECT", {}),
    ("gw_spaced",      " groundwater ", "groundwater", "ENVIRONMENTAL_FEATURE", "CANONICAL", "P-G149", "SUBJECT", {}),
    ("gw_shallow",     "shallow groundwater", "groundwater", "ENVIRONMENTAL_FEATURE", "CANONICAL", "P-G142", "SUBJECT", {"DEPTH_CLASS": "shallow"}),
    ("gw_deep",        "deep borehole groundwater", "groundwater", "ENVIRONMENTAL_FEATURE", "CANONICAL", "P-G063", "SUBJECT", {"DEPTH_CLASS": "deep"}),
    ("gw_table",       "groundwater table", "groundwater table", "ENVIRONMENTAL_FEATURE", "CANONICAL", "P-G142", "SUBJECT", {}),
    ("gw_quality",     "groundwater quality", "groundwater quality", "PROPERTY_METRIC", "CANONICAL", "P-G149", "OUTCOME", {}),
    ("borehole",       "borehole", "borehole", "INFRASTRUCTURE_DEVICE", "CANONICAL", "P-G063", "MEDIUM", {}),
    ("well",           "well", "well", "INFRASTRUCTURE_DEVICE", "CANONICAL", "P-G149", "MEDIUM", {}),
    ("rainwater",      "harvested rainwater", "rainwater", "ENVIRONMENTAL_FEATURE", "CANONICAL", "P-G147", "SUBJECT", {}),

    # ---- vernacular aliases: the case the corpus exists to serve ------------
    ("va_binomial",    "Vernonia amygdalina", "Vernonia amygdalina", "ORGANISM", "CANONICAL", "P-A001", "SUBJECT", {}),
    ("va_bitterleaf",  "bitter leaf", "bitter leaf", "ORGANISM", "CANONICAL", "P-A002", "SUBJECT", {}),
    ("va_onugbu",      "onugbu", "onugbu", "ORGANISM", "CANONICAL", "P-A003", "SUBJECT", {}),
    ("va_gymn",        "Gymnanthemum amygdalinum", "Gymnanthemum amygdalinum", "ORGANISM", "CANONICAL", "P-A004", "SUBJECT", {}),

    # ---- acronym expansion --------------------------------------------------
    ("rha_short",      "RHA", "RHA", "MATERIAL", "CANONICAL", "P-M001", "SUBJECT", {}),
    ("rha_long",       "rice husk ash", "rice husk ash", "MATERIAL", "CANONICAL", "P-M002", "SUBJECT", {}),
    ("rha_hull",       "rice hull ash", "rice hull ash", "MATERIAL", "CANONICAL", "P-M003", "SUBJECT", {}),

    # ---- scientific vs common name -----------------------------------------
    ("cg_binomial",    "Clarias gariepinus", "Clarias gariepinus", "ORGANISM", "CANONICAL", "P-G088", "SUBJECT", {}),
    ("cg_common",      "African catfish", "African catfish", "ORGANISM", "CANONICAL", "P-G088b", "SUBJECT", {}),

    # ---- homonyms: identical strings, different referents -------------------
    ("niger_state",    "Niger", "Niger", "PLACE", "CANONICAL", "P-G089", "PLACE", {"ADMINISTRATIVE_LEVEL": "state", "COUNTRY": "NG"}),
    ("niger_country",  "Niger", "Niger", "PLACE", "CANONICAL", "P-X001", "PLACE", {"ADMINISTRATIVE_LEVEL": "country", "COUNTRY": "NE"}),
    ("niger_river",    "Niger", "Niger", "ENVIRONMENTAL_FEATURE", "CANONICAL", "P-G086", "SUBJECT", {"FEATURE_CLASS": "river"}),
    ("spring_water",   "spring", "spring", "ENVIRONMENTAL_FEATURE", "CANONICAL", "P-G142", "MEDIUM", {}),
    ("spring_season",  "spring", "spring", "TIME_PERIOD", "CANONICAL", "P-X002", "CONTEXT", {}),

    # ---- places: same referent, different wording --------------------------
    ("minna_plain",    "Minna", "Minna", "PLACE", "CANONICAL", "P-G089", "PLACE", {"COUNTRY": "NG"}),
    ("minna_nested",   "Bosso, Minna", "Minna", "PLACE", "CANONICAL", "P-G149", "PLACE", {"COUNTRY": "NG"}),
    ("minna_state",    "Minna, Niger State", "Minna", "PLACE", "CANONICAL", "P-G089b", "PLACE", {"COUNTRY": "NG"}),

    # ---- study-local instances: must never merge across papers -------------
    ("bh_sample_a",    "wellhead water sample", "water sample", "SAMPLE_SPECIMEN", "STUDY_INSTANCE", "P-G142", "SUBJECT", {"DEPTH_CLASS": "deep"}),
    ("bh_sample_b",    "wellhead water sample", "water sample", "SAMPLE_SPECIMEN", "STUDY_INSTANCE", "P-G063", "SUBJECT", {"DEPTH_CLASS": "deep"}),
    ("bh_sample_a2",   "wellhead sample", "water sample", "SAMPLE_SPECIMEN", "STUDY_INSTANCE", "P-G142", "SUBJECT", {"DEPTH_CLASS": "deep"}),

    # ---- chemicals ---------------------------------------------------------
    ("nitrate",        "nitrate", "nitrate", "CHEMICAL", "CANONICAL", "P-G142", "AGENT", {}),
    ("nitrate_ion",    "NO3-", "nitrate", "CHEMICAL", "CANONICAL", "P-G063", "AGENT", {}),
    ("nitrite",        "nitrite", "nitrite", "CHEMICAL", "CANONICAL", "P-G149", "AGENT", {}),
    ("cadmium",        "cadmium", "cadmium", "CHEMICAL", "CANONICAL", "P-M004", "AGENT", {}),
    ("cd_symbol",      "Cd", "cadmium", "CHEMICAL", "CANONICAL", "P-M005", "AGENT", {}),

    # ---- unicode / OCR damage ----------------------------------------------
    ("catfish_ocr",    "Clarias gariepinus", "Clarias  gariepinus", "ORGANISM", "CANONICAL", "P-G088c", "SUBJECT", {}),
    ("turbidity",      "turbidity", "turbidity", "PROPERTY_METRIC", "CANONICAL", "P-G142", "OUTCOME", {}),
    ("turbidity_caps", "Turbidity", "turbidity", "PROPERTY_METRIC", "CANONICAL", "P-G149", "OUTCOME", {}),
]

# (left_key, right_key, label, why)
PAIRS: list[tuple[str, str, str, str]] = [
    # --- must connect: this is what the current golden file cannot test ------
    ("gw_plain", "gw_upper", "SAME_CONCEPT", "case variation only"),
    ("gw_plain", "gw_spaced", "SAME_CONCEPT", "whitespace variation only"),
    ("gw_plain", "gw_shallow", "SAME_CONCEPT", "depth is a qualifier, not identity"),
    ("gw_shallow", "gw_deep", "SAME_CONCEPT", "same concept, differing qualifier"),
    ("turbidity", "turbidity_caps", "SAME_CONCEPT", "case variation only"),
    ("minna_plain", "minna_nested", "SAME_CONCEPT", "nested place, same referent"),
    ("minna_plain", "minna_state", "SAME_CONCEPT", "administrative suffix, same referent"),
    ("va_binomial", "va_bitterleaf", "SAME_CONCEPT", "vernacular alias"),
    ("va_binomial", "va_onugbu", "SAME_CONCEPT", "Igbo vernacular alias"),
    ("va_binomial", "va_gymn", "SAME_CONCEPT", "taxonomic synonym"),
    ("rha_short", "rha_long", "SAME_CONCEPT", "acronym expansion"),
    ("rha_long", "rha_hull", "SAME_CONCEPT", "husk/hull spelling variant"),
    ("cg_binomial", "cg_common", "SAME_CONCEPT", "scientific and common name"),
    ("cg_binomial", "catfish_ocr", "SAME_CONCEPT", "double space from OCR"),
    ("nitrate", "nitrate_ion", "SAME_CONCEPT", "ionic notation"),
    ("cadmium", "cd_symbol", "SAME_CONCEPT", "element symbol"),

    # --- must not connect ----------------------------------------------------
    ("gw_plain", "gw_table", "DIFFERENT_CONCEPT", "medium vs hydrological feature"),
    ("gw_plain", "gw_quality", "DIFFERENT_CONCEPT", "medium vs property of it"),
    ("gw_plain", "borehole", "DIFFERENT_CONCEPT", "medium vs infrastructure"),
    ("gw_plain", "well", "DIFFERENT_CONCEPT", "medium vs infrastructure"),
    ("gw_plain", "rainwater", "DIFFERENT_CONCEPT", "different water sources"),
    ("borehole", "well", "DIFFERENT_CONCEPT", "related but distinct infrastructure"),
    ("niger_state", "niger_country", "DIFFERENT_CONCEPT", "homonym, different admin level"),
    ("niger_state", "niger_river", "DIFFERENT_CONCEPT", "homonym, different entity type"),
    ("spring_water", "spring_season", "DIFFERENT_CONCEPT", "homonym, different entity type"),
    ("nitrate", "nitrite", "DIFFERENT_CONCEPT", "one letter apart, different chemicals"),
    ("minna_plain", "niger_state", "DIFFERENT_CONCEPT", "city is not its state"),

    # --- study instances -----------------------------------------------------
    ("bh_sample_a", "bh_sample_b", "DIFFERENT_CONCEPT", "identical wording, different papers: hard invariant"),
    ("bh_sample_a", "bh_sample_a2", "SAME_INSTANCE", "same sample, same paper, shortened wording"),
    ("bh_sample_a", "gw_plain", "INSTANCE_OF", "a sample is an instance, not the concept"),
]


# Alternative names extraction is expected to emit for each mention, as agreed:
# the scientific name and common English name whenever the entity is local or
# vernacular, acronym expansions both ways, and chemical formula/name pairs.
# Negatives deliberately share no alias: nitrite never lists nitrate, and
# groundwater never lists borehole.
ALIASES: dict[str, list[str]] = {
    "va_binomial":   ["bitter leaf", "onugbu", "ewuro", "Gymnanthemum amygdalinum"],
    "va_bitterleaf": ["Vernonia amygdalina", "onugbu", "ewuro"],
    "va_onugbu":     ["Vernonia amygdalina", "bitter leaf", "ewuro"],
    "va_gymn":       ["Vernonia amygdalina", "bitter leaf"],

    "rha_short":     ["rice husk ash", "rice hull ash"],
    "rha_long":      ["RHA", "rice hull ash"],
    "rha_hull":      ["RHA", "rice husk ash"],

    "cg_binomial":   ["African catfish", "North African catfish", "Clarias lazera"],
    "cg_common":     ["Clarias gariepinus", "North African catfish"],
    "catfish_ocr":   ["African catfish", "Clarias lazera"],

    "nitrate":       ["NO3-", "nitrate ion"],
    "nitrate_ion":   ["nitrate", "nitrate ion"],
    "cadmium":       ["Cd"],
    "cd_symbol":     ["cadmium"],

    "minna_nested":  ["Minna"],
    "minna_state":   ["Minna"],

    "gw_upper":      ["groundwater"],
    "gw_spaced":     ["groundwater"],
    "gw_shallow":    ["groundwater"],
    "gw_deep":       ["groundwater"],
    "turbidity_caps": ["turbidity"],

    # negatives keep their own names only
    "nitrite":       ["NO2-"],
    "borehole":      ["bore hole"],
    "well":          ["water well"],
    "spring_water":  ["natural spring"],
    "spring_season": ["springtime"],
    "niger_state":   ["Niger State"],
    "niger_country": ["Republic of the Niger"],
    "niger_river":   ["River Niger"],
    "gw_table":      ["water table"],
    "gw_quality":    ["groundwater quality index"],
}


# Extraction labels each physical thing inside a paper. bh_sample_a and
# bh_sample_a2 are two wordings for one sample; bh_sample_b is another paper's
# sample, and paper_id still leads the key so it can never merge across papers.
INSTANCE_IDS: dict[str, str] = {
    "bh_sample_a":  "SAMPLE_1",
    "bh_sample_a2": "SAMPLE_1",
    "bh_sample_b":  "SAMPLE_1",
}

