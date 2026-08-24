"""The property axis of the coverage gate.

A facet is *what kind of fact* a claim can supply, or a question asks for. It is
the axis lexical search cannot see. retrieval-v1 scored recall@10 = 25/25 and
abstention 0/5 precisely because BM25 has only a topic axis: asked which
bacteria were in the Bosso samples it found the corpus's physicochemical
measurements of those exact samples and answered anyway.

Facets are a small controlled vocabulary, versioned with the corpus. Claim
facets are derived from claim_type, measurement keys, entity types and text.
Question facets come from a keyword lexicon. The gate requires a claim to
satisfy both axes at once.
"""

from __future__ import annotations

import re
from typing import Any

from .normalize import norm

FACET_VOCABULARY_VERSION = "facets-v1"

#: Human-readable names, used in the abstention sentence shown to the user.
FACET_LABELS: dict[str, str] = {
    "concentration_measurement": "measured concentrations",
    "physicochemical_property": "physicochemical measurements",
    "microbiology": "bacteriological or microbial findings",
    "toxicity_dose_response": "dose-response toxicity results",
    "histopathology": "tissue or organ damage findings",
    "health_risk_estimate": "modelled health-risk estimates",
    "clinical_incidence": "clinically confirmed disease rates",
    "hydrogeology_geophysics": "hydrogeological or geophysical results",
    "water_access_supply": "water access and supply figures",
    "statistical_model": "statistical model results",
    "forecast_validation": "forecast validation results",
    "rainfall_climate": "rainfall and climate measurements",
    "spatial_distribution": "spatial distribution results",
    "demography_population": "population figures",
    "method_protocol": "method and protocol descriptions",
    "recommendation_action": "recommendations",
    "evidence_gap": "stated evidence gaps",
    "cost_economics": "costs or economic figures",
    "technology_device": "device, sensor or software architecture details",
    "longitudinal_followup": "long-term follow-up results",
}

# --------------------------------------------------------------------------
# Claim side
# --------------------------------------------------------------------------

# (facet, regex over the claim's searchable text)
_CLAIM_TEXT_RULES: list[tuple[str, str]] = [
    ("microbiology", r"\b(bacteri\w*|coliform|e\.? ?coli|salmonell\w*|pathogen\w*|"
                     r"microbial|microbiolog\w*|faecal|fecal|colony|cfu)\b"),
    ("concentration_measurement", r"\b(concentration|mg/l|µg/l|ug/l|ppm|ppb|nitrate|"
                                  r"benzene|cadmium|lead|arsenic|contaminant level)\b"),
    ("physicochemical_property", r"\b(ph\b|turbidity|conductivity|electrical[- ]conductivity|"
                                 r"hardness|alkalinity|dissolved solids|temperature|physico)"),
    ("toxicity_dose_response", r"\b(lc50|ld50|median lethal|mortality|acute toxic\w*|"
                               r"sublethal|exposure concentration|96[- ]hour)\b"),
    ("histopathology", r"\b(histopatholog\w*|lesion\w*|necrosis|gill|liver|kidney|tissue)\b"),
    ("clinical_incidence", r"\b(incidence rate|prevalence of|diagnos\w+|clinically|"
                           r"cancer (?:cases|incidence|rate)|case[- ]control|cohort study)\b"),
    ("health_risk_estimate", r"\b(health risk|hazard quotient|risk index|carcinogenic risk|"
                             r"risk score|exposure risk)\b"),
    ("hydrogeology_geophysics", r"\b(resistivity|aquifer|borehole depth|water table|"
                                r"geophysic\w*|electrical resistivity|leachate|plume|"
                                r"vertical electrical|traverse)\b"),
    ("water_access_supply", r"\b(litres per person|l/person|water stress|water point|"
                            r"household[s]? (?:surveyed|sampled)|water supply|"
                            r"rainwater harvest\w*|water demand)\b"),
    ("statistical_model", r"\b(rmse|aic|sbc|bic|r-?squared|regression|sarima|arima|kalman|"
                          r"random forest|knn|imputation|missforest|model fit|"
                          r"meta[- ]regression)\b"),
    ("forecast_validation", r"\b(forecast|validation (?:year|period|set)|predicted versus|"
                            r"out[- ]of[- ]sample|hold[- ]out)\b"),
    ("rainfall_climate", r"\b(rainfall|precipitation|rainy season|dry season|"
                         r"rain rate|attenuation|seasonalit\w*|climate)\b"),
    ("spatial_distribution", r"\b(spatial|distribution across|east[- ]west|north[- ]south|"
                             r"traverse \d|mapped|transect|profile)\b"),
    ("demography_population", r"\b(population of|residents|inhabitants|urban population|"
                              r"million people|census)\b"),
    ("cost_economics", r"\b(cost|price|budget|naira|usd|\$|funding|expenditure|"
                       r"economic|capital outlay)\b"),
    ("technology_device", r"\b(sensor|smartphone|mobile app|application architecture|"
                          r"microcontroller|sms|gsm|hardware|firmware|telemetry|"
                          r"early[- ]warning system)\b"),
    ("longitudinal_followup", r"\b(years? (?:after|later|of follow)|follow[- ]up|"
                              r"longitudinal|long[- ]term monitoring)\b"),
]

_CLAIM_TYPE_FACETS: dict[str, tuple[str, ...]] = {
    "measurement": ("concentration_measurement",),
    "context_measurement": ("concentration_measurement",),
    "derived_measurement": ("concentration_measurement",),
    "measurement_comparison": ("concentration_measurement", "spatial_distribution"),
    "measurement_interpretation": ("concentration_measurement",),
    "experimental_result": ("toxicity_dose_response",),
    "experimental_observation": ("toxicity_dose_response",),
    "meta_regression_result": ("statistical_model",),
    "model_result": ("statistical_model",),
    "model_comparison": ("statistical_model",),
    "model_selection_result": ("statistical_model",),
    "model_diagnostic": ("statistical_model",),
    "model_hyperparameter_result": ("statistical_model",),
    "model_validation_result": ("statistical_model", "forecast_validation"),
    "data_quality_measurement": ("statistical_model",),
    "seasonality_result": ("rainfall_climate",),
    "spatial_result": ("spatial_distribution",),
    "method": ("method_protocol",),
    "method_result": ("method_protocol",),
    "recommendation": ("recommendation_action",),
    "design_proposal": ("recommendation_action",),
    "design_recommendation": ("recommendation_action",),
    "evidence_gap": ("evidence_gap",),
    "projection": ("demography_population",),
    "review_synthesis": (),
    "respondent_report": (),
    "field_observation": (),
    "author_interpretation": (),
}

_ENTITY_TYPE_FACETS: dict[str, tuple[str, ...]] = {
    "Organism": ("toxicity_dose_response",),
    "Pesticide": ("toxicity_dose_response",),
    "ToxicityMetric": ("toxicity_dose_response",),
    "HistopathologyOutcome": ("histopathology",),
    "HealthOutcome": ("health_risk_estimate",),
    "Contaminant": ("concentration_measurement",),
    "ContaminantPlume": ("hydrogeology_geophysics",),
    "Chemical": ("concentration_measurement",),
    "HydrogeologicSetting": ("hydrogeology_geophysics",),
    "GeophysicalFeature": ("hydrogeology_geophysics",),
    "GeologicFeature": ("hydrogeology_geophysics",),
    "StatisticalModel": ("statistical_model",),
    "ImputationMethod": ("statistical_model",),
    "ImputationMethodGroup": ("statistical_model",),
    "EvaluationMetric": ("statistical_model",),
    "ModelOutput": ("forecast_validation",),
    "TimeSeries": ("statistical_model",),
    "MonitoringStation": ("spatial_distribution",),
    "WaterAccessOutcome": ("water_access_supply",),
    "WaterInfrastructure": ("water_access_supply",),
    "WaterDemand": ("water_access_supply",),
    "Household": ("water_access_supply",),
    "HouseholdGroup": ("water_access_supply",),
    "Population": ("demography_population",),
    "EarlyWarning": ("technology_device",),
    "CommunicationSystem": ("technology_device",),
}

_MEASUREMENT_KEY_FACETS: list[tuple[str, str]] = [
    ("toxicity_dose_response", r"^(mortality_pct|exposure_time|exposure_hours|exposure_days|"
                               r"concentrations?(_mg_l)?|ci95_(low|high)|recovery_days|"
                               r"severity_scores|prior_concentration)$"),
    ("hydrogeology_geophysics", r"^(resistivity_\w+|depth(_\w+)?|groundwater_depth_\w+|"
                                r"traverse_\w+|horizontal_\w+|distance_\w+|profiles|"
                                r"inside_profiles|outside_profiles)$"),
    ("statistical_model", r"^(RSS|AIC|SBC|chosen_\w+|competitor_\w+|alpha|estimate|"
                          r"coefficient|p|RF|kNN|missForest|PMM|best_k_values|repeats|"
                          r"\w*_RMSE|missing_(counts|pct)|n_studies)$"),
    ("forecast_validation", r"^(validation_year|forecast_horizon|training_(start|end))$"),
    ("rainfall_climate", r"^(rainfall\w*|mean_annual_rainfall|rainy_season|dry_season|"
                         r"rain_rate\w*|attenuation\w*|frequency\w*|fade_margin\w*)$"),
    ("water_access_supply", r"^(households?|sampled_households|total_households|"
                            r"functional\w*|not_functional|abandoned|epileptic|seasonal|"
                            r"available_\w+|equivalent_litres|daily_shortfall_per_person|"
                            r"household_members|roof_area\w*|runoff_coefficient|"
                            r"min_households|max_households|selected_households|"
                            r"usable_responses|response_rate_pct)$"),
    ("demography_population", r"^(population|communities|districts|share_pct)$"),
    ("concentration_measurement", r"^(concentration_unit|surface_water_\w+|sediment\w*|"
                                  r"water|fish|time_exceedance_pct)$"),
]

_UNIT_FACETS: dict[str, tuple[str, ...]] = {
    "mg/l": ("concentration_measurement",),
    "ug/l": ("concentration_measurement",),
    "µg/l": ("concentration_measurement",),
    "us/cm": ("physicochemical_property",),
    "µs/cm": ("physicochemical_property",),
    "ntu": ("physicochemical_property",),
    "ph": ("physicochemical_property",),
    "l/person/day": ("water_access_supply",),
    "mm": ("rainfall_climate",),
    "mm/h": ("rainfall_climate",),
    "rmse": ("statistical_model",),
    "db": ("rainfall_climate",),
}


def _flat_measurement_keys(measurement: Any) -> list[str]:
    if not isinstance(measurement, dict):
        return []
    return [str(k) for k in measurement]


def claim_facets(claim: dict[str, Any]) -> set[str]:
    """Facets a claim can answer for."""
    found: set[str] = set()

    haystack = " ".join(
        str(x)
        for x in (
            claim.get("text", ""),
            claim.get("quote", "") or (claim.get("evidence") or {}).get("quote", ""),
            claim.get("predicate", ""),
            " ".join(e.get("name", "") for e in claim.get("entities") or []),
        )
    )
    hay = norm(haystack)
    for facet, pattern in _CLAIM_TEXT_RULES:
        if re.search(pattern, hay):
            found.add(facet)

    found.update(_CLAIM_TYPE_FACETS.get(claim.get("claim_type") or "", ()))

    for ent in claim.get("entities") or []:
        found.update(_ENTITY_TYPE_FACETS.get(ent.get("type") or "", ()))

    measurement = claim.get("measurement") or {}
    keys = _flat_measurement_keys(measurement)
    for facet, pattern in _MEASUREMENT_KEY_FACETS:
        if any(re.match(pattern, k) for k in keys):
            found.add(facet)

    unit = norm(str(measurement.get("unit") or ""))
    found.update(_UNIT_FACETS.get(unit, ()))

    return found


# --------------------------------------------------------------------------
# Question side
# --------------------------------------------------------------------------

# Ordered: the first matching rule per facet is enough. Phrased around what the
# asker wants back, not around the topic they are asking about.
_QUESTION_RULES: list[tuple[str, str]] = [
    ("microbiology", r"\b(bacteri\w*|microb\w*|pathogen\w*|coliform|e\.? ?coli|"
                     r"organisms? (?:detected|found|present)|species (?:were )?detected)\b"),
    ("cost_economics", r"\b(cost|costs?|price|how much (?:would|does|did) it (?:cost|take)|"
                       r"budget|expenditure|economic|afford|naira|dollars?)\b"),
    ("technology_device", r"\b(sensor|smartphone|mobile[- ]app|app architecture|hardware|"
                          r"device|firmware|which (?:technology|platform|system) "
                          r"(?:was |were )?(?:deploy|use)\w*)\b"),
    ("clinical_incidence", r"\b(clinically confirmed|diagnos\w+|incidence rate|"
                           r"cancer (?:rate|incidence|cases)|how many (?:patients|cases)|"
                           r"prevalence (?:rate )?of)\b"),
    ("longitudinal_followup", r"\b(years? (?:after|later)|follow[- ]up|long[- ]term|"
                              r"over the following \w+ years)\b"),
    ("toxicity_dose_response", r"\b(lc50|ld50|lethal concentration|median lethal|mortality|"
                               r"toxicit\w*|how toxic|exposure concentration)\b"),
    ("histopathology", r"\b(histopatholog\w*|tissue damage|lesion\w*|organ damage|"
                       r"gill|liver|kidney)\b"),
    ("physicochemical_property", r"\b(ph\b|turbidity|conductivit\w*|electrical[- ]conductivity|"
                                 r"hardness|physicochemical|dissolved solids)\b"),
    ("concentration_measurement", r"\b(concentration|level of|how much \w+ was (?:found|measured)|"
                                  r"mg/l|nitrate|benzene|cadmium|lead|arsenic|contamination level)\b"),
    ("hydrogeology_geophysics", r"\b(resistivity|aquifer|water table|leachate|plume|"
                                r"geophysic\w*|depth of|borehole depth|traverse)\b"),
    ("water_access_supply", r"\b(water stress|litres per person|l/person|water points?|"
                            r"water supply|water access|households? (?:were )?(?:surveyed|"
                            r"distributed|sampled)|rainwater harvest\w*)\b"),
    ("statistical_model", r"\b(rmse|aic|sbc|bic|which model|model (?:fit|perform\w*|compar\w*)|"
                          r"regression|imputation|sarima|arima|kalman|random forest|"
                          r"meta[- ]regression)\b"),
    ("forecast_validation", r"\b(forecast|validat\w+|predicted versus|accuracy of the model)\b"),
    ("rainfall_climate", r"\b(rainfall|precipitation|rainy season|dry season|rain rate|"
                         r"attenuation|seasonal\w*)\b"),
    ("spatial_distribution", r"\b(where|spatial|distribut\w+ across|which direction|"
                             r"east|west|north|south|mapped)\b"),
    ("demography_population", r"\b(population|how many people|residents|inhabitants)\b"),
    ("recommendation_action", r"\b(recommend\w*|what should|proposed|suggest\w*|"
                              r"what action)\b"),
    ("method_protocol", r"\b(how (?:was|were) .* (?:measured|sampled|collected|analysed)|"
                        r"what method|which method|protocol|procedure)\b"),
    ("evidence_gap", r"\b(evidence gap|what is missing|not been studied|no data)\b"),
]


def question_facets(question: str) -> list[str]:
    """Facets a question asks for, most specific first.

    Order is the rule order above, which runs from the narrow kinds of fact
    (microbiology, cost, device architecture, clinical incidence) to the broad
    ones. That ordering is load-bearing: "what was the clinically confirmed
    cancer incidence rate caused by benzene in Ogale?" mentions benzene, so it
    also matches `concentration_measurement`, and the corpus is full of benzene
    concentrations. Matching on any requested facet would answer it from the
    wrong kind of evidence. Only the first — the head of the question — decides.
    """
    hay = norm(question)
    return [facet for facet, pattern in _QUESTION_RULES if re.search(pattern, hay)]


def primary_question_facet(question: str, exclude: frozenset[str] | set[str] = frozenset()) -> str | None:
    """The single kind of fact this question is actually asking for."""
    for facet in question_facets(question):
        if facet not in exclude:
            return facet
    return None


def label(facet: str) -> str:
    return FACET_LABELS.get(facet, facet.replace("_", " "))
