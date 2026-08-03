import json
with open("C:/CodingWorld/Hackathons/AfricanDeepTechChallenge/MUFASA/01-data-engineering/data-extraction/openalex_ng_science_2000_2026/classification_output/results/result_0005.jsonl","r") as f:
    lines = f.readlines()
print(f"Total records: {len(lines)}")
decisions = {}
evidence_levels = {}
hard_excl = 0
for l in lines:
    r = json.loads(l)
    d = r["decision"]
    decisions[d] = decisions.get(d, 0) + 1
    e = r["evidence_level"]
    evidence_levels[e] = evidence_levels.get(e, 0) + 1
    if r["hard_exclusion"]:
        hard_excl += 1
print(f"Decisions: {decisions}")
print(f"Evidence levels: {evidence_levels}")
print(f"Hard exclusions: {hard_excl}")
# Verify all required fields present
required = ["openalex_id","title","decision","hard_exclusion","hard_exclusion_reason","evidence_level","african_centrality","local_specificity","scientific_depth","knowledge_value","local_applicability","total_score","african_focus","scientific_evidence","african_country_codes","african_relevance_tags","evidence","inference_basis","reason","field_name","primary_topic","work_type"]
first = json.loads(lines[0])
missing = [k for k in required if k not in first]
print(f"Missing fields in first record: {missing}")
# Check total_score consistency
errors = 0
for i, l in enumerate(lines):
    r = json.loads(l)
    expected = r["african_centrality"] + r["local_specificity"] + r["scientific_depth"] + r["knowledge_value"] + r["local_applicability"]
    if r["total_score"] != expected:
        print(f"Line {i}: total_score mismatch ({r['total_score']} vs expected {expected})")
        errors += 1
print(f"Total score errors: {errors}")
