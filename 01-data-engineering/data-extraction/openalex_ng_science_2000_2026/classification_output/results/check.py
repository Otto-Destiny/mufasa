import json
with open("C:/CodingWorld/Hackathons/AfricanDeepTechChallenge/MUFASA/01-data-engineering/data-extraction/openalex_ng_science_2000_2026/classification_output/results/result_0005.jsonl","r") as f:
    lines = f.readlines()
print("Total output lines:", len(lines))
last = json.loads(lines[-1])
print("Last record title:", last["title"][:80])
with open("C:/CodingWorld/Hackathons/AfricanDeepTechChallenge/MUFASA/01-data-engineering/data-extraction/openalex_ng_science_2000_2026/classification_output/sub_partitions_200/sub_0005.jsonl","r") as f:
    inlines = f.readlines()
print("Total input lines:", len(inlines))
lastin = json.loads(inlines[-1])
print("Last input title:", lastin["title"][:80])
glofas_found = False
for i, l in enumerate(inlines):
    rec = json.loads(l)
    if "GLOFAS" in rec.get("title",""):
        print("GLOFAS found in input at line", i)
        glofas_found = True
if not glofas_found:
    print("GLOFAS NOT found in input file")
