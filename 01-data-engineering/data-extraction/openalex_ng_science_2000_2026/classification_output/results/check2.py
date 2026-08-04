import json
with open("C:/CodingWorld/Hackathons/AfricanDeepTechChallenge/MUFASA/01-data-engineering/data-extraction/openalex_ng_science_2000_2026/classification_output/results/result_0005.jsonl","r") as f:
    lines = f.readlines()
print("Total output lines:", len(lines))
# Show last 5 records
for i in range(max(0,len(lines)-5), len(lines)):
    rec = json.loads(lines[i])
    print(f"Line {i}: {rec['title'][:60]}")
# Check GLOFAS specifically
for i, l in enumerate(lines):
    rec = json.loads(l)
    if "GLOFAS" in rec.get("title",""):
        print(f"GLOFAS found in output at line {i}: {rec['decision']}")
