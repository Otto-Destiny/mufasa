#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("metadata.json").read_text(encoding="utf-8"))

assert data["domain"] == "math_scientific_reasoning"
assert data["budget_laptop_claim"] is True
assert data["model"]["runtime"] == "llama.cpp"
assert data["model"]["quantization"].startswith("GGUF ")
assert isinstance(data["model"]["parameters_estimate"], str)
assert len(data["test_prompts"]) == 2
assert data["_runtime"]["model_path"] == "model/MUFASA-Gemma3-1B-SFT-Q4_K_M.gguf"
assert data["submitter"]["github_handle"] == "willieseun"

assert data["team_id"] == "mufasa", "Devpost Team ID must be 'mufasa'"

print("metadata structural checks passed")
PY

grep -q '^model/' .gitignore
grep -q '^\*.gguf' .gitignore
test -x download_model.sh
echo "repository checks passed"
