#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"
MODEL_FILE="$MODEL_DIR/MUFASA-Gemma3-1B-SFT-Q4_K_M.gguf"
REPO="DestinyOtto/mufasa-gemma3-1b-sft-gguf"
TMP="$MODEL_FILE.partial"

mkdir -p "$MODEL_DIR"

is_valid_gguf() {
  python3 - "$1" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file() or p.stat().st_size < 100_000_000:
    raise SystemExit(1)
with p.open("rb") as f:
    if f.read(4) != b"GGUF":
        raise SystemExit(1)
PY
}

if [[ -f "$MODEL_FILE" ]]; then
  if is_valid_gguf "$MODEL_FILE"; then
    echo "model already present and valid at $MODEL_FILE — skipping download"
    exit 0
  fi
  rm -f "$MODEL_FILE"
fi

REMOTE_FILE="$(
python3 - "$REPO" <<'PY'
import json, sys, urllib.request
repo = sys.argv[1]
with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=60) as r:
    data = json.load(r)

ggufs = [
    x.get("rfilename", "")
    for x in data.get("siblings", [])
    if x.get("rfilename", "").lower().endswith(".gguf")
]
preferred = [n for n in ggufs if "q4_k_m" in n.lower()]
if not preferred:
    raise SystemExit(
        "No Q4_K_M GGUF found in the public repository. Available GGUFs: "
        + ", ".join(ggufs)
    )
preferred.sort(key=lambda n: ("/" in n, len(n), n.lower()))
print(preferred[0])
PY
)"

ENCODED_FILE="$(
python3 - "$REMOTE_FILE" <<'PY'
import sys, urllib.parse
print(urllib.parse.quote(sys.argv[1], safe="/"))
PY
)"

URL="https://huggingface.co/${REPO}/resolve/main/${ENCODED_FILE}?download=true"
echo "Downloading ${REPO}/${REMOTE_FILE}"
rm -f "$TMP"
trap 'rm -f "$TMP"' EXIT

if command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 4 --retry-delay 2 --progress-bar -o "$TMP" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget --tries=4 --show-progress -O "$TMP" "$URL"
else
  echo "error: curl or wget is required" >&2
  exit 1
fi

if ! is_valid_gguf "$TMP"; then
  echo "error: downloaded file is not a valid GGUF" >&2
  exit 1
fi

mv "$TMP" "$MODEL_FILE"
trap - EXIT
echo "Model ready: $MODEL_FILE"
