# MUFASA — ADTC 2026 Submission Repository

This folder is prepared for the Africa Deep Tech Challenge 2026 Laptop LLM track.

## Before pushing to GitHub

### 1. Confirmed Devpost Team ID

Devpost project URL:

```text
https://devpost.com/software/mufasa
```

Therefore the confirmed Devpost project ID / Team ID is:

```text
mufasa
```

This value is already written to `metadata.json`.

### 2. Validate the repo

```bash
bash validate_submission.sh
```

### 3. Download the public Q4_K_M model

```bash
bash download_model.sh
```

### 4. Install the official profiler

```bash
python3 -m pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"
```

### 5. Smoke test

```bash
adtc-profiler run \
  --submission . \
  --mode participant \
  --output submission.json \
  --skip-accuracy
```

Confirm:

```json
"measured_on": "participant_laptop"
```

### 6. Final profiler run

```bash
adtc-profiler run \
  --submission . \
  --mode participant \
  --output submission.json
```

## Required files

- `metadata.json`
- `download_model.sh`
- `REPORT.md`
- `.gitignore`

`PROJECT_STORY.md` is provided for the Devpost project page.

The model itself is intentionally not committed to Git.
