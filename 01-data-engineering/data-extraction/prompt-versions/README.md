# Extraction prompt archive

The system prompts in `llm-claim-extraction.ipynb` are f-strings that
interpolate the controlled vocabularies, so what is stored here is the
**rendered** text the model actually receives, not the template. Both versions
are rendered the same way, so they diff cleanly.

Each version folder holds three files:

| file | task |
|---|---|
| `context-system-prompt.txt` | study contexts + the paper profile |
| `observation-system-prompt.txt` | atomic observations |
| `training-system-prompt.txt` | factual / reasoning / reranker / preference pairs |

## Comparing versions

```bash
git diff --no-index \
  01-data-engineering/data-extraction/prompt-versions/mufasa-extraction-2.0-candidate.1 \
  01-data-engineering/data-extraction/prompt-versions/mufasa-extraction-2.3-candidate.1
```

## Versions

| version | rendered size (context / observation / training) | state |
|---|---|---|
| `mufasa-extraction-2.0-candidate.1` | 10,524 / 7,925 / 7,707 | never ran; preflight failed on `model_not_found` |
| `mufasa-extraction-2.1-candidate.6` | 13,553 / 10,042 / 11,247 | superseded before production |
| `mufasa-extraction-2.2-candidate.1` | 13,553 / 10,042 / 11,429 | superseded before production |
| `mufasa-extraction-2.2-candidate.2` | 14,328 / 10,349 / 9,246 | failed exact-grounding pilot; superseded |
| `mufasa-extraction-2.3-candidate.1` | 16,972 / 12,937 / 9,394 | current production candidate |

## What changed between them

Measured on the prompt files only: **+151 lines, −20**
(context +46/−3, observation +35/−2, training +70/−15). The substantive
differences:

**All three prompts**

- An explicit, bounded exception to the outside-knowledge rule for entity
  **aliases**. Version 2.0 forbade outside knowledge and then asked for aliases
  from the model's own knowledge in the same prompt; 2.1 states that names are
  exempt because a name is not a claim, and that everything else is not.

**Context / paper profile**

- New `discipline` and `discipline_secondary` fields over a 42-value academic
  taxonomy, recorded alongside — not replacing — the six MUFASA domains.
- New `OTH` MUFASA domain for papers outside the taxonomy, which forces review.

**Observations**

- New `HOW MANY TO EXTRACT` section: **10–50 per paper**, where 2.0 had no
  quota at all. The maximum is enforced by the validator; the minimum is a
  stated target, never a rejection.
- Prioritisation rules for what to keep when the ceiling binds, including
  keeping a table's extremes and optimum rather than its first rows.

**Training pairs**

- Per-chunk caps replaced by a **per-paper budget** stated in the user message.
  2.0 capped each chunk at 10/8/3/3, so a paper's yield depended on how many
  chunks it happened to have; 2.1 targets 20/20/5/5 per paper regardless.
- `AIM FOR BRILLIANT` quality bar, with an explicit warning that it means
  insight rather than confident-sounding prose.
- Floors requiring **5 CONCEPT** and **3 INNOVATION** pairs of the 20 reasoning
  pairs, so the fundamentals of the field and the locally distinctive work
  cannot both be dropped in favour of easier kinds.
- A longer `INNOVATION` definition asking for the constraint an adaptation
  answers, not just the novelty.

The rendered prompts are the record; this summary is a reading aid and the diff
is the authority.

## What changed in 2.2-candidate.1

- Removed the training prompt's permission to use general scientific
  background. Every accepted scientific assertion must derive from supplied
  paper text and its linked evidence quote.
- Numbers in answers, reasoning and chosen preference answers must appear in
  the linked evidence; only an intentionally wrong rejected answer may contain
  an unsupported number.

## What changed in 2.2-candidate.2

- Context and observation prompts explicitly request stated `COUNTRY` and
  `FEATURE_CLASS` identity qualifiers without permitting inference.
- Context profiles echo deterministic full-versus-partial coverage; partial
  profiles cannot claim missing content or exclude resolver input.
- The training prompt uses a task-specific grounding prelude and no longer
  repeats entity, alias or observation-schema instructions it never outputs.

## What changed in 2.3-candidate.1

- Made the validator's literal-copy rules explicit, including body-only source
  spans, page ownership, parser artifacts and line breaks.
- Defined how all atoms from one source phrase share the complete surface text,
  evidence ID and provenance while retaining distinct semantic atoms.
- Added the exact context, observation and reranker grounding requirements that
  the old prompt left implicit.
- Repair calls now receive the rejected JSON and all validation errors, so array
  paths identify something the model can actually correct.
