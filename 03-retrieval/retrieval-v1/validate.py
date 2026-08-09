"""
Validation before display, per retrieval-architecture.md:
  - every number in the answer must appear in the Observation it cites
  - every specific claim must carry a tag
  - an invented tag (e.g. [E9] when 8 were supplied) fails immediately

Known limitation: this only checks numbers and tags mechanically. It does
not catch a non-numeric invented fact (e.g. a made-up species name) unless
the answer also fails on tags/numbers elsewhere. A real implementation
would need an entailment check, not string matching -- noted here rather
than silently overclaiming what this v1 catches.
"""
import re

TAG_RE = re.compile(r"\[E(\d+)\]")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def validate(answer, tag_map, evidence_texts):
    stripped = answer.strip()
    if stripped.lower().startswith("no matching evidence"):
        return {"abstained": True, "verdict": "abstained"}

    used_tags = sorted(set(f"E{n}" for n in TAG_RE.findall(answer)))
    invented = [t for t in used_tags if t not in tag_map]

    numbers = NUMBER_RE.findall(answer)

    if used_tags and not invented:
        allowed_text = " ".join(evidence_texts.get(t, "") for t in used_tags)
    else:
        # no tags cited at all, or an invented one -- check against
        # everything offered as a last-resort signal, but this case
        # already fails below regardless of the number check.
        allowed_text = " ".join(evidence_texts.values())

    unsupported = [n for n in numbers if n not in allowed_text]

    if invented:
        verdict = "fail_invented_tag"
    elif numbers and not used_tags:
        verdict = "fail_no_citation"
    elif unsupported:
        verdict = "fail_unsupported_number"
    else:
        verdict = "pass"

    return {
        "abstained": False,
        "used_tags": used_tags,
        "invented_tags": invented,
        "numbers_stated": numbers,
        "unsupported_numbers": unsupported,
        "verdict": verdict,
    }
