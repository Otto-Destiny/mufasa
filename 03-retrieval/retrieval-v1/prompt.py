"""
Formats retrieved claims into a tagged evidence block and a full prompt,
per retrieval-architecture.md's "What goes to the model": Observations
tagged [E1], [E2]..., with the quoted span attached to each.
"""
from retrieve import get_claim


def format_evidence_block(con, claim_ids):
    """Returns (block_text, tag_map) where tag_map is {"E1": claim_id, ...}."""
    lines = []
    tag_map = {}
    for i, cid in enumerate(claim_ids, start=1):
        row = get_claim(con, cid)
        if not row:
            continue
        _, paper_id, text, quote, page, section = row
        tag = f"E{i}"
        tag_map[tag] = cid
        lines.append(f'[{tag}] (paper {paper_id}, page {page}): "{quote}"')
    return "\n".join(lines), tag_map


def evidence_texts(con, tag_map):
    """Returns {tag: "claim text + quote"} for validation lookups."""
    out = {}
    for tag, cid in tag_map.items():
        row = get_claim(con, cid)
        if not row:
            continue
        _, paper_id, text, quote, page, section = row
        out[tag] = f"{text or ''} {quote or ''}"
    return out


def build_prompt(question, evidence_block):
    return f"""Answer the question using ONLY the evidence listed below. Do not use outside knowledge.

Evidence:
{evidence_block}

Question: {question}

Rules:
- Cite the tag (like [E1]) for every specific fact, number or name you state.
- Only use facts that are explicitly present in the evidence text above.
- If the evidence does not contain the answer, reply with exactly this sentence and nothing else: No matching evidence in this corpus.

Answer:"""
