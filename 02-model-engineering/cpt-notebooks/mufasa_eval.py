"""Measuring what continued pretraining did, against the model it started from.

Everything here works on a raw completion model - no chat template, no
instruction following - because that is what a CPT checkpoint is.

Two things are measured, and they answer different questions.

  WITHIN a model: base vs CPT
      Perplexity is fine here. Both sides use the same tokenizer, so the ratio
      between them is honest, and it is the headline CPT result.

  ACROSS models: Gemma vs Qwen vs LFM
      Perplexity is NOT comparable and must never be used for this. It is a
      per-TOKEN quantity, and the three tokenizers cut the same text into
      different numbers of tokens. Measured on this corpus:

          LFM2.5   0.295 tokens/byte
          Gemma 3  0.314 tokens/byte
          Qwen3.5  0.324 tokens/byte

      For three models of IDENTICAL quality (1.10 bits per byte) that produces
      perplexities of 13.30, 11.37 and 10.54 - a 26% spread that is purely an
      artifact of tokenisation, and it would rank LFM last for no reason.

      Bits per byte divides the same total information by BYTES instead of
      tokens, so the tokenizer cancels out. Use it, and only it, to compare
      the three models with each other.

  span NLL       Negative log-likelihood of a target span - a place, a technical
                 term, a measured figure - with the surrounding prose masked out.
                 Called span NLL and not "entity recall" because that is what
                 it is: a likelihood, not a recall rate. Nothing is retrieved
                 and nothing is counted as correct or incorrect. LOWER means
                 the model finds the true continuation less surprising.

                 Measured twice: on TRAIN papers (retention - did it absorb
                 what it read) and on EVALUATE papers (generalisation - does
                 that help on research it never saw). Report both.

  standard benchmarks   ARC, HellaSwag, PIQA, MMLU via lm-evaluation-harness.
                 HIGHER is better. These detect capability lost while the
                 language-modelling numbers looked fine.
"""

import math
import re
from collections import Counter

import pandas as pd
import torch

WINDOW = 2048          # fixed, so base and CPT are scored on identical spans


# ------------------------------------------------- perplexity / bits-per-byte --

@torch.no_grad()
def score_text(model, tokenizer, texts, window=WINDOW, limit=None, progress=None):
    """Perplexity AND bits-per-byte over the same forward passes.

    Returns a dict. Use `perplexity` to compare a model against its own base;
    use `bits_per_byte` to compare different models with each other. They come
    from one measurement because computing them separately would double the
    cost for no reason.

    Scored in fixed windows rather than whole documents, so a long paper and a
    short one contribute in proportion to their length and the number does not
    move when max_seq_length changes.
    """
    model.eval()
    total_nll, total_tokens, total_bytes = 0.0, 0, 0
    for index, text in enumerate(texts[:limit] if limit else texts):
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        scored_to = 0
        for start in range(0, max(len(ids) - 1, 0), window):
            chunk = ids[start:start + window]
            if len(chunk) < 32:                      # too short to score fairly
                continue
            chunk = chunk.unsqueeze(0).to(model.device)
            out = model(chunk, labels=chunk)
            counted = chunk.numel() - 1
            total_nll += out.loss.item() * counted
            total_tokens += counted
            scored_to = start + len(chunk[0])
        # Only the bytes actually scored count, or a document skipped for being
        # short would still inflate the denominator.
        if scored_to:
            decoded = tokenizer.decode(ids[:scored_to], skip_special_tokens=True)
            total_bytes += len(decoded.encode("utf-8"))
        if progress:
            progress(index + 1)
    if not total_tokens or not total_bytes:
        return {"perplexity": float("nan"), "bits_per_byte": float("nan"),
                "tokens": 0, "bytes": 0}
    return {
        "perplexity": math.exp(total_nll / total_tokens),
        "bits_per_byte": total_nll / math.log(2) / total_bytes,
        "tokens": total_tokens,
        "bytes": total_bytes,
    }


def perplexity(model, tokenizer, texts, window=WINDOW, limit=None, progress=None):
    """Back-compatible wrapper returning (perplexity, tokens)."""
    got = score_text(model, tokenizer, texts, window, limit, progress)
    return got["perplexity"], got["tokens"]


# ------------------------------------------------------------- span targets --

PLACE = re.compile(r"\b(?:in|at|from|across)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})")
# Not a taxonomy matcher. This is a capitalised technical bigram - "Atterberg
# limits", "Uyo metropolis", sometimes a real binomial. Calling it SPECIES would
# be a lie: on this corpus most matches are domain terms, not organisms. What
# makes it usable as a probe is the recurrence test in _recurring(), which drops
# one-off sentence fragments like "Studying land".
TERM = re.compile(r"\b([A-Z][a-z]+\s+[a-z]{4,})\b")
SPECIES = TERM      # kept so older references still resolve
FIGURE = re.compile(r"\b(\d+\.\d+)\s*(?:%|mg|kg|ml|cfu|ppm|ppb|mm|cm|km|ha|mS|C\b)")

# Words that are page furniture, not knowledge. A model predicting "Table" after
# "shown in" has demonstrated nothing about African research.
STRUCTURAL = {
    "Fig", "Figure", "Figures", "Table", "Tables", "Section", "Appendix", "Plate",
    "Chart", "Map", "Photo", "Equation", "Eq", "Source", "Note", "Total", "Chapter",
    "Page", "Annex", "Box", "Panel", "Sci", "Vol", "No", "Ref", "Abstract",
}
MONTHS = {"January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"}
# Sentence scaffolding that the capitalisation heuristic picks up by accident.
GENERIC = {
    "The", "This", "These", "Those", "There", "However", "Although", "Therefore",
    "Parts", "Part", "Results", "Result", "Discussion", "Introduction", "Methods",
    "Conclusion", "Study", "Data", "Analysis", "Modified", "Based", "Using", "Both",
    "Each", "Other", "Some", "Such", "Their", "Its", "It", "We", "Our", "All",
}
REJECT = STRUCTURAL | MONTHS | GENERIC


def _worth_scoring(target):
    """Is this span actually carrying knowledge?"""
    words = target.split()
    if not words or len(target) < 4:
        return False
    if any(word in REJECT for word in words):
        return False
    return True


def _recurring(texts, minimum=2):
    """Candidate terms that appear more than once across the sampled papers.

    A term used twice is part of the literature's vocabulary; one used once is
    usually a sentence fragment the capitalisation rule caught by accident.
    """
    seen = Counter()
    for text in texts:
        for match in TERM.finditer(text):
            seen[match.group(1)] += 1
    return {term for term, n in seen.items() if n >= minimum}


def entity_items(texts, per_pattern=2, limit=400, max_share=0.04):
    """(context, target) pairs where the target is a place, term or figure.

    Four things this gets right that a single quota loop does not.

    Each pattern has its OWN quota. With one shared quota the first pattern
    fills it every time: measured on 40 papers, PLACE produced 240 of 240
    targets and terms and figures were never sampled at all.

    Terms must RECUR across the sampled papers. A capitalised bigram used once
    is usually a sentence fragment ("Studying land"); one used twice is part of
    the literature's vocabulary ("Atterberg limits").

    Page furniture is rejected. "Fig", "Table", "November", "Modified" all match
    a capitalised-word-after-preposition rule and none of them test knowledge.

    No single target may exceed `max_share` of the set. Unfiltered, "Nigeria"
    was 60 of 240 targets - a quarter of the measurement spent on the one word
    every paper in the corpus contains.
    """
    items, used = [], Counter()
    cap = max(1, int(limit * max_share))
    recurring = _recurring(texts)
    for text in texts:
        for pattern in (PLACE, TERM, FIGURE):
            found = 0
            for match in pattern.finditer(text):
                if found >= per_pattern or len(items) >= limit:
                    break
                start = match.start(1)
                target = match.group(1)
                if start < 200:                      # need real context first
                    continue
                if not _worth_scoring(target) or used[target] >= cap:
                    continue
                if pattern is TERM and target not in recurring:
                    continue
                items.append((text[max(0, start - 600):start], target))
                used[target] += 1
                found += 1
        if len(items) >= limit:
            break
    return items[:limit]


@torch.no_grad()
def entity_nll(model, tokenizer, items, progress=None):
    """Mean NLL of the target spans, with the context masked out of the loss."""
    model.eval()
    total, counted = 0.0, 0
    for index, (context, target) in enumerate(items):
        context_ids = tokenizer(context, return_tensors="pt").input_ids[0][-1024:]
        target_ids = tokenizer(target, add_special_tokens=False,
                               return_tensors="pt").input_ids[0]
        if not len(target_ids):
            continue
        ids = torch.cat([context_ids, target_ids]).unsqueeze(0).to(model.device)
        labels = ids.clone()
        labels[0, :len(context_ids)] = -100          # score the target only
        out = model(ids, labels=labels)
        total += out.loss.item() * len(target_ids)
        counted += len(target_ids)
        if progress:
            progress(index + 1)
    return (total / counted) if counted else float("nan"), counted


def describe_items(items):
    """What the probe is actually made of - check this before trusting a score."""
    targets = [t for _, t in items]
    kinds = Counter()
    for t in targets:
        if re.fullmatch(r"\d+\.\d+", t):
            kinds["figure"] += 1
        elif TERM.fullmatch(t):
            kinds["term"] += 1
        else:
            kinds["place"] += 1
    common = Counter(targets).most_common(1)
    return {"targets": len(targets), "distinct": len(set(targets)),
            "by kind": dict(kinds),
            "most common": f"{common[0][0]} x{common[0][1]}" if common else "-",
            "top share": f"{100 * common[0][1] / len(targets):.0f}%" if common else "-"}


# --------------------------------------------------------------- reporting --

LOWER_IS_BETTER = {
    "domain perplexity": True, "general perplexity": True,
    "domain bits/byte": True, "general bits/byte": True,
    "span NLL (train papers)": True, "span NLL (held out)": True,
}


def compare(base, tuned, base_name="base", tuned_name="CPT"):
    """One table. `LOWER_IS_BETTER` decides which direction counts as a win."""
    rows = []
    for metric in base:
        before, after = base[metric], tuned[metric]
        lower = LOWER_IS_BETTER.get(metric, True)
        change = (after - before) / before * 100 if before else float("nan")
        improved = (after < before) if lower else (after > before)
        rows.append({"metric": metric, base_name: round(before, 4),
                     tuned_name: round(after, 4),
                     "change %": round(change, 1),
                     "better?": "yes" if improved else "no"})
    return pd.DataFrame(rows)


def read_out(frame, base_name="base", tuned_name="CPT"):
    """Say in words what the table means, including the awkward combinations."""
    def get(metric, column):
        row = frame[frame.metric == metric]
        return float(row[column].iloc[0]) if len(row) else float("nan")

    lines = []
    domain = get("domain perplexity", "change %")
    general = get("general perplexity", "change %")
    retention = get("span NLL (train papers)", "change %")
    held = get("span NLL (held out)", "change %")

    if domain < -5:
        lines.append(f"Domain perplexity fell {abs(domain):.0f}% - the model predicts "
                     "African research writing markedly better than the base did. "
                     "This is the result CPT exists for.")
    elif domain < 0:
        lines.append(f"Domain perplexity fell only {abs(domain):.0f}%. Real but small: "
                     "more epochs, a higher rank, or more tokens would be the levers.")
    else:
        lines.append(f"Domain perplexity ROSE {domain:.0f}%. Something is wrong - "
                     "check the learning rate, and that the corpus cell is feeding "
                     "papers rather than the template's example dataset.")

    if general > 25:
        lines.append(f"General perplexity rose {general:.0f}%. That is catastrophic "
                     "forgetting: the model is paying for the domain with its general "
                     "ability. Lower the learning rate or train fewer steps.")
    elif general > 5:
        lines.append(f"General perplexity rose {general:.0f}% - the ordinary price of "
                     "domain adaptation, and the SFT stage will recover some of it.")
    else:
        lines.append(f"General perplexity moved {general:+.0f}% - essentially intact, "
                     "so nothing was traded away for the domain gain.")

    if retention < -5 and held < -5:
        lines.append("Span NLL improved on BOTH trained and held-out papers, so the "
                     "model learned the vocabulary of this literature, not just the "
                     "specific papers it read.")
    elif retention < -5 <= held:
        lines.append("Span NLL improved on trained papers but not held-out ones: it "
                     "memorised rather than generalised. Fine if you only need recall "
                     "of the corpus, weak if you want it to handle new research.")
    elif retention >= -5:
        lines.append("Span NLL barely moved even on trained papers. The spans are not "
                     "being absorbed - the usual cause is embed_tokens and lm_head "
                     "missing from target_modules.")

    lines.append("To compare this model with the other two, use bits/byte and NOT "
                 "perplexity: perplexity is per-token and the three tokenizers cut "
                 "the same text differently, which alone moves it by ~26%.")
    return lines


def plot(frame, base_name="base", tuned_name="CPT", title="CPT vs base"):
    """Grouped bars per metric, plus the percentage change beside them.

    Perplexity and bits-per-byte live on different scales, so the raw-value
    panel is split - drawing them on one axis would flatten bits/byte to
    nothing against a perplexity of 12.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    ppl = frame[~frame.metric.str.contains("bits/byte")].reset_index(drop=True)
    bpb = frame[frame.metric.str.contains("bits/byte")].reset_index(drop=True)
    panels = 3 if len(bpb) else 2
    widths = [3, 1.3, 2] if panels == 3 else [3, 2]
    figure, axes = plt.subplots(1, panels, figsize=(4.4 * panels, 4.2),
                                gridspec_kw={"width_ratios": widths})

    def bars(ax, data, heading):
        x = np.arange(len(data))
        width = 0.38
        ax.bar(x - width / 2, data[base_name], width, label=base_name, color="#9aa4b2")
        ax.bar(x + width / 2, data[tuned_name], width, label=tuned_name, color="#2f5d8c")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" (", "\n(").replace(" perplexity", "\nperplexity")
                            for m in data.metric], fontsize=8)
        ax.set_title(heading, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    bars(axes[0], ppl, f"{title} - lower is better")
    if panels == 3:
        bars(axes[1], bpb, "bits/byte\n(cross-model)")

    change = frame["change %"]
    axes[-1].barh(frame.metric, change,
                  color=["#3e7a5e" if c < 0 else "#a34434" for c in change])
    axes[-1].axvline(0, color="#333", linewidth=0.8)
    axes[-1].set_title("change % (green = improved)", fontsize=10)
    axes[-1].tick_params(labelsize=8)
    axes[-1].grid(axis="x", alpha=0.25)
    plt.tight_layout()
    return figure


def across_models(csv_paths):
    """Rank the three models honestly, on bits per byte only.

    Each path is a cpt_evaluation.csv written by one model's notebook.
    """
    rows = []
    for path in csv_paths:
        frame = pd.read_csv(path)
        got = frame[frame.metric == "domain bits/byte"]
        if not len(got):
            continue
        rows.append({"model": str(path).replace("cpt_evaluation_", "").replace(".csv", ""),
                     "base bits/byte": got["base"].iloc[0],
                     "CPT bits/byte": got["CPT"].iloc[0],
                     "change %": got["change %"].iloc[0]})
    table = pd.DataFrame(rows).sort_values("CPT bits/byte")
    return table.reset_index(drop=True)
