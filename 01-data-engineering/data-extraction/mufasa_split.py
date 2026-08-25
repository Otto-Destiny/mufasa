"""Partition the corpus once, before anything is trained on it.

Every later stage - CPT, SFT, DPO, benchmarks - reads the same manifest, so a
paper can never be trained on in one stage and evaluated on in another.

  train     raw papers may enter CPT; derived SFT and DPO examples may train
  evaluate  absent from all training; used while iterating
  test      locked, untouched until the final evaluation

The corpus was already deduplicated by study family before download, so each
family_id holds exactly one paper and a paper-level split IS a family-level
split. That is why there is no clustering step here. What there IS is an audit:
near-identical titles are looked for across splits, so anything that slipped
through the earlier dedup shows up rather than quietly inflating a held-out
score.

One distinction worth keeping straight. A paper used in CPT-train AND SFT-train
is intentional curriculum overlap, not leakage - CPT teaches the language and
facts, SFT teaches how to answer about them. Evaluating on a trained paper is
legitimate too, but it measures knowledge RETENTION and must be reported as
that, never as generalisation to unseen research.
"""

import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)
FIELD = re.compile(r'^(\w+):\s*"?(.*?)"?\s*$', re.M)

STOP = {
    "the", "of", "and", "in", "on", "for", "a", "an", "to", "from", "with", "by",
    "its", "their", "some", "using", "study", "studies", "analysis", "assessment",
    "evaluation", "investigation", "determination", "characterization", "effect",
    "effects", "impact", "case", "review", "among", "between", "during", "based",
    "nigeria", "nigerian", "africa", "african", "state", "local", "government",
}


def read_manifest(markdown_dir):
    """One row per paper, from the front matter each file already carries."""
    rows = []
    for path in sorted(Path(markdown_dir).glob("*.md")):
        head = path.open(encoding="utf-8", errors="replace").read(1200)
        found = FRONT.search(head)
        fields = dict(FIELD.findall(found.group(1))) if found else {}
        rows.append({
            "paper_id": fields.get("paper_id") or path.stem,
            "family_id": fields.get("family_id") or path.stem,
            "title": fields.get("title", ""),
            "domain": fields.get("mufasa_domain", ""),
            "licence": fields.get("licence", ""),
            "doi": fields.get("doi", ""),
            "chars": path.stat().st_size,
        })
    return pd.DataFrame(rows)


def split(frame, evaluate=0.03, test=0.03, seed=7, floor=0):
    """Assign whole families to train / evaluate / test, balanced by domain.

    Stratifying by domain matters: health is 37% of the corpus, so an
    unstratified draw can leave a split with almost none of a smaller domain.

    A flat percentage keeps the SHARES right but not the COUNTS. Technology is
    2.9% of the corpus, so 3% of it is nine papers - enough for a pooled score,
    far too few to say anything about technology on its own. `floor` sets a
    minimum papers-per-domain-per-split for exactly that case; it is capped at a
    fifth of the domain so a small domain cannot be hollowed out to supply it.
    """
    frame = frame.copy()
    by_domain = defaultdict(list)
    for family, group in frame.groupby("family_id"):
        by_domain[group.domain.iloc[0]].append(family)

    assignment, rng = {}, random.Random(seed)
    for domain, families in by_domain.items():
        families = sorted(families)
        rng.shuffle(families)
        n = len(families)
        n_eval, n_test = int(n * evaluate), int(n * test)
        if floor:
            n_eval = min(max(n_eval, floor), n // 5)
            n_test = min(max(n_test, floor), n // 5)
        for family in families[:n_eval]:
            assignment[family] = "evaluate"
        for family in families[n_eval:n_eval + n_test]:
            assignment[family] = "test"
        for family in families[n_eval + n_test:]:
            assignment[family] = "train"
    frame["split"] = frame.family_id.map(assignment)
    return frame


def title_key(title, keep=5):
    """The rare words of a title - a fingerprint for spotting a repeat."""
    words = [w for w in re.findall(r"[a-z]{4,}", str(title).lower()) if w not in STOP]
    return tuple(sorted(set(words))[:keep])


def audit(frame):
    """Everything that could make this split dishonest, checked not assumed."""
    problems = []
    straddle = frame.groupby("family_id").split.nunique() > 1
    if straddle.any():
        problems.append(f"{int(straddle.sum())} families span more than one split")
    if frame.split.isna().any():
        problems.append(f"{int(frame.split.isna().sum())} papers have no split")
    if frame.paper_id.duplicated().any():
        problems.append(f"{int(frame.paper_id.duplicated().sum())} duplicate paper_id")

    # the safety net: near-identical titles landing on opposite sides would mean
    # the earlier family dedup missed a pair, and a held-out score would be soft
    seen = defaultdict(set)
    for row in frame.itertuples():
        key = title_key(row.title)
        if len(key) >= 4:
            seen[key].add(row.split)
    crossing = [k for k, splits in seen.items() if len(splits) > 1]
    if crossing:
        problems.append(f"{len(crossing)} near-identical titles appear in more than "
                        f"one split - the family dedup may have missed them")
    return problems, crossing


def materialise(frame, markdown_dir, raw_dir, out_root, mode="copy"):
    """Lay the split out on disk, markdown and extraction JSON side by side.

        out_root/train/markdown/*.md      out_root/train/raw/*.json
        out_root/evaluate/...             out_root/test/...

    mode="copy" duplicates (safest), mode="link" makes hard links on the same
    volume and costs no disk. Originals are never moved or altered.
    """
    import os
    import shutil

    markdown_dir, raw_dir, out_root = Path(markdown_dir), Path(raw_dir), Path(out_root)
    counts = {}
    for name, group in frame.groupby("split"):
        placed = {"markdown": 0, "raw": 0, "no markdown": 0, "no raw": 0, "stale": 0}
        wanted = set(group.paper_id)
        for kind, source_dir, suffix in (("markdown", markdown_dir, ".md"),
                                         ("raw", raw_dir, ".json")):
            target_dir = out_root / name / kind
            target_dir.mkdir(parents=True, exist_ok=True)
            # Clear anything this split no longer owns. Without this a rerun
            # with different proportions leaves the previous copies behind, and
            # a paper ends up in two folders at once - which is leakage, not
            # clutter: the model trains on a paper it is later scored against.
            for existing in target_dir.iterdir():
                if existing.stem not in wanted:
                    existing.unlink()
                    placed["stale"] += 1
            for paper_id in group.paper_id:
                source = source_dir / f"{paper_id}{suffix}"
                if not source.is_file():
                    placed[f"no {kind}"] += 1
                    continue
                target = target_dir / source.name
                if target.exists():
                    target.unlink()
                if mode == "link":
                    try:
                        os.link(source, target)
                    except OSError:
                        shutil.copy2(source, target)     # different volume
                else:
                    shutil.copy2(source, target)
                placed[kind] += 1
        counts[name] = placed
    return counts


def write_manifest(frame, path):
    """The single source of truth every later stage reads.

    Lives inside the split folder it describes, so the partition travels as one
    thing: copy corpus_splits/ to Colab and the manifest goes with it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path

def heal(frame):
    """Pull near-duplicate titles onto the training side.

    The audit finds titles that fingerprint alike but landed in different
    splits - papers the earlier family dedup did not catch, usually the same
    group reporting adjacent work. Left alone they make a held-out score softer
    than it looks, because the model saw the twin during training.

    Moving them to train, rather than dropping them, keeps the data and costs
    only a handful of held-out papers.
    """
    _, crossing = audit(frame)
    if not crossing:
        return frame, []
    frame = frame.copy()
    keys = frame.title.map(title_key)
    moved = []
    for key in crossing:
        touched = frame[(keys == key) & (frame.split != "train")]
        for row in touched.itertuples():
            moved.append({"paper_id": row.paper_id, "was": row.split,
                          "title": str(row.title)[:70]})
        frame.loc[keys == key, "split"] = "train"
    return frame, moved
