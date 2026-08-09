"""
Scores retrieve.py against questions.jsonl per milestone1.md Task 6:
  1. Recall@10 on answerable questions.
  2. Abstention behaviour on unanswerable questions.

Caveat: the abstention threshold is calibrated on this same 30-question
set (there's no held-out dev/test split at this scale), so the abstention
accuracy below is optimistic, not a generalisation estimate. Fine for a
milestone-1 proof; needs a real split once the corpus grows.
"""
import json
import os
import sqlite3

from retrieve import DB_PATH, retrieve

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(BASE, "..", "milestone1-test-data"))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    con = sqlite3.connect(DB_PATH)
    questions = list(load_jsonl(os.path.join(DATA, "questions.jsonl")))

    results = []
    for q in questions:
        ranked, best_score = retrieve(con, q["question"], k=10)
        results.append((q, ranked, best_score))

    ans_scores = [r[2] for r in results if r[0]["answerable"] and r[2] is not None]
    unans_scores = [r[2] for r in results if not r[0]["answerable"] and r[2] is not None]
    print("answerable best_scores:  ", sorted(round(s, 2) for s in ans_scores))
    print("unanswerable best_scores:", sorted(round(s, 2) for s in unans_scores))

    candidates = sorted(set(ans_scores + unans_scores))
    best_thr, best_acc = None, -1
    for thr in candidates:
        correct = sum(
            1
            for q, _, best_score in results
            if ((best_score is not None and best_score <= thr) == bool(q["answerable"]))
        )
        acc = correct / len(results)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    print(f"\nCalibrated abstention threshold (bm25 <= thr => answer): "
          f"{best_thr:.3f}  (train-set accuracy={best_acc:.3f})")

    hit = 0
    total = 0
    rows = []
    for q, ranked, best_score in results:
        if not q["answerable"]:
            continue
        total += 1
        ranked_ids = {cid for cid, _ in ranked}
        expected = set(q["expected_claim_ids"])
        found = expected & ranked_ids
        is_hit = len(found) > 0
        hit += is_hit
        rows.append((q["id"], is_hit, len(found), len(expected), best_score))

    print(f"\nRecall@10 (>=1 expected claim found in top 10): {hit}/{total} = {hit/total:.3f}")

    correct_abstain = 0
    unans_total = 0
    for q, ranked, best_score in results:
        if q["answerable"]:
            continue
        unans_total += 1
        predicted_answerable = best_score is not None and best_score <= best_thr
        status = "WRONGLY ANSWERED" if predicted_answerable else "correctly abstained"
        print(f"  {q['id']} (unanswerable): best_bm25={best_score} -> {status}")
        if not predicted_answerable:
            correct_abstain += 1
    print(f"\nCorrect abstention on unanswerable: {correct_abstain}/{unans_total}")

    print("\nPer-question detail (answerable only):")
    for qid, is_hit, found_n, exp_n, score in rows:
        score_str = f"{score:.3f}" if score is not None else "no_fts_match"
        mark = "OK  " if is_hit else "MISS"
        print(f"  [{mark}] {qid}: found={found_n}/{exp_n} best_bm25={score_str}")


if __name__ == "__main__":
    main()
