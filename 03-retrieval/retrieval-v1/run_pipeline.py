"""
End-to-end: question -> retrieve -> tagged prompt -> generate -> validate.
Run against a chosen subset of questions.jsonl to check whether generation
+ validation catches what retrieval-time abstention could not (see the
conversation this was built from for why a BM25/coverage threshold cannot
separate answerable from unanswerable in this test set).
"""
import argparse
import json
import os

from generate import LlamaServer, generate
from prompt import build_prompt, evidence_texts, format_evidence_block
from retrieve import connect, retrieve
from validate import validate

DATA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "milestone1-test-data"))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def answer_question(con, server, question_text, k=8):
    ranked, _ = retrieve(con, question_text, k=k)
    claim_ids = [cid for cid, _ in ranked]
    evidence_block, tag_map = format_evidence_block(con, claim_ids)
    ev_texts = evidence_texts(con, tag_map)
    prompt = build_prompt(question_text, evidence_block)
    answer = generate(prompt, server)
    result = validate(answer, tag_map, ev_texts)
    return answer, result, tag_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None, help="question IDs to run, default: a fixed sample")
    args = ap.parse_args()

    target_ids = set(args.ids) if args.ids else {
        "Q-026", "Q-027", "Q-028", "Q-029", "Q-030",  # unanswerable
        "Q-001", "Q-009", "Q-023",                     # answerable sanity check
    }

    con = connect()
    questions = {q["id"]: q for q in load_jsonl(os.path.join(DATA, "questions.jsonl"))}

    with LlamaServer() as server:
        for qid in sorted(target_ids):
            q = questions[qid]
            answer, result, tag_map = answer_question(con, server, q["question"])
            print("=" * 90)
            print(f"{q['id']} (answerable={q['answerable']}): {q['question']}")
            print(f"-> ANSWER: {answer}")
            print(f"-> VALIDATION: {result}")


if __name__ == "__main__":
    main()
