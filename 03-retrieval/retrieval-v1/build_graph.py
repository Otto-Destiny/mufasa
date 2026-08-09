"""
Builds the milestone-1 retrieval graph from papers.jsonl / claims.jsonl into
a SQLite + FTS5 database.

Why SQLite+FTS5 instead of Ladybug for this milestone: see 03-retrieval/README
or the conversation this was built from. Short version: it's the architecture
doc's own named fallback (retrieval-architecture.md, "SQLite + FTS5"), needs
no extension binaries, and one hop is a join -- which is all this design
needs. Ladybug packaging is real work for the full 6,000-paper build, not
something the 10-paper proof-of-concept benefits from.

Entities are stored as a generic Entity{name, type} node rather than the
fixed Material/Property/Application tables from the architecture doc,
because this test corpus (water/health/environmental) doesn't use those
categories at all -- see the schema survey in the conversation.
"""
import json
import os
import sqlite3

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(BASE, "..", "milestone1-test-data"))
DB_PATH = os.path.join(BASE, "retrieval.sqlite")

SCHEMA = """
CREATE TABLE paper (
    id TEXT PRIMARY KEY,
    benchmark_id TEXT,
    study_family_id TEXT,
    title TEXT,
    authors TEXT,
    year INTEGER,
    doi TEXT,
    study_type TEXT,
    license TEXT,
    local_pdf TEXT
);

CREATE TABLE claim (
    id TEXT PRIMARY KEY,
    paper_id TEXT REFERENCES paper(id),
    study_family_id TEXT,
    claim_type TEXT,
    predicate TEXT,
    text TEXT,
    direction TEXT,
    measurement_json TEXT,
    conditions_json TEXT,
    limitations_json TEXT,
    quote TEXT,
    page INTEGER,
    section TEXT,
    extraction_confidence REAL,
    review_status TEXT
);

CREATE TABLE entity (
    name TEXT,
    type TEXT,
    PRIMARY KEY (name, type)
);

CREATE TABLE claim_entity (
    claim_id TEXT REFERENCES claim(id),
    entity_name TEXT,
    entity_type TEXT,
    role TEXT
);

CREATE INDEX idx_claim_entity_claim ON claim_entity(claim_id);
CREATE INDEX idx_claim_entity_entity ON claim_entity(entity_name, entity_type);
CREATE INDEX idx_claim_paper ON claim(paper_id);

CREATE VIRTUAL TABLE claim_fts USING fts5(
    claim_id UNINDEXED,
    text,
    quote,
    entities,
    tokenize = 'porter unicode61'
);
"""


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)

    n_papers = 0
    for p in load_jsonl(os.path.join(DATA, "papers.jsonl")):
        con.execute(
            "INSERT INTO paper VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                p["paper_id"], p.get("benchmark_id"), p.get("study_family_id"),
                p.get("title"), json.dumps(p.get("authors", [])), p.get("year"),
                p.get("doi"), p.get("study_type"), p.get("license"), p.get("local_pdf"),
            ),
        )
        n_papers += 1

    n_claims = 0
    n_entity_links = 0
    for c in load_jsonl(os.path.join(DATA, "claims.jsonl")):
        evidence = c.get("evidence") or {}
        quote = c.get("quote") or evidence.get("quote")
        page = c.get("page") if c.get("page") is not None else evidence.get("pdf_page")
        section = c.get("section") or evidence.get("section")

        con.execute(
            "INSERT INTO claim VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                c["id"], c.get("paper_id"), c.get("study_family_id"), c.get("claim_type"),
                c.get("predicate"), c.get("text"), c.get("direction"),
                json.dumps(c["measurement"]) if c.get("measurement") else None,
                json.dumps(c["conditions"]) if c.get("conditions") else None,
                json.dumps(c["limitations"]) if c.get("limitations") else None,
                quote, page, section,
                c.get("extraction_confidence"), c.get("review_status"),
            ),
        )
        n_claims += 1

        ent_names = []
        for e in c.get("entities", []):
            name, etype, role = e.get("name"), e.get("type"), e.get("role")
            if not name:
                continue
            con.execute("INSERT OR IGNORE INTO entity VALUES (?,?)", (name, etype))
            con.execute(
                "INSERT INTO claim_entity VALUES (?,?,?,?)",
                (c["id"], name, etype, role),
            )
            ent_names.append(name)
            n_entity_links += 1

        con.execute(
            "INSERT INTO claim_fts (claim_id, text, quote, entities) VALUES (?,?,?,?)",
            (c["id"], c.get("text") or "", quote or "", " ".join(ent_names)),
        )

    con.commit()
    n_entities = con.execute("SELECT COUNT(*) FROM entity").fetchone()[0]
    con.close()
    print(f"Loaded {n_papers} papers, {n_claims} claims, "
          f"{n_entities} distinct entities, {n_entity_links} claim-entity links")
    print(f"Database: {DB_PATH}")


if __name__ == "__main__":
    build()
