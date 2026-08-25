"""Build plane: JSONL records -> the shipped evidence store.

Runs on your machine with as much time as it needs. Nothing here happens on the
judging laptop. The output is a single SQLite file plus a manifest, both hashed.

Determinism is a requirement, not a nicety: entity ids are assigned in sorted
order and the manifest records every input hash, so the same records always
produce the same database and `test_build.py` can assert it.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import properties
from .embed import DIM, get_embedder, quantize
from .normalize import norm

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
CORPUS_SCHEMA_VERSION = "corpus-schema-v2"


# Copyright protects expression, not facts; but we still only ship verbatim text
# for licences that allow it. Unknown maps to tier 3, the conservative default.
# See 03-retrieval/licence-tiers.md.
def licence_tier(licence: str | None) -> int:
    if not licence:
        return 3
    lic = norm(licence)
    if any(k in lic for k in ("cc0", "public domain", "cc by", "cc-by", "creative commons attribution")):
        if "nd" in lic.split() or "no deriv" in lic:
            return 3
        return 1
    if "open access" in lic:
        return 1
    if any(k in lic for k in ("tdm", "text and data mining", "training permitted")):
        return 2
    return 3


@dataclass(frozen=True)
class BuildStats:
    papers: int
    claims: int
    spans: int
    entities: int
    facets: int
    vectors: int
    coverage: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _span_of(claim: dict[str, Any]) -> dict[str, Any]:
    ev = claim.get("evidence") or {}
    return {
        "quote": ev.get("quote") or claim.get("quote"),
        "page": ev.get("pdf_page") if ev.get("pdf_page") is not None else claim.get("page"),
        "printed_page": ev.get("printed_page"),
        "section": ev.get("section") or claim.get("section"),
        "kind": ev.get("kind") or "text",
    }


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def build(
    *,
    claims_path: str | Path,
    papers_path: str | Path,
    db_path: str | Path,
    corpus_version: str = "corpus_v1",
    embed_backend: str | None = None,
    aliases_path: str | Path | None = None,
) -> BuildStats:
    claims_path, papers_path, db_path = Path(claims_path), Path(papers_path), Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    for sidecar in (db_path.with_suffix(db_path.suffix + "-wal"),
                    db_path.with_suffix(db_path.suffix + "-shm")):
        if sidecar.exists():
            sidecar.unlink()

    papers = _read_jsonl(papers_path)
    claims = _read_jsonl(claims_path)

    conn = connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    # -- papers ------------------------------------------------------------
    conn.executemany(
        """INSERT INTO paper (paper_id, study_family_id, title, authors_json, year, journal,
                              doi, study_type, licence_tier, licence,
                              geographic_scope_json, topics_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                p["paper_id"],
                p.get("study_family_id"),
                p.get("title", ""),
                json.dumps(p.get("authors", []), ensure_ascii=False),
                p.get("year"),
                p.get("journal"),
                p.get("doi"),
                p.get("study_type"),
                licence_tier(p.get("license")),
                p.get("license"),
                json.dumps(p.get("geographic_scope", []), ensure_ascii=False),
                json.dumps(p.get("topics", []), ensure_ascii=False),
            )
            for p in sorted(papers, key=lambda p: p["paper_id"])
        ],
    )

    # -- entities (deterministic ids) --------------------------------------
    pairs: set[tuple[str, str, str]] = set()
    for c in claims:
        for e in c.get("entities") or []:
            name, typ = (e.get("name") or "").strip(), (e.get("type") or "Unknown").strip()
            if name:
                pairs.add((norm(name), typ, name))
    ordered = sorted(pairs, key=lambda t: (t[0], t[1]))
    entity_id = {(n, t): i + 1 for i, (n, t, _) in enumerate(ordered)}
    conn.executemany(
        "INSERT INTO entity (entity_id, name, type, norm_name) VALUES (?,?,?,?)",
        [(entity_id[(n, t)], raw, t, n) for n, t, raw in ordered],
    )

    # -- claims, spans, edges, facets --------------------------------------
    claim_rows, span_rows, edge_rows, facet_rows, fts_rows = [], [], [], [], []
    for c in sorted(claims, key=lambda c: c["id"]):
        cid = c["id"]
        claim_rows.append(
            (
                cid,
                c["paper_id"],
                c.get("study_family_id"),
                c.get("text", ""),
                c.get("claim_type"),
                c.get("predicate"),
                json.dumps(c.get("measurement") or {}, ensure_ascii=False, sort_keys=True),
                json.dumps(c.get("conditions") or {}, ensure_ascii=False, sort_keys=True),
                json.dumps(c.get("limitations") or [], ensure_ascii=False),
                c.get("direction"),
                c.get("extraction_confidence"),
                c.get("review_status"),
            )
        )
        span = _span_of(c)
        span_rows.append(
            (f"{cid}-S1", cid, span["quote"], span["page"], span["printed_page"],
             span["section"], span["kind"])
        )
        names = []
        for e in c.get("entities") or []:
            name, typ = (e.get("name") or "").strip(), (e.get("type") or "Unknown").strip()
            if not name:
                continue
            names.append(name)
            edge_rows.append((cid, entity_id[(norm(name), typ)], e.get("role") or "unspecified"))
        for facet in sorted(properties.claim_facets(c)):
            facet_rows.append((cid, facet))
        fts_rows.append((cid, c.get("text", ""), span["quote"] or "", " ; ".join(names)))

    conn.executemany(
        """INSERT INTO claim (claim_id, paper_id, study_family_id, text, claim_type, predicate,
                              measurement_json, conditions_json, limitations_json, direction,
                              extraction_confidence, review_status)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        claim_rows,
    )
    conn.executemany(
        """INSERT INTO evidence_span (span_id, claim_id, quote, page, printed_page, section, kind)
           VALUES (?,?,?,?,?,?,?)""",
        span_rows,
    )
    conn.executemany(
        "INSERT OR IGNORE INTO claim_entity (claim_id, entity_id, role) VALUES (?,?,?)", edge_rows
    )
    conn.executemany("INSERT OR IGNORE INTO claim_facet (claim_id, facet) VALUES (?,?)", facet_rows)
    conn.executemany(
        "INSERT INTO claim_fts (claim_id, text, quote, entities) VALUES (?,?,?,?)", fts_rows
    )

    # -- aliases -----------------------------------------------------------
    if aliases_path:
        _load_aliases(conn, Path(aliases_path))

    # -- vectors -----------------------------------------------------------
    embedder = get_embedder(embed_backend)
    # Embed the claim sentence together with its entity names: the dense channel
    # earns its place on paraphrase, and the entity wording is what a paraphrase
    # usually keeps.
    texts = [f"{row[3]} — {fts[3]}" for row, fts in zip(claim_rows, fts_rows, strict=True)]
    vectors = embedder.encode(texts)
    q, scale = quantize(vectors)
    conn.executemany(
        "INSERT INTO claim_vec (claim_id, dim, scale, vec) VALUES (?,?,?,?)",
        [
            (row[0], int(q.shape[1]) if q.size else DIM, scale, q[i].tobytes())
            for i, row in enumerate(claim_rows)
        ],
    )

    # -- coverage records (licence tier 3) ---------------------------------
    cov_rows = []
    tiers = dict(conn.execute("SELECT paper_id, licence_tier FROM paper").fetchall())
    for i, p in enumerate(sorted(papers, key=lambda p: p["paper_id"]), start=1):
        if tiers.get(p["paper_id"], 3) >= 3:
            cov_rows.append(
                (
                    f"COV-{i:04d}",
                    p["paper_id"],
                    p.get("title", ""),
                    p.get("study_type"),
                    "; ".join(p.get("geographic_scope") or []) or None,
                    json.dumps(p.get("topics") or [], ensure_ascii=False),
                )
            )
    conn.executemany(
        """INSERT INTO coverage (cov_id, paper_id, studied, method, place,
                                 properties_measured_json) VALUES (?,?,?,?,?,?)""",
        cov_rows,
    )

    # -- manifest ----------------------------------------------------------
    counts = {
        "papers": len(papers),
        "claims": len(claim_rows),
        "spans": len(span_rows),
        "entities": len(ordered),
        "facets": len(facet_rows),
        "vectors": len(claim_rows),
        "coverage": len(cov_rows),
    }
    years = [p.get("year") for p in papers if p.get("year")]
    manifest = {
        "corpus_version": corpus_version,
        "schema_version": CORPUS_SCHEMA_VERSION,
        "facet_vocabulary": properties.FACET_VOCABULARY_VERSION,
        "embedder": embedder.name,
        "embedding_dim": str(DIM),
        "claims_sha256": _sha256(claims_path),
        "papers_sha256": _sha256(papers_path),
        "year_min": str(min(years)) if years else "",
        "year_max": str(max(years)) if years else "",
        **{f"count_{k}": str(v) for k, v in counts.items()},
    }
    conn.executemany(
        "INSERT INTO manifest (key, value) VALUES (?,?)", sorted(manifest.items())
    )

    conn.commit()
    conn.execute("PRAGMA optimize")
    conn.close()
    return BuildStats(**counts)


def _load_aliases(conn: sqlite3.Connection, path: Path) -> None:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = []
    for canonical, spec in (data.get("entities") or {}).items():
        for alias in spec.get("aliases", []) or []:
            rows.append((alias, norm(alias), canonical, spec.get("lang"), str(path.name)))
        rows.append((canonical, norm(canonical), canonical, spec.get("lang"), str(path.name)))
    conn.executemany(
        """INSERT OR IGNORE INTO alias (alias, norm_alias, entity_name, lang, source)
           VALUES (?,?,?,?,?)""",
        rows,
    )


def manifest(conn: sqlite3.Connection) -> dict[str, str]:
    return {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM manifest")}


def vector_matrix(conn: sqlite3.Connection) -> tuple[list[str], np.ndarray, float]:
    """Load every int8 claim vector as one contiguous matrix."""
    ids, blobs, scale, dim = [], [], 1.0, DIM
    for row in conn.execute("SELECT claim_id, dim, scale, vec FROM claim_vec ORDER BY claim_id"):
        ids.append(row["claim_id"])
        blobs.append(np.frombuffer(row["vec"], dtype=np.int8))
        scale, dim = row["scale"], row["dim"]
    if not ids:
        return [], np.zeros((0, dim), dtype=np.int8), 1.0
    return ids, np.vstack(blobs), float(scale)
