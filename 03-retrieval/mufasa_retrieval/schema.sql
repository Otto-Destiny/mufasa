-- MUFASA evidence store, corpus schema v2.
--
-- Deliberately generic. The draft in retrieval-architecture.md fixes
-- Material / Property / Application node tables; the actual corpus is water,
-- health and environmental science whose entity types (WaterSample, Population,
-- StatisticalModel, ContaminantPlume, ...) do not fit those three. Entities are
-- therefore (name, type) rows joined to claims through a role-bearing edge, and
-- measurements stay opaque JSON because 112 claims already carry 57 distinct
-- measurement key-shapes.
--
-- One hop is a join. See 03-retrieval/docs/retrieval-v2.md.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS manifest (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper (
    paper_id        TEXT PRIMARY KEY,
    study_family_id TEXT,
    title           TEXT NOT NULL,
    authors_json    TEXT NOT NULL DEFAULT '[]',
    year            INTEGER,
    journal         TEXT,
    doi             TEXT,
    study_type      TEXT,
    -- 1 = quotable verbatim, 2 = paraphrase + citation only, 3 = coverage record only.
    -- Unknown licences default to 3; see 03-retrieval/licence-tiers.md.
    licence_tier    INTEGER NOT NULL DEFAULT 3,
    licence         TEXT,
    geographic_scope_json TEXT NOT NULL DEFAULT '[]',
    topics_json     TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS claim (
    claim_id        TEXT PRIMARY KEY,
    paper_id        TEXT NOT NULL REFERENCES paper(paper_id),
    study_family_id TEXT,
    text            TEXT NOT NULL,
    claim_type      TEXT,
    predicate       TEXT,
    measurement_json TEXT NOT NULL DEFAULT '{}',
    conditions_json TEXT NOT NULL DEFAULT '{}',
    limitations_json TEXT NOT NULL DEFAULT '[]',
    direction       TEXT,
    extraction_confidence REAL,
    review_status   TEXT
);
CREATE INDEX IF NOT EXISTS idx_claim_paper  ON claim(paper_id);
CREATE INDEX IF NOT EXISTS idx_claim_family ON claim(study_family_id);

CREATE TABLE IF NOT EXISTS evidence_span (
    span_id         TEXT PRIMARY KEY,
    claim_id        TEXT NOT NULL REFERENCES claim(claim_id),
    quote           TEXT,
    page            INTEGER,
    printed_page    INTEGER,
    section         TEXT,
    kind            TEXT
);
CREATE INDEX IF NOT EXISTS idx_span_claim ON evidence_span(claim_id);

CREATE TABLE IF NOT EXISTS entity (
    entity_id       INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    type            TEXT NOT NULL,
    norm_name       TEXT NOT NULL,
    UNIQUE (norm_name, type)
);
CREATE INDEX IF NOT EXISTS idx_entity_norm ON entity(norm_name);

CREATE TABLE IF NOT EXISTS claim_entity (
    claim_id        TEXT NOT NULL REFERENCES claim(claim_id),
    entity_id       INTEGER NOT NULL REFERENCES entity(entity_id),
    role            TEXT NOT NULL,
    PRIMARY KEY (claim_id, entity_id, role)
);
CREATE INDEX IF NOT EXISTS idx_ce_entity ON claim_entity(entity_id);

-- The property axis of the coverage gate. A claim carries the facets it can
-- actually answer for; a question carries the facets it asks for. Lexical
-- overlap alone cannot tell "same topic" from "contains the requested fact",
-- which is why retrieval-v1 scored 0/5 on the unanswerable questions.
CREATE TABLE IF NOT EXISTS claim_facet (
    claim_id        TEXT NOT NULL REFERENCES claim(claim_id),
    facet           TEXT NOT NULL,
    PRIMARY KEY (claim_id, facet)
);
CREATE INDEX IF NOT EXISTS idx_facet ON claim_facet(facet);

-- Licence tier 3: the study exists and is described, no finding is reproduced.
CREATE TABLE IF NOT EXISTS coverage (
    cov_id          TEXT PRIMARY KEY,
    paper_id        TEXT NOT NULL REFERENCES paper(paper_id),
    studied         TEXT NOT NULL,
    method          TEXT,
    place           TEXT,
    properties_measured_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS alias (
    alias           TEXT NOT NULL,
    norm_alias      TEXT NOT NULL,
    entity_name     TEXT NOT NULL,
    lang            TEXT,
    source          TEXT,
    PRIMARY KEY (norm_alias, entity_name)
);
CREATE INDEX IF NOT EXISTS idx_alias_norm ON alias(norm_alias);

CREATE TABLE IF NOT EXISTS claim_vec (
    claim_id        TEXT PRIMARY KEY REFERENCES claim(claim_id),
    dim             INTEGER NOT NULL,
    scale           REAL NOT NULL,
    vec             BLOB NOT NULL          -- int8, dim components
);

CREATE VIRTUAL TABLE IF NOT EXISTS claim_fts USING fts5(
    claim_id UNINDEXED,
    text,
    quote,
    entities,
    tokenize = 'porter unicode61'
);
