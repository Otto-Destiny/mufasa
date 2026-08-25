from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


MODULE_DIR = Path(__file__).parents[1] / "01-data-engineering" / "data-extraction"
sys.path.insert(0, str(MODULE_DIR))
SPEC = importlib.util.spec_from_file_location(
    "mufasa_citations_test", MODULE_DIR / "mufasa_citations.py",
)
citations = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = citations
SPEC.loader.exec_module(citations)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    markdown = tmp_path / "markdown"
    markdown.mkdir()
    manifest = tmp_path / "manifest.parquet"
    documents = tmp_path / "documents.parquet"
    authors = tmp_path / "authors.parquet"

    pd.DataFrame(
        [
            {"paper_id": "W100", "title": "Rice communities in Ogun"},
            {"paper_id": "W200", "title": "Water quality in Sokoto"},
            {"paper_id": "W201", "title": "Water treatment in Sokoto"},
            {"paper_id": "W300", "title": "Cassava yields in Enugu"},
            {"paper_id": "W400", "title": "A manifest-only paper"},
        ]
    ).to_parquet(manifest, index=False)
    pd.DataFrame(
        [
            {
                "paper_id": "W100",
                "openalex_id": "https://openalex.org/W100",
                "title": "Rice communities in Ogun",
                "publication_date": "2022-07-07",
                "markdown_path": "/stale/kaggle/W100.md",
            },
            {
                "paper_id": "W200",
                "openalex_id": "https://openalex.org/W200",
                "title": "Water quality in Sokoto",
                "publication_date": "2020-01-01",
                "markdown_path": "",
            },
            {
                "paper_id": "W201",
                "openalex_id": "https://openalex.org/W201",
                "title": "Water treatment in Sokoto",
                "publication_date": "2020-02-01",
                "markdown_path": "",
            },
            {
                "paper_id": "W300",
                "openalex_id": "https://openalex.org/W300",
                "title": "Cassava yields in Enugu",
                "publication_date": "2019",
                "markdown_path": "",
            },
        ]
    ).to_parquet(documents, index=False)
    pd.DataFrame(
        [
            {
                "openalex_id": "https://openalex.org/W100",
                "authors_json": json.dumps(["Sammy Olufemi Sam-Wobo", "Olushola Akintola"]),
            },
            {
                "openalex_id": "https://openalex.org/W200",
                "authors_json": json.dumps(["Ada Njoku", "Binta Musa", "Chidi Okafor"]),
            },
            {
                "openalex_id": "https://openalex.org/W201",
                "authors_json": json.dumps(["Ada Njoku", "Kemi Bello", "Femi Yusuf"]),
            },
            {
                "openalex_id": "https://openalex.org/W300",
                "authors_json": json.dumps(["Jane Alpha", "John Beta", "Alice Gamma"]),
            },
        ]
    ).to_parquet(authors, index=False)

    (markdown / "W100.md").write_text(
        """<!-- MUFASA_PDF_PAGE: 1 -->
# Rice communities in Ogun
#### Sammy Olufemi Sam-Wobo, Olushola Akintola
Journal of Rural Health - 2013, Volume 10, Number 2
## Abstract
Field work occurred from 2009 to 2010.
<!-- MUFASA_PDF_PAGE: 2 -->
References from 2022 do not belong to page one.
""",
        encoding="utf-8",
    )
    for paper_id, title in (("W200", "Water quality in Sokoto"), ("W201", "Water treatment in Sokoto")):
        (markdown / f"{paper_id}.md").write_text(
            f"""<!-- MUFASA_PDF_PAGE: 1 -->
# {title}
#### Ada Njoku, Other Authors
Published 2020
## Abstract
""",
            encoding="utf-8",
        )
    (markdown / "W300.md").write_text(
        """<!-- MUFASA_PDF_PAGE: 1 -->
# Cassava yields in Enugu
#### John Beta, Jane Alpha, Alice Gamma
Published in 2019
## Abstract
""",
        encoding="utf-8",
    )
    return {
        "split_manifest": manifest,
        "documents_path": documents,
        "authors_cache_path": authors,
        "markdown_root": markdown,
    }


def test_document_year_corrects_openalex_without_dropping_rows(tmp_path: Path):
    frame = citations.prepare_citation_metadata(**_inputs(tmp_path))

    assert frame["paper_id"].tolist() == ["W100", "W200", "W201", "W300", "W400"]
    row = frame.set_index("paper_id").loc["W100"]
    assert row["openalex_label"] == "Sam-Wobo & Akintola, 2022"
    assert row["citation_label"] == "Sam-Wobo & Akintola, 2013"
    assert row["author_status"] == citations.STATUS_VERIFIED
    assert row["year_status"] == citations.STATUS_CORRECTED
    assert row["citation_status"] == citations.STATUS_CORRECTED
    assert row["audit_mode"] == "AUDIT_ONLY"


def test_document_byline_can_correct_openalex_order(tmp_path: Path):
    frame = citations.prepare_citation_metadata(**_inputs(tmp_path)).set_index("paper_id")
    row = frame.loc["W300"]

    assert row["openalex_label"] == "Alpha et al., 2019"
    assert row["citation_label"] == "Beta et al., 2019"
    assert row["document_first_author_family"] == "Beta"
    assert row["author_status"] == citations.STATUS_CORRECTED


def test_fallback_is_observed_but_never_filters(tmp_path: Path):
    frame = citations.prepare_citation_metadata(
        **_inputs(tmp_path), paper_ids=["W100", "W400", "W999"],
    ).set_index("paper_id")

    assert set(frame.index) == {"W100", "W400", "W999"}
    assert bool(frame.loc["W400", "fallback_used"])
    assert frame.loc["W400", "citation_status"] == citations.STATUS_INVALID
    assert frame.loc["W999", "citation_parenthetical"] == "(Unattributed study, n.d.)"


def test_collision_counts_are_audit_metadata_only(tmp_path: Path):
    frame = citations.prepare_citation_metadata(**_inputs(tmp_path)).set_index("paper_id")

    assert frame.loc["W200", "citation_label"] == "Njoku et al., 2020"
    assert frame.loc["W201", "citation_label"] == "Njoku et al., 2020"
    assert frame.loc["W200", "collision_count"] == 2
    assert frame.loc["W201", "collision_count"] == 2


def test_atomic_parquet_round_trip_and_lookup(tmp_path: Path):
    paths = _inputs(tmp_path)
    output = tmp_path / "citation_metadata.parquet"
    frame = citations.prepare_citation_metadata(**paths, output_path=output)
    lookup = citations.load_citation_metadata(output)

    assert pq.ParquetFile(output).schema_arrow == citations.CITATION_SCHEMA
    assert len(lookup) == len(frame)
    assert citations.citation_for_paper("https://openalex.org/W100", lookup)["citation_label"].endswith("2013")
    fallback = citations.citation_for_paper("W999", lookup, title="Unknown field study")
    assert fallback["fallback_used"] is True
    assert "Unknown field study" in fallback["citation_label"]
