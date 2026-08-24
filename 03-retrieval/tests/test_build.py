"""Build-plane determinism and manifest correctness."""

from __future__ import annotations

import hashlib
from pathlib import Path

from mufasa_retrieval import build, connect, manifest


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def test_build_is_deterministic(claims_path, papers_path, tmp_path: Path) -> None:
    aliases = Path(__file__).resolve().parents[1] / "mufasa_retrieval" / "aliases" / "flagship.yaml"
    a = tmp_path / "a.db"
    b = tmp_path / "b.db"
    kwargs = dict(
        claims_path=claims_path,
        papers_path=papers_path,
        corpus_version="corpus_test",
        aliases_path=aliases,
    )
    build(db_path=a, **kwargs)
    build(db_path=b, **kwargs)
    assert _file_sha(a) == _file_sha(b)


def test_manifest_records_counts(db_path: Path) -> None:
    conn = connect(db_path)
    man = manifest(conn)
    conn.close()
    assert int(man["count_papers"]) == 10
    assert int(man["count_claims"]) == 112
    assert man["corpus_version"] == "corpus_test"
    assert "embedder" in man
