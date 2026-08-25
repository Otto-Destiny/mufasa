import json
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.entity_resolution.cli import _load_embedding
from scripts.entity_resolution.io import canonical_json_hash, sha256_file


def test_embedding_vector_artifact_tampering_fails_closed(workspace_tmp):
    mention_path = workspace_tmp / "mention_vectors.parquet"
    target_path = workspace_tmp / "target_vectors.parquet"
    manifest_path = workspace_tmp / "embedding_manifest.json"
    pd.DataFrame([{"mention_id": "M1", "vector": [1.0, 0.0]}]).to_parquet(
        mention_path, index=False
    )
    pd.DataFrame([{"target_id": "C1", "vector": [1.0, 0.0]}]).to_parquet(
        target_path, index=False
    )
    descriptor = {
        "model_id": "offline/test-encoder",
        "model_hash": "a" * 64,
        "mention_vectors_sha256": sha256_file(mention_path),
        "target_vectors_sha256": sha256_file(target_path),
    }
    manifest_path.write_text(
        json.dumps({**descriptor, "vector_set_fingerprint": canonical_json_hash(descriptor)}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        embedding_manifest=str(manifest_path),
        mention_vectors=str(mention_path),
        target_vectors=str(target_path),
    )
    backend = _load_embedding(args)
    assert backend.vector_set_fingerprint == canonical_json_hash(descriptor)

    pd.DataFrame([{"mention_id": "M1", "vector": [0.0, 1.0]}]).to_parquet(
        mention_path, index=False
    )
    with pytest.raises(SystemExit, match="SHA-256"):
        _load_embedding(args)

