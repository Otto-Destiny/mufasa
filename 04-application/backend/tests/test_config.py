"""Model registry and env-driven settings."""

from __future__ import annotations

from pathlib import Path

import pytest
from mufasa_app.config import Settings, reset_settings_cache


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    reset_settings_cache()
    yield
    reset_settings_cache()


def test_unknown_model_key_fails_loudly(tmp_path: Path) -> None:
    registry = tmp_path / "models.toml"
    registry.write_text(
        '[models.qwen3-1_7b]\nfile = "qwen.gguf"\nlabel = "Qwen"\n',
        encoding="utf-8",
    )
    settings = Settings(
        mufasa_model="does-not-exist",
        mufasa_model_registry=registry,
        mufasa_models_dir=tmp_path,
    )
    with pytest.raises(KeyError, match="does-not-exist"):
        settings.model_entry()


def test_valid_model_key_resolves_path(tmp_path: Path) -> None:
    registry = tmp_path / "models.toml"
    registry.write_text(
        '[models.gemma-3-270m]\nfile = "gemma.gguf"\nlabel = "Gemma"\n',
        encoding="utf-8",
    )
    (tmp_path / "gemma.gguf").write_bytes(b"fake")
    settings = Settings(
        mufasa_model="gemma-3-270m",
        mufasa_model_registry=registry,
        mufasa_models_dir=tmp_path,
    )
    entry = settings.model_entry()
    assert entry["present"] is True
    assert entry["label"] == "Gemma"


def test_llama_server_url_must_be_loopback() -> None:
    with pytest.raises(ValueError, match="loopback"):
        Settings(llama_server_url="http://example.com:8080")
