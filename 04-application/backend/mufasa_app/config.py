"""Configuration, all of it from the environment.

Nothing in the application hard-codes a model, a voice or a path. Swapping
Gemma for Qwen for MUFASA itself is one line in `.env`; the registry in
`models.toml` holds the rest. That is the whole point — a standard system does
not make you edit code to change a model.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
    )

    # -- model -------------------------------------------------------------
    mufasa_model: str = "qwen3-1_7b"
    mufasa_models_dir: Path = Path("./models")
    mufasa_model_registry: Path = Path("./models.toml")
    mufasa_generator: str = "llama-server"
    llama_server_url: str = "http://127.0.0.1:8080"
    llama_server_bin: Path = Path("./bin/llama-server")

    # -- resource governor -------------------------------------------------
    mufasa_max_context_tokens: int = 2048
    mufasa_max_output_tokens: int = 400
    mufasa_llama_threads: int = 4
    mufasa_single_flight: bool = True
    mufasa_max_evidence_records: int = 10

    # -- retrieval ---------------------------------------------------------
    mufasa_db: Path = Path("../../packages/corpus_v1/mufasa.db")
    mufasa_top_k: int = 10
    mufasa_use_vectors: bool = True
    mufasa_embed_backend: str = "hashing"
    mufasa_embed_model_dir: str = ""
    mufasa_embed_threads: int = 2

    # -- voice -------------------------------------------------------------
    mufasa_tts_enabled: bool = True
    mufasa_tts_voice: str = "en_US-lessac-medium"
    mufasa_tts_bin: Path = Path("./bin/piper")
    mufasa_voices_dir: Path = Path("./voices")

    # -- server ------------------------------------------------------------
    mufasa_host: str = "127.0.0.1"
    mufasa_port: int = 8756
    mufasa_share_lan: bool = False
    mufasa_feedback_path: Path = Path("./feedback.jsonl")

    registry: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @field_validator("llama_server_url")
    @classmethod
    def _loopback_only(cls, v: str) -> str:
        if not any(v.startswith(p) for p in ("http://127.0.0.1", "http://localhost", "http://[::1]")):
            raise ValueError(
                f"LLAMA_SERVER_URL must stay on loopback (got {v!r}). The evaluation runs "
                "with no network; a non-local URL is a disqualification, not a bug."
            )
        return v

    # -- resolved paths ----------------------------------------------------
    def _abs(self, p: Path) -> Path:
        return p if p.is_absolute() else (BACKEND_ROOT / p).resolve()

    @property
    def db_path(self) -> Path:
        return self._abs(self.mufasa_db)

    @property
    def models_dir(self) -> Path:
        return self._abs(self.mufasa_models_dir)

    @property
    def voices_dir(self) -> Path:
        return self._abs(self.mufasa_voices_dir)

    @property
    def registry_path(self) -> Path:
        return self._abs(self.mufasa_model_registry)

    @property
    def feedback_path(self) -> Path:
        return self._abs(self.mufasa_feedback_path)

    @property
    def tts_bin(self) -> Path:
        return self._abs(self.mufasa_tts_bin)

    # -- registry ----------------------------------------------------------
    def load_registry(self) -> dict[str, Any]:
        if self.registry:
            return self.registry
        path = self.registry_path
        if not path.exists():
            raise FileNotFoundError(f"model registry not found at {path}")
        self.registry = tomllib.loads(path.read_text(encoding="utf-8"))
        return self.registry

    def model_entry(self, key: str | None = None) -> dict[str, Any]:
        key = key or self.mufasa_model
        models = self.load_registry().get("models", {})
        if key not in models:
            raise KeyError(
                f"MUFASA_MODEL={key!r} is not in {self.registry_path.name}. "
                f"Available: {', '.join(sorted(models)) or '(none)'}"
            )
        entry = dict(models[key])
        entry["key"] = key
        entry["path"] = str(self.models_dir / entry["file"])
        entry["present"] = (self.models_dir / entry["file"]).exists()
        return entry

    def voice_entry(self, key: str | None = None) -> dict[str, Any]:
        key = key or self.mufasa_tts_voice
        voices = self.load_registry().get("voices", {})
        if key not in voices:
            raise KeyError(
                f"MUFASA_TTS_VOICE={key!r} is not in {self.registry_path.name}. "
                f"Available: {', '.join(sorted(voices)) or '(none)'}"
            )
        entry = dict(voices[key])
        entry["key"] = key
        entry["path"] = str(self.voices_dir / entry["file"])
        entry["present"] = (self.voices_dir / entry["file"]).exists()
        return entry

    def available_models(self) -> list[dict[str, Any]]:
        return [self.model_entry(k) for k in sorted(self.load_registry().get("models", {}))]

    def available_voices(self) -> list[dict[str, Any]]:
        return [self.voice_entry(k) for k in sorted(self.load_registry().get("voices", {}))]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
