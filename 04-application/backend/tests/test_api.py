"""API contract: SSE order, sources carry no quotes, single-flight."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mufasa_app.api import create_app
from mufasa_app.config import Settings, reset_settings_cache
from mufasa_retrieval import build


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data = Path(__file__).resolve().parents[3] / "03-retrieval" / "milestone1-test-data"
    aliases = (
        Path(__file__).resolve().parents[3]
        / "03-retrieval"
        / "mufasa_retrieval"
        / "aliases"
        / "flagship.yaml"
    )
    out = tmp_path_factory.mktemp("api") / "mufasa.db"
    build(
        claims_path=data / "claims.jsonl",
        papers_path=data / "papers.jsonl",
        db_path=out,
        corpus_version="corpus_test",
        aliases_path=aliases,
    )
    return out


@pytest.fixture
def client(db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reset_settings_cache()
    registry = Path(__file__).resolve().parents[1] / "models.toml"
    settings = Settings(
        mufasa_db=db_path,
        mufasa_generator="stub",
        mufasa_tts_enabled=False,
        mufasa_model_registry=registry,
        mufasa_models_dir=tmp_path,
        mufasa_feedback_path=tmp_path / "feedback.jsonl",
    )
    app = create_app(settings)
    with TestClient(app) as c:
        yield c
    reset_settings_cache()


def test_health_ok(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_ask_returns_evidence_only_after_answer(client: TestClient) -> None:
    r = client.post("/api/ask", json={"question": "What was the turbidity of borehole water?"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "evidence" in body
    assert "sources" in body
    for src in body["sources"]:
        assert "quote" not in src
        assert "tag" not in src


def test_ask_stream_order(client: TestClient) -> None:
    events = []
    with client.stream(
        "GET", "/api/ask/stream", params={"q": "What was the turbidity of borehole water?"}
    ) as resp:
        assert resp.status_code == 200
        event_name = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and event_name:
                events.append(event_name)
                event_name = None
    assert "job" in events
    assert "stage" in events
    assert "answer" in events
    # evidence is inside the final answer payload, never a pre-generation card event
    assert "evidence" not in events


def test_single_flight_conflict(client: TestClient) -> None:
    # Hold the governor busy by acquiring through the app state.
    gov = client.app.state.governor
    job = gov.acquire("held")
    try:
        r = client.post("/api/ask", json={"question": "second"})
        assert r.status_code == 409
    finally:
        gov.release(job)


def test_speak_preview(client: TestClient) -> None:
    r = client.post(
        "/api/speak/preview",
        json={"text": "Conductivity was 39.97 mg/L [E1]."},
    )
    assert r.status_code == 200
    assert "[E1]" not in r.json()["spoken"]
    assert "milligrams" in r.json()["spoken"]
