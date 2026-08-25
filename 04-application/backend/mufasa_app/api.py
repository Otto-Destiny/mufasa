"""The laptop-local service.

One request path for the desktop shell and the paired phone. It binds to
127.0.0.1 by default and works with every radio disabled; nothing here reaches
outside the machine.

Stage order is deliberate and is the one place the drafted architecture is
overridden: `sources` (paper title, journal, year) during generation,
`evidence` (tagged, quoted, cited-only) only after validation. A card shown
before the validator runs is a candidate, not evidence.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from mufasa_retrieval import connect
from mufasa_retrieval.compare import compare
from mufasa_retrieval.coverage import card, papers, statistics, total_papers
from mufasa_retrieval.generate import StubGenerator
from mufasa_retrieval.pipeline import answer, answer_stream
from mufasa_retrieval.search import load_claims

from . import integrity
from .config import BACKEND_ROOT, Settings, get_settings
from .governor import Busy, Cancelled, Governor
from .runtime import generator_status, resolve_generator

STATIC_DIR = BACKEND_ROOT / "mufasa_app" / "static"
HISTORY_PATH = BACKEND_ROOT / "history.jsonl"

CSP = (
    "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob: data:; "
    "style-src 'self' 'unsafe-inline'; script-src 'self'; font-src 'self' data:; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    governor = Governor(
        single_flight=settings.mufasa_single_flight,
        max_context_tokens=settings.mufasa_max_context_tokens,
        max_output_tokens=settings.mufasa_max_output_tokens,
        threads=settings.mufasa_llama_threads,
    )
    governor.apply_thread_caps()

    app = FastAPI(title="MUFASA", version="0.1.0", docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.governor = governor
    app.state.generator = None

    @app.on_event("startup")
    def _boot_generator() -> None:
        app.state.generator = resolve_generator(settings)

    if settings.mufasa_share_lan:
        # Only ever the bundled interface's own origin; there is no third party.
        app.add_middleware(
            CORSMiddleware,
            allow_origin_regex=r"^http://(127\.0\.0\.1|localhost|\d+\.\d+\.\d+\.\d+)(:\d+)?$",
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def _headers(request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def db() -> sqlite3.Connection:
        path = settings.db_path
        if not path.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"No evidence database at {path}. Build one with: "
                    f"uv run mufasa-build --out {path}"
                ),
            )
        conn = connect(path)
        conn.execute("PRAGMA query_only = ON")
        return conn

    def generator():  # type: ignore[no-untyped-def]
        if app.state.generator is None:
            app.state.generator = resolve_generator(settings)
        return app.state.generator

    # -- system ------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        try:
            model = settings.model_entry()
        except KeyError as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": settings.db_path.exists(),
            "database": {"path": str(settings.db_path), "present": settings.db_path.exists()},
            "model": {"key": model["key"], "label": model.get("label"), "present": model["present"]},
            "generator": settings.mufasa_generator,
            "generator_runtime": generator_status(settings),
            "busy": governor.busy,
        }

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        conn = db()
        try:
            coverage_card = card(conn).as_dict()
        finally:
            conn.close()
        voice: dict[str, Any] | None = None
        if settings.mufasa_tts_enabled:
            try:
                voice = settings.voice_entry()
            except KeyError:
                voice = None
        return {
            "model": settings.model_entry(),
            "models": settings.available_models(),
            "generator": settings.mufasa_generator,
            "generator_runtime": generator_status(settings),
            "voice": voice,
            "voices": settings.available_voices() if settings.mufasa_tts_enabled else [],
            "tts_enabled": settings.mufasa_tts_enabled,
            "limits": governor.as_dict(),
            "retrieval": {
                "top_k": settings.mufasa_top_k,
                "use_vectors": settings.mufasa_use_vectors,
                "embed_backend": settings.mufasa_embed_backend,
                "max_evidence_records": settings.mufasa_max_evidence_records,
            },
            "share_lan": settings.mufasa_share_lan,
            "corpus": coverage_card,
        }

    @app.get("/api/integrity")
    def integrity_report() -> dict[str, Any]:
        return integrity.report(settings)

    # -- corpus ------------------------------------------------------------

    @app.get("/api/corpus")
    def corpus() -> dict[str, Any]:
        conn = db()
        try:
            return card(conn).as_dict()
        finally:
            conn.close()

    @app.get("/api/statistics")
    def stats() -> dict[str, Any]:
        conn = db()
        try:
            return statistics(conn).as_dict()
        finally:
            conn.close()

    @app.get("/api/papers")
    def paper_list(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
        conn = db()
        try:
            return {"total": total_papers(conn), "papers": papers(conn, limit=limit, offset=offset)}
        finally:
            conn.close()

    @app.get("/api/claims/{claim_id}")
    def claim_detail(claim_id: str) -> dict[str, Any]:
        conn = db()
        try:
            found = load_claims(conn, [claim_id])
            if claim_id not in found:
                raise HTTPException(status_code=404, detail=f"no claim {claim_id}")
            c = found[claim_id]
            return {
                "claim_id": c.claim_id,
                "text": c.text,
                "quote": c.quote if c.licence_tier == 1 else None,
                "quote_withheld": c.licence_tier >= 2,
                "page": c.page,
                "section": c.section,
                "measurement": c.measurement,
                "conditions": c.conditions,
                "limitations": c.limitations,
                "facets": c.facets,
                "entities": c.entities,
                "study_family_id": c.study_family_id,
                "paper": {
                    "paper_id": c.paper_id,
                    "title": c.paper_title,
                    "year": c.paper_year,
                    "journal": c.paper_journal,
                    "doi": c.paper_doi,
                    "licence_tier": c.licence_tier,
                },
            }
        finally:
            conn.close()

    # -- ask ---------------------------------------------------------------

    def _run(question: str, job) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        if settings.mufasa_demo_mode:
            text = generator().generate(
                question, max_tokens=governor.max_output_tokens, temperature=0.0,
            )
            return {"question": question, "answer": text, "text": text,
                    "evidence": [], "records": [], "citations": [],
                    "validated": False, "demo_mode": True}
        conn = db()
        try:
            result = answer(
                conn,
                question,
                generator(),
                k=settings.mufasa_top_k,
                max_records=settings.mufasa_max_evidence_records,
                max_tokens=governor.max_output_tokens,
                temperature=0.0,
                use_vectors=settings.mufasa_use_vectors,
                on_stage=lambda stage, _payload: governor.checkpoint(job, stage),
            )
            return result.as_dict()
        finally:
            conn.close()

    @app.post("/api/ask")
    def ask(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        question = (payload.get("question") or "").strip()
        if not question:
            raise HTTPException(status_code=422, detail="question is required")
        try:
            job = governor.acquire(question)
        except Busy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        try:
            out = _run(question, job)
        except Cancelled as exc:
            raise HTTPException(status_code=499, detail=str(exc)) from exc
        finally:
            governor.release(job)
        _append_history(question, out)
        return out

    def _demo_events(question: str, job) -> Iterator[str]:  # type: ignore[no-untyped-def]
        """Stream tokens straight from llama-server. No database, no validator."""
        import json as _json
        import urllib.request as _url

        started = time.perf_counter()
        yield _sse("job", {"job_id": job.job_id})
        yield _sse("stage", {"stage": "generating", "demo_mode": True})

        payload = {
            "model": settings.mufasa_model,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": governor.max_output_tokens,
            "temperature": 0.0,
            "stream": True,
        }
        request = _url.Request(
            f"{settings.llama_server_url.rstrip('/')}/v1/chat/completions",
            data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        pieces: list[str] = []
        with _url.urlopen(request, timeout=240) as response:  # noqa: S310
            for raw in response:
                if job.cancelled.is_set():
                    raise Cancelled("cancelled")
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    delta = _json.loads(body)["choices"][0]["delta"].get("content") or ""
                except (KeyError, IndexError, ValueError):
                    continue
                if not delta:
                    continue
                pieces.append(delta)
                # Two shapes, so a UI that understands either will render it.
                yield _sse("token", {"text": delta})
                yield _sse("stage", {"stage": "writing", "partial": "".join(pieces)})

        answer_text = "".join(pieces).strip()
        out = {
            "question": question,
            "answer": answer_text,
            "text": answer_text,
            "evidence": [],
            "records": [],
            "citations": [],
            "validated": False,
            "demo_mode": True,
            "generator": settings.mufasa_model,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
        _append_history(question, out)
        yield _sse("answer", out)

    @app.get("/api/ask/stream")
    def ask_stream(q: str = Query(..., min_length=1)) -> StreamingResponse:
        try:
            job = governor.acquire(q)
        except Busy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        def events() -> Iterator[str]:
            started = time.perf_counter()
            try:
                if settings.mufasa_demo_mode:
                    yield from _demo_events(q, job)
                    return
                yield _sse("job", {"job_id": job.job_id})
                # Runs on a worker thread, so each stage reaches the client while
                # the model is still writing rather than all at once at the end.
                for stage, payload in answer_stream(
                    db,
                    q,
                    generator(),
                    k=settings.mufasa_top_k,
                    max_records=settings.mufasa_max_evidence_records,
                    max_tokens=governor.max_output_tokens,
                    temperature=0.0,
                    use_vectors=settings.mufasa_use_vectors,
                    cancel_check=job.cancelled.is_set,
                ):
                    job.stage = stage
                    if stage == "done":
                        payload["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
                        _append_history(q, payload)
                        yield _sse("answer", payload)
                    elif stage == "error":
                        yield _sse("error", payload)
                    else:
                        yield _sse("stage", {"stage": stage, **payload})
            except Cancelled:
                yield _sse("cancelled", {"job_id": job.job_id})
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                yield _sse("error", {"error": str(exc)})
            finally:
                governor.release(job)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/cancel")
    def cancel(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        return {"cancelled": governor.cancel(payload.get("job_id"))}

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return governor.as_dict()

    # -- compare -----------------------------------------------------------

    @app.post("/api/compare")
    def compare_studies(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        claim_ids = payload.get("claim_ids") or []
        if not claim_ids:
            raise HTTPException(status_code=422, detail="claim_ids is required")
        conn = db()
        try:
            groups = compare(conn, list(claim_ids))
            return {"groups": [g.as_dict() for g in groups]}
        finally:
            conn.close()

    # -- voice -------------------------------------------------------------

    @app.post("/api/speak")
    def speak(payload: dict[str, Any] = Body(...)) -> Response:
        from .tts import PiperVoice, VoiceUnavailable, speakable

        if not settings.mufasa_tts_enabled:
            raise HTTPException(status_code=503, detail="voice output is disabled")
        text = (payload.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=422, detail="text is required")
        try:
            entry = settings.voice_entry(payload.get("voice"))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        spoken = speakable(text, keep_citations=bool(payload.get("keep_citations")))
        voice = PiperVoice(
            settings.tts_bin,
            Path(entry["path"]),
            Path(entry["path"] + ".json") if entry.get("config") else None,
        )
        try:
            wav = voice.synthesize(spoken, length_scale=float(payload.get("length_scale", 1.0)))
        except VoiceUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(content=wav, media_type="audio/wav",
                        headers={"Cache-Control": "no-store"})

    @app.post("/api/speak/preview")
    def speak_preview(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """What the synthesiser will actually say. Testable without Piper."""
        from .tts import speakable, split_sentences

        text = (payload.get("text") or "").strip()
        spoken = speakable(text, keep_citations=bool(payload.get("keep_citations")))
        return {"spoken": spoken, "sentences": split_sentences(text)}

    # -- history and feedback ----------------------------------------------

    def _append_history(question: str, result: dict[str, Any]) -> None:
        entry = {
            "ts": time.time(),
            "question": question,
            "verdict": result.get("verdict"),
            "answerable": result.get("answerable"),
            "answer": result.get("answer"),
            "sources": result.get("sources", []),
            "cited": [r["tag"] for r in result.get("evidence", [])],
        }
        with HISTORY_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @app.get("/api/history")
    def history(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
        if not HISTORY_PATH.exists():
            return {"entries": []}
        lines = HISTORY_PATH.read_text(encoding="utf-8").splitlines()[-limit:]
        return {"entries": [json.loads(ln) for ln in reversed(lines) if ln.strip()]}

    @app.post("/api/feedback")
    def feedback(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Written to a local file. Nothing enters training automatically."""
        entry = {"ts": time.time(), **payload}
        path = settings.feedback_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return {"saved": True, "path": str(path)}

    # -- the bundled interface ---------------------------------------------

    if STATIC_DIR.exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str) -> FileResponse:
            candidate = STATIC_DIR / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()

__all__ = ["app", "create_app", "StubGenerator"]
