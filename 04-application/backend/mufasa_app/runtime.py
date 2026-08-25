"""Local llama-server lifecycle for the app process.

Prefers a real GGUF via llama-server. Falls back to the stub generator only when
the model file or server binary is missing, so the UI still works on a fresh
checkout.
"""

from __future__ import annotations

import atexit
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mufasa_retrieval.generate import Generator, LlamaServerGenerator, StubGenerator, get_generator

from .config import Settings

_llama_proc: subprocess.Popen[bytes] | None = None


def llama_server_bin(settings: Settings) -> Path:
    """Resolve llama-server, including the Windows .exe next to the configured path."""
    configured = settings._abs(settings.llama_server_bin)
    candidates = [configured]
    if configured.suffix.lower() != ".exe":
        candidates.append(configured.with_name(configured.name + ".exe"))
    bin_dir = configured.parent
    candidates.extend(
        [
            bin_dir / "llama-server.exe",
            bin_dir / "llama-server",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return configured


def _health(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=timeout) as resp:  # noqa: S310
            return 200 <= resp.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        try:
            with urllib.request.urlopen(  # noqa: S310
                f"{url.rstrip('/')}/v1/models", timeout=timeout
            ) as resp:
                return 200 <= resp.status < 500
        except (urllib.error.URLError, TimeoutError, OSError):
            return False


def ensure_llama_server(settings: Settings) -> bool:
    """Start llama-server if needed. Returns True when it looks reachable."""
    global _llama_proc
    url = settings.llama_server_url
    if _health(url):
        return True

    model = settings.model_entry()
    if not model.get("present"):
        print(
            f"No GGUF at {model['path']}. Place the file from models.toml under "
            f"{settings.models_dir}, then restart. Using evidence-only answers until then.",
            file=sys.stderr,
        )
        return False

    bin_path = llama_server_bin(settings)
    if not bin_path.exists():
        print(
            f"llama-server binary not found at {bin_path}. "
            "Add it (or set LLAMA_SERVER_BIN), then restart. Using evidence-only answers until then.",
            file=sys.stderr,
        )
        return False

    if _llama_proc is not None and _llama_proc.poll() is None:
        # Still starting
        for _ in range(40):
            if _health(url):
                return True
            time.sleep(0.25)
        return False

    host = "127.0.0.1"
    port = 8080
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8080
    except Exception:  # noqa: BLE001
        pass

    cmd = [
        str(bin_path),
        "-m",
        str(model["path"]),
        "--host",
        host,
        "--port",
        str(port),
        "-c",
        str(min(int(model.get("context") or 4096), settings.mufasa_max_context_tokens)),
        "-t",
        str(settings.mufasa_llama_threads),
    ]
    print(f"Starting llama-server: {bin_path.name} · {model['file']}", file=sys.stderr)
    _llama_proc = subprocess.Popen(  # noqa: S603
        cmd,
        cwd=str(settings.models_dir.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def _stop() -> None:
        global _llama_proc
        if _llama_proc is not None and _llama_proc.poll() is None:
            _llama_proc.terminate()
            try:
                _llama_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _llama_proc.kill()
        _llama_proc = None

    atexit.register(_stop)

    for _ in range(80):
        if _llama_proc.poll() is not None:
            print("llama-server exited while starting. Using evidence-only answers.", file=sys.stderr)
            return False
        if _health(url):
            return True
        time.sleep(0.25)
    print("llama-server did not become ready in time. Using evidence-only answers.", file=sys.stderr)
    return False


def resolve_generator(settings: Settings) -> Generator:
    """Pick llama-server when configured and available, else stub."""
    kind = (settings.mufasa_generator or "llama-server").lower()
    if kind == "stub":
        return StubGenerator()

    if kind in ("llama-server", "llama_server", "llama"):
        if ensure_llama_server(settings):
            return LlamaServerGenerator(
                base_url=settings.llama_server_url,
                model=settings.mufasa_model,
            )
        return StubGenerator()

    return get_generator(kind)


def generator_status(settings: Settings) -> dict[str, Any]:
    kind = (settings.mufasa_generator or "llama-server").lower()
    model = settings.model_entry()
    if kind == "stub":
        return {"requested": kind, "active": "stub", "reason": "configured"}
    ready = _health(settings.llama_server_url)
    if ready and model.get("present"):
        return {"requested": kind, "active": "llama-server", "reason": "ready"}
    if not model.get("present"):
        return {"requested": kind, "active": "stub", "reason": "model_missing"}
    bin_path = llama_server_bin(settings)
    if not bin_path.exists():
        return {"requested": kind, "active": "stub", "reason": "bin_missing"}
    return {"requested": kind, "active": "stub" if not ready else "llama-server", "reason": "starting"}
