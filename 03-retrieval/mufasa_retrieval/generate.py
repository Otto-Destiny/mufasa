"""Generation behind one interface: ``generate(prompt) -> str``.

Everything downstream — the bundle, the prompt, the validator, the whole test
suite — depends only on this signature, which is why most of the suite runs with
no model file present at all.

The shipped default talks to a local ``llama-server`` over 127.0.0.1. Greedy by
default: llama-server's own default temperature is 0.8, and retrieval-v1's
results were not reproducible run to run because of it.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .bundle import EvidenceBundle
from .validate import split_sentences


class GenerationError(RuntimeError):
    pass


class Generator(Protocol):
    name: str

    def generate(self, prompt: str, *, max_tokens: int = 400, temperature: float = 0.0) -> str: ...


@dataclass
class LlamaServerGenerator:
    """HTTP client for llama.cpp's server.

    retrieval-v1 first shelled out to ``llama-cli`` and parsed the terminal
    transcript; its interactive UI truncates long echoed lines, so build banners
    leaked into extracted answers and were then flagged as invented facts. The
    HTTP API returns clean JSON with the chat template already applied.
    """

    base_url: str = "http://127.0.0.1:8080"
    model: str = "mufasa"
    timeout: float = 240.0
    name: str = "llama-server"

    def generate(self, prompt: str, *, max_tokens: int = 400, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        req = urllib.request.Request(  # noqa: S310 - fixed loopback URL
            f"{self.base_url.rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise GenerationError(f"llama-server unreachable at {self.base_url}: {exc}") from exc
        try:
            return body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise GenerationError(f"unexpected llama-server response: {body!r}") from exc


@dataclass
class StubGenerator:
    """Deterministic generator for tests and CI.

    Echoes the required abstention sentence when the prompt carries one, and
    otherwise composes a cited answer from the first evidence record — enough to
    exercise the validator without a model file.
    """

    name: str = "stub"

    def generate(self, prompt: str, *, max_tokens: int = 400, temperature: float = 0.0) -> str:
        for line in prompt.splitlines():
            if line.startswith("REQUIRED OPENING: "):
                return line[len("REQUIRED OPENING: "):].strip()
        first = next((ln for ln in prompt.splitlines() if ln.startswith("[E1] ")), None)
        if first:
            body = first[len("[E1] "):].strip()
            lead = (split_sentences(body) or [body])[0]
            return f"{lead} [E1]"
        return "No verified matching evidence in this corpus."

    @staticmethod
    def from_evidence(bundle: EvidenceBundle) -> str:
        if not bundle.decision.answerable or not bundle.records:
            return bundle.decision.message or "No verified matching evidence in this corpus."
        rec = bundle.records[0]
        body = (rec.text or "").strip()
        lead = (split_sentences(body) or [body])[0]
        return f"{lead} [{rec.tag}]"


def get_generator(kind: str | None = None) -> Generator:
    kind = (kind or os.getenv("MUFASA_GENERATOR") or "llama-server").lower()
    if kind == "stub":
        return StubGenerator()
    if kind in ("llama-server", "llama_server", "llama"):
        return LlamaServerGenerator(
            base_url=os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080"),
            model=os.getenv("MUFASA_MODEL", "mufasa"),
        )
    raise ValueError(f"unknown generator {kind!r}")
